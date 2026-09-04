#!/usr/bin/env python3
"""Compare pooled and learned second-order predictors of per-sample LoRA gradients."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradient_geometry.compression import row_cosine  # noqa: E402


PRIMARY_REPRESENTATIONS = (
    "last_width256",
    "mean_width256",
    "second_order_mean_dr64_width256",
)
CONTROL_REPRESENTATIONS = (
    "last_width318",
    "mean_width318",
    "second_order_sum_dr64_width256",
)
ALL_REPRESENTATIONS = PRIMARY_REPRESENTATIONS + CONTROL_REPRESENTATIONS
OBJECTIVES = ("joint", "A_only", "B_only")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def factor_slices(experiment: Path) -> dict[str, slice]:
    layout = json.loads((experiment / "parameter_layout.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[int, int]]] = {"A": [], "B": []}
    for entry in layout["parameters"]:
        factor = entry.get("factor", "A" if "lora_A" in entry["name"] else "B")
        grouped.setdefault(factor, []).append(
            (int(entry["flat_start"]), int(entry["flat_stop"]))
        )
    slices = {}
    for factor in ("A", "B"):
        blocks = grouped.get(factor, [])
        if len(blocks) != 1:
            raise RuntimeError(
                f"This single-module ablation expects one contiguous {factor} block, got {blocks}"
            )
        slices[factor] = slice(*blocks[0])
    if slices["A"].start != 0 or slices["A"].stop != slices["B"].start:
        raise RuntimeError(f"Expected contiguous A then B layout, got {slices}")
    if slices["B"].stop != int(layout["raw_gradient_dim"]):
        raise RuntimeError("Factor slices do not cover the raw gradient")
    return slices


class RaggedHiddenStore:
    def __init__(self, directory: Path, role: str):
        self.tokens = np.load(directory / f"{role}_tokens.npy", mmap_mode="r")
        self.offsets = np.load(directory / f"{role}_offsets.npy")
        if self.tokens.ndim != 2 or self.offsets.ndim != 1:
            raise RuntimeError(f"{role}: invalid ragged hidden cache")
        if len(self.offsets) < 2 or self.offsets[0] != 0 or self.offsets[-1] != len(self.tokens):
            raise RuntimeError(f"{role}: invalid ragged hidden offsets")
        if np.any(np.diff(self.offsets) <= 0):
            raise RuntimeError(f"{role}: every hidden sequence must be non-empty")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def hidden_dim(self) -> int:
        return self.tokens.shape[1]

    def sequence(self, index: int) -> np.ndarray:
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        # Copy out of the read-only memmap so torch.from_numpy receives writable
        # storage and never exposes undefined in-place behavior.
        return np.array(self.tokens[start:stop], dtype=np.float32, copy=True)

    def pooled(self, pooling: str) -> np.ndarray:
        output = np.empty((len(self), self.hidden_dim), dtype=np.float32)
        for index in range(len(self)):
            sequence = self.sequence(index)
            if pooling == "last":
                output[index] = sequence[-1]
            elif pooling == "mean":
                output[index] = sequence.mean(axis=0, dtype=np.float64).astype(np.float32)
            else:
                raise ValueError(f"Unknown pooling {pooling!r}")
        return output


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    kind: str
    bottleneck_width: int
    projection_width: int | None = None
    second_order_reduction: str | None = None
    role: str = "primary"


def representation_spec(name: str) -> RepresentationSpec:
    specs = {
        "last_width256": RepresentationSpec(name, "last", 256),
        "mean_width256": RepresentationSpec(name, "mean", 256),
        "second_order_mean_dr64_width256": RepresentationSpec(
            name, "second_order", 256, 64, "mean", "primary"
        ),
        "last_width318": RepresentationSpec(name, "last", 318, role="capacity_control"),
        "mean_width318": RepresentationSpec(name, "mean", 318, role="capacity_control"),
        "second_order_sum_dr64_width256": RepresentationSpec(
            name, "second_order", 256, 64, "sum", "length_confound_control"
        ),
    }
    return specs[name]


class GradientPredictor(torch.nn.Module):
    def __init__(
        self,
        spec: RepresentationSpec,
        hidden_dim: int,
        output_dim_a: int,
        output_dim_b: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.spec = spec
        if spec.kind == "second_order":
            assert spec.projection_width is not None
            self.a2 = torch.nn.Linear(hidden_dim, spec.projection_width, bias=False)
            self.b1 = torch.nn.Linear(hidden_dim, spec.projection_width, bias=False)
            encoder_input = spec.projection_width**2
        else:
            self.a2 = None
            self.b1 = None
            encoder_input = hidden_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_input, spec.bottleneck_width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.a_head = torch.nn.Linear(spec.bottleneck_width, output_dim_a)
        self.b_head = torch.nn.Linear(spec.bottleneck_width, output_dim_b)

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spec.kind == "second_order":
            if lengths is None or inputs.ndim != 3:
                raise RuntimeError("Second-order input requires padded sequences and lengths")
            x = self.a2(inputs)
            y = self.b1(inputs)
            core = torch.bmm(x.transpose(1, 2), y)
            if self.spec.second_order_reduction == "mean":
                core = core / lengths.to(core.dtype).view(-1, 1, 1)
            elif self.spec.second_order_reduction != "sum":
                raise RuntimeError(
                    f"Unknown second-order reduction {self.spec.second_order_reduction!r}"
                )
            features = core.flatten(start_dim=1)
        else:
            if inputs.ndim != 2:
                raise RuntimeError("Pooled input must be a matrix")
            features = inputs
        latent = self.encoder(features)
        return self.a_head(latent), self.b_head(latent)


def objective_factors(objective: str) -> tuple[str, ...]:
    if objective == "joint":
        return ("A", "B")
    if objective == "A_only":
        return ("A",)
    if objective == "B_only":
        return ("B",)
    raise ValueError(f"Unknown objective {objective!r}")


def cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(
        prediction, target, dim=1, eps=1e-8
    ).mean()


def pad_sequences(
    store: RaggedHiddenStore, indices: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [torch.from_numpy(store.sequence(int(index))) for index in indices]
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded.to(device), lengths.to(device)


def model_inputs(
    spec: RepresentationSpec,
    store: RaggedHiddenStore,
    pooled: np.ndarray | None,
    indices: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if spec.kind == "second_order":
        return pad_sequences(store, indices, device)
    assert pooled is not None
    inputs = torch.from_numpy(np.asarray(pooled[indices], dtype=np.float32)).to(device)
    return inputs, None


def loss_dict(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    objective: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    active = objective_factors(objective)
    loss_a = cosine_loss(prediction_a, target_a) if "A" in active else None
    loss_b = cosine_loss(prediction_b, target_b) if "B" in active else None
    total = sum(loss for loss in (loss_a, loss_b) if loss is not None)
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_A": None if loss_a is None else float(loss_a.detach().cpu()),
        "loss_B": None if loss_b is None else float(loss_b.detach().cpu()),
    }


def train_epoch(
    model: GradientPredictor,
    optimizer: torch.optim.Optimizer,
    spec: RepresentationSpec,
    store: RaggedHiddenStore,
    pooled: np.ndarray | None,
    gradients_a: np.ndarray,
    gradients_b: np.ndarray,
    fit_indices: np.ndarray,
    objective: str,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
) -> dict:
    model.train()
    order = rng.permutation(fit_indices)
    totals = {"loss": 0.0, "loss_A": 0.0, "loss_B": 0.0}
    active = objective_factors(objective)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        inputs, lengths = model_inputs(spec, store, pooled, indices, device)
        target_a = torch.from_numpy(np.asarray(gradients_a[indices], dtype=np.float32)).to(device)
        target_b = torch.from_numpy(np.asarray(gradients_b[indices], dtype=np.float32)).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction_a, prediction_b = model(inputs, lengths)
        loss, values = loss_dict(
            prediction_a, prediction_b, target_a, target_b, objective
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = len(indices)
        totals["loss"] += values["loss"] * count
        for factor in active:
            totals[f"loss_{factor}"] += values[f"loss_{factor}"] * count
    return {
        key: (None if key[-1:] in ("A", "B") and key[-1] not in active else value / len(order))
        for key, value in totals.items()
    }


@torch.no_grad()
def predict(
    model: GradientPredictor,
    spec: RepresentationSpec,
    store: RaggedHiddenStore,
    pooled: np.ndarray | None,
    indices: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prediction_a = np.empty((len(indices), model.a_head.out_features), dtype=np.float32)
    prediction_b = np.empty((len(indices), model.b_head.out_features), dtype=np.float32)
    for output_start in range(0, len(indices), batch_size):
        batch_indices = indices[output_start : output_start + batch_size]
        inputs, lengths = model_inputs(spec, store, pooled, batch_indices, device)
        batch_a, batch_b = model(inputs, lengths)
        output_stop = output_start + len(batch_indices)
        prediction_a[output_start:output_stop] = batch_a.cpu().numpy()
        prediction_b[output_start:output_stop] = batch_b.cpu().numpy()
    return prediction_a, prediction_b


def metric_summary(truth: np.ndarray, prediction: np.ndarray) -> dict:
    cosines = row_cosine(np.asarray(prediction), np.asarray(truth))
    true_norm = np.linalg.norm(truth, axis=1)
    predicted_norm = np.linalg.norm(prediction, axis=1)
    norm_correlation = (
        None
        if true_norm.std() <= 1e-12 or predicted_norm.std() <= 1e-12
        else float(np.corrcoef(true_norm, predicted_norm)[0, 1])
    )
    return {
        "mean_cosine": float(cosines.mean()),
        "median_cosine": float(np.median(cosines)),
        "cosine_std": float(cosines.std()),
        "cosine_loss": float(1.0 - cosines.mean()),
        "gradient_norm_pearson": norm_correlation,
        "true_norm_mean": float(true_norm.mean()),
        "predicted_norm_mean": float(predicted_norm.mean()),
    }


def target_alignment(
    candidate_truth: np.ndarray,
    candidate_prediction: np.ndarray,
    target_mean: np.ndarray,
) -> dict:
    true_similarity = row_cosine(candidate_truth, target_mean[None, :])
    predicted_similarity = row_cosine(candidate_prediction, target_mean[None, :])
    correlation = spearmanr(true_similarity, predicted_similarity)
    return {
        "spearman": finite_or_none(correlation.statistic),
        "spearman_pvalue": finite_or_none(correlation.pvalue),
        "definition": (
            "Spearman over candidate_test between factor-wise true and predicted "
            "cosine alignment to the corresponding mean target-validation gradient factor"
        ),
    }


def split_loss(metrics: dict[str, dict], objective: str) -> dict:
    active = objective_factors(objective)
    return {
        "loss": float(sum(metrics[factor]["cosine_loss"] for factor in active)),
        "loss_A": metrics["A"]["cosine_loss"] if "A" in active else None,
        "loss_B": metrics["B"]["cosine_loss"] if "B" in active else None,
    }


def evaluate_split(
    model: GradientPredictor,
    spec: RepresentationSpec,
    store: RaggedHiddenStore,
    pooled: np.ndarray | None,
    gradients_a: np.ndarray,
    gradients_b: np.ndarray,
    indices: np.ndarray,
    objective: str,
    batch_size: int,
    device: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    prediction_a, prediction_b = predict(
        model, spec, store, pooled, indices, batch_size, device
    )
    all_metrics = {
        "A": metric_summary(np.asarray(gradients_a[indices]), prediction_a),
        "B": metric_summary(np.asarray(gradients_b[indices]), prediction_b),
    }
    active = objective_factors(objective)
    metrics = {
        factor: (all_metrics[factor] if factor in active else None) for factor in ("A", "B")
    }
    return {
        "losses": split_loss(all_metrics, objective),
        "factors": metrics,
    }, prediction_a, prediction_b


def train_one(
    experiment: Path,
    output_root: Path,
    spec: RepresentationSpec,
    objective: str,
    train_store: RaggedHiddenStore,
    test_store: RaggedHiddenStore,
    train_pooled: np.ndarray | None,
    test_pooled: np.ndarray | None,
    train_a: np.ndarray,
    train_b: np.ndarray,
    test_a: np.ndarray,
    test_b: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    target_mean_by_factor: dict[str, np.ndarray] | None,
    args,
) -> dict:
    started = time.time()
    set_seed(args.seed)
    model = GradientPredictor(
        spec,
        train_store.hidden_dim,
        train_a.shape[1],
        train_b.shape[1],
        args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    best_state = None
    best_epoch = 0
    best_validation_loss = np.inf
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        train_losses = train_epoch(
            model,
            optimizer,
            spec,
            train_store,
            train_pooled,
            train_a,
            train_b,
            fit_indices,
            objective,
            args.batch_size,
            args.device,
            rng,
        )
        validation, _, _ = evaluate_split(
            model,
            spec,
            train_store,
            train_pooled,
            train_a,
            train_b,
            validation_indices,
            objective,
            args.batch_size,
            args.device,
        )
        row = {
            "epoch": epoch,
            "train": train_losses,
            "validation": validation["losses"],
        }
        history.append(row)
        print(json.dumps({"representation": spec.name, "objective": objective, **row}), flush=True)
        validation_loss = validation["losses"]["loss"]
        if validation_loss < best_validation_loss - args.minimum_improvement:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(best_state)

    train_metrics, _, _ = evaluate_split(
        model, spec, train_store, train_pooled, train_a, train_b, fit_indices,
        objective, args.batch_size, args.device,
    )
    validation_metrics, _, _ = evaluate_split(
        model, spec, train_store, train_pooled, train_a, train_b, validation_indices,
        objective, args.batch_size, args.device,
    )
    test_indices = np.arange(len(test_store))
    test_metrics, prediction_a, prediction_b = evaluate_split(
        model, spec, test_store, test_pooled, test_a, test_b, test_indices,
        objective, args.batch_size, args.device,
    )
    active = objective_factors(objective)
    alignment = None
    if target_mean_by_factor is not None:
        alignment = {}
        for factor, truth, prediction in (
            ("A", test_a, prediction_a), ("B", test_b, prediction_b)
        ):
            if factor in active:
                alignment[factor] = target_alignment(
                    truth, prediction, target_mean_by_factor[factor]
                )

    output = output_root / spec.name / objective
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "representation": spec.__dict__,
            "objective": objective,
            "hidden_dim": train_store.hidden_dim,
            "output_dim_A": train_a.shape[1],
            "output_dim_B": train_b.shape[1],
            "dropout": args.dropout,
        },
        output / "model.pt",
    )
    if "A" in active:
        np.save(output / "candidate_predicted_A_gradient.npy", prediction_a)
    if "B" in active:
        np.save(output / "candidate_predicted_B_gradient.npy", prediction_b)
    if objective == "joint":
        np.save(
            output / "candidate_predicted_raw_gradient.npy",
            np.concatenate((prediction_a, prediction_b), axis=1),
        )
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "status": "passed",
        "representation": spec.__dict__,
        "same_tensor_H": True,
        "second_order_formula": (
            None
            if spec.kind != "second_order"
            else (
                "((H @ A2.T).T @ (H @ B1)) / T"
                if spec.second_order_reduction == "mean"
                else "(H @ A2.T).T @ (H @ B1)"
            )
        ),
        "objective": objective,
        "active_factors": list(active),
        "model": {
            "trainable_parameters": trainable_parameters,
            "dropout": args.dropout,
            "output_dim_A": train_a.shape[1],
            "output_dim_B": train_b.shape[1],
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "input_transform": "none; all representations derive directly from cached H",
            "target_transform": "none; raw LoRA factor gradients",
            "loss": "sum of active factor-wise cosine losses",
            "magnitude_loss": False,
            "gradient_clip_norm": 1.0,
        },
        "selection": {
            "fit_samples": len(fit_indices),
            "validation_samples": len(validation_indices),
            "metric": "validation active-factor cosine loss",
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "minimum_improvement": args.minimum_improvement,
            "history": history,
            "refit_on_validation": False,
        },
        "results": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "target_alignment_by_factor": alignment,
        "norm_metric_warning": (
            "Pure cosine loss does not identify prediction magnitude; norm correlation is diagnostic."
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    del model, optimizer, best_state
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return summary


def comparison_rows(summaries: list[dict]) -> list[dict]:
    rows = []
    for summary in summaries:
        active = set(summary["active_factors"])
        for factor in ("A", "B"):
            if factor not in active:
                continue
            test = summary["results"]["test"]["factors"][factor]
            alignment = (summary["target_alignment_by_factor"] or {}).get(factor, {})
            rows.append(
                {
                    "representation": summary["representation"]["name"],
                    "representation_role": summary["representation"]["role"],
                    "objective": summary["objective"],
                    "factor": factor,
                    "trainable_parameters": summary["model"]["trainable_parameters"],
                    "best_epoch": summary["selection"]["best_epoch"],
                    "train_loss": summary["results"]["train"]["losses"][f"loss_{factor}"],
                    "validation_loss": summary["results"]["validation"]["losses"][f"loss_{factor}"],
                    "test_loss": summary["results"]["test"]["losses"][f"loss_{factor}"],
                    "test_mean_cosine": test["mean_cosine"],
                    "test_median_cosine": test["median_cosine"],
                    "test_gradient_norm_pearson": test["gradient_norm_pearson"],
                    "target_alignment_spearman": alignment.get("spearman"),
                }
            )
    return rows


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Hidden representation gradient-prediction ablation",
        "",
        "| Representation | Role | Objective | Factor | Params | Train loss | Val loss | Test loss | Mean cosine | Median cosine | Norm r | Target rho |",
        "|:---|:---|:---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fmt = lambda value: "n/a" if value is None else f"{value:.4f}"
        lines.append(
            f"| {row['representation']} | {row['representation_role']} | "
            f"{row['objective']} | {row['factor']} | {row['trainable_parameters']} | "
            f"{fmt(row['train_loss'])} | {fmt(row['validation_loss'])} | "
            f"{fmt(row['test_loss'])} | {fmt(row['test_mean_cosine'])} | "
            f"{fmt(row['test_median_cosine'])} | "
            f"{fmt(row['test_gradient_norm_pearson'])} | "
            f"{fmt(row['target_alignment_spearman'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--representations", nargs="+", choices=ALL_REPRESENTATIONS,
                        default=list(ALL_REPRESENTATIONS))
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES,
                        default=list(OBJECTIVES))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = time.time()
    experiment = args.experiment.resolve()
    cache_summary = json.loads(
        (experiment / "prompt_hidden" / "summary.json").read_text(encoding="utf-8")
    )
    if cache_summary.get("status") != "passed":
        raise RuntimeError("Prompt hidden cache is not complete")
    if cache_summary.get("hidden_source") != "target_module_input":
        raise RuntimeError("Ablation requires a target-module-input sequence cache")
    train_store = RaggedHiddenStore(experiment / "prompt_hidden", "predictor_train")
    test_store = RaggedHiddenStore(experiment / "prompt_hidden", "candidate_test")
    gradients_train = np.load(
        experiment / "extraction" / "predictor_train_raw_gradients.npy", mmap_mode="r"
    )
    gradients_test = np.load(
        experiment / "extraction" / "candidate_test_raw_gradients.npy", mmap_mode="r"
    )
    if len(train_store) != len(gradients_train) or len(test_store) != len(gradients_test):
        raise RuntimeError("Hidden cache and gradient sample counts differ")
    slices = factor_slices(experiment)
    train_a, train_b = gradients_train[:, slices["A"]], gradients_train[:, slices["B"]]
    test_a, test_b = gradients_test[:, slices["A"]], gradients_test[:, slices["B"]]
    all_indices = np.arange(len(train_store))
    fit_indices, validation_indices = train_test_split(
        all_indices,
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
    )
    output_root = experiment / "hidden_representation_ablation"
    output_root.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "seed": args.seed,
        "fit_indices": fit_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "candidate_test_indices": list(range(len(test_store))),
        "refit_on_validation": False,
    }
    split_path = output_root / "split.json"
    if split_path.exists() and json.loads(split_path.read_text(encoding="utf-8")) != split_payload:
        raise RuntimeError("Refusing to mix ablation runs with a different split")
    write_json(split_path, split_payload)

    needed_pooling = {
        representation_spec(name).kind for name in args.representations
    } & {"last", "mean"}
    train_pool = {kind: train_store.pooled(kind) for kind in needed_pooling}
    test_pool = {kind: test_store.pooled(kind) for kind in needed_pooling}
    target_mean_by_factor = None
    target_path = experiment / "target_alignment" / "target_validation_mean_raw_gradient.npy"
    if target_path.exists():
        target_mean = np.load(target_path)
        if target_mean.shape != (gradients_train.shape[1],):
            raise RuntimeError(f"Unexpected target mean gradient shape {target_mean.shape}")
        target_mean_by_factor = {
            factor: np.asarray(target_mean[block], dtype=np.float32)
            for factor, block in slices.items()
        }

    summaries = []
    for representation_name in args.representations:
        spec = representation_spec(representation_name)
        pooled_train = train_pool.get(spec.kind)
        pooled_test = test_pool.get(spec.kind)
        for objective in args.objectives:
            summary_path = output_root / spec.name / objective / "summary.json"
            if summary_path.exists():
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                if existing.get("status") == "passed":
                    expected_optimization = {
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "batch_size": args.batch_size,
                        "seed": args.seed,
                    }
                    actual_optimization = existing.get("optimization", {})
                    mismatches = {
                        key: (actual_optimization.get(key), value)
                        for key, value in expected_optimization.items()
                        if actual_optimization.get(key) != value
                    }
                    if (
                        existing.get("representation") != spec.__dict__
                        or existing.get("objective") != objective
                        or existing.get("model", {}).get("dropout") != args.dropout
                        or existing.get("selection", {}).get("fit_samples") != len(fit_indices)
                        or existing.get("selection", {}).get("validation_samples")
                        != len(validation_indices)
                        or existing.get("selection", {}).get("max_epochs") != args.max_epochs
                        or existing.get("selection", {}).get("patience") != args.patience
                        or existing.get("selection", {}).get("minimum_improvement")
                        != args.minimum_improvement
                        or mismatches
                    ):
                        raise RuntimeError(
                            f"Existing result {summary_path} has incompatible settings: "
                            f"optimization mismatches={mismatches}"
                        )
                    summaries.append(existing)
                    print(json.dumps({"skipped_completed": str(summary_path)}), flush=True)
                    continue
            summaries.append(
                train_one(
                    experiment,
                    output_root,
                    spec,
                    objective,
                    train_store,
                    test_store,
                    pooled_train,
                    pooled_test,
                    train_a,
                    train_b,
                    test_a,
                    test_b,
                    fit_indices,
                    validation_indices,
                    target_mean_by_factor,
                    args,
                )
            )
    rows = comparison_rows(summaries)
    comparison = {
        "status": "passed",
        "experiment": str(experiment),
        "protocol": {
            "same_cached_H_for_all_representations": True,
            "hidden_source": cache_summary["hidden_source"],
            "hidden_module_suffix": cache_summary["hidden_module_suffix"],
            "cached_last_vs_existing_max_abs_difference": {
                role: cache_summary["roles"][role]["last_hidden_max_abs_difference"]
                for role in ("predictor_train", "candidate_test")
            },
            "fit_samples": len(fit_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_store),
            "primary_representations": list(PRIMARY_REPRESENTATIONS),
            "control_representations": list(CONTROL_REPRESENTATIONS),
            "objectives": list(args.objectives),
            "loss": "factor-wise cosine only",
        },
        "rows": rows,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_root / "comparison.json", comparison)
    write_markdown(output_root / "comparison.md", rows)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
