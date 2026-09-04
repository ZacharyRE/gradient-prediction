#!/usr/bin/env python3
"""Predict flattened LoRA A+B gradients with a shared bottleneck and separate heads."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradient_geometry.compression import row_cosine


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class StructuredLoRAGradientBottleneck(torch.nn.Module):
    """Use one hidden representation while preserving separate LoRA A/B output heads."""

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        factor_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, bottleneck_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.a_head = torch.nn.Linear(bottleneck_dim, factor_dim)
        self.b_head = torch.nn.Linear(bottleneck_dim, factor_dim)

        # The external train means are the initial predictions. Train only residuals.
        torch.nn.init.zeros_(self.a_head.weight)
        torch.nn.init.zeros_(self.a_head.bias)
        torch.nn.init.zeros_(self.b_head.weight)
        torch.nn.init.zeros_(self.b_head.bias)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(hidden)
        return self.a_head(encoded), self.b_head(encoded)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_factors(gradients: np.ndarray, factor_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return gradients[:, :factor_dim], gradients[:, factor_dim:]


def fit_factor_transform(factor: np.ndarray) -> tuple[np.ndarray, float]:
    mean = np.asarray(factor, dtype=np.float64).mean(axis=0)
    scale = float(
        np.sqrt(np.mean(np.square(np.asarray(factor, dtype=np.float64) - mean)))
    )
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid target scale: {scale}")
    return mean, scale


def normalize_factor(factor: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    return ((np.asarray(factor) - mean) / scale).astype(np.float32)


@torch.no_grad()
def predict_normalized(
    model: StructuredLoRAGradientBottleneck,
    x: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    factor_dim = model.a_head.out_features
    output_a = np.empty((len(x), factor_dim), dtype=np.float32)
    output_b = np.empty((len(x), factor_dim), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        inputs = torch.from_numpy(np.asarray(x[start:stop], dtype=np.float32)).to(device)
        prediction_a, prediction_b = model(inputs)
        output_a[start:stop] = prediction_a.cpu().numpy()
        output_b[start:stop] = prediction_b.cpu().numpy()
    return output_a, output_b


def denormalize_and_join(
    normalized_a: np.ndarray,
    normalized_b: np.ndarray,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    scale_a: float,
    scale_b: float,
) -> np.ndarray:
    prediction_a = normalized_a * scale_a + mean_a
    prediction_b = normalized_b * scale_b + mean_b
    return np.concatenate((prediction_a, prediction_b), axis=1).astype(np.float32)


def mean_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(row_cosine(np.asarray(y_pred), np.asarray(y_true)).mean())


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, baseline_mse: float) -> dict:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    mse = float(np.mean(np.square(error)))
    cosines = row_cosine(np.asarray(y_pred), np.asarray(y_true))
    true_norm = np.linalg.norm(y_true, axis=1)
    predicted_norm = np.linalg.norm(y_pred, axis=1)
    norm_correlation = (
        None
        if true_norm.std() == 0 or predicted_norm.std() == 0
        else float(np.corrcoef(true_norm, predicted_norm)[0, 1])
    )
    return {
        "mean_cosine": float(cosines.mean()),
        "median_cosine": float(np.median(cosines)),
        "cosine_std": float(cosines.std()),
        "r2_uniform": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "r2_variance_weighted": float(
            r2_score(y_true, y_pred, multioutput="variance_weighted")
        ),
        "mse": mse,
        "relative_mse_to_train_mean": mse / baseline_mse,
        "gradient_norm_pearson": norm_correlation,
        "true_norm_mean": float(true_norm.mean()),
        "predicted_norm_mean": float(predicted_norm.mean()),
    }


def train_epoch(
    model: StructuredLoRAGradientBottleneck,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    batch_size: int,
    device: str,
    rng: np.random.Generator,
    loss_weight_a: float,
    loss_weight_b: float,
) -> dict[str, float]:
    model.train()
    order = rng.permutation(len(x))
    totals = {"loss": 0.0, "loss_a": 0.0, "loss_b": 0.0}
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        inputs = torch.from_numpy(np.asarray(x[indices], dtype=np.float32)).to(device)
        targets_a = torch.from_numpy(np.asarray(y_a[indices], dtype=np.float32)).to(device)
        targets_b = torch.from_numpy(np.asarray(y_b[indices], dtype=np.float32)).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction_a, prediction_b = model(inputs)
        loss_a = torch.nn.functional.mse_loss(prediction_a, targets_a)
        loss_b = torch.nn.functional.mse_loss(prediction_b, targets_b)
        loss = loss_weight_a * loss_a + loss_weight_b * loss_b
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = len(indices)
        totals["loss"] += float(loss.detach().cpu()) * count
        totals["loss_a"] += float(loss_a.detach().cpu()) * count
        totals["loss_b"] += float(loss_b.detach().cpu()) * count
    return {key: value / len(x) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bottleneck-width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-weight-a", type=float, default=1.0)
    parser.add_argument("--loss-weight-b", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bottleneck_width <= 0:
        parser.error("--bottleneck-width must be positive")
    if args.loss_weight_a < 0 or args.loss_weight_b < 0:
        parser.error("loss weights must be non-negative")
    if args.loss_weight_a + args.loss_weight_b <= 0:
        parser.error("at least one loss weight must be positive")

    started = time.time()
    set_seed(args.seed)
    extraction = args.experiment / "extraction"
    x_all = np.load(extraction / "predictor_train_hidden.npy", mmap_mode="r")
    y_all = np.load(extraction / "predictor_train_raw_gradients.npy", mmap_mode="r")
    x_test = np.load(extraction / "candidate_test_hidden.npy", mmap_mode="r")
    y_test = np.load(extraction / "candidate_test_raw_gradients.npy", mmap_mode="r")
    if x_all.ndim != 2 or y_all.ndim != 2 or x_test.ndim != 2 or y_test.ndim != 2:
        raise RuntimeError("Expected two-dimensional hidden-state and gradient arrays")
    if len(x_all) != len(y_all) or len(x_test) != len(y_test):
        raise RuntimeError("Hidden-state and gradient sample counts do not match")
    if x_all.shape[1] != x_test.shape[1] or y_all.shape[1] != y_test.shape[1]:
        raise RuntimeError("Train and test dimensions do not match")
    if y_all.shape[1] % 2 != 0:
        raise RuntimeError(f"A+B gradient dimension must be even, got {y_all.shape[1]}")
    if not 0 < args.validation_size < len(x_all):
        raise RuntimeError("validation size must be between zero and the training sample count")

    input_dim = x_all.shape[1]
    output_dim = y_all.shape[1]
    factor_dim = output_dim // 2
    all_a, all_b = split_factors(y_all, factor_dim)
    test_a, test_b = split_factors(y_test, factor_dim)
    fit_indices, validation_indices = train_test_split(
        np.arange(len(x_all)),
        test_size=args.validation_size,
        random_state=args.seed,
        shuffle=True,
    )

    # Select the epoch using only predictor_train. A and B get independent scales,
    # preventing the higher-energy factor from dominating the optimization loss.
    selection_scaler = StandardScaler().fit(x_all[fit_indices])
    x_fit = selection_scaler.transform(x_all[fit_indices]).astype(np.float32)
    x_validation = selection_scaler.transform(x_all[validation_indices]).astype(np.float32)
    selection_mean_a, selection_scale_a = fit_factor_transform(all_a[fit_indices])
    selection_mean_b, selection_scale_b = fit_factor_transform(all_b[fit_indices])
    y_fit_a = normalize_factor(all_a[fit_indices], selection_mean_a, selection_scale_a)
    y_fit_b = normalize_factor(all_b[fit_indices], selection_mean_b, selection_scale_b)

    model = StructuredLoRAGradientBottleneck(
        input_dim, args.bottleneck_width, factor_dim, args.dropout
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    history = []
    best_epoch = 0
    best_validation_cosine = -np.inf
    epochs_without_improvement = 0
    for epoch in range(1, args.max_epochs + 1):
        train_losses = train_epoch(
            model,
            optimizer,
            x_fit,
            y_fit_a,
            y_fit_b,
            args.batch_size,
            args.device,
            rng,
            args.loss_weight_a,
            args.loss_weight_b,
        )
        validation_a_normalized, validation_b_normalized = predict_normalized(
            model, x_validation, args.device, args.batch_size
        )
        validation_prediction = denormalize_and_join(
            validation_a_normalized,
            validation_b_normalized,
            selection_mean_a,
            selection_mean_b,
            selection_scale_a,
            selection_scale_b,
        )
        validation_true = np.asarray(y_all[validation_indices])
        validation_cosine = mean_cosine(validation_true, validation_prediction)
        validation_prediction_a, validation_prediction_b = split_factors(
            validation_prediction, factor_dim
        )
        row = {
            "epoch": epoch,
            "train_weighted_normalized_mse": train_losses["loss"],
            "train_a_normalized_mse": train_losses["loss_a"],
            "train_b_normalized_mse": train_losses["loss_b"],
            "validation_combined_mean_cosine": validation_cosine,
            "validation_a_mean_cosine": mean_cosine(
                all_a[validation_indices], validation_prediction_a
            ),
            "validation_b_mean_cosine": mean_cosine(
                all_b[validation_indices], validation_prediction_b
            ),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_cosine > best_validation_cosine + 1e-5:
            best_validation_cosine = validation_cosine
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    del model, optimizer
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    # Refit from scratch on all predictor_train examples for the selected epoch count.
    set_seed(args.seed)
    final_scaler = StandardScaler().fit(x_all)
    x_train = final_scaler.transform(x_all).astype(np.float32)
    x_test_scaled = final_scaler.transform(x_test).astype(np.float32)
    train_mean_a, train_scale_a = fit_factor_transform(all_a)
    train_mean_b, train_scale_b = fit_factor_transform(all_b)
    y_train_a = normalize_factor(all_a, train_mean_a, train_scale_a)
    y_train_b = normalize_factor(all_b, train_mean_b, train_scale_b)

    final_model = StructuredLoRAGradientBottleneck(
        input_dim, args.bottleneck_width, factor_dim, args.dropout
    ).to(args.device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    final_rng = np.random.default_rng(args.seed)
    final_losses = []
    for _ in range(best_epoch):
        final_losses.append(
            train_epoch(
                final_model,
                final_optimizer,
                x_train,
                y_train_a,
                y_train_b,
                args.batch_size,
                args.device,
                final_rng,
                args.loss_weight_a,
                args.loss_weight_b,
            )
        )

    train_normalized_a, train_normalized_b = predict_normalized(
        final_model, x_train, args.device, args.batch_size
    )
    test_normalized_a, test_normalized_b = predict_normalized(
        final_model, x_test_scaled, args.device, args.batch_size
    )
    train_prediction = denormalize_and_join(
        train_normalized_a,
        train_normalized_b,
        train_mean_a,
        train_mean_b,
        train_scale_a,
        train_scale_b,
    )
    test_prediction = denormalize_and_join(
        test_normalized_a,
        test_normalized_b,
        train_mean_a,
        train_mean_b,
        train_scale_a,
        train_scale_b,
    )
    train_mean = np.concatenate((train_mean_a, train_mean_b))
    mean_test_prediction = np.repeat(train_mean[None, :], len(y_test), axis=0)
    train_baseline_mse = float(np.mean(np.square(np.asarray(y_all) - train_mean)))
    test_baseline_mse = float(np.mean(np.square(np.asarray(y_test) - mean_test_prediction)))
    results = {
        "structured_bottleneck_train": metric_summary(
            y_all, train_prediction, train_baseline_mse
        ),
        "structured_bottleneck_test": metric_summary(
            y_test, test_prediction, test_baseline_mse
        ),
        "mean_gradient_test": metric_summary(
            y_test, mean_test_prediction, test_baseline_mse
        ),
    }
    factor_results = {}
    for factor, truth, prediction, mean in (
        ("A", all_a, train_prediction[:, :factor_dim], train_mean_a),
        ("B", all_b, train_prediction[:, factor_dim:], train_mean_b),
    ):
        test_truth = test_a if factor == "A" else test_b
        test_factor_prediction = (
            test_prediction[:, :factor_dim]
            if factor == "A"
            else test_prediction[:, factor_dim:]
        )
        test_mean_prediction = np.repeat(mean[None, :], len(test_truth), axis=0)
        baseline_mse = float(np.mean(np.square(np.asarray(test_truth) - test_mean_prediction)))
        train_factor_baseline_mse = float(
            np.mean(np.square(np.asarray(truth) - mean[None, :]))
        )
        factor_results[factor] = {
            "structured_bottleneck_train": metric_summary(
                truth, prediction, train_factor_baseline_mse
            ),
            "structured_bottleneck_test": metric_summary(
                test_truth, test_factor_prediction, baseline_mse
            ),
            "mean_gradient_test": metric_summary(
                test_truth, test_mean_prediction, baseline_mse
            ),
        }

    output = args.experiment / f"structured_bottleneck_width{args.bottleneck_width}"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in final_model.state_dict().items()
            },
            "input_scaler_mean": torch.from_numpy(final_scaler.mean_.astype(np.float32)),
            "input_scaler_scale": torch.from_numpy(final_scaler.scale_.astype(np.float32)),
            "target_mean_a": torch.from_numpy(train_mean_a.astype(np.float32)),
            "target_mean_b": torch.from_numpy(train_mean_b.astype(np.float32)),
            "target_scale_a": train_scale_a,
            "target_scale_b": train_scale_b,
            "input_dim": input_dim,
            "factor_dim": factor_dim,
            "bottleneck_width": args.bottleneck_width,
            "dropout": args.dropout,
        },
        output / "model.pt",
    )
    np.save(output / "candidate_structured_bottleneck_raw_gradient.npy", test_prediction)
    trainable_parameters = sum(
        parameter.numel() for parameter in final_model.parameters() if parameter.requires_grad
    )
    summary = {
        "status": "passed",
        "model": {
            "architecture": (
                f"shared_{input_dim}-{args.bottleneck_width}-GELU-Dropout_"
                f"with_separate_A_and_B_{args.bottleneck_width}-{factor_dim}_heads"
            ),
            "bottleneck_width": args.bottleneck_width,
            "dropout": args.dropout,
            "trainable_parameters": trainable_parameters,
            "flattened_storage_and_label": True,
            "factor_boundary": {"A": [0, factor_dim], "B": [factor_dim, output_dim]},
            "gradient_projection_or_sketch": "none",
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "loss": "weighted_sum_of_separately_normalized_A_and_B_MSE",
            "loss_weight_a": args.loss_weight_a,
            "loss_weight_b": args.loss_weight_b,
            "target_transform": "separate_A_B_train_mean_and_global_RMS_scale",
            "target_scale_a": train_scale_a,
            "target_scale_b": train_scale_b,
        },
        "epoch_selection": {
            "metric": "validation_combined_raw_A_plus_B_mean_cosine",
            "fit_samples": len(fit_indices),
            "validation_samples": len(validation_indices),
            "best_epoch": best_epoch,
            "best_validation_mean_cosine": best_validation_cosine,
            "history": history,
        },
        "final_train_normalized_mse": final_losses,
        "results": results,
        "results_by_lora_factor": factor_results,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
