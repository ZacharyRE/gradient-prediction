#!/usr/bin/env python3
"""Train an MLP from one layer's hidden state to its complete raw LoRA gradient."""

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


class GradientMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        # Start exactly at the train-mean-gradient baseline and learn only its residual.
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_normalized(model, x: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    output = np.empty((len(x), model.network[-1].out_features), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        batch = torch.from_numpy(np.asarray(x[start:stop], dtype=np.float32)).to(device)
        output[start:stop] = model(batch).cpu().numpy()
    return output


def cosine_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(row_cosine(y_pred, y_true).mean())


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, baseline_mse: float) -> dict:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    mse = float(np.mean(np.square(error)))
    cosines = row_cosine(np.asarray(y_pred), np.asarray(y_true))
    true_norm = np.linalg.norm(y_true, axis=1)
    predicted_norm = np.linalg.norm(y_pred, axis=1)
    norm_correlation = (
        None if true_norm.std() == 0 or predicted_norm.std() == 0
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


def train_epoch(model, optimizer, x: np.ndarray, y_normalized: np.ndarray, batch_size: int,
                device: str, rng: np.random.Generator) -> float:
    model.train()
    order = rng.permutation(len(x))
    total_loss = 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        inputs = torch.from_numpy(np.asarray(x[indices], dtype=np.float32)).to(device)
        targets = torch.from_numpy(np.asarray(y_normalized[indices], dtype=np.float32)).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(indices)
    return total_loss / len(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-width", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = time.time()
    set_seed(args.seed)
    extraction = args.experiment / "extraction"
    x_all = np.load(extraction / "predictor_train_hidden.npy", mmap_mode="r")
    y_all = np.load(extraction / "predictor_train_raw_gradients.npy", mmap_mode="r")
    x_test = np.load(extraction / "candidate_test_hidden.npy", mmap_mode="r")
    y_test = np.load(extraction / "candidate_test_raw_gradients.npy", mmap_mode="r")
    if x_all.shape != (2000, 1536) or y_all.shape != (2000, 12288):
        raise RuntimeError(f"Unexpected train shapes: {x_all.shape}, {y_all.shape}")
    fit_indices, validation_indices = train_test_split(
        np.arange(len(x_all)), test_size=args.validation_size,
        random_state=args.seed, shuffle=True,
    )

    # Select epoch count using only a held-out portion of predictor_train.
    selection_scaler = StandardScaler().fit(x_all[fit_indices])
    x_fit = selection_scaler.transform(x_all[fit_indices]).astype(np.float32)
    x_validation = selection_scaler.transform(x_all[validation_indices]).astype(np.float32)
    selection_mean = np.asarray(y_all[fit_indices], dtype=np.float64).mean(axis=0)
    selection_scale = float(np.sqrt(np.mean(np.square(
        np.asarray(y_all[fit_indices], dtype=np.float64) - selection_mean
    ))))
    y_fit_normalized = ((np.asarray(y_all[fit_indices]) - selection_mean) /
                        selection_scale).astype(np.float32)
    model = GradientMLP(1536, args.hidden_width, 12288, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    history = []
    best_epoch = 0
    best_validation_cosine = -np.inf
    epochs_without_improvement = 0
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_epoch(
            model, optimizer, x_fit, y_fit_normalized, args.batch_size, args.device, rng
        )
        validation_residual = predict_normalized(
            model, x_validation, args.device, args.batch_size
        )
        validation_prediction = validation_residual * selection_scale + selection_mean
        validation_cosine = cosine_metric(y_all[validation_indices], validation_prediction)
        history.append({"epoch": epoch, "train_normalized_mse": train_loss,
                        "validation_mean_cosine": validation_cosine})
        print(json.dumps(history[-1]), flush=True)
        if validation_cosine > best_validation_cosine + 1e-5:
            best_validation_cosine = validation_cosine
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    del model, optimizer
    torch.cuda.empty_cache()

    # Refit from scratch on all 2000 training examples for the selected epoch count.
    set_seed(args.seed)
    final_scaler = StandardScaler().fit(x_all)
    x_train = final_scaler.transform(x_all).astype(np.float32)
    x_test_scaled = final_scaler.transform(x_test).astype(np.float32)
    train_mean = np.asarray(y_all, dtype=np.float64).mean(axis=0)
    target_scale = float(np.sqrt(np.mean(np.square(
        np.asarray(y_all, dtype=np.float64) - train_mean
    ))))
    y_train_normalized = ((np.asarray(y_all) - train_mean) / target_scale).astype(np.float32)
    final_model = GradientMLP(1536, args.hidden_width, 12288, args.dropout).to(args.device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    final_rng = np.random.default_rng(args.seed)
    final_losses = []
    for _ in range(best_epoch):
        final_losses.append(train_epoch(
            final_model, final_optimizer, x_train, y_train_normalized,
            args.batch_size, args.device, final_rng,
        ))
    train_prediction = (
        predict_normalized(final_model, x_train, args.device, args.batch_size)
        * target_scale + train_mean
    ).astype(np.float32)
    test_prediction = (
        predict_normalized(final_model, x_test_scaled, args.device, args.batch_size)
        * target_scale + train_mean
    ).astype(np.float32)
    mean_test_prediction = np.repeat(train_mean[None, :], len(y_test), axis=0)
    train_baseline_mse = float(np.mean(np.square(np.asarray(y_all) - train_mean)))
    test_baseline_mse = float(np.mean(np.square(np.asarray(y_test) - mean_test_prediction)))
    results = {
        "mlp_train": metric_summary(y_all, train_prediction, train_baseline_mse),
        "mlp_test": metric_summary(y_test, test_prediction, test_baseline_mse),
        "mean_gradient_test": metric_summary(y_test, mean_test_prediction, test_baseline_mse),
    }
    factor_results = {}
    for factor, factor_slice in {"A": slice(0, 6144), "B": slice(6144, 12288)}.items():
        baseline_mse = float(np.mean(np.square(
            np.asarray(y_test[:, factor_slice]) - mean_test_prediction[:, factor_slice]
        )))
        factor_results[factor] = {
            "mlp_test": metric_summary(
                y_test[:, factor_slice], test_prediction[:, factor_slice], baseline_mse
            ),
            "mean_gradient_test": metric_summary(
                y_test[:, factor_slice], mean_test_prediction[:, factor_slice], baseline_mse
            ),
        }
    output = args.experiment / "mlp_width512"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
        "scaler_mean": torch.from_numpy(final_scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(final_scaler.scale_.astype(np.float32)),
        "target_mean": torch.from_numpy(train_mean.astype(np.float32)),
        "target_global_scale": target_scale,
    }, output / "model.pt")
    np.save(output / "candidate_mlp_raw_gradient.npy", test_prediction)
    summary = {
        "status": "passed",
        "model": {"architecture": "1536-512-GELU-Dropout-12288",
                  "hidden_width": args.hidden_width, "dropout": args.dropout,
                  "raw_gradient_output": True, "gradient_projection_or_sketch": "none"},
        "optimization": {"optimizer": "AdamW", "learning_rate": args.learning_rate,
                         "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                         "target_transform": "subtract_train_mean_and_one_global_scale"},
        "epoch_selection": {"fit_samples": len(fit_indices),
                            "validation_samples": len(validation_indices),
                            "best_epoch": best_epoch,
                            "best_validation_mean_cosine": best_validation_cosine,
                            "history": history},
        "final_train_normalized_mse": final_losses,
        "results": results,
        "results_by_lora_factor": factor_results,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
