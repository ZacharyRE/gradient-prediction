#!/usr/bin/env python3
"""Cross-validate a Ridge probe on saved hidden states and full raw LoRA gradients."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gradient_geometry.compression import row_cosine  # noqa: E402


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def factor_slices(experiment: Path) -> dict[str, slice]:
    layout = json.loads((experiment / "parameter_layout.json").read_text(encoding="utf-8"))
    slices = {}
    offset = 0
    for entry in layout["parameters"]:
        start = int(entry.get("flat_start", offset))
        stop = int(entry.get("flat_stop", start + int(entry["numel"])))
        factor = entry.get("factor", "A" if "lora_A" in entry["name"] else "B")
        slices[factor] = slice(start, stop)
        offset = stop
    if set(slices) != {"A", "B"} or offset != int(layout["raw_gradient_dim"]):
        raise RuntimeError(f"Invalid LoRA parameter layout: {layout}")
    return slices


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


def ridge_path_predictions(x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray,
                           alphas: list[float], device: str):
    """Evaluate a Ridge path from one eigendecomposition of X^T X."""
    scaler = StandardScaler().fit(x_train)
    x_fit = torch.from_numpy(scaler.transform(x_train).astype(np.float64)).to(device)
    x_val = torch.from_numpy(scaler.transform(x_validation).astype(np.float64)).to(device)
    y_mean = np.asarray(y_train, dtype=np.float64).mean(axis=0)
    y_fit = torch.from_numpy(np.asarray(y_train, dtype=np.float64) - y_mean).to(device)
    dual = x_fit.shape[1] > x_fit.shape[0]
    gram = x_fit @ x_fit.mT if dual else x_fit.mT @ x_fit
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    if dual:
        transformed_rhs = eigenvectors.mT @ y_fit
        validation_basis = (x_val @ x_fit.mT) @ eigenvectors
    else:
        transformed_rhs = eigenvectors.mT @ (x_fit.mT @ y_fit)
        validation_basis = x_val @ eigenvectors
    predictions = []
    for alpha in alphas:
        coefficient_basis = transformed_rhs / (eigenvalues[:, None] + alpha)
        prediction = (validation_basis @ coefficient_basis).cpu().numpy() + y_mean
        predictions.append(prediction.astype(np.float32))
    diagnostics = {
        "formulation": "dual" if dual else "primal",
        "xtx_eigenvalue_min": float(eigenvalues.min().cpu()),
        "xtx_eigenvalue_max": float(eigenvalues.max().cpu()),
    }
    del x_fit, x_val, y_fit, eigenvalues, eigenvectors, transformed_rhs, validation_basis
    torch.cuda.empty_cache()
    return predictions, diagnostics


def fit_final(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
              alpha: float, device: str):
    scaler = StandardScaler().fit(x_train)
    x_fit = torch.from_numpy(scaler.transform(x_train).astype(np.float64)).to(device)
    x_test_tensor = torch.from_numpy(scaler.transform(x_test).astype(np.float64)).to(device)
    y_mean = np.asarray(y_train, dtype=np.float64).mean(axis=0)
    y_fit = torch.from_numpy(np.asarray(y_train, dtype=np.float64) - y_mean).to(device)
    dual = x_fit.shape[1] > x_fit.shape[0]
    if dual:
        gram = x_fit @ x_fit.mT
        gram.diagonal().add_(alpha)
        coefficient = torch.cholesky_solve(y_fit, torch.linalg.cholesky(gram))
        train_prediction = ((x_fit @ x_fit.mT) @ coefficient).cpu().numpy() + y_mean
        test_prediction = ((x_test_tensor @ x_fit.mT) @ coefficient).cpu().numpy() + y_mean
    else:
        gram = x_fit.mT @ x_fit
        gram.diagonal().add_(alpha)
        coefficient = torch.cholesky_solve(x_fit.mT @ y_fit, torch.linalg.cholesky(gram))
        train_prediction = (x_fit @ coefficient).cpu().numpy() + y_mean
        test_prediction = (x_test_tensor @ coefficient).cpu().numpy() + y_mean
    result = {
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "intercept": y_mean.astype(np.float32),
        "train_prediction": train_prediction.astype(np.float32),
        "test_prediction": test_prediction.astype(np.float32),
    }
    if dual:
        result["dual_coefficient"] = coefficient.cpu().numpy().astype(np.float32)
        result["standardized_train_hidden"] = x_fit.cpu().numpy().astype(np.float32)
        result["solver_formulation"] = "dual"
    else:
        result["coefficient"] = coefficient.cpu().numpy().astype(np.float32)
        result["solver_formulation"] = "primal"
    del x_fit, x_test_tensor, y_fit, gram, coefficient
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True,
                        help="Completed direct-probe experiment containing extraction arrays.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000,
                                 100000, 300000, 1000000])
    args = parser.parse_args()
    started = time.time()
    slices = factor_slices(args.experiment)
    extraction = args.experiment / "extraction"
    x_train = np.load(extraction / "predictor_train_hidden.npy", mmap_mode="r")
    y_train = np.load(extraction / "predictor_train_raw_gradients.npy", mmap_mode="r")
    x_test = np.load(extraction / "candidate_test_hidden.npy", mmap_mode="r")
    y_test = np.load(extraction / "candidate_test_raw_gradients.npy", mmap_mode="r")
    if x_train.ndim != 2 or y_train.ndim != 2 or len(x_train) != len(y_train):
        raise RuntimeError(f"Unexpected train shapes: {x_train.shape}, {y_train.shape}")
    if (
        x_test.ndim != 2
        or y_test.ndim != 2
        or len(x_test) != len(y_test)
        or x_train.shape[1] != x_test.shape[1]
        or y_train.shape[1] != y_test.shape[1]
    ):
        raise RuntimeError(f"Unexpected test shapes: {x_test.shape}, {y_test.shape}")
    alphas = [float(alpha) for alpha in args.alphas]
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    cv = {str(alpha): [] for alpha in alphas}
    fold_diagnostics = []
    for fold, (fit_indices, validation_indices) in enumerate(splitter.split(x_train)):
        predictions, diagnostics = ridge_path_predictions(
            np.asarray(x_train[fit_indices]), np.asarray(y_train[fit_indices]),
            np.asarray(x_train[validation_indices]), alphas, args.device,
        )
        fold_diagnostics.append({"fold": fold, **diagnostics})
        truth = np.asarray(y_train[validation_indices])
        for alpha, prediction in zip(alphas, predictions, strict=True):
            cv[str(alpha)].append(float(row_cosine(prediction, truth).mean()))
    cv_summary = {
        alpha: {"fold_mean_cosines": scores, "mean_cosine": float(np.mean(scores)),
                "std_cosine": float(np.std(scores))}
        for alpha, scores in cv.items()
    }
    selected_alpha = float(max(alphas, key=lambda alpha: cv_summary[str(alpha)]["mean_cosine"]))
    fitted = fit_final(x_train, y_train, x_test, selected_alpha, args.device)
    train_mean = np.asarray(y_train, dtype=np.float64).mean(axis=0)
    test_mean_prediction = np.repeat(train_mean[None, :], len(y_test), axis=0)
    test_baseline_mse = float(np.mean(np.square(np.asarray(y_test) - test_mean_prediction)))
    train_baseline_mse = float(np.mean(np.square(np.asarray(y_train) - train_mean)))
    results = {
        "ridge_train": metric_summary(y_train, fitted["train_prediction"], train_baseline_mse),
        "ridge_test": metric_summary(y_test, fitted["test_prediction"], test_baseline_mse),
        "mean_gradient_test": metric_summary(y_test, test_mean_prediction, test_baseline_mse),
    }
    factor_results = {}
    for factor, factor_slice in slices.items():
        factor_baseline_mse = float(np.mean(np.square(
            np.asarray(y_test[:, factor_slice]) - test_mean_prediction[:, factor_slice]
        )))
        factor_results[factor] = {
            "ridge_test": metric_summary(
                y_test[:, factor_slice], fitted["test_prediction"][:, factor_slice],
                factor_baseline_mse,
            ),
            "mean_gradient_test": metric_summary(
                y_test[:, factor_slice], test_mean_prediction[:, factor_slice],
                factor_baseline_mse,
            ),
        }
    output = args.experiment / "ridge"
    output.mkdir(parents=True, exist_ok=True)
    for key, value in fitted.items():
        if isinstance(value, np.ndarray) and key not in {"train_prediction", "test_prediction"}:
            np.save(output / f"{key}.npy", value)
    np.save(output / "candidate_ridge_raw_gradient.npy", fitted["test_prediction"])
    summary = {
        "status": "passed",
        "model": "direct_raw_gradient_ridge",
        "gradient_projection_or_sketch": "none",
        "solver_formulation": fitted["solver_formulation"],
        "selection": {"metric": "five_fold_train_only_mean_cosine",
                      "folds": args.folds, "selected_alpha": selected_alpha,
                      "alphas": alphas, "scores": cv_summary,
                      "fold_diagnostics": fold_diagnostics},
        "results": results,
        "results_by_lora_factor": factor_results,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
