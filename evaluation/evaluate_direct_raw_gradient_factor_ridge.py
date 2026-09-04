#!/usr/bin/env python3
"""Select separate Ridge penalties for raw LoRA A and B gradient blocks."""

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


def safe_norm_correlation(true_norm: np.ndarray, predicted_norm: np.ndarray):
    if true_norm.std() <= 1e-12 or predicted_norm.std() <= 1e-12:
        return None
    return float(np.corrcoef(true_norm, predicted_norm)[0, 1])


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, baseline_mse: float) -> dict:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    mse = float(np.mean(np.square(error)))
    cosines = row_cosine(np.asarray(y_pred), np.asarray(y_true))
    true_norm = np.linalg.norm(y_true, axis=1)
    predicted_norm = np.linalg.norm(y_pred, axis=1)
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
        "gradient_norm_pearson": safe_norm_correlation(true_norm, predicted_norm),
        "true_norm_mean": float(true_norm.mean()),
        "predicted_norm_mean": float(predicted_norm.mean()),
    }


def block_statistics(prediction: np.ndarray, truth: np.ndarray, block: slice) -> dict:
    predicted = np.asarray(prediction[:, block], dtype=np.float64)
    target = np.asarray(truth[:, block], dtype=np.float64)
    dot = np.einsum("ij,ij->i", predicted, target)
    predicted_sq = np.einsum("ij,ij->i", predicted, predicted)
    target_sq = np.einsum("ij,ij->i", target, target)
    denominator = np.sqrt(predicted_sq * target_sq)
    cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 1e-24)
    return {"dot": dot, "predicted_sq": predicted_sq, "target_sq": target_sq,
            "mean_cosine": float(cosine.mean())}


def fold_path_statistics(x_fit_raw: np.ndarray, y_fit_raw: np.ndarray,
                         x_validation_raw: np.ndarray, y_validation: np.ndarray,
                         alphas: list[float], device: str, slices: dict[str, slice]):
    scaler = StandardScaler().fit(x_fit_raw)
    x_fit = torch.from_numpy(scaler.transform(x_fit_raw).astype(np.float64)).to(device)
    x_validation = torch.from_numpy(
        scaler.transform(x_validation_raw).astype(np.float64)
    ).to(device)
    y_mean = np.asarray(y_fit_raw, dtype=np.float64).mean(axis=0)
    y_fit = torch.from_numpy(np.asarray(y_fit_raw, dtype=np.float64) - y_mean).to(device)
    eigenvalues, eigenvectors = torch.linalg.eigh(x_fit.mT @ x_fit)
    transformed_rhs = eigenvectors.mT @ (x_fit.mT @ y_fit)
    validation_basis = x_validation @ eigenvectors
    statistics = {"A": {}, "B": {}}
    for alpha in alphas:
        coefficient_basis = transformed_rhs / (eigenvalues[:, None] + alpha)
        prediction = (validation_basis @ coefficient_basis).cpu().numpy() + y_mean
        statistics["A"][str(alpha)] = block_statistics(prediction, y_validation, slices["A"])
        statistics["B"][str(alpha)] = block_statistics(prediction, y_validation, slices["B"])
    diagnostics = {"xtx_eigenvalue_min": float(eigenvalues.min().cpu()),
                   "xtx_eigenvalue_max": float(eigenvalues.max().cpu())}
    del x_fit, x_validation, y_fit, eigenvalues, eigenvectors, transformed_rhs, validation_basis
    torch.cuda.empty_cache()
    return statistics, diagnostics


def combined_cosine(a_stats: dict, b_stats: dict) -> float:
    numerator = a_stats["dot"] + b_stats["dot"]
    denominator = np.sqrt(
        (a_stats["predicted_sq"] + b_stats["predicted_sq"])
        * (a_stats["target_sq"] + b_stats["target_sq"])
    )
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                       where=denominator > 1e-24)
    return float(values.mean())


def fit_final(x_train_raw: np.ndarray, y_train_raw: np.ndarray, x_test_raw: np.ndarray,
              alpha_a: float, alpha_b: float, device: str, slices: dict[str, slice]):
    scaler = StandardScaler().fit(x_train_raw)
    x_train = torch.from_numpy(scaler.transform(x_train_raw).astype(np.float64)).to(device)
    x_test = torch.from_numpy(scaler.transform(x_test_raw).astype(np.float64)).to(device)
    y_mean = np.asarray(y_train_raw, dtype=np.float64).mean(axis=0)
    y_centered = torch.from_numpy(np.asarray(y_train_raw, dtype=np.float64) - y_mean).to(device)
    gram = x_train.mT @ x_train
    rhs = x_train.mT @ y_centered

    gram_a = gram.clone()
    gram_a.diagonal().add_(alpha_a)
    coefficient_a = torch.cholesky_solve(rhs[:, slices["A"]], torch.linalg.cholesky(gram_a))
    gram_b = gram.clone()
    gram_b.diagonal().add_(alpha_b)
    coefficient_b = torch.cholesky_solve(rhs[:, slices["B"]], torch.linalg.cholesky(gram_b))
    coefficient = torch.cat([coefficient_a, coefficient_b], dim=1)
    train_prediction = (x_train @ coefficient).cpu().numpy() + y_mean
    test_prediction = (x_test @ coefficient).cpu().numpy() + y_mean
    result = {
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "coefficient_a": coefficient_a.cpu().numpy().astype(np.float32),
        "coefficient_b": coefficient_b.cpu().numpy().astype(np.float32),
        "intercept": y_mean.astype(np.float32),
        "train_prediction": train_prediction.astype(np.float32),
        "test_prediction": test_prediction.astype(np.float32),
    }
    del x_train, x_test, y_centered, gram, rhs, gram_a, gram_b, coefficient_a, coefficient_b,
    coefficient
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[1000, 3000, 10000, 30000, 100000, 300000, 1000000])
    args = parser.parse_args()
    started = time.time()
    slices = factor_slices(args.experiment)
    extraction = args.experiment / "extraction"
    x_train = np.load(extraction / "predictor_train_hidden.npy", mmap_mode="r")
    y_train = np.load(extraction / "predictor_train_raw_gradients.npy", mmap_mode="r")
    x_test = np.load(extraction / "candidate_test_hidden.npy", mmap_mode="r")
    y_test = np.load(extraction / "candidate_test_raw_gradients.npy", mmap_mode="r")
    if (
        x_train.ndim != 2
        or y_train.ndim != 2
        or len(x_train) != len(y_train)
        or x_test.ndim != 2
        or y_test.ndim != 2
        or len(x_test) != len(y_test)
        or x_train.shape[1] != x_test.shape[1]
        or y_train.shape[1] != y_test.shape[1]
        or y_train.shape[1] != slices["B"].stop
    ):
        raise RuntimeError(f"Unexpected train shapes: {x_train.shape}, {y_train.shape}")
    alphas = [float(alpha) for alpha in args.alphas]
    pair_scores = {f"A={alpha_a},B={alpha_b}": []
                   for alpha_a in alphas for alpha_b in alphas}
    factor_scores = {"A": {str(alpha): [] for alpha in alphas},
                     "B": {str(alpha): [] for alpha in alphas}}
    diagnostics = []
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (fit_indices, validation_indices) in enumerate(splitter.split(x_train)):
        statistics, fold_diagnostics = fold_path_statistics(
            np.asarray(x_train[fit_indices]), np.asarray(y_train[fit_indices]),
            np.asarray(x_train[validation_indices]), np.asarray(y_train[validation_indices]),
            alphas, args.device, slices,
        )
        diagnostics.append({"fold": fold, **fold_diagnostics})
        for factor in ("A", "B"):
            for alpha in alphas:
                factor_scores[factor][str(alpha)].append(
                    statistics[factor][str(alpha)]["mean_cosine"]
                )
        for alpha_a in alphas:
            for alpha_b in alphas:
                pair_scores[f"A={alpha_a},B={alpha_b}"].append(combined_cosine(
                    statistics["A"][str(alpha_a)], statistics["B"][str(alpha_b)]
                ))
    pair_summary = {key: {"fold_mean_cosines": scores,
                          "mean_cosine": float(np.mean(scores)),
                          "std_cosine": float(np.std(scores))}
                    for key, scores in pair_scores.items()}
    selected_key = max(pair_summary, key=lambda key: pair_summary[key]["mean_cosine"])
    selected_alpha_a = float(selected_key.split(",")[0].split("=")[1])
    selected_alpha_b = float(selected_key.split(",")[1].split("=")[1])
    factor_summary = {
        factor: {alpha: {"fold_mean_cosines": scores,
                         "mean_cosine": float(np.mean(scores)),
                         "std_cosine": float(np.std(scores))}
                 for alpha, scores in by_alpha.items()}
        for factor, by_alpha in factor_scores.items()
    }
    independent_factor_optima = {
        factor: float(max(alphas, key=lambda alpha: factor_summary[factor][str(alpha)]["mean_cosine"]))
        for factor in ("A", "B")
    }
    fitted = fit_final(
        x_train, y_train, x_test, selected_alpha_a, selected_alpha_b, args.device, slices
    )
    train_mean = np.asarray(y_train, dtype=np.float64).mean(axis=0)
    mean_test_prediction = np.repeat(train_mean[None, :], len(y_test), axis=0)
    train_baseline_mse = float(np.mean(np.square(np.asarray(y_train) - train_mean)))
    test_baseline_mse = float(np.mean(np.square(np.asarray(y_test) - mean_test_prediction)))
    results = {
        "factor_ridge_train": metric_summary(
            y_train, fitted["train_prediction"], train_baseline_mse
        ),
        "factor_ridge_test": metric_summary(
            y_test, fitted["test_prediction"], test_baseline_mse
        ),
        "mean_gradient_test": metric_summary(y_test, mean_test_prediction, test_baseline_mse),
    }
    factor_results = {}
    for factor, block in slices.items():
        baseline_mse = float(np.mean(np.square(
            np.asarray(y_test[:, block]) - mean_test_prediction[:, block]
        )))
        factor_results[factor] = {
            "factor_ridge_test": metric_summary(
                y_test[:, block], fitted["test_prediction"][:, block], baseline_mse
            ),
            "mean_gradient_test": metric_summary(
                y_test[:, block], mean_test_prediction[:, block], baseline_mse
            ),
        }
    shared_summary_path = args.experiment / "ridge" / "summary.json"
    shared_comparison = None
    if shared_summary_path.exists():
        shared = json.loads(shared_summary_path.read_text(encoding="utf-8"))
        shared_comparison = {
            "selected_alpha": shared["selection"]["selected_alpha"],
            "test": shared["results"]["ridge_test"],
        }
    output = args.experiment / "factor_ridge"
    output.mkdir(parents=True, exist_ok=True)
    for key in ("scaler_mean", "scaler_scale", "coefficient_a", "coefficient_b", "intercept"):
        np.save(output / f"{key}.npy", fitted[key])
    np.save(output / "candidate_factor_ridge_raw_gradient.npy", fitted["test_prediction"])
    summary = {
        "status": "passed",
        "model": "direct_raw_gradient_factor_ridge",
        "objective": "separate L2 penalties for raw LoRA A and B output blocks",
        "gradient_projection_or_sketch": "none",
        "selection": {
            "metric": "five_fold_train_only_combined_raw_A_plus_B_mean_cosine",
            "folds": args.folds,
            "alphas": alphas,
            "selected_alpha_A": selected_alpha_a,
            "selected_alpha_B": selected_alpha_b,
            "selected_pair_cv": pair_summary[selected_key],
            "independent_factor_optima": independent_factor_optima,
            "factor_scores": factor_summary,
            "pair_scores": pair_summary,
            "fold_diagnostics": diagnostics,
        },
        "results": results,
        "results_by_lora_factor": factor_results,
        "shared_alpha_baseline": shared_comparison,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
