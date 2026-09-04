#!/usr/bin/env python3
"""Fit independently solved rank-constrained Ridge predictors for LoRA A and B."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_direct_raw_gradient_ridge import metric_summary, write_json
from evaluation.evaluate_low_rank_ridge_bottleneck import fit_factor_transform


def fit_factor(
    x_fit: torch.Tensor,
    y_factor: np.ndarray,
    alpha: float,
    rank: int,
    oversampling: int,
    power_iterations: int,
) -> dict:
    target_mean, target_scale = fit_factor_transform(y_factor)
    y_normalized = (np.asarray(y_factor) - target_mean) / target_scale
    target = torch.from_numpy(y_normalized.astype(np.float64)).to(x_fit.device)
    gram = x_fit.mT @ x_fit
    gram.diagonal().add_(alpha)
    coefficient = torch.cholesky_solve(x_fit.mT @ target, torch.linalg.cholesky(gram))
    fitted_response = x_fit @ coefficient
    q = min(rank + oversampling, min(fitted_response.shape))
    _, _, approximate_basis = torch.pca_lowrank(
        fitted_response, q=q, center=False, niter=power_iterations
    )
    output_basis = approximate_basis[:, :rank]
    encoder = coefficient @ output_basis
    retained_energy = float(
        torch.sum(torch.square(fitted_response @ output_basis))
        / torch.sum(torch.square(fitted_response))
    )
    return {
        "mean": target_mean,
        "scale": target_scale,
        "encoder": encoder,
        "basis": output_basis,
        "retained_energy": retained_energy,
    }


def predict_factor(x: torch.Tensor, fitted: dict) -> np.ndarray:
    normalized = (x @ fitted["encoder"]) @ fitted["basis"].mT
    prediction = normalized.cpu().numpy() * fitted["scale"] + fitted["mean"]
    return prediction.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank-a", type=int, default=64)
    parser.add_argument("--rank-b", type=int, default=64)
    parser.add_argument("--alpha-a", type=float, default=30000.0)
    parser.add_argument("--alpha-b", type=float, default=30000.0)
    parser.add_argument("--oversampling", type=int, default=16)
    parser.add_argument("--power-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.rank_a, args.rank_b) <= 0:
        parser.error("factor ranks must be positive")
    if min(args.alpha_a, args.alpha_b) < 0:
        parser.error("factor alphas must be non-negative")
    if args.oversampling < 0:
        parser.error("oversampling must be non-negative")

    started = time.time()
    torch.manual_seed(args.seed)
    extraction = args.experiment / "extraction"
    x_train = np.load(extraction / "predictor_train_hidden.npy", mmap_mode="r")
    y_train = np.load(extraction / "predictor_train_raw_gradients.npy", mmap_mode="r")
    x_test = np.load(extraction / "candidate_test_hidden.npy", mmap_mode="r")
    y_test = np.load(extraction / "candidate_test_raw_gradients.npy", mmap_mode="r")
    if x_train.ndim != 2 or y_train.ndim != 2 or x_test.ndim != 2 or y_test.ndim != 2:
        raise RuntimeError("Expected two-dimensional hidden-state and gradient arrays")
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise RuntimeError("Hidden-state and gradient sample counts do not match")
    if x_train.shape[1] != x_test.shape[1] or y_train.shape[1] != y_test.shape[1]:
        raise RuntimeError("Train and test dimensions do not match")
    if y_train.shape[1] % 2:
        raise RuntimeError("This A+B experiment requires two equally sized gradient factors")
    maximum_rank = min(x_train.shape[1], len(x_train), y_train.shape[1] // 2)
    if max(args.rank_a, args.rank_b) > maximum_rank:
        raise RuntimeError(f"Requested rank exceeds maximum available rank {maximum_rank}")

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    factor_dim = output_dim // 2
    train_a = y_train[:, :factor_dim]
    train_b = y_train[:, factor_dim:]
    test_a = y_test[:, :factor_dim]
    test_b = y_test[:, factor_dim:]
    scaler = StandardScaler().fit(x_train)
    x_fit = torch.from_numpy(scaler.transform(x_train).astype(np.float64)).to(args.device)
    x_candidate = torch.from_numpy(scaler.transform(x_test).astype(np.float64)).to(args.device)

    fitted_a = fit_factor(
        x_fit,
        train_a,
        args.alpha_a,
        args.rank_a,
        args.oversampling,
        args.power_iterations,
    )
    fitted_b = fit_factor(
        x_fit,
        train_b,
        args.alpha_b,
        args.rank_b,
        args.oversampling,
        args.power_iterations,
    )
    train_prediction_a = predict_factor(x_fit, fitted_a)
    train_prediction_b = predict_factor(x_fit, fitted_b)
    test_prediction_a = predict_factor(x_candidate, fitted_a)
    test_prediction_b = predict_factor(x_candidate, fitted_b)
    train_prediction = np.concatenate((train_prediction_a, train_prediction_b), axis=1)
    test_prediction = np.concatenate((test_prediction_a, test_prediction_b), axis=1)
    train_mean = np.concatenate((fitted_a["mean"], fitted_b["mean"]))
    mean_test_prediction = np.repeat(train_mean[None, :], len(y_test), axis=0)
    train_baseline_mse = float(np.mean(np.square(np.asarray(y_train) - train_mean)))
    test_baseline_mse = float(np.mean(np.square(np.asarray(y_test) - mean_test_prediction)))
    results = {
        "separate_ridge_bottlenecks_train": metric_summary(
            y_train, train_prediction, train_baseline_mse
        ),
        "separate_ridge_bottlenecks_test": metric_summary(
            y_test, test_prediction, test_baseline_mse
        ),
        "mean_gradient_test": metric_summary(
            y_test, mean_test_prediction, test_baseline_mse
        ),
    }
    factor_results = {}
    for factor, truth, prediction, mean in (
        ("A", test_a, test_prediction_a, fitted_a["mean"]),
        ("B", test_b, test_prediction_b, fitted_b["mean"]),
    ):
        baseline = np.repeat(mean[None, :], len(truth), axis=0)
        baseline_mse = float(np.mean(np.square(np.asarray(truth) - baseline)))
        factor_results[factor] = {
            "separate_ridge_bottleneck_test": metric_summary(
                truth, prediction, baseline_mse
            ),
            "mean_gradient_test": metric_summary(truth, baseline, baseline_mse),
        }

    output = args.experiment / f"separate_ridge_bottlenecks_rank{args.rank_a}_{args.rank_b}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "input_scaler_mean.npy", scaler.mean_.astype(np.float32))
    np.save(output / "input_scaler_scale.npy", scaler.scale_.astype(np.float32))
    for name, fitted in (("a", fitted_a), ("b", fitted_b)):
        raw_head = (
            fitted["basis"].mT.cpu().numpy().astype(np.float32) * fitted["scale"]
        )
        np.save(output / f"{name}_encoder.npy", fitted["encoder"].cpu().numpy().astype(np.float32))
        np.save(output / f"{name}_head.npy", raw_head)
        np.save(output / f"{name}_intercept.npy", fitted["mean"].astype(np.float32))
    np.save(output / "candidate_separate_ridge_bottlenecks_raw_gradient.npy", test_prediction)

    parameter_count = (
        input_dim * args.rank_a
        + args.rank_a * factor_dim
        + input_dim * args.rank_b
        + args.rank_b * factor_dim
        + output_dim
    )
    summary = {
        "status": "passed",
        "model": {
            "name": "independently_solved_reduced_rank_ridge_A_and_B",
            "architecture": {
                "A": f"standardized_{input_dim}-{args.rank_a}-linear-{factor_dim}",
                "B": f"standardized_{input_dim}-{args.rank_b}-linear-{factor_dim}",
            },
            "shared_encoder": False,
            "nonlinearity": "none",
            "dropout": 0.0,
            "trainable_parameter_equivalent": parameter_count,
            "flattened_storage_and_label": True,
            "factor_boundary": {"A": [0, factor_dim], "B": [factor_dim, output_dim]},
        },
        "fit": {
            "alpha_a": args.alpha_a,
            "alpha_b": args.alpha_b,
            "alpha_source": "independent factor-Ridge train-only CV optima unless overridden",
            "rank_a": args.rank_a,
            "rank_b": args.rank_b,
            "ridge_fitted_response_energy_retained_a": fitted_a["retained_energy"],
            "ridge_fitted_response_energy_retained_b": fitted_b["retained_energy"],
            "target_transform": "independent_A_B_train_mean_and_global_RMS_scale",
            "target_scale_a": fitted_a["scale"],
            "target_scale_b": fitted_b["scale"],
            "randomized_svd_oversampling": args.oversampling,
            "randomized_svd_power_iterations": args.power_iterations,
            "seed": args.seed,
        },
        "results": results,
        "results_by_lora_factor": factor_results,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
