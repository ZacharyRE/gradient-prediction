#!/usr/bin/env python3
"""Fit a pure-linear reduced-rank Ridge predictor with an explicit bottleneck."""

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


def fit_factor_transform(factor: np.ndarray) -> tuple[np.ndarray, float]:
    mean = np.asarray(factor, dtype=np.float64).mean(axis=0)
    scale = float(
        np.sqrt(np.mean(np.square(np.asarray(factor, dtype=np.float64) - mean)))
    )
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid target scale: {scale}")
    return mean, scale


def predict_raw(
    x_scaled: torch.Tensor,
    encoder: torch.Tensor,
    output_basis: torch.Tensor,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
) -> np.ndarray:
    normalized = (x_scaled @ encoder) @ output_basis.mT
    prediction = normalized.cpu().numpy() * target_scale + target_mean
    return prediction.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument(
        "--alpha",
        type=float,
        default=30000.0,
        help="Ridge alpha; 30000 is the train-only CV optimum of the full-rank baseline.",
    )
    parser.add_argument("--oversampling", type=int, default=16)
    parser.add_argument("--power-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rank <= 0 or args.alpha < 0 or args.oversampling < 0:
        parser.error("rank and oversampling must be positive, and alpha must be non-negative")

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
    if args.rank > min(x_train.shape[1], len(x_train), y_train.shape[1]):
        raise RuntimeError(f"Requested rank {args.rank} exceeds the available matrix dimensions")

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
    mean_a, scale_a = fit_factor_transform(train_a)
    mean_b, scale_b = fit_factor_transform(train_b)
    target_mean = np.concatenate((mean_a, mean_b))
    target_scale = np.concatenate(
        (np.full(factor_dim, scale_a), np.full(factor_dim, scale_b))
    )
    y_normalized = np.concatenate(
        (
            (np.asarray(train_a) - mean_a) / scale_a,
            (np.asarray(train_b) - mean_b) / scale_b,
        ),
        axis=1,
    )
    y_fit = torch.from_numpy(y_normalized.astype(np.float64)).to(args.device)

    # First solve the ordinary multivariate Ridge problem. Then find the leading
    # right-singular subspace of its fitted responses. Projecting the Ridge map
    # into this subspace gives an explicit rank-constrained linear map:
    #     x -> x @ encoder (rank dims) -> latent @ output_basis.T.
    gram = x_fit.mT @ x_fit
    gram.diagonal().add_(args.alpha)
    coefficient = torch.cholesky_solve(x_fit.mT @ y_fit, torch.linalg.cholesky(gram))
    ridge_fitted = x_fit @ coefficient
    q = min(args.rank + args.oversampling, min(ridge_fitted.shape))
    _, _, approximate_basis = torch.pca_lowrank(
        ridge_fitted,
        q=q,
        center=False,
        niter=args.power_iterations,
    )
    output_basis = approximate_basis[:, : args.rank]
    encoder = coefficient @ output_basis

    train_prediction = predict_raw(
        x_fit, encoder, output_basis, target_mean, target_scale
    )
    test_prediction = predict_raw(
        x_candidate, encoder, output_basis, target_mean, target_scale
    )
    mean_test_prediction = np.repeat(target_mean[None, :], len(y_test), axis=0)
    train_baseline_mse = float(np.mean(np.square(np.asarray(y_train) - target_mean)))
    test_baseline_mse = float(np.mean(np.square(np.asarray(y_test) - mean_test_prediction)))
    results = {
        "ridge_bottleneck_train": metric_summary(
            y_train, train_prediction, train_baseline_mse
        ),
        "ridge_bottleneck_test": metric_summary(y_test, test_prediction, test_baseline_mse),
        "mean_gradient_test": metric_summary(
            y_test, mean_test_prediction, test_baseline_mse
        ),
    }
    factor_results = {}
    for factor, factor_slice in {
        "A": slice(0, factor_dim),
        "B": slice(factor_dim, output_dim),
    }.items():
        factor_mean_prediction = mean_test_prediction[:, factor_slice]
        factor_baseline_mse = float(
            np.mean(
                np.square(np.asarray(y_test[:, factor_slice]) - factor_mean_prediction)
            )
        )
        factor_results[factor] = {
            "ridge_bottleneck_test": metric_summary(
                y_test[:, factor_slice],
                test_prediction[:, factor_slice],
                factor_baseline_mse,
            ),
            "mean_gradient_test": metric_summary(
                y_test[:, factor_slice], factor_mean_prediction, factor_baseline_mse
            ),
        }

    # Convert the normalized output basis into two raw-gradient heads. The saved
    # arrays therefore directly implement the deployed 1536 -> rank -> A/B map.
    raw_output_head = output_basis.mT.cpu().numpy().astype(np.float32) * target_scale
    encoder_array = encoder.cpu().numpy().astype(np.float32)
    output = args.experiment / f"ridge_bottleneck_rank{args.rank}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "input_scaler_mean.npy", scaler.mean_.astype(np.float32))
    np.save(output / "input_scaler_scale.npy", scaler.scale_.astype(np.float32))
    np.save(output / "encoder.npy", encoder_array)
    np.save(output / "a_head.npy", raw_output_head[:, :factor_dim])
    np.save(output / "b_head.npy", raw_output_head[:, factor_dim:])
    np.save(output / "a_intercept.npy", mean_a.astype(np.float32))
    np.save(output / "b_intercept.npy", mean_b.astype(np.float32))
    np.save(output / "candidate_ridge_bottleneck_raw_gradient.npy", test_prediction)

    explained_energy = float(
        torch.sum(torch.square(ridge_fitted @ output_basis))
        / torch.sum(torch.square(ridge_fitted))
    )
    parameter_count = input_dim * args.rank + args.rank * output_dim + output_dim
    summary = {
        "status": "passed",
        "model": {
            "name": "reduced_rank_ridge_with_shared_linear_bottleneck",
            "architecture": (
                f"standardized_{input_dim}-{args.rank}-linear_to_separate_"
                f"A_and_B_{factor_dim}_heads"
            ),
            "rank": args.rank,
            "trainable_parameter_equivalent": parameter_count,
            "nonlinearity": "none",
            "dropout": 0.0,
            "flattened_storage_and_label": True,
            "factor_boundary": {"A": [0, factor_dim], "B": [factor_dim, output_dim]},
            "ridge_fitted_response_energy_retained": explained_energy,
        },
        "fit": {
            "alpha": args.alpha,
            "alpha_source": "full-rank Ridge train-only CV optimum unless overridden",
            "target_transform": "separate_A_B_train_mean_and_global_RMS_scale",
            "target_scale_a": scale_a,
            "target_scale_b": scale_b,
            "rank_reduction": "right singular subspace of Ridge fitted train responses",
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
