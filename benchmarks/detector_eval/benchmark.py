# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Reproducible calibration benchmark for the Weighted Mean detector."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from sklearn import metrics

from synthid_text import detector_mean

_SCHEMA_VERSION = 1


def _parse_csv(value: str, cast):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def _threshold_at_fpr(negative_scores: np.ndarray, target_fpr: float) -> float:
    """Returns the least restrictive threshold with empirical FPR <= target."""
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    if negative_scores.ndim != 1 or negative_scores.size == 0:
        raise ValueError("negative_scores must be a non-empty one-dimensional array")
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must be between 0 and 1")

    allowed_false_positives = math.floor(target_fpr * negative_scores.size)
    if allowed_false_positives >= negative_scores.size:
        return float(np.min(negative_scores))
    if allowed_false_positives == 0:
        return float(np.nextafter(np.max(negative_scores), np.inf))

    descending = np.sort(negative_scores)[::-1]
    cutoff = descending[allowed_false_positives - 1]
    if np.count_nonzero(negative_scores >= cutoff) <= allowed_false_positives:
        return float(cutoff)
    return float(np.nextafter(cutoff, np.inf))


def _score_fixture(
    rng: np.random.Generator,
    sample_count: int,
    token_length: int,
    watermarking_depth: int,
    watermarked_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic g-values and scores both fixture classes."""
    shape = (sample_count, token_length, watermarking_depth)
    unwatermarked = rng.binomial(1, 0.5, size=shape).astype(np.float32)
    watermarked = rng.binomial(1, watermarked_probability, size=shape).astype(
        np.float32
    )
    mask = np.ones((sample_count, token_length), dtype=np.float32)

    negative_scores = detector_mean.weighted_mean_score(
        jnp.asarray(unwatermarked), jnp.asarray(mask)
    )
    positive_scores = detector_mean.weighted_mean_score(
        jnp.asarray(watermarked), jnp.asarray(mask)
    )
    return np.asarray(negative_scores), np.asarray(positive_scores)


def _bootstrap_metrics(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    target_fprs: Sequence[float],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Bootstraps ROC-AUC and TPR at recalibrated target FPRs."""
    if iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")

    auc_samples = np.empty(iterations, dtype=np.float64)
    tpr_samples = {
        target_fpr: np.empty(iterations, dtype=np.float64) for target_fpr in target_fprs
    }
    labels = np.concatenate(
        [np.zeros(negative_scores.size), np.ones(positive_scores.size)]
    )

    for iteration in range(iterations):
        negative_sample = rng.choice(
            negative_scores, size=negative_scores.size, replace=True
        )
        positive_sample = rng.choice(
            positive_scores, size=positive_scores.size, replace=True
        )
        scores = np.concatenate([negative_sample, positive_sample])
        auc_samples[iteration] = metrics.roc_auc_score(labels, scores)
        for target_fpr in target_fprs:
            threshold = _threshold_at_fpr(negative_sample, target_fpr)
            tpr_samples[target_fpr][iteration] = np.mean(positive_sample >= threshold)

    return auc_samples, tpr_samples


def _confidence_interval(samples: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"ci_low": float(low), "ci_high": float(high)}


def evaluate_scores(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    target_fprs: Sequence[float],
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Evaluates detector scores with calibrated thresholds and bootstrap CIs."""
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    if negative_scores.ndim != 1 or positive_scores.ndim != 1:
        raise ValueError("score arrays must be one-dimensional")
    if negative_scores.size == 0 or positive_scores.size == 0:
        raise ValueError("score arrays must be non-empty")

    labels = np.concatenate(
        [np.zeros(negative_scores.size), np.ones(positive_scores.size)]
    )
    scores = np.concatenate([negative_scores, positive_scores])
    auc = float(metrics.roc_auc_score(labels, scores))
    auc_samples, tpr_samples = _bootstrap_metrics(
        negative_scores,
        positive_scores,
        target_fprs,
        bootstrap_iterations,
        rng,
    )

    tpr_at_fpr = []
    for target_fpr in target_fprs:
        threshold = _threshold_at_fpr(negative_scores, target_fpr)
        empirical_fpr = float(np.mean(negative_scores >= threshold))
        tpr = float(np.mean(positive_scores >= threshold))
        tpr_at_fpr.append(
            {
                "target_fpr": float(target_fpr),
                "threshold": threshold,
                "empirical_fpr": empirical_fpr,
                "tpr": tpr,
                **_confidence_interval(tpr_samples[target_fpr]),
            }
        )

    return {
        "roc_auc": {"estimate": auc, **_confidence_interval(auc_samples)},
        "tpr_at_fpr": tpr_at_fpr,
    }


def run_benchmark(
    token_lengths: Sequence[int] = (32, 64, 128, 256, 512),
    target_fprs: Sequence[float] = (0.001, 0.01),
    sample_count: int = 1000,
    watermarking_depth: int = 5,
    watermarked_probability: float = 0.65,
    bootstrap_iterations: int = 200,
    seed: int = 20260831,
) -> dict[str, object]:
    """Runs the deterministic synthetic Weighted Mean calibration benchmark."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if watermarking_depth <= 0:
        raise ValueError("watermarking_depth must be positive")
    if not 0.5 < watermarked_probability <= 1.0:
        raise ValueError("watermarked_probability must be in (0.5, 1.0]")
    if not token_lengths or any(length <= 0 for length in token_lengths):
        raise ValueError("token_lengths must contain positive values")
    if not target_fprs:
        raise ValueError("target_fprs must not be empty")

    rng = np.random.default_rng(seed)
    length_results = []
    for token_length in token_lengths:
        negative_scores, positive_scores = _score_fixture(
            rng,
            sample_count,
            token_length,
            watermarking_depth,
            watermarked_probability,
        )
        length_results.append(
            {
                "effective_token_length": int(token_length),
                "metrics": evaluate_scores(
                    negative_scores,
                    positive_scores,
                    target_fprs,
                    bootstrap_iterations,
                    rng,
                ),
            }
        )

    return {
        "schema_version": _SCHEMA_VERSION,
        "benchmark": "weighted_mean_synthetic_calibration",
        "fixture": {
            "unwatermarked_g_probability": 0.5,
            "watermarked_g_probability": float(watermarked_probability),
            "sample_count_per_class": int(sample_count),
            "watermarking_depth": int(watermarking_depth),
        },
        "config": {
            "seed": int(seed),
            "bootstrap_iterations": int(bootstrap_iterations),
            "target_fprs": [float(value) for value in target_fprs],
        },
        "lengths": length_results,
    }


def _print_summary(result: dict[str, object]) -> None:
    print("length\troc_auc\ttarget_fpr\tempirical_fpr\ttpr\tthreshold")
    for length_result in result["lengths"]:
        auc = length_result["metrics"]["roc_auc"]["estimate"]
        for tpr_result in length_result["metrics"]["tpr_at_fpr"]:
            print(
                f"{length_result['effective_token_length']}\t{auc:.6f}\t"
                f"{tpr_result['target_fpr']:.6g}\t{tpr_result['empirical_fpr']:.6g}\t"
                f"{tpr_result['tpr']:.6f}\t{tpr_result['threshold']:.8f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="32,64,128,256,512")
    parser.add_argument("--fpr", default="0.001,0.01")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--watermarking-depth", type=int, default=5)
    parser.add_argument("--watermarked-probability", type=float, default=0.65)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    result = run_benchmark(
        token_lengths=_parse_csv(args.lengths, int),
        target_fprs=_parse_csv(args.fpr, float),
        sample_count=args.samples,
        watermarking_depth=args.watermarking_depth,
        watermarked_probability=args.watermarked_probability,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    _print_summary(result)
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
