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

import json

import numpy as np
import pytest

from benchmarks.detector_eval import benchmark


def test_threshold_respects_target_fpr_with_ties():
    negative_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.4])

    threshold = benchmark._threshold_at_fpr(negative_scores, 0.2)

    assert np.mean(negative_scores >= threshold) <= 0.2
    assert threshold > 0.4


def test_evaluate_scores_calibrates_each_target_fpr():
    negative_scores = np.linspace(0.0, 0.49, 100)
    positive_scores = np.linspace(0.51, 1.0, 100)

    result = benchmark.evaluate_scores(
        negative_scores,
        positive_scores,
        target_fprs=(0.01, 0.05),
        bootstrap_iterations=20,
        rng=np.random.default_rng(7),
    )

    assert result["roc_auc"]["estimate"] == pytest.approx(1.0)
    for calibrated in result["tpr_at_fpr"]:
        assert calibrated["empirical_fpr"] <= calibrated["target_fpr"]
        assert 0.0 <= calibrated["ci_low"] <= calibrated["ci_high"] <= 1.0


def test_run_benchmark_is_deterministic_and_serializable():
    kwargs = dict(
        token_lengths=(16, 32),
        target_fprs=(0.05,),
        sample_count=64,
        watermarking_depth=3,
        watermarked_probability=0.7,
        bootstrap_iterations=10,
        seed=123,
    )

    first = benchmark.run_benchmark(**kwargs)
    second = benchmark.run_benchmark(**kwargs)

    assert first == second
    assert first["schema_version"] == 1
    assert first["benchmark"] == "weighted_mean_synthetic_calibration"
    assert [item["effective_token_length"] for item in first["lengths"]] == [16, 32]
    json.dumps(first)


def test_run_benchmark_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="sample_count"):
        benchmark.run_benchmark(sample_count=0)
    with pytest.raises(ValueError, match="token_lengths"):
        benchmark.run_benchmark(token_lengths=())
    with pytest.raises(ValueError, match="target_fprs"):
        benchmark.run_benchmark(target_fprs=())
    with pytest.raises(ValueError, match="watermarked_probability"):
        benchmark.run_benchmark(watermarked_probability=0.5)
