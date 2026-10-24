"""Test the demo detector without downloading or loading GPT-2 weights."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def demo():
    script = Path(__file__).resolve().parents[1] / "scripts" / "prompt_demo.py"
    spec = importlib.util.spec_from_file_location("stage2_prompt_demo", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_text_returns_no_score(demo):
    ngram_len = demo.watermark_config()["ngram_len"]
    for tokens in ([], [71] * (ngram_len - 1)):
        result = demo.score_tokens(tokens, eos_token_id=50256)
        assert result["decision"] == "insufficient_text"
        assert result["valid_ngram_count"] == 0
        assert result["mean_score"] is None
        assert result["weighted_mean_score"] is None


def test_eos_and_its_suffix_do_not_change_scores(demo):
    prefix = [101, 202, 303, 404, 505, 606, 707, 808, 909]
    clean = demo.score_tokens(prefix, eos_token_id=50256)
    terminated = demo.score_tokens(
        prefix + [50256, 111, 222, 333, 444, 555, 666],
        eos_token_id=50256,
    )
    assert clean["valid_ngram_count"] > 0
    assert terminated["valid_ngram_count"] == clean["valid_ngram_count"]
    assert terminated["mean_score"] == pytest.approx(clean["mean_score"], abs=1e-7)
    assert terminated["weighted_mean_score"] == pytest.approx(
        clean["weighted_mean_score"], abs=1e-7
    )
    assert terminated["decision"] == "not_calibrated"


def test_scores_are_finite_and_deterministic(demo):
    tokens = [31, 41, 59, 26, 53, 58, 97, 93, 23, 84, 62, 64, 33]
    first = demo.score_tokens(tokens, eos_token_id=50256)
    second = demo.score_tokens(tokens, eos_token_id=50256)
    assert first == second
    assert first["valid_ngram_count"] > 0
    for key in ("mean_score", "weighted_mean_score"):
        assert math.isfinite(first[key])
        assert 0 <= first[key] <= 1
    assert first["decision"] == "not_calibrated"


def test_repeated_context_is_excluded_from_official_scores(demo):
    # The context [11, 12, 13, 14] occurs before 21 and later before 22.
    # Only the first occurrence should contribute to the scores.
    # The four contexts between them are distinct and remain valid.
    config = demo.watermark_config()
    assert config["ngram_len"] == 5
    tokens = [11, 12, 13, 14, 21, 11, 12, 13, 14, 22]
    expected_mask = demo.np.array([[True, True, True, True, True, False]])
    processor = demo.logits_processing.SynthIDLogitsProcessor(
        **config, top_k=40, temperature=0.7
    )
    ids = demo.torch.tensor([tokens], dtype=demo.torch.long)
    g_values = processor.compute_g_values(ids).numpy().astype(demo.np.float32)
    expected_mean = float(demo.detector_mean.mean_score(g_values.copy(), expected_mask)[0])
    expected_weighted = float(
        demo.detector_mean.weighted_mean_score(g_values.copy(), expected_mask)[0]
    )

    result = demo.score_tokens(tokens, eos_token_id=50256)
    assert result["valid_ngram_count"] == 5
    assert result["mean_score"] == pytest.approx(expected_mean, abs=1e-7)
    assert result["weighted_mean_score"] == pytest.approx(expected_weighted, abs=1e-7)
