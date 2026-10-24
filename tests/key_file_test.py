"""Test external watermark keys without downloading or loading pretrained weights."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def demo():
    """Load the demo when this file is installed in the project's tests directory."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "prompt_demo.py"
    spec = importlib.util.spec_from_file_location("key_file_prompt_demo", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_keys_are_copied_and_detection_uses_cpu(demo):
    upstream_keys = demo.synthid_mixin.DEFAULT_WATERMARKING_CONFIG["keys"]
    original = list(upstream_keys)
    keys = demo.load_keys(None)
    assert keys == original
    assert keys is not upstream_keys

    keys[0] += 1
    assert list(upstream_keys) == original
    config = demo.watermark_config()
    assert config["keys"] == original
    assert config["device"].type == "cpu"


def test_key_file_accepts_utf8_bom_and_int64_boundaries(demo, tmp_path):
    expected = [1234, -55, -(2**63), 2**63 - 1]
    key_file = tmp_path / "watermark_keys.txt"
    key_file.write_text(json.dumps(expected), encoding="utf-8-sig")

    assert demo.load_keys(key_file) == expected
    config = demo.watermark_config(demo.load_keys(key_file))
    assert config["keys"] == expected
    assert config["device"].type == "cpu"


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        "[true, 2]",
        "[false]",
        "[1, 1]",
        "[9223372036854775808]",
        "[-9223372036854775809]",
        '{"keys": [1, 2]}',
        '"[1, 2]"',
        "null",
        "[1.5, 2]",
        "[1,",
    ],
    ids=[
        "empty",
        "true-is-not-an-integer-key",
        "false-is-not-an-integer-key",
        "duplicates",
        "int64-overflow",
        "int64-underflow",
        "object-not-array",
        "string-not-array",
        "null-not-array",
        "float-not-integer",
        "invalid-json",
    ],
)
def test_invalid_key_files_are_rejected(demo, tmp_path, contents):
    key_file = tmp_path / "invalid_keys.txt"
    key_file.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        demo.load_keys(key_file)


def test_different_keys_change_scores_without_classifying_authorship(demo):
    tokens = list(range(101, 133))
    first_keys = [10101 + 23 * index for index in range(30)]
    other_keys = [20202 + 29 * index for index in range(30)]
    first = demo.score_tokens(tokens, eos_token_id=50256, keys=first_keys)
    other = demo.score_tokens(tokens, eos_token_id=50256, keys=other_keys)

    assert first["valid_ngram_count"] == other["valid_ngram_count"] == 28
    for result in (first, other):
        assert result["decision"] == "not_calibrated"
        assert math.isfinite(result["mean_score"])
        assert math.isfinite(result["weighted_mean_score"])
    # These are arbitrary tokens, not a generated positive sample: no ordering
    # of the two scores or claim of correct-key detection is asserted here.
    assert (first["mean_score"], first["weighted_mean_score"]) != (
        other["mean_score"], other["weighted_mean_score"]
    )


def test_configured_model_builds_warper_with_custom_keys(demo):
    """A tiny random GPT-2 checks model integration without pretrained files."""
    upstream_keys = list(demo.synthid_mixin.DEFAULT_WATERMARKING_CONFIG["keys"])
    tiny_config = demo.transformers.GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = demo.ConfiguredSynthIDGPT2LMHeadModel(tiny_config)
    custom_keys = [8101, 8203, 8309]
    model.watermarking_config = demo.watermark_config(custom_keys)
    warpers = model._construct_warper_list({"top_k": 10, "temperature": 0.5})

    assert isinstance(warpers, demo.transformers.LogitsProcessorList)
    assert len(warpers) == 1
    processor = warpers[0]
    assert isinstance(processor, demo.logits_processing.SynthIDLogitsProcessor)
    assert processor.keys.tolist() == custom_keys
    assert processor.keys.device.type == "cpu"
    assert processor.top_k == 10
    assert processor.temperature == 0.5
    assert list(demo.synthid_mixin.DEFAULT_WATERMARKING_CONFIG["keys"]) == upstream_keys
