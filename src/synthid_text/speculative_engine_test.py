# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for the dual-key speculative decoding engine."""

import collections
from typing import Any

from absl.testing import absltest
import numpy as np
from scipy import stats
import torch
import transformers

from synthid_text import detector_frequentist
from synthid_text import speculative_engine
from synthid_text import torch_testing


def _make_config(**kwargs) -> speculative_engine.DualKeyWatermarkingConfig:
  device = torch_testing.torch_device()
  return speculative_engine.DualKeyWatermarkingConfig(
      watermark_device=device,
      target_device=device,
      draft_device=device,
      **kwargs,
  )


class _UncachedLogitsChecker(torch.nn.Module):
  """Wraps a causal LM and compares cached logits to an uncached recomputation.

  The engine only feeds the tokens not yet in the KV cache, and cropping only
  ever removes tokens from the end of the cache. The cache length before each
  call is therefore the number of retained prefix tokens, which lets the
  wrapper reconstruct the full sequence and recompute its logits without a
  cache.
  """

  def __init__(self, model: transformers.PreTrainedModel):
    super().__init__()
    self.model = model
    self.config = model.config
    self.tokens: torch.Tensor | None = None
    self.max_abs_diff = 0.0
    self.num_calls = 0

  def forward(self, input_ids, past_key_values, use_cache):
    offset = past_key_values.get_seq_length()
    if self.tokens is None:
      self.tokens = input_ids
    else:
      self.tokens = torch.cat([self.tokens[:, :offset], input_ids], dim=1)
    out = self.model(
        input_ids, past_key_values=past_key_values, use_cache=use_cache
    )
    ref_logits = self.model(self.tokens, use_cache=False).logits[:, offset:]
    self.max_abs_diff = max(
        self.max_abs_diff, float((out.logits - ref_logits).abs().max())
    )
    self.num_calls += 1
    return out


class SpeculativeEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.device = torch_testing.torch_device()
    if self.device.type == "cpu":
      torch.set_num_threads(1)
    self.vocab_size = 120
    torch.manual_seed(0)
    target_config = transformers.GPT2Config(
        vocab_size=self.vocab_size,
        n_positions=256,
        n_embd=32,
        n_layer=2,
        n_head=2,
    )
    draft_config = transformers.GPT2Config(
        vocab_size=self.vocab_size,
        n_positions=256,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    self.target_model = (
        transformers.GPT2LMHeadModel(target_config).eval().to(self.device)
    )
    self.draft_model = (
        transformers.GPT2LMHeadModel(draft_config).eval().to(self.device)
    )

  def test_config_validation(self):
    with self.assertRaisesRegex(
        ValueError, "temperature must be strictly positive"
    ):
      speculative_engine.DualKeyWatermarkingConfig(temperature=0.0)
    with self.assertRaisesRegex(ValueError, "top_k must be >= 2"):
      speculative_engine.DualKeyWatermarkingConfig(top_k=1)
    with self.assertRaisesRegex(ValueError, "draft_len must be >= 1"):
      speculative_engine.DualKeyWatermarkingConfig(draft_len=0)
    with self.assertRaisesRegex(ValueError, "key_draft and key_target"):
      speculative_engine.DualKeyWatermarkingConfig(key_draft=7, key_target=7)
    with self.assertRaisesRegex(ValueError, "num_leaves must be 2"):
      speculative_engine.DualKeyWatermarkingConfig(num_leaves=3)
    other_model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=self.vocab_size + 1,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
    )
    with self.assertRaisesRegex(ValueError, "same vocab_size"):
      speculative_engine.DualKeySpeculativeEngine(
          self.target_model, other_model, _make_config()
      )

  def test_speculative_generation_end_to_end(self):
    config = _make_config(
        ngram_len=5,
        key_draft=42,
        key_target=84,
        depth=10,
        top_k=30,
        draft_len=3,
    )
    engine = speculative_engine.DualKeySpeculativeEngine(
        self.target_model, self.draft_model, config
    )

    generator = torch.Generator(device=self.device).manual_seed(123)
    prompt_ids = torch.randint(
        2, self.vocab_size, (1, 8), generator=generator, device=self.device
    )
    max_new_tokens = 15

    output = engine.generate(
        prompt_ids, max_new_tokens=max_new_tokens, generator=generator
    )

    self.assertEqual(output.tokens_generated, max_new_tokens)
    self.assertEqual(
        output.sequences.shape, (1, prompt_ids.shape[1] + max_new_tokens)
    )
    self.assertGreater(output.num_draft_tokens, 0)
    self.assertBetween(output.acceptance_rate, 0.0, 1.0)
    self.assertLen(output.token_origins, max_new_tokens)

    valid_origins = {"draft", "target_residual", "target_bonus"}
    generated_tokens = output.sequences[0, prompt_ids.shape[1] :].tolist()
    for (origin, tok_id), expected_tok in zip(
        output.token_origins, generated_tokens, strict=True
    ):
      self.assertIn(origin, valid_origins)
      self.assertEqual(tok_id, expected_tok)

  def test_speculative_generation_reproducibility_and_edge_cases(self):
    config = _make_config(
        ngram_len=4,
        key_draft=11,
        key_target=22,
        depth=8,
        top_k=20,
        draft_len=3,
    )
    engine = speculative_engine.DualKeySpeculativeEngine(
        self.target_model, self.draft_model, config
    )
    prompt_ids = torch.tensor([[10, 11, 12, 13, 14]], device=self.device)

    # Zero new tokens returns the prompt unchanged.
    zero_out = engine.generate(prompt_ids, max_new_tokens=0)
    self.assertEqual(zero_out.tokens_generated, 0)
    self.assertTrue(torch.equal(zero_out.sequences, prompt_ids))

    # Same generator seed produces identical sequences and token origins.
    gen1 = torch.Generator(device=self.device).manual_seed(99)
    gen2 = torch.Generator(device=self.device).manual_seed(99)
    out1 = engine.generate(prompt_ids, max_new_tokens=10, generator=gen1)
    out2 = engine.generate(prompt_ids, max_new_tokens=10, generator=gen2)
    self.assertTrue(torch.equal(out1.sequences, out2.sequences))
    self.assertEqual(out1.token_origins, out2.token_origins)

    # Early stopping works with a sequence of EOS token IDs.
    first_tok = int(out1.sequences[0, prompt_ids.shape[1]].item())
    gen3 = torch.Generator(device=self.device).manual_seed(99)
    out_eos = engine.generate(
        prompt_ids,
        max_new_tokens=10,
        eos_token_id=[self.vocab_size + 1, first_tok],
        generator=gen3,
    )
    self.assertEqual(out_eos.tokens_generated, 1)
    self.assertEqual(int(out_eos.sequences[0, -1].item()), first_tok)

    # Also supports a default CPU generator regardless of self.device.
    cpu_gen = torch.Generator().manual_seed(99)
    out_cpu_gen = engine.generate(
        prompt_ids, max_new_tokens=5, generator=cpu_gen
    )
    self.assertEqual(out_cpu_gen.tokens_generated, 5)

    # Invalid input shapes raise ValueError.
    with self.assertRaisesRegex(ValueError, "prompt_ids must have shape"):
      engine.generate(
          torch.ones((2, 5), dtype=torch.long, device=self.device),
          max_new_tokens=5,
      )
    with self.assertRaisesRegex(ValueError, "prompt_ids must have shape"):
      engine.generate(
          torch.ones((1, 0), dtype=torch.long, device=self.device),
          max_new_tokens=5,
      )
    with self.assertRaisesRegex(
        ValueError, "max_new_tokens must be a non-negative int"
    ):
      engine.generate(prompt_ids, max_new_tokens=-1)

  def test_speculative_watermarked_text_has_small_p_value(self):
    config = _make_config(
        ngram_len=4,
        key_draft=42,
        key_target=84,
        depth=30,
        top_k=40,
        draft_len=3,
    )
    engine = speculative_engine.DualKeySpeculativeEngine(
        self.target_model, self.draft_model, config
    )
    eos_token_id = self.vocab_size + 1
    detector = detector_frequentist.DualKeyFrequentistDetector.from_config(
        config, eos_token_id
    )
    wrong_key_detector = (
        detector_frequentist.DualKeyFrequentistDetector.from_config(
            _make_config(ngram_len=4, key_draft=100, key_target=200, depth=30),
            eos_token_id,
        )
    )

    generator = torch.Generator(device=self.device).manual_seed(7)
    prompt_ids = torch.randint(
        2, self.vocab_size, (1, 6), generator=generator, device=self.device
    )
    output = engine.generate(
        prompt_ids, max_new_tokens=200, generator=generator
    )
    generated_ids = output.sequences[:, prompt_ids.shape[1] :]
    self.assertLess(detector.p_value(generated_ids)[0], 1e-6)
    self.assertGreater(wrong_key_detector.p_value(generated_ids)[0], 0.05)

  def _top_k_probs(
      self,
      model: torch.nn.Module,
      prompt_ids: torch.LongTensor,
      top_k: int,
      temperature: float,
  ) -> torch.Tensor:
    with torch.no_grad():
      logits = model(prompt_ids).logits[0, -1] / temperature
    values, indices = torch.topk(logits, top_k)
    return torch.zeros_like(logits).scatter(
        0, indices, torch.softmax(values, dim=-1)
    )

  def test_first_token_follows_top_k_target_distribution(self):
    # With skip_first_ngram_calls=True the first token is not watermarked, so
    # speculative sampling must reproduce the top-k target distribution p(x)
    # and acceptance rate sum_x min(p(x), q(x)).
    top_k = 5
    # A low temperature makes the tiny random models' distributions peaked
    # enough that the draft and target top-k distributions differ
    # substantially, while the top-k still truncates a large part of the
    # full distributions.
    temperature = 0.2
    config = _make_config(
        ngram_len=4, top_k=top_k, temperature=temperature, draft_len=1
    )
    engine = speculative_engine.DualKeySpeculativeEngine(
        self.target_model, self.draft_model, config
    )
    prompt_ids = torch.tensor([[10, 11, 12, 13, 14]], device=self.device)

    p = self._top_k_probs(self.target_model, prompt_ids, top_k, temperature)
    q = self._top_k_probs(self.draft_model, prompt_ids, top_k, temperature)
    # Guard the power of the test: the draft must be a poor match.
    self.assertGreater(float(0.5 * (p - q).abs().sum()), 0.3)
    expected_acceptance = float(torch.minimum(p, q).sum())

    generator = torch.Generator(device=self.device).manual_seed(0)
    num_samples = 4000
    num_accepted = 0
    counts: collections.Counter[int] = collections.Counter()
    for _ in range(num_samples):
      out = engine.generate(prompt_ids, max_new_tokens=1, generator=generator)
      num_accepted += out.num_accepted_tokens
      counts[int(out.sequences[0, -1])] += 1

    self.assertAlmostEqual(
        num_accepted / num_samples, expected_acceptance, delta=0.015
    )
    support = torch.nonzero(p).squeeze(1).tolist()
    self.assertContainsSubset(counts, support)
    observed = np.array([counts[i] for i in support], dtype=np.float64)
    expected_probs = p[support].cpu().double().numpy()
    expected_probs = expected_probs / expected_probs.sum()
    np.testing.assert_allclose(
        observed / num_samples, expected_probs, atol=0.015
    )
    _, chi2_p_value = stats.chisquare(observed, expected_probs * num_samples)
    self.assertGreater(chi2_p_value, 1e-2)

  def test_watermarking_preserves_acceptance_rate_and_target_distribution(self):
    # With skip_first_ngram_calls=False the first token is watermarked. For a
    # single fixed key pair, tournament watermarking concentrates mass on the
    # tournament winners; averaged over key pairs, binary tournament
    # watermarking (num_leaves=2) is non-distortionary, so both the acceptance
    # rate sum_x min(p(x), q(x)) and the marginal token distribution p(x) match
    # unwatermarked speculative sampling.
    top_k = 5
    temperature = 0.2
    prompt_ids = torch.tensor([[10, 11, 12, 13, 14]], device=self.device)

    p = self._top_k_probs(self.target_model, prompt_ids, top_k, temperature)
    q = self._top_k_probs(self.draft_model, prompt_ids, top_k, temperature)
    expected_acceptance = float(torch.minimum(p, q).sum())
    self.assertBetween(expected_acceptance, 0.1, 0.7)
    support = torch.nonzero(p).squeeze(1).tolist()
    expected_probs = p[support].cpu().double().numpy()
    expected_probs = expected_probs / expected_probs.sum()

    # Guard that watermarking is active on the first token: for a single fixed
    # key pair, the token distribution differs substantially from p(x).
    fixed_gen = torch.Generator(device=self.device).manual_seed(0)
    fixed_engine = speculative_engine.DualKeySpeculativeEngine(
        self.target_model,
        self.draft_model,
        _make_config(
            ngram_len=4,
            key_draft=42,
            key_target=84,
            depth=30,
            top_k=top_k,
            temperature=temperature,
            draft_len=1,
            skip_first_ngram_calls=False,
        ),
    )
    fixed_counts = collections.Counter(
        int(
            fixed_engine.generate(
                prompt_ids, max_new_tokens=1, generator=fixed_gen
            ).sequences[0, -1]
        )
        for _ in range(200)
    )
    fixed_probs = np.array([fixed_counts[i] / 200 for i in support])
    self.assertGreater(
        float(0.5 * np.abs(fixed_probs - expected_probs).sum()), 0.25
    )

    # Averaged across key pairs, both the acceptance rate and the marginal
    # token distribution match unwatermarked speculative sampling.
    generator = torch.Generator(device=self.device).manual_seed(0)
    num_samples = 4000
    num_accepted = 0
    counts: collections.Counter[int] = collections.Counter()
    config = _make_config(
        ngram_len=4,
        key_draft=0,
        key_target=1,
        depth=8,
        top_k=top_k,
        temperature=temperature,
        draft_len=1,
        skip_first_ngram_calls=False,
    )
    engine = speculative_engine.DualKeySpeculativeEngine(
        self.target_model, self.draft_model, config
    )
    for i in range(num_samples):
      config.key_draft = i
      config.key_target = i + num_samples
      out = engine.generate(prompt_ids, max_new_tokens=1, generator=generator)
      num_accepted += out.num_accepted_tokens
      counts[int(out.sequences[0, -1])] += 1

    self.assertAlmostEqual(
        num_accepted / num_samples, expected_acceptance, delta=0.015
    )
    self.assertContainsSubset(counts, support)
    observed = np.array([counts[i] for i in support], dtype=np.float64)
    np.testing.assert_allclose(
        observed / num_samples, expected_probs, atol=0.015
    )
    _, chi2_p_value = stats.chisquare(observed, expected_probs * num_samples)
    self.assertGreater(chi2_p_value, 1e-2)

  def test_identical_draft_and_target_accept_all_draft_tokens(self):
    config = _make_config(ngram_len=4, top_k=20, draft_len=3)
    engine = speculative_engine.DualKeySpeculativeEngine(
        self.target_model, self.target_model, config
    )
    generator = torch.Generator(device=self.device).manual_seed(5)
    prompt_ids = torch.tensor([[10, 11, 12, 13, 14]], device=self.device)
    output = engine.generate(prompt_ids, max_new_tokens=40, generator=generator)
    self.assertEqual(output.acceptance_rate, 1.0)
    self.assertNotIn(
        "target_residual", {origin for origin, _ in output.token_origins}
    )

  def test_numerical_tie_with_zero_residual_accepts_draft_token(self):
    config = _make_config(ngram_len=4, top_k=20, draft_len=3)
    engine = speculative_engine.DualKeySpeculativeEngine(
        self.target_model, self.target_model, config
    )
    prompt_ids = torch.tensor([[10, 11, 12, 13, 14]], device=self.device)
    # Force u_draws > 1.0 so `u_draws[j] <= min(1.0, p_x / q_x)` is False while
    # `residual = max(0, p - q)` is identically zero (`res_sum == 0.0`).
    with absltest.mock.patch.object(
        torch,
        "rand",
        side_effect=lambda n, **kwargs: torch.full(
            (n,), 1.1, device=kwargs.get("device")
        ),
    ):
      output = engine.generate(prompt_ids, max_new_tokens=6)
    self.assertEqual(output.tokens_generated, 6)
    self.assertEqual(output.acceptance_rate, 1.0)
    self.assertNotIn(
        "target_residual", {origin for origin, _ in output.token_origins}
    )

  def test_accepted_draft_eos_stops_generation(self):
    # With identical draft and target models every draft token is accepted, so
    # making the first drafted token an EOS exercises both the drafting loop's
    # EOS break and the accepted-EOS exit (no bonus token, generation ends).
    engine = speculative_engine.DualKeySpeculativeEngine(
        self.target_model,
        self.target_model,
        _make_config(ngram_len=4, top_k=20, draft_len=3),
    )
    prompt_ids = torch.tensor([[10, 11, 12, 13, 14]], device=self.device)
    gen1 = torch.Generator(device=self.device).manual_seed(3)
    out = engine.generate(prompt_ids, max_new_tokens=10, generator=gen1)
    first_origin, first_tok = out.token_origins[0]
    self.assertEqual(first_origin, "draft")

    gen2 = torch.Generator(device=self.device).manual_seed(3)
    out_eos = engine.generate(
        prompt_ids,
        max_new_tokens=10,
        eos_token_id=[self.vocab_size + 1, first_tok],
        generator=gen2,
    )
    self.assertEqual(out_eos.token_origins, [("draft", first_tok)])
    self.assertEqual(out_eos.tokens_generated, 1)
    self.assertEqual(out_eos.num_draft_tokens, 1)
    self.assertEqual(out_eos.num_accepted_tokens, 1)
    self.assertEqual(out_eos.num_steps, 1)
    self.assertEqual(int(out_eos.sequences[0, -1].item()), first_tok)

  def test_half_precision_models_upcast_logits_to_float32(self):
    # Extreme float16 logit gaps (> 18 after temperature scaling) would
    # underflow in float16 exp(log_softmax(...)), producing q_x == 0 and
    # ZeroDivisionError if _top_k_log_probs did not upcast to float32.
    logits_fp16 = torch.tensor([[20.0, 0.0, -5.0]], dtype=torch.float16)
    log_probs, _ = speculative_engine._top_k_log_probs(
        logits_fp16, top_k=3, temperature=0.5
    )
    self.assertEqual(log_probs.dtype, torch.float32)
    self.assertTrue(torch.all(log_probs.exp() > 0))
    self.assertAlmostEqual(float(log_probs.exp().sum()), 1.0, places=6)

    for dtype in (torch.float16, torch.bfloat16):
      target_half = (
          transformers.GPT2LMHeadModel(self.target_model.config)
          .eval()
          .to(device=self.device, dtype=dtype)
      )
      draft_half = (
          transformers.GPT2LMHeadModel(self.draft_model.config)
          .eval()
          .to(device=self.device, dtype=dtype)
      )
      engine = speculative_engine.DualKeySpeculativeEngine(
          target_half,
          draft_half,
          _make_config(ngram_len=4, top_k=10, temperature=0.1, draft_len=3),
      )
      generator = torch.Generator(device=self.device).manual_seed(0)
      prompt_ids = torch.tensor([[10, 11, 12, 13, 14]], device=self.device)
      out = engine.generate(prompt_ids, max_new_tokens=10, generator=generator)
      self.assertEqual(out.tokens_generated, 10)

  def test_sliding_window_cache_cropping(self):
    sizes = dict(
        vocab_size=self.vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=4,
    )
    torch.manual_seed(0)
    target_gemma = (
        transformers.Gemma3ForCausalLM(transformers.Gemma3TextConfig(**sizes))
        .eval()
        .to(self.device)
    )
    torch.manual_seed(1)
    draft_gemma = (
        transformers.Gemma3ForCausalLM(transformers.Gemma3TextConfig(**sizes))
        .eval()
        .to(self.device)
    )
    config = _make_config(ngram_len=4, top_k=10, draft_len=3)
    prompt_ids = torch.tensor([[10, 11, 12, 13, 14]], device=self.device)

    # Different draft/target models exercise rejection cropping past the window.
    engine = speculative_engine.DualKeySpeculativeEngine(
        target_gemma, draft_gemma, config
    )
    gen1 = torch.Generator(device=self.device).manual_seed(0)
    out = engine.generate(prompt_ids, max_new_tokens=12, generator=gen1)
    self.assertEqual(out.tokens_generated, 12)

    # Identical draft/target models exercise crop(0) when all draft tokens are
    # accepted past the sliding window.
    identical_engine = speculative_engine.DualKeySpeculativeEngine(
        target_gemma, target_gemma, config
    )
    gen2 = torch.Generator(device=self.device).manual_seed(0)
    out_id = identical_engine.generate(
        prompt_ids, max_new_tokens=12, generator=gen2
    )
    self.assertEqual(out_id.tokens_generated, 12)
    self.assertEqual(out_id.acceptance_rate, 1.0)

  def test_cached_logits_match_uncached_recomputation(self):
    # Every logit the engine consumes from the draft and target KV caches must
    # equal the logit obtained by recomputing the same sequence without a
    # cache. This covers cropping after rejections, crop(0) after full
    # acceptance, and the sliding-window stash/restore across draft steps.
    sizes = dict(
        vocab_size=self.vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
    )
    torch.manual_seed(0)
    target_gemma = (
        transformers.Gemma3ForCausalLM(transformers.Gemma3TextConfig(**sizes))
        .eval()
        .to(self.device)
    )
    torch.manual_seed(1)
    draft_gemma = (
        transformers.Gemma3ForCausalLM(transformers.Gemma3TextConfig(**sizes))
        .eval()
        .to(self.device)
    )
    self.assertTrue(
        any(
            getattr(layer, "is_sliding", False)
            for layer in transformers.DynamicCache(
                config=target_gemma.config
            ).layers
        )
    )

    model_pairs = {
        "gpt2": (self.target_model, self.draft_model),
        "gemma3": (target_gemma, draft_gemma),
    }
    for name, (target, draft) in model_pairs.items():
      for prompt_len in (2, 5):
        for draft_len in (1, 3, 6):
          with self.subTest(
              name=name, prompt_len=prompt_len, draft_len=draft_len
          ):
            target_checker: Any = _UncachedLogitsChecker(target)
            draft_checker: Any = _UncachedLogitsChecker(draft)
            engine = speculative_engine.DualKeySpeculativeEngine(
                target_checker,
                draft_checker,
                _make_config(ngram_len=4, top_k=10, draft_len=draft_len),
            )
            prompt_ids = torch.arange(
                10, 10 + prompt_len, device=self.device
            ).unsqueeze(0)
            generator = torch.Generator(device=self.device).manual_seed(0)
            out = engine.generate(
                prompt_ids, max_new_tokens=20, generator=generator
            )
            self.assertEqual(out.tokens_generated, 20)
            origins = {origin for origin, _ in out.token_origins}
            self.assertIn("target_residual", origins)
            self.assertGreater(draft_checker.num_calls, 1)
            self.assertGreater(target_checker.num_calls, 1)
            self.assertLess(draft_checker.max_abs_diff, 1e-4)
            self.assertLess(target_checker.max_abs_diff, 1e-4)


if __name__ == "__main__":
  absltest.main()
