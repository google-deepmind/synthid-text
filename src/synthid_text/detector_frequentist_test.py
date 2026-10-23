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

"""Tests for the Frequentist detector."""

import fractions
import math

from absl.testing import absltest
import numpy as np
import torch

from synthid_text import detector_frequentist
from synthid_text import logits_processing
from synthid_text import torch_testing

_EOS = 1


def _make_detector(
    depth: int = 30,
    eos_token_id: int | list[int] = _EOS,
) -> tuple[
    detector_frequentist.FrequentistDetector,
    logits_processing.SynthIDLogitsProcessor,
]:
  processor = logits_processing.SynthIDLogitsProcessor(
      ngram_len=5,
      key=987654321,
      depth=depth,
      device=torch_testing.torch_device(),
  )
  return (
      detector_frequentist.FrequentistDetector(processor, eos_token_id),
      processor,
  )


def _generate_watermarked(processor, batch_size, num_steps, seed):
  """Returns watermarked tokens (batch_size, num_steps), prompt removed."""
  device = torch_testing.torch_device()
  vocab_size = 100
  generator = torch.Generator().manual_seed(seed)
  # A single prompt token; it is stripped before detection.
  input_ids = torch.randint(
      low=2, high=vocab_size, size=(batch_size, 1), generator=generator
  ).to(device)
  for _ in range(num_steps):
    scores = torch.zeros((batch_size, vocab_size), device=device)
    scores[:, _EOS] = -torch.inf
    watermarked = processor(input_ids, scores)
    probs = torch.softmax(watermarked, dim=-1).cpu()
    next_tokens = torch.multinomial(probs, 1, generator=generator)
    input_ids = torch.cat([input_ids, next_tokens.to(device)], dim=1)
  return input_ids[:, 1:]


class FrequentistDetectorTest(absltest.TestCase):

  def test_unwatermarked_p_values_have_correct_size(self):
    """Under the null, P(p <= alpha) should be close to (at most) alpha."""
    detector, _ = _make_detector()
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(
        low=2, high=1000, size=(2000, 60), generator=generator
    ).to(torch_testing.torch_device())
    p = detector.p_value(tokens)
    self.assertEqual(p.shape, (2000,))
    self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
    for alpha in (0.05, 0.2):
      rate = float(np.mean(p <= alpha))
      # 2000 samples: the standard deviation of the rate is at most ~0.009.
      self.assertBetween(rate, alpha - 0.03, alpha + 0.03)

  def test_watermarked_text_has_small_p_value(self):
    detector, processor = _make_detector()
    p = detector.p_value(_generate_watermarked(processor, 4, 200, seed=1))
    self.assertTrue(np.all(p < 1e-6), msg=f'p-values: {p}')

  def test_eos_and_trailing_tokens_do_not_change_p_value(self):
    detector, processor = _make_detector(eos_token_id=[_EOS, 106])
    tokens = _generate_watermarked(processor, 2, 60, seed=2)
    generator = torch.Generator().manual_seed(3)
    trailing = torch.randint(0, 100, (2, 20), generator=generator)
    trailing[0, 0] = _EOS
    trailing[1, 0] = 106
    extended = torch.cat([tokens, trailing.to(tokens.device)], dim=1)
    np.testing.assert_array_equal(
        detector.p_value(extended), detector.p_value(tokens)
    )

  def test_p_value_skips_repeated_context_and_matches_binomial_tail(self):
    depth = 30
    detector, processor = _make_detector(depth=depth)
    # With ngram_len=5, position j has context tokens[j:j+4]. Positions 0-4
    # are scored; position 5 repeats the context of position 0; positions 6-8
    # contain or follow EOS.
    tokens = torch.tensor(
        [[10, 11, 12, 13, 14, 10, 11, 12, 13, 15, _EOS, 20, 21]],
        device=torch_testing.torch_device(),
    )
    g_values, mask = processor.compute_g_values_and_mask(tokens, _EOS)
    self.assertEqual(mask.cpu().tolist(), [[True] * 5 + [False] * 4])
    num_ones, num_trials = int(g_values[0, :5].sum()), 5 * depth
    # Exact P(X >= num_ones) for X ~ Binomial(num_trials, 0.5).
    tail = sum(
        math.comb(num_trials, i) for i in range(num_ones, num_trials + 1)
    )
    expected = float(fractions.Fraction(tail, 2**num_trials))
    np.testing.assert_allclose(detector.p_value(tokens), [expected], rtol=1e-9)

  def test_no_scored_positions_gives_p_value_one(self):
    detector, _ = _make_detector()
    device = torch_testing.torch_device()
    all_eos = torch.full((2, 20), _EOS, device=device)
    np.testing.assert_array_equal(detector.p_value(all_eos), [1.0, 1.0])
    too_short = torch.tensor([[5, 6, 7], [8, 9, 10]], device=device)
    np.testing.assert_array_equal(detector.p_value(too_short), [1.0, 1.0])


def _make_dual_key_detector(
    depth: int = 30,
    eos_token_id: int | list[int] = _EOS,
) -> detector_frequentist.DualKeyFrequentistDetector:
  device = torch_testing.torch_device()
  draft_processor = logits_processing.SynthIDLogitsProcessor(
      ngram_len=5, key=42, depth=depth, device=device
  )
  target_processor = logits_processing.SynthIDLogitsProcessor(
      ngram_len=5, key=84, depth=depth, device=device
  )
  return detector_frequentist.DualKeyFrequentistDetector(
      draft_processor, target_processor, eos_token_id
  )


class DualKeyFrequentistDetectorTest(absltest.TestCase):

  def test_validation(self):
    device = torch_testing.torch_device()
    p1 = logits_processing.SynthIDLogitsProcessor(
        ngram_len=5, key=42, depth=10, device=device
    )
    p_same_key = logits_processing.SynthIDLogitsProcessor(
        ngram_len=5, key=42, depth=10, device=device
    )
    p_diff_depth = logits_processing.SynthIDLogitsProcessor(
        ngram_len=5, key=84, depth=20, device=device
    )
    p_diff_ngram = logits_processing.SynthIDLogitsProcessor(
        ngram_len=4, key=84, depth=10, device=device
    )
    with self.assertRaisesRegex(ValueError, 'distinct keys'):
      detector_frequentist.DualKeyFrequentistDetector(p1, p_same_key, _EOS)
    with self.assertRaisesRegex(ValueError, 'same depth'):
      detector_frequentist.DualKeyFrequentistDetector(p1, p_diff_depth, _EOS)
    with self.assertRaisesRegex(ValueError, 'same ngram_len'):
      detector_frequentist.DualKeyFrequentistDetector(p1, p_diff_ngram, _EOS)
    p_diff_history = logits_processing.SynthIDLogitsProcessor(
        ngram_len=5, key=84, depth=10, context_history_size=512, device=device
    )
    with self.assertRaisesRegex(ValueError, 'same context_history_size'):
      detector_frequentist.DualKeyFrequentistDetector(p1, p_diff_history, _EOS)
    # Tests may only have one device available, so set the attribute that the
    # check compares rather than constructing a processor on a second device.
    p_diff_device = logits_processing.SynthIDLogitsProcessor(
        ngram_len=5, key=84, depth=10, device=device
    )
    p_diff_device.device = torch.device('meta')
    with self.assertRaisesRegex(ValueError, 'same device'):
      detector_frequentist.DualKeyFrequentistDetector(p1, p_diff_device, _EOS)

  def test_unwatermarked_p_values_have_correct_size(self):
    """Under the null, P(p <= alpha) should be close to (at most) alpha."""
    detector = _make_dual_key_detector()
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(
        low=2, high=1000, size=(2000, 60), generator=generator
    ).to(torch_testing.torch_device())
    p = detector.p_value(tokens)
    self.assertEqual(p.shape, (2000,))
    self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
    for alpha in (0.05, 0.2):
      rate = float(np.mean(p <= alpha))
      self.assertBetween(rate, alpha - 0.03, alpha + 0.03)

  def test_eos_and_repeated_context_masking_matches_binomial_tail(self):
    depth = 20
    detector = _make_dual_key_detector(depth=depth)
    device = torch_testing.torch_device()
    # With ngram_len=5, positions 0-4 are scored; position 5 repeats the
    # context of position 0; positions 6-8 contain or follow EOS.
    tokens = torch.tensor(
        [[10, 11, 12, 13, 14, 10, 11, 12, 13, 15, _EOS, 20, 21]],
        device=device,
    )
    draft_g, mask = detector.draft_processor.compute_g_values_and_mask(
        tokens, _EOS
    )
    target_g, _ = detector.target_processor.compute_g_values_and_mask(
        tokens, _EOS
    )
    self.assertEqual(mask.cpu().tolist(), [[True] * 5 + [False] * 4])
    num_ones = int(draft_g[0, :5].sum() + target_g[0, :5].sum())
    num_trials = 2 * 5 * depth
    tail = sum(
        math.comb(num_trials, i) for i in range(num_ones, num_trials + 1)
    )
    expected = float(fractions.Fraction(tail, 2**num_trials))
    np.testing.assert_allclose(detector.p_value(tokens), [expected], rtol=1e-9)

    # Trailing tokens after EOS (including a sequence of EOS IDs) do not affect
    # p_value.
    np.testing.assert_array_equal(
        detector.p_value(tokens), detector.p_value(tokens[:, :11])
    )
    multi_eos_detector = _make_dual_key_detector(
        depth=depth, eos_token_id=[_EOS, 106]
    )
    tokens_alt_eos = tokens.clone()
    tokens_alt_eos[0, 10] = 106
    np.testing.assert_array_equal(
        multi_eos_detector.p_value(tokens_alt_eos),
        detector.p_value(tokens[:, :10]),
    )

    # Sequences with no scored positions return p_value == 1.0.
    all_eos = torch.full((2, 20), _EOS, device=device)
    np.testing.assert_array_equal(detector.p_value(all_eos), [1.0, 1.0])
    too_short = torch.tensor([[5, 6, 7], [8, 9, 10]], device=device)
    np.testing.assert_array_equal(detector.p_value(too_short), [1.0, 1.0])


if __name__ == '__main__':
  absltest.main()
