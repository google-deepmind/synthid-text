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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import tqdm

from synthid_text import g_value_expectations
from synthid_text import logits_processing
from synthid_text import torch_testing


def does_mean_g_value_matches_theoretical(
    vocab_size: int,
    ngram_len: int,
    batch_size: int,
    key: int,
    atol: float,
    device: torch.device,
    depth: int = 30,
    num_leaves: int = 2,
) -> tuple[torch.Tensor, float, torch.Tensor]:
  """Returns the mean g-value, its theoretical value and whether they match.

  SynthIDLogitsProcessor is tested on its own using random input tokens.

  Args:
    vocab_size: Vocab size of the model.
    ngram_len: Length of the ngram.
    batch_size: Batch size of the model.
    key: Key used for watermarking.
    atol: Absolute tolerance for the mean g-value.
    device: Device to use for the test.
    depth: Number of tournament layers.
    num_leaves: Number of leaves per node in the tournament tree.

  Returns:
    A tuple of the mean g-value of each layer (depth,), the expected mean
    g-value and whether every layer's mean is within atol of the expected one.
  """
  generator = torch.Generator(device=device).manual_seed(0)
  # Use 10**9 rather than vocab_size to ensure variety in the contexts.
  context = torch.randint(
      low=0,
      high=10**9,
      size=(batch_size, ngram_len - 1),
      dtype=torch.int64,
      generator=generator,
      device=device,
  )

  context_history_size = 1024
  logits_processor = logits_processing.SynthIDLogitsProcessor(
      ngram_len=ngram_len,
      key=key,
      depth=depth,
      context_history_size=context_history_size,
      device=device,
      num_leaves=num_leaves,
  )

  scores = torch.ones(
      (batch_size, vocab_size),
      dtype=torch.float64,
      device=device,
  )
  # The processor appends the last input token to its context from the second
  # call onwards, so after ngram_len calls its context is `context`. A dummy
  # first token keeps the input growing by one token per call.
  input_ids = torch.concat((torch.zeros_like(context[:, :1]), context), dim=1)
  for input_len in range(1, ngram_len):
    _ = logits_processor(input_ids[:, :input_len], scores)
  updated_scores = logits_processor(input_ids, scores)

  probs = torch.nn.functional.softmax(updated_scores, dim=1)
  generator = torch.Generator(device=device).manual_seed(0)
  next_tokens = torch.multinomial(
      probs,
      num_samples=1,
      generator=generator,
  )

  tokens = torch.concat((context, next_tokens), dim=1)
  g_values = logits_processor.compute_g_values(tokens)
  mean_g_values = g_values.mean(dtype=torch.float64, dim=(0, 1))

  expected_mean_g_value = g_value_expectations.expected_mean_g_value(
      vocab_size=vocab_size, num_leaves=num_leaves
  )
  is_close = torch.all(
      torch.isclose(
          mean_g_values,
          torch.tensor(
              expected_mean_g_value, dtype=torch.float64, device=device
          ),
          atol=atol,
          rtol=0,
      )
  )

  return mean_g_values, expected_mean_g_value, is_close


class LogitsProcessorCorrectnessTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='vocab_size_10k_ngram_len_5_num_layers_3',
          ngram_len=5,
          vocab_size=10000,
          num_layers=3,
      ),
      dict(
          testcase_name='vocab_size_1k_ngram_len_10_num_layers_5',
          ngram_len=10,
          vocab_size=1000,
          num_layers=5,
      ),
  )
  def test_g_value_uniformity_for_random_ngrams(
      self, vocab_size, ngram_len, num_layers
  ):
    device = torch_testing.torch_device()
    batch_size = 100000
    torch.manual_seed(0)
    ngrams = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, ngram_len),
        device=device,
    )

    logits_processor = logits_processing.SynthIDLogitsProcessor(
        ngram_len=ngram_len,
        key=913391291,
        depth=num_layers,
        context_history_size=512,
        device=device,
    )
    # One ngram per batch row: (batch_size, num_ngrams=1, ngram_len).
    g_values = logits_processor.compute_g_values_from_ngrams(ngrams[:, None, :])
    g_values_mean = torch.mean(torch.mean(g_values.float(), dim=0))
    self.assertAlmostEqual(g_values_mean, 0.5, delta=0.01)

  @parameterized.named_parameters(
      dict(
          testcase_name='vocab_size_10k',
          vocab_size=10000,
          num_layers=3,
      ),
      dict(
          testcase_name='vocab_size_1k',
          vocab_size=1000,
          num_layers=20,
      ),
  )
  def test_g_values_uniformity_across_vocab_size(self, vocab_size, num_layers):
    batch_size = 1000
    ngram_len = 5
    device = torch_testing.torch_device()
    generator = torch.Generator(device=device).manual_seed(0)
    n_minus_1_grams = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, ngram_len - 1),
        device=device,
        generator=generator,
    )

    logits_processor = logits_processing.SynthIDLogitsProcessor(
        ngram_len=ngram_len,
        key=913391291,
        depth=num_layers,
        context_history_size=512,
        device=device,
    )
    vocab = (
        torch.arange(vocab_size, device=device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch_size, -1, 1)
    )
    ctx = n_minus_1_grams.unsqueeze(1).expand(-1, vocab_size, -1)
    ngrams = torch.cat([ctx, vocab], dim=-1)
    g_values = logits_processor.compute_g_values_from_ngrams(ngrams)
    # g_values shape should be (batch_size, vocab_size, depth).
    g_values_mean = torch.mean(torch.mean(g_values.float(), dim=1))
    self.assertAlmostEqual(g_values_mean, 0.5, delta=0.001)

  def test_distributional_convergence(self):
    """Checks that the watermarked distribution converges to the input one."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    vocab_size = 2
    batch_size = 1500
    ngram_len = 5
    num_keys = 1000
    device = torch_testing.torch_device()
    updated_softmaxes = 0
    for _ in tqdm.tqdm(range(num_keys)):
      logits_processor = logits_processing.SynthIDLogitsProcessor(
          ngram_len=ngram_len,
          key=int(rng.integers(0, 2**63)),
          depth=1,
          context_history_size=1024,
          device=device,
      )

      ngrams = torch.randint(
          low=0,
          high=vocab_size,
          size=(batch_size, ngram_len),
          device=device,
      )

      # Insert the first ngram_len - 2 tokens into the logits processor state.
      # The processor ignores the last input token at the first call.
      for idx in range(ngram_len - 1):
        _ = logits_processor(
            ngrams[:, :idx], torch.ones((batch_size, vocab_size), device=device)
        )

      # This call appends token ngram_len - 2, so the context is
      # ngrams[:, : ngram_len - 1].
      scores = torch.ones((batch_size, vocab_size), device=device)
      updated_scores = logits_processor(ngrams[:, : ngram_len - 1], scores)
      updated_softmaxes += (
          torch.nn.functional.softmax(updated_scores, dim=1).cpu().numpy()
      )
    updated_softmaxes = np.mean(updated_softmaxes, axis=0) / num_keys
    for softmax in updated_softmaxes:
      self.assertAlmostEqual(softmax, 0.5, delta=0.005)

  @parameterized.named_parameters(
      dict(
          testcase_name='vocab_size_2_ngram_len_10',
          vocab_size=2,
          ngram_len=10,
          atol=0.01,
      ),
      dict(
          testcase_name='vocab_size_100_ngram_len_5',
          vocab_size=100,
          ngram_len=5,
          atol=0.01,
      ),
      dict(
          testcase_name='vocab_size_100_ngram_len_10',
          vocab_size=100,
          ngram_len=10,
          atol=0.02,
      ),
      dict(
          testcase_name='vocab_size_2_ngram_len_10_num_leaves_3',
          vocab_size=2,
          ngram_len=10,
          num_leaves=3,
          atol=0.02,
      ),
      dict(
          testcase_name='vocab_size_100_ngram_len_10_num_leaves_3',
          vocab_size=100,
          ngram_len=10,
          num_leaves=3,
          atol=0.02,
      ),
  )
  def test_bias_from_logits_processor(
      self,
      vocab_size,
      ngram_len,
      atol,
      num_leaves: int = 2,
  ):
    """Checks that the mean g-value matches the single layer theoretical value.

    The mean g-value is computed on watermarked text. The theoretical value is
    for a single tournament layer, so depth is fixed to 1.
    """
    device = torch_testing.torch_device()
    mean, expected, passes = does_mean_g_value_matches_theoretical(
        vocab_size=vocab_size,
        ngram_len=ngram_len,
        batch_size=50_000,
        key=1,
        depth=1,
        atol=atol,
        device=device,
        num_leaves=num_leaves,
    )
    print('Mean', mean)
    print('Expected', expected)
    self.assertTrue(passes)


class LogitsProcessorTest(absltest.TestCase):

  def set_up_logits_processor(
      self,
      batch_size,
      sequence_len,
      num_layers,
      ngram_len,
      vocab_size,
  ):
    """Sets up a logits processor and inputs for the tests."""
    device = torch_testing.torch_device()
    logits_processor = logits_processing.SynthIDLogitsProcessor(
        ngram_len=ngram_len,
        key=913391291,
        depth=num_layers,
        context_history_size=512,
        device=device,
    )
    sequences = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_len),
        device=device,
        generator=torch.Generator(device=device).manual_seed(0),
    )
    return logits_processor, sequences, device

  def test_compute_g_values_shape(self):
    batch_size, sequence_len, num_layers, ngram_len = 1000, 50, 3, 5
    logits_processor, sequences, _ = self.set_up_logits_processor(
        batch_size,
        sequence_len,
        num_layers,
        ngram_len,
        vocab_size=32,
    )
    g_values = logits_processor.compute_g_values(sequences)
    self.assertEqual(
        g_values.shape, (batch_size, sequence_len - (ngram_len - 1), num_layers)
    )

  def test_compute_context_repetition_mask_shape(self):
    batch_size, sequence_len, num_layers, ngram_len = 1000, 50, 3, 5
    logits_processor, sequences, _ = self.set_up_logits_processor(
        batch_size,
        sequence_len,
        num_layers,
        ngram_len,
        vocab_size=32,
    )
    context_mask = logits_processor.compute_context_repetition_mask(sequences)
    self.assertEqual(
        context_mask.shape, (batch_size, sequence_len - (ngram_len - 1))
    )

  def test_compute_eos_token_mask_shape(self):
    batch_size, sequence_len, num_layers, ngram_len = 1000, 50, 3, 5
    logits_processor, sequences, _ = self.set_up_logits_processor(
        batch_size,
        sequence_len,
        num_layers,
        ngram_len,
        vocab_size=32,
    )
    eos_mask = logits_processor.compute_eos_token_mask(
        sequences, eos_token_id=0
    )
    self.assertEqual(eos_mask.shape, (batch_size, sequence_len))

  def test_compute_eos_token_mask_values(self):
    device = torch_testing.torch_device()
    logits_processor = logits_processing.SynthIDLogitsProcessor(
        ngram_len=3, key=1234, depth=2, device=device
    )
    eos = 1
    input_ids = torch.tensor(
        [
            [5, 7, 8, 9, 6],  # No EOS.
            [1, 7, 8, 9, 6],  # EOS first.
            [5, 7, 1, 9, 6],  # EOS in the middle.
            [5, 1, 8, 1, 1],  # Several EOS tokens.
            [5, 7, 8, 9, 1],  # EOS last.
        ],
        device=device,
    )
    expected = torch.tensor(
        [
            [True, True, True, True, True],
            [False, False, False, False, False],
            [True, True, False, False, False],
            [True, False, False, False, False],
            [True, True, True, True, False],
        ],
        device=device,
    )
    eos_mask = logits_processor.compute_eos_token_mask(
        input_ids, eos_token_id=eos
    )
    self.assertEqual(eos_mask.dtype, torch.bool)
    torch.testing.assert_close(eos_mask, expected)

    multi_eos_mask = logits_processor.compute_eos_token_mask(
        input_ids, eos_token_id=[1, 8]
    )
    expected_multi = torch.tensor(
        [
            [True, True, False, False, False],
            [False, False, False, False, False],
            [True, True, False, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False],
        ],
        device=device,
    )
    torch.testing.assert_close(multi_eos_mask, expected_multi)

  def test_compute_g_values_from_ngrams_shape(self):
    batch_size, sequence_len, num_layers, ngram_len = 1000, 50, 3, 5
    vocab_size = 32
    logits_processor, _, device = self.set_up_logits_processor(
        batch_size,
        sequence_len,
        num_layers,
        ngram_len,
        vocab_size=vocab_size,
    )

    num_ngrams = 20
    ngrams = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, num_ngrams, ngram_len),
        device=device,
    )
    g_values = logits_processor.compute_g_values_from_ngrams(ngrams)
    self.assertEqual(g_values.shape, (batch_size, num_ngrams, num_layers))

  def test_unsupported_keys_arg_raises_type_error(self):
    device = torch_testing.torch_device()
    legacy_kwargs = {'keys': [654, 400]}
    with self.assertRaises(TypeError):
      logits_processing.SynthIDLogitsProcessor(
          ngram_len=5,
          device=device,
          **legacy_kwargs,  # pyrefly: ignore[unexpected-keyword]
      )

  def test_repetition_mask_respects_history_size(self):
    device = torch_testing.torch_device()
    lp = logits_processing.SynthIDLogitsProcessor(
        ngram_len=2, key=3, depth=1, context_history_size=2, device=device
    )
    # Contexts (1 token each) for the 7 scored positions: 10 11 10 12 13 14 10.
    # The second 10 is 2 contexts after the first (within the history of 2);
    # the third 10 is 4 contexts after the second (beyond it).
    tokens = torch.tensor([[10, 11, 10, 12, 13, 14, 10, 15]], device=device)
    mask = lp.compute_context_repetition_mask(tokens)
    self.assertEqual(mask.cpu().tolist(), [[1, 1, 0, 1, 1, 1, 1]])

  def test_compute_g_values_and_mask(self):
    device = torch_testing.torch_device()
    eos = 1
    lp = logits_processing.SynthIDLogitsProcessor(
        ngram_len=3, key=7, depth=4, device=device
    )
    # Contexts (ngram_len - 1 = 2 tokens) for the 6 scored positions:
    # (5,6) (6,7) (7,5) (5,6)=repeat (6,8) (8,EOS). The ngrams ending at the
    # EOS token and after it are masked.
    tokens = torch.tensor([[5, 6, 7, 5, 6, 8, eos, 9]], device=device)
    g_values, mask = lp.compute_g_values_and_mask(tokens, eos_token_id=eos)
    self.assertEqual(tuple(g_values.shape), (1, 6, 4))
    self.assertEqual(mask.cpu().tolist(), [[1, 1, 1, 0, 0, 0]])
    torch.testing.assert_close(g_values, lp.compute_g_values(tokens))
    _, mask_multi = lp.compute_g_values_and_mask(tokens, eos_token_id=[eos, 8])
    self.assertEqual(mask_multi.cpu().tolist(), [[1, 1, 1, 0, 0, 0]])

  def test_compute_g_values_and_mask_short_sequence(self):
    device = torch_testing.torch_device()
    lp = logits_processing.SynthIDLogitsProcessor(
        ngram_len=3, key=7, depth=4, device=device
    )
    for seq_len, expected_ngrams in [(0, 0), (2, 0), (3, 1)]:
      tokens = torch.zeros((2, seq_len), dtype=torch.int64, device=device)
      g_values, mask = lp.compute_g_values_and_mask(tokens, eos_token_id=1)
      self.assertEqual(tuple(g_values.shape), (2, expected_ngrams, 4))
      self.assertEqual(tuple(mask.shape), (2, expected_ngrams))

  def test_call_on_truncated_scores_preserves_sparsity(self):
    """__call__ on pre-truncated scores preserves sparsity across steps."""
    device = torch_testing.torch_device()
    batch_size, vocab_size, top_k = 8, 40, 6
    num_steps = 12
    lp = logits_processing.SynthIDLogitsProcessor(
        ngram_len=5, key=12345, depth=4, device=device
    )
    generator = torch.Generator(device=device).manual_seed(0)
    input_ids = torch.randint(
        0, vocab_size, (batch_size, 3), device=device, generator=generator
    )
    for _ in range(num_steps):
      raw = torch.randn(
          (batch_size, vocab_size), device=device, generator=generator
      )
      kth = torch.topk(raw, top_k, dim=1).values[:, -1:]
      processed = torch.where(raw < kth, -float('inf'), raw)
      scores = lp(input_ids, processed)

      # Only the top-k entries are finite.
      self.assertTrue((torch.isfinite(scores).sum(dim=1) == top_k).all())
      # Advance the sequence so that the context changes.
      next_tokens = torch.multinomial(
          torch.softmax(scores, dim=1), num_samples=1, generator=generator
      )
      input_ids = torch.cat([input_ids, next_tokens], dim=1)


def _get_state(
    lp: logits_processing.SynthIDLogitsProcessor,
) -> logits_processing.SynthIDState:
  """Returns the state of the processor, which must have been called."""
  if lp.state is None:
    raise AssertionError('The logits processor has not been called.')
  return lp.state


class LogitsProcessorCallTest(parameterized.TestCase):

  @parameterized.parameters(2, 3)
  def test_distortionary_matches_float64_closed_form(self, num_leaves):
    device = torch_testing.torch_device()
    generator = torch.Generator(device=device).manual_seed(0)
    scores = torch.randn((4, 8), device=device, generator=generator)
    g_values = torch.randint(
        0, 2, (4, 8, 5), device=device, generator=generator
    ).float()
    g_values[1] = 1.0  # g-mass 1 at every depth.
    # Row 2: g-mass ~3e-10, where the closed form cancels to 0 in float32.
    scores[2], scores[2, 0] = 0.0, -20.0
    g_values[2], g_values[2, 0] = 0.0, 1.0
    probs = torch.softmax(scores.double(), dim=1)
    for i in range(g_values.shape[2]):
      g = g_values[:, :, i].double()
      g_mass = (g * probs).sum(dim=1, keepdim=True)
      probs = probs * torch.where(
          g == 1,
          (1 - (1 - g_mass) ** num_leaves) / g_mass,
          (1 - g_mass) ** (num_leaves - 1),
      )
    torch.testing.assert_close(
        logits_processing.update_scores_distortionary(
            scores, g_values, num_leaves
        ),
        torch.log(probs).float(),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_skip_first_ngram_calls(self):
    """The first ngram_len - 1 calls are unchanged and not in the history."""
    device = torch_testing.torch_device()
    ngram_len = 4
    generator = torch.Generator(device=device).manual_seed(0)
    tokens = torch.randint(
        0, 16, (3, ngram_len), device=device, generator=generator
    )
    lp = logits_processing.SynthIDLogitsProcessor(
        ngram_len=ngram_len, key=11, depth=4, device=device
    )
    # The first token is never appended to the context, which starts as zeros.
    padded = torch.concat(
        (torch.zeros_like(tokens[:, 1:]), tokens[:, 1:]), dim=1
    )
    for step in range(ngram_len):
      scores = torch.log_softmax(
          torch.randn((3, 16), device=device, generator=generator), dim=1
      )
      out = lp(tokens[:, : step + 1], scores)
      state = _get_state(lp)
      # The context advances at every call, including the skipped ones.
      torch.testing.assert_close(
          state.context, padded[:, step : step + ngram_len - 1]
      )
      is_last = step == ngram_len - 1
      self.assertEqual(torch.equal(out, scores), not is_last)
      self.assertEqual(bool(state.context_history[:, 0].ne(0).all()), is_last)
      self.assertFalse(state.context_history[:, 1:].any())

  def test_repeated_context_returns_input_scores(self):
    device = torch_testing.torch_device()
    lp = logits_processing.SynthIDLogitsProcessor(
        ngram_len=2,
        key=21,
        depth=4,
        device=device,
        skip_first_ngram_calls=False,
    )
    # Contexts (1 token) at the 4 calls: 0, 5, 6, then 5 (repeated) in row 0
    # and 7 (fresh) in row 1. The first token is not used as context.
    tokens = torch.tensor([[9, 5, 6, 5], [9, 5, 6, 7]], device=device)
    generator = torch.Generator(device=device).manual_seed(0)
    for step in range(4):
      scores = torch.log_softmax(
          torch.randn((2, 10), device=device, generator=generator), dim=1
      )
      out = lp(tokens[:, : step + 1], scores)
      is_unchanged = [torch.equal(out[row], scores[row]) for row in range(2)]
      self.assertEqual(is_unchanged, [step == 3, False])

  @parameterized.parameters(2, 3)
  def test_call_with_neg_inf_rows_matches_single_rows(self, num_leaves):
    """Rows with different numbers of -inf entries are processed separately."""
    device = torch_testing.torch_device()
    vocab_size = 12
    # Number of finite scores in each row; row 3 is entirely -inf.
    num_finite = torch.tensor([12, 5, 1, 0, 3], device=device)
    batch_size = num_finite.shape[0]

    def make_processor():
      return logits_processing.SynthIDLogitsProcessor(
          ngram_len=3,
          key=77,
          depth=4,
          device=device,
          num_leaves=num_leaves,
          skip_first_ngram_calls=False,
      )

    batch_lp = make_processor()
    # The all -inf row has nothing to watermark; it gets no per-row processor.
    row_lps = {r: make_processor() for r in range(batch_size) if num_finite[r]}
    generator = torch.Generator(device=device).manual_seed(0)
    input_ids = torch.randint(
        0, vocab_size, (batch_size, 2), device=device, generator=generator
    )
    num_changed = 0
    for _ in range(6):
      shape = (batch_size, vocab_size)
      raw = torch.randn(shape, device=device, generator=generator)
      # Keep num_finite[row] randomly chosen entries of each row finite.
      ranks = (
          torch.rand(shape, device=device, generator=generator)
          .argsort(dim=1)
          .argsort(dim=1)
      )
      scores = torch.where(ranks < num_finite[:, None], raw, -float('inf'))
      out = batch_lp(input_ids, scores)
      self.assertFalse(torch.isnan(out).any())
      self.assertTrue(torch.equal(torch.isneginf(out), torch.isneginf(scores)))
      self.assertTrue(torch.equal(torch.isfinite(out).sum(dim=1), num_finite))
      for row, row_lp in row_lps.items():
        row_out = row_lp(input_ids[row : row + 1], scores[row : row + 1])
        torch.testing.assert_close(out[row : row + 1], row_out)
      num_changed += not torch.allclose(out[0], scores[0].log_softmax(dim=0))
      next_tokens = torch.randint(
          0, vocab_size, (batch_size, 1), device=device, generator=generator
      )
      input_ids = torch.cat([input_ids, next_tokens], dim=1)
    self.assertGreater(num_changed, 0)


if __name__ == '__main__':
  absltest.main()
