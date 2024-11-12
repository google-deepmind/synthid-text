# Copyright 2024 DeepMind Technologies Limited
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

from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
import numpy as np
import torch
import tqdm

from synthid_text import logits_processing
from synthid_text import g_value_expectations
from synthid_text import torch_testing


def does_mean_g_value_matches_theoretical(
    vocab_size: int,
    ngram_len: int,
    batch_size: int,
    keys: Sequence[int],
    atol: float,
    device: torch.device,
    num_leaves: int = 2,
) -> tuple[float, float, bool]:
  """Tests that the mean g-value is close to theoretical value.

  SynthIDLogitsProcessor is tested on its own using random input tokens.

  Args:
    vocab_size: vocab size of the model.
    ngram_len: length of the ngram.
    batch_size: batch size of the model.
    keys: keys used for watermarking.
    atol: absolute tolerance for the mean g-value.
    device: device to use for the test.
    num_leaves: number of children per node in the tournament tree.

  Returns:
    A tuple of mean g-value, the expected mean g-value and the boolean result
    of the test.
  """
  generator = torch.Generator(device=device).manual_seed(0)
  # Use 10**9 rather than vocab_size to ensure variety in (n-1)-grams.
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
      keys=keys,
      sampling_table_size=2**16,
      sampling_table_seed=0,
      context_history_size=context_history_size,
      device=device,
      top_k=vocab_size,
      temperature=0.7,
      num_leaves=num_leaves,
  )

  scores = torch.ones(
      (batch_size, vocab_size),
      dtype=torch.float64,
      device=device,
  )
  # Init state of the logits processor.
  logits_processor.watermarked_call(context, scores)
  # insert context into the state.
  for idx in range(1, ngram_len - 1):
    _ = logits_processor.watermarked_call(context[:, :idx], scores)

  updated_scores, indices_mapping, _ = logits_processor.watermarked_call(
      context, scores
  )

  probs = torch.nn.functional.softmax(updated_scores, dim=1)
  next_tokens = torch.multinomial(
      probs,
      num_samples=1,
      generator=generator,
  )
  # Re-map to dense indices with indices_mapping.
  next_tokens = torch.vmap(torch.take, in_dims=0, out_dims=0)(
      indices_mapping, next_tokens
  )

  ngrams = torch.concat((context, next_tokens), dim=1)
  g_values = logits_processor.compute_g_values(ngrams)
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
    watermarking_config = immutabledict.immutabledict({
        'ngram_len': ngram_len,
        'keys': np.random.randint(low=0, high=2**16, size=(num_layers,)),
        'sampling_table_size': 2**16,
        'sampling_table_seed': 0,
        'context_history_size': 512,
        'device': device,
    })
    batch_size = 100000
    torch.manual_seed(0)
    ngrams = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, ngram_len),
        device=device,
    )

    logits_processor = logits_processing.SynthIDLogitsProcessor(
        **watermarking_config, top_k=10, temperature=1.0
    )
    g_values = logits_processor.compute_g_values(ngrams)
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
    watermarking_config = immutabledict.immutabledict({
        'ngram_len': ngram_len,
        'keys': np.random.randint(low=0, high=2**16, size=(num_layers,)),
        'sampling_table_size': 2**16,
        'sampling_table_seed': 0,
        'context_history_size': 512,
        'device': device,
    })
    n_minus_1_grams = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, watermarking_config['ngram_len'] - 1),
        device=device,
    )

    logits_processor = logits_processing.SynthIDLogitsProcessor(
        **watermarking_config, top_k=10, temperature=1.0
    )
    ngram_keys, _ = logits_processor._compute_keys(
        n_minus_1_grams,
        torch.stack(
            [torch.arange(vocab_size, device=device) for _ in range(batch_size)]
        ),
    )

    g_values = logits_processor.sample_g_values(ngram_keys)
    # g_values shape should be [batch_size, vocab_size, num_layers]
    g_values_mean = torch.mean(torch.mean(g_values.float(), dim=1))
    self.assertAlmostEqual(g_values_mean, 0.5, delta=0.001)

  def test_distributional_convergence(self):
    """Check if watermarked distribution converges to input distribution."""
    vocab_size = 2
    batch_size = 1500
    num_keys = 1000
    device = torch_testing.torch_device()
    temperature = 1.0

    updated_softmaxes = 0
    for _ in tqdm.tqdm(range(num_keys)):
      watermarking_config = immutabledict.immutabledict({
          'ngram_len': 5,
          'keys': np.random.randint(0, 10**9, size=(1,), dtype=np.int64),
          'sampling_table_size': 2**16,
          'sampling_table_seed': 0,
          'context_history_size': 1024,
          'device': device,
      })

      logits_processor = logits_processing.SynthIDLogitsProcessor(
          **watermarking_config,
          top_k=vocab_size,
          temperature=temperature,
          apply_top_k=False,
      )

      ngrams = torch.randint(
          low=0,
          high=vocab_size,
          size=(batch_size, watermarking_config['ngram_len']),
          device=device,
      )

      # Insert ngram-1 into logit_processor state.
      for idx in range(watermarking_config['ngram_len'] - 1):
        _ = logits_processor.watermarked_call(
            ngrams[:, :idx], torch.ones((batch_size, vocab_size), device=device)
        )

      scores = torch.ones((batch_size, vocab_size), device=device)
      updated_scores, _, _ = logits_processor.watermarked_call(ngrams, scores)
      updated_softmaxes += (
          torch.nn.functional.softmax(updated_scores, dim=1).cpu().numpy()
      )
    updated_softmaxes = np.mean(updated_softmaxes, axis=0) / num_keys
    for softmax in updated_softmaxes:
      self.assertAlmostEqual(softmax, 0.5, delta=0.002)

  @parameterized.named_parameters(
      dict(
          testcase_name='vocab_size_100_ngram_len_10_num_layers_1',
          vocab_size=2,
          ngram_len=10,
          num_layers=1,
          atol=0.01,
      ),
      dict(
          testcase_name='vocab_size_100_ngram_len_5_num_layers_1',
          vocab_size=100,
          ngram_len=5,
          num_layers=1,
          atol=0.01,
      ),
      dict(
          testcase_name='vocab_size_100_ngram_len_10_num_layers_2',
          vocab_size=100,
          ngram_len=10,
          num_layers=2,
          atol=0.02,
      ),
      dict(
          testcase_name='vocab_size_2_ngram_len_10_num_layers_1_num_leaves_3',
          vocab_size=2,
          ngram_len=10,
          num_layers=1,
          num_leaves=3,
          atol=0.02,
      ),
      dict(
          testcase_name='vocab_size_100_ngram_len_10_num_layers_1_num_leaves_3',
          vocab_size=100,
          ngram_len=10,
          num_layers=1,
          num_leaves=3,
          atol=0.02,
      ),
  )
  def test_bias_from_logits_processor(
      self, vocab_size, ngram_len, num_layers, atol, num_leaves: int = 2,
  ):
    """Check if watermarked distribution converges to input distribution."""
    device = torch_testing.torch_device()
    mean, expected, passes = does_mean_g_value_matches_theoretical(
        vocab_size=vocab_size,
        ngram_len=ngram_len,
        batch_size=20_000,
        keys=[np.random.randint(0, 10**9) for _ in range(num_layers)],
        atol=atol,
        device=device,
        num_leaves=num_leaves,
    )
    self.assertTrue(passes)


class LogitsProcessorTest(absltest.TestCase):

  def set_up_logits_processor(
      self,
      batch_size,
      sequence_len,
      num_layers,
      ngram_len,
      top_k,
      vocab_size,
  ):
    """Setup function for all the tests."""
    device = torch_testing.torch_device()
    watermarking_config = immutabledict.immutabledict({
        'ngram_len': ngram_len,
        'keys': np.random.randint(low=0, high=2**16, size=(num_layers,)),
        'sampling_table_size': 2**16,
        'sampling_table_seed': 0,
        'context_history_size': 512,
        'device': device,
    })
    logits_processor = logits_processing.SynthIDLogitsProcessor(
        **watermarking_config, top_k=top_k, temperature=1.0
    )
    sequences = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_len),
        device=device,
    )
    return logits_processor, sequences, device

  def test_compute_g_values_shape(self):
    batch_size, sequence_len, num_layers, ngram_len = 1000, 50, 3, 5
    logits_processor, sequences, _ = self.set_up_logits_processor(
        batch_size,
        sequence_len,
        num_layers,
        ngram_len,
        top_k=10,
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
        top_k=10,
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
        top_k=10,
        vocab_size=32,
    )
    eos_mask = logits_processor.compute_eos_token_mask(
        sequences, eos_token_id=0
    )
    self.assertEqual(eos_mask.shape, (batch_size, sequence_len))

  def test_compute_ngram_keys_shape(self):
    batch_size, sequence_len, num_layers, ngram_len = 1000, 50, 3, 5
    vocab_size = 32
    logits_processor, _, device = self.set_up_logits_processor(
        batch_size,
        sequence_len,
        num_layers,
        ngram_len,
        top_k=10,
        vocab_size=vocab_size,
    )

    num_ngrams = 20
    ngrams = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, num_ngrams, ngram_len),
        device=device,
    )
    ngram_keys = logits_processor.compute_ngram_keys(ngrams)
    self.assertEqual(ngram_keys.shape, (batch_size, num_ngrams, num_layers))

  def test_watermarked_call_shape(self):
    batch_size, sequence_len, num_layers, ngram_len = 1000, 50, 3, 5
    vocab_size = 32
    top_k = 10
    logits_processor, sequences, device = self.set_up_logits_processor(
        batch_size,
        sequence_len,
        num_layers,
        ngram_len,
        top_k=top_k,
        vocab_size=vocab_size,
    )
    scores = torch.ones((batch_size, vocab_size), device=device)
    watermarked_scores, top_k_indices, original_scores = (
        logits_processor.watermarked_call(sequences, scores)
    )
    self.assertEqual(watermarked_scores.shape, (batch_size, top_k))
    self.assertEqual(top_k_indices.shape, (batch_size, top_k))
    self.assertEqual(original_scores.shape, (batch_size, top_k))


if __name__ == '__main__':
  absltest.main()
