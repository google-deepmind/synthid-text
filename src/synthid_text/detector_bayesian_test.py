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

"""Tests for the Bayesian detector."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torch

from synthid_text import detector_bayesian
from synthid_text import logits_processing

_EOS = 1
_DEPTH = 4
_CPU = torch.device('cpu')


def _make_processor(key: int) -> logits_processing.SynthIDLogitsProcessor:
  return logits_processing.SynthIDLogitsProcessor(
      ngram_len=5, key=key, depth=_DEPTH, device=_CPU
  )


class BayesianDetectorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.processor = _make_processor(key=42)
    generator = torch.Generator().manual_seed(0)
    self.tokens = torch.randint(
        low=2, high=50, size=(3, 30), generator=generator
    )
    self.tokens[0, 20] = _EOS
    g_values, mask = self.processor.compute_g_values_and_mask(
        self.tokens, eos_token_id=_EOS
    )
    self.g_values = g_values.numpy()
    self.mask = mask.numpy()
    # Randomly initialised (untrained) parameters are enough to check that
    # score() wires tokens, g-values and masks together correctly.
    self.params = detector_bayesian.BayesianDetectorModule(
        watermarking_depth=_DEPTH
    ).init(jax.random.PRNGKey(0), self.g_values[:1], self.mask[:1])
    self.detector = detector_bayesian.BayesianDetector(
        self.processor, _EOS, self.params
    )

  def test_score_from_tokens_matches_score_from_g_values(self):
    scores = np.asarray(self.detector.score(self.tokens))
    log_odds = jnp.asarray(
        detector_bayesian.BayesianDetectorModule(
            watermarking_depth=_DEPTH
        ).apply(self.params, self.g_values, self.mask)
    )
    expected = jax.nn.sigmoid(log_odds)
    self.assertEqual(scores.shape, (3,))
    self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))
    np.testing.assert_allclose(scores, np.asarray(expected), rtol=1e-6)

  def test_raises_if_num_leaves_is_not_2(self):
    processor = logits_processing.SynthIDLogitsProcessor(
        ngram_len=5, key=42, depth=_DEPTH, device=_CPU, num_leaves=3
    )
    with self.assertRaisesRegex(ValueError, 'num_leaves=3'):
      detector_bayesian.BayesianDetector(processor, _EOS, self.params)
    with self.assertRaisesRegex(ValueError, 'num_leaves=3'):
      detector_bayesian.BayesianDetector.train_best_detector_given_g_values(
          train_g_values=self.g_values,
          train_masks=self.mask,
          train_labels=jnp.array([1, 0, 1]),
          cv_g_values=self.g_values,
          cv_masks=self.mask,
          cv_labels=jnp.array([1, 0, 1]),
          logits_processor=processor,
          eos_token_id=_EOS,
      )

  def test_init_uses_rng(self):
    module = detector_bayesian.BayesianDetectorModule(watermarking_depth=_DEPTH)
    other = module.init(jax.random.PRNGKey(1), self.g_values[:1], self.mask[:1])
    for name in ('beta', 'delta'):
      self.assertFalse(
          np.array_equal(
              self.params['params']['likelihood_model_watermarked'][name],
              other['params']['likelihood_model_watermarked'][name],
          )
      )


class _DummyDetector:

  def apply(self, params, g_values, masks):
    del params, masks
    return g_values


class TprAtFprTest(absltest.TestCase):

  def test_ties_do_not_exceed_target_fpr(self):
    negatives = [0.0] * 95 + [5.0] * 5
    positives = [5.0, 5.0, 6.0]
    scores = np.array(positives + negatives)
    labels = np.array([1] * len(positives) + [0] * len(negatives))
    dummy = _DummyDetector()

    # Counting scores >= 5 as positive would give FPR 5% > 1%, so positives
    # tied with the top negatives must not be counted.
    self.assertAlmostEqual(
        detector_bayesian.tpr_at_fpr(
            params={},
            detector_inputs=(scores, scores),
            w_true=labels,
            minibatch_size=32,
            detector_module=dummy,
            target_fpr=0.01,
        ),
        1 / 3,
    )
    # With target 5% all five tied negatives may be false positives.
    self.assertAlmostEqual(
        detector_bayesian.tpr_at_fpr(
            params={},
            detector_inputs=(scores, scores),
            w_true=labels,
            minibatch_size=32,
            detector_module=dummy,
            target_fpr=0.05,
        ),
        1.0,
    )


class TrainingPreprocessingTest(absltest.TestCase):
  """Training and inference preprocessing must give the same log-odds."""

  def _check_matches_inference(
      self, truncation_length, num_kept, eos_token_id=_EOS
  ):
    processor = _make_processor(key=7)
    generator = torch.Generator().manual_seed(3)
    tokens = torch.randint(low=2, high=30, size=(6, 25), generator=generator)
    tokens[0, 12:] = torch.arange(12, 25) + 100  # EOS, then other tokens.
    tokens[0, 12] = _EOS
    tokens[1, 3] = (
        eos_token_id[-1] if isinstance(eos_token_id, list) else _EOS
    )  # Fewer than ngram_len tokens before EOS.
    tokens[2, 10:15] = tokens[2, 0:5]  # A repeated context.
    masks, g_values = detector_bayesian.process_outputs_for_training(
        [tokens],
        logits_processor=processor,
        eos_token_id=eos_token_id,
        truncation_length=truncation_length,
        max_length=40,
        torch_device=_CPU,
    )
    if truncation_length is not None:
      num_before_eos = processor.compute_eos_token_mask(
          tokens, eos_token_id
      ).sum(1)
      tokens = tokens[num_before_eos >= truncation_length, :truncation_length]
    inference_g_values, inference_mask = processor.compute_g_values_and_mask(
        tokens, eos_token_id=eos_token_id
    )
    module = detector_bayesian.BayesianDetectorModule(watermarking_depth=_DEPTH)
    params = module.init(
        jax.random.PRNGKey(0),
        inference_g_values[:1].numpy(),
        inference_mask[:1].numpy(),
    )
    # Move the parameters away from their near-symmetric initial values.
    rng = np.random.default_rng(0)
    params = jax.tree_util.tree_map(
        lambda x: x + 0.5 * rng.normal(size=np.shape(x)).astype(np.float32),
        params,
    )

    def log_odds(g, m):
      return np.asarray(module.apply(params, g.numpy(), m.numpy()))

    train_log_odds = log_odds(g_values[0], masks[0])
    self.assertEqual(train_log_odds.shape, (num_kept,))
    np.testing.assert_array_equal(masks[0].sum(1), inference_mask.sum(1))
    np.testing.assert_allclose(
        train_log_odds,
        log_odds(inference_g_values, inference_mask),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_matches_inference_without_truncation(self):
    self._check_matches_inference(truncation_length=None, num_kept=6)
    self._check_matches_inference(
        truncation_length=None, num_kept=6, eos_token_id=[_EOS, 200]
    )

  def test_matches_inference_with_truncation(self):
    # Rows 0 and 1 have fewer than 15 tokens before EOS and are dropped.
    self._check_matches_inference(truncation_length=15, num_kept=4)
    self._check_matches_inference(
        truncation_length=15, num_kept=4, eos_token_id=[_EOS, 200]
    )


class TrainBestDetectorTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.processor = _make_processor(key=1234)
    generator = torch.Generator().manual_seed(0)
    num_batches, batch_size, length, vocab_size = 6, 8, 80, 100
    num_train, num_held_out = num_batches * batch_size, 16
    cls.truncation_length, cls.max_padded_length = 60, 90

    # Watermarked sequences: sample from a uniform distribution after
    # watermarking. The first (prompt) token is dropped.
    input_ids = torch.randint(
        2, vocab_size, (num_train + num_held_out, 1), generator=generator
    )
    for _ in range(length):
      scores = torch.zeros((input_ids.shape[0], vocab_size))
      scores[:, _EOS] = -torch.inf
      probs = torch.softmax(cls.processor(input_ids, scores), dim=-1)
      next_tokens = torch.multinomial(probs, 1, generator=generator)
      input_ids = torch.cat([input_ids, next_tokens], dim=1)
    wm_tokens = input_ids[:, 1:]
    uwm_tokens = torch.randint(
        2, vocab_size, (num_train + num_held_out, length), generator=generator
    )
    # Held-out rows are used neither for training nor for model selection.
    cls.wm_held_out = wm_tokens[num_train:]
    cls.uwm_held_out = uwm_tokens[num_train:]

    def training_batches(tokens):
      batches = [b.clone() for b in tokens[:num_train].split(batch_size)]
      for batch in batches:
        # Row 0 is kept (70 tokens before EOS); row 1 is dropped (30).
        for row, eos_pos in ((0, 70), (1, 30)):
          batch[row, eos_pos] = _EOS
          batch[row, eos_pos + 1 :] = torch.randint(
              0, vocab_size, (length - eos_pos - 1,), generator=generator
          )
      # A batch shorter than the truncation length is dropped entirely.
      batches.append(
          torch.randint(2, vocab_size, (batch_size, 50), generator=generator)
      )
      return [b.numpy() for b in batches]

    cls.wm_batches = training_batches(wm_tokens)
    cls.uwm_batches = training_batches(uwm_tokens)

  def _process(self, wm_batches, seed=0):
    return detector_bayesian.BayesianDetector.process_raw_model_outputs(
        tokenized_wm_outputs=wm_batches,
        tokenized_uwm_outputs=self.uwm_batches,
        logits_processor=self.processor,
        eos_token_id=_EOS,
        torch_device=_CPU,
        pos_truncation_length=self.truncation_length,
        neg_truncation_length=self.truncation_length,
        max_padded_length=self.max_padded_length,
        seed=seed,
    )

  def test_same_seed_is_reproducible(self):
    first = self._process(self.wm_batches, seed=3)
    second = self._process(self.wm_batches, seed=3)
    for a, b in zip(first, second):
      np.testing.assert_array_equal(a, b)

  def test_trains_on_cpu_and_separates_held_out_text(self):
    detector, selection_loss = (
        detector_bayesian.BayesianDetector.train_best_detector(
            tokenized_wm_outputs=self.wm_batches,
            tokenized_uwm_outputs=self.uwm_batches,
            logits_processor=self.processor,
            eos_token_id=_EOS,
            torch_device=_CPU,
            pos_truncation_length=self.truncation_length,
            neg_truncation_length=self.truncation_length,
            max_padded_length=self.max_padded_length,
            n_epochs=10,
            l2_weights=np.zeros((1,)),
        )
    )
    self.assertBetween(selection_loss, -1.0, 0.0)
    wm_scores = np.asarray(detector.score(self.wm_held_out))
    uwm_scores = np.asarray(detector.score(self.uwm_held_out))
    self.assertGreater(float(np.min(wm_scores)), float(np.max(uwm_scores)))


if __name__ == '__main__':
  absltest.main()
