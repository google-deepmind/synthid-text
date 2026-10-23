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

"""Frequentist watermark detection: exact p-value detector."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats
import torch

from synthid_text import logits_processing


class FrequentistDetector:
  """Watermark detector based on the count of g-values equal to 1.

  Under the null hypothesis that the text was generated without the watermark
  key, the unmasked g-values are modelled as independent Bernoulli(0.5)
  variables: the mask keeps only positions whose ngram does not contain or
  follow an EOS token and whose context has not occurred among the previous
  `context_history_size` contexts, so each g-value comes from a distinct PRNG
  input. The number of ones is then Binomial(num_unmasked * depth, 0.5), and
  `p_value` returns the exact one-sided tail probability.

  Repeated contexts are only detected within the last `context_history_size`
  contexts of the logits processor, so for sequences longer than that the
  independence assumption can be violated.
  """

  def __init__(
      self,
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      eos_token_id: int | Sequence[int],
  ):
    """Initializes the detector.

    Args:
      logits_processor: The SynthIDLogitsProcessor with the same watermarking
        config as used for generation.
      eos_token_id: EOS token ID or sequence of EOS token IDs that mark the end
        of generation.
    """
    self.logits_processor = logits_processor
    self.eos_token_id = eos_token_id

  def p_value(self, outputs: torch.LongTensor) -> np.ndarray:
    """Computes the p-value of the watermark test for each sequence.

    Small values are evidence that the text was watermarked with this key.
    Sequences with no scored positions get a p-value of 1.0.

    Args:
      outputs: Generated token IDs (batch_size, input_len), with the prompt
        removed.

    Returns:
      p-values (batch_size,).
    """
    g_values, mask = self.logits_processor.compute_g_values_and_mask(
        input_ids=outputs,
        eos_token_id=self.eos_token_id,
    )
    depth = g_values.shape[-1]
    num_ones = (g_values * mask.unsqueeze(-1)).sum(dim=(1, 2)).cpu().numpy()
    num_trials = (mask.sum(dim=1) * depth).cpu().numpy()
    # P(X >= num_ones) for X ~ Binomial(num_trials, 0.5). For num_trials == 0
    # this is P(X >= 0) = 1.
    return stats.binom.sf(num_ones - 1, num_trials, 0.5)


class DualKeyFrequentistDetector:
  """Frequentist watermark detector pooling g-values from draft and target keys.

  Dual-key speculative decoding interleaves tokens watermarked with `key_draft`
  and tokens watermarked with `key_target`. Because the detector does not know
  which key produced each token, it evaluates every unmasked position under
  both keys.

  Under the null hypothesis that the text was generated without either
  watermark key (with `key_draft != key_target`), the unmasked g-values across
  both keys are modelled as independent Bernoulli(0.5) variables: for each key,
  the mask keeps only positions whose ngram does not contain or follow an EOS
  token and whose context has not occurred among the previous
  `context_history_size` contexts. The pooled count of ones is then
  Binomial(num_trials, 0.5), and `p_value` returns the exact one-sided tail
  probability per sequence.
  """

  def __init__(
      self,
      draft_processor: logits_processing.SynthIDLogitsProcessor,
      target_processor: logits_processing.SynthIDLogitsProcessor,
      eos_token_id: int | Sequence[int],
  ):
    """Initializes the dual-key detector.

    Args:
      draft_processor: SynthIDLogitsProcessor configured with `key_draft`.
      target_processor: SynthIDLogitsProcessor configured with `key_target`.
      eos_token_id: EOS token ID or sequence of EOS token IDs that mark the end
        of generation.

    Raises:
      ValueError: If draft_processor and target_processor share the same key or
        have different `depth`, `ngram_len`, `context_history_size` or
        `device`.
    """
    if draft_processor.key == target_processor.key:
      raise ValueError(
          "draft_processor and target_processor must have distinct keys, but"
          f" both have key={draft_processor.key}."
      )
    if draft_processor.depth != target_processor.depth:
      raise ValueError(
          "draft_processor and target_processor must have the same depth, got"
          f" {draft_processor.depth} and {target_processor.depth}."
      )
    if draft_processor.ngram_len != target_processor.ngram_len:
      raise ValueError(
          "draft_processor and target_processor must have the same ngram_len,"
          f" got {draft_processor.ngram_len} and {target_processor.ngram_len}."
      )
    if (
        draft_processor.context_history_size
        != target_processor.context_history_size
    ):
      raise ValueError(
          "draft_processor and target_processor must have the same"
          " context_history_size, got"
          f" {draft_processor.context_history_size} and"
          f" {target_processor.context_history_size}."
      )
    if draft_processor.device != target_processor.device:
      raise ValueError(
          "draft_processor and target_processor must be on the same device,"
          f" got {draft_processor.device} and {target_processor.device}."
      )
    self.draft_processor = draft_processor
    self.target_processor = target_processor
    self.eos_token_id = eos_token_id

  @classmethod
  def from_config(
      cls,
      config: Any,
      eos_token_id: int | Sequence[int],
  ) -> "DualKeyFrequentistDetector":
    """Constructs a detector from a DualKeyWatermarkingConfig."""
    return cls(
        draft_processor=config.make_draft_processor(),
        target_processor=config.make_target_processor(),
        eos_token_id=eos_token_id,
    )

  def p_value(self, outputs: torch.LongTensor) -> np.ndarray:
    """Computes the exact one-sided binomial p-value for each sequence.

    Small values are evidence that the text was watermarked with `(key_draft,
    key_target)`. Sequences with no scored positions get a p-value of 1.0.

    Args:
      outputs: Generated token IDs of shape (batch_size, input_len), with the
        prompt removed, on the processors' device.

    Returns:
      p-values of shape (batch_size,).
    """
    g_draft, mask_draft = self.draft_processor.compute_g_values_and_mask(
        input_ids=outputs, eos_token_id=self.eos_token_id
    )
    g_target, mask_target = self.target_processor.compute_g_values_and_mask(
        input_ids=outputs, eos_token_id=self.eos_token_id
    )
    depth = g_draft.shape[-1]
    ones_draft = (g_draft * mask_draft.unsqueeze(-1)).sum(dim=(1, 2))
    ones_target = (g_target * mask_target.unsqueeze(-1)).sum(dim=(1, 2))
    num_ones = (ones_draft + ones_target).cpu().numpy()
    num_trials = (
        ((mask_draft.sum(dim=1) + mask_target.sum(dim=1)) * depth).cpu().numpy()
    )
    return stats.binom.sf(num_ones - 1, num_trials, 0.5)
