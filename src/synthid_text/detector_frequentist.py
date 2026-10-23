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
