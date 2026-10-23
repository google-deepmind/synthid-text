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

"""Expected g-value for watermarking."""


def expected_mean_g_value(
    vocab_size: int,
    num_leaves: int = 2,
) -> float:
  """Computes the expected mean g-value of watermarked text for a uniform LM.

  This is the theoretical expected value for a single layer of tournament
  watermarking, using a Bernoulli(0.5) g-value distribution and N=num_leaves
  samples, assuming that the LM distribution p_LM is uniform.

  Args:
    vocab_size: The size of the vocabulary.
    num_leaves: Number of leaves per node in the tournament tree (N in the
      paper). Defaults to 2.

  Returns:
    The expected mean g-value for watermarked text.

  Raises:
    ValueError: If num_leaves is not 2 or 3.
  """
  if num_leaves == 2:
    # This equation is from Corollary 27 in Supplementary Information of paper,
    # in the case where p_LM is uniform.
    return 0.5 + 0.25 * (1 - (1 / vocab_size))
  elif num_leaves == 3:
    # This case can be derived from Theorem 25 in Supplementary Information of
    # the paper, in the case where N=3 and p_LM is uniform.
    return 7 / 8 - (3 / (8 * vocab_size))
  else:
    raise ValueError(
        f'Only 2 or 3 leaves are supported for the expected mean g-value'
        f' computation, but got {num_leaves}.'
    )
