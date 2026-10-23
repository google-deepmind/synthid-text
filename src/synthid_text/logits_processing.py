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

"""SynthID text logits processing and watermarked score updates."""

from collections.abc import Sequence

import torch
import transformers

from synthid_text import hashing_function


def update_scores(
    scores: torch.FloatTensor,
    g_values: torch.FloatTensor,
) -> torch.FloatTensor:
  """Updates scores using the g-values for tournament watermarking.

  This is the num_leaves=2 case of `update_scores_distortionary`. The scores
  are assumed to be in log space.

  Args:
    scores: Log-space scores (batch_size, vocab_size).
    g_values: Binary g-values (batch_size, vocab_size, depth).

  Returns:
    Updated log-probabilities (batch_size, vocab_size). Entries with zero
    probability are set to -1e12.
  """
  _, _, depth = g_values.shape

  probs = torch.softmax(scores, dim=1)

  for i in range(depth):
    g_values_at_depth = g_values[:, :, i]
    g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
    probs = probs * (1 + g_values_at_depth - g_mass_at_depth)

  log_probs = torch.log(probs)
  return torch.where(torch.isfinite(log_probs), log_probs, -1e12)


def update_scores_distortionary(
    scores: torch.FloatTensor,
    g_values: torch.FloatTensor,
    num_leaves: int,
) -> torch.FloatTensor:
  """Updates scores using the g-values for distortionary tournaments.

  The scores are assumed to be in log space.

  Args:
    scores: Log-space scores (batch_size, vocab_size).
    g_values: Binary g-values (batch_size, vocab_size, depth).
    num_leaves: Number of leaves per node in the tournament tree.

  Returns:
    Updated log-probabilities (batch_size, vocab_size). Entries with zero
    probability are set to -1e12.
  """
  _, _, depth = g_values.shape

  probs = torch.softmax(scores, dim=1)

  for i in range(depth):
    g_values_at_depth = g_values[:, :, i]
    g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
    coeff_not_in_g = (1 - g_mass_at_depth)**(num_leaves - 1)
    # (1 - (1 - G)**N) / G, computed without cancellation for small G. Rounding
    # can push G slightly above 1, where log1p(-G) is NaN, so G is clamped to
    # 1, where the coefficient is exactly 1. For G == 0 this is NaN, but it is
    # only selected for tokens with g == 1 and probs > 0, which imply G > 0.
    g_mass_clamped = torch.clamp(g_mass_at_depth, max=1.0)
    coeff_in_g = (
        -torch.expm1(num_leaves * torch.log1p(-g_mass_clamped)) / g_mass_clamped
    )
    coeffs = torch.where(
        torch.logical_and(g_values_at_depth == 1, probs > 0),
        coeff_in_g, coeff_not_in_g)
    probs = probs * coeffs

  log_probs = torch.log(probs)
  return torch.where(torch.isfinite(log_probs), log_probs, -1e12)


class SynthIDState:
  """SynthID watermarking state.

  Attributes:
    context: Last ngram_len - 1 tokens (batch_size, ngram_len - 1).
    context_history: Packed int64 keys of the previously seen contexts, most
      recent first (batch_size, context_history_size).
    num_calls: Number of calls to `_watermark_candidates`.
  """

  def __init__(
      self,
      batch_size: int,
      ngram_len: int,
      context_history_size: int,
      device: torch.device,
  ):
    """Initializes the state.

    Args:
      batch_size: Batch size.
      ngram_len: Ngram length.
      context_history_size: Size of the tensor to keep track of seen contexts.
      device: Device to use.
    """
    self.context = torch.zeros(
        (batch_size, ngram_len - 1),
        dtype=torch.int64,
        device=device,
    )
    self.context_history = torch.zeros(
        (batch_size, context_history_size),
        dtype=torch.int64,
        device=device,
    )
    self.num_calls = 0


class SynthIDLogitsProcessor(transformers.LogitsProcessor):
  """SynthID watermarking logits processor.

  Logits processor updates the provided scores based on the binary g-values
  derived from the Threefry PRNG given a master secret key.

  An instance is stateful and must be used for a single generation: it must be
  called exactly once per generated token, with a fixed batch size and with
  input_ids one token longer than at the previous call. The prompt tokens are
  not used as watermarking context: the context starts as ngram_len - 1 zeros
  and is filled with the generated tokens.
  """

  def __init__(
      self,
      *,
      ngram_len: int,
      key: int = 654,
      depth: int = 30,
      context_history_size: int = 1024,
      device: torch.device,
      skip_first_ngram_calls: bool = True,
      num_leaves: int = 2,
  ):
    """Initializes the logits processor.

    Args:
      ngram_len: Ngram length (context length + 1), >= 2.
      key: Master secret key for Threefry PRNG watermarking, an int in [0,
        2**63). Defaults to 654.
      depth: Number of tournament layers (D), >= 1. Defaults to 30.
      context_history_size: Size of the tensor to keep track of seen contexts.
        Defaults to 1024.
      device: Device to use.
      skip_first_ngram_calls: If True, the first ngram_len - 1 calls return the
        scores unmodified and do not add their contexts to the context history;
        the context is still advanced. These calls have contexts that contain
        the initial zeros, and the tokens they produce are not scored by
        detection. Defaults to True.
      num_leaves: Number of leaves per node in the tournament tree, >= 2.
        Defaults to 2.

    Raises:
      ValueError: If key, depth, ngram_len or num_leaves is not an int or is
        outside its valid range.
    """
    # bool is a subclass of int; reject it explicitly.
    if isinstance(key, bool) or not isinstance(key, int):
      raise ValueError(f"`key` has to be an int, but is {key!r}")
    if not 0 <= key < 2**63:
      raise ValueError(f"`key` has to be in [0, 2**63), but is {key}")
    if isinstance(depth, bool) or not isinstance(depth, int):
      raise ValueError(f"`depth` has to be an int, but is {depth!r}")
    if depth < 1:
      raise ValueError(f"`depth` has to be >= 1, but is {depth}")
    if isinstance(ngram_len, bool) or not isinstance(ngram_len, int):
      raise ValueError(f"`ngram_len` has to be an int, but is {ngram_len!r}")
    # With ngram_len=1 the context is empty, so every step after the first
    # repeats it and nothing is watermarked.
    if ngram_len < 2:
      raise ValueError(f"`ngram_len` has to be >= 2, but is {ngram_len}")
    if isinstance(num_leaves, bool) or not isinstance(num_leaves, int):
      raise ValueError(f"`num_leaves` has to be an int, but is {num_leaves!r}")
    # num_leaves=1 leaves the scores unchanged; num_leaves<=0 is meaningless.
    if num_leaves < 2:
      raise ValueError(f"`num_leaves` has to be >= 2, but is {num_leaves}")

    self.ngram_len = ngram_len
    self.key = key
    self.depth = depth
    self.base_key = hashing_function.threefry_seed(
        torch.tensor(self.key, device=device, dtype=torch.int64)
    )

    self.context_history_size = context_history_size
    self.device = self.base_key.device
    self.state = None
    self.skip_first_ngram_calls = skip_first_ngram_calls
    self.num_leaves = num_leaves

  def _init_state(self, batch_size: int) -> SynthIDState:
    """Initializes and returns the state."""
    return SynthIDState(
        batch_size=batch_size,
        ngram_len=self.ngram_len,
        context_history_size=self.context_history_size,
        device=self.device,
    )

  def _watermark_candidates(
      self,
      input_ids: torch.LongTensor,
      candidate_scores: torch.FloatTensor,
      candidate_indices: torch.LongTensor,
  ) -> torch.FloatTensor:
    """Updates the state and watermarks a set of candidate tokens.

    Args:
      input_ids: Input token IDs (batch_size, input_len).
      candidate_scores: Log-space scores of the candidates (batch_size,
        num_candidates). Entries of -inf have zero probability.
      candidate_indices: Vocabulary indices of the candidates (batch_size,
        num_candidates).

    Returns:
      Watermarked log-space scores of the candidates (batch_size,
      num_candidates). Candidates with zero probability remain at -inf.

    Raises:
      ValueError: If input_ids is not 2-D, if candidate_scores is on an
        unexpected device, or if the batch sizes of input_ids and
        candidate_scores differ.
    """
    self._check_input_ids_shape(input_ids)
    if candidate_scores.device != self.device:
      raise ValueError(
          "SynthIDLogitsProcessor received inputs with unexpected device.",
      )
    batch_size = input_ids.shape[0]
    if candidate_scores.shape[0] != batch_size:
      raise ValueError(
          f"input_ids has batch size {batch_size}, but the scores have batch"
          f" size {candidate_scores.shape[0]}."
      )

    if self.state is None:
      self.state = self._init_state(batch_size)
    else:
      self.state.context = torch.concat(
          (self.state.context, input_ids[:, -1:]),
          dim=1,
      )
      self.state.context = self.state.context[:, 1:]

    self.state.num_calls += 1

    if self.skip_first_ngram_calls and self.state.num_calls < self.ngram_len:
      return candidate_scores

    # 1. Compute context key and binary g-values using Threefry PRNG
    base_key_batch = self.base_key.expand(batch_size, 2)
    context_key = hashing_function.recursively_fold_in(
        base_key_batch, self.state.context
    )
    g_values = hashing_function.compute_g_values_boolean(
        context_key, candidate_indices, depth=self.depth
    )

    # 2. Modify scores
    if self.num_leaves == 2:
      updated_scores = update_scores(candidate_scores, g_values)
    else:
      updated_scores = update_scores_distortionary(
          candidate_scores, g_values, self.num_leaves
      )
    # update_scores maps zero-probability entries to -1e12; keep them at -inf.
    updated_scores = torch.where(
        torch.isneginf(candidate_scores), candidate_scores, updated_scores
    )

    # 3. Check if the current watermarking context was previously used
    packed_ctx = hashing_function.pack_key_to_int64(context_key)[:, None]
    is_repeated_context = (self.state.context_history == packed_ctx).any(
        dim=1,
        keepdim=True,
    )
    self.state.context_history = torch.concat(
        (packed_ctx, self.state.context_history),
        dim=1,
    )[:, :-1]

    return torch.where(
        is_repeated_context,
        input=candidate_scores,
        other=updated_scores,
    )

  @torch.no_grad
  def __call__(
      self,
      input_ids: torch.LongTensor,
      scores: torch.FloatTensor,
  ) -> torch.FloatTensor:
    """Watermarks already-processed scores as a standard LogitsProcessor.

    This is intended to run after all other logits processors and sampling
    warpers (temperature, top-k, top-p, ...), so that it watermarks the exact
    distribution that would otherwise be sampled from. Only tokens with finite
    scores are candidates; g-values are computed for the largest number of
    finite entries in any batch row. Sampling warpers such as top-k keep this
    number small. Without truncation every vocabulary entry is a candidate,
    which is correct but slow.

    The processor is stateful: an instance must be used for a single
    generation and called exactly once per generated token, with input_ids
    growing by one token per call. The prompt tokens are not used as
    watermarking context; the context starts as ngram_len - 1 zeros.

    Args:
      input_ids: Input token IDs (batch_size, input_len).
      scores: Processed log-space scores (batch_size, vocab_size). Entries may
        be -inf, but not NaN or +inf.

    Returns:
      Watermarked scores (batch_size, vocab_size). Entries that were -inf
      remain -inf.

    Raises:
      ValueError: If scores contains NaN or +inf, if every score is -inf, if
        input_ids is not 2-D, if scores is on an unexpected device, or if the
        batch sizes of input_ids and scores differ.
    """
    if not (torch.isfinite(scores) | torch.isneginf(scores)).all():
      raise ValueError(
          "Scores contain NaN or +inf entries; only finite values and -inf are"
          " allowed."
      )
    num_candidates = int(torch.isfinite(scores).sum(dim=1).max())
    if num_candidates == 0:
      raise ValueError("All scores are -inf; nothing to sample from.")
    candidates = torch.topk(scores, k=num_candidates, dim=1)
    watermarked = self._watermark_candidates(
        input_ids, candidates.values, candidates.indices
    )
    return torch.full_like(scores, -float("inf")).scatter(
        1, candidates.indices, watermarked
    )

  def compute_g_values_from_ngrams(
      self,
      ngrams: torch.LongTensor,
  ) -> torch.Tensor:
    """Computes g-values for each ngram in a batch of ngrams.

    Args:
      ngrams: Ngrams (batch_size, num_ngrams, ngram_len).

    Returns:
      float32 g-values in {0.0, 1.0} (batch_size, num_ngrams, depth).

    Raises:
      ValueError: If ngrams is not 3-D or its last dimension is not ngram_len.
    """
    if len(ngrams.shape) != 3:
      raise ValueError(
          "Ngrams should be of shape (batch_size, num_ngrams, ngram_len), but"
          f" is {ngrams.shape}"
      )
    if ngrams.shape[2] != self.ngram_len:
      raise ValueError(
          "Ngrams should be of shape (batch_size, num_ngrams, ngram_len),"
          f" where ngram_len is {self.ngram_len}, but is {ngrams.shape}"
      )
    batch_size, num_ngrams, _ = ngrams.shape
    contexts = ngrams[..., :-1].reshape(
        batch_size * num_ngrams, self.ngram_len - 1
    )
    target_tokens = ngrams[..., -1:].reshape(batch_size * num_ngrams, 1)

    base_key_expanded = self.base_key.expand(batch_size * num_ngrams, 2)
    context_keys = hashing_function.recursively_fold_in(
        base_key_expanded, contexts
    )
    g_values = hashing_function.compute_g_values_boolean(
        context_keys, target_tokens, depth=self.depth
    ).view(batch_size, num_ngrams, self.depth)

    return g_values

  def _check_input_ids_shape(self, input_ids: torch.LongTensor):
    """Checks the shape of input token IDs."""
    if len(input_ids.shape) != 2:
      raise ValueError(
          "Input ids should be of shape (batch_size, input_len), but is"
          f" {input_ids.shape}"
      )

  def compute_g_values(
      self,
      input_ids: torch.LongTensor,
  ) -> torch.Tensor:
    """Computes g-values for each ngram from the given sequence of tokens.

    Args:
      input_ids: Input token IDs (batch_size, input_len).

    Returns:
      float32 g-values in {0.0, 1.0} (batch_size, input_len - (ngram_len - 1),
      depth).

    Raises:
      ValueError: If input_ids is not 2-D.
    """
    self._check_input_ids_shape(input_ids)
    batch_size, seq_len = input_ids.shape
    num_ngrams = seq_len - (self.ngram_len - 1)
    if num_ngrams <= 0:
      return torch.empty(
          (batch_size, 0, self.depth),
          device=self.device,
          dtype=torch.float32,
      )
    ngrams = input_ids.unfold(dimension=1, size=self.ngram_len, step=1)
    return self.compute_g_values_from_ngrams(ngrams)

  def compute_context_repetition_mask(
      self,
      input_ids: torch.LongTensor,
  ) -> torch.BoolTensor:
    """Computes a mask of the positions whose context has not been seen before.

    Position j refers to the ngram ending at token j + ngram_len - 1, and its
    context is the preceding ngram_len - 1 tokens. A context counts as repeated
    only if it occurs among the previous context_history_size contexts in the
    same row.

    Args:
      input_ids: Input token IDs (batch_size, input_len).

    Returns:
      Boolean mask (batch_size, input_len - (ngram_len - 1)) that is True where
      the context has not been seen within the previous context_history_size
      contexts, and False where the context is repeated.

    Raises:
      ValueError: If input_ids is not 2-D.
    """
    self._check_input_ids_shape(input_ids)
    batch_size, seq_len = input_ids.shape
    num_contexts = seq_len - (self.ngram_len - 1)
    if num_contexts <= 0:
      return torch.empty((batch_size, 0), device=self.device, dtype=torch.bool)

    # Packed keys of the previously seen contexts, most recent first. As in
    # SynthIDState, it starts as zeros.
    context_history = torch.zeros(
        (batch_size, self.context_history_size),
        dtype=torch.int64,
        device=self.device,
    )
    contexts = input_ids[:, :-1].unfold(
        dimension=1,
        size=self.ngram_len - 1,
        step=1,
    )

    base_key_batch = self.base_key.expand(batch_size, num_contexts, 2)
    # Hash all contexts at once: (batch_size, num_contexts).
    context_hashes = hashing_function.pack_key_to_int64(
        hashing_function.recursively_fold_in(base_key_batch, contexts)
    )
    are_repeated_contexts = []
    for i in range(num_contexts):
      context_hash = context_hashes[:, i : i + 1]
      is_repeated_context = (context_history == context_hash).any(
          dim=1,
          keepdim=True,
      )
      are_repeated_contexts.append(is_repeated_context)
      context_history = torch.concat(
          (context_hash, context_history),
          dim=1,
      )[:, :-1]
    are_repeated_contexts = torch.concat(are_repeated_contexts, dim=1)

    return torch.logical_not(are_repeated_contexts)

  def compute_eos_token_mask(
      self,
      input_ids: torch.LongTensor,
      eos_token_id: int | Sequence[int],
  ) -> torch.BoolTensor:
    """Computes a mask of the tokens that precede the first EOS token.

    `compute_g_values_and_mask` drops the first ngram_len - 1 columns of this
    mask, so that each remaining entry refers to the ngram ending at that
    position. An ngram is then kept only if it neither contains nor follows an
    EOS token.

    Args:
      input_ids: Input token IDs (batch_size, input_len).
      eos_token_id: EOS token ID or sequence of EOS token IDs.

    Returns:
      Boolean mask (batch_size, input_len) that is True at positions before the
      first EOS token of each row, and False at the first EOS token and every
      position after it. Rows without an EOS token are all True.

    Raises:
      ValueError: If input_ids is not 2-D.
    """
    self._check_input_ids_shape(input_ids)
    eos_ids = torch.as_tensor(
        eos_token_id, dtype=input_ids.dtype, device=input_ids.device
    )
    # Number of EOS tokens up to and including each position. It is zero
    # exactly at the positions before the first EOS token.
    num_eos_so_far = torch.isin(input_ids, eos_ids).long().cumsum(dim=1)
    return num_eos_so_far == 0

  def compute_g_values_and_mask(
      self,
      input_ids: torch.LongTensor,
      eos_token_id: int | Sequence[int],
  ) -> tuple[torch.Tensor, torch.BoolTensor]:
    """Computes g-values and the mask of positions that should be scored.

    A position is scored if its ngram does not contain or follow an EOS token
    and its context (the preceding ngram_len - 1 tokens) has not occurred
    within the previous context_history_size contexts.

    Args:
      input_ids: Generated token IDs (batch_size, input_len), with the prompt
        removed.
      eos_token_id: EOS token ID or sequence of EOS token IDs.

    Returns:
      A tuple (g_values, mask) of float32 g-values (batch_size, input_len -
      (ngram_len - 1), depth) and a boolean mask (batch_size, input_len -
      (ngram_len - 1)) that is True at the positions to score.

    Raises:
      ValueError: If input_ids is not 2-D.
    """
    eos_token_mask = self.compute_eos_token_mask(
        input_ids=input_ids,
        eos_token_id=eos_token_id,
    )[:, self.ngram_len - 1 :]
    context_repetition_mask = self.compute_context_repetition_mask(
        input_ids=input_ids,
    )
    mask = context_repetition_mask * eos_token_mask
    g_values = self.compute_g_values(input_ids=input_ids)
    return g_values, mask
