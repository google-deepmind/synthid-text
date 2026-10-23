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

"""Dual-key speculative decoding engine for SynthID tournament watermarking.

Reference:
  Dathathri et al., "Scalable watermarking for identifying large language
  model outputs", Nature 634, 818-823 (2024),
  https://doi.org/10.1038/s41586-024-08025-4.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

import torch
import transformers

from synthid_text import logits_processing


def _top_k_log_probs(
    logits: torch.Tensor, top_k: int, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
  """Returns the temperature-scaled top-k distribution of logits (..., vocab_size).

  Args:
    logits: Logits (..., vocab_size).
    top_k: Number of highest-probability tokens to keep.
    temperature: Positive temperature for scaling logits.

  Returns:
    A tuple (log_probs, tokens), both of shape (..., k), where `tokens` are the
    vocabulary token IDs of the k highest-logit tokens and `log_probs` are
    renormalized over those k tokens in float32.
  """
  values, tokens = torch.topk(
      logits.to(torch.float32), k=min(top_k, logits.shape[-1]), dim=-1
  )
  return torch.log_softmax(values / temperature, dim=-1), tokens


def _lookup_probs(
    probs: torch.Tensor, tokens: torch.Tensor, query_tokens: torch.Tensor
) -> torch.Tensor:
  """Evaluates a 1D (k,) distribution (probs, tokens) at 1D query_tokens."""
  matches = query_tokens[:, None] == tokens[None, :]
  return (probs[None, :] * matches).sum(dim=1)


def _watermark_and_sample(
    processor: logits_processing.SynthIDLogitsProcessor,
    input_ids: torch.LongTensor,
    candidate_log_probs: torch.Tensor,
    candidate_tokens: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
  """Watermarks a (1, k) candidate distribution and samples a (1, 1) token."""
  watermarked = processor.watermark_candidates(
      input_ids, candidate_log_probs, candidate_tokens
  )
  probs = torch.softmax(watermarked, dim=-1)
  if generator is not None:
    probs = probs.to(generator.device)
  sampled_idx = torch.multinomial(probs, num_samples=1, generator=generator)
  return torch.gather(
      candidate_tokens, -1, sampled_idx.to(candidate_tokens.device)
  )


def _crop_cache(cache: Any, target_length: int) -> None:
  """Crops a Transformers KV cache in-place to target_length."""
  tokens_to_remove = cache.get_seq_length() - target_length
  if tokens_to_remove >= 0:
    cache.crop(-tokens_to_remove)


def _trim_sliding_window_for_step(
    cache: Any,
    stashed_kv: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]],
) -> None:
  """Trims sliding-window layers to sliding_window - 1 before a 1-token step."""
  for idx, layer in enumerate(getattr(cache, "layers", ())):
    if getattr(layer, "is_sliding", False) and getattr(
        layer, "is_initialized", False
    ):
      max_cached = layer.sliding_window - 1
      excess = layer.keys.shape[-2] - max_cached
      if excess > 0:
        k_stash, v_stash = stashed_kv.setdefault(idx, ([], []))
        k_stash.append(layer.keys[:, :, :excess, :])
        v_stash.append(layer.values[:, :, :excess, :])
        layer.keys = layer.keys[:, :, excess:, :]
        layer.values = layer.values[:, :, excess:, :]


def _restore_sliding_window_stash(
    cache: Any,
    stashed_kv: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]],
) -> None:
  """Restores stashed sliding-window keys/values before cropping the cache."""
  for idx, (k_stash, v_stash) in stashed_kv.items():
    layer = cache.layers[idx]
    layer.keys = torch.cat([*k_stash, layer.keys], dim=-2)
    layer.values = torch.cat([*v_stash, layer.values], dim=-2)


def _default_model_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda:0")
  return torch.device("cpu")


@dataclasses.dataclass
class DualKeyWatermarkingConfig:
  """Configuration for dual-key tournament watermarked speculative decoding.

  Attributes:
    ngram_len: Ngram length (context length is ngram_len - 1), >= 2.
    key_draft: Secret watermarking key for candidate tokens drafted by the draft
      model, an int in [0, 2**63).
    key_target: Secret watermarking key for residual replacement and bonus
      tokens from the target model, an int in [0, 2**63) distinct from
      key_draft.
    depth: Number of tournament layers (D), >= 1.
    context_history_size: Number of recently seen contexts to remember.
    temperature: Positive temperature for scaling model logits.
    top_k: Number of highest-probability candidate tokens kept when sampling, >=
      2.
    draft_len: Number of candidate tokens drafted per speculative step, >= 1.
    watermark_device: Device for watermarking, rejection sampling and the
      generated sequence. Generation runs with batch size 1, so watermarking
      operates on tensors with at most top_k entries; on a GPU it is dominated
      by kernel launch overhead and is usually faster on the CPU. The top-k
      candidates are copied to this device at every step. Defaults to CPU.
    target_device: Device of the target model. Defaults to cuda:0 if available,
      otherwise CPU.
    draft_device: Device of the draft model. Defaults to cuda:0 if available,
      otherwise CPU.
    skip_first_ngram_calls: If True, the first ngram_len - 1 generated tokens
      are not watermarked (matching detection, which never scores them).
    num_leaves: Number of leaves per node in the tournament tree; must be 2.
      Rejection sampling compares against the unwatermarked draft distribution
      q_j, which is only correct if the watermark is non-distortionary for a
      single token (averaged over keys), and that holds only for num_leaves=2.
  """

  ngram_len: int = 5
  key_draft: int = 42
  key_target: int = 84
  depth: int = 15
  context_history_size: int = 1024
  temperature: float = 1.0
  top_k: int = 40
  draft_len: int = 4
  watermark_device: torch.device = dataclasses.field(
      default_factory=lambda: torch.device("cpu")
  )
  target_device: torch.device = dataclasses.field(
      default_factory=_default_model_device
  )
  draft_device: torch.device = dataclasses.field(
      default_factory=_default_model_device
  )
  skip_first_ngram_calls: bool = True
  num_leaves: int = 2

  def __post_init__(self):
    """Validates the configuration.

    Raises:
      ValueError: If any parameter is outside its valid range or if key_draft
        equals key_target.
    """
    if self.temperature <= 0.0:
      raise ValueError(
          f"temperature must be strictly positive, got {self.temperature}"
      )
    if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
      raise ValueError(f"top_k must be an int, got {self.top_k!r}")
    if self.top_k < 2:
      raise ValueError(f"top_k must be >= 2, got {self.top_k}")
    if isinstance(self.draft_len, bool) or not isinstance(self.draft_len, int):
      raise ValueError(f"draft_len must be an int, got {self.draft_len!r}")
    if self.draft_len < 1:
      raise ValueError(f"draft_len must be >= 1, got {self.draft_len}")
    if self.num_leaves != 2:
      raise ValueError(
          "num_leaves must be 2: speculative sampling assumes the watermark is"
          " non-distortionary for a single token, which only holds for"
          f" num_leaves=2, got {self.num_leaves!r}."
      )
    # Constructing the processors validates ngram_len, keys and depth.
    self.make_draft_processor()
    self.make_target_processor()
    if self.key_draft == self.key_target:
      raise ValueError(
          "key_draft and key_target must be distinct, but both are"
          f" {self.key_draft}."
      )

  def make_draft_processor(self) -> logits_processing.SynthIDLogitsProcessor:
    """Returns a fresh SynthIDLogitsProcessor configured with key_draft."""
    return self._make_processor(self.key_draft)

  def make_target_processor(self) -> logits_processing.SynthIDLogitsProcessor:
    """Returns a fresh SynthIDLogitsProcessor configured with key_target."""
    return self._make_processor(self.key_target)

  def _make_processor(
      self, key: int
  ) -> logits_processing.SynthIDLogitsProcessor:
    return logits_processing.SynthIDLogitsProcessor(
        ngram_len=self.ngram_len,
        key=key,
        depth=self.depth,
        context_history_size=self.context_history_size,
        device=self.watermark_device,
        skip_first_ngram_calls=self.skip_first_ngram_calls,
        num_leaves=self.num_leaves,
    )


@dataclasses.dataclass
class SpeculativeGenerationOutput:
  """Output of dual-key speculative generation.

  Attributes:
    sequences: Prompt and generated token IDs of shape (1, total_len).
    tokens_generated: Total number of new tokens generated.
    num_draft_tokens: Total number of candidate tokens proposed by the draft
      model.
    num_accepted_tokens: Total number of candidate tokens accepted from the
      draft model.
    acceptance_rate: Ratio of accepted draft tokens to total proposed draft
      tokens (0.0 if no draft tokens were proposed).
    token_origins: List of (origin_type, token_id) tuples recording the source
      of each generated token. origin_type is one of 'draft', 'target_residual',
      or 'target_bonus'.
    num_steps: Number of speculative steps executed.
  """

  sequences: torch.LongTensor
  tokens_generated: int
  num_draft_tokens: int
  num_accepted_tokens: int
  acceptance_rate: float
  token_origins: list[tuple[str, int]] = dataclasses.field(default_factory=list)
  num_steps: int = 0


class DualKeySpeculativeEngine:
  """Dual-key watermarked speculative decoding engine.

  Let p_j and q_j be the target and draft next-token distributions at position
  j after temperature scaling and top-k truncation (renormalized over the top_k
  tokens). At each speculative step:
    1. The draft model proposes up to `draft_len` candidate tokens, each
       sampled from q_j watermarked with `key_draft`.
    2. The target model evaluates the candidate sequence in a single forward
       pass.
    3. Each candidate token x_j is accepted with probability
       min(1, p_j(x_j) / q_j(x_j)). On the first rejection, a replacement token
       is sampled from the residual distribution max(0, p_j - q_j) (normalized)
       watermarked with `key_target`.
    4. If all candidate tokens are accepted, one bonus token is sampled from
       the target distribution p_j watermarked with `key_target`.

  With num_leaves=2 the tournament watermark is non-distortionary for each
  token, so averaged over watermarking keys each generated token follows the
  top-k target distribution p_j, as in autoregressive top-k sampling. For a
  fixed pair of keys the output is not the key_target-watermarked target
  distribution, because accepted draft tokens carry the key_draft watermark.

  The draft and target models must share a tokenizer and have the same
  vocabulary size.
  """

  def __init__(
      self,
      target_model: transformers.PreTrainedModel,
      draft_model: transformers.PreTrainedModel,
      config: DualKeyWatermarkingConfig,
  ):
    """Initializes the speculative decoding engine.

    Args:
      target_model: Target causal language model.
      draft_model: Draft causal language model.
      config: Dual-key watermarking and speculative decoding configuration.

    Raises:
      ValueError: If target_model and draft_model have different vocabulary
        sizes.
    """
    target_vocab_size = target_model.config.get_text_config().vocab_size
    draft_vocab_size = draft_model.config.get_text_config().vocab_size
    if target_vocab_size != draft_vocab_size:
      raise ValueError(
          "target_model and draft_model must have the same vocab_size, got"
          f" {target_vocab_size} and {draft_vocab_size}."
      )
    self.target_model = target_model
    self.draft_model = draft_model
    self.config = config
    self.vocab_size = target_vocab_size

  def _top_k_log_probs(
      self, logits: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the top-k (log_probs, tokens) and moves them to the watermark device."""
    log_probs, tokens = _top_k_log_probs(
        logits, self.config.top_k, self.config.temperature
    )
    device = self.config.watermark_device
    return log_probs.to(device), tokens.to(device)

  @torch.no_grad()
  def generate(
      self,
      prompt_ids: torch.LongTensor,
      max_new_tokens: int,
      eos_token_id: int | Sequence[int] | None = None,
      generator: torch.Generator | None = None,
  ) -> SpeculativeGenerationOutput:
    """Generates watermarked text using dual-key speculative decoding.

    Args:
      prompt_ids: Prompt token IDs of shape (1, prompt_len).
      max_new_tokens: Maximum number of new tokens to generate, >= 0.
      eos_token_id: Optional EOS token ID or sequence of EOS token IDs for early
        stopping.
      generator: Optional PyTorch RNG for reproducible sampling.

    Returns:
      SpeculativeGenerationOutput with the generated sequence and statistics.

    Raises:
      ValueError: If prompt_ids does not have shape (1, prompt_len) with
        prompt_len >= 1, or if max_new_tokens < 0.
    """
    if (
        prompt_ids.ndim != 2
        or prompt_ids.shape[0] != 1
        or prompt_ids.shape[1] < 1
    ):
      raise ValueError(
          "prompt_ids must have shape (1, prompt_len) with prompt_len >= 1,"
          f" got {tuple(prompt_ids.shape)}."
      )
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 0
    ):
      raise ValueError(
          f"max_new_tokens must be a non-negative int, got {max_new_tokens!r}."
      )

    if eos_token_id is None:
      eos_ids: set[int] = set()
    elif isinstance(eos_token_id, int):
      eos_ids = {eos_token_id}
    else:
      eos_ids = set(eos_token_id)

    self.target_model.eval()
    self.draft_model.eval()

    generated = prompt_ids.clone().to(
        device=self.config.watermark_device, dtype=torch.int64
    )
    draft_processor = self.config.make_draft_processor()
    target_processor = self.config.make_target_processor()

    draft_cache: Any = transformers.DynamicCache(config=self.draft_model.config)
    draft_cache.activate_past_recording()
    draft_cached_len = 0
    target_cache: Any = transformers.DynamicCache(
        config=self.target_model.config
    )
    target_cache.activate_past_recording()
    target_cached_len = 0

    tokens_produced = 0
    total_draft_tokens = 0
    total_accepted_tokens = 0
    num_steps = 0
    token_origins: list[tuple[str, int]] = []

    while tokens_produced < max_new_tokens:
      draft_len = min(self.config.draft_len, max_new_tokens - tokens_produced)
      num_steps += 1
      cur_len = generated.shape[1]

      # Step 1: Draft up to `draft_len` candidate tokens with key_draft.
      draft_seq = generated
      if draft_processor.state is None:
        draft_states = [None]
      else:
        draft_states = [draft_processor.state.clone()]
      draft_top_k_probs: list[torch.Tensor] = []
      draft_top_k_tokens: list[torch.Tensor] = []
      stashed_sliding_kv: dict[
          int, tuple[list[torch.Tensor], list[torch.Tensor]]
      ] = {}

      for draft_step in range(draft_len):
        if draft_step > 0:
          _trim_sliding_window_for_step(draft_cache, stashed_sliding_kv)
        draft_out = self.draft_model(
            draft_seq[:, draft_cached_len:].to(self.config.draft_device),
            past_key_values=draft_cache,
            use_cache=True,
        )
        draft_cache = draft_out.past_key_values
        draft_cached_len = draft_seq.shape[1]
        if draft_step == 0:
          # All `cur_len` tokens in `generated` are already committed, so trim
          # any sliding-window layers at `cur_len` before drafting further.
          _crop_cache(draft_cache, cur_len)
        q_log_probs, q_tokens = self._top_k_log_probs(
            draft_out.logits[:, -1, :]
        )
        next_tok_tensor = _watermark_and_sample(
            draft_processor, draft_seq, q_log_probs, q_tokens, generator
        )
        assert draft_processor.state is not None
        draft_states.append(draft_processor.state.clone())
        draft_top_k_probs.append(q_log_probs[0].exp())
        draft_top_k_tokens.append(q_tokens[0])
        draft_seq = torch.cat([draft_seq, next_tok_tensor], dim=1)
        if int(next_tok_tensor.item()) in eos_ids:
          break

      _restore_sliding_window_stash(draft_cache, stashed_sliding_kv)

      draft_tokens: list[int] = draft_seq[0, cur_len:].tolist()
      actual_draft_len = len(draft_tokens)
      total_draft_tokens += actual_draft_len

      # Step 2: Target model parallel verification pass.
      target_out = self.target_model(
          draft_seq[:, target_cached_len:].to(self.config.target_device),
          past_key_values=target_cache,
          use_cache=True,
      )
      target_cache = target_out.past_key_values
      p_log_probs, p_tokens = self._top_k_log_probs(
          target_out.logits[:, -(actual_draft_len + 1) :, :]
      )
      target_top_k_probs = p_log_probs[0, :actual_draft_len].exp()
      target_top_k_tokens = p_tokens[0, :actual_draft_len]

      # Step 3: Rejection sampling of the draft tokens.
      if generator is not None:
        rand_device = generator.device
      else:
        rand_device = self.config.watermark_device
      u_draws = torch.rand(
          actual_draft_len, generator=generator, device=rand_device
      ).tolist()

      accepted_tokens: list[int] = []
      num_accepted_draft = 0
      all_accepted = True

      for j, x_j in enumerate(draft_tokens):
        prefix_ids = draft_seq[:, : cur_len + j]
        p_probs_j = target_top_k_probs[j]
        p_tokens_j = target_top_k_tokens[j]
        q_probs_j = draft_top_k_probs[j]
        q_tokens_j = draft_top_k_tokens[j]

        # Look up p_j(x_j) and q_j(x_j) in the 1D (k,) top-k tables (0.0 if x_j
        # is outside the target top-k).
        p_x = float(p_probs_j[p_tokens_j == x_j].sum())
        q_x = float(q_probs_j[q_tokens_j == x_j].sum())

        # The residual max(0, p_j - q_j) is zero outside p_tokens_j. If
        # floating-point rounding makes p_x < q_x while max(0, p_j - q_j) == 0
        # everywhere, p_j and q_j are numerically tied and x_j is accepted.
        residual: torch.Tensor | None = None
        res_sum = 0.0
        accepted = u_draws[j] <= min(1.0, p_x / q_x)
        if not accepted:
          q_at_p = _lookup_probs(q_probs_j, q_tokens_j, p_tokens_j)
          residual = torch.clamp(p_probs_j - q_at_p, min=0.0)
          res_sum = float(residual.sum())
          if res_sum <= 0.0:
            accepted = True

        if accepted:
          accepted_tokens.append(x_j)
          token_origins.append(("draft", x_j))
          num_accepted_draft += 1
          total_accepted_tokens += 1
          # Record the accepted context in target_processor so its context
          # history stays synchronized with draft_processor.
          target_processor.advance(prefix_ids)
          if x_j in eos_ids:
            all_accepted = False
            break
        else:
          assert residual is not None
          all_accepted = False
          # Roll draft_processor back to its state before x_j was drafted,
          # discarding any context history recorded at or after step j.
          draft_processor.state = draft_states[j]
          # Sample the replacement token from the normalized residual
          # distribution watermarked with key_target, and advance
          # draft_processor over the same prefix so both states match.
          replacement_tok = int(
              _watermark_and_sample(
                  target_processor,
                  prefix_ids,
                  torch.log(residual / res_sum).unsqueeze(0),
                  p_tokens_j.unsqueeze(0),
                  generator,
              ).item()
          )
          draft_processor.advance(prefix_ids)
          accepted_tokens.append(replacement_tok)
          token_origins.append(("target_residual", replacement_tok))
          break

      # Step 4: Bonus token if all candidate tokens were accepted.
      if all_accepted and (
          tokens_produced + len(accepted_tokens) < max_new_tokens
      ):
        prefix_ids = draft_seq[:, : cur_len + actual_draft_len]
        bonus_tok = int(
            _watermark_and_sample(
                target_processor,
                prefix_ids,
                p_log_probs[:, actual_draft_len, :],
                p_tokens[:, actual_draft_len, :],
                generator,
            ).item()
        )
        draft_processor.advance(prefix_ids)
        accepted_tokens.append(bonus_tok)
        token_origins.append(("target_bonus", bonus_tok))

      # Step 5: Crop KV caches to discard rejected draft tokens and advance.
      valid_cache_len = cur_len + num_accepted_draft
      _crop_cache(draft_cache, min(draft_cached_len, valid_cache_len))
      draft_cached_len = draft_cache.get_seq_length()
      _crop_cache(target_cache, valid_cache_len)
      target_cached_len = target_cache.get_seq_length()

      new_tokens = torch.tensor(
          [accepted_tokens],
          device=self.config.watermark_device,
          dtype=torch.int64,
      )
      generated = torch.cat([generated, new_tokens], dim=1)
      tokens_produced += len(accepted_tokens)

      if eos_ids.intersection(accepted_tokens):
        break

    if total_draft_tokens > 0:
      acceptance_rate = total_accepted_tokens / total_draft_tokens
    else:
      acceptance_rate = 0.0
    return SpeculativeGenerationOutput(
        sequences=generated,
        tokens_generated=tokens_produced,
        num_draft_tokens=total_draft_tokens,
        num_accepted_tokens=total_accepted_tokens,
        acceptance_rate=acceptance_rate,
        token_origins=token_origins,
        num_steps=num_steps,
    )
