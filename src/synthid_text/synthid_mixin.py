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

"""SynthID watermarked mixin class."""

import inspect
import warnings

import immutabledict
import torch
import transformers

from synthid_text import logits_processing


# Watermarking config used by the mixin for all watermarked generation. It is
# for demonstration only: key 654 is a public example key, not a secret.
DEFAULT_WATERMARKING_CONFIG = immutabledict.immutabledict({
    "ngram_len": 5,  # This corresponds to H=4 context window size in the paper.
    "key": 654,
    "depth": 30,
    "context_history_size": 1024,
    # Device used when this config is passed directly to a
    # SynthIDLogitsProcessor, e.g. for detection. Watermarked generation
    # instead uses the device that `generate()` passes to
    # `_get_logits_processor`.
    "device": (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    ),
    # The first ngram_len - 1 generated tokens are not watermarked. Their
    # contexts are partly the all-zero initial context rather than generated
    # tokens, and detection, which only sees the generated tokens, never
    # scores them.
    "skip_first_ngram_calls": True,
})


class SynthIDSparseTopKMixin(transformers.GenerationMixin):
  """Mixin class of transformers library with watermarking enabled.

  Watermarking is applied by appending a SynthIDLogitsProcessor after all of
  the logits processors and sampling warpers that `generate()` builds
  (repetition penalty, temperature, top-k, top-p, ...). The watermark is
  therefore applied to exactly the distribution that would otherwise be
  sampled from, and all logits processors and sampling warpers are supported.

  g-values are only computed for tokens with finite scores after processing,
  so `top_k` should be set to keep the per-step cost bounded.

  With `return_dict_in_generate=True, output_scores=True`, the returned `scores`
  are the processed scores *after* watermarking, so they reflect the watermarked
  distribution rather than the model's. A warning is issued in that case. Use
  `output_logits=True` for the model's raw logits, e.g. to compute perplexity.

  Only plain sampling (`do_sample=True`, `num_beams=1`) through `generate()`
  produces correctly watermarked text.
  """

  def _get_logits_processor(  # pyrefly: ignore[bad-override]
      self,
      generation_config: transformers.GenerationConfig,
      *args,
      **kwargs,
  ) -> transformers.LogitsProcessorList:
    """Returns the default processors with the watermarking processor added.

    Args:
      generation_config: The generation config of the current generate() call.
      *args: Positional arguments passed to the parent implementation.
      **kwargs: Keyword arguments passed to the parent implementation. The
        watermarking processor is created on `device` if it is given
        (positionally or as a keyword), and on `self.device` otherwise.

    Returns:
      The default processors with the watermarking processor added last, or
      just before a trailing LogitNormalization.
    """
    # 1. Build the base logits processors and warpers (temperature, top-k,
    # etc.).
    parent = super()._get_logits_processor
    bound_args = inspect.signature(parent).bind(
        generation_config, *args, **kwargs
    )
    processors = parent(  # pyrefly: ignore[bad-argument-type]
        generation_config, *args, **kwargs
    )
    if (
        generation_config.return_dict_in_generate
        and generation_config.output_scores
    ):
      warnings.warn(
          "The `scores` returned by watermarked generation include the SynthID"
          " watermark. Use `output_logits=True` for the model's raw logits,"
          " e.g. to compute perplexity."
      )

    # 2. Create a fresh processor instance per generate() call to maintain
    # isolated per-sequence state (context window and repetition history).
    # generate() passes the device of the input tensor.
    device = bound_args.arguments.get("device")
    if device is None:
      device = self.device  # pyrefly: ignore[missing-attribute]
    config = dict(DEFAULT_WATERMARKING_CONFIG)
    config["device"] = device
    watermarking_processor = logits_processing.SynthIDLogitsProcessor(
        **config,  # pyrefly: ignore[bad-argument-type]
    )

    # 3. Insert the watermarking processor after all sampling warpers (so only
    # finite top-k candidates are watermarked), but before LogitNormalization,
    # which transformers requires to remain the final processor in the chain.
    if processors and isinstance(
        processors[-1], transformers.LogitNormalization
    ):
      processors.insert(len(processors) - 1, watermarking_processor)
    else:
      processors.append(watermarking_processor)
    return processors


class SynthIDGPT2LMHeadModel(
    SynthIDSparseTopKMixin, transformers.GPT2LMHeadModel
):
  """GPT-2 language model with SynthID watermarked generation."""


class SynthIDGemmaForCausalLM(
    SynthIDSparseTopKMixin, transformers.GemmaForCausalLM
):
  """Gemma language model with SynthID watermarked generation."""


class SynthIDGemma3ForCausalLM(
    SynthIDSparseTopKMixin, transformers.Gemma3ForCausalLM
):
  """Text-only Gemma 3 language model with SynthID watermarked generation."""
