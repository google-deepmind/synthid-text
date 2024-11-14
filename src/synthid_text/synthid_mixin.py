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

"""SynthID watermarked mixin class."""

from collections.abc import Mapping
from typing import Any, Optional, Union

import immutabledict
import torch
import transformers

from synthid_text import logits_processing


DEFAULT_WATERMARKING_CONFIG = immutabledict.immutabledict({
    "ngram_len": 5,  # This corresponds to H=4 context window size in the paper.
    "keys": [
        654,
        400,
        836,
        123,
        340,
        443,
        597,
        160,
        57,
        29,
        590,
        639,
        13,
        715,
        468,
        990,
        966,
        226,
        324,
        585,
        118,
        504,
        421,
        521,
        129,
        669,
        732,
        225,
        90,
        960,
    ],
    "sampling_table_size": 2**16,
    "sampling_table_seed": 0,
    "context_history_size": 1024,
    "device": (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    ),
})


class SynthIDSparseTopKMixin(transformers.GenerationMixin):
  """Mixin class of transformers library with watermarking enabled."""

  def _construct_warper_list(
      self, extra_params: Mapping[str, Any]
  ) -> transformers.LogitsProcessorList:
    """Instantiate warpers list."""
    warpers = transformers.LogitsProcessorList()
    warpers.append(
        logits_processing.SynthIDLogitsProcessor(
            **DEFAULT_WATERMARKING_CONFIG, **extra_params
        )
    )
    return warpers

  def _get_logits_warper(
      self,
      generation_config: transformers.GenerationConfig,
      **unused_kw,
  ) -> transformers.LogitsProcessorList:
    """Constructs and returns a list of warpers.

    This overrides the base class's implementation to control how we apply top_k
    and temperature. Only the SynthIDLogitsProcessor warper is constructed that
    performs top_k and temperature scaling before applying watermark. This is
    to improve the latency impact by watermarking by only considering the top_k
    indices for watermarking.

    Args:
     generation_config: Config used for generation with this model.

    Returns:
     List of logits processors to be applied at inference time.
    """
    extra_params = {}

    # Add temperature to extra params
    if not (
        generation_config.temperature is not None
        and 0.0 <= generation_config.temperature <= 1.0
    ):
      raise ValueError(
          f"Invalid temperature {generation_config.temperature} when sampling"
          " with watermarking. Temperature should be between 0.0 and 1.0."
      )
    extra_params["temperature"] = generation_config.temperature

    # Add top_k to extra params.
    if not (
        generation_config.top_k is not None and generation_config.top_k >= 1
    ):
      raise ValueError(
          f"Invalid top_k {generation_config.top_k} when sampling with"
          " watermarking. Top_k should >= 1."
      )
    extra_params["top_k"] = generation_config.top_k

    return self._construct_warper_list(extra_params)

  def _sample(
      self,
      input_ids: torch.LongTensor,
      logits_processor: transformers.LogitsProcessorList,
      stopping_criteria: transformers.StoppingCriteriaList,
      generation_config: transformers.GenerationConfig,
      synced_gpus: bool,
      streamer: Optional["transformers.BaseStreamer"],
      logits_warper: Optional[transformers.LogitsProcessorList] = None,
      **model_kwargs,
  ) -> Union[
      transformers.generation.utils.GenerateNonBeamOutput, torch.LongTensor
  ]:
    r"""Sample sequence of tokens.

    Generates sequences of token ids for models with a language modeling head
    using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and
    vision-to-text models.

    This function is copied and changed minimally from the HuggingFace
    repository to support watermarking implementation.

    This overrides the base class implementation to achieve watermarking of the
    logits before they are sampled. This is done specifically so as to preserve
    the top_k indices separately without making the logits dense with all the
    indices. This removes extra overhead of considering all possible indices for
    watermarking.

    Args:
        input_ids: The sequence used as a prompt for the generation.
        logits_processor: List of instances of class derived from
          [`LogitsProcessor`] used to modify the prediction scores of the
          language modeling head applied at each generation step.
        stopping_criteria: An instance of [`StoppingCriteriaList`]. List of
          instances of class derived from [`StoppingCriteria`] used to tell if
          the generation loop should stop.
        generation_config: The generation configuration to be used as
          parametrization of the decoding method.
        synced_gpus: Whether to continue running the while loop until max_length
          (needed for ZeRO stage 3)
        streamer: Streamer object that will be used to stream the generated
          sequences. Generated tokens are passed through
          `streamer.put(token_ids)` and the streamer is responsible for any
          further processing.
        logits_warper: List of instances of class derived from [`LogitsWarper`]
          used to warp the prediction score distribution of the language
          modeling head applied before multinomial sampling at each generation
          step. Only required with sampling strategies (i.e. `do_sample` is set
          in `generation_config`)
        **model_kwargs: Additional model specific kwargs will be forwarded to
          the `forward` function of the model. If model is an encoder-decoder
          model the kwargs should include `encoder_outputs`.

    Returns:
        A `torch.LongTensor` containing the generated tokens (default behaviour)
        or a
        [`~generation.GenerateDecoderOnlyOutput`] if
        `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a
        [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config.pad_token_id
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample
    if do_sample and not isinstance(
        logits_warper, transformers.LogitsProcessorList
    ):
      raise ValueError(
          "`do_sample` is set to `True`, `logits_warper` must be a"
          f" `LogitsProcessorList` instance (it is {logits_warper})."
      )
    if has_eos_stopping_criteria and pad_token_id is None:
      raise ValueError(
          "`stopping_criteria` is not empty, `pad_token_id` must be set in "
          "`generation_config`. See "
          "https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig"
          "for more on how to configure the `pad_token_id`."
      )
    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = (
        () if (return_dict_in_generate and output_attentions) else None
    )
    cross_attentions = (
        () if (return_dict_in_generate and output_attentions) else None
    )
    decoder_hidden_states = (
        () if (return_dict_in_generate and output_hidden_states) else None
    )

    # if model is an encoder-decoder, retrieve encoder attention weights and
    # hidden states
    encoder_attentions = None
    encoder_hidden_states = None
    if return_dict_in_generate and self.config.is_encoder_decoder:  # pytype: disable=attribute-error
      encoder_attentions = (
          model_kwargs["encoder_outputs"].get("attentions")
          if output_attentions
          else None
      )
      encoder_hidden_states = (
          model_kwargs["encoder_outputs"].get("hidden_states")
          if output_hidden_states
          else None
      )

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(
        batch_size, dtype=torch.long, device=input_ids.device
    )
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)  # pytype: disable=attribute-error

    while self._has_unfinished_sequences(  # pytype: disable=attribute-error
        this_peer_finished, synced_gpus, device=input_ids.device
    ):
      # prepare model inputs
      model_inputs = self.prepare_inputs_for_generation(  # pytype: disable=attribute-error
          input_ids, **model_kwargs
      )

      # forward pass to get next token
      outputs = self(  # pytype: disable=not-callable
          **model_inputs,
          return_dict=True,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
      )

      if synced_gpus and this_peer_finished:
        continue  # don't waste resources running the code we don't need

      # Clone is needed to avoid keeping a hanging ref to outputs.logits which
      # may be very large for first iteration (the clone itself is always small)
      next_token_logits = outputs.logits[:, -1, :].clone()

      # pre-process distribution
      next_token_scores = logits_processor(input_ids, next_token_logits)
      indices_mapping = None
      unwatermarked_scores = None
      if do_sample:
        *regular_warpers, watermarking_logits_warper = logits_warper
        if not isinstance(
            watermarking_logits_warper,
            logits_processing.SynthIDLogitsProcessor,
        ):
          raise ValueError(
              "SynthIDLogitsProcessor should be the final warper in the list"
              " while watermarking."
          )
        for logit_warper in regular_warpers:
          next_token_scores = logit_warper(input_ids, next_token_scores)
        # Watermark final scores with sparse top_k.
        next_token_scores, indices_mapping, unwatermarked_scores = (
            watermarking_logits_warper.watermarked_call(
                input_ids, next_token_scores
            )
        )

      # token selection
      if do_sample:
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
      else:
        next_tokens = torch.argmax(next_token_scores, dim=-1)

      # Store scores, attentions and hidden_states when required
      if return_dict_in_generate:
        if output_scores:
          assert unwatermarked_scores is not None
          score = torch.gather(
              -torch.log(torch.nn.Softmax(dim=1)(unwatermarked_scores)),
              1,
              next_tokens[:, None],
          )
          scores += (score,)
        if output_logits:
          raw_logits += (next_token_logits,)
        if output_attentions:
          decoder_attentions += (
              (outputs.decoder_attentions,)
              if self.config.is_encoder_decoder  # pytype: disable=attribute-error
              else (outputs.attentions,)
          )
          if self.config.is_encoder_decoder:  # pytype: disable=attribute-error
            cross_attentions += (outputs.cross_attentions,)

        if output_hidden_states:
          decoder_hidden_states += (
              (outputs.decoder_hidden_states,)
              if self.config.is_encoder_decoder  # pytype: disable=attribute-error
              else (outputs.hidden_states,)
          )

      assert indices_mapping is not None
      # re-mapping to dense indices with indices_mapping
      next_tokens = torch.vmap(torch.take, in_dims=0, out_dims=0)(
          indices_mapping, next_tokens
      )

      # finished sentences should have their next token be a padding token
      if has_eos_stopping_criteria:
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )

      # update generated ids, model inputs, and length for next step
      input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
      if streamer is not None:
        streamer.put(next_tokens.cpu())
      model_kwargs = self._update_model_kwargs_for_generation(  # pytype: disable=attribute-error
          outputs,
          model_kwargs,
          is_encoder_decoder=self.config.is_encoder_decoder,  # pytype: disable=attribute-error
      )

      unfinished_sequences = unfinished_sequences & ~stopping_criteria(
          input_ids, scores
      )
      this_peer_finished = unfinished_sequences.max() == 0

      # This is needed to properly delete outputs.logits which may be very large
      # for first iteration. Otherwise a reference to outputs is kept which
      # keeps the logits alive in the next iteration
      del outputs

    if streamer is not None:
      streamer.end()

    if return_dict_in_generate:
      if self.config.is_encoder_decoder:  # pytype: disable=attribute-error
        return transformers.generation.utils.GenerateEncoderDecoderOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            encoder_attentions=encoder_attentions,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            decoder_hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
      else:
        return transformers.generation.utils.GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
    else:
      return input_ids


class SynthIDGPT2LMHeadModel(
    SynthIDSparseTopKMixin, transformers.GPT2LMHeadModel
):
  pass


class SynthIDGemmaForCausalLM(
    SynthIDSparseTopKMixin, transformers.GemmaForCausalLM
):
  pass
