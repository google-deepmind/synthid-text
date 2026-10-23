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

import warnings

from absl.testing import absltest
from absl.testing import parameterized
import mock
import torch
import transformers

from synthid_text import logits_processing
from synthid_text import synthid_mixin


_TINY_CONFIG = transformers.GPT2Config(
    vocab_size=50, n_positions=64, n_embd=16, n_layer=1, n_head=2
)


def _tiny_model() -> synthid_mixin.SynthIDGPT2LMHeadModel:
  torch.manual_seed(0)
  return synthid_mixin.SynthIDGPT2LMHeadModel(_TINY_CONFIG).eval()


class SynthidMixinTest(parameterized.TestCase):

  def _generate_and_record(self, model, **generate_kwargs):
    """Runs generate() recording raw logits and watermark inputs/outputs."""
    raw_logits = []
    old_forward = model.forward

    def recording_forward(*args, **kwargs):
      out = old_forward(*args, **kwargs)
      raw_logits.append(out.logits[:, -1, :].to(torch.float32).clone())
      return out

    wm_inputs, wm_outputs = [], []
    old_call = logits_processing.SynthIDLogitsProcessor.__call__

    def recording_call(self_, input_ids, scores):
      wm_inputs.append(scores.clone())
      out = old_call(self_, input_ids, scores)
      wm_outputs.append(out.clone())
      return out

    with mock.patch.object(
        logits_processing.SynthIDLogitsProcessor,
        "__call__",
        autospec=True,
        side_effect=recording_call,
    ), mock.patch.object(model, "forward", side_effect=recording_forward):
      sequences = model.generate(
          input_ids=torch.ones(2, 6, dtype=torch.long),
          attention_mask=torch.ones(2, 6, dtype=torch.long),
          do_sample=True,
          pad_token_id=0,
          **generate_kwargs,
      )
    return sequences, raw_logits, wm_inputs, wm_outputs

  def test_watermark_is_applied_once_after_all_warpers(self):
    """The watermark must see exactly the scores HF would sample from."""
    model = _tiny_model()
    temperature, top_k, top_p, new_tokens = 0.5, 10, 0.9, 4
    _, raw_logits, wm_inputs, _ = self._generate_and_record(
        model,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=new_tokens,
    )

    expected_warpers = transformers.LogitsProcessorList([
        transformers.TemperatureLogitsWarper(temperature),
        transformers.TopKLogitsWarper(top_k),
        transformers.TopPLogitsWarper(top_p),
    ])
    self.assertLen(wm_inputs, new_tokens)
    self.assertLen(raw_logits, new_tokens)
    for raw, got in zip(raw_logits, wm_inputs):
      want = expected_warpers(None, raw)
      torch.testing.assert_close(got, want)

  def test_watermark_preserves_truncation_and_changes_scores(self):
    model = _tiny_model()
    ngram_len = synthid_mixin.DEFAULT_WATERMARKING_CONFIG["ngram_len"]
    new_tokens = ngram_len + 5
    sequences, _, wm_inputs, wm_outputs = self._generate_and_record(
        model,
        temperature=0.7,
        top_k=10,
        top_p=0.8,
        max_new_tokens=new_tokens,
    )
    self.assertLen(wm_inputs, new_tokens)
    generated = sequences[:, -new_tokens:]
    any_changed = False
    for step, (inp, out) in enumerate(zip(wm_inputs, wm_outputs)):
      # Tokens removed by top-k/top-p stay removed.
      torch.testing.assert_close(torch.isneginf(out), torch.isneginf(inp))
      # Sampled tokens always come from the surviving set.
      sampled_scores = torch.gather(out, 1, generated[:, step : step + 1])
      self.assertTrue(torch.isfinite(sampled_scores).all())
      if step < ngram_len - 1:
        # skip_first_ngram_calls=True: the first ngram_len - 1 steps are not
        # watermarked.
        torch.testing.assert_close(out, inp)
        continue
      finite = torch.isfinite(inp)
      inp_logp = torch.log_softmax(inp, dim=1)[finite]
      out_logp = torch.log_softmax(out, dim=1)[finite]
      if not torch.allclose(inp_logp, out_logp, atol=1e-5):
        any_changed = True
    self.assertTrue(any_changed, "Watermarking never modified the scores.")

  def test_generate_requires_top_k(self):
    model = _tiny_model()
    with self.assertRaisesRegex(ValueError, "top_k"):
      model.generate(
          input_ids=torch.ones(1, 6, dtype=torch.long),
          do_sample=True,
          top_k=0,
          max_new_tokens=2,
          pad_token_id=0,
      )

  def test_generate_requires_sampling(self):
    model = _tiny_model()
    with self.assertRaisesRegex(ValueError, "do_sample"):
      model.generate(
          input_ids=torch.ones(1, 6, dtype=torch.long),
          do_sample=False,
          max_new_tokens=2,
          pad_token_id=0,
      )

  # DoLa and contrastive search are Hub repos in Transformers v5, which
  # raises unless trust_remote_code=True; the mixin must then reject them
  # before anything is loaded. The mode name's case depends on Python.
  @parameterized.parameters(
      (
          dict(assistant_model=transformers.GPT2LMHeadModel(_TINY_CONFIG)),
          "assisted generation",
      ),
      (dict(prompt_lookup_num_tokens=3), "assisted generation"),
      (dict(use_mtp=True), "assisted generation"),
      (dict(num_beams=2), "num_beams=1"),
      (
          dict(dola_layers="low", trust_remote_code=True),
          "(?i)plain sampling.*dola_generation",
      ),
      (
          dict(do_sample=False, penalty_alpha=0.6, trust_remote_code=True),
          "(?i)plain sampling.*contrastive_search",
      ),
      (dict(custom_generate=lambda **kw: None), "custom_generate"),
      (dict(cache_implementation="paged"), "continuous batching"),
      (
          dict(watermarking_config=transformers.WatermarkingConfig()),
          "watermarking_config",
      ),
  )
  def test_unsupported_generation_paths_raise(self, kwargs, regex):
    model = _tiny_model()
    generate_kwargs = dict(
        input_ids=torch.ones(1, 6, dtype=torch.long),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        do_sample=True,
        top_k=5,
        max_new_tokens=4,
        pad_token_id=0,
    )
    generate_kwargs.update(kwargs)
    with self.assertRaisesRegex(ValueError, regex):
      model.generate(**generate_kwargs)

  def test_generation_and_detection_agree(self):
    """Detection recomputes the generation-time contexts and repetition mask."""
    config = transformers.GPT2Config(
        vocab_size=50, n_positions=128, n_embd=16, n_layer=1, n_head=2
    )
    torch.manual_seed(0)
    model = synthid_mixin.SynthIDGPT2LMHeadModel(config).eval()
    unwatermarked_model = transformers.GPT2LMHeadModel(config).eval()
    unwatermarked_model.load_state_dict(model.state_dict())

    ngram_len = synthid_mixin.DEFAULT_WATERMARKING_CONFIG["ngram_len"]
    prompt_len, new_tokens = 6, 80
    prompts = torch.tensor(
        [[7, 8, 9, 10, 11, 12], [3, 3, 5, 6, 7, 8], [3, 3, 3, 2, 4, 5]]
    )
    # Rows 1 and 2 are left-padded.
    attention_mask = torch.tensor(
        [[1] * 6, [0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    )
    generate_kwargs = dict(
        input_ids=prompts,
        attention_mask=attention_mask,
        do_sample=True,
        top_k=10,
        temperature=1.0,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        pad_token_id=3,
    )

    # Per call: (context used for hashing, whether it was treated as repeated,
    # context history before the call).
    calls = []
    original = logits_processing.SynthIDLogitsProcessor.watermark_candidates

    def recording_watermark_candidates(self_, input_ids, scores, indices):
      history = (
          None if self_.state is None else self_.state.context_history.clone()
      )
      out = original(self_, input_ids, scores, indices)
      state = self_.state
      assert state is not None
      if history is None:
        history = torch.zeros_like(state.context_history)
      packed = state.context_history[:, :1]
      calls.append((
          state.context.clone(),
          (history == packed).any(dim=1),
          history,
      ))
      return out

    with mock.patch.object(
        logits_processing.SynthIDLogitsProcessor,
        "watermark_candidates",
        autospec=True,
        side_effect=recording_watermark_candidates,
    ):
      torch.manual_seed(1)
      sequences = model.generate(**generate_kwargs)
    generated = sequences[:, prompt_len:]
    self.assertLen(calls, new_tokens)

    # With skip_first_ngram_calls=True the first ngram_len - 1 calls neither
    # watermark nor record their context (their "repeated" entry above is
    # meaningless), so the history is still empty at the first watermarked
    # call, as it is at the start of detection.
    self.assertFalse(calls[ngram_len - 1][2].any())

    config = dict(synthid_mixin.DEFAULT_WATERMARKING_CONFIG)
    config["device"] = generated.device
    detector = logits_processing.SynthIDLogitsProcessor(
        **config  # pyrefly: ignore[bad-argument-type]
    )
    mask = detector.compute_context_repetition_mask(generated)
    num_scored = new_tokens - (ngram_len - 1)
    self.assertEqual(tuple(mask.shape), (len(prompts), num_scored))
    for j in range(num_scored):
      context, repeated, _ = calls[j + ngram_len - 1]
      torch.testing.assert_close(context, generated[:, j : j + ngram_len - 1])
      torch.testing.assert_close(mask[:, j], ~repeated)

    def mean_g(tokens):
      g_values = detector.compute_g_values(tokens)
      m = detector.compute_context_repetition_mask(tokens)[..., None]
      return float((g_values * m).sum() / (m.sum() * g_values.shape[-1]))

    torch.manual_seed(1)
    unwatermarked = unwatermarked_model.generate(**generate_kwargs)[
        :, prompt_len:
    ]
    watermarked_mean = mean_g(generated)
    unwatermarked_mean = mean_g(unwatermarked)
    self.assertGreater(watermarked_mean, 0.55)
    self.assertBetween(unwatermarked_mean, 0.46, 0.54)

  def test_generate_ignores_config_device(self):
    """generate() does not use the device in DEFAULT_WATERMARKING_CONFIG."""
    model = _tiny_model().to("cpu")
    # Simulate DEFAULT_WATERMARKING_CONFIG having a different device
    # (e.g. cuda:0).
    fake_config = dict(synthid_mixin.DEFAULT_WATERMARKING_CONFIG)
    fake_config["device"] = torch.device("cuda:0")

    with mock.patch.object(
        synthid_mixin, "DEFAULT_WATERMARKING_CONFIG", fake_config
    ):
      # Must not raise device mismatch ValueError.
      out = model.generate(
          input_ids=torch.ones(1, 6, dtype=torch.long),
          attention_mask=torch.ones(1, 6, dtype=torch.long),
          do_sample=True,
          top_k=5,
          max_new_tokens=2,
          pad_token_id=0,
      )
    self.assertEqual(out.device.type, "cpu")

  def test_processor_device(self):
    """The processor uses the `device` argument, else the model's device."""
    model = _tiny_model()
    config = transformers.GenerationConfig(do_sample=True, top_k=5)
    passed = torch.device("cuda:3")
    for kwargs, want in [(dict(device=passed), passed), ({}, model.device)]:
      # Mocked so that no tensors are created on `passed`.
      with mock.patch.object(
          logits_processing, "SynthIDLogitsProcessor", autospec=True
      ) as processor_cls:
        model._get_logits_processor(config, **kwargs)
      self.assertEqual(processor_cls.call_args.kwargs["device"], want)

  def test_continuous_batching_raises(self):
    model = _tiny_model()
    with self.assertRaisesRegex(ValueError, "continuous batching"):
      model.init_continuous_batching()

  def test_watermark_precedes_logit_normalization(self):
    """With renormalize_logits=True the watermark runs before normalization."""
    model = _tiny_model()
    temperature, top_k, new_tokens = 0.5, 10, 5
    _, raw_logits, wm_inputs, _ = self._generate_and_record(
        model,
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=new_tokens,
        renormalize_logits=True,
    )
    expected_warpers = transformers.LogitsProcessorList([
        transformers.TemperatureLogitsWarper(temperature),
        transformers.TopKLogitsWarper(top_k),
    ])
    self.assertLen(wm_inputs, new_tokens)
    for raw, wm_input in zip(raw_logits, wm_inputs):
      want = expected_warpers(None, raw)
      # The watermark sees unnormalized scores, so it precedes normalization.
      torch.testing.assert_close(wm_input, want)

  def test_output_scores_warns_that_scores_are_watermarked(self):
    model = _tiny_model()
    # assertWarns inspects every loaded module, which triggers transformers'
    # lazy imports of optional dependencies; record the warnings instead.
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      out = model.generate(  # pyrefly: ignore[bad-argument-type]
          input_ids=torch.ones(1, 6, dtype=torch.long),
          attention_mask=torch.ones(1, 6, dtype=torch.long),
          do_sample=True,
          top_k=5,
          max_new_tokens=2,
          pad_token_id=0,
          return_dict_in_generate=True,
          output_scores=True,
      )
    messages = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    self.assertTrue(any("output_logits=True" in m for m in messages), messages)
    self.assertLen(out.scores, 2)

  def test_num_return_sequences(self):
    model = _tiny_model()
    sequences, _, wm_inputs, _ = self._generate_and_record(
        model, top_k=5, max_new_tokens=6, num_return_sequences=3
    )
    # _generate_and_record uses 2 prompts of length 6.
    self.assertEqual(tuple(sequences.shape), (6, 12))
    self.assertLen(wm_inputs, 6)
    for scores in wm_inputs:
      self.assertEqual(scores.shape[0], 6)

  def test_gemma_generate(self):
    sizes = dict(
        vocab_size=50,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    for name, make_model in (
        (
            "gemma",
            lambda: synthid_mixin.SynthIDGemmaForCausalLM(
                transformers.GemmaConfig(**sizes)
            ),
        ),
        (
            "gemma3",
            # A window shorter than the sequence exercises the sliding cache.
            lambda: synthid_mixin.SynthIDGemma3ForCausalLM(
                transformers.Gemma3TextConfig(**sizes, sliding_window=4)
            ),
        ),
    ):
      with self.subTest(name):
        torch.manual_seed(0)
        model = make_model().eval()
        sequences, _, wm_inputs, _ = self._generate_and_record(
            model, top_k=5, max_new_tokens=6, min_new_tokens=6
        )
        self.assertEqual(tuple(sequences.shape), (2, 12))
        self.assertLen(wm_inputs, 6)


if __name__ == "__main__":
  absltest.main()
