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

import dataclasses

from absl.testing import absltest
import mock
import torch
import transformers
from transformers import utils as transformers_utils

from synthid_text import synthid_mixin
from synthid_text import logits_processing


@dataclasses.dataclass(frozen=True)
class Config:

  is_encoder_decoder: bool = True


class MockSynthIDModel(synthid_mixin.SynthIDSparseTopKMixin):

  def __init__(self, config: Config):
    super().__init__()
    self.config = config

  def _get_initial_cache_position(self, input_ids, model_kwargs):
    model_kwargs["cache_position"] = torch.zeros(input_ids.shape[1])
    del input_ids
    return model_kwargs

  def prepare_inputs_for_generation(self, input_ids, **kwargs):
    return {
        "x": input_ids,
        **kwargs,
    }

  def __call__(
      self,
      x: torch.Tensor,
      **kwargs,
  ) -> transformers_utils.ModelOutput:
    return transformers_utils.ModelOutput(
        logits=torch.ones(x.shape[0], 5, 7),
    )


class SynthidMixinTest(absltest.TestCase):

  def test_sampling_from_mixin_includes_watermarking(self):
    old_watermarked_call = (
        logits_processing.SynthIDLogitsProcessor.watermarked_call
    )
    with mock.patch.object(
        logits_processing.SynthIDLogitsProcessor,
        "watermarked_call",
        autospec=True,
    ) as mock_watermarked_call:
      mock_watermarked_call.side_effect = old_watermarked_call
      synthid_model = MockSynthIDModel(config=Config())
      generation_config = transformers.GenerationConfig(
          top_k=5,
          temperature=0.5,
          do_sample=True,
      )

      logits_warper = synthid_model._get_logits_warper(generation_config)

      synthid_model._sample(
          input_ids=torch.ones(3, 11, dtype=torch.long),
          logits_processor=transformers.LogitsProcessorList(),
          stopping_criteria=transformers.StoppingCriteriaList(
              [lambda *_: True]
          ),
          generation_config=generation_config,
          synced_gpus=False,
          streamer=None,
          logits_warper=logits_warper,
      )

      self.assertEqual(mock_watermarked_call.call_count, 1)


if __name__ == "__main__":
  absltest.main()
