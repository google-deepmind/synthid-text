# Copyright 2026 DeepMind Technologies Limited
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

from absl.testing import absltest
import mock
import torch

from synthid_text import detector_bayesian


class BayesianDetectorDeviceTest(absltest.TestCase):

  def _train_with_device(self, device):
    processed = (
        mock.sentinel.train_g_values,
        mock.sentinel.train_masks,
        mock.sentinel.train_labels,
        mock.sentinel.cv_g_values,
        mock.sentinel.cv_masks,
        mock.sentinel.cv_labels,
    )
    expected = (mock.sentinel.detector, 0.25)
    with mock.patch.object(
        detector_bayesian.BayesianDetector,
        "process_raw_model_outputs",
        return_value=processed,
    ) as process_mock, mock.patch.object(
        detector_bayesian.BayesianDetector,
        "train_best_detector_given_g_values",
        return_value=expected,
    ) as train_mock:
      result = detector_bayesian.BayesianDetector.train_best_detector(
          tokenized_wm_outputs=[],
          tokenized_uwm_outputs=[],
          logits_processor=mock.sentinel.logits_processor,
          tokenizer=mock.sentinel.tokenizer,
          torch_device=device,
      )
    return result, process_mock, train_mock

  def test_cuda_device_is_allowed(self):
    result, process_mock, train_mock = self._train_with_device(
        torch.device("cuda")
    )

    self.assertEqual(result, (mock.sentinel.detector, 0.25))
    process_mock.assert_called_once()
    train_mock.assert_called_once()

  def test_cpu_device_is_rejected_before_processing(self):
    with mock.patch.object(
        detector_bayesian.BayesianDetector, "process_raw_model_outputs"
    ) as process_mock:
      with self.assertRaisesRegex(ValueError, "training unstable on CPUs"):
        detector_bayesian.BayesianDetector.train_best_detector(
            tokenized_wm_outputs=[],
            tokenized_uwm_outputs=[],
            logits_processor=mock.sentinel.logits_processor,
            tokenizer=mock.sentinel.tokenizer,
            torch_device=torch.device("cpu"),
        )

    process_mock.assert_not_called()


if __name__ == "__main__":
  absltest.main()
