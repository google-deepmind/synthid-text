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

"""Validation cross entropy must weight examples, not equally weight batches."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from synthid_text import detector_bayesian as detector


def run_training(batch_size, val_g, val_mask, val_y, learning_rate=0.0):
  model = detector.BayesianDetectorModule(watermarking_depth=2, baserate=0.8)
  train_g = jnp.zeros((4, 3, 2))
  history, minimum = detector.train(
      detector_module=model,
      g_values=train_g,
      mask=jnp.ones((4, 3)),
      watermarked=jnp.array([1.0, 1.0, 0.0, 0.0]),
      epochs=2,
      learning_rate=learning_rate,
      minibatch_size=batch_size,
      shuffle=False,
      g_values_val=val_g,
      mask_val=val_mask,
      watermarked_val=val_y,
      validation_metric=detector.ValidationMetric.CROSS_ENTROPY,
  )
  return model, history, minimum


class ValidationBatchWeightingTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 4, 5, 8)
  def test_reported_cross_entropy_matches_all_validation_examples(
      self, batch_size
  ):
    g = jnp.zeros((5, 3, 2))
    mask = jnp.ones((5, 3))
    labels = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0])
    model, history, minimum = run_training(batch_size, g, mask, labels)
    expected_losses = []
    for epoch in history.values():
      scores = model.apply({"params": epoch["params"]}, g, mask)
      expected = detector.xentropy_loss(labels, scores)
      np.testing.assert_allclose(epoch["val_loss"], expected, rtol=2e-6)
      expected_losses.append(float(expected))
    np.testing.assert_allclose(minimum, min(expected_losses), rtol=2e-6)

  def test_reordering_validation_examples_cannot_reweight_the_last_example(
      self,
  ):
    g = jnp.zeros((5, 3, 2))
    mask = jnp.ones((5, 3))
    labels = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0])
    _, first, minimum1 = run_training(4, g, mask, labels)
    _, second, minimum2 = run_training(4, g[::-1], mask[::-1], labels[::-1])
    np.testing.assert_allclose(minimum1, minimum2, rtol=2e-6)
    for epoch in first:
      np.testing.assert_allclose(
          first[epoch]["val_loss"], second[epoch]["val_loss"], rtol=2e-6
      )

  def test_real_parameter_updates_and_selected_snapshot_match_full_validation(
      self,
  ):
    g = jnp.array(
        [[[i % 2, (i // 2) % 2]] * 3 for i in range(5)], dtype=jnp.float32
    )
    mask = jnp.ones((5, 3)).at[-1, -1].set(0.0)
    labels = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])
    model, history, minimum = run_training(
        4, g, mask, labels, learning_rate=0.01
    )
    losses = []
    for epoch in history.values():
      expected = detector.xentropy_loss(
          labels, model.apply({"params": epoch["params"]}, g, mask)
      )
      losses.append(float(expected))
      np.testing.assert_allclose(epoch["val_loss"], expected, rtol=2e-6)
    np.testing.assert_allclose(minimum, min(losses), rtol=2e-6)
    selected = detector.xentropy_loss(
        labels, model.apply(model.params, g, mask)
    )
    np.testing.assert_allclose(selected, min(losses), rtol=2e-6)


if __name__ == "__main__":
  absltest.main()
