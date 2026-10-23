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

"""Bayesian detector class."""

import abc
from collections.abc import Mapping, Sequence
import enum
import functools
import gc
import logging
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import numpy as np
import optax
from sklearn import model_selection
import torch
import tqdm

from synthid_text import logits_processing


def pad_to_len(
    arr: torch.Tensor,
    target_len: int,
    *,
    left_pad: bool,
    pad_value: int,
    device: torch.device,
) -> torch.Tensor:
  """Pad or truncate array to given length."""
  if arr.shape[1] < target_len:
    shape_for_ones = list(arr.shape)
    shape_for_ones[1] = target_len - shape_for_ones[1]
    padded = (
        torch.ones(
            shape_for_ones,
            device=device,
            dtype=torch.long,
        )
        * pad_value
    )
    if not left_pad:
      return torch.concatenate((arr, padded), dim=1)
    else:
      return torch.concatenate((padded, arr), dim=1)
  else:
    return arr[:, :target_len]


def filter_and_truncate(
    outputs: torch.Tensor,
    truncation_length: int | None,
    eos_token_mask: torch.Tensor,
) -> torch.Tensor:
  """Filter and truncate outputs to given length.

  Args:
   outputs: output tensor of shape [batch_size, output_len]
   truncation_length: Length to truncate the final output. If None, then no
     truncation is applied.
   eos_token_mask: EOS token mask of shape [batch_size, output_len]

  Returns:
   output tensor of shape [batch_size, truncation_length].
  """
  if truncation_length:
    outputs = outputs[:, :truncation_length]
    truncation_mask = torch.sum(eos_token_mask, dim=1) >= truncation_length
    return outputs[truncation_mask, :]
  return outputs


def process_outputs_for_training(
    all_outputs: Sequence[torch.Tensor],
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    eos_token_id: int | Sequence[int],
    *,
    truncation_length: int | None,
    max_length: int,
    torch_device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
  """Process raw model outputs into format understandable by the detector.

  Args:
   all_outputs: sequence of outputs of shape [batch_size, output_len].
   logits_processor: logits processor used for watermarking.
   eos_token_id: EOS token ID or sequence of EOS token IDs.
   truncation_length: Length to truncate the outputs. If None, then no
     truncation is applied.
   max_length: Length to pad truncated outputs so that all processed entries.
     have same shape.
   torch_device: torch device to use.

  Returns:
    Tuple of
      all_masks: list of masks of shape [batch_size, max_length].
      all_g_values: list of g_values of shape [batch_size, max_length, depth].
  """
  if truncation_length is not None and truncation_length > max_length:
    raise ValueError(
        f"max_length ({max_length}) must be at least the truncation length"
        f" ({truncation_length}); otherwise rows are silently truncated"
        " further."
    )
  pad_value = (
      eos_token_id[0] if isinstance(eos_token_id, Sequence) else eos_token_id
  )
  all_masks = []
  all_g_values = []
  num_total = 0
  num_kept = 0
  for outputs in tqdm.tqdm(all_outputs):
    # outputs is of shape (batch_size, output_len).
    # output_len can differ from batch to batch.
    num_total += outputs.shape[0]
    eos_token_mask = logits_processor.compute_eos_token_mask(
        input_ids=outputs,
        eos_token_id=eos_token_id,
    )
    outputs = filter_and_truncate(outputs, truncation_length, eos_token_mask)
    num_kept += outputs.shape[0]

    # If no filtered outputs skip this batch.
    if outputs.shape[0] == 0:
      continue

    # All outputs are padded to max-length with eos-tokens.
    outputs = pad_to_len(
        outputs,
        max_length,
        left_pad=False,
        pad_value=pad_value,
        device=torch_device,
    )
    # outputs shape (num_kept, max_length).

    eos_token_mask = logits_processor.compute_eos_token_mask(
        input_ids=outputs,
        eos_token_id=eos_token_id,
    )

    context_repetition_mask = logits_processor.compute_context_repetition_mask(
        input_ids=outputs,
    )

    # Pad on left to match max_length shape:
    # (num_kept, max_length - (ngram_len - 1)) -> (num_kept, max_length).
    context_repetition_mask = pad_to_len(
        context_repetition_mask,
        max_length,
        left_pad=True,
        pad_value=0,
        device=torch_device,
    )
    combined_mask = context_repetition_mask * eos_token_mask

    g_values = logits_processor.compute_g_values(
        input_ids=outputs,
    )

    # Pad on left to match max_length shape:
    # (num_kept, max_length - (ngram_len - 1), watermarking_depth) ->
    # (num_kept, max_length, watermarking_depth).
    g_values = pad_to_len(
        g_values, max_length, left_pad=True, pad_value=0, device=torch_device
    )
    all_masks.append(combined_mask)
    all_g_values.append(g_values)
  logging.info(
      "Processed outputs: kept %d of %d rows (truncation length %s).",
      num_kept,
      num_total,
      truncation_length,
  )
  return all_masks, all_g_values


class LikelihoodModel(abc.ABC):
  """Watermark likelihood model base class defining __call__ interface."""

  @abc.abstractmethod
  def __call__(self, g_values: jnp.ndarray) -> jnp.ndarray:
    """Computes likelihoods given g-values and a mask.

    Args:
      g_values: g-values (all are 0 or 1) of shape [batch_size, seq_len,
        watermarking_depth, ...].

    Returns:
      an array of shape [batch_size, seq_len, watermarking_depth] or
      [batch_size, seq_len, 1] corresponding to the likelihoods
      of the g-values given either the watermarked hypothesis or
      the unwatermarked hypothesis; i.e. either P(g|watermarked)
      or P(g|unwatermarked).
    """


class LikelihoodModelWatermarked(nn.Module, LikelihoodModel):
  """Watermarked likelihood model for binary-valued g-values.

  This takes in g-values and returns P(g_values|watermarked).
  """

  watermarking_depth: int

  def setup(self):
    """Initializes the model parameters."""
    self.beta = self.param(
        "beta",
        lambda rng: -2.5
        + 0.001 * jax.random.normal(rng, (1, 1, self.watermarking_depth)),
    )
    self.delta = self.param(
        "delta",
        lambda rng: 0.001
        * jax.random.normal(
            rng, (1, 1, self.watermarking_depth, self.watermarking_depth)
        ),
    )

  def l2_loss(self) -> jnp.ndarray:
    return jnp.einsum("ijkl->", self.delta**2)

  def _compute_latents(
      self, g_values: jnp.ndarray
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the unique token probability distribution given g-values.

    Args:
      g_values: Pseudorandom function values of shape [batch_size, seq_len,
        watermarking_depth].

    Returns:
      p_one_unique_token and p_two_unique_tokens, both of shape
        [batch_size, seq_len, watermarking_depth]. p_one_unique_token[i,t,l]
        gives the probability of there being one unique token in a tournament
        match on layer l, on timestep t, for batch item i.
        p_one_unique_token[i,t,l] + p_two_unique_token[i,t,l] = 1.
    """
    # Tile g-values to produce feature vectors for predicting the latents
    # for each layer in the tournament; our model for the latents psi is a
    # logistic regression model psi = sigmoid(delta * x + beta).
    x = jnp.repeat(
        jnp.expand_dims(g_values, axis=-2), self.watermarking_depth, axis=-2
    )  # [batch_size, seq_len, watermarking_depth, watermarking_depth]

    x = jnp.tril(
        x, k=-1
    )  # [batch_size, seq_len, watermarking_depth, watermarking_depth]

    logits = (
        jnp.einsum("ijkl,ijkl->ijk", self.delta, x) + self.beta
    )  # [batch_size, seq_len, watermarking_depth]

    p_two_unique_tokens = jax.nn.sigmoid(logits)
    p_one_unique_token = 1 - p_two_unique_tokens
    return p_one_unique_token, p_two_unique_tokens

  def __call__(self, g_values: jnp.ndarray) -> jnp.ndarray:
    """Computes the likelihoods P(g_values|watermarked).

    Args:
      g_values: g-values (values 0 or 1) of shape [batch_size, seq_len,
        watermarking_depth]

    Returns:
      p(g_values|watermarked) of shape [batch_size, seq_len,
      watermarking_depth].
    """
    p_one_unique_token, p_two_unique_tokens = self._compute_latents(g_values)

    # P(g_tl | watermarked) is equal to
    # 0.5 * [ (g_tl+0.5) * p_two_unique_tokens + p_one_unique_token].
    return 0.5 * ((g_values + 0.5) * p_two_unique_tokens + p_one_unique_token)


class LikelihoodModelUnwatermarked(nn.Module, LikelihoodModel):
  """Unwatermarked likelihood model for binary-valued g-values.

  This takes in g-values and returns p(g_values | not watermarked).
  """

  @nn.compact
  def __call__(self, g_values: jnp.ndarray) -> jnp.ndarray:
    """Computes the likelihoods P(g-values|not watermarked).

    Args:
      g_values: g-values (0 or 1 values) of shape [batch_size, seq_len,
        watermarking_depth, ...].

    Returns:
      Likelihoods of g-values given text is unwatermarked --
      p(g_values | not watermarked) of shape [batch_size, seq_len,
      watermarking_depth].
    """
    return 0.5 * jnp.ones_like(g_values)  # all g-values have prob 0.5.


class BayesianDetectorModule(nn.Module):
  """Bayesian classifier for watermark detection Flax Module.

  This detector uses Bayes' rule to compute a watermarking score, which is
  the posterior probability P(watermarked|g_values) that the text is
  watermarked, given its g_values.

  Note that this detector only works with Tournament-based watermarking using
  the Bernoulli(0.5) g-value distribution.
  """

  watermarking_depth: int  # The number of tournament layers.
  params: Mapping[str, Mapping[str, Any]] | None = None
  baserate: float = 0.5  # Prior probability P(w) that a text is watermarked.

  def l2_loss(self) -> jnp.ndarray:
    return self.likelihood_model_watermarked.l2_loss()

  def setup(self):
    """Initializes the model parameters."""
    self.likelihood_model_watermarked = LikelihoodModelWatermarked(
        watermarking_depth=self.watermarking_depth,
    )
    self.likelihood_model_unwatermarked = LikelihoodModelUnwatermarked()
    self.prior_log_odds = self.param(
        "prior_log_odds",
        lambda *x: jax.scipy.special.logit(self.baserate),
        (1,),
    )

  def __call__(
      self,
      g_values: jnp.ndarray,
      mask: jnp.ndarray,
  ) -> jnp.ndarray:
    """Computes the posterior log-odds of the text being watermarked.

    Args:
      g_values: g-values (with values 0 or 1) of shape (batch_size, seq_len,
        watermarking_depth).
      mask: Boolean or 0/1 array (batch_size, seq_len) indicating which g-values
        are used; g-values where the mask is False (0) are ignored.

    Returns:
      log(P(watermarked | g-values) / P(unwatermarked | g-values)), of shape
      (batch_size,). The posterior P(watermarked | g-values) is the sigmoid of
      this value.
    """
    likelihoods_watermarked = self.likelihood_model_watermarked(g_values)
    likelihoods_unwatermarked = self.likelihood_model_unwatermarked(g_values)
    mask = jnp.expand_dims(mask, -1)
    log_likelihoods_watermarked = jnp.log(likelihoods_watermarked)
    log_likelihoods_unwatermarked = jnp.log(likelihoods_unwatermarked)
    log_odds = log_likelihoods_watermarked - log_likelihoods_unwatermarked

    # Sum relative surprisals (log odds) across all token positions and layers.
    relative_surprisal_likelihood = jnp.einsum(
        "i...->i", log_odds * mask
    )  # (batch_size,).

    # Combine prior and likelihood log-odds.
    return self.prior_log_odds + relative_surprisal_likelihood


def xentropy_loss_from_logits(y: jnp.ndarray, logits: Any) -> jnp.ndarray:
  """Computes the mean binary cross-entropy from logits.

  Computed with log-sigmoid, so the loss and its gradient stay finite and
  non-zero for confidently wrong predictions, unlike clipping probabilities.

  Args:
    y: Labels in {0, 1}, shape (batch_size,).
    logits: Predicted log-odds, shape (batch_size,).

  Returns:
    Mean cross-entropy.
  """
  return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))


def loss_fn(
    params: Mapping[str, Any],
    detector_inputs: Any,
    w_true: jnp.ndarray,
    l2_batch_weight: float,
    detector_module: BayesianDetectorModule,
) -> jnp.ndarray:
  """Calculates the loss for a batch of data given parameters."""
  w_logits = detector_module.apply(params, *detector_inputs)
  unweighted_l2 = detector_module.apply(params, method=detector_module.l2_loss)
  l2_loss = l2_batch_weight * unweighted_l2  # pyrefly: ignore[unsupported-operation]
  return xentropy_loss_from_logits(w_true, w_logits) + l2_loss


def tpr_at_fpr(
    params: Mapping[str, Any],
    detector_inputs: tuple[Any, Any],
    w_true: jnp.ndarray | np.ndarray,
    minibatch_size: int,
    detector_module: Any,
    target_fpr: float = 0.01,
) -> float:
  """Computes TPR of the lowest threshold with empirical FPR <= target_fpr.

  Examples are scored by posterior log-odds in minibatches to avoid GPU memory
  overflow. With the negative scores sorted in descending order,
  s_0 >= s_1 >= ... >= s_{n-1}, and k = floor(target_fpr * n), the threshold
  is t = s_k. An example is classified as positive iff its score is strictly
  greater than t. Ties are handled as follows:
    * At most k negatives are strictly greater than s_k (only s_0, ..., s_{k-1}
      can be), so the empirical FPR is at most k / n <= target_fpr, even if
      several negatives are tied with s_k.
    * Any lower threshold t' < s_k would classify all of s_0, ..., s_k (k + 1
      negatives) as positive, so t = s_k gives the highest TPR among the rules
      "score > t" that satisfy the FPR constraint.
    * Positives tied with s_k are counted as negatives (not detected), and
      negatives tied with s_k are not false positives.

  Args:
    params: Parameters of detector_module.
    detector_inputs: Tuple (g_values, mask) where g_values has shape
      (num_examples, seq_len, watermarking_depth) and mask has shape
      (num_examples, seq_len).
    w_true: Labels in {0, 1} (1: watermarked), shape (num_examples,).
    minibatch_size: Number of examples scored per call to detector_module.
    detector_module: The detector module to apply.
    target_fpr: Target false positive rate, in [0, 1). Defaults to 0.01 (1%).

  Returns:
    Fraction of positive scores strictly greater than the threshold.

  Raises:
    ValueError: If target_fpr is not in [0, 1), if any score is NaN, or if
      w_true has no watermarked or no unwatermarked examples.
  """
  if not 0.0 <= target_fpr < 1.0:
    raise ValueError(f"target_fpr must be in [0, 1), but is {target_fpr}")

  g_values, masks = detector_inputs
  if len(g_values) == 0:
    raise ValueError("detector_inputs cannot be empty.")

  scores = np.concatenate([
      np.asarray(
          detector_module.apply(
              params,
              g_values[i : i + minibatch_size],
              masks[i : i + minibatch_size],
          )
      )
      for i in range(0, len(g_values), minibatch_size)
  ])
  if np.isnan(scores).any():
    raise ValueError("Scores contain NaN; cannot compute TPR at FPR.")

  labels = np.asarray(w_true)
  positives = scores[labels == 1]
  negatives = scores[labels == 0]
  if positives.size == 0 or negatives.size == 0:
    raise ValueError(
        "TPR at FPR requires positive and negative examples, but got"
        f" {positives.size} positives and {negatives.size} negatives."
    )

  k = int(np.floor(target_fpr * negatives.size))
  threshold = np.sort(negatives)[::-1][k]
  return float(np.mean(positives > threshold))


@enum.unique
class ValidationMetric(enum.Enum):
  """Validation metric used to select the best training epoch."""

  # Negative TPR at 1% FPR.
  TPR_AT_FPR = "tpr_at_fpr"
  # Mean cross-entropy, without the L2 penalty.
  CROSS_ENTROPY = "cross_entropy"


def train(
    *,
    detector_module: BayesianDetectorModule,
    g_values: jnp.ndarray,
    mask: jnp.ndarray,
    watermarked: jnp.ndarray,
    g_values_val: jnp.ndarray,
    mask_val: jnp.ndarray,
    watermarked_val: jnp.ndarray,
    epochs: int = 250,
    learning_rate: float = 1e-3,
    minibatch_size: int = 64,
    l2_weight: float = 0.0,
    shuffle: bool = True,
    seed: int = 0,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
    verbose: bool = False,
) -> tuple[Mapping[int, Mapping[str, PyTree]], float]:
  """Train detector based on wm and uwm data.

  Args:
    detector_module: Bayesian detector module.
    g_values: g_values of shape [num_train, seq_len, watermarking_depth].
    mask: binary mask of shape [num_train, seq_len].
    watermarked: binary array indicating whether each entry is watermarked of
      shape [num_train].
    g_values_val: validation g-values of shape [num_val, seq_len,
      watermarking_depth].
    mask_val: validation mask of shape [num_val, seq_len].
    watermarked_val: validation watermark labels of shape [num_val].
    epochs: number of epochs to train.
    learning_rate: learning rate to use.
    minibatch_size: minibatch size to use. We suggest setting this such that
      minibatch_size * seq_len * depth is not too large so that the memory
      doesn't exceed the GPU memory. Each entry of g-values of shape
      [batch_size, seq_len, depth] creates a tensor of shape [batch_size,
      seq_len, depth, depth] for predictions which approximately takes ~ 32 *
      minibatch_size * seq_len * depth * depth bits of memory.
    l2_weight: weight to apply to L2 regularization for delta parameters.
    shuffle: whether to shuffle before training.
    seed: seed for parameter initialization.
    validation_metric: validation metric to use.
    verbose: boolean indicating verbosity of training. If true, the loss will be
      printed. Defaulted to False.

  Returns:
    Tuple of
      training_history: training history keyed by epoch number where the values
      are
        dictionaries containing the loss, validation loss, and model parameters,
        keyed by
        'loss', 'val_loss', and 'params', respectively.
      min_val_loss: minimum validation loss achieved during training.
  """
  if epochs < 1:
    raise ValueError(f"epochs must be >= 1, but is {epochs}")
  minibatch_inds = jnp.arange(0, len(g_values), minibatch_size)
  minibatch_inds_val = jnp.arange(0, len(g_values_val), minibatch_size)

  rng = jax.random.PRNGKey(seed)
  param_rng, shuffle_rng = jax.random.split(rng)

  def coshuffle(*args):
    return [jax.random.permutation(shuffle_rng, x) for x in args]

  if shuffle:
    g_values, mask, watermarked = coshuffle(g_values, mask, watermarked)

  def update_fn_if_fpr_tpr(params):
    """Loss function for negative TPR@FPR=1% as the validation loss."""
    tpr_ = tpr_at_fpr(
        params=params,
        detector_inputs=(g_values_val, mask_val),
        w_true=watermarked_val,
        minibatch_size=minibatch_size,
        detector_module=detector_module,
    )
    return -tpr_

  n_minibatches = len(g_values) / minibatch_size
  l2_batch_weight_train = l2_weight / n_minibatches
  l2_batch_weight_val = 0.0
  loss_fn_train = functools.partial(
      loss_fn,
      l2_batch_weight=l2_batch_weight_train,
      detector_module=detector_module,
  )
  loss_fn_jitted_val = jax.jit(
      functools.partial(
          loss_fn,
          l2_batch_weight=l2_batch_weight_val,
          detector_module=detector_module,
      )
  )

  @jax.jit
  def update(gvalues, masks, labels, params, opt_state):
    loss_fn_partialed = functools.partial(
        loss_fn_train,
        detector_inputs=(gvalues, masks),
        w_true=labels,
    )
    loss, grads = jax.value_and_grad(loss_fn_partialed)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state

  def update_with_minibatches(gvalues, masks, labels, inds, params, opt_state):
    """Update params iff opt_state is not None and always returns the loss."""
    # loss_fn returns a mean over the minibatch and the last minibatch can be
    # smaller, so the epoch loss is weighted by minibatch size.
    weighted_loss_sum = 0.0
    num_examples = 0
    for start in inds:
      end = start + minibatch_size
      batch_labels = labels[start:end]
      loss, params, opt_state = update(
          gvalues[start:end],
          masks[start:end],
          batch_labels,
          params,
          opt_state,
      )
      weighted_loss_sum += loss * batch_labels.shape[0]
      num_examples += batch_labels.shape[0]
    return float(weighted_loss_sum / num_examples), params, opt_state

  def validate_with_minibatches(gvalues, masks, labels, inds, params):
    """Update params iff opt_state is not None and always returns the loss."""
    weighted_loss_sum = 0.0
    num_examples = 0
    for start in inds:
      end = start + minibatch_size
      batch_labels = labels[start:end]
      loss = loss_fn_jitted_val(
          params,
          detector_inputs=(gvalues[start:end], masks[start:end]),
          w_true=batch_labels,
      )
      weighted_loss_sum += loss * batch_labels.shape[0]
      num_examples += batch_labels.shape[0]
    return float(weighted_loss_sum / num_examples)

  def update_fn(opt_state, params):
    """Updates the model parameters and returns the loss."""
    loss, params, opt_state = update_with_minibatches(
        g_values, mask, watermarked, minibatch_inds, params, opt_state
    )
    if validation_metric == ValidationMetric.TPR_AT_FPR:
      val_loss = update_fn_if_fpr_tpr(params)
    else:
      val_loss = validate_with_minibatches(
          g_values_val,
          mask_val,
          watermarked_val,
          minibatch_inds_val,
          params,
      )

    return opt_state, params, loss, val_loss

  params = detector_module.params
  if params is None:
    params = detector_module.init(param_rng, g_values[:1], mask[:1])

  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  history = {}
  epochs_completed = 0
  while epochs_completed < epochs:
    opt_state, params, loss, val_loss = update_fn(opt_state, params)
    epochs_completed += 1
    if not np.isfinite(loss):
      raise ValueError(
          f"Training loss is {loss} at epoch {epochs_completed}"
          f" (learning_rate={learning_rate}, l2_weight={l2_weight})."
      )
    if not np.isfinite(val_loss):
      raise ValueError(
          f"Validation loss is {val_loss} at epoch {epochs_completed}"
          f" (learning_rate={learning_rate}, l2_weight={l2_weight})."
      )

    history[epochs_completed] = {
        "loss": loss,
        "val_loss": val_loss,
        "params": params["params"],
    }
    if verbose:
      if validation_metric == ValidationMetric.TPR_AT_FPR:
        val_desc = f"TPR@1%FPR {-val_loss:.3f} (val)"
      else:
        val_desc = f"cross-entropy {val_loss:.4f} (val)"
      print(
          f"Epoch {epochs_completed}: cross-entropy {loss:.4f} (train),"
          f" {val_desc}"
      )
  val_losses = np.array(
      [history[epoch]["val_loss"] for epoch in range(1, epochs + 1)],
      dtype=np.float64,
  )
  best_val_epoch = int(np.argmin(val_losses)) + 1
  min_val_loss = float(val_losses[best_val_epoch - 1])
  if validation_metric == ValidationMetric.TPR_AT_FPR:
    print(
        f"Best val epoch: {best_val_epoch}, TPR@1%FPR: {-min_val_loss:.3f}"
        f" (min_val_loss: {min_val_loss:.3f})"
    )
  else:
    print(f"Best val epoch: {best_val_epoch}, min_val_loss: {min_val_loss:.4f}")
  detector_module.params = {"params": history[best_val_epoch]["params"]}
  return history, min_val_loss


def _check_num_leaves_is_two(
    logits_processor: logits_processing.SynthIDLogitsProcessor,
) -> None:
  """Raises if the likelihood model does not hold for the processor."""
  if logits_processor.num_leaves != 2:
    raise ValueError(
        "The Bayesian detector's likelihood model assumes num_leaves=2, but"
        f" the logits processor has num_leaves={logits_processor.num_leaves}."
    )


class BayesianDetector:
  """Bayesian detector class used for watermark detection."""

  detector_module: BayesianDetectorModule
  eos_token_id: int | Sequence[int]
  logits_processor: logits_processing.SynthIDLogitsProcessor

  def __init__(
      self,
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      eos_token_id: int | Sequence[int],
      params: Mapping[str, Mapping[str, Any]],
  ):

    _check_num_leaves_is_two(logits_processor)
    self.detector_module = BayesianDetectorModule(
        watermarking_depth=logits_processor.depth,
        params=params,
    )
    self.logits_processor = logits_processor
    self.eos_token_id = eos_token_id

  def score(self, outputs: torch.LongTensor) -> jnp.ndarray:
    """Score the model output for possibility of being watermarked.

    Score is within [0, 1] where 0 is not watermarked and 1 is watermarked.

    Args:
      outputs: model output of shape [batch_size, output_len]

    Returns:
      scores of shape [batch_size]
    """
    if self.detector_module.params is None:
      raise ValueError("params must be set before calling score")
    g_values, mask = self.logits_processor.compute_g_values_and_mask(
        input_ids=outputs,
        eos_token_id=self.eos_token_id,
    )
    log_odds: Any = self.detector_module.apply(
        self.detector_module.params,
        g_values.cpu().numpy(),
        mask.cpu().numpy(),
    )
    return jax.nn.sigmoid(log_odds)

  @classmethod
  def process_raw_model_outputs(
      cls,
      *,
      tokenized_wm_outputs: Sequence[np.ndarray] | np.ndarray,
      tokenized_uwm_outputs: Sequence[np.ndarray] | np.ndarray,
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      eos_token_id: int | Sequence[int],
      torch_device: torch.device,
      test_size: float = 0.3,
      pos_truncation_length: int | None = 200,
      neg_truncation_length: int | None = 100,
      max_padded_length: int | None = None,
      seed: int = 0,
  ) -> tuple[
      jnp.ndarray,
      jnp.ndarray,
      jnp.ndarray,
      jnp.ndarray,
      jnp.ndarray,
      jnp.ndarray,
  ]:
    """Process raw models outputs into inputs we can train.

    Args:
      tokenized_wm_outputs: tokenized outputs of watermarked data.
      tokenized_uwm_outputs: tokenized outputs of unwatermarked data.
      logits_processor: logits processor used for watermarking.
      eos_token_id: EOS token ID or sequence of EOS token IDs.
      torch_device: torch device to use.
      test_size: test size to use for train-test split.
      pos_truncation_length: Length to truncate wm outputs. If None, no
        truncation is applied.
      neg_truncation_length: Length to truncate uwm outputs. If None, no
        truncation is done.
      max_padded_length: Length to pad truncated outputs so that all processed
        entries have same shape.
      seed: seed for train-test split.

    Returns:
      Tuple of 6 arrays (train_g_values, train_masks, train_labels, cv_g_values,
      cv_masks, cv_labels).
    """
    if max_padded_length is None:
      if pos_truncation_length is None or neg_truncation_length is None:
        raise ValueError(
            "max_padded_length must be set if pos_truncation_length or"
            " neg_truncation_length is None."
        )
      max_padded_length = max(pos_truncation_length, neg_truncation_length)
    for name, truncation_length in (
        ("pos_truncation_length", pos_truncation_length),
        ("neg_truncation_length", neg_truncation_length),
    ):
      if truncation_length is None:
        continue
      if max_padded_length < truncation_length:
        raise ValueError(
            f"max_padded_length ({max_padded_length}) must be at least"
            f" {name} ({truncation_length})."
        )

    # Independent streams for the two splits and the two shuffles.
    wm_split_seq, uwm_split_seq, train_perm_seq, cv_perm_seq = (
        np.random.SeedSequence(seed).spawn(4)
    )

    # Split data into train and CV
    train_wm_outputs, cv_wm_outputs = model_selection.train_test_split(
        tokenized_wm_outputs,
        test_size=test_size,
        random_state=int(wm_split_seq.generate_state(1)[0]),
    )

    train_uwm_outputs, cv_uwm_outputs = model_selection.train_test_split(
        tokenized_uwm_outputs,
        test_size=test_size,
        random_state=int(uwm_split_seq.generate_state(1)[0]),
    )

    def process_split(
        outputs: Sequence[np.ndarray] | np.ndarray,
        *,
        is_pos: bool,
        is_cv: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
      truncation_length = (
          pos_truncation_length if (is_pos or is_cv) else neg_truncation_length
      )
      masks, g_values = process_outputs_for_training(
          [
              torch.as_tensor(batch, device=torch_device, dtype=torch.long)
              for batch in outputs
          ],
          logits_processor=logits_processor,
          eos_token_id=eos_token_id,
          truncation_length=truncation_length,
          max_length=max_padded_length,
          torch_device=torch_device,
      )
      if not masks:
        split_name = (
            f"{'watermarked' if is_pos else 'unwatermarked'} "
            f"{'cross-validation' if is_cv else 'training'}"
        )
        raise ValueError(
            f"No {split_name} rows left after filtering with truncation"
            f" length {truncation_length}: rows need at least that many"
            " tokens before the first EOS token."
        )
      return (
          torch.cat(g_values, dim=0).cpu().numpy(),
          torch.cat(masks, dim=0).cpu().numpy(),
      )

    def build_dataset(
        wm_outputs: Sequence[np.ndarray] | np.ndarray,
        uwm_outputs: Sequence[np.ndarray] | np.ndarray,
        *,
        is_cv: bool,
        perm_seq: np.random.SeedSequence,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
      wm_g, wm_m = process_split(wm_outputs, is_pos=True, is_cv=is_cv)
      uwm_g, uwm_m = process_split(uwm_outputs, is_pos=False, is_cv=is_cv)
      g_values = np.concatenate((wm_g, uwm_g), axis=0)
      masks = np.concatenate((wm_m, uwm_m), axis=0)
      labels = np.concatenate((
          np.ones(len(wm_m), dtype=bool),
          np.zeros(len(uwm_m), dtype=bool),
      ))
      perm = np.random.default_rng(perm_seq).permutation(len(g_values))
      return (
          jnp.asarray(g_values[perm]),
          jnp.asarray(masks[perm]),
          jnp.asarray(labels[perm]),
      )

    train_g_values, train_masks, train_labels = build_dataset(
        train_wm_outputs,
        train_uwm_outputs,
        is_cv=False,
        perm_seq=train_perm_seq,
    )
    cv_g_values, cv_masks, cv_labels = build_dataset(
        cv_wm_outputs,
        cv_uwm_outputs,
        is_cv=True,
        perm_seq=cv_perm_seq,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return (
        train_g_values,
        train_masks,
        train_labels,
        cv_g_values,
        cv_masks,
        cv_labels,
    )

  @classmethod
  def train_best_detector_given_g_values(
      cls,
      *,
      train_g_values: jnp.ndarray,
      train_masks: jnp.ndarray,
      train_labels: jnp.ndarray,
      cv_g_values: jnp.ndarray,
      cv_masks: jnp.ndarray,
      cv_labels: jnp.ndarray,
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      eos_token_id: int | Sequence[int],
      n_epochs: int = 50,
      learning_rate: float = 2.1e-2,
      l2_weights: np.ndarray = np.logspace(-3, -2, num=4),
      verbose: bool = False,
      seed: int = 0,
  ) -> tuple["BayesianDetector", float]:
    """Train best detector given g_values, mask and labels."""
    _check_num_leaves_is_two(logits_processor)
    if np.size(l2_weights) == 0:
      raise ValueError("l2_weights must not be empty.")
    # train() raises on non-finite losses, so every min_val_loss is finite.
    results = []
    for l2_weight in l2_weights:
      detector_module = BayesianDetectorModule(
          watermarking_depth=logits_processor.depth,
      )
      _, min_val_loss = train(
          detector_module=detector_module,
          g_values=train_g_values,
          mask=train_masks,
          watermarked=train_labels,
          g_values_val=cv_g_values,
          mask_val=cv_masks,
          watermarked_val=cv_labels,
          learning_rate=learning_rate,
          l2_weight=l2_weight,
          epochs=n_epochs,
          seed=seed,
          verbose=verbose,
      )
      results.append((min_val_loss, detector_module.params))
    # min() returns the first minimum, i.e. the smallest index on ties.
    lowest_loss, best_params = min(results, key=lambda result: result[0])
    return cls(logits_processor, eos_token_id, best_params), lowest_loss

  @classmethod
  def train_best_detector(
      cls,
      *,
      tokenized_wm_outputs: Sequence[np.ndarray] | np.ndarray,
      tokenized_uwm_outputs: Sequence[np.ndarray] | np.ndarray,
      logits_processor: logits_processing.SynthIDLogitsProcessor,
      eos_token_id: int | Sequence[int],
      torch_device: torch.device,
      test_size: float = 0.3,
      pos_truncation_length: int | None = 200,
      neg_truncation_length: int | None = 100,
      max_padded_length: int | None = None,
      n_epochs: int = 50,
      learning_rate: float = 2.1e-2,
      l2_weights: np.ndarray = np.logspace(-3, -2, num=4),
      verbose: bool = False,
      seed: int = 0,
  ) -> tuple["BayesianDetector", float]:
    """Construct, train and return the best detector based on wm and uwm data.

    In practice, we have found that tuning pos_truncation_length,
    neg_truncation_length, n_epochs, learning_rate and l2_weights can help
    improve the performance of the detector. We recommend tuning these
    parameters for your data.

    Args:
      tokenized_wm_outputs: tokenized outputs of watermarked data.
      tokenized_uwm_outputs: tokenized outputs of unwatermarked data.
      logits_processor: logits processor used for watermarking.
      eos_token_id: EOS token ID or sequence of EOS token IDs.
      torch_device: torch device to use.
      test_size: test size to use for train-test split.
      pos_truncation_length: Length to truncate wm outputs. If None, no
        truncation is applied.
      neg_truncation_length: Length to truncate uwm outputs. If None, no
        truncation is done.
      max_padded_length: Length to pad truncated outputs so that all processed
        entries have same shape.
      n_epochs: Number of epochs to train the detector.
      learning_rate: Learning rate to use for training the detector.
      l2_weights: L2 weights to use for training the detector.
      verbose: Whether to print training progress.
      seed: Seed for training.

    Returns:
      Tuple of trained detector and loss achieved on CV data.
    """
    _check_num_leaves_is_two(logits_processor)
    (
        train_g_values,
        train_masks,
        train_labels,
        cv_g_values,
        cv_masks,
        cv_labels,
    ) = cls.process_raw_model_outputs(
        tokenized_wm_outputs=tokenized_wm_outputs,
        tokenized_uwm_outputs=tokenized_uwm_outputs,
        logits_processor=logits_processor,
        eos_token_id=eos_token_id,
        torch_device=torch_device,
        test_size=test_size,
        pos_truncation_length=pos_truncation_length,
        neg_truncation_length=neg_truncation_length,
        max_padded_length=max_padded_length,
        seed=seed,
    )

    return cls.train_best_detector_given_g_values(
        train_g_values=train_g_values,
        train_masks=train_masks,
        train_labels=train_labels,
        cv_g_values=cv_g_values,
        cv_masks=cv_masks,
        cv_labels=cv_labels,
        logits_processor=logits_processor,
        eos_token_id=eos_token_id,
        verbose=verbose,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        l2_weights=l2_weights,
        seed=seed,
    )
