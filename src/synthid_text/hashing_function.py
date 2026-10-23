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

"""Threefry 2x32 PRNG implementation in PyTorch with direct bit extraction.

This module implements the Threefry 2x32 counter-based PRNG for generation-time
watermarking and detection. It operates on 64-bit keys (two 32-bit words) and
extracts independent Bernoulli(0.5) g-value bits directly from the 64-bit
output stream for maximum throughput and compilation efficiency.

References:
  - Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. (2011).
    "Parallel Random Numbers: As Easy as 1, 2, 3", SC 2011.
    https://doi.org/10.1145/2063384.2063405
"""

import torch


def _ensure_int64(x: torch.Tensor) -> torch.Tensor:
  """Casts to int64 and keeps the low 32 bits."""
  return x.to(torch.int64) & 0xFFFFFFFF


def _rotate_left_u32(x: torch.Tensor, d: int) -> torch.Tensor:
  """Rotates the 32-bit unsigned integer x left by d bits.

  Args:
    x: int64 tensor with values in [0, 2**32).
    d: Rotation amount in [1, 31].

  Returns:
    int64 tensor with values in [0, 2**32).
  """
  return ((x << d) | (x >> (32 - d))) & 0xFFFFFFFF


def _apply_round(v0: torch.Tensor, v1: torch.Tensor, rot: int):
  """Applies one Threefry mix round."""
  v0 = (v0 + v1) & 0xFFFFFFFF
  v1 = _rotate_left_u32(v1, rot)
  v1 = v0 ^ v1
  return v0, v1


def threefry2x32(
    key1: torch.Tensor,
    key2: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Applies the Threefry 2x32 block cipher (20 rounds, fully unrolled).

  Follows the 20-round Threefry 2x32 specification from Salmon et al. (SC '11),
  using the 32-bit Skein key parity constant 0x1BD11BDA and rotation schedules
  [13, 15, 26, 6] and [17, 29, 16, 24].

  All arguments are int64 tensors representing uint32 values.
  Arbitrary batch dimensions are supported via broadcasting.

  Args:
    key1: First word of the key.
    key2: Second word of the key.
    x1: First word of the counter/input.
    x2: Second word of the counter/input.

  Returns:
    Tuple (out1, out2) of int64 tensors representing uint32 outputs.
  """
  skein_ks_parity32 = 0x1BD11BDA
  rot_0_0, rot_0_1, rot_0_2, rot_0_3 = 13, 15, 26, 6
  rot_1_0, rot_1_1, rot_1_2, rot_1_3 = 17, 29, 16, 24

  key1 = _ensure_int64(key1)
  key2 = _ensure_int64(key2)
  x1 = _ensure_int64(x1)
  x2 = _ensure_int64(x2)

  ks0 = key1
  ks1 = key2
  ks2 = (key1 ^ key2 ^ skein_ks_parity32) & 0xFFFFFFFF

  # Initial key injection.
  x1 = (x1 + ks0) & 0xFFFFFFFF
  x2 = (x2 + ks1) & 0xFFFFFFFF

  # Round group 1: rotations[0], key inject with ks1, ks2, round_num=1
  x1, x2 = _apply_round(x1, x2, rot_0_0)
  x1, x2 = _apply_round(x1, x2, rot_0_1)
  x1, x2 = _apply_round(x1, x2, rot_0_2)
  x1, x2 = _apply_round(x1, x2, rot_0_3)
  x1 = (x1 + ks1) & 0xFFFFFFFF
  x2 = (x2 + ks2 + 1) & 0xFFFFFFFF

  # Round group 2: rotations[1], key inject with ks2, ks0, round_num=2
  x1, x2 = _apply_round(x1, x2, rot_1_0)
  x1, x2 = _apply_round(x1, x2, rot_1_1)
  x1, x2 = _apply_round(x1, x2, rot_1_2)
  x1, x2 = _apply_round(x1, x2, rot_1_3)
  x1 = (x1 + ks2) & 0xFFFFFFFF
  x2 = (x2 + ks0 + 2) & 0xFFFFFFFF

  # Round group 3: rotations[0], key inject with ks0, ks1, round_num=3
  x1, x2 = _apply_round(x1, x2, rot_0_0)
  x1, x2 = _apply_round(x1, x2, rot_0_1)
  x1, x2 = _apply_round(x1, x2, rot_0_2)
  x1, x2 = _apply_round(x1, x2, rot_0_3)
  x1 = (x1 + ks0) & 0xFFFFFFFF
  x2 = (x2 + ks1 + 3) & 0xFFFFFFFF

  # Round group 4: rotations[1], key inject with ks1, ks2, round_num=4
  x1, x2 = _apply_round(x1, x2, rot_1_0)
  x1, x2 = _apply_round(x1, x2, rot_1_1)
  x1, x2 = _apply_round(x1, x2, rot_1_2)
  x1, x2 = _apply_round(x1, x2, rot_1_3)
  x1 = (x1 + ks1) & 0xFFFFFFFF
  x2 = (x2 + ks2 + 4) & 0xFFFFFFFF

  # Round group 5: rotations[0], key inject with ks2, ks0, round_num=5
  x1, x2 = _apply_round(x1, x2, rot_0_0)
  x1, x2 = _apply_round(x1, x2, rot_0_1)
  x1, x2 = _apply_round(x1, x2, rot_0_2)
  x1, x2 = _apply_round(x1, x2, rot_0_3)
  x1 = (x1 + ks2) & 0xFFFFFFFF
  x2 = (x2 + ks0 + 5) & 0xFFFFFFFF

  return x1, x2


def threefry_seed(data: torch.Tensor) -> torch.Tensor:
  """Converts integer(s) to Threefry 2x32 key(s).

  Splits a 64-bit integer into high and low 32-bit words.

  Args:
    data: Integer tensor of arbitrary shape.

  Returns:
    int64 tensor (*data.shape, 2) representing uint32 key pairs.
  """
  data = data.to(torch.int64)
  k1 = (data >> 32) & 0xFFFFFFFF  # High 32 bits.
  k2 = data & 0xFFFFFFFF  # Low 32 bits.
  return torch.stack([k1, k2], dim=-1)


def fold_in(key: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
  """Folds data into a Threefry 2x32 key.

  The data is split into high and low 32-bit words, which are encrypted with
  Threefry 2x32 under the key.

  Args:
    key: int64 tensor (..., 2) representing uint32 key pairs.
    data: Integer tensor (...). Its shape must broadcast with key[..., 0].

  Returns:
    int64 tensor (..., 2) representing the new uint32 key pairs.
  """
  seed = threefry_seed(data)  # (..., 2)
  key1 = key[..., 0]
  key2 = key[..., 1]
  x1 = seed[..., 0]
  x2 = seed[..., 1]
  out1, out2 = threefry2x32(key1, key2, x1, x2)
  return torch.stack([out1, out2], dim=-1)


def recursively_fold_in(
    key: torch.Tensor,
    data: torch.Tensor,
) -> torch.Tensor:
  """Sequentially folds each element of data along the last dim into a key.

  Args:
    key: int64 tensor (..., 2) representing uint32 key pairs.
    data: Integer tensor (..., k). The elements data[..., 0], ..., data[..., k -
      1] are folded in with `fold_in` in that order.

  Returns:
    int64 tensor (..., 2) representing the resulting uint32 key pairs. If
    k == 0, the key is returned unchanged.
  """
  k = data.shape[-1]
  for i in range(k):
    key = fold_in(key, data[..., i])
  return key


def pack_key_to_int64(key: torch.Tensor) -> torch.Tensor:
  """Packs a (..., 2) 32-bit key pair into a single int64 tensor (...)."""
  return (key[..., 0] << 32) | (key[..., 1] & 0xFFFFFFFF)


def compute_g_values_boolean(
    context_key: torch.Tensor,
    token_ids: torch.Tensor,
    depth: int = 30,
) -> torch.Tensor:
  """Computes binary g-values {0.0, 1.0} directly from Threefry 64-bit outputs.

  This evaluates Threefry (depth + 63) // 64 times per (context, token) pair
  and extracts the depth Bernoulli(0.5) bits from the outputs with vectorised
  bit shifts.

  Args:
    context_key: int64 tensor (..., 2): precomputed context key(s).
    token_ids: int64 tensor (..., V): candidate token IDs. The leading dims of
      context_key and token_ids must broadcast.
    depth: Number of tournament layers (D). Defaults to 30.

  Returns:
    float32 tensor (..., V, D) containing binary g-values (0.0 or 1.0).
  """
  num_blocks = (depth + 63) // 64

  # ctx keys: (..., 1, 1), x1: (..., V, 1)
  ctx_k1 = context_key[..., 0:1].unsqueeze(-1)
  ctx_k2 = context_key[..., 1:2].unsqueeze(-1)
  x1 = (token_ids.to(torch.int64) & 0xFFFFFFFF).unsqueeze(-1)

  # x2 (block index counter): (1, ..., 1, num_blocks)
  x2_shape = [1] * (token_ids.ndim + 1)
  x2_shape[-1] = num_blocks
  x2 = torch.arange(
      num_blocks, device=token_ids.device, dtype=torch.int64
  ).view(*x2_shape)

  # Single Threefry broadcast evaluates all blocks in parallel ->
  # (..., V, num_blocks)
  bits1, bits2 = threefry2x32(ctx_k1, ctx_k2, x1, x2)
  packed_64 = (bits1 << 32) | (bits2 & 0xFFFFFFFF)

  # Slicing D depth bits via parallel shift & mask
  depth_indices = torch.arange(
      depth, device=token_ids.device, dtype=torch.int64
  )
  block_indices = depth_indices // 64
  bit_shifts = depth_indices % 64

  # (..., V, D)
  selected_blocks = packed_64[..., block_indices]
  g_values = ((selected_blocks >> bit_shifts) & 1).to(torch.float32)
  return g_values
