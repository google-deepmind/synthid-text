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

"""Hashing function implementation."""

import torch


def accumulate_hash(
    current_hash: torch.LongTensor,
    data: torch.LongTensor,
    multiplier: int = 6364136223846793005,
    increment: int = 1,
) -> torch.LongTensor:
  """Accumulate hash of data on current hash.

  Method uses adapted linear congruential generator (LCG)with newlib/musl
  parameters.

  This function has following property -
  f(x, data[T]) = f(f(x, data[:T - 1]), data[T])

  This function expects current_hash.shape and data.shape[:-1] to
  match/broadcastable.

  Args:
    current_hash: (shape,)
    data: (shape, tensor_len)
    multiplier: (int) multiplier of linear congruential generator
    increment: (int) increment of linear congruential generator

  Returns:
    updated hash (shape,)
  """
  for i in range(data.shape[-1]):
    current_hash = torch.add(current_hash, data[..., i])
    current_hash = torch.mul(current_hash, multiplier)
    current_hash = torch.add(current_hash, increment)
  return current_hash
