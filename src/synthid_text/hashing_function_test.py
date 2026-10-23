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

"""Tests for Threefry 2x32 PRNG in hashing_function."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torch

from synthid_text import hashing_function


class HashingFunctionTest(parameterized.TestCase):

  # Known Answer Test (KAT) vectors from the original Threefry reference
  # implementation (Salmon et al., SC 2011, "Parallel Random Numbers: As Easy
  # as 1, 2, 3"). Test vectors are published in DEShawResearch Random123-Boost:
  # test/test_threefry.cpp.
  @parameterized.parameters(
      dict(
          key=[0x00000000, 0x00000000],
          counter=[0x00000000, 0x00000000],
          expected=[0x6B200159, 0x99BA4EFE],
      ),
      dict(
          key=[0xFFFFFFFF, 0xFFFFFFFF],
          counter=[0xFFFFFFFF, 0xFFFFFFFF],
          expected=[0x1CB996FC, 0xBB002BE7],
      ),
      dict(
          key=[0x13198A2E, 0x03707344],
          counter=[0x243F6A88, 0x85A308D3],
          expected=[0xC4923A9C, 0x483DF7A0],
      ),
  )
  def test_threefry2x32_kat_vectors(self, key, counter, expected):
    """Verifies threefry2x32 against published Random123 KAT vectors."""
    out1, out2 = hashing_function.threefry2x32(
        torch.tensor(key[0], dtype=torch.int64),
        torch.tensor(key[1], dtype=torch.int64),
        torch.tensor(counter[0], dtype=torch.int64),
        torch.tensor(counter[1], dtype=torch.int64),
    )
    self.assertEqual(out1.item(), expected[0])
    self.assertEqual(out2.item(), expected[1])

  def test_threefry_seed_and_pack(self):
    val = 0x123456789ABCDEF0
    key = hashing_function.threefry_seed(torch.tensor(val, dtype=torch.int64))
    self.assertEqual(key.shape, (2,))
    self.assertEqual(key[0].item(), 0x12345678)
    self.assertEqual(key[1].item(), 0x9ABCDEF0)

    packed = hashing_function.pack_key_to_int64(key)
    self.assertEqual(packed.item(), val)

  def test_pack_key_to_int64_round_trip(self):
    # High words with the top bit set pack to negative int64 values.
    keys = torch.tensor(
        [[0, 0], [0x12345678, 0x9ABCDEF0], [0xFFFFFFFF, 1], [0x80000000, 0]]
    )
    packed = hashing_function.pack_key_to_int64(keys)
    high, low = (packed >> 32) & 0xFFFFFFFF, packed & 0xFFFFFFFF
    self.assertTrue(torch.equal(keys, torch.stack([high, low], dim=-1)))

  def test_recursively_fold_in_shapes(self):
    base_key = hashing_function.threefry_seed(
        torch.tensor(42, dtype=torch.int64)
    ).expand(3, 2)
    context = torch.randint(0, 1000, (3, 4), dtype=torch.int64)
    ctx_key = hashing_function.recursively_fold_in(base_key, context)
    self.assertEqual(ctx_key.shape, (3, 2))

  @parameterized.parameters(30, 64, 100)
  def test_compute_g_values_boolean(self, depth):
    batch_size = 2
    vocab_size = 50

    base_key = hashing_function.threefry_seed(
        torch.tensor(42, dtype=torch.int64)
    ).expand(batch_size, 2)
    context = torch.randint(0, 1000, (batch_size, 4), dtype=torch.int64)
    ctx_key = hashing_function.recursively_fold_in(base_key, context)

    cand_indices = torch.randint(
        0, 10000, (batch_size, vocab_size), dtype=torch.int64
    )
    g_vals = hashing_function.compute_g_values_boolean(
        ctx_key, cand_indices, depth=depth
    )

    self.assertEqual(g_vals.shape, (batch_size, vocab_size, depth))
    # All values must be binary 0.0 or 1.0
    self.assertTrue(torch.all((g_vals == 0.0) | (g_vals == 1.0)))

    if depth > 64:
      # Verify that bits in block 0 and block 1 are not duplicate copies.
      self.assertFalse(torch.equal(g_vals[..., :36], g_vals[..., 64:100]))

    # Over a large sample, empirical mean should be close to 0.5
    large_cand = torch.arange(10000, dtype=torch.int64).unsqueeze(0)
    single_ctx = ctx_key[0:1]
    g_large = hashing_function.compute_g_values_boolean(
        single_ctx, large_cand, depth=depth
    )
    mean_g = g_large.mean().item()
    self.assertAlmostEqual(mean_g, 0.5, delta=0.03)

  def test_threefry_seed_matches_jax(self):
    """Verifies that threefry_seed matches JAX threefry2x32 key generation."""
    with jax.enable_x64():
      seeds = [0, 1, 42, 654, 2**31 - 1, 2**32, 2**63 - 1]
      for seed in seeds:
        pt_seed = hashing_function.threefry_seed(
            torch.tensor(seed, dtype=torch.int64)
        )
        jax_key = jax.random.key(seed, impl="threefry2x32")
        jax_seed = jax.random.key_data(jax_key)
        np.testing.assert_array_equal(pt_seed.numpy(), np.array(jax_seed))

  def test_fold_in_matches_jax(self):
    """Verifies that fold_in matches JAX fold_in across sample values."""
    np.random.seed(54321)
    for _ in range(50):
      k1 = int(np.random.randint(0, 0xFFFFFFFF, dtype=np.uint32))
      k2 = int(np.random.randint(0, 0xFFFFFFFF, dtype=np.uint32))
      token_val = int(np.random.randint(0, 50000))

      pt_key = torch.tensor([k1, k2], dtype=torch.int64)
      pt_folded = hashing_function.fold_in(
          pt_key, torch.tensor(token_val, dtype=torch.int64)
      )

      jax_key = jax.random.wrap_key_data(
          jnp.array([k1, k2], dtype=jnp.uint32), impl="threefry2x32"
      )
      jax_folded = jax.random.fold_in(jax_key, np.uint32(token_val))
      jax_folded_data = jax.random.key_data(jax_folded)
      np.testing.assert_array_equal(
          pt_folded.numpy(), np.array(jax_folded_data)
      )

  def test_recursively_fold_in_matches_jax(self):
    """Verifies that recursively_fold_in matches sequential JAX fold_in."""
    np.random.seed(999)
    batch_size = 4
    context_len = 6
    base_seed = 654

    base_key_pt = hashing_function.threefry_seed(
        torch.tensor(base_seed, dtype=torch.int64)
    ).expand(batch_size, 2)
    tokens_np = np.random.randint(
        0, 10000, size=(batch_size, context_len), dtype=np.int64
    )
    tokens_pt = torch.from_numpy(tokens_np)

    pt_folded = hashing_function.recursively_fold_in(base_key_pt, tokens_pt)

    for b in range(batch_size):
      jax_key = jax.random.key(base_seed, impl="threefry2x32")
      for t in range(context_len):
        jax_key = jax.random.fold_in(jax_key, np.uint32(tokens_np[b, t]))
      np.testing.assert_array_equal(
          pt_folded[b].numpy(), np.array(jax.random.key_data(jax_key))
      )

  # (context_key, token, block) -> threefry2x32(context_key, (token, block)),
  # generated with jax.extend.random.threefry_2x32 (JAX 0.11.1).
  _THREEFRY_REFERENCE = {
      ((0x00000000, 0x00000000), 1, 0): (0x508EFB2C, 0xC0DE3F32),
      ((0x00000000, 0x00000000), 1, 1): (0x9375D35F, 0x37C5FA2C),
      ((0x00000000, 0x00000000), 50000, 0): (0xFDB89356, 0x812EA27E),
      ((0x00000000, 0x00000000), 50000, 1): (0x1CA129CD, 0xF6AA2B4B),
      ((0x00000000, 0x00000000), 256000, 0): (0x193F7654, 0xFD16F115),
      ((0x00000000, 0x00000000), 256000, 1): (0xB39BFFC3, 0x69D3125E),
      ((0x9E3779B9, 0xF1234567), 1, 0): (0x2887434A, 0xD6B8637C),
      ((0x9E3779B9, 0xF1234567), 1, 1): (0x5EE94091, 0x66624B79),
      ((0x9E3779B9, 0xF1234567), 50000, 0): (0x34911A4C, 0x85F68A2E),
      ((0x9E3779B9, 0xF1234567), 50000, 1): (0x44E4CF9C, 0x0843C5E9),
      ((0x9E3779B9, 0xF1234567), 256000, 0): (0x4426E3A7, 0xB839526A),
      ((0x9E3779B9, 0xF1234567), 256000, 1): (0xB5AEF048, 0x4C6DCA5D),
      ((0xFFFFFFFF, 0x00000001), 1, 0): (0xF3AB21F7, 0x1EF2D992),
      ((0xFFFFFFFF, 0x00000001), 1, 1): (0x48627E9A, 0xDBA02E7A),
      ((0xFFFFFFFF, 0x00000001), 50000, 0): (0xC2F16654, 0x854FC64D),
      ((0xFFFFFFFF, 0x00000001), 50000, 1): (0x6AA47185, 0x50954C12),
      ((0xFFFFFFFF, 0x00000001), 256000, 0): (0x5CA930B1, 0xF2004456),
      ((0xFFFFFFFF, 0x00000001), 256000, 1): (0x94BC3CD3, 0x59FCA497),
  }

  def test_compute_g_values_boolean_bit_layout(self):
    """Pins the g-value bit layout to reference Threefry 2x32 outputs.

    Layer d of the g-values for (context_key, token) is bit d % 64 of
    (out1 << 32) | out2, where (out1, out2) = threefry2x32(context_key,
    (token, d // 64)).
    """
    depth = 100  # Two 64-bit blocks.
    context_keys = [
        (0x00000000, 0x00000000),
        (0x9E3779B9, 0xF1234567),
        (0xFFFFFFFF, 0x00000001),
    ]
    tokens = [1, 50000, 256000]
    g_values = hashing_function.compute_g_values_boolean(
        torch.tensor(context_keys, dtype=torch.int64),
        torch.tensor([tokens] * len(context_keys), dtype=torch.int64),
        depth=depth,
    ).numpy()
    self.assertEqual(g_values.shape, (len(context_keys), len(tokens), depth))

    expected = np.zeros_like(g_values)
    for b, context_key in enumerate(context_keys):
      for v, token in enumerate(tokens):
        for d in range(depth):
          out1, out2 = self._THREEFRY_REFERENCE[(context_key, token, d // 64)]
          expected[b, v, d] = (((out1 << 32) | out2) >> (d % 64)) & 1
    np.testing.assert_array_equal(g_values, expected)


if __name__ == "__main__":
  absltest.main()
