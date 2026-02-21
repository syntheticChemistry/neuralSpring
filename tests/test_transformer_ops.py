# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Unit tests for NumPy transformer operations.

Tests mathematical properties of softmax, GELU, layer norm, causal mask,
and scaled dot-product attention against analytical known values.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        from transformer_inference import softmax

        x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        s = softmax(x, axis=-1)
        np.testing.assert_allclose(s.sum(axis=-1), [1.0, 1.0], atol=1e-15)

    def test_nonnegative(self) -> None:
        from transformer_inference import softmax

        rng = np.random.default_rng(42)
        x = rng.standard_normal((10, 20))
        assert np.all(softmax(x, axis=-1) >= 0.0)

    def test_uniform_on_equal_inputs(self) -> None:
        from transformer_inference import softmax

        x = np.full((4, 5), 3.14)
        s = softmax(x, axis=-1)
        np.testing.assert_allclose(s, np.full((4, 5), 0.2), atol=1e-15)

    def test_numerically_stable(self) -> None:
        from transformer_inference import softmax

        x = np.array([[1e10, 1e10 + 1, 1e10 + 2]])
        s = softmax(x, axis=-1)
        assert np.all(np.isfinite(s))
        np.testing.assert_allclose(s.sum(), 1.0, atol=1e-14)

    def test_argmax_preserved(self) -> None:
        from transformer_inference import softmax

        x = np.array([[1.0, 5.0, 2.0]])
        s = softmax(x, axis=-1)
        assert np.argmax(s) == 1


# ---------------------------------------------------------------------------
# GELU
# ---------------------------------------------------------------------------


class TestGELU:
    def test_zero(self) -> None:
        from transformer_inference import gelu_numpy

        np.testing.assert_allclose(gelu_numpy(np.array([0.0])), [0.0], atol=1e-10)

    def test_positive_for_large(self) -> None:
        from transformer_inference import gelu_numpy

        x = np.array([3.0, 5.0, 10.0])
        np.testing.assert_allclose(gelu_numpy(x), x, atol=0.01)

    def test_negative_for_large_negative(self) -> None:
        from transformer_inference import gelu_numpy

        x = np.array([-5.0, -10.0])
        result = gelu_numpy(x)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=0.01)

    def test_monotonic_around_origin(self) -> None:
        from transformer_inference import gelu_numpy

        x = np.linspace(-1, 3, 1000)
        g = gelu_numpy(x)
        assert np.all(np.diff(g[x > 0]) > 0)


# ---------------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------------


class TestLayerNorm:
    def test_zero_mean_unit_variance(self) -> None:
        from transformer_inference import layer_norm_numpy

        rng = np.random.default_rng(42)
        x = rng.standard_normal((8, 32)) * 10 + 5
        gamma = np.ones(32)
        beta = np.zeros(32)
        out = layer_norm_numpy(x, gamma, beta)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=1e-5)

    def test_scale_and_shift(self) -> None:
        from transformer_inference import layer_norm_numpy

        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.full(4, 2.0)
        beta = np.full(4, 1.0)
        out = layer_norm_numpy(x, gamma, beta)
        np.testing.assert_allclose(out.mean(axis=-1), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Causal Mask
# ---------------------------------------------------------------------------


class TestCausalMask:
    def test_shape(self) -> None:
        from transformer_inference import causal_mask

        m = causal_mask(8)
        assert m.shape == (8, 8)

    def test_lower_triangle_zero(self) -> None:
        from transformer_inference import causal_mask

        m = causal_mask(5)
        lower = np.tril(m)
        np.testing.assert_array_equal(lower, 0.0)

    def test_upper_triangle_negative(self) -> None:
        from transformer_inference import causal_mask

        m = causal_mask(5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert m[i, j] < -1e6


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
# ---------------------------------------------------------------------------


class TestSDPA:
    def test_weights_sum_to_one(self) -> None:
        from transformer_inference import scaled_dot_product_attention

        rng = np.random.default_rng(42)
        Q = rng.standard_normal((4, 8))
        K = rng.standard_normal((4, 8))
        V = rng.standard_normal((4, 8))
        _, w = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-14)

    def test_output_shape(self) -> None:
        from transformer_inference import scaled_dot_product_attention

        Q = np.zeros((6, 10))
        K = np.zeros((6, 10))
        V = np.zeros((6, 10))
        out, _ = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (6, 10)

    def test_identity_attention(self) -> None:
        """When Q=K and V=I, attention should approximate identity."""
        from transformer_inference import scaled_dot_product_attention

        d = 4
        Q = np.eye(d) * 100
        K = Q.copy()
        V = np.eye(d)
        out, w = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(out, np.eye(d), atol=0.01)

    def test_causal_blocks_future(self) -> None:
        from transformer_inference import causal_mask, scaled_dot_product_attention

        rng = np.random.default_rng(42)
        s = 8
        Q = rng.standard_normal((s, 4))
        K = rng.standard_normal((s, 4))
        V = rng.standard_normal((s, 4))
        mask = causal_mask(s)
        _, w = scaled_dot_product_attention(Q, K, V, mask=mask)
        upper = np.triu(w, k=1)
        np.testing.assert_allclose(upper, 0.0, atol=1e-6)
