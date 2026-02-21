# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Determinism tests: verify that control scripts produce identical results
across runs with the same seed.

These tests import the data-generation and metric functions directly and
verify that outputs are bitwise identical when given the same seed.
"""

import numpy as np
import pytest


class TestSurrogateDeterminism:
    """Surrogate benchmark functions are pure math — zero tolerance."""

    def test_rastrigin_deterministic(self) -> None:
        from surrogate_validation import rastrigin_2d

        rng1 = np.random.default_rng(42)
        x1 = np.column_stack([rng1.uniform(-5.12, 5.12, 100), rng1.uniform(-5.12, 5.12, 100)])
        y1 = rastrigin_2d(x1)

        rng2 = np.random.default_rng(42)
        x2 = np.column_stack([rng2.uniform(-5.12, 5.12, 100), rng2.uniform(-5.12, 5.12, 100)])
        y2 = rastrigin_2d(x2)

        np.testing.assert_array_equal(y1, y2)

    def test_rosenbrock_deterministic(self) -> None:
        from surrogate_validation import rosenbrock_2d

        rng1 = np.random.default_rng(42)
        x1 = np.column_stack([rng1.uniform(-5, 10, 100), rng1.uniform(-5, 10, 100)])
        y1 = rosenbrock_2d(x1)

        rng2 = np.random.default_rng(42)
        x2 = np.column_stack([rng2.uniform(-5, 10, 100), rng2.uniform(-5, 10, 100)])
        y2 = rosenbrock_2d(x2)

        np.testing.assert_array_equal(y1, y2)


class TestSequenceDeterminism:
    """Synthetic weather generation must be deterministic."""

    def test_michigan_weather_deterministic(self) -> None:
        from sequence_forecasting import generate_michigan_weather

        w1 = generate_michigan_weather(365, seed=42)
        w2 = generate_michigan_weather(365, seed=42)

        np.testing.assert_array_equal(w1["tmax"], w2["tmax"])
        np.testing.assert_array_equal(w1["tmin"], w2["tmin"])
        np.testing.assert_array_equal(w1["precip"], w2["precip"])


class TestTransformerDeterminism:
    """NumPy transformer ops are deterministic for same seed + input."""

    def test_softmax_deterministic(self) -> None:
        from transformer_inference import softmax

        rng = np.random.default_rng(42)
        x = rng.standard_normal((8, 32))
        s1 = softmax(x, axis=-1)
        s2 = softmax(x, axis=-1)
        np.testing.assert_array_equal(s1, s2)

    def test_sdpa_deterministic(self) -> None:
        from transformer_inference import scaled_dot_product_attention

        rng = np.random.default_rng(42)
        q = rng.standard_normal((8, 8))
        k = rng.standard_normal((8, 8))
        v = rng.standard_normal((8, 8))
        o1, w1 = scaled_dot_product_attention(q, k, v)
        o2, w2 = scaled_dot_product_attention(q, k, v)
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(w1, w2)


class TestMetricsDeterminism:
    """Metric functions are pure math."""

    def test_r2_known_value(self) -> None:
        from surrogate_validation import compute_r2

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_r2(y_true, y_pred) == pytest.approx(1.0)

    def test_r2_mean_gives_zero(self) -> None:
        from surrogate_validation import compute_r2

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full(5, 3.0)
        assert compute_r2(y_true, y_pred) == pytest.approx(0.0)
