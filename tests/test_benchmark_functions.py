# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Unit tests for benchmark functions and metrics in control scripts.

Tests analytical known-values at global minima, boundary conditions,
and cross-checks between Python and expected mathematical properties.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Benchmark functions: analytical known-values
# ---------------------------------------------------------------------------


class TestRastrigin:
    """Rastrigin 2D: f(0,0) = 0, always ≥ 0."""

    def test_global_minimum(self) -> None:
        from surrogate_validation import rastrigin_2d

        x = np.array([[0.0, 0.0]])
        assert rastrigin_2d(x) == pytest.approx(0.0, abs=1e-15)

    def test_nonnegative(self) -> None:
        from surrogate_validation import rastrigin_2d

        rng = np.random.default_rng(99)
        x = rng.uniform(-5.12, 5.12, (1000, 2))
        assert np.all(rastrigin_2d(x) >= 0.0)

    def test_symmetric(self) -> None:
        from surrogate_validation import rastrigin_2d

        x = np.array([[1.5, 2.3]])
        x_flip = np.array([[-1.5, -2.3]])
        np.testing.assert_allclose(rastrigin_2d(x), rastrigin_2d(x_flip), atol=1e-14)

    def test_known_value(self) -> None:
        from surrogate_validation import rastrigin_2d

        x = np.array([[1.0, 1.0]])
        expected = 20 + 1 - 10 * np.cos(2 * np.pi) + 1 - 10 * np.cos(2 * np.pi)
        assert rastrigin_2d(x) == pytest.approx(expected, abs=1e-14)


class TestRosenbrock:
    """Rosenbrock 2D: f(1,1) = 0."""

    def test_global_minimum(self) -> None:
        from surrogate_validation import rosenbrock_2d

        x = np.array([[1.0, 1.0]])
        assert rosenbrock_2d(x) == pytest.approx(0.0, abs=1e-15)

    def test_known_value(self) -> None:
        from surrogate_validation import rosenbrock_2d

        x = np.array([[0.0, 0.0]])
        assert rosenbrock_2d(x) == pytest.approx(1.0, abs=1e-15)

    def test_nonnegative(self) -> None:
        from surrogate_validation import rosenbrock_2d

        rng = np.random.default_rng(99)
        x = rng.uniform(-5, 10, (1000, 2))
        assert np.all(rosenbrock_2d(x) >= 0.0)


class TestAckley:
    """Ackley 2D: f(0,0) = 0."""

    def test_global_minimum(self) -> None:
        from surrogate_validation import ackley_2d

        x = np.array([[0.0, 0.0]])
        assert ackley_2d(x) == pytest.approx(0.0, abs=1e-14)

    def test_positive_away_from_origin(self) -> None:
        from surrogate_validation import ackley_2d

        rng = np.random.default_rng(99)
        x = rng.uniform(0.5, 5.0, (500, 2))
        assert np.all(ackley_2d(x) > 0.0)


# ---------------------------------------------------------------------------
# Metrics: known-value unit tests
# ---------------------------------------------------------------------------


class TestComputeR2:
    def test_perfect(self) -> None:
        from surrogate_validation import compute_r2

        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_r2(y, y) == pytest.approx(1.0)

    def test_mean_prediction(self) -> None:
        from surrogate_validation import compute_r2

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full(5, 3.0)
        assert compute_r2(y_true, y_pred) == pytest.approx(0.0)

    def test_negative_r2(self) -> None:
        from surrogate_validation import compute_r2

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert compute_r2(y_true, y_pred) < 0.0


class TestComputeRMSE:
    def test_zero_error(self) -> None:
        from surrogate_validation import compute_rmse

        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        from surrogate_validation import compute_rmse

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        assert compute_rmse(y_true, y_pred) == pytest.approx(0.1, abs=1e-10)

    def test_order_independent_magnitude(self) -> None:
        from surrogate_validation import compute_rmse

        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        expected = np.sqrt((9 + 16) / 2)
        assert compute_rmse(y_true, y_pred) == pytest.approx(expected, abs=1e-10)


class TestComputeMAE:
    def test_zero_error(self) -> None:
        from surrogate_validation import compute_mae

        y = np.array([1.0, 2.0, 3.0])
        assert compute_mae(y, y) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        from surrogate_validation import compute_mae

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert compute_mae(y_true, y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------


class TestCreateSequences:
    def test_output_shapes(self) -> None:
        from sequence_forecasting import create_sequences

        data = np.arange(100, dtype=float)
        X, y = create_sequences(data, seq_len=10, horizon=1)
        assert X.shape[1] == 10
        assert len(X) == len(y)
        assert len(X) == 100 - 10

    def test_first_target(self) -> None:
        from sequence_forecasting import create_sequences

        data = np.arange(50, dtype=float)
        X, y = create_sequences(data, seq_len=5, horizon=1)
        np.testing.assert_array_equal(X[0], [0, 1, 2, 3, 4])
        assert y[0] == 5.0

    def test_horizon_offset(self) -> None:
        from sequence_forecasting import create_sequences

        data = np.arange(50, dtype=float)
        X, y = create_sequences(data, seq_len=5, horizon=3)
        assert y[0] == 7.0


class TestPersistenceForecast:
    def test_last_value_persisted(self) -> None:
        from sequence_forecasting import persistence_forecast

        X = np.array(
            [[[1, 0.5], [2, 0.6], [3, 0.7]], [[4, 0.8], [5, 0.9], [6, 1.0]]],
            dtype=float,
        )
        pred = persistence_forecast(X, horizon=1)
        np.testing.assert_array_equal(pred, [3, 6])


class TestMichiganWeather:
    def test_output_keys(self) -> None:
        from sequence_forecasting import generate_michigan_weather

        w = generate_michigan_weather(100, seed=0)
        assert "tmax" in w
        assert "tmin" in w
        assert "precip" in w
        assert len(w["tmax"]) == 100

    def test_physical_bounds(self) -> None:
        from sequence_forecasting import generate_michigan_weather

        w = generate_michigan_weather(1000, seed=42)
        assert np.all(w["tmax"] > -50)
        assert np.all(w["tmax"] < 60)
        assert np.all(w["tmin"] <= w["tmax"])
        assert np.all(w["precip"] >= 0)
