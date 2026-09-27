"""Metrics tests: closed-form anchors and bounds."""

from __future__ import annotations

import numpy as np
import pytest

from gaussforge.core.errors import DataError
from gaussforge.eval.metrics import (
    accuracy,
    coverage90,
    ece,
    gaussian_nll,
    log_loss,
    rmse,
)


def test_rmse_known_value():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert abs(rmse(y, p) - np.sqrt(4.0 / 3.0)) < 1e-12


def test_rmse_shape_mismatch_raises():
    with pytest.raises(DataError):
        rmse(np.zeros(3), np.zeros(4))


def test_gaussian_nll_anchor():
    y = np.array([0.5, -0.5])
    nll = gaussian_nll(y, y, np.ones(2))
    assert abs(nll - 0.5 * np.log(2 * np.pi)) < 1e-12


def test_gaussian_nll_rejects_nonpositive_var():
    with pytest.raises(DataError):
        gaussian_nll(np.zeros(2), np.zeros(2), np.array([1.0, 0.0]))


def test_coverage90_perfect_std():
    y = np.array([0.0, 1.0])
    assert coverage90(y, y, np.full(2, 1.0)) == 1.0


def test_accuracy_log_loss_ece_bounds():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.2, 0.9, 0.6, 0.4])
    assert accuracy(y, p) == 1.0
    assert log_loss(y, p) > 0
    e = ece(y, p)
    assert 0.0 <= e <= 1.0


def test_ece_perfect_calibration_small():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=500)
    y = (rng.uniform(0, 1, size=500) < p).astype(float)
    assert ece(y, p) < 0.15
