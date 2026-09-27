"""Backend baseline tests + offline fallback path."""

from __future__ import annotations

import numpy as np

from gaussforge.backends.baselines import (
    NumpyKNN,
    NumpyLogistic,
    NumpyRidge,
    available_sklearn,
)


def test_numpy_ridge_recovers_linear():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(60, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0
    r = NumpyRidge().fit(X, y)
    pred = r.predict(X)
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.1


def test_numpy_knn_exact_on_duplicates():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0])
    knn = NumpyKNN(k=1).fit(X, y)
    assert np.allclose(knn.predict(X), y)


def test_numpy_logistic_separable():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    p = NumpyLogistic().fit(X, y).predict_proba(X)
    assert np.mean((p >= 0.5).astype(int) == y) > 0.9


def test_available_sklearn_true_in_env():
    assert available_sklearn() is True
