"""AutoKernel search tests."""

from __future__ import annotations

import pytest

from gaussforge.core.errors import ConfigError
from gaussforge.data.synthetic import make_regression_split
from gaussforge.kernels import MEAN_CATALOG, SPEC_CATALOG
from gaussforge.search.autokernel import search


def test_search_returns_valid_structure():
    sp = make_regression_split("smooth", 0)
    res = search(sp.X_train, sp.y_train, sp.X_val, sp.y_val, n_trials=3, seed=0, maxiter=10)
    assert res["spec"] in SPEC_CATALOG
    assert res["mean"] in MEAN_CATALOG
    assert res["val_score"] < 1e8


def test_search_deterministic():
    sp = make_regression_split("smooth", 1)
    r1 = search(sp.X_train, sp.y_train, sp.X_val, sp.y_val, n_trials=3, seed=2, maxiter=10)
    r2 = search(sp.X_train, sp.y_train, sp.X_val, sp.y_val, n_trials=3, seed=2, maxiter=10)
    assert r1 == r2


def test_search_invalid_objective_raises():
    sp = make_regression_split("smooth", 0)
    with pytest.raises(ConfigError):
        search(sp.X_train, sp.y_train, sp.X_val, sp.y_val, objective="auc")
