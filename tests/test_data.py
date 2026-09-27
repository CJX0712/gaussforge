"""Data generator tests: shapes, determinism, validity."""

from __future__ import annotations

import numpy as np
import pytest

from gaussforge.core.errors import DataError
from gaussforge.data.synthetic import (
    available_datasets,
    make_classification_split,
    make_regression_split,
)

N_TRAIN, N_VAL, N_TEST = 150, 75, 75


@pytest.mark.parametrize(
    "name",
    ["quasi_periodic", "periodic_noisy", "multiscale", "smooth", "diabetes"],
)
def test_regression_split_shapes(name):
    sp = make_regression_split(name, 0)
    assert sp.X_train.shape == (N_TRAIN, sp.X_train.shape[1])
    assert sp.X_val.shape == (N_VAL, sp.X_train.shape[1])
    assert sp.X_test.shape == (N_TEST, sp.X_train.shape[1])
    assert sp.y_train.shape == (N_TRAIN,)
    assert np.isfinite(sp.X_train).all() and np.isfinite(sp.y_train).all()


@pytest.mark.parametrize("name", ["blobs_overlap", "moons_overlap"])
def test_classification_split_validity(name):
    sp = make_classification_split(name, 0)
    assert sp.task == "classification"
    assert set(np.unique(sp.y_train)) <= {0, 1}
    assert sp.X_train.shape[1] == 2
    assert len(sp.y_test) == N_TEST


def test_split_determinism():
    a = make_regression_split("quasi_periodic", 5)
    b = make_regression_split("quasi_periodic", 5)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.y_test, b.y_test)
    c = make_regression_split("quasi_periodic", 6)
    assert not np.array_equal(a.X_train, c.X_train)


def test_hole_datasets_train_excludes_hole():
    sp = make_regression_split("quasi_periodic", 0)
    in_hole = (sp.X_train[:, 0] >= 6.0) & (sp.X_train[:, 0] <= 10.0)
    assert not np.any(in_hole)
    # val and test cover the hole region (deployment distribution)
    assert np.any((sp.X_val[:, 0] >= 6.0) & (sp.X_val[:, 0] <= 10.0))


def test_unknown_dataset_raises():
    with pytest.raises(DataError):
        make_regression_split("unknown", 0)
    with pytest.raises(DataError):
        make_classification_split("unknown", 0)


def test_available_datasets_registry():
    reg = available_datasets()
    assert "diabetes" in reg["regression"]
    assert "moons_overlap" in reg["classification"]
