"""LaplaceGPC tests."""

from __future__ import annotations

import numpy as np
import pytest

from gaussforge.core.errors import FitError
from gaussforge.data.synthetic import make_classification_split
from gaussforge.gp.classify import LaplaceGPC


def test_blobs_accuracy_and_proba_bounds():
    sp = make_classification_split("blobs_overlap", 0)
    gpc = LaplaceGPC(seed=0).fit(sp.X_train, sp.y_train)
    p = gpc.predict_proba(sp.X_test)
    assert p.min() > 0.0 and p.max() < 1.0
    acc = float(np.mean((p >= 0.5).astype(int) == sp.y_test))
    assert acc >= 0.85


def test_moons_accuracy():
    sp = make_classification_split("moons_overlap", 0)
    p = LaplaceGPC(seed=0).fit(sp.X_train, sp.y_train).predict_proba(sp.X_test)
    acc = float(np.mean((p >= 0.5).astype(int) == sp.y_test))
    assert acc >= 0.75


def test_determinism():
    sp = make_classification_split("blobs_overlap", 1)
    p1 = LaplaceGPC(seed=3).fit(sp.X_train, sp.y_train).predict_proba(sp.X_test)
    p2 = LaplaceGPC(seed=3).fit(sp.X_train, sp.y_train).predict_proba(sp.X_test)
    assert np.array_equal(p1, p2)


def test_predict_before_fit_raises():
    with pytest.raises(FitError):
        LaplaceGPC().predict_proba(np.zeros((2, 2)))


def test_non_binary_labels_raise():
    sp = make_classification_split("blobs_overlap", 0)
    y = sp.y_train.copy()
    y[0] = 5
    with pytest.raises(FitError):
        LaplaceGPC().fit(sp.X_train, y)
