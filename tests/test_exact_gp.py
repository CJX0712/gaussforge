"""ExactGP tests: fit/predict contract, determinism, interpolation, invariants."""

from __future__ import annotations

import numpy as np
import pytest

from gaussforge.core.errors import FitError
from gaussforge.data.synthetic import make_regression_split
from gaussforge.gp.exact import ExactGP
from gaussforge.kernels import MEAN_CATALOG, SPEC_CATALOG


def test_fit_predict_shapes_and_nonnegative_var():
    sp = make_regression_split("smooth", 0)
    gp = ExactGP(spec="rbf", mean="const", n_restarts=1, seed=0, maxiter=20).fit(
        sp.X_train, sp.y_train
    )
    m, s = gp.predict(sp.X_test)
    assert m.shape == (sp.X_test.shape[0],)
    assert s.shape == m.shape
    assert np.all(s > 0)
    assert np.isfinite(gp.log_marginal_likelihood_)


def test_predict_without_fit_raises():
    gp = ExactGP()
    with pytest.raises(FitError):
        gp.predict(np.zeros((3, 1)))


def test_interpolation_low_noise():
    # noise-free targets + tiny noise bound => near-perfect train fit
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 6, size=(40, 1))
    y = np.sin(2 * np.pi * X[:, 0] / 2.0)
    gp = ExactGP(spec="rbf", mean="const", noise_init=1e-3, n_restarts=1, seed=0, maxiter=30)
    gp.fit(X, y)
    m, _ = gp.predict(X)
    assert np.sqrt(np.mean((m - y) ** 2)) < 0.05


def test_determinism_same_seed():
    sp = make_regression_split("multiscale", 0)
    r = []
    for _ in range(2):
        gp = ExactGP(spec="rq", mean="const", n_restarts=2, seed=7, maxiter=20).fit(
            sp.X_train, sp.y_train
        )
        m, s = gp.predict(sp.X_test)
        r.append((m.copy(), s.copy(), gp.theta_.copy()))
    assert np.array_equal(r[0][0], r[1][0])
    assert np.array_equal(r[0][1], r[1][1])
    assert r[0][2] == r[1][2]


def test_all_mean_kinds_fit():
    sp = make_regression_split("smooth", 1)
    for mean in MEAN_CATALOG:
        gp = ExactGP(spec="rbf", mean=mean, n_restarts=1, seed=0, maxiter=15)
        gp.fit(sp.X_train, sp.y_train)
        m, s = gp.predict(sp.X_val)
        assert np.isfinite(m).all() and np.isfinite(s).all()


def test_all_specs_fit_on_small_data():
    sp = make_regression_split("smooth", 2)
    Xs, ys = sp.X_train[:40], sp.y_train[:40]
    for spec in SPEC_CATALOG:
        gp = ExactGP(spec=spec, mean="const", n_restarts=1, seed=0, maxiter=10)
        gp.fit(Xs, ys)
        m, _s = gp.predict(sp.X_val)
        assert np.isfinite(m).all(), spec


def test_unknown_mean_raises():
    with pytest.raises(FitError):
        ExactGP(mean="cubic")


def test_bad_shapes_raise():
    gp = ExactGP()
    with pytest.raises(FitError):
        gp.fit(np.zeros((3, 1)), np.zeros(4))
    with pytest.raises(FitError):
        gp.fit(np.zeros((1, 1)), np.zeros(1))
