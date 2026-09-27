"""Kernel DSL tests: PSD, symmetry, periodicity, spec parsing."""

from __future__ import annotations

import numpy as np
import pytest

from gaussforge.kernels import SPEC_CATALOG, KernelModel, parse_spec


def _th(km: KernelModel) -> dict[str, float]:
    return {p.key: p.init for p in km.free_params()}


def test_all_specs_psd_and_symmetric():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(25, 1))
    for spec in SPEC_CATALOG:
        km = parse_spec(spec)
        K = km.cov(X, X, _th(km))
        assert K.shape == (25, 25)
        assert np.allclose(K, K.T, atol=1e-12), spec
        eig = np.linalg.eigvalsh(K)
        assert eig.min() > -1e-8, (spec, eig.min())


def test_diag_matches_matrix_diagonal():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(10, 1))
    for spec in SPEC_CATALOG:
        km = parse_spec(spec)
        K = km.cov(X, X, _th(km))
        assert abs(np.diag(K).mean() - km.diag_kss(_th(km))) < 1e-9, spec


def test_periodic_kernel_repeats_with_period():
    km = parse_spec("periodic")
    th = {p.key: p.init for p in km.free_params()}
    th["periodic.p"] = 2.0
    a = np.array([[0.0]])
    b = np.array([[4.0]])  # exactly two periods away
    k = km.cov(a, b, th)[0, 0]
    assert abs(k - th["periodic.var"]) < 1e-10


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        parse_spec("nope")
    with pytest.raises(ValueError):
        parse_spec("")


def test_mixed_spec_raises():
    with pytest.raises(ValueError):
        parse_spec("rbf+periodic*rq")


def test_product_vs_sum_different():
    X = np.linspace(0, 1, 8)[:, None]
    th_sum = _th(parse_spec("rbf+periodic"))
    th_prod = _th(parse_spec("rbf*periodic"))
    ks = parse_spec("rbf+periodic").cov(X, X, th_sum)
    kp = parse_spec("rbf*periodic").cov(X, X, th_prod)
    assert not np.allclose(ks, kp)
