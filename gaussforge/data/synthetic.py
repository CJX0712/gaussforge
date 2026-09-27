"""Synthetic datasets + bundled real data (sklearn's offline registry).

Hole-style construction benchmarks (quasi_periodic / periodic_noisy): the
train/val regions exclude a deterministic x-interval while the test region
covers it. Fixed-RBF GP smooths across the hole; periodic-aware kernels
recover it. This creates honest headroom for structure search.
"""

from __future__ import annotations

import numpy as np

from gaussforge.core.errors import DataError
from gaussforge.core.types import Split

N_TRAIN, N_VAL, N_TEST = 150, 75, 75

_REG_GENERATORS = {"quasi_periodic", "periodic_noisy", "multiscale", "smooth"}
_HOLES = {"quasi_periodic": (6.0, 10.0), "periodic_noisy": (3.0, 7.0)}
_CLF_GENERATORS = {"blobs_overlap", "moons_overlap"}


def _x_with_hole(rng: np.random.Generator, lo: float, hi: float, hole: tuple[float, float]) -> np.ndarray:
    """Train x excludes the hole; val and test cover the full range so that
    structure selection sees hole-crossing performance like deployment."""
    x_full = rng.uniform(lo, hi, size=1200)
    outside = x_full[(x_full < hole[0]) | (x_full > hole[1])]
    x_train = outside[:N_TRAIN]
    x_val = rng.uniform(lo, hi, size=N_VAL)
    x_test = rng.uniform(lo, hi, size=N_TEST)
    return x_train, x_val, x_test


def _x_plain(rng: np.random.Generator, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.uniform(lo, hi, size=N_TRAIN + N_VAL + N_TEST)
    return x[:N_TRAIN], x[N_TRAIN : N_TRAIN + N_VAL], x[N_TRAIN + N_VAL :]


def _y_quasi_periodic(x: np.ndarray) -> np.ndarray:
    return 0.3 * x + 3.0 * np.sin(2.0 * np.pi * x / 4.0) + 0.3 * np.sin(2.0 * np.pi * x / 13.0)


def _y_periodic_noisy(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(2.0 * np.pi * x / 3.0)


def _y_multiscale(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(2.0 * np.pi * x / 2.0) * np.exp(-x / 8.0) + 0.5 * np.sin(2.0 * np.pi * x / 9.0)


def _y_smooth(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.exp(-(((x - 3.0) / 1.5) ** 2)) + 1.5 * np.sin(2.0 * np.pi * x / 5.0)


_GEN = {
    "quasi_periodic": (_y_quasi_periodic, (0.0, 20.0), 0.2),
    "periodic_noisy": (_y_periodic_noisy, (0.0, 12.0), 0.5),
    "multiscale": (_y_multiscale, (0.0, 10.0), 0.1),
    "smooth": (_y_smooth, (0.0, 10.0), 0.05),
}


def make_regression_split(name: str, seed: int) -> Split:
    if name == "diabetes":
        return _diabetes_split(seed)
    if name not in _GEN:
        raise DataError(f"unknown regression dataset {name!r}")
    fn, (lo, hi), noise_std = _GEN[name]
    rng = np.random.default_rng(seed)
    if name in _HOLES:
        x_train, x_val, x_test = _x_with_hole(rng, lo, hi, _HOLES[name])
    else:
        x_train, x_val, x_test = _x_plain(rng, lo, hi)
    mk = lambda x: fn(x) + rng.normal(0.0, noise_std, size=x.shape)
    col = lambda x: x[:, None]
    return Split(
        name=name,
        X_train=col(x_train),
        y_train=mk(x_train),
        X_val=col(x_val),
        y_val=mk(x_val),
        X_test=col(x_test),
        y_test=mk(x_test),
    )


def _diabetes_split(seed: int) -> Split:
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X, y = data.data.astype(np.float64), data.target.astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    y = (y - y.mean()) / y.std()
    n = 300
    idx = np.random.default_rng(seed).permutation(X.shape[0])[:n]
    X, y = X[idx], y[idx]
    return Split(
        name="diabetes",
        X_train=X[:N_TRAIN],
        y_train=y[:N_TRAIN],
        X_val=X[N_TRAIN : N_TRAIN + N_VAL],
        y_val=y[N_TRAIN : N_TRAIN + N_VAL],
        X_test=X[N_TRAIN + N_VAL :],
        y_test=y[N_TRAIN + N_VAL :],
    )


def make_classification_split(name: str, seed: int) -> Split:
    if name not in _CLF_GENERATORS:
        raise DataError(f"unknown classification dataset {name!r}")
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL + N_TEST
    if name == "blobs_overlap":
        c = np.array([-1.5, 1.5])
        y = rng.integers(0, 2, size=n)
        X = rng.normal(0.0, 1.0, size=(n, 2)) + c[y][:, None]
        X += rng.normal(0.0, 1.1, size=(n, 2))  # overlap injection
    else:  # moons_overlap
        t = rng.uniform(0.0, np.pi, size=n)
        half = rng.integers(0, 2, size=n)
        X = np.zeros((n, 2))
        m0 = half == 0
        m1 = ~m0
        X[m0, 0] = np.cos(t[m0])
        X[m0, 1] = np.sin(t[m0])
        X[m1, 0] = 1.0 - np.cos(t[m1])
        X[m1, 1] = 0.5 - np.sin(t[m1])
        X += rng.normal(0.0, 0.42, size=(n, 2))
        y = half.astype(np.float64)
    sl = lambda a: (
        a[:N_TRAIN],
        a[N_TRAIN : N_TRAIN + N_VAL],
        a[N_TRAIN + N_VAL :],
    )
    Xtr, Xva, Xte = sl(X)
    ytr, yva, yte = sl(y)
    return Split(
        name=name,
        X_train=Xtr,
        y_train=ytr.astype(int),
        X_val=Xva,
        y_val=yva.astype(int),
        X_test=Xte,
        y_test=yte.astype(int),
        task="classification",
    )


def available_datasets() -> dict[str, list[str]]:
    return {
        "regression": sorted(_REG_GENERATORS | {"diabetes"}),
        "classification": sorted(_CLF_GENERATORS),
    }
