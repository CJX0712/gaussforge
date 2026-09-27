"""Baseline backends with availability probing.

Tier-0 (SOTA references, optional): scikit-learn GPR / GPC / RandomForest /
LogisticRegression. Tier-1 (always available): pure numpy Ridge / k-NN /
logistic IRLS. `available_sklearn()` is monkeypatch-friendly for tests.
"""

from __future__ import annotations

import numpy as np

from gaussforge.core.errors import BackendUnavailableError

_SKLEARN_STATE: bool | None = None


def available_sklearn() -> bool:
    global _SKLEARN_STATE
    if _SKLEARN_STATE is None:
        try:
            import sklearn  # noqa: F401

            _SKLEARN_STATE = True
        except Exception:  # noqa: BLE001 - import probe must catch all failures
            _SKLEARN_STATE = False
    return _SKLEARN_STATE


def _force_sklearn(state: bool) -> None:
    """Test hook: override probing result."""
    global _SKLEARN_STATE
    _SKLEARN_STATE = state


class NumpyRidge:
    """Closed-form ridge on standardized features (lambda = 1.0)."""

    def __init__(self, lam: float = 1.0) -> None:
        self.lam = float(lam)
        self.coef_: np.ndarray | None = None
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> NumpyRidge:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0) + 1e-12
        Z = (X - self.mu_) / self.sigma_
        self.intercept_ = float(y.mean())
        r = y - self.intercept_
        A = Z.T @ Z + self.lam * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ r)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Z = (X - self.mu_) / self.sigma_
        out = Z @ self.coef_ + self.intercept_
        if return_std:
            # homoscedastic residual std estimate is not tracked; return zeros
            return out, np.zeros(X.shape[0])
        return out


class NumpyKNN:
    """k-nearest-neighbour mean regressor, Euclidean."""

    def __init__(self, k: int = 5) -> None:
        self.k = int(k)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NumpyKNN:
        self.X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self.y = np.asarray(y, dtype=np.float64).ravel()
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        d2 = (
            np.sum(X * X, axis=1)[:, None]
            + np.sum(self.X * self.X, axis=1)[None, :]
            - 2.0 * (X @ self.X.T)
        )
        idx = np.argpartition(d2, kth=min(self.k, self.X.shape[0] - 1), axis=1)[
            :, : self.k
        ]
        out = self.y[idx].mean(axis=1)
        if return_std:
            std = self.y[idx].std(axis=1) + 1e-6
            return out, std
        return out


class NumpyLogistic:
    """Binary logistic regression via IRLS (Tier-1 fallback)."""

    def __init__(self, lam: float = 1e-3, max_iter: int = 50) -> None:
        self.lam = float(lam)
        self.max_iter = int(max_iter)
        self.w_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NumpyLogistic:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])
        y = np.asarray(y, dtype=np.float64).ravel()
        w = np.zeros(Xa.shape[1])
        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(Xa @ w, -500, 500)))
            W = np.maximum(p * (1.0 - p), 1e-8)
            H = Xa.T @ (Xa * W[:, None]) + self.lam * np.eye(Xa.shape[1])
            g = Xa.T @ (y - p) - self.lam * w
            step = np.linalg.solve(H, g)
            w = w + step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.w_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])
        return 1.0 / (1.0 + np.exp(-np.clip(Xa @ self.w_, -500, 500)))


def _require_sklearn(feature: str) -> None:
    if not available_sklearn():
        raise BackendUnavailableError(f"scikit-learn unavailable: {feature} skipped")


def make_sklearn_gpr(seed: int):
    _require_sklearn("sklearn-gp-rbf")
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=seed,
    )


def make_sklearn_rf(seed: int):
    _require_sklearn("sklearn-rf")
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)


def make_sklearn_gpc(seed: int):
    _require_sklearn("sklearn-gpc")
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF

    return GaussianProcessClassifier(kernel=RBF(1.0), random_state=seed)
