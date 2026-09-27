"""Binary Gaussian Process classification via Laplace approximation (numpy).

Rasmussen & Williams Alg 3.1/3.2: Newton iterations on the posterior mode,
probit-approximated predictive probabilities. Pure numpy, zero downloads.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve

from gaussforge.core.errors import FitError
from gaussforge.gp.exact import _chol
from gaussforge.kernels import parse_spec


def _sigmoid(x: np.ndarray) -> np.ndarray:
    xc = np.clip(x, -500, 500)
    return np.where(xc >= 0, 1.0 / (1.0 + np.exp(-xc)), np.exp(xc) / (1.0 + np.exp(xc)))


class LaplaceGPC:
    """Binary GP classifier, Laplace approximation, fixed kernel theta defaults."""

    def __init__(self, spec: str = "rbf", seed: int = 0, max_iter: int = 25) -> None:
        self.kernel = parse_spec(spec)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.X_train_: np.ndarray | None = None
        self.theta_: dict[str, float] | None = None
        self._f_hat: np.ndarray | None = None
        self._a: np.ndarray | None = None  # y - p at mode
        self._sw: np.ndarray | None = None
        self._Lm: np.ndarray | None = None  # chol of M = I + sqrt(W) K sqrt(W)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LaplaceGPC:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        if set(np.unique(y)) - {0.0, 1.0}:
            raise FitError("LaplaceGPC supports binary labels in {0, 1}")
        th = self._default_theta()
        K = self.kernel.cov(X, X, th)
        _chol(K)  # PSD validation before Newton iterations
        f = np.zeros(X.shape[0])
        p = _sigmoid(f)
        for _ in range(self.max_iter):
            w = np.maximum(p * (1.0 - p), 1e-8)
            sw = np.sqrt(w)
            b = sw * (f + K @ (y - p))
            M = np.eye(X.shape[0]) + K * np.outer(sw, sw)
            Lm = _chol(M)
            z = cho_solve((Lm, True), b)
            f_new = z / sw
            p = _sigmoid(f_new)
            if float(np.max(np.abs(f_new - f))) < 1e-8:
                f = f_new
                break
            f = f_new
        w = np.maximum(p * (1.0 - p), 1e-8)
        self.X_train_ = X
        self.theta_ = th
        self._f_hat = f
        self._a = y - p
        self._sw = np.sqrt(w)
        self._Lm = _chol(np.eye(X.shape[0]) + K * np.outer(self._sw, self._sw))
        return self

    def _default_theta(self) -> dict[str, float]:
        return {p.key: p.init for p in self.kernel.free_params()}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if any(
            v is None
            for v in (self._f_hat, self._a, self._sw, self._Lm, self.X_train_, self.theta_)
        ):
            raise FitError("model is not fitted; call fit() first")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Ks = self.kernel.cov(self.X_train_, X, self.theta_)  # (n_train, m)
        m_star = Ks.T @ self._a  # latent posterior mean approx
        v = solve_lower(self._Lm, self._sw[:, None] * Ks)
        kss = self.kernel.diag_kss(self.theta_ or {})
        var = np.maximum(kss - np.sum(v * v, axis=0), 1e-12)
        return _sigmoid(m_star / np.sqrt(1.0 + var))


def solve_lower(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    import scipy.linalg as sla

    return sla.solve_triangular(L, B, lower=True)
