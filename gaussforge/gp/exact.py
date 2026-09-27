"""Exact GP regression, pure numpy/scipy (Tier-1 offline engine).

Math core: Rasmussen & Williams ch.2. Hyperparameters optimized in log-space
by L-BFGS-B on the negative log marginal likelihood, with jitter escalation
and seeded multi-restart. Deterministic given (seed, data, config).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

from gaussforge.core.errors import FitError
from gaussforge.kernels import MEAN_CATALOG, KernelModel, parse_spec


def _mean_values(X: np.ndarray, kind: str, coefs: np.ndarray | None = None):
    if kind == "zero":
        return np.zeros(X.shape[0]), None
    if kind == "const":
        if coefs is None:
            return None, None  # filled at fit time
        return np.full(X.shape[0], float(coefs[0])), coefs
    if kind == "linear":
        Xa = _aug(X)
        if coefs is None:
            return None, None
        return Xa @ coefs, coefs
    raise FitError(f"unknown mean kind: {kind!r}")


def _aug(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))])


class ExactGP:
    """Exact GP regressor on a parsed kernel spec. Scores larger predictions
    carry uncertainty through `predict(..., return_std=True)`."""

    def __init__(
        self,
        spec: str = "rbf",
        mean: str = "const",
        noise_init: float = 0.1,
        noise_bounds: tuple[float, float] = (1e-6, 25.0),
        n_restarts: int = 2,
        seed: int = 0,
        maxiter: int = 40,
    ) -> None:
        if mean not in MEAN_CATALOG:
            raise FitError(f"unknown mean {mean!r}")
        self.spec = spec
        self.mean_kind = mean
        self.kernel: KernelModel = parse_spec(spec)
        self.noise_init = float(noise_init)
        self.noise_bounds = noise_bounds
        self.n_restarts = int(n_restarts)
        self.seed = int(seed)
        self.maxiter = int(maxiter)
        self.X_train_: np.ndarray | None = None
        self.theta_: dict[str, float] | None = None
        self._noise: float | None = None
        self._alpha: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._mean_coefs: np.ndarray | None = None
        self.log_marginal_likelihood_: float | None = None

    # ---------------- mean helpers ----------------
    def _init_mean_coefs(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self.mean_kind == "const":
            return np.array([float(np.mean(y))])
        if self.mean_kind == "linear":
            Xa = _aug(X)
            coef, *_ = np.linalg.lstsq(Xa, y, rcond=None)
            return coef
        return np.zeros(0)

    def _mean_at(self, X: np.ndarray) -> np.ndarray:
        m, _ = _mean_values(X, self.mean_kind, self._mean_coefs)
        if m is None:  # const without coef -> handled in fit only
            raise FitError("mean coefficients not initialized")
        return m

    # ---------------- optimization ----------------
    def _pack(self, th: dict[str, float], noise: float) -> np.ndarray:
        keys = [p.key for p in self.kernel.free_params()]
        return np.log(np.array([th[k] for k in keys] + [noise]))

    def _unpack(self, z: np.ndarray) -> tuple[dict[str, float], float]:
        keys = [p.key for p in self.kernel.free_params()]
        vals = np.exp(z)
        th = {k: float(v) for k, v in zip(keys, vals[:-1])}
        return th, float(vals[-1])

    def _neg_lml(self, z: np.ndarray, X: np.ndarray, r: np.ndarray) -> float:
        th, noise = self._unpack(z)
        try:
            K = self.kernel.cov(X, X, th) + noise * np.eye(X.shape[0])
            L = _chol(K)
        except FitError:
            return 1e12
        a = cho_solve((L, True), r)
        nll = 0.5 * float(r @ a) + float(np.sum(np.log(np.diag(L)))) + 0.5 * len(r) * np.log(2 * np.pi)
        return nll

    def fit(self, X: np.ndarray, y: np.ndarray) -> ExactGP:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.shape[0] != y.shape[0] or X.shape[0] < 2:
            raise FitError(f"bad shapes X={X.shape} y={y.shape}")
        self._mean_coefs = self._init_mean_coefs(y, X)
        r = y - self._mean_at(X)
        free = self.kernel.free_params()
        bounds = [np.log(p.lo) for p in free] + [np.log(self.noise_bounds[0])], [
            np.log(p.hi) for p in free
        ] + [np.log(self.noise_bounds[1])]
        rng = np.random.default_rng(self.seed)
        best = (np.inf, None, None)
        for it in range(max(1, self.n_restarts)):
            if it == 0:
                z0 = np.log(
                    np.array([p.init for p in free] + [self.noise_init])
                )
            else:
                lo = np.log(np.array([p.lo for p in free]))
                hi = np.log(np.array([p.hi for p in free]))
                z0 = np.concatenate([
                    rng.uniform(lo, hi),
                    [np.log(rng.uniform(1e-4, 1.0))],
                ])
            try:
                res = minimize(
                    self._neg_lml,
                    z0,
                    args=(X, r),
                    method="L-BFGS-B",
                    bounds=list(zip(*bounds)),
                    options={"maxiter": self.maxiter},
                )
            except Exception as exc:  # pragma: no cover - numerical guard
                raise FitError(f"L-BFGS failed: {exc}") from exc
            if np.isfinite(res.fun) and res.fun < best[0]:
                best = (float(res.fun), res.x.copy(), None)
        if best[1] is None:
            raise FitError("all restarts failed to produce finite LML")
        th, noise = self._unpack(best[1])
        K = self.kernel.cov(X, X, th) + noise * np.eye(X.shape[0])
        L = _chol(K)
        alpha = cho_solve((L, True), r)
        self.X_train_ = X
        self.theta_ = th
        self._noise = noise
        self._L = L
        self._alpha = alpha
        self.log_marginal_likelihood_ = -best[0]
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self._alpha is None or self.X_train_ is None or self.theta_ is None:
            raise FitError("model is not fitted; call fit() first")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Ks = self.kernel.cov(self.X_train_, X, self.theta_)
        mean = Ks.T @ self._alpha + self._mean_at(X)
        if not return_std:
            return mean
        v = solve_triangular(self._L, Ks, lower=True)
        kss = self.kernel.diag_kss(self.theta_) + (self._noise or 0.0)
        var = np.maximum(kss - np.sum(v * v, axis=0), 1e-12)
        return mean, np.sqrt(var)


def _chol(K: np.ndarray) -> np.ndarray:
    jitter = 0.0
    base = float(np.mean(np.diag(K))) or 1.0
    for _ in range(6):
        try:
            return cholesky(K + jitter * np.eye(K.shape[0]), lower=True)
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, 1e-10 * base)
    raise FitError("kernel matrix not PSD even after jitter escalation")
