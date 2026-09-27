"""Kernel DSL: primitive covariance terms and composition (Sum / Product).

Design:
- Each primitive exposes `params()`: (key, init, lo, hi) in natural space.
  Optimization happens in log-space (done by gp.exact).
- cov() receives both squared-distance (D2) and distance (D) matrices so the
  same DSL works for 1-D and multi-D inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEC_CATALOG: tuple[str, ...] = (
    "rbf",
    "rq",
    "matern52",
    "periodic",
    "rbf+periodic",
    "rq+periodic",
    "matern52+periodic",
    "rbf*periodic",
    "rq*periodic",
)
MEAN_CATALOG: tuple[str, ...] = ("zero", "const", "linear")


@dataclass(frozen=True)
class _Param:
    key: str
    init: float
    lo: float
    hi: float


class _Term:
    kind: str = ""

    def params(self, prefix: str) -> list[_Param]:  # pragma: no cover - interface
        raise NotImplementedError

    def cov(self, D2: np.ndarray, D: np.ndarray, th: dict[str, float]) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover

    def diag(self, th: dict[str, float]) -> np.ndarray | float:
        """Self-covariance k(x, x) as scalar (same for all x)."""
        raise NotImplementedError  # pragma: no cover


class _RBF(_Term):
    kind = "rbf"

    def params(self, prefix: str) -> list[_Param]:
        return [
            _Param(f"{prefix}.ls", 1.0, 0.05, 60.0),
            _Param(f"{prefix}.var", 1.0, 0.01, 100.0),
        ]

    def cov(self, D2, D, th):
        return th[f"{self.kind}.var"] * np.exp(-0.5 * D2 / th[f"{self.kind}.ls"] ** 2)

    def diag(self, th):
        return th[f"{self.kind}.var"]


class _RQ(_Term):
    kind = "rq"

    def params(self, prefix: str) -> list[_Param]:
        return [
            _Param(f"{self.kind}.ls", 1.0, 0.05, 60.0),
            _Param(f"{self.kind}.alpha", 1.0, 0.1, 50.0),
            _Param(f"{self.kind}.var", 1.0, 0.01, 100.0),
        ]

    def cov(self, D2, D, th):
        ls, a, v = th[f"{self.kind}.ls"], th[f"{self.kind}.alpha"], th[f"{self.kind}.var"]
        return v * (1.0 + 0.5 * D2 / (a * ls**2)) ** (-a)

    def diag(self, th):
        return th[f"{self.kind}.var"]


class _Matern52(_Term):
    kind = "matern52"

    def params(self, prefix: str) -> list[_Param]:
        return [
            _Param(f"{self.kind}.ls", 1.0, 0.05, 60.0),
            _Param(f"{self.kind}.var", 1.0, 0.01, 100.0),
        ]

    def cov(self, D2, D, th):
        ls, v = th[f"{self.kind}.ls"], th[f"{self.kind}.var"]
        r = np.sqrt(5.0) * D / ls
        return v * (1.0 + r + r * r / 3.0) * np.exp(-r)

    def diag(self, th):
        return th[f"{self.kind}.var"]


class _Periodic(_Term):
    kind = "periodic"

    def params(self, prefix: str) -> list[_Param]:
        return [
            _Param(f"{self.kind}.ls", 1.0, 0.05, 60.0),
            _Param(f"{self.kind}.p", 2.0, 0.1, 80.0),
            _Param(f"{self.kind}.var", 1.0, 0.01, 100.0),
        ]

    def cov(self, D2, D, th):
        ls, p, v = th[f"{self.kind}.ls"], th[f"{self.kind}.p"], th[f"{self.kind}.var"]
        return v * np.exp(-2.0 * np.sin(np.pi * D / p) ** 2 / ls**2)

    def diag(self, th):
        return th[f"{self.kind}.var"]


_TERM_KINDS: dict[str, type[_Term]] = {
    t.kind: t for t in (_RBF, _RQ, _Matern52, _Periodic)
}


class KernelModel:
    """Parsed kernel structure. spec like 'rbf+periodic' or 'rbf*periodic'."""

    def __init__(self, spec: str) -> None:
        spec = spec.strip()
        if "*" in spec and "+" in spec:
            raise ValueError(f"mixed +- spec not supported: {spec!r}")
        if "*" in spec:
            parts, op = spec.split("*"), "product"
        elif "+" in spec:
            parts, op = spec.split("+"), "sum"
        else:
            parts, op = [spec], "sum"
        self.spec = spec
        self.op = op
        self.terms: list[tuple[str, _Term]] = []
        counts: dict[str, int] = {}
        for raw in parts:
            kind = raw.strip()
            if kind not in _TERM_KINDS:
                raise ValueError(f"unknown kernel kind {kind!r} in spec {spec!r}")
            idx = counts.get(kind, 0)
            counts[kind] = idx + 1
            prefix = kind if idx == 0 else f"{kind}{idx}"
            self.terms.append((prefix, _TERM_KINDS[kind]()))
        if not self.terms:
            raise ValueError(f"empty spec: {spec!r}")

    def free_params(self) -> list[_Param]:
        out: list[_Param] = []
        for prefix, term in self.terms:
            out.extend(term.params(prefix))
        return out

    def cov(self, X1: np.ndarray, X2: np.ndarray, th: dict[str, float]) -> np.ndarray:
        X1 = np.atleast_2d(np.asarray(X1, dtype=np.float64))
        X2 = np.atleast_2d(np.asarray(X2, dtype=np.float64))
        d2 = _sq_dists(X1, X2)
        d = np.sqrt(d2)
        covs = [term.cov(d2, d, th) for _, term in self.terms]
        if self.op == "product":
            k = np.ones_like(covs[0])
            for c in covs:
                k = k * c
            return k
        return sum(covs[1:], covs[0])

    def diag_kss(self, th: dict[str, float]) -> float:
        vals = [term.diag(th) for _, term in self.terms]
        if self.op == "product":
            out = 1.0
            for v in vals:
                out *= float(v)
            return out
        return float(sum(vals))


def _sq_dists(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    a2 = np.sum(X1 * X1, axis=1)[:, None]
    b2 = np.sum(X2 * X2, axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * (X1 @ X2.T)
    return np.maximum(d2, 0.0)


def parse_spec(spec: str) -> KernelModel:
    return KernelModel(spec)
