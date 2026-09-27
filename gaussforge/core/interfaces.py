"""Protocol interfaces for pluggable backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RegressorLike(Protocol):
    """Contract: fit then predict; predict returns (mean, std) with std >= 0."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegressorLike: ...

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class ClassifierLike(Protocol):
    """Contract: fit then predict_proba returning probs in [0, 1]."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClassifierLike: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
