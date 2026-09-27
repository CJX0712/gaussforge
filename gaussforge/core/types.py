"""Shared typed containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Split:
    """Train/val/test split for one dataset."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    task: str = "regression"  # or "classification"

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])


@dataclass
class Row:
    """One benchmark cell: model x dataset x seed."""

    dataset: str
    model: str
    seed: int
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "seed": self.seed,
            **self.metrics,
            **self.extra,
        }
