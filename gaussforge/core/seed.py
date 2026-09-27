"""Global determinism: single seed entry point."""

from __future__ import annotations

import random

import numpy as np


def set_all(seed: int) -> None:
    """Seed python.random and numpy once. All stochastic parts derive from this."""
    random.seed(seed)
    np.random.seed(seed)
