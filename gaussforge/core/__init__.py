"""Core primitives shared across modules."""

from gaussforge.core.config import GaussConfig
from gaussforge.core.errors import (
    BackendUnavailableError,
    ConfigError,
    DataError,
    FitError,
    GaussForgeError,
    RuntimeFailure,
)
from gaussforge.core.seed import set_all
from gaussforge.core.types import Row, Split

__all__ = [
    "BackendUnavailableError",
    "ConfigError",
    "DataError",
    "FitError",
    "GaussConfig",
    "GaussForgeError",
    "Row",
    "RuntimeFailure",
    "Split",
    "set_all",
]
