"""Data module exports."""

from gaussforge.data.synthetic import (
    available_datasets,
    make_classification_split,
    make_regression_split,
)

__all__ = [
    "available_datasets",
    "make_classification_split",
    "make_regression_split",
]
