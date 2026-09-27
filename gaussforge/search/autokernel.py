"""AutoKernel: Optuna TPE search over the kernel DSL + mean functions.

Objective is the validation-set score of a fully fitted exact GP.
`nll` (default) selects for calibrated uncertainty; `rmse` is the ablation arm.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna

from gaussforge.core.errors import ConfigError
from gaussforge.eval.metrics import gaussian_nll, rmse
from gaussforge.gp.exact import ExactGP
from gaussforge.kernels import MEAN_CATALOG, SPEC_CATALOG

optuna.logging.set_verbosity(optuna.logging.WARNING)


def search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    objective: str = "nll",
    n_trials: int = 12,
    seed: int = 0,
    maxiter: int = 25,
) -> dict[str, Any]:
    if objective not in ("nll", "rmse"):
        raise ConfigError(f"unknown search objective {objective!r}")
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _objective(trial: optuna.Trial) -> float:
        spec = trial.suggest_categorical("spec", list(SPEC_CATALOG))
        mean = trial.suggest_categorical("mean", list(MEAN_CATALOG))
        gp = ExactGP(
            spec=spec,
            mean=mean,
            n_restarts=1,
            seed=seed * 1000 + trial.number,
            maxiter=maxiter,
        )
        try:
            gp.fit(X_train, y_train)
            m, s = gp.predict(X_val)
            if objective == "nll":
                return float(gaussian_nll(y_val, m, s * s))
            return float(rmse(y_val, m))
        except Exception:  # noqa: BLE001 - a broken trial must not kill the study
            return 1e9

    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    return {
        "spec": str(best.params["spec"]),
        "mean": str(best.params["mean"]),
        "val_score": float(best.value),
    }
