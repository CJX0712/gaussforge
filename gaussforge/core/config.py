"""Configuration with ENV override support (ENV_GF_*)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from .errors import ConfigError


@dataclass(frozen=True)
class GaussConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    search_seed: int = 0
    n_trials: int = 10
    search_maxiter: int = 25
    final_maxiter: int = 30
    n_restarts_final: int = 2
    reg_datasets: tuple[str, ...] = (
        "quasi_periodic",
        "periodic_noisy",
        "multiscale",
        "smooth",
        "diabetes",
    )
    clf_datasets: tuple[str, ...] = ("blobs_overlap", "moons_overlap")
    perf_budget_sec: float = 60.0

    def validate(self) -> None:
        if not self.seeds:
            raise ConfigError("seeds must be non-empty")
        if self.n_trials < 1:
            raise ConfigError("n_trials must be >= 1")
        if self.search_maxiter < 1 or self.final_maxiter < 1:
            raise ConfigError("maxiter values must be >= 1")
        if not self.reg_datasets:
            raise ConfigError("reg_datasets must be non-empty")

    @staticmethod
    def from_env() -> GaussConfig:
        cfg = GaussConfig()
        env = os.environ

        def _int(name: str, cur: int) -> int:
            return int(env[f"ENV_GF_{name}"]) if f"ENV_GF_{name}" in env else cur

        def _float(name: str, cur: float) -> float:
            return float(env[f"ENV_GF_{name}"]) if f"ENV_GF_{name}" in env else cur

        cfg = replace(
            cfg,
            n_trials=_int("N_TRIALS", cfg.n_trials),
            search_maxiter=_int("SEARCH_MAXITER", cfg.search_maxiter),
            final_maxiter=_int("FINAL_MAXITER", cfg.final_maxiter),
            n_restarts_final=_int("N_RESTARTS", cfg.n_restarts_final),
            perf_budget_sec=_float("PERF_BUDGET", cfg.perf_budget_sec),
        )
        if "ENV_GF_SEEDS" in env:
            cfg = replace(
                cfg, seeds=tuple(int(s) for s in env["ENV_GF_SEEDS"].split(","))
            )
        return cfg
