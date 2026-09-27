"""GaussPipeline: end-to-end benchmark orchestration.

Flow per regression dataset:
  1. build split per seed (fixed seed => reproducible)
  2. AutoKernel structure search once per dataset on the search_seed split
  3. per seed: refit candidates on train+val, score on untouched test
  4. aggregate mean+-std, win-check vs strongest fixed-kernel baseline,
     ablation table, failure-case extraction.

Call direction: pipeline -> {data, search, gp, backends, eval} -> core.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from gaussforge.backends import baselines as _bl
from gaussforge.backends.baselines import (
    NumpyKNN,
    NumpyRidge,
    make_sklearn_gpc,
    make_sklearn_gpr,
    make_sklearn_rf,
)
from gaussforge.core.config import GaussConfig
from gaussforge.core.types import Row
from gaussforge.data.synthetic import (
    make_classification_split,
    make_regression_split,
)
from gaussforge.eval.metrics import (
    accuracy,
    coverage90,
    ece,
    gaussian_nll,
    log_loss,
    rmse,
)
from gaussforge.gp.classify import LaplaceGPC
from gaussforge.gp.exact import ExactGP
from gaussforge.search.autokernel import search as ak_search


def _standardizer(X_train: np.ndarray):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-12
    return mu, sd


def _fit_on(tr_x, tr_y, spec: str, mean: str, cfg: GaussConfig, seed: int) -> ExactGP:
    gp = ExactGP(
        spec=spec,
        mean=mean,
        n_restarts=cfg.n_restarts_final,
        seed=seed,
        maxiter=cfg.final_maxiter,
    )
    return gp.fit(tr_x, tr_y)


class GaussPipeline:
    def __init__(self, cfg: GaussConfig | None = None) -> None:
        self.cfg = cfg or GaussConfig()
        self.cfg.validate()

    # ------------------------------------------------------------------ reg
    def _reg_rows_for_seed(
        self,
        name: str,
        seed: int,
        structure: dict[str, Any],
        structure_rmse: dict[str, Any],
    ) -> list[Row]:
        cfg = self.cfg
        split = make_regression_split(name, seed)
        mu, sd = _standardizer(split.X_train)
        tr_x = (split.X_train - mu) / sd
        te_x = (split.X_test - mu) / sd
        # All models fit on the train split only; val is reserved for structure
        # selection, test is untouched. Uniform口径 across every model.
        fit_x = tr_x
        fit_y = split.y_train
        sk_ok = _bl.available_sklearn()
        rows: list[Row] = []

        def record(model: str, m, s, extra: dict | None = None) -> None:
            t0 = time.perf_counter()
            metrics: dict[str, float] = {"rmse": rmse(split.y_test, m)}
            if s is not None:
                var = np.asarray(s) ** 2
                metrics["nll"] = gaussian_nll(split.y_test, m, var)
                metrics["coverage90"] = coverage90(split.y_test, m, np.asarray(s))
            elapsed = time.perf_counter() - t0
            rows.append(
                Row(
                    dataset=name,
                    model=model,
                    seed=seed,
                    metrics=metrics,
                    extra={**(extra or {}), "score_sec": elapsed},
                )
            )

        # autokernel (NLL objective)
        gp = _fit_on(fit_x, fit_y, structure["spec"], structure["mean"], cfg, seed)
        m, s = gp.predict(te_x)
        record("autokernel", m, s, {"spec": structure["spec"], "mean_fn": structure["mean"]})

        # ablation: rmse-objective search, same engine
        gp2 = _fit_on(fit_x, fit_y, structure_rmse["spec"], structure_rmse["mean"], cfg, seed)
        m2, s2 = gp2.predict(te_x)
        record("autokernel-rmse", m2, s2, {"spec": structure_rmse["spec"]})

        # fixed RBF on the same numpy engine (isolates search contribution)
        gp3 = _fit_on(fit_x, fit_y, "rbf", "const", cfg, seed)
        m3, s3 = gp3.predict(te_x)
        record("np-gp-rbf", m3, s3)

        if sk_ok:
            gpr = make_sklearn_gpr(seed)
            gpr.fit(fit_x, fit_y)
            m4, s4 = gpr.predict(te_x, return_std=True)
            noise = 0.0
            try:
                noise = float(gpr.kernel_.k2.noise_level)
            except Exception:  # noqa: BLE001 - kernel layout varies by sklearn version
                noise = 0.0
            s4 = np.sqrt(np.asarray(s4) ** 2 + noise)
            record("sklearn-gp-rbf", m4, s4)

            rf = make_sklearn_rf(seed)
            rf.fit(fit_x, fit_y)
            record("rf", rf.predict(te_x), None)
        else:
            record("rf", NumpyKNN(5).fit(fit_x, fit_y).predict(te_x), None, {"note": "rf unavailable, knn proxy"})

        ridge = NumpyRidge().fit(fit_x, fit_y)
        record("ridge", ridge.predict(te_x), None)
        knn = NumpyKNN(5).fit(fit_x, fit_y)
        record("knn", knn.predict(te_x), None)

        if seed == self.cfg.search_seed:
            worst = np.argsort(-np.abs(split.y_test - m))[:3]
            self._failure_cache.append(
                {
                    "dataset": name,
                    "model": "autokernel",
                    "spec": structure["spec"],
                    "cases": [
                        {
                            "x": [round(float(v), 4) for v in te_x[i]],
                            "y_true": float(split.y_test[i]),
                            "y_pred": float(m[i]),
                            "abs_err": float(abs(split.y_test[i] - m[i])),
                        }
                        for i in worst
                    ],
                }
            )
        return rows

    def _search_structures(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {"nll": {}, "rmse": {}}
        for name in self.cfg.reg_datasets:
            split = make_regression_split(name, self.cfg.search_seed)
            mu, sd = _standardizer(split.X_train)
            for obj in ("nll", "rmse"):
                out[obj][name] = ak_search(
                    (split.X_train - mu) / sd,
                    split.y_train,
                    (split.X_val - mu) / sd,
                    split.y_val,
                    objective=obj,
                    n_trials=self.cfg.n_trials,
                    seed=self.cfg.search_seed,
                    maxiter=self.cfg.search_maxiter,
                )
        return out

    # ------------------------------------------------------------------ clf
    def _classification_rows(self, name: str, seed: int) -> list[Row]:
        split = make_classification_split(name, seed)
        mu, sd = _standardizer(split.X_train)
        tr_x = (split.X_train - mu) / sd
        te_x = (split.X_test - mu) / sd
        sk_ok = _bl.available_sklearn()
        rows: list[Row] = []

        def record(model: str, proba: np.ndarray) -> None:
            rows.append(
                Row(
                    dataset=name,
                    model=model,
                    seed=seed,
                    metrics={
                        "accuracy": accuracy(split.y_test, proba),
                        "log_loss": log_loss(split.y_test, proba),
                        "ece": ece(split.y_test, proba),
                    },
                )
            )

        record("numpy-laplace-gpc", LaplaceGPC(seed=seed).fit(tr_x, split.y_train).predict_proba(te_x))
        if sk_ok:
            gpc = make_sklearn_gpc(seed)
            gpc.fit(tr_x, split.y_train)
            record("sklearn-gpc", gpc.predict_proba(te_x)[:, 1])
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(max_iter=1000).fit(tr_x, split.y_train)
            record("logistic", lr.predict_proba(te_x)[:, 1])
        else:
            from gaussforge.backends.baselines import NumpyLogistic

            record("logistic", NumpyLogistic().fit(tr_x, split.y_train).predict_proba(te_x))
        return rows

    # ------------------------------------------------------------------ run
    def run(self) -> dict[str, Any]:
        self._failure_cache: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        structures = self._search_structures()
        rows: list[Row] = []
        for name in self.cfg.reg_datasets:
            for seed in self.cfg.seeds:
                rows.extend(
                    self._reg_rows_for_seed(name, seed, structures["nll"][name], structures["rmse"][name])
                )
        clf_rows: list[Row] = []
        for name in self.cfg.clf_datasets:
            for seed in self.cfg.seeds:
                clf_rows.extend(self._classification_rows(name, seed))
        elapsed = time.perf_counter() - t0
        summary = self._summarize(rows)
        failures = self._failure_cache
        return {
            "config": {
                "seeds": list(self.cfg.seeds),
                "n_trials": self.cfg.n_trials,
                "reg_datasets": list(self.cfg.reg_datasets),
                "clf_datasets": list(self.cfg.clf_datasets),
            },
            "structures": structures,
            "rows": [r.as_dict() for r in rows],
            "clf_rows": [r.as_dict() for r in clf_rows],
            "summary": summary,
            "failures": failures,
            "elapsed_sec": elapsed,
            "perf_budget_sec": self.cfg.perf_budget_sec,
            "within_budget": elapsed <= self.cfg.perf_budget_sec,
        }

    # -------------------------------------------------------------- summary
    def _summarize(self, rows: list[Row]) -> dict[str, Any]:
        by_cell: dict[tuple[str, str], list[Row]] = {}
        for r in rows:
            by_cell.setdefault((r.dataset, r.model), []).append(r)

        def cell_stats(model: str, ds: str) -> dict[str, float] | None:
            cell = by_cell.get((ds, model))
            if not cell:
                return None
            out: dict[str, float] = {}
            for key in ("rmse", "nll", "coverage90", "accuracy", "log_loss", "ece"):
                vals = [float(c.metrics[key]) for c in cell if key in c.metrics]
                if vals:
                    out[f"{key}_mean"] = float(np.mean(vals))
                    out[f"{key}_std"] = float(np.std(vals))
            return out

        models = sorted({r.model for r in rows})
        datasets = sorted({r.dataset for r in rows})
        table: dict[str, Any] = {}
        for ds in datasets:
            table[ds] = {m: cell_stats(m, ds) for m in models if cell_stats(m, ds)}

        # win-check vs strongest fixed-kernel GP baseline
        baseline_candidates = [m for m in ("sklearn-gp-rbf", "np-gp-rbf") if m in models]
        wins: dict[str, Any] = {}
        for ds in datasets:
            if not baseline_candidates or "autokernel" not in table[ds]:
                continue
            best_base = min(
                (m for m in baseline_candidates if m in table[ds]),
                key=lambda m: table[ds][m]["rmse_mean"],
            )
            ak = table[ds]["autokernel"]
            base = table[ds][best_base]
            delta = base["rmse_mean"] - ak["rmse_mean"]
            improve = delta / base["rmse_mean"] if base["rmse_mean"] else 0.0
            sig = delta > 0.5 * (base["rmse_std"] + ak["rmse_std"])
            wins[ds] = {
                "baseline": best_base,
                "baseline_rmse_mean": base["rmse_mean"],
                "baseline_rmse_std": base["rmse_std"],
                "autokernel_rmse_mean": ak["rmse_mean"],
                "autokernel_rmse_std": ak["rmse_std"],
                "rmse_improvement_pct": improve * 100.0,
                "significant": bool(sig),
            }
        agg_improve = (
            float(np.mean([w["rmse_improvement_pct"] for w in wins.values()])) if wins else 0.0
        )
        return {
            "table": table,
            "win_check": wins,
            "aggregate_rmse_improvement_pct": agg_improve,
            "win_threshold_pct": 5.0,
            "win_all_significant": bool(wins) and all(w["significant"] for w in wins.values()),
        }
