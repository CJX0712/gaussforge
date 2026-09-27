"""Pipeline integration tests (tiny configs) + offline fallback path."""

from __future__ import annotations

import pytest

import gaussforge.backends.baselines as bl
from gaussforge.core.config import GaussConfig
from gaussforge.pipeline.gauss_pipeline import GaussPipeline


def _tiny_cfg() -> GaussConfig:
    return GaussConfig(
        seeds=(0,),
        n_trials=2,
        search_maxiter=10,
        final_maxiter=10,
        n_restarts_final=1,
        reg_datasets=("smooth",),
        clf_datasets=("blobs_overlap",),
    )


def test_pipeline_tiny_run_contract():
    res = GaussPipeline(_tiny_cfg()).run()
    assert res["rows"] and res["clf_rows"]
    models = {r["model"] for r in res["rows"]}
    assert "autokernel" in models and "np-gp-rbf" in models and "ridge" in models
    assert res["summary"]["win_check"]
    assert "autokernel_rmse_mean" in next(iter(res["summary"]["win_check"].values()))
    assert res["within_budget"] or res["elapsed_sec"] > 0
    # failure cases derived from seed-0 test predictions
    assert res["failures"] and len(res["failures"][0]["cases"]) == 3


def test_pipeline_tiny_run_deterministic():
    r1 = GaussPipeline(_tiny_cfg()).run()
    r2 = GaussPipeline(_tiny_cfg()).run()
    t1 = r1["summary"]["table"]["smooth"]
    t2 = r2["summary"]["table"]["smooth"]
    for model in t1:
        for key in ("rmse_mean", "nll_mean"):
            if key in t1[model] and key in t2[model]:
                assert t1[model][key] == t2[model][key], (model, key)


def test_pipeline_offline_fallback(monkeypatch):
    monkeypatch.setattr(bl, "available_sklearn", lambda: False)
    res = GaussPipeline(_tiny_cfg()).run()
    models = {r["model"] for r in res["rows"]}
    assert "sklearn-gp-rbf" not in models
    # sklearn-dependent rows are skipped; numpy arms still produce results
    assert "autokernel" in models
    clf_models = {r["model"] for r in res["clf_rows"]}
    assert "sklearn-gpc" not in clf_models
    assert "numpy-laplace-gpc" in clf_models
    assert "logistic" in clf_models


def test_config_validation():
    from gaussforge.core.errors import ConfigError

    with pytest.raises(ConfigError):
        cfg = GaussConfig(n_trials=0)
        cfg.validate()
