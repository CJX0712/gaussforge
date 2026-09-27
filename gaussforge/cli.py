"""CLI entry: python -m gaussforge.cli {info|demo|determinism|classify}"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _print_table(results: dict) -> None:
    summary = results["summary"]
    print("\n=== Regression benchmark (mean over seeds; rmse / nll / cov90) ===")
    header = f"{'dataset':<16}{'model':<18}{'rmse':>8}{'nll':>9}{'cov90':>8}"
    print(header)
    for ds, models in summary["table"].items():
        for model, st in models.items():
            rm = st.get("rmse_mean", float("nan"))
            nl = st.get("nll_mean", float("nan"))
            cv = st.get("coverage90_mean", float("nan"))
            print(
                f"{ds:<16}{model:<18}{rm:>8.4f}{nl:>9.4f}{cv:>8.3f}"
            )
    print("\n=== Win check: autokernel vs strongest fixed-kernel baseline ===")
    for ds, w in summary["win_check"].items():
        print(
            f"{ds:<16} vs {w['baseline']:<16} "
            f"improve={w['rmse_improvement_pct']:>6.2f}%  significant={w['significant']}"
        )
    agg = summary["aggregate_rmse_improvement_pct"]
    print(f"\naggregate rmse improvement: {agg:.2f}% (threshold {summary['win_threshold_pct']}%)")

    if results.get("clf_rows"):
        print("\n=== Classification (binary) ===")
        print(f"{'dataset':<16}{'model':<20}{'acc':>8}{'logloss':>9}{'ece':>8}")
        cells: dict[tuple[str, str], list[float]] = {}
        for r in results["clf_rows"]:
            cells.setdefault((r["dataset"], r["model"]), []).append(r["accuracy"])
            cells.setdefault((r["dataset"], r["model"]), []).append(r["log_loss"])
            cells.setdefault((r["dataset"], r["model"]), []).append(r["ece"])
        acc_cells: dict[tuple[str, str], list[float]] = {}
        ll_cells: dict[tuple[str, str], list[float]] = {}
        ece_cells: dict[tuple[str, str], list[float]] = {}
        for r in results["clf_rows"]:
            acc_cells.setdefault((r["dataset"], r["model"]), []).append(r["accuracy"])
            ll_cells.setdefault((r["dataset"], r["model"]), []).append(r["log_loss"])
            ece_cells.setdefault((r["dataset"], r["model"]), []).append(r["ece"])
        for (ds, model) in sorted(acc_cells):
            acc = float(np.mean(acc_cells[(ds, model)]))
            ll = float(np.mean(ll_cells[(ds, model)]))
            ec = float(np.mean(ece_cells[(ds, model)]))
            print(f"{ds:<16}{model:<20}{acc:>8.4f}{ll:>9.4f}{ec:>8.4f}")

    print(f"\nstructures searched: {json.dumps(results['structures'])[:400]}")
    print(f"elapsed: {results['elapsed_sec']:.1f}s (budget {results['perf_budget_sec']:.0f}s, within={results['within_budget']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gaussforge")
    parser.add_argument(
        "command", choices=["info", "demo", "determinism"], help="run mode"
    )
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001, S110 - best-effort console encoding fix
        pass

    if args.command == "info":
        from gaussforge import __version__
        from gaussforge.backends.baselines import available_sklearn
        from gaussforge.data.synthetic import available_datasets

        print(f"GaussForge v{__version__}")
        print(f"sklearn backend available: {available_sklearn()}")
        print(f"datasets: {available_datasets()}")
        return 0

    from gaussforge.core.config import GaussConfig
    from gaussforge.pipeline.gauss_pipeline import GaussPipeline

    cfg = GaussConfig.from_env()
    pipe = GaussPipeline(cfg)
    if args.command == "demo":
        results = pipe.run()
        _print_table(results)
        with open("benchmark.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print("benchmark.json written")
        return 0 if results["within_budget"] else 1

    # determinism: two full runs, compare core metrics
    r1 = pipe.run()
    r2 = pipe.run()
    keys = ("rmse_mean", "nll_mean", "coverage90_mean")
    diffs: list[tuple[str, str, float]] = []
    for ds, models in r1["summary"]["table"].items():
        for model, st in models.items():
            st2 = r2["summary"]["table"][ds][model]
            for k in keys:
                if k in st and k in st2:
                    d = abs(st[k] - st2[k])
                    if d > 0:
                        diffs.append((ds, f"{model}.{k}", d))
    if diffs:
        for ds, cell, d in diffs:
            print(f"MISMATCH {ds}/{cell}: |delta|={d:.3e}")
        return 1
    print("determinism OK: all core metrics bit-identical across runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
