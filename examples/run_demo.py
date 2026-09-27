"""End-to-end demo: full benchmark -> benchmark.json.

Usage: python examples/run_demo.py
Determinism check: make determinism  (or run twice and diff benchmark.json
core metrics; elapsed_sec is excluded by design).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussforge.cli import _print_table
from gaussforge.core.config import GaussConfig
from gaussforge.pipeline.gauss_pipeline import GaussPipeline


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001, S110 - best-effort console encoding fix
        pass
    cfg = GaussConfig.from_env()
    results = GaussPipeline(cfg).run()
    _print_table(results)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nbenchmark.json written to {out}")
    return 0 if results["within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
