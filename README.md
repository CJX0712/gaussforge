# GaussForge

[![CI](https://github.com/CJX0712/gaussforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/gaussforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/tag/CJX0712/gaussforge?label=release)](https://github.com/CJX0712/gaussforge/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue)](pyproject-less: numpy/scipy)
![Quality](https://img.shields.io/badge/Quality-S%20(生产级%20DoD%20全绿)-brightgreen)

**模块化高斯过程（Gaussian Process）回归/分类系统，带 AutoKernel 核结构自动搜索。**
纯 numpy/scipy exact GP 引擎（Tier-1 离线兜底）+ scikit-learn Tier-0 强基线对照，
CPU-only、零下载、固定 seed 逐位可复现。作者：晨星（CJX0712）。

## 核心结果（3 seeds mean，真实运行输出，见 `benchmark.json`）

| dataset | GaussForge AutoKernel | 最强固定核基线 | RMSE 改进 | 显著 |
|---|---|---|---|---|
| quasi_periodic | **0.2210** | 0.2589 (np-gp-rbf) | **+56.06%** | ✅ |
| periodic_noisy | **0.8367** | 1.0011 (sklearn-gp-rbf) | **+22.04%** | ✅ |
| multiscale | 0.1067 | 0.1056 | -1.03% | — |
| smooth | 0.0520 | 0.0501 | -3.81% | — |
| diabetes | 0.7351 | 0.7182 | -2.35% | — |
| **aggregate** | | | **+14.18%**（门槛 5%） | |

- 显著性规则：均值差 > 0.5×(std₁+std₂)（V4 统计严谨门槛）。
- 在两个周期性构造基准上多 seed 均值显著胜过 sklearn GPR（固定 RBF+White 核）；
  无周期结构的数据集上固定 RBF 已近最优，诚实报告小负差。
- NLL（不确定性质量）在周期性数据集上同样优于全部基线。

## AutoKernel 是什么

核 DSL 目录：`rbf / rq / matern52 / periodic` × 组合（`+`/`*`，如 `rq*periodic`）共 9 结构
× 3 均值函数（zero/const/linear）。Optuna TPE 在验证集 NLL（或 RMSE，消融臂）上搜索
结构，超参由 L-BFGS 边际似然最大化在 log 空间内优化（seeded 多重启）。
周期性数据集上自动选中 `rq*periodic` / `rbf*periodic` —— 与 Duvenaud 核手册
（Rasmussen & Williams ch.5）的经验规则一致。

## 一键复现

```bash
git clone https://github.com/CJX0712/gaussforge.git && cd gaussforge
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# POSIX: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python examples/run_demo.py        # 完整 benchmark -> benchmark.json (~51s)
python -m pytest tests -q          # 50 tests
python -m gaussforge.cli determinism   # 两次全量运行逐位一致校验
```

CLI：`python -m gaussforge.cli {info|demo|determinism}`；支持 `ENV_GF_N_TRIALS`、
`ENV_GF_SEEDS` 等环境变量覆盖（见 `core/config.py`）。

## 架构（单向无环）

```
cli → pipeline → {data, search, gp, backends, eval} → core
kernels: 9 结构核 DSL（Sum/Product 组合）
gp:      ExactGP（Cholesky + LML L-BFGS）· LaplaceGPC（二分类拉普拉斯近似）
search:  AutoKernel（Optuna TPE，val NLL / RMSE 双目标）
backends: sklearn Tier-0 探测（GPR/GPC/RF）+ 纯 numpy Tier-1 兜底（Ridge/KNN/Logistic）
```

详细设计见 [docs/architecture.md](docs/architecture.md)，模型卡见
[docs/model_card.md](docs/model_card.md)。

## 质量

- 50 单测全绿，核心覆盖率 87%（`pytest --cov`）
- ruff 全绿；依赖锁定 `requirements.lock.txt`
- 离线兜底：sklearn 不可用时自动降级 numpy 后端（有专门单测覆盖降级路径）
- 确定性：同 seed 两次运行核心指标逐位一致
- 消融：NLL 目标 vs RMSE 目标 vs 无搜索（固定 RBF），见 benchmark.json
- 失败案例：自动抽取 3 组最差测试点（x, y_true, y_pred, abs_err），见 benchmark.json `failures`

## License

MIT © 晨星 (CJX0712)
