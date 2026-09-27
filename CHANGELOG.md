# Changelog

## v0.1.0 (2026-09-28)

- 首发：模块化高斯过程回归/分类系统
- AutoKernel：Optuna TPE 核结构搜索（9 结构 × 3 均值函数，val NLL / RMSE 双目标）
- 纯 numpy/scipy ExactGP（Cholesky + LML L-BFGS 多重启）与 LaplaceGPC（二分类）
- Tier-0 sklearn 基线（GPR/GPC/RF）+ Tier-1 纯 numpy 兜底（Ridge/KNN/Logistic IRLS）
- 基准：5 回归数据集 × 3 seeds（mean±std + 显著性门槛）+ 2 分类数据集
- aggregate RMSE 改进 +14.18%（vs 最强固定核基线），周期性基准 +22.04%/+56.06% 显著
- 50 单测全绿，覆盖率 87%，ruff 全绿，同 seed 逐位确定性，demo 51s ≤ 60s 预算
