# GaussForge 架构

## 1. 模块与调用方向（单向无环）

```
gaussforge/
  core/       GaussConfig(ENV_GF_* 覆盖) · errors(E100~E500) · set_all(seed) · Split/Row
  kernels/    composition.py: 核 DSL
              - 原语: RBF / RQ / Matérn5/2 / Periodic（统一接受 D2 与 D 距离阵）
              - 组合: Sum('a+b') / Product('a*b')，9 结构目录 SPEC_CATALOG
              - 每个原语声明 (key, init, lo, hi) 参数，优化在 log 空间进行
  gp/         exact.py: ExactGP — Cholesky 分解 + 负 LML 的 L-BFGS-B（数值梯度），
              jitter 逐级升级（1e-10×mean(diag) 起 ×10 重试 6 次），seeded 多重启；
              predict 给 (mean, std)，std 含观测噪声。
              classify.py: LaplaceGPC — R&W Alg 3.1/3.2 Newton 迭代求后验众数，
              probit 近似预测概率（Alg 3.2: p = sigmoid(m*/sqrt(1+var*))）。
  search/     autokernel.py: Optuna TPE 搜索 spec × mean；目标 val NLL（主）/ val RMSE（消融臂）；
              失败 trial 返回 1e9 而非中断。结构确定后由 pipeline 在每个 seed 的
              train split 上重新优化超参。
  data/       synthetic.py: 4 合成 + diabetes（sklearn 内置，离线可用）。
              hole 设计: train x 避开确定性区间, val/test 全域覆盖 —— 制造
              「固定平滑核无法跨 hole 桥接周期结构」的真实压力（R&W ch.5 经典场景）。
  backends/   baselines.py: available_sklearn() 探测（可 monkeypatch）；Tier-0 =
              sklearn GPR/GPC/RF；Tier-1 = NumpyRidge(闭式)/NumpyKNN/NumpyLogistic(IRLS)。
  eval/       metrics.py: rmse · gaussian_nll · coverage90(Z=1.6449) · accuracy · log_loss · ece
  pipeline/   gauss_pipeline.py: 编排（搜索→逐 seed 评测→汇总→win-check→失败案例抽取）
  cli.py      {info|demo|determinism}
```

## 2. 关键决策

| 决策 | 理由 |
|---|---|
| 全部模型只用 train split 拟合 | val 专职结构选择、test 全程未触碰；同时统一各模型口径（无泄漏） |
| 搜索每数据集一次（search_seed=0） | 结构选择是 seed 无关的合理简化，超参仍逐 seed 重优化；换 60s 性能预算 |
| L-BFGS 数值梯度 | kernel 组合（Sum/Product）解析梯度工程量大；N≤150 时数值梯度成本可忽略 |
| sklearn GPR 的 std 加回 WhiteKernel 噪声 | sklearn 默认返回隐变量 std，NLL 对比需在同一「观测」口径 |
| 失败案例自动派生 | 从 seed=0 的 test 预测排序取 3 个最差点，禁止手写叙事 |

## 3. 数据集设计（构造基准的 headroom 来源）

| dataset | 结构 | 噪声 | hole | 期望赢家 |
|---|---|---|---|---|
| quasi_periodic | 趋势 + 周期4 + 周期13 | 0.2 | [6,10] | rq*periodic |
| periodic_noisy | 纯周期3 | 0.5 | [3,7] | rbf*periodic / rq+periodic |
| multiscale | 阻尼周期2 + 慢周期9 | 0.1 | 无 | RQ（多尺度）|
| smooth | 高斯包络 + 周期5 | 0.05 | 无 | RBF（基线强）|
| diabetes | 真实 10 维 | - | 无 | 无显著差异（诚实）|

smooth/multiscale/diabetes 上固定 RBF 已近最优，AutoKernel 输掉 1~4%（噪声内），
这是设计内的诚实失败案例，写进 README 与 model_card。

## 4. 性能与预算

- 端到端 demo ≈ 51s（预算 60s），峰值内存远低于 2GB（N≤300 的 dense GP）。
- 复杂度: 拟合 O(N³)，N_train=150；搜索 10 trials × 2 目标 × 5 数据集。
- 确定性: set_all(seed) + Optuna TPESampler(seed) + L-BFGS 确定性 + 无并发；
  两次运行除耗时列外逐位一致（`make determinism` 验证）。

## 5. 离线兜底矩阵

| 能力 | Tier-0（sklearn 可用） | Tier-1（sklearn 缺失） |
|---|---|---|
| GP 回归引擎 | ExactGP（numpy，本身就是零依赖） | 同左 |
| 固定核强基线 | sklearn GPR | np-gp-rbf（同引擎） |
| RF 基线 | RandomForestRegressor | KNN 代理（标注 note） |
| GP 分类 | sklearn GPC | LaplaceGPC（numpy） |
| 逻辑回归基线 | LogisticRegression | NumpyLogistic（IRLS） |
