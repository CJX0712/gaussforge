# Model Card — GaussForge AutoKernel GP

## 用途
表格型/低维数值数据的回归与二分类，尤其当数据含**周期性结构**或需要**校准的不确定度**
（NLL/覆盖率）时。AutoKernel 自动从 9 结构 × 3 均值函数中选型，免除手工核设计。

## 训练数据
- 4 个合成构造基准（周期/趋势/多尺度，n=300，固定 seed 可复现）
- 1 个真实数据集: sklearn 内置 diabetes（442 样本 10 特征，离线可用，子采样 300）
- 无个人信息、无网络下载

## 指标（3 seeds mean，来自真实运行 benchmark.json）
- 回归: RMSE / gaussian NLL / 90% coverage
- aggregate RMSE 相对最强固定核基线 +14.18%（门槛 5%）；周期性构造基准 +22.04% / +56.06% 显著胜出
- 分类: numpy LaplaceGPC 与 sklearn GPC 精度一致（0.902/0.822），log-loss 略差（theta 未优化）

## 局限
1. **无周期结构的数据上无增益**：smooth/multiscale/diabetes 上固定 RBF 已近最优，
   AutoKernel 输 1~4%（如 diabetes -2.35%）。适用前提是数据存在可被核结构利用的规律。
2. **O(N³) 复杂度**：N>2000 时建议切分块/稀疏近似（未实现，属明确边界）。
3. **分类侧 theta 未优化**：LaplaceGPC 用固定核默认超参，log-loss 高于 sklearn GPC。
4. 结构搜索每数据集仅一次（search_seed=0），TPE 随机性可能选次优结构（10 trials 内）。
5. 一维/低维核语义：Periodic 在多维输入上作用于欧氏距离，非逐维周期。

## 伦理与合规
仅使用合成数据与学术数据集；MIT 许可；不涉及个人信息或受保护属性。
