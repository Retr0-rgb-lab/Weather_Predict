# 2026-09-22 R2：Lasso 变量选择 + 特征组消融（Wave 3，提前于 09-25 完成）

> 关联：spec v1.2 §2.3/§2.4；`src/weatherlib/dataset.py`、`analysis/r2_lasso_ablation.py`
> 运行：`"/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/r2_lasso_ablation.py`

## 问题定义
① 日际变温的可预测信号集中在哪些特征组（levels/deltas/气压梯度/季节/缺失指示）？② Lasso 能否选出少数"物理通道"特征并保留大部分 skill？为报告 model 章节提供消融证据与可解释性。

## 方法与配置
- 全部线性模型，α 用 val 2008 选（无内部 CV，无洗牌）；block bootstrap(7d, 1000, seed 42)。
- 消融 10 组：levels/deltas 全量及组合、±grads、±season（受控）、±missings（受控）、上游 7 站子集、本站。
- Lasso：L1 梯度（Adam，lr 1e-2，batch 256，200 epochs），α 网格 geomspace(1e-3,1,8)，种子 0；levels+deltas 共 326 特征。
- 压缩模型：lasso_l1_full / 非零集 L2 重拟合 / top25 L2。

## 结果（test 366 天，MAE [95% CI]）
| 模型 | MAE [CI] | R² | skill vs zero |
|---|---|---|---|
| zero_change | 2.393 [2.104,2.680] | −0.000 | — |
| yesterday_Δ | 3.447 | −1.141 | −44.0% |
| levels_all163 | 1.845 [1.651,2.033] | 0.437 | +22.9% |
| deltas_all163 | 1.978 | 0.344 | +17.4% |
| **lvl+d_all163** | **1.827 [1.623,2.014]** | 0.429 | **+23.6%** |
| lvl+d+grad | 1.829 | 0.424 | +23.6% |
| lvl+d+grad+season | 1.834 | 0.422 | +23.4% |
| lvl+d+grad+missing | 1.842 | 0.415 | +23.0% |
| up7_lvl+d | 2.056 | 0.245 | +14.1% |
| up7_d | 2.115 | 0.199 | +11.6% |
| **up7_d+grad** | **2.075** | 0.242 | **+13.3%** |
| own_lvl+d | 2.392 | 0.015 | +0.0% |
| lasso_l1_full | 1.773 [1.574,1.955] | 0.445 | +25.9% |
| lasso_top324_l2 | 1.830 | 0.431 | +23.5% |
| lasso_top25_l2 | 1.859 [1.655,2.045] | 0.412 | +22.3% |

**要点**：
1. **受控消融成立**：season（+0.0→+23.4%）与 missings（+23.0%）不带来提升 → Δ 空间没有"季节免费午餐"，缺失指示无用。
2. **气压梯度在子集上才有边际价值**：全量下 grads ≈ 0（1.829 vs 1.827），但上游 only 时 +grad 从 11.6%→13.3%（+1.7pp）——梯度信息与上游 Δ 互补。
3. **levels > deltas 单独，但互补**：levels 22.9%、deltas 17.4%、合 23.6%。
4. **Lasso 几乎不稀疏化（诚实发现）**：val 最优 α=0.052 下 324/326 非零——20~25% 的 skill 分布在大量特征上，不是少数魔法特征；这本身是结论。
5. **压缩模型可行**：top25 L2 保留 22.3% skill（vs 全量 23.6%）；lasso_l1_full（≈弱正则 OLS）1.773 与 L2 1.827 差 0.05，在 CI 噪声内，不做 overclaim。
6. **物理解释（top 系数，标准化尺度）**：气压主导——d_HEATHROW_pressure −0.30、d_BASEL_pressure +0.28、d_MONTELIMAR_pressure +0.23、TOURS_pressure −0.17；次级为上游湿度（d_MAASTRICHT_humidity +0.13、DRESDEN_humidity −0.12）。→ 气旋/反气旋过境（气压场）是变温主通道，湿度次之。注：原始尺度下气压系数很大（59）是单位假象，解释用标准化系数。

## 遇到的问题与解法
- 无失败；L1 网格若更宽可更稀疏，但 val 判 0.052 最优 → 尊重 val，把"不稀疏"写为诚实结论。

## 结论与下一步
- 消融 + 物理解释齐备，报告 model 章节核心素材到位。
- 下一步 **R3（Wave 4）**：MLP 网格 {32}/{64,32}/{128,64} × Dropout{0.2,0.4} × wd{1e-4,1e-3} × Huber，3 种子 mean±std，CUDA；并接多站合并扩展。
