# 2026-09-22 R4：最终结果表 + 配对显著性 + 图表（Wave 5，提前于 09-27 完成）

> 关联：spec v1.2 §2.2；`analysis/r4_eval_figures.py`
> 运行：`"/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/r4_eval_figures.py`

## 问题定义
把 R1–R3 统一为一份可写进报告的最终表：全部模型在同一协议下重算（train-only 统计、val 选参、test 只评一次），并给出模型间**配对 block bootstrap** 的 MAE 差异显著性。

## 方法与配置
- 模型：zero / yesterday_Δ / monthly_Δ 气候 + 线性×5（levels / lvl+d / +grad / up7 / own）+ Lasso(L1 梯度, α∈geomspace(1e-3,1,8), 200 ep) + MLP 最佳配置((64,32),do0.4,wd1e-4,3 种子 ensemble)。
- 指标：MAE + block-CI(7d,1000,seed42)；配对差异用同块 bootstrap；sig_95 = CI 不含 0。
- 图：fig10 阶梯、fig11 散点、fig12 分月、fig13 Lasso 系数、fig14 逐站。

## 结果（test 366 天）
| 模型 | MAE [CI] | RMSE | R² | skill vs zero |
|---|---|---|---|---|
| zero（门槛） | 2.393 [2.104,2.680] | 3.176 | −0.000 | — |
| yesterday_Δ | 3.447 | 4.647 | −1.141 | −44.0% |
| monthly_Δ 气候 | 2.393 | 3.174 | 0.001 | +0.0% |
| lin_levels | 1.845 | 2.383 | 0.437 | +22.9% |
| lin_lvld | 1.827 [1.623,2.014] | 2.400 | 0.429 | +23.7% |
| lin_lvld+grad | 1.829 | 2.410 | 0.424 | +23.6% |
| lin_up7 | 2.056 | 2.760 | 0.245 | +14.1% |
| lin_own | 2.392 | 3.152 | 0.015 | +0.0% |
| lasso_l1 | 1.773 [1.574,1.955] | 2.365 | 0.445 | +25.9% |
| **mlp_ens_best** | **1.751 [1.546,1.946]** | 2.341 | 0.457 | **+26.9%** |

**配对显著性（diff = A − B MAE，95% CI）**
| A − B | diff | CI | sig |
|---|---|---|---|
| mlp − zero | −0.642 | [−0.843, −0.467] | ✓ |
| lasso − zero | −0.620 | [−0.834, −0.416] | ✓ |
| lin_lvld − zero | −0.566 | [−0.769, −0.383] | ✓ |
| mlp − lin_lvld | −0.076 | [−0.149, 0.000] | ✗（临界） |
| mlp − lasso | −0.023 | [−0.119, 0.068] | ✗ |
| lasso − lin_lvld | −0.054 | [−0.101, 0.002] | ✗ |
| lin_lvld − up7 | −0.229 | [−0.367, −0.115] | ✓ |

**要点**：
1. 三个学习模型（线性/Lasso/MLP）全部**显著**跑赢 zero 基线（CI 不含 0）。
2. 模型间差异（MLP vs Lasso vs 线性）CI 含 0 → **在噪声内**；MLP vs 线性在临界（上界恰为 0.000），report 用词："consistent with no improvement beyond linear"。
3. **全量 vs 上游 7 站显著**（−0.229）→ 完整场信息确凿优于子集。
4. 图 11–14 已产出（docs/figures/，gitignored，可复现）。

## 遇到的问题与解法
- 无实验失败。fig10 里 zero/yesterday/clim 用不同灰阶区分、学习模型用蓝/橙/红，报告配色可沿用。

## 结论与下一步
- **评估章节素材齐备**：一条完整、诚实、可复现的能力阶梯 + 显著性。
- 下一步 **R5（Wave 6）**：英文报告成文（formulation + model 为重心）+ 依赖/运行说明 + 打包自检 + 发送（09-29）。
