# 2026-09-21 Formulation 探针：时间记忆 vs 空间记忆（补录）

> 补录于 2026-09-22：progress 留痕机制当日才建立。数字与 `docs/formulation_probe_{horizons,stations}.csv` 一致，可由 `analysis/formulation_probe.py` 复现。
> 运行：`uv run --no-project --with pandas --with numpy --with scikit-learn python analysis/formulation_probe.py`

## 问题定义
对主任务候选做可行性论证：距平空间下，预测目标站（BASEL）t+h 的温度距平，信息量主要来自**本站历史（时间记忆）**还是**其余 17 站观测（空间记忆）**？h=1..10。为"跨站点预测 ΔT"提供证据。

## 方法与切分
- 去季节化用训练期（2000–2007）月气候（`train_years` 写法，无泄漏）。
- 模型：RidgeCV（alphas logspace(−2,4,25)）+ StandardScaler。
- 消融特征组：clim（零距平）/ persist（今天距平当明天）/ own（本站 9 变量）/ others（他站 temp_mean）/ oth+d（他站 level+Δ，平流代理）/ all（163）/ no-own（163−本站）。
- 评估：test 目标从 2009-01-01 起（val 2008 不用），RMSE；skill = 1 − RMSE_others/RMSE_persist。

## 结果（h=1，test RMSE °C）
| 组 | RMSE | vs persist | vs clim |
|---|---|---|---|
| clim（零距平） | 3.378 | — | — |
| persist | 2.288 | — | +32% |
| others（17 站 temp_mean） | 2.013 | +12.0% | +40% |
| oth+d（level+Δ） | 1.955 | +15% | +42% |
| all（163） | 1.807 | +21% | +47% |
| no-own | 1.835 | — | — |

- skill 随 h 衰减：h=1 空间 skill +12.0%，h=7 相对 clim 仅 +1.3%（长 h 只剩 persistence）。
- 站表（train 相关，对 BASEL h=1 距平）：MAASTRICHT 0.804 / TOURS 0.796 / DUSSELDORF 0.778 / MUENCHEN 0.757 / KASSEL 0.738 / DE_BILT 0.725 / HEATHROW 0.688 领先（西风带上游）；OSLO 0.310 垫底。

## 遇到的问题与解法
- 无失败项；`train_years` 写法被后续所有实验沿用为标准模板。

## 结论与下一步
1. 跨站空间信息在 h=1 有 ~12% 相对 persistence 的 skill；**Δ 特征（oth+d）再贡献 ~+3pp** → 直接支撑 ΔT_max 主线的"平流通道"叙事。
2. 长 h 可预报性枯竭 → 主线锁定 h=1（2026-09-22 决策落实为 ΔT_max(t+1)）。
3. 下一步：ΔT_max 可行性 probe（09-22，`analysis/delta_t_probe.py`）。
