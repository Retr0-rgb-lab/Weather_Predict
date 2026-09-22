# 2026-09-22 R1：正式数据管线 + 基线表（Wave 2，提前于 09-23 完成）

> 关联：spec v1.1 §2.2/§2.4/§5；计划 `docs/superpowers/plans/2026-09-22-r1-pipeline-baselines.md`
> 代码：`src/weatherlib/`（共享实验库）、`analysis/r1_baselines.py`
> 运行：`uv run --no-project --with pandas --with numpy --with scikit-learn python analysis/r1_baselines.py`

## 问题定义
把 R0 probe 升级为正式可复用的数据管线 + 基线表：清洗（含 temp_min>temp_max）、train-only 站点×月中位数填补、ΔT_max 对齐、气压梯度特征、block bootstrap CI。目标：锁定 zero-change 门槛与 ridge 上界，为 R2（Lasso/树）与 R3（MLP）提供统一基座。

## 代码架构（已落盘 src/weatherlib）
```
data.py      load_raw / clean(范围+temp_min>temp_max, 先清洗后差分) / impute_station_month(train-only 站点×月中位数+缺失指示,默认关)
dataset.py   make_delta_max_dataset: levels[t]⊕deltas[t]⊕grads[t] → y[t]=ΔT_max(t+1)；train/val/test 按目标日掩码
features.py  pressure_gradients(P_upstream−P_BASEL, 7 特征) / season_features(受控,默认关)
metrics.py   MAE/RMSE/R²/corr_safe/skill + moving block bootstrap(block=7d) CI（单模型&配对差异）
baselines.py zero / yesterday_Δ / monthly_Δ climatology
models.py    fit_ridge / fit_lasso（R2 加树，R3 加 MLP）
```

## 数据与切分
- train/val/test = **2920/366/366**（目标日掩码），无 shuffle。
- 清洗掩蔽合计 **77 格** = 3(cloud)+29(sunshine)+4(pressure)+41(temp_min>temp_max)——与 EDA §3.2 完全一致。
- 填补：站点×月**中位数**（probe 用均值）；气压梯度 7 个（上游 7 站 − BASEL）。
- 超参：RidgeCV alphas=logspace(−2,4,25)，StandardScaler 仅 fit 于 train；bootstrap n=1000, block=7, seed=42。

## 结果（test 366 天，目标 train std 3.199 °C；MAE 后为 95% block-CI）
| 模型 | MAE [CI] | RMSE | R² | skill vs zero |
|---|---|---|---|---|
| zero_change（门槛） | 2.393 [2.104,2.680] | 3.176 | −0.000 | — |
| yesterday_Δ | 3.447 [3.033,3.916] | 4.647 | −1.141 | **−44.0%** |
| monthly_Δ 气候 | 2.393 [2.098,2.682] | 3.174 | 0.001 | +0.0% |
| ridge levels 全 163 | 1.859 [1.665,2.039] | 2.388 | 0.435 | +22.3% |
| ridge levels+Δ 全 163 | 1.851 [1.646,2.024] | 2.414 | 0.422 | **+22.7%** |
| ridge levels+Δ+grad | 1.847 [1.642,2.019] | 2.412 | 0.423 | +22.8% |
| ridge 上游7站 level+Δ | 2.058 [1.838,2.286] | 2.743 | 0.254 | +14.0% |
| ridge 上游7站 Δ | 2.115 [1.873,2.342] | 2.835 | 0.203 | +11.6% |
| **ridge 本站 level+Δ** | 2.422 [2.144,2.691] | 3.122 | 0.034 | **−1.2%** |

**要点**：
1. **一致性**：ridge lvl+d_all163 MAE 1.851 vs probe 1.859（中位数 vs 均值填补，偏移 0.008 °C）——管线正确。
2. **显著性**：ridge 全量 CI 上限 2.024 < zero 下限 2.104 → 提升在 block 自举层面**不重叠**，+22.7% skill 非噪声。
3. **本站对照**：仅本站 level+Δ 是 −1.2%（无 skill）→ 坐实"本站日际变化近白噪声"（lag-1=0.075），信息来自他站（物理通道）。
4. **Δ 与梯度贡献**：上游 7 站 Δ（11.6%）→ level+Δ（14.0%）；全量上 +Δ 仅 +0.4pp、+grad 再 +0.1pp——全量下边际有限（特征冗余 62/163），但子集叙事成立。
5. **yesterday_Δ −44%** 再次坐实日际变化不可用自身惯性预测。

## 遇到的问题与解法（含失败）
1. `load_raw` 返回 Series 而非 DatetimeIndex → `.month` 不存在；改用 `pd.DatetimeIndex(pd.to_datetime(...))`。
2. 清洗时 int64 列置 NaN → `LossySetitemError`；`clean()` 内先 `astype(float)`。
3. `DatetimeIndex < datetime64` 已返回 ndarray，多余 `.to_numpy()` → 报错；移除。
4. `fit_ridge` 未在 `__init__.py` 导出 → 补导出。
5. 气压梯度误传全 18 站 → 14 特征；按 spec 改传上游 7 站 → 7 特征。
6. `record()` 对已切片的 ridge 预测再按 te 切片 → IndexError；统一传入全长度预测、record 内切片。

## 结论与下一步
- 正式基座就绪：zero-change 门槛 **MAE 2.393**、ridge 全量上界 **MAE 1.851 / R² 0.422 / skill +22.7%**（CI 不重叠）。
- R1 验收全过。下一步 **R2（Wave 3）**：Lasso 变量选择 + 特征组消融成表 + RF/HistGBM（09-25 前）。
