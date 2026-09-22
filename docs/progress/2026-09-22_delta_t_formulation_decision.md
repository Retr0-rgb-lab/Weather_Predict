# 2026-09-22 ΔT_max formulation 决策记录（主线转向 + 可行性 probe）

> 类型：formulation 决策 + R0 实验记录
> 关联：spec v1.1（docs/superpowers/specs/2026-09-22-delta-t-max-spec.md）、R0 计划（docs/superpowers/plans/）、probe（analysis/delta_t_probe.py）
> 状态：已按 AGENTS.md §0 commit + push

---

## 1. 决策：主任务从原始次日气温改为日际变温 ΔT_max

**定义**：输入第 t 天 18 站全部观测 x_t ∈ R¹⁶³（+工程特征），预测 BASEL 站次日最高温的日际变化

$$y_t = \Delta T_{\max}(t+1) = T_{\max}(t+1) - T_{\max}(t)$$

**等价性论证（报告必写）**：若 T_max(t) 在输入中，预测 Δ 与预测 T(t+1) 属同一假设空间；本决策的本质是把 persistence 从模型手里拿走、放进评价里——Δ 空间 persistence≡0，skill 第一次有物理意义。

**动机（证据）**：
- 原始 T 的 74.4% 方差是季节循环、R²=0.92 由 persistence 白拿（EDA §5.3/§7.4），"预测原始 T" 展示的是日历+惯性；
- 教学 notebook 主实验是原始 T / 日照；Kaggle 已有原始温度预测（用户核查：bernie21/temperature-prediction）；差异化必须靠评估框架而非换皮；
- Δ 空间第一次让"打得过 persistence"成为有意义的门槛；
- 物理对应冷暖空气平流，上游站信息已被 probe 证实（§5 表）。

**否决清单**（本次明确不做，理由进报告 discussion）：

| 目标 | 否决理由 |
|---|---|
| 原始 T_mean / T_max | persistence 污染 + Kaggle/教学已做 |
| 日照时长 | 教学 notebook 主实验 |
| picnic / 月份分类 | 另两份教学 notebook；且 picnic 是"无雨且暖"的派生规则 |
| 降水量数值回归 | 零膨胀 35–75%，8 天讲不清 |
| 次日降水发生（0/1） | 保留为 extension（BSS/PR-AUC），因 lag-1=0.211 属低技能诚实评估叙事，不进主线 |

## 2. 方法选择：四个问题的答案

1. **专门训一个模型吗？** 是。MLP 为专门自研小模型（无预训练模型适用于此表格任务），定位为**容量上限对照**：先验预期相对 ridge 提升有限（~2.9k 训练样本 × 低信噪比），"提升有限"本身是结论，写入 discussion。
2. **采用什么模型？** 阶梯：Zero / Yesterday-Δ / Monthly-Δ → Ridge、Lasso（变量选择→物理通道证据）→ RandomForest、HistGBM（非线性对照 + permutation importance）→ MLP（主模型）→ 扩展：多站合并 ridge/MLP。
3. **MLP 结构**：输入标准化；隐藏层网格 {32} / {64,32} / {128,64}，GELU，Dropout ∈ {0.2,0.4}，weight decay ∈ {1e-4,1e-3}，Huber（δ = 1 × train σ_Δ，目标标准化后取 1），Adam lr 1e-3，batch 64，EarlyStopping（val MAE, patience 30, max 300 epochs），种子 {0,1,2} 报 mean±std；按 mean val MAE 选配置。理由：样本少、SNR 低，过拟合是主要风险，容量由 val 裁决。
4. **其他方法怎么开展？** 统一协议（同切分/管线/指标/种子），每模型一节：动机→配置→结果→消融或解释。消融矩阵：全 163 levels+Δ / 仅上游子集 / 去 Δ 特征 / 去气压梯度 / 季节特征（受控对照，默认关）。

## 3. 研究规则（本次锁定）

**3.1 开展顺序**（spec §5 的 Wave 结构）：

| 阶段 | 内容 | 期限 |
|---|---|---|
| R0（已完成） | 决策记录 + task_brief 改向 + 可行性 probe | 09-22 |
| R1 | 正式管线 src/data_pipeline.py + 基线表 | 09-23 |
| R2 | Lasso 变量选择 + 特征组消融 + RF/HistGBM | 09-25 |
| R3 | MLP 网格 + 3 种子 + 多站合并扩展 | 09-26 |
| R4 | block bootstrap CI + 消融汇总 + 图表 | 09-27 |
| R5 | 报告成文 + 打包自检发送 | 09-29 |

**3.2 途中遇到问题的处理协议**：
1. 任何失败/异常/作废 → 记录到 docs/progress（问题→假设→尝试→结论），失败的也是 report 的 discussion/limitations 证据，不删；
2. 报告数字与代码输出对不上 → 停止写作，先复现再改文档，**禁止手改数字**；
3. 模型打不过 zero 基线 → 也是结果：查特征组消融找原因，写进 limitations，不硬凑、不调测试集；
4. 时间不够 → 按 R1→R2→R3→R4 从后往前砍：保"基线+ridge+消融"核心表，MLP 降级为小对照，多站合并可砍；
5. 每个实验单元 = 代码 + progress 记录 + commit + push（AGENTS.md §0）。

**3.3 骨架**：formulation（ΔT_max 定义 + 等价性论证 + 基线设计）→ 数据管线（清洗→填补→差分特征→标准化，train-only）→ 模型阶梯 → 消融矩阵 → 评估（skill vs zero + block bootstrap CI(7d) + 分季节）→ 报告（英文，重心 formulation+model）。

## 4. R0 实验：ΔT_max 可行性 probe

- **问题**：day-t 多站观测能否预测 BASEL ΔT_max(t+1)，并显著 beat zero-change 基线？
- **方法**：物理范围置 NaN（先清洗后差分）→ train-only 月度均值填补 → 样本对齐（levels[t] ⊕ Δ[t−1→t] → y[t]=ΔT_max(t+1)）→ 基线 ×3 + RidgeCV（alphas logspace(−2,4,25)，StandardScaler）×5 特征子集；时间切分 train/val/test = 2920/366/366，无 shuffle。
- **结果**（test 366 天，目标 train std 3.199 °C）：

| 模型 | MAE | RMSE | R² | skill vs zero | corr |
|---|---|---|---|---|---|
| zero_change（门槛） | 2.393 | 3.176 | −0.000 | — | —（常数预测） |
| yesterday_Δ | 3.447 | 4.647 | −1.141 | **−44%** | −0.072 |
| monthly_Δ 气候 | 2.393 | 3.174 | 0.001 | 0% | +0.034 |
| ridge levels 全 163 | 1.867 | 2.395 | 0.431 | **+22.0%** | +0.663 |
| ridge levels+Δ 全 163 | 1.859 | 2.419 | 0.420 | **+22.3%** | +0.655 |
| ridge 上游7站 ΔT_max | 2.116 | 2.837 | 0.202 | +11.6% | +0.452 |
| ridge 上游7站 Δ+本站Δ | 2.103 | 2.777 | 0.235 | +12.1% | +0.489 |
| ridge 上游7站 level+Δ | 2.060 | 2.745 | 0.253 | +13.9% | +0.503 |

平流证据（train 相关，ΔT_max(t) vs ΔT_max(t+1)）：TOURS Δ 0.349 > MAASTRICHT 0.201 > DE_BILT 0.190 > HEATHROW 0.182 > DUSSELDORF 0.158——西风带上游领先，与 formulation_probe 站表一致。
- **遇到的问题**：① `DataFrame.between` 不存在（pandas 为 Series 方法）→ 改 `(v<lo)|(v>hi)`；② monthly_Δ 基线 reindex 误用产生全 NaN → 改 `Series.map`；③ zero 基线 corr 触发 RuntimeWarning（常数预测无定义）→ 条件计算，记 NaN（spec P2-3 已预防）。
- **结论**：可行性确认。ridge 全量特征 R²=0.43、skill +22%（MAE 改进 0.53 °C，约为标准误量级数倍，需 bootstrap 验证）；上游 7 站 Δ 特征独立贡献 +12~14% skill，平流叙事成立；yesterday-Δ 比 zero 差 44%，坐实"日际变化近白噪声"。
- **下一步（R1）**：正式管线（站点×月统计 + temp_min>temp_max 检查）+ 基线表成文。

## 5. 评估协议红线（沿用 spec §2.2，违反=作废）

时间切分唯一合法；一切统计量 train-only 估计；哨兵值先清洗后差分；指标 MAE 主 + skill vs zero + block bootstrap(7d) CI；分季节附报；zero/月气候基线 corr 记 "—"。
