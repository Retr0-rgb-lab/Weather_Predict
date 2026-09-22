# ΔT_max 日际变温预测 — Spec v1.1

> 日期: 2026-09-22
> 依据: docs/data_analysis.md §5.3/§7.3、docs/formulation_probe_horizons.csv、AGENTS.md §0、docs/progress/2026-09-22_delta_t_formulation_decision.md（同批落盘）
> 基线分支: main
> 前置依赖: EDA 完成（eda.py + data_analysis.md）；formulation 探针完成（formulation_probe.py + 两个 CSV）；AGENTS.md 轨迹规则已生效

---

## 摘要

主线问题从"原始次日气温"改为**日际变温** ΔT_max(t+1) = T_max(t+1) − T_max(t)（BASEL 为主目标站）。原始 T 的预测被 persistence 与季节循环污染（74.4% 方差是日历），Δ 空间 persistence≡0，skill 第一次有物理意义。探针数据已证可行性（temp_mean 距平空间 h=1：全特征 ridge 1.807 vs persistence 2.288 vs 零基线 3.378 °C RMSE）。模型采用"基线→线性→树→MLP"阶梯，MLP 为专门训练的小模型（64→32，重正则），定位为容量上限对照；评估以 MAE + skill-vs-zero + block bootstrap CI 为核心。产出为英文报告（评审核心 = formulation + model），截止 2026-09-30。

## 0. 前置依赖

- `analysis/eda.py` → `docs/data_analysis.md` + `docs/eda_results.json`（全部 EDA 数字）
- `analysis/formulation_probe.py` → `docs/formulation_probe_{horizons,stations}.csv`（距平空间 skill 证据）
- 本 spec 与同批的 progress 决策记录、`analysis/delta_t_probe.py` 共同构成 R0

## 1. 现状

- `docs/task_brief.md` §4.1 的主任务仍是"原始 BASEL_temp_mean(t+1) 回归"——**已偏离研究方向，本 spec 负责改向**。
- `src/l1.ipynb` 为 torch MLP 草稿（2 cell），未成实验。
- 探针证据（formulation_probe_horizons.csv，temp_mean 距平，h=1，test 366 天）：

| 模型 | RMSE (°C) | vs persistence | vs 零 |
|---|---|---|---|
| 零（气候距平） | 3.378 | — | — |
| persistence | 2.288 | — | +32% |
| 其他 17 站 temp_mean | 2.013 | +12% | +40% |
| 17 站 level+Δ（平流代理） | 1.955 | +15% | +42% |
| 全 163 特征 | 1.807 | +21% | +47% |

- 站表（formulation_probe_stations.csv）：对 BASEL 次日距平，h=1 领先者为 MAASTRICHT 0.804 / TOURS 0.796 / DUSSELDORF 0.778 / MUENCHEN 0.757 / KASSEL 0.738 / DE_BILT 0.725 / MONTELIMAR 0.695（第 7 位；HEATHROW 0.688 居第 8，因西风带物理位置入选上游 7 站，见 2.4）；末位为 OSLO 0.310 / STOCKHOLM 0.361 / ROMA 0.426。
- EDA 关键约束：日际变化 lag-1 自相关 0.075（近乎白噪声）；季节循环占 T_mean 方差 74.4%；77 个物理越界格（含 `STOCKHOLM_cloud_cover=−99`，20090625，测试期内）；量纲未标准化；2010 年仅 1 天。

## 2. 需要做的工作

### 2.1 Formulation 定义（本次锁定，写进报告 §Problem Formulation）

- **输入**：第 t 天 18 站全部观测 x_t ∈ R¹⁶³（levels）+ 工程特征（上游站 Δ、气压梯度，见 2.4）
- **目标**：y_t = ΔT_max(t+1) = BASEL_temp_max(t+1) − BASEL_temp_max(t)
- **等价性论证（必写）**：若 T_max(t) 在输入中，预测 Δ 与预测 T(t+1) 属同一假设空间；Δ 公式化是把 persistence 从模型手里拿走、放进评价里。此句是防 reviewer 质疑的关键。
- **物理叙事**：ΔT 对应冷暖空气平流；西风带使上游站（MAASTRICHT/TOURS/DE_BILT）当日观测领先 BASEL 次日变化（probe 站表证据）。

### 2.2 评估协议（红线，违反 = 作废）

1. **时间切分**：train 2000–2007（2922 天，对齐后 2920 样本）/ val 2008（366 天）/ test 2009-01-01～2010-01-01（366 天）。严禁 shuffle。
2. **基线**：zero-change（Δ 空间 persistence，主门槛）→ yesterday-Δ（预期惨败，坐实 lag-1=0.075）→ monthly-Δ 气候（≈0）→ Ridge/Lasso → RF/HistGBM → MLP。
3. **指标**：MAE（°C，主指标，重尾稳健）、RMSE、R²（Δ 空间≈skill vs 均值）、skill = 1 − MAE_model/MAE_zero、corr(Δ̂, Δ)（zero/月气候基线预测近常数，corr 无定义，记 —）。分季节附报。
4. **不确定性**：moving block bootstrap（block=7 天 ≥ 气压记忆 3–5 天），1000 次，95% 分位 CI；模型间差异用同块配对 bootstrap。
5. **泄漏控制**：清洗/填补/标准化/气候统计一律 train-only 估计再施加到 val/test；特征只用第 t 天及以前。
6. **哨兵值必须在差分前清洗**：−99 直接差分会产生 ±100 °C 假尖刺。

### 2.3 模型阶梯与方法选择（回答四个问题）

1. **专门训一个模型吗？** 是。MLP 为专门自研小模型（无预训练模型适用于此表格任务），但定位为**容量上限对照**：先验预期是相对 ridge 提升有限（~2.9k 训练样本 × 低信噪比），"提升有限"本身是结论（样本量-容量权衡），写入 discussion。
2. **什么模型？** 阶梯：Zero / Yesterday-Δ / Monthly-Δ → Ridge、Lasso（变量选择，物理通道证据）→ RandomForest、HistGBM（非线性对照，permutation importance）→ MLP（主模型）→ 扩展：多站合并 ridge/MLP。
3. **MLP 结构**：输入标准化；隐藏层网格 {32} / {64,32} / {128,64}，GELU，Dropout ∈ {0.2,0.4}，weight decay ∈ {1e-4,1e-3}，Huber(δ = 1 × train σ_Δ；目标按 train σ_Δ 标准化后 δ 取 1，等价于原始尺度 1 个 σ_Δ)，Adam lr 1e-3，batch 64，EarlyStopping(val MAE, patience 30, max 300 epochs)，种子 {0,1,2} 报 mean±std；按 mean val MAE 选配置。理由：~2.9k 训练样本、低 SNR，过拟合是主要风险，容量由 val 集裁决。
4. **其他方法怎么开展？** 统一协议（同一切分/管线/指标/种子），每模型一节：动机→配置→结果→消融或解释。消融矩阵：全 163 levels+Δ / 仅上游子集 / 去 Δ 特征 / 去气压梯度 / 季节特征（受控对照，默认关）。

### 2.4 数据管线

```
CSV → ①物理范围置 NaN（哨兵清洗，先于差分）→ ②站点×月填补（train-only 统计；缺失指示特征默认不加，列为消融可选项）
    → ③特征工程：levels x_t(163) │ 上游站 Δtemp_max/Δtemp_mean │ 气压梯度(站对差) │ (受控)季节特征
    → ④标准化（train fit）→ 模型阶梯 → 评估（skill + block bootstrap）
```

- 上游 7 站（西风带）：MAASTRICHT, TOURS, DUSSELDORF, MUENCHEN, KASSEL, DE_BILT, HEATHROW。（h=1 相关前 6 + HEATHROW；MONTELIMAR 0.695 高于 HEATHROW 0.688，但位置偏南、非典型西风上游，取舍属物理判断，消融可对照。）
- 差分特征用 t−1→t 的变化，样本 t 对齐目标日 t+1。

## 3. 文件变更清单

| 文件 | 变更 | 节号 |
|---|---|---|
| docs/progress/2026-09-22_delta_t_formulation_decision.md | 新建：决策记录 + 研究规则（顺序/问题处理/骨架）+ 方法四问答案 | 2 |
| docs/superpowers/specs/2026-09-22-delta-t-max-spec.md | 新建：本文件 | 全部 |
| docs/superpowers/plans/2026-09-22-r0-execution-plan.md | 新建：R0 执行计划 | 5 |
| analysis/delta_t_probe.py | 新建：ΔT_max 可行性 probe | 2 |
| docs/delta_t_probe.csv, docs/delta_t_probe_lead.csv | 新建：probe 机器可读结果（commit 作证据） | 2 |
| docs/task_brief.md | 改向：§4.1 主任务、§4.2 扩展、§4.3 协议、§5 模型、§7 时间线、§9 风险、新增 §10 变更日志 | 2.5 |

## 4. 架构图

```
data/weather_prediction_dataset.csv
   │ ① 物理范围清洗（train-only 统计）
   ▼
填补 → levels x_t(163) ──┐
   │ diff(t−1→t)          │ ④ 标准化(train fit)
   ▼                       │
Δ 上游特征 + 气压梯度 ─────┤
   │                       ▼
   └──► Zero/Yday-Δ/ClimΔ → Ridge/Lasso → RF/HistGBM → MLP(64→32, Huber, ES)
                               │
                               ▼
              test: MAE/RMSE/R²/skill-vs-zero + block-CI(7d) + 分季节
                               │
                               ▼
              docs/progress 记录 → commit → push（每个实验单元，AGENTS.md §0）
```

## 5. 实施顺序

| 阶段 | 任务 | 依赖 | 工作量 | 完成期限 |
|---|---|---|---|---|
| **Wave 1（本会话，单执行者顺序执行）** | Plan A: probe 脚本+运行+记录；Plan B: task_brief 改向 + 决策记录 | 无 | 0.5d | 09-22 |
| Wave 2 | R1: 正式管线 src/data_pipeline.py + 基线表（zero/yday/climΔ/ridge 全量/子集） | Wave 1 | 1d | 09-23 |
| Wave 3 | R2: Lasso 变量选择 + 特征组消融 + RF/HistGBM | Wave 2 | 1.5d | 09-25 |
| Wave 4 | R3: MLP 网格 + 3 种子 + 多站合并扩展 | Wave 2 | 1.5d | 09-26 |
| Wave 5 | R4: block bootstrap CI + 消融汇总 + 图表 | Wave 3,4 | 1d | 09-27 |
| Wave 6 | R5: 报告成文 + 打包自检发送 | Wave 5 | 1.5d | 09-29 |

Wave 2–6 进入时各自生成 plan 文件；本 spec 的 Wave 1 计划见 plans/2026-09-22-r0-execution-plan.md。

## 6. 验收标准

- [ ] probe 产出 test MAE（zero vs ridge 全量 vs 上游子集），写入 docs/delta_t_probe.csv，且 `uv run --no-project --with pandas --with numpy --with scikit-learn python analysis/delta_t_probe.py` 可复现
- [ ] task_brief §4.1 主任务 = ΔT_max，无"原始 temp_mean(t+1) 主线"残留；§10 变更日志含日期、理由、progress 链接
- [ ] progress 决策记录含：4 个方法问题的答案 + 研究规则三件套（顺序/问题处理/骨架）
- [ ] 全程时间切分 + train-only 统计（oracle 重点核对此项）
- [ ] 报告每个数字可追溯到 commit 的代码输出
- [ ] oracle 审查结论为 APPROVE 或 APPROVE-WITH-MINOR，P0 问题 = 0

## 7. 不在范围

- 原始 T_mean/T_max 作为主线任务（Kaggle/教学已做；仅可作对照行）
- picnic / 月份 / 日照任务（已否决，理由见决策记录）
- 降水量数值回归（零膨胀）；LSTM（stretch，仅当 Wave 5 提前完成时加跑）
- 超出 §2.3 网格的超参搜索

## 8. 版本变更

- v1.0 (2026-09-22): 初稿，待 oracle 审查。
- v1.1 (2026-09-22): oracle APPROVE-WITH-MINOR（P0=0）。修正：Huber δ 表述钉死（P1-1）；§2.4 显式声明缺失指示特征默认不加（P1-2）；corr 对近常数基线记 —（P2-3）；train 天数 2921→2922/对齐 2920（P2-1，EDA/AGENTS 照抄的 off-by-one 一并修正）；站表第 7/8 位与末位表述（P2-2）；3654→~2.9k 训练样本口径（P2-4）；排期零冗余，报告骨架从 Wave 2 起并行成文（P2-6）。
