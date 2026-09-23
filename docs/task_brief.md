# Weather Prediction 任务说明

> 任务来源：Ms. PAN Yubing（yubing.pan@connect.polyu.hk）课题组面试前的筛选作业
> 文档整理日期：2026-09-19 ｜ **改向日期：2026-09-22（§10 变更日志）** ｜ 提交截止：**2026-09-30**
> 本仓库根目录：`E:\College_Projects\Weather_Predict`

---

## 1. 导师邮件要点（原文归纳）

| 项 | 内容 |
|---|---|
| 任务 | 使用 `https://github.com/florian-huber/weather_prediction_dataset` 的数据，构建一个天气预测模型 |
| 必交物 | ① PDF 报告（**必须体现 problem formulation 与 model**，实验结果可选）② 对应代码 |
| 提交方式 | 把 CV、报告、代码压缩为**单个 ZIP**，命名为 `Task Report_YourName` |
| 收件人 | yubing.pan@connect.polyu.hk |
| 邮件主题 | `Task Report Submission_YourName` |
| 截止 | 30 Sep |
| 附加条件 | 只有**提交了作业**的候选人才会被安排面试；截止后统一发面试通知 |

解读：评审重点明确写在括号里——**problem formulation（问题如何定义）与 model（模型设计与选择）**。报告应把"我为什么这样定义问题、为什么选这个模型、如何避免自欺欺人的评估"讲清楚。本仓库已于 2026-09-22 将主线形式化为**日际变温预测 ΔT_max**（§4.1），差异化正是建立在评估诚实性之上。

---

## 2. 项目背景：这个数据集是什么

（2026-09-22 注：本节数据集描述与主线无关，保持原样；数据commit仍为 `83d70ee`。）

数据集由 Florian Huber 等人构建，**目标是教学用途**（ECML 2022 "Teaching Machine Learning" workshop），刻意设计成"小而真实"：单机几分钟训完，但天然包含真实数据的坑。

- **原始来源**：ECA&D 地面站点日观测，file created 22-04-2021。
- **时空范围**：18 个欧洲站点，2000-01-01 ~ 2010-01-01，共 **3654 天**。
- **站点**：BASEL、BUDAPEST、DE_BILT、DRESDEN、DUSSELDORF、HEATHROW、KASSEL、LJUBLJANA、MAASTRICHT、MALMO、MONTELIMAR、MUENCHEN、OSLO、PERPIGNAN、ROMA、SONNBLICK、STOCKHOLM、TOURS。
- **变量**：`temp_mean/max/min`（`temp_max` 全 18 站可得，`temp_min` 缺 BUDAPEST）、`cloud_cover`、`wind_speed`、`wind_gust`、`humidity`、`pressure`、`global_radiation`、`precipitation`、`sunshine`（部分站点有）。
- **上游已做的预处理**（报告需显式声明并讨论）：
  - 缺失率 > 5% 的列被整列删除；≤ 5% 的列以**均值填补**（作者注明这是对原始数据的操纵）。
  - 单位"压范围"但**未标准化**：温度 °C、风 m/s、湿度 0~1、海平面气压 ×1000 hPa、辐射 ×100 W/m²、降水 ×10 mm、日照 0.1 h。
  - 校验：当前数据已无 `-9999`（0 格），但 EDA 发现 **77 个物理不可能值**（哨兵残留等，见 `docs/data_analysis.md` §3.2）。
- **主表**：`DATE` + `MONTH` + **163 个特征**（18×11 理论格子缺 35）。
- **附带分类标签**：`weather_prediction_picnic_labels.csv`（17 站点 picnic 布尔，无 ROMA）——**主线不用**（§4.2 已否决），仅供对照讨论。
- **许可与引用**：MIT；须引 Klein Tank et al. (2002) 与 Huber et al.

---

## 3. 已下载的数据清单

```
data/
├── weather_prediction_dataset.csv          2.7 MB  主表 3654 × 165（DATE, MONTH, 163 features）
├── weather_prediction_picnic_labels.csv    0.4 MB  17 站点 picnic 布尔标签（主线不用）
├── weather_prediction_dataset_map.png      1.7 MB  18 站点位置地图
├── metadata.txt                            4.6 KB  上游字段与单位说明
├── README_upstream.md                      6.6 KB  上游仓库 README
└── LICENSE                                 1.1 KB  上游 MIT License
```

（未纳入仓库：上游 `notebooks/` 教学示例——"官方参考答案"性质，其任务选择已被本仓库否决，见 §10。）

---

## 4. 问题形式化（Problem Formulation）——主线：日际变温 ΔT_max

### 4.1 主任务：跨站点预测 BASEL 次日最高温的日际变化

- **定义输入**：第 $t$ 天全部 18 个站点的观测向量 $x_t \in \mathbb{R}^{163}$，及工程特征（上游站 Δ、气压梯度，见 spec §2.4）
- **预测目标**：$$y_t = \Delta T_{\max}(t+1) = T_{\max}(t+1) - T_{\max}(t)\quad (\text{BASEL})$$
- **学习目标**：$f: \mathbb{R}^{163} \to \mathbb{R}$，最小化 $\mathbb{E}[\ell(f(x_t), y_t)]$，$\ell$ 取 Huber/MAE
- **等价性论证（报告必写）**：若 $T_{\max}(t)$ 在输入中，预测 Δ 与预测 $T(t+1)$ 属同一假设空间；本形式化是把 **persistence 从模型手里拿走、放进评价里**——Δ 空间 persistence 恒为 0，skill 第一次有物理意义。
- **为什么选它**：
  1. 原始 $T$ 的 74.4% 方差是季节循环，且 R²=0.92 可被 persistence 白拿（EDA §5.3/§7.4）——"预测原始 T"展示的是日历+惯性；
  2. 教学 notebook 主实验是原始 T / 日照，Kaggle 已有原始温度预测；ΔT 避开重复，差异化落在评估框架；
  3. 物理对应冷暖空气平流：probe 实测上游站 Δ 特征 train 相关 TOURS 0.349 / MAASTRICHT 0.201 / DE_BILT 0.190（`docs/delta_t_probe_lead.csv`），西风带平流叙事成立；
  4. $\Delta T_{\max}$ 用全 18 站都有的 `temp_max`，无需缺测多的列。
- **可行性已验证**（2026-09-22，R0 probe）：ridge 全量特征 test MAE **1.867** vs zero-change 基线 **2.393**（skill **+22%**，R²=0.431）。详见 `docs/progress/2026-09-22_delta_t_formulation_decision.md`。

### 4.2 扩展任务与已否决清单

**扩展（可选，按序）**：

| 方向 | 定义 | 价值 |
|---|---|---|
| 多站合并 ΔT_max | 18 站 × 366 天 = 6582 station-days 联合建模 | Δ 空间站点气候偏移抵消，跨站合并第一次物理正当；缓解测试期功效 |
| 多步长 h=1..10 | 预测 ΔT_max(t+h) | skill 衰减曲线 = 可预报性极限叙事 |
| 次日降水发生 | 0/1 分类 | 低技能任务的诚实评估（lag-1=0.211），BSS/PR-AUC |

**已否决（2026-09-22 决策，理由进报告 discussion）**：

| 目标 | 原因 |
|---|---|
| 原始 T_mean / T_max 次日回归 | persistence 污染 + Kaggle/教学 notebook 已做 |
| 日照时长回归 | 教学 notebook 主实验 |
| picnic / 月份分类 | 另两份教学 notebook；picnic 是"无雨且暖"的派生规则 |
| 降水量数值回归 | 零膨胀 35–75%，篇幅内讲不清 |

### 4.3 评估协议（红线，违反 = 报告作废）

- **严禁随机划分**：日序数据 lag-1 自相关 0.957。唯一合法切分：**train 2000–2007（2922 天，Δ 对齐后 2920 样本）/ val 2008（366）/ test 2009-01-01～2010-01-01（366）**；验证集用于选模型、早停、选配置。
- **基线阶梯**：zero-change（Δ 空间 persistence，**主门槛**，test MAE 2.393 °C）→ yesterday-Δ（预期惨败，坐实日际变化近白噪声 lag-1=0.075）→ monthly-Δ 气候（≈zero）→ Ridge/Lasso → RF/HistGBM → MLP。
- **指标**：MAE（°C，主指标，重尾稳健）、RMSE、R²（Δ 空间≈skill vs 均值）、**skill = 1 − MAE_model/MAE_zero**、corr(Δ̂,Δ)（常数基线记 "—"）。分季节附报。
- **不确定性**：moving block bootstrap（block=7 天 ≥ 气压记忆 3–5 天），95% CI；模型间差异用同块配对 bootstrap。
- **泄漏控制**：清洗/填补/标准化/气候统计一律 train-only 估计再施加到 val/test；特征只用第 t 天及以前；**哨兵值（−99 等）必须先清洗再差分**，否则 Δ 特征出现 ±100 °C 假尖刺。
- **必须讨论的坑**：ΔT 重尾（锋面过境 ±10 °C）→ MAE/Huber；冬季方差大于夏季 → 分季节报告；测试期仅 366 天 → bootstrap CI + 多站合并扩展；上游均值填补对 ≤5% 缺失列的方差压缩效应。

---

## 5. 模型设计要点（Model）

按"基线→线性→MLP"阶梯组织，**100% PyTorch**（2026-09-22 框架决策，砍掉树模型；完整定义见 spec v1.2 §2.3）：

1. **基线**：zero-change、yesterday-Δ、monthly-Δ 气候——界定"什么叫预测得有用"。
2. **线性（torch）**：L2 封闭解 Ridge（`torch.linalg.solve`，瞬时且精确）+ L1 Lasso（梯度，R2）。**α 用 val 2008 选择**（替代 sklearn 内部 CV，符合"无时间洗牌"协议）。系数即物理通道证据（上游 Δ、气压梯度、风/湿）。
3. **MLP（专门训练的主模型）**：小网格 {32}/{64,32}/{128,64}，GELU，Dropout∈{0.2,0.4}，weight decay∈{1e-4,1e-3}，Huber(δ=1×train σ_Δ)，Adam lr 1e-3，batch 64，EarlyStopping(val MAE, patience 30)，种子 {0,1,2} 报 mean±std，按 mean val MAE 选配置；CUDA 可用。定位：**容量上限对照**——相对线性提升有限（~2.9k 样本 × 低 SNR）本身是结论（样本量-容量权衡），写入 discussion。
4. **消融矩阵**：全 163 levels+Δ / 仅上游 7 站子集 / 去 Δ 特征 / 去气压梯度 / 季节特征（受控对照，默认关）/ 缺失指示特征（默认关）。

---

## 6. 交付物与提交清单

- [ ] `report.pdf` —— **英文**，学术报告格式；必须包含 problem formulation 与 model；建议章节：Introduction / Dataset / Problem Formulation / Methodology / Experiments / Discussion & Limitations / Conclusion / References
- [ ] 代码 —— 可复现（含依赖说明、随机种子、一键运行入口），与报告图表/数字一一对应
- [ ] CV
- [ ] 打包为 `Task Report_<YourName>.zip`
- [ ] 邮件发送至 `yubing.pan@connect.polyu.hk`，主题 `Task Report Submission_<YourName>`

**提交前自检**：ZIP 内代码能否干净环境跑通？报告数字是否与代码输出一致（溯源到 docs/progress 记录）？PDF 所有图是否有编号、单位、图注？数据来源与引用是否声明？

---

## 7. 时间线（2026-09-22 重排，R 阶段对应 spec §5 的 Wave）

| 阶段 | 内容 | 完成期限 |
|---|---|---|
| R0 | formulation 决策落盘 + task_brief 改向 + ΔT_max 可行性 probe（skill +22% 已确认） | 09-22 ✅ |
| R1 | 正式数据管线（清洗→填补→差分→标准化）+ 基线表 | 09-23 |
| R2 | Lasso(torch) 变量选择 + 特征组消融成表 | 09-25 |
| R3 | MLP 网格 + 3 种子 + 多站合并扩展 | 09-26 |
| R4 | block bootstrap CI + 消融汇总 + 图表 | 09-27 |
| R5 | 报告成文（骨架自 R1 起并行撰写）+ 打包自检 + 发送 | **09-29**（留 1 天缓冲） |

---

## 8. 参考与引用

- 数据仓库：https://github.com/florian-huber/weather_prediction_dataset （MIT）
- Zenodo 备份：https://doi.org/10.5281/zenodo.7525955
- 上游教学 notebook（任务选择已被否决，仅参考其基线设定）：`notebooks/deep_learning_regression_BASEL_tomorrow_10years.ipynb` 等
- 必引文献：
  - Klein Tank, A.M.G. and Coauthors, 2002. *Daily dataset of 20th-century surface air temperature and precipitation series for the European Climate Assessment.* Int. J. of Climatol., 22, 1441–1453.
  - Florian Huber, Dafne van Kuppevelt, Peter Steinbach, Colin Sauze, Yang Liu, Berend Weel. *Will the sun shine? – An accessible dataset for teaching machine learning and deep learning.*

---

## 9. 风险与待确认

1. **低 R² 被误读**：ΔT 的 R² 天花板由大气可预报性决定（日际变化 lag-1=0.075），报告须预设"可预报比例"叙事，全部数字用 skill vs zero + CI 呈现。
2. **样本量-容量权衡**：~2.9k 训练样本下 MLP 过拟合是主要风险——小网格、early stopping、3 种子、val 裁决；负结果是合法结论。
3. **测试期功效**：366 天、MAE 标准误 ≈ 0.17 °C（ΔT 尺度）；0.1 量级"提升"须 bootstrap CI 佐证；多站合并扩展可翻倍功效。
4. **Kaggle 重叠待核**：`bernie21/temperature-prediction` 的具体内容未核实（网络受限）；即使做过原始 T，本工作差异化在 Δ 空间评估框架 + 平流通道物理解释。建议提交前再确认。
5. **命名/语言/篇幅**：`YourName` 与 CV 一致；报告英文；建议 8–12 页，重心给 formulation 与 methodology。
6. **排期零冗余**：任何阶段 overrun 都吃掉 09-30 缓冲——按 R1→R4 从后往前砍单（见 progress 记录 §3.2）。

---

## 10. 变更日志

- **2026-09-22 框架决策**：建模 100% PyTorch，砍掉树模型；运行时 = Windows Python 3.12（`D:\Program Files\Pythons\python3.12`，torch 2.13+cu126）。线性 L2 用封闭解、α 用 val 2008 选择。R1 复算：linear_lvl+d_all163 test MAE **1.827**（sklearn 为 1.851）。记录：`docs/progress/2026-09-22_framework_decision_torch.md`。
- **2026-09-22 主线改向**：主任务由"原始 BASEL_temp_mean(t+1) 回归"（原 §4.1）改为"**日际变温 ΔT_max(t+1) 预测**"。理由：原始 T 被 persistence 与季节循环污染（74.4% 方差是日历）、教学 notebook 与 Kaggle 已覆盖；Δ 空间 skill 第一次有物理意义，且上游站平流信号已被 R0 probe 证实。已否决日照/picnic/月份/原始气温/降水量数值，保留降水发生为 extension。
  - 决策全记录：`docs/progress/2026-09-22_delta_t_formulation_decision.md`
  - 工程 spec（oracle 审查 v1.1）：`docs/superpowers/specs/2026-09-22-delta-t-max-spec.md`
  - R0 probe：`analysis/delta_t_probe.py` → `docs/delta_t_probe.csv` / `docs/delta_t_probe_lead.csv`
  - 注意：§2/§3 中数据集事实与 163 特征描述对 formulation 无关，保持原样；`temp_min` 缺 BUDAPEST 不影响主线（用 `temp_max`，18/18 站可得）。
