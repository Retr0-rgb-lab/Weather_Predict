# 研究总结：日际变温 ΔT_max 预测（截至 2026-09-22）

> 本文档用朴素语言解释整个研究的设计、实施与发现，给未来的我自己/同事读。
> 配套代码在 `src/weatherlib/`（共享实验库）与 `analysis/`（每次实验一个脚本）；详细实验轨迹见 `docs/progress/` 其它按日期命名的记录；最终结果在 `docs/r4_final_table.csv` 与 `docs/r4_pairwise.csv`。

---

## 1. 研究对象

我们要预测的目标是：**巴塞尔明天最高温比今天高（低）几度**。记成公式：

> ΔT_max(t+1) = T_max(t+1) − T_max(t)

这不是预测"明天多少度"，而是预测"明天比今天**变**几度"。

**为什么预测"变化"而不是"原始温度"？**
直接预测明天气温是送分题——明天大约等于今天（天气有惯性），再叠加"一月冷、七月热"的季节循环，任何模型抄今天的数 + 日历就能拿到 R² ≈ 0.92。这种"准"是日历+惯性给的，不是模型本事。我们换成"变化量"后：
- 惯性预测退化成"预测不变"（值 = 0），成为必须**打过的及格线**；
- 季节循环在变化量里几乎消失（平均一天既不开温也不降温），没有"日历免费午餐"；
- 第一次真正测出模型学到多少气象知识。

**数据集**：18 个欧洲气象站（2000–2010，3654 天）的日观测，原始由 florian-huber/weather_prediction_dataset 发布（基于 ECA&D），包含温度、气压、湿度、风速、辐射等。主表 165 列，去掉 DATE/MONTH 后剩 **163 个特征**（18 站 × 11 变量，缺 35 格）。许可 MIT；上游做了缺失填补与单位缩放但**未标准化**——这点后面会用上。

---

## 2. 实验方法与协议

为了结果可被信赖，我们立了几条硬规矩（详细佐证见 `docs/data_analysis.md`）：

**切分**：日序时间序列，自相关很高，相邻日 shuffle 会泄漏信息。**唯一合法**切分：
- 训练 2000–2007（**2920 样本**，对齐后）
- 验证 2008（**366**）
- 测试 2009-01-01 至 2010-01-01（**366**，2010 年实际只有 1 天边界样本）

**统计量**：清洗参数、填补中位数、模型 α 都只用训练期估计，再施加到验证与测试——严格无泄漏。

**指标**：以 **MAE（°C）为主**（目标分布重尾，MAE 抗噪）；附 RMSE、R²、skill vs zero、Corr。重尾下选 MAE 而非 MSE。366 天测试期上 MAE 的标准误约 0.17 °C——任何 0.1 量级的"提升"必须在 block bootstrap 95% CI 下显著。

**不确定性**：moving block bootstrap（**block = 7 天** ≥ 气压记忆 3–5 天，覆盖天气尺度去相关）；模型间差异用**同块配对** bootstrap。

**基线门槛**：**zero-change**（Δ 空间里的"惯性预测"，预测 = 0）；test MAE 2.393 °C。任何模型都必须报告 skill = 1 − MAE_model/MAE_zero + bootstrap CI。

**运行栈**：模型与分析全程用 **Windows Python 3.12**（含 torch 2.13 + CUDA、numpy、pandas），从 WSL 经 `/mnt/d/Program Files/Pythons/python3.12/python.exe` 调用。**100% PyTorch**——你的选择，树模型砍掉，线性 L2 用 `torch.linalg.solve` 封闭解（秒级、精确），α 用验证集选（不用 sklearn 内部 CV，避免跨时间褶皱）。

---

## 3. 实验过程（一条流水线，从 idea 到终表）

按时间顺序，每一步生成 commit + progress 记录 + 推送，证据齐全：

| 阶段 | 做了什么 | 关键产物 |
|---|---|---|
| EDA（09-19） | 摸底：163 特征、77 异常格、季节循环 74.4%、persistence 1.667/R²0.92（原始 T）、0.075 日变化自相关、最强 lead 相关 MAASTRICHT 0.40 | `analysis/eda.py`、`docs/data_analysis.md` |
| formulation 探针（09-21） | 距平空间下 h=1..10 测时间记忆 vs 空间记忆 → 全量 ridge h=1 RMSE 1.807 vs persistence 2.288 vs clim 3.378 | `analysis/formulation_probe.py`、两个 CSV |
| **R0 决策（09-22）** | 把主任务从原始 T 改向 **ΔT_max**（evidence: persistence 污染、Kaggle/教学已做、平流通道成立）| `docs/progress/2026-09-22_delta_t_formulation_decision.md` |
| R0 probe | ΔT_max 可行性：zero 2.393 / yesterday −44% / ridge 全量 1.867 skill +22% | `analysis/delta_t_probe.py` |
| 框架决策 | 切到 Windows py3.12，100% PyTorch，砍树，封闭解；re-run 后 linear 1.827 | `docs/progress/2026-09-22_framework_decision_torch.md` |
| **R1 正式管线 + 基线表** | 共享库 `weatherlib` + 基线×3 + 线性×6 → `docs/r1_baseline.csv`；77 格掩蔽与 EDA 完全吻合 | `src/weatherlib/`、`analysis/r1_baselines.py` |
| **R2 Lasso + 消融** | 10 组特征消融（含受控 season/missing）+ Lasso L1 系数分析 + 压缩模型 top25 | `analysis/r2_lasso_ablation.py`、`docs/r2_ablation.csv`、`r2_lasso_coefs.csv` |
| **R3 MLP 网格 + 多站合并** | 12 配置 × 3 种子 MLP（容量上限对照）+ pooled 多站合并（负结果）| `analysis/r3_mlp.py`、`docs/r3_mlp.csv`、`r3_pooled.csv` |
| **R4 最终表 + 配对显著性 + 图** | 规范表 + 配对 CI + 5 张报告图 | `analysis/r4_eval_figures.py`、`docs/r4_{final_table,pairwise}.csv`、`docs/figures/fig10..14` |

---

## 4. 特征工程

从原始 163 个观测列出发，按以下顺序构造模型输入：

### 4.1 清洗（先于差分）
哨兵值如果先做差分会产生 ±100 °C 假尖刺污染模型，所以我们**先清洗后差分**。总掩蔽 **77 格**：
- cloud_cover 3 格（STOCKHOLM 残存 −99 等）
- sunshine 29 格（STOCKHOLM −1.7）
- pressure 4 格（STOCKHOLM −0.099×3 + TOURS 0.0003×1）
- temp_min > temp_max 41 格（HEATHROW 40 + ROMA 1；成对置 NaN 后填补）

填补用**训练期各月站点×变量中位数**（升级自 R0 的月均值）。

### 4.2 特征块（`make_delta_max_dataset` 一次产出）
- **levels（163）**：第 t 天全站原始观测，标准化（train-only 均值/方差）。
- **deltas（163）**：第 t 天各变量相对 t−1 的变化量——核心信号。
- **grads（7）**：上游 7 站（西风带：MAASTRICHT/TOURS/DUSSELDORF/MUENCHEN/KASSEL/DE_BILT/HEATHROW）的气压减去巴塞尔气压——**地转风/平流代理**。
- **season（2，受控对照默认关）**：日序的 sin/cos 编码。
- **missings（163，受控对照默认关）**：清洗前的缺失指示。

"受控对照默认关"的意思是：主结果**不**用 season 和 missings；它们出现在消融表里，是为了证明"Δ 空间没有日历免费午餐"和"缺失指示无用"——这是协议的一部分，不是装饰。

### 4.3 特征集定义（消融矩阵的列）
| 特征集 | 用法 |
|---|---|
| levels_all163 | 163 levels 单独 |
| deltas_all163 | 163 Δ 单独 |
| lvl+d_all163 | levels+Δ = 326，主实验输入 |
| lvl+d+grad | 333 |
| lvl+d+grad+season | 335（受控）|
| lvl+d+grad+missing | 496（受控）|
| up7_lvl+d | 上游 7 站子集（14 特征）|
| up7_d | 上游 7 站 Δ（7 特征）|
| own_lvl+d | 仅本站（2 特征）|

**事实**：162/163 特征两两相关 |r|>0.95 的有 62 个（含 163 维）；所以"全量"与"精选子集"的对比才是真问题，而不是简单的"维度越高越被低劣"。

---

## 5. 模型选择

### 5.1 模型阶梯（"基线 → 线性 → 树 → MLP"简化成"基线 → 线性 → MLP"）
树模型（RF/HistGBM）你主动砍掉（torch 里没有，且参数化希望统一）。阶梯：

1. **基线 ×3**
   - zero-change：预测 0（主门槛，MAE 2.393）
   - yesterday_Δ：今天的变温当明天的（预期失败；实测 −44%）
   - monthly_Δ climatology：每月平均变温（≈0）

2. **线性（L2 封闭解 Ridge + alpha L1 Lasso，PyTorch）**
   - L2 用 `torch.linalg.solve`（精确、无梯度循环、瞬时）
   - L1 用 Adam + L1 惩罚（α 由验证集选）
   - α 网格用 logspace(−2, 4, 25) 或 geomspace(1e-3, 1, 8)
   - 系数即物理通道证据（这是线性最大的卖点——可解释）

3. **MLP（容量上限对照）**
   - 输入：levels+deltas（326）；标准化后 y 按 train σ_Δ 标准化
   - 架构网格：hidden {32} / {64,32} / {128,64}；GELU+Dropout∈{0.2, 0.4}；wd∈{1e-4, 1e-3}；Huber(δ=1×train σ_Δ)；Adam lr 1e-3, batch 64
   - EarlyStopping(val MAE, patience 30, max 300)
   - 3 种子（0,1,2）报 mean±std + ensemble

4. **多站合并扩展**（pooled 18 站 × 3652 = 65736 station-days）
   - 用同一全局特征场（326 features）作为输入，每个站点的 ΔT_max 为独立目标
   - Δ 空间站点气候偏移抵消 → 跨站合并"理论上"正当

### 5.2 选参与评估的协议（"test 只评一次"硬规矩）
- 模型选 α 全部在 val 2008 上做；test 仅用于**最终**评估；
- MLP 配置由 mean val MAE 选；选定后 test 取 ensemble + 配对 block bootstrap CI；
- 模型间差异显著性用同块配对 bootstrap，CI 不含 0 即为 sig。

---

## 6. 实验结果（核心数字与结论）

### 6.1 模型阶梯（test 366 天，全带 95% block-CI）

| 模型 | MAE [CI] | skill vs zero |
|---|---|---|
| zero（门槛） | 2.393 [2.104, 2.680] | — |
| yesterday_Δ | 3.447 | **−44.0%** |
| linear (lvl+d) | 1.827 [1.623, 2.014] | **+23.7%** |
| Lasso(L1, 326) | 1.773 [1.574, 1.955] | **+25.9%** |
| **MLP ensemble** | **1.751 [1.546, 1.946]** | **+26.9%** |

### 6.2 配对显著性（同块 bootstrap）
- 所有学习模型 vs zero：**显著**（CI 全部不含 0）
- linear vs lasso：−0.054 [−0.101, 0.002] **不显著**
- lasso vs mlp：−0.023 [−0.119, 0.068] **不显著**
- mlp vs linear：−0.076 [−0.149, 0.000] **不显著**（临界）
- **全量 vs 上游 7 站：−0.229 显著** → 完整场信息确实优于子集

### 6.3 消融表里的故事
- **levels 22.9% > deltas 17.4% > 合 23.7%**：单独 levels 比 deltas 强，但合起来再涨 1pp
- **grads 在全量下几乎没用**（+grad 1.829 vs 1.827）但在**上游子集上 +1.7pp**（up7_d 11.6% → up7_d+grad 13.3%）：梯度信息被上游 Δ 互补
- **season 23.4% < 不加 23.7%** —— 季节特征不加（受控消融干净：Δ 空间没有日历免费午餐）
- **missing 指示 23.0% < 不加 23.7%** —— 缺失指示无帮助
- **本站 lvl+d 0.0%** —— 仅靠本站观测做不出预测

### 6.4 Lasso 物理通道（top 标准化系数）
变温的主要驱动力是**气压场**及其日变化：
- d_BASEL_pressure +0.28, d_HEATHROW_pressure −0.30, d_MONTELIMAR_pressure +0.23, TOURS_pressure −0.17
- 次级是上游湿度：d_MAASTRICHT_humidity +0.13, DRESDEN_humidity −0.12
- **诚实发现**：Lasso 在验证最优 α 下 324/326 系数非零 → 可预报信号是**分布式**的，不是几个魔法特征；但 top25 重拟合仍保留 22.3% skill（vs 全量 23.7%），说明有一个可解释的"核心"

### 6.5 多站合并（负结果，discussion 素材）
| 模型 | MAE (6588 station-days) | skill vs pooled zero |
|---|---|---|
| pooled zero | 2.186 | — |
| **per-station L2（18 个独立模型）** | **1.705** | **+22.0%** |
| 单一全局 pooled L2 | 2.066 | +5.5% |
| pooled MLP | 2.038 | +6.8% |

**单一全局模型比 18 个独立模型差 0.36 °C**（远超 CI）：每个站对同一气压场的响应不同，全球共享一个权重向量装不下这种"地方特性"——这是有价值的负面结果，撑得起讨论一节。

### 6.6 总体结论（一句话）
在预测"明天比今天热/冷几度"这个任务上，**学习模型（线性 / Lasso / MLP）能在 zero-change 基线之上稳定、相似地多拿 ~24–27% 的 skill**，差异在噪声内——这 ~27% 的可预测比例是**大气可预报性的边界**，不是模型容量的问题。信号来源是大尺度环流（西风带上游平流 + 气压场），而非本站自身（本站观测 skill 0%）。这正好印证了"日际变温大部分是天气尺度的随机噪声"这一气象常识——把基线剥干净后，模型确实在物理上说得通的。

---

## 7. 后续工作

- **R5（09-29 前）**：英文报告成文（formulation + model 是评分重心，篇幅重心），引用 Klein Tank et al. (2002) 与 Huber et al. *Will the sun shine?*；把本文档浓缩为正文 + 数字与图表一一对应；
- **打包自检**：ZIP 在干净环境跑通；报告数字与 `docs/r4_final_table.csv` 一致；所有图有编号、单位、图注；声明上游数据 commit `83d70ee`；
- **发送**：邮件 `yubing.pan@connect.polyu.hk`，主题 `Task Report Submission_<YourName>`，附件 `Task Report_<YourName>.zip`。