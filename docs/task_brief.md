# Weather Prediction 任务说明

> 任务来源：Ms. PAN Yubing（yubing.pan@connect.polyu.hk）课题组面试前的筛选作业
> 文档整理日期：2026-09-19 ｜ 提交截止：**2026-09-30**
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

解读：这是一份"卡门槛"的筛选作业。评审重点明确写在了括号里——**problem formulation（问题如何定义）与 model（模型设计与选择）**，而非刷指标。实验结果"可选"，说明报告的逻辑完整性 > 精度数字。因此报告应把"我为什么这样定义问题、为什么选这个模型、如何避免自欺欺人的评估"讲清楚。

---

## 2. 项目背景：这个数据集是什么

数据集由 Florian Huber 等人构建，**目标是教学用途**（用于 ECML 2022 "Teaching Machine Learning" workshop），因此刻意设计成"小而真实"：单机笔记本几分钟就能训完，但任务难度可调，并且天然包含真实数据中的坑（过拟合、类别不平衡、缺失值、分布漂移）。

- **原始来源**：ECA&D（European Climate Assessment & Dataset）地面站点日观测，file created 22-04-2021。
- **时空范围**：18 个欧洲站点，2000-01-01 ~ 2010-01-01，共 **3654 天**。
- **站点**：BASEL、BUDAPEST、DE_BILT、DRESDEN、DUSSELDORF、HEATHROW、KASSEL、LJUBLJANA、MAASTRICHT、MALMO、MONTELIMAR、MUENCHEN、OSLO、PERPIGNAN、ROMA、SONNBLICK、STOCKHOLM、TOURS。
- **变量**：`temp_mean/max/min`（全站点都有）、`cloud_cover`、`wind_speed`、`wind_gust`、`humidity`、`pressure`、`global_radiation`、`precipitation`、`sunshine`（部分站点有）。
- **上游已做的预处理**（重要，写报告时需要显式声明并讨论）：
  - 缺失率 > 5% 的列被整列删除；缺失率 ≤ 5% 的列，缺失值（原始标记 `-9999`）**用均值填补**。作者自己在 metadata 里注明了这是对原始数据的操纵，仅出于教学目的。
  - 单位做了"压范围"但**没有标准化**：温度 °C、风速/阵风 m/s、湿度 0~1 比例、海平面气压 ×1000 hPa、全球辐射 ×100 W/m²、降水 ×10 mm、日照 0.1 h。→ 不同特征的数值尺度差异很大，建模时是否标准化是一个需要交代的设计决策。
  - 校验结果：当前数据中已不存在 `-9999`（0 个单元格），说明填补已完成。
- **实操细节**：18 个站点的列**不等宽**（有的站点缺 `cloud_cover`、`sunshine`、`wind_*`）。BUDAPEST 缺 `temp_min`，因此主表共 `DATE` + `MONTH` + **163 个特征**，不是 18×11。
- **附带的分类标签**：`weather_prediction_picnic_labels.csv` 给出 17 个站点"当天是否适合野餐"的布尔标签（无 ROMA），可用于分类任务。
- **许可与引用**：MIT License；数据集本身要求引用 ECA&D 的 Klein Tank et al. (2002) 以及数据集作者的论文。报告中应给出引用。

---

## 3. 已下载的数据清单

```
data/
├── weather_prediction_dataset.csv          2.7 MB  主表 3654 × 165（DATE, MONTH, 163 features）
├── weather_prediction_picnic_labels.csv    0.4 MB  17 站点 picnic 布尔标签，3654 行
├── weather_prediction_dataset_map.png      1.7 MB  18 站点位置地图（报告配图可直接用）
├── metadata.txt                            4.6 KB  上游数据说明与单位定义
├── README_upstream.md                      6.6 KB  上游仓库 README（原文留存）
└── LICENSE                                1.1 KB  上游 MIT License
```

（未纳入仓库：上游 `notebooks/` 下的教学示例 notebook。它们是"官方参考答案"性质的基线实现，若需要可再次从 GitHub 拉取，见第 8 节。）

---

## 4. 问题形式化（Problem Formulation）

报告的核心。给出一套可直接落笔的定义：

### 4.1 推荐主任务：跨站点次日气温回归

- 定义输入：第 $t$ 天全部 18 个站点的全部观测向量 $x_t \in \mathbb{R}^{163}$
- 预测目标：**BASEL 站点第 $t+1$ 天的 `temp_mean`**，即 $y_t = \text{BASEL\_temp\_mean}(t+1)$
- 学习目标：$f: \mathbb{R}^{163} \to \mathbb{R}$，最小化 $\mathbb{E}[\ell(f(x_t), y_t)]$，$\ell$ 取 MSE 或 MAE
- 为什么选它：目标变量与某个输入特征强相关（`BASEL_temp_mean(t)`），存在明确且不平凡的**持续性基线**（persistence），因此"精度"有解释力；同时任务有多站点、多物理量的输入结构，足以体现建模取舍。这也是上游教学 notebook 采用的设定，便于对标与说明。
- 物理可解释性：跨站点的气压/辐射/湿度场含有次日气温的可预报信息，模型选择与特征重要性可以直接对照气象常识来讨论。

### 4.2 可选扩展任务

| 方向 | 定义 | 价值 |
|---|---|---|
| 多目标回归 | 一次预测 18 个站点的次日 `temp_mean` | 展示多任务学习 / 空间相关性建模 |
| 月分类 | 由当日观测预测 `MONTH`（12 类） | 上游示例任务之一，考察不平衡与特征工程 |
| picnic 二分类 | 预测某站点当日是否适合野餐 | 天然类别不平衡，适合讨论 PR-AUC、阈值选择 |
| 多步序列预测 | 用过去 $k$ 天序列预测未来第 $h$ 天 | 可引入 LSTM / TCN / Transformer，但需注意样本量仅 3654 |

建议：**主任务做 4.1，再补一个分类任务作为扩展**，这样报告既能体现回归建模的严谨，也能展示对不平衡问题与评估指标选择的理解。

### 4.3 评估协议（必须写清，也是最能体现功力的部分）

- **严禁随机划分**：这是日序时序数据，随机 shuffle 会造成相邻日跨 train/test 的信息泄漏。必须按时间切分（如 2000–2007 训练 / 2008 验证 / 2009–2010 测试），并说明验证集用于选模型与早停。
- **基线对照**：至少包含
  - Persistence：$\hat{y}_t = \text{BASEL\_temp\_mean}(t)$
  - 气候学基线：以训练集中"同月均值"作为预测
  - 线性模型（Ridge / Lasso）
  - 树模型（Random Forest / Gradient Boosting）
  - 神经网络（MLP，注意 163 维输入、3654 样本下的过拟合风险）
- **指标**：MAE、RMSE、$R^2$。回归要求报告**相对于 persistence 基线的提升**，否则数字没有意义。
- **必须讨论的坑**：目标自相关导致的指标虚高；均值填补对 ≤5% 缺失列引入的偏差；单位不统一对距离类模型（KNN、SVM、神经网络）的影响；单年异常天气（极端寒潮/热浪）在测试集上的影响。

---

## 5. 模型设计要点（Model）

报告需说明"为什么是这些模型"，建议按以下层次组织：

1. **物理/统计基线**：persistence 与同月气候均值 —— 界定"什么叫预测得有用"。
2. **线性模型**：Ridge/Lasso，回答"是否线性可分信息足够"，Lasso 顺便做变量选择，可讨论哪些站点/变量被选中（可解释性）。
3. **树集成**：Random Forest / Gradient Boosting（如 XGBoost/LightGBM），通常是最强非时序基线，特征重要性可对照气象常识。
4. **神经网络**：MLP（BatchNorm + Dropout + EarlyStopping），说明正则化手段的必要性；若做序列扩展再用 LSTM/TCN。
5. **可选**：多任务/多站点联合建模，或加入时间特征（day-of-year 的正弦编码）——注意这会让"预测"退化为"记忆季节性"，需在报告中明确区分并说明其合理性边界。

---

## 6. 交付物与提交清单

- [ ] `report.pdf` —— **英文**，学术报告格式；必须包含 problem formulation 与 model；建议章节：Introduction / Dataset / Problem Formulation / Methodology / Experiments(可选) / Discussion & Limitations / Conclusion / References
- [ ] 代码 —— 可复现（含依赖说明、随机种子、一键运行入口），与报告中的图表/数字一一对应
- [ ] CV
- [ ] 打包为 `Task Report_<YourName>.zip`
- [ ] 邮件发送至 `yubing.pan@connect.polyu.hk`，主题 `Task Report Submission_<YourName>`

**提交前自检**：ZIP 内文件能否在干净环境跑通？报告里的数字是否与代码输出一致？PDF 里是否所有图都有编号、单位、图注？是否声明了数据来源与引用？

---

## 7. 时间线建议（今天 2026-09-19，剩 11 天）

| 阶段 | 内容 | 建议完成 |
|---|---|---|
| P0 | 数据核验、基线（persistence/线性）跑通、评估协议定稿 | 09-21 |
| P1 | 问题形式化与报告骨架（Dataset / Formulation / Methodology 三章成文） | 09-23 |
| P2 | 树模型 + MLP 实验、消融（是否标准化、是否加时间特征、特征子集） | 09-26 |
| P3 | 图表定稿、报告成文、代码整理为可复现结构 | 09-28 |
| P4 | 打包、自检、发送邮件 | **09-29**（留 1 天缓冲，勿压 09-30 当天） |

---

## 8. 参考与引用

- 数据仓库：https://github.com/florian-huber/weather_prediction_dataset （MIT License）
- Zenodo 备份：https://doi.org/10.5281/zenodo.7525955
- 上游教学 notebook（本仓库未收录，如需可拉取）：
  `notebooks/deep_learning_regression_BASEL_tomorrow_10years.ipynb`（对应 4.1 的回归任务）、
  `notebooks/machine_learning_classification_months.ipynb`、`notebooks/machine_learning_classification_bbq.ipynb`
- 拉取命令：`git clone --depth 1 https://github.com/florian-huber/weather_prediction_dataset.git`
- 必引文献：
  - Klein Tank, A.M.G. and Coauthors, 2002. *Daily dataset of 20th-century surface air temperature and precipitation series for the European Climate Assessment.* Int. J. of Climatol., 22, 1441–1453.
  - Florian Huber, Dafne van Kuppevelt, Peter Steinbach, Colin Sauze, Yang Liu, Berend Weel. *Will the sun shine? – An accessible dataset for teaching machine learning and deep learning.*

---

## 9. 风险与待确认

1. **命名占位符**：ZIP 与邮件主题中的 `YourName` 需替换为实际姓名（与 CV 一致，建议 `Task Report_<GivenName FamilyName>`）。
2. **报告语言**：邮件用英文，报告建议英文；若考虑中文可先与 Ms. PAN 确认。
3. **时长与篇幅**：邮件未规定页数。建议 8–12 页（含图表与参考文献），重点篇幅给 formulation 与 methodology。
4. **是否需要实验**：写的是 optional。但"模型"章节若无任何数值支撑会显得空泛，建议至少给出 §4.3 的基线与主模型对照表。
5. **数据版本**：本仓库数据取自主仓库 `main` 分支，克隆于 2026-09-19，commit `83d70ee054d538ec3700948367b921c0f6a08874`，与 Zenodo 版一致；报告中可标注该版本以保证可复现。
