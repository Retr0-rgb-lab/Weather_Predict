# AGENTS.md

> 本项目 = PolyU Ms. PAN Yubing 课题组的筛选作业（面试门槛）。**截止 2026-09-30**。
> 交付：英文 PDF 报告（评审核心是 problem formulation 与 model）+ 可复现代码 + CV → 打包 `Task Report_<YourName>.zip` → 邮件 `yubing.pan@connect.polyu.hk`（主题 `Task Report Submission_<YourName>`）。
> 任务全貌与时间线（P0–P4）见 `docs/task_brief.md`，先读它再动手。

## 0. 最高优先级规则：一切实验轨迹必须留痕

**所有代码痕迹、实验过程、问题与解法尝试，都必须落盘到 `docs/progress/` 并 commit + push 到 GitHub**（remote 已配置：`github.com/Retr0-rgb-lab/Weather_Predict`，branch `main`）。写 report 时要能凭这些记录追溯每一个数字、每一次失败、每一个设计决策的来龙去脉。

- 每个实验/决策一条记录，文件命名 `YYYY-MM-DD_<slug>.md`（如 `2026-09-22_mlp_baseline.md`），按日期自然排序。
- 记录内容至少包含：**问题定义 → 方法/模型 → 数据与切分 → 超参与随机种子 → 结果（精确数字+指标）→ 遇到的问题 → 尝试过的解法（含失败的）→ 结论与下一步**。
- **失败的、作废的、走不通的实验同样要记录**——它们是 report 里 discussion/limitations 的证据，不要删。
- 记录里的每个数字必须能从已提交的代码一键复现（见 §2 的运行命令）；报告中的数字必须与代码输出一致。
- 对应的实验代码必须一并 commit（见 §1 的目录约定）。**禁止只把结果留在本地内存或未提交的 Notebook 输出里。**
- `docs/progress/` 当前为空且**未被 gitignore**——这是故意的：它就是为版本管理而设的。记录写完后 `git add docs/progress && git commit && git push`。

## 1. 仓库结构

```
analysis/            # 可复现的 .py 实验脚本（eda.py 已产出 docs/data_analysis.md 的全部数字/图）
src/                 # 建模代码（当前只有 l1.ipynb，torch MLP 草稿）
docs/
  task_brief.md      # 任务说明（导师邮件要点、交付清单、时间线）——必读
  data_analysis.md   # EDA 报告：163 特征构成、77 个异常格、基线数字、可执行清单
  eda_results.json   # 机器可读的全部统计结果（写 report 引用精确数值用）
  formulation_probe_{horizons,stations}.csv  #  formulation 探针结果
  progress/          # ★ 实验轨迹记录（见 §0），全部要 commit
data/                # 上游数据集（gitignored，2.7MB CSV，勿提交）
docs/figures/        # EDA 图表 fig01–fig09（gitignored，可由 eda.py 重新生成）
```

- 仓库在 Windows 的 `E:\College_Projects\Weather_Predict`（WSL 挂载为 `/mnt/e/...`）。
- 偏好可 diff 的 `.py` 脚本而非 Notebook 做实验；Notebook 可用，但输出即证据，必须 commit。

## 2. 环境与运行命令（最容易踩的坑）

- **系统 Python（3.14）没有装 pandas/numpy/sklearn/matplotlib/torch，也不要全局 pip install。** 一律用 `uv`（已装 0.11.3）拉起隔离环境：
  ```bash
  # EDA（重新生成 docs/data_analysis.md 全部数字与图）
  uv run --no-project --with pandas --with numpy --with matplotlib python analysis/eda.py

  # formulation 探针（需 sklearn）
  uv run --no-project --with pandas --with numpy --with scikit-learn python analysis/formulation_probe.py
  ```
- 新增依赖时按同样模式 `--with <pkg>` 追加，并把完整命令写进脚本的 module docstring 和 progress 记录。
- 无 pyproject/requirements.txt；report 交付前需整理出依赖说明与一键运行入口（task_brief §6）。

## 3. 科学红线（评估协议，违反 = 报告作废）

这些坑全部有数据佐证，详见 `docs/data_analysis.md` §3/§5/§7 与 `docs/task_brief.md` §4.3：

- **严禁随机划分**。日序数据 lag-1 自相关 0.957，shuffle 必然泄漏。唯一合法切分：**train 2000–2007（2921 天）/ val 2008（366 天）/ test 2009-01-01～2010-01-01（366 天）**；验证集用于选模型与早停。
- **persistence 基线是必须越过的门槛**：test MAE **1.667 °C**、R² **0.9206**。任何模型必须报告相对它的提升，只报 R²≈0.9 不构成证据。气候基线：test MAE 2.662 / R² 0.8118。
- **气候/标准化统计量只能用训练期估计**（月度气候均值、站点均值、scaler 参数），再施加到 val/test。formulation_probe.py 里的 `train_years` 写法照抄。
- **季节循环占目标方差 74.4%**。加 day-of-year 等季节特征必须配"不加"的对照组，否则高 R² 只是日历。
- 原始相关被季节循环灌水（随便两条温度序列 r>0.9）；论证模型价值只用**去季节化**后的相关。
- **2010 年只有 1 天**（2010-01-01 边界日），对它做任何年度统计都无意义。
- 测试期仅 366 天，MAE 标准误 ≈ 0.11 °C——0.1 °C 量级的"提升"是噪声，须报 bootstrap 区间或合并更长测试期。
- **77 个物理不可能值**（`−99`/`−0.099` 哨兵残留、`temp_min>temp_max` 等，集中於 STOCKHOLM/HEATHROW）；`STOCKHOLM_cloud_cover = −99` 落在测试期。清洗方案见 data_analysis.md §3.4，清洗本身也是要在 progress 记录的实验。
- **量纲未标准化**（pressure≈1.0 且 std 仅 0.013，wind_gust 达 41）→ NN/KNN/SVM 必须标准化，树模型可跳过但要在报告中说明理由。
- 列名拆分用**最长前缀匹配**，不能按第一个下划线切（`DE_BILT_x` 会退化成 `DE`；`split_column()` 已实现）。
- picnic 分类任务：标签无 ROMA；De Bilt 列名是 `DE_picnic_weather`；`SONNBLICK_picnic_weather` 恒为 False（常数列，必须剔除）；正例率 24.7% 且季节差 120 倍 → 用 PR-AUC、时间切分、月份基线对照。

## 4. 报告与写作约束

- 报告**英文**，学术格式，建议 8–12 页；篇幅重心给 problem formulation 与 model（这才是评分点，实验结果 optional 但建议至少有基线对照表）。
- 必引：Klein Tank et al. (2002)；Huber et al.（*Will the sun shine?*）；数据 commit `83d70ee`。
- 上游已做的数据操纵（均值填补、单位缩放未标准化）必须在报告中显式声明并讨论其影响。
- 每个图有编号/单位/图注；每个数字区分 train/val/test；提交前在干净环境复跑一遍代码核对数字（task_brief §6 自检清单）。

## 5. Git 工作流

- 每完成一个实验单元（代码 + progress 记录）就 commit；commit message 简洁描述实验内容（如 `exp: ridge baseline h=1, test MAE 1.52`），并及时 `git push` 到 `main`——本地 E 盘不等于备份，GitHub 才是。
- 不要 commit：`data/`、`docs/figures/`、`__pycache__/`、`.ipynb_checkpoints/`、`.venv/`（已被 .gitignore 覆盖）。
- 小型机器可读结果（CSV/JSON，如 formulation_probe 的两个 csv）**要 commit**，它们是 report 数字的直接证据。
