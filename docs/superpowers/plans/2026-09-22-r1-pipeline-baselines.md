# Plan R1: 正式数据管线 + 基线表（Wave 2）

> **For agentic workers:** 独立模块化任务，单执行者，~1d（提前于 09-23 目标执行，日期 09-22）
> **Spec:** `docs/superpowers/specs/2026-09-22-delta-t-max-spec.md` §2.4/§2.2（v1.1）
> **工作目录:** `/mnt/e/College_Projects/Weather_Predict`

## 代码架构（回答"代码怎么写"）

```
src/weatherlib/                 # 共享实验库：R1 起所有实验共用，杜绝复制粘贴管线
  __init__.py                   # 统一出口：from weatherlib import *
  data.py                       # load_raw / clean(范围+temp_min>temp_max, 先清洗后差分)
                                #   / impute_station_month(train-only 站点×月中位数 + 缺失指示, 默认关)
  dataset.py                    # make_delta_max_dataset: levels[t] ⊕ deltas[t] ⊕ grads[t]
                                #   → y[t]=ΔT_max(t+1)；train/val/test 掩码(按目标日)
  features.py                   # pressure_gradients(上游站 − BASEL, 地转风代理)、season(受控,默认关)
  metrics.py                    # MAE/RMSE/R²/corr_safe/skill + moving block bootstrap(7d) CI
                                #   （单模型 & 配对差异）
  baselines.py                  # zero / yesterday_Δ / monthly_Δ climatology
  models.py                     # fit_ridge / fit_lasso（R2 加树，R3 加 MLP——增量演进）
analysis/
  r1_baselines.py               # R1 实验：基线 + ridge 特征集消融 → docs/r1_baseline.csv
docs/
  progress/2026-09-22_r1_pipeline_baselines.md   # 本波记录（含数字）
  superpowers/plans/2026-09-22-r1-*.md           # 本文件
```

数据流（与 spec §4 架构图一致）：
```
CSV → data.load_raw → data.clean(置NaN) → data.impute_station_month(train-only)
    → dataset.make_delta_max_dataset(levels/deltas/grads + y + masks)
    → baselines.* / models.fit_ridge
    → metrics.*(skill + block-CI)
    → docs/r1_baseline.csv + progress 记录 → commit → push
```

设计决策（相对 probe 的升级，写进记录）：
1. 填补从"全列月均值"升级为"站点×月中位数"（data_analysis §3.4 建议）；预计与 probe 数字有小偏移，需一致性核对。
2. 新增 `temp_min>temp_max` 处理：41 格（HEATHROW/ROMA）成对置 NaN 后填补——不在目标列，纯健壮性。
3. 新增气压梯度特征 `P_upstream − P_BASEL`（7 个，地转风/环流代理）作为独立消融组。
4. bootstrap CI 提前进 R1（R4 的子集，成本低），基线表直接带不确定性。

---

### Task A: 建共享库 src/weatherlib
**Files:**
- Create: `src/weatherlib/__init__.py`, `data.py`, `dataset.py`, `features.py`, `metrics.py`, `baselines.py`, `models.py`

- [ ] **Step 1: 按上文架构写 7 个模块**（无 torch；R3 再加 MLP）
- [ ] **Step 2: 冒烟检查 import**
```bash
uv run --no-project --with pandas --with numpy --with scikit-learn python -c "import sys; sys.path.insert(0,'src'); import weatherlib; print('ok', weatherlib.STATIONS[0])"
```

### Task B: R1 基线表实验
**Files:**
- Create: `analysis/r1_baselines.py`

- [ ] **Step 1: 写脚本**（基线 ×3 + ridge ×6 特征集；skill vs zero；block-CI(1000, block=7, seed=42)；写 docs/r1_baseline.csv）
- [ ] **Step 2: 运行**
```bash
uv run --no-project --with pandas --with numpy --with scikit-learn python analysis/r1_baselines.py
```
- [ ] **Step 3: 一致性核对**：ridge lvl+d_all163 test MAE 应 ≈ probe 的 1.859（±0.05，中位数 vs 均值填补导致的小偏移可接受）
- [ ] **Step 4: Commit**
```bash
git add src/weatherlib analysis/r1_baselines.py docs/r1_baseline.csv docs/progress/2026-09-22_r1_pipeline_baselines.md
git commit -m "feat(r1): weatherlib pipeline + baseline table (zero 2.393, ridge full MAE <1.87, skill +22%, block-CI)"
```

### Task C: 历史分析补录（用户指出的 progress 缺口）
**Files:**
- Create: `docs/progress/2026-09-19_eda.md`（补录）
- Create: `docs/progress/2026-09-21_formulation_probe.md`（补录）

- [ ] **Step 1: 补录两条记录**（标注"补录于 2026-09-22"，数字与 data_analysis.md / probe CSV 一致、可复现）
- [ ] **Step 2: 与 Task B 同批 commit**

---

## Acceptance
- [ ] `docs/r1_baseline.csv` 落盘；ridge 全量 MAE 与 probe 一致性核对通过
- [ ] 全程无泄漏：清洗/填补/标准化/气候统计 train-only；时间切分 2920/366/366
- [ ] `docs/progress/` 补录 EDA + formulation_probe 两条历史记录
- [ ] 全部 commit + push（AGENTS.md §0）
