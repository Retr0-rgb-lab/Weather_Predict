# Plan R4: 最终结果表 + 配对显著性 + 图表（Wave 5）

> **For agentic workers:** 单任务、单执行者，~1d（提前于 09-27 目标，日期 09-22）
> **Spec:** `docs/superpowers/specs/2026-09-22-delta-t-max-spec.md` §2.2（v1.2）
> **工作目录:** `/mnt/e/College_Projects/Weather_Predict`

### Task A: 规范最终表 + 配对显著性
**Files:**
- Create: `analysis/r4_eval_figures.py`、`docs/r4_final_table.csv`、`docs/r4_pairwise.csv`

- [x] **Step 1: 统一重算 10 个模型**（基线×3 + 线性×5 + Lasso + MLP ensemble，同一协议）
- [x] **Step 2: block-CI(7d) 进表**；配对同块 bootstrap 算 MAE 差异 CI + sig_95 标志
- [x] **Step 3: 运行**
```bash
"/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/r4_eval_figures.py
```

### Task B: 报告图表（docs/figures/，gitignored 可再生）
**Files:**
- Create: `docs/figures/fig10..fig14`（英文标注）

- [x] fig10 模型阶梯（MAE+CI 柱状） / fig11 MLP 测试散点 / fig12 分月误差 /
      fig13 Lasso 物理通道系数 / fig14 逐站 skill

### Task C: 记录 + 提交
- Create: `docs/progress/2026-09-22_r4_eval_figures.md`、本 plan

- [ ] **Step 1: 写 progress 记录**
- [ ] **Step 2: Commit + push**
```bash
git add analysis/r4_eval_figures.py docs/r4_final_table.csv docs/r4_pairwise.csv docs/progress/2026-09-22_r4_eval_figures.md docs/superpowers/plans/2026-09-22-r4-eval-figures.md
git commit -m "feat(r4): canonical final table + paired significance + figures (all models sig vs zero; inter-model diffs within CI; full vs up7 sig)"
git push origin main
```

## Acceptance
- [ ] docs/r4_final_table.csv + r4_pairwise.csv 落盘，数字与 R1–R3 一致
- [ ] 显著性表正确：零基线差异 sig；模型间差异非 sig；全量 vs 上游 sig
- [ ] 5 张报告图产出（可再生、英文标注）
