# Plan R0: ΔT_max formulation 决策落盘 + 可行性 probe

> **For agentic workers:** 单波次、两任务、文件不相交，单执行者顺序完成（≤0.5d）
> **Spec:** `docs/superpowers/specs/2026-09-22-delta-t-max-spec.md` §2（v1.1）
> **工作目录:** `/mnt/e/College_Projects/Weather_Predict`

---

### Task A: ΔT_max 可行性 probe
**Files:**
- Create: `analysis/delta_t_probe.py`（✅ 已建，含 oracle P1-2/P2-5 docstring 修正）
- Create: `docs/delta_t_probe.csv`, `docs/delta_t_probe_lead.csv`

- [x] **Step 1: 写 probe 脚本**（清洗→填补→对齐→基线→ridge 子集→advection 表）
- [x] **Step 2: 运行**
```bash
uv run --no-project --with pandas --with numpy --with scikit-learn python analysis/delta_t_probe.py
```
- [ ] **Step 3: 修 bug 至通过**（首跑 `DataFrame.between` 不存在 → 已改为 `(v<lo)|(v>hi)`）
- [ ] **Step 4: Commit**
```bash
git add analysis/delta_t_probe.py docs/delta_t_probe.csv docs/delta_t_probe_lead.csv docs/progress/2026-09-22_delta_t_formulation_decision.md
git commit -m "exp(probe): delta_t_probe.py first dT_max feasibility numbers"
```

---

### Task B: 决策记录 + task_brief 改向 + spec/plan 落盘
**Files:**
- Create: `docs/progress/2026-09-22_delta_t_formulation_decision.md`
- Modify: `docs/task_brief.md`（§4.1 主任务→ΔT_max；§4.2 扩展；§4.3 基线/指标；§5 模型阶梯；§7 时间线；§9 风险；新增 §10 变更日志）
- Create: `docs/superpowers/specs/2026-09-22-delta-t-max-spec.md`（✅ v1.1 已建）
- Create: 本文件

- [x] **Step 1: Spec v1.0 → oracle 审查 → v1.1**（APPROVE-WITH-MINOR，P0=0，7 条 diff 已应用）
- [ ] **Step 2: 写决策记录**（4 个方法问题答案 + 研究规则三件套：顺序/问题处理/骨架）
- [ ] **Step 3: task_brief 改向**
- [ ] **Step 4: Commit + push**
```bash
git add docs/task_brief.md docs/superpowers
git commit -m "docs(brief): re-point task_brief to dT_max formulation + spec v1.1 + R0 plan"
git push origin main
```

---

## Acceptance
- [ ] probe 打印 test MAE：zero / yesterday-Δ / ridge 全量 / ridge 上游子集，且 CSV 落盘
- [ ] task_brief §4.1 主任务 = ΔT_max，§10 变更日志含日期/理由/progress 链接
- [ ] 决策记录含方法四问答案 + 规则三件套
- [ ] spec v1.1、plan、probe 脚本、progress 记录同批 commit 并 push（oracle 前置依赖要求）
- [ ] 后续 Wave（2–6）进入时各生成 plan 文件（本计划不预先展开）
