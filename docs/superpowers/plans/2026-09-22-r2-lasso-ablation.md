# Plan R2: Lasso 变量选择 + 特征组消融（Wave 3）

> **For agentic workers:** 单任务、单执行者，~1.5d（提前于 09-25 目标，日期 09-22）
> **Spec:** `docs/superpowers/specs/2026-09-22-delta-t-max-spec.md` §2.3/§2.4（v1.2）
> **工作目录:** `/mnt/e/College_Projects/Weather_Predict`

### Task A: 特征组消融表（L2 线性，α on val 2008）
**Files:**
- Modify: `src/weatherlib/dataset.py`（加 season / missings 两个受控消融块，向后兼容）
- Create: `analysis/r2_lasso_ablation.py`、`docs/r2_ablation.csv`、`docs/r2_lasso_coefs.csv`

- [x] **Step 1: dataset 加受控块**（season=doy sin/cos；missings=缺失指示，默认关）
- [x] **Step 2: 消融表**：zero/yesterday + 10 组 L2（levels/deltas/grads/season/missing 组合 + 上游/本站子集）
- [x] **Step 3: Lasso（L1 梯度）**：levels+deltas 326 特征，α 网格 geomspace(1e-3,1,8)，val 选；top25 系数落盘
- [x] **Step 4: 压缩模型**：lasso_l1_full / top非零 L2 重拟合 / top25 L2，均带 block-CI
- [x] **Step 5: 运行**
```bash
"/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/r2_lasso_ablation.py
```

### Task B: 记录 + 提交
- Create: `docs/progress/2026-09-22_r2_lasso_ablation.md`、本 plan

- [x] **Step 1: 写 progress 记录**（含诚实发现：L1 在 val 最优 α 下几乎不稀疏化）
- [ ] **Step 2: Commit + push**
```bash
git add src/weatherlib/dataset.py analysis/r2_lasso_ablation.py docs/r2_ablation.csv docs/r2_lasso_coefs.csv docs/progress/2026-09-22_r2_lasso_ablation.md docs/superpowers/plans/2026-09-22-r2-lasso-ablation.md
git commit -m "feat(r2): Lasso var-selection + ablation table (lvl+d 1.827, lasso_l1 1.773, top25 1.859; pressure field = top channel)"
git push origin main
```

## Acceptance
- [ ] docs/r2_ablation.csv + r2_lasso_coefs.csv 落盘且可复现
- [ ] 受控消融证明：season/missing 不带来提升（Δ 空间无季节免费午餐）
- [ ] 物理解释成立：气压场（level+Δ）+ 上游湿度为 top 通道；top25 保留 ~22% skill
- [ ] 全程 train-only + 时间切分 + val 选参
