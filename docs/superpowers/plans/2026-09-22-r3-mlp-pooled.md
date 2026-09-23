# Plan R3: MLP 网格 + 多站合并（Wave 4）

> **For agentic workers:** 单任务、单执行者，~1.5d（提前于 09-26 目标，日期 09-22）
> **Spec:** `docs/superpowers/specs/2026-09-22-delta-t-max-spec.md` §2.3.3/§4.2（v1.2）
> **工作目录:** `/mnt/e/College_Projects/Weather_Predict`

### Task A: MLP 网格（BASEL，容量上限对照）
**Files:**
- Modify: `src/weatherlib/models.py`（Fitted 加 y 反标准化；新增 `fit_mlp`：GELU+Dropout+Huber+EarlyStopping(val MAE, patience 30)）
- Create: `analysis/r3_mlp.py`、`docs/r3_mlp.csv`

- [x] **Step 1: fit_mlp 实现**（target 按 train σ_Δ 标准化，δ=1；best state 恢复；CPU 返回）
- [x] **Step 2: 计时验证**（(64,32) 5.8s/次，early stop ~epoch14 → 全网格可行）
- [x] **Step 3: 网格**：hidden{32}/{64,32}/{128,64} × dropout{0.2,0.4} × wd{1e-4,1e-3} × 3 种子；**按 mean val MAE 选配置**（test 仅最终评估一次，不用于选参）
- [x] **Step 4: 选定配置 ensemble + block-CI 报告**

### Task B: 多站合并扩展（18 站 pooled）
**Files:**
- Create: `docs/r3_pooled.csv`

- [x] **Step 1: pooled 构造**：18 站 × 3652 = 65736 station-days；目标=各站自身 ΔT_max(t+1)，特征=同一全局 326 lvl+d 场（Δ 空间站点偏移抵消→可干净合并）
- [x] **Step 2: 对照**：pooled zero / per-station L2 均值 / pooled L2 / pooled MLP(精简预算)

### Task C: 记录 + 提交
- Create: `docs/progress/2026-09-22_r3_mlp_pooled.md`、本 plan

- [x] **Step 1: 写 progress 记录**（含 pooled 负结果）
- [ ] **Step 2: Commit + push**
```bash
git add src/weatherlib/models.py src/weatherlib/__init__.py analysis/r3_mlp.py docs/r3_mlp.csv docs/r3_pooled.csv docs/progress/2026-09-22_r3_mlp_pooled.md docs/superpowers/plans/2026-09-22-r3-mlp-pooled.md
git commit -m "feat(r3): MLP grid + pooled multi-station (MLP ens 1.751, capacity not bottleneck; pooling negative: per-station 1.705 >> pooled 2.066)"
git push origin main
```

## Acceptance
- [ ] docs/r3_mlp.csv + r3_pooled.csv 落盘可复现
- [ ] 选参只在 val，test 仅最终评估一次
- [ ] MLP 结论诚实：相对线性提升在 CI 噪声内
- [ ] pooled 负结果写入记录（站点特异性 → discussion 素材）
