# 2026-09-22 R3：MLP 网格（容量上限对照）+ 多站合并（Wave 4，提前于 09-26 完成）

> 关联：spec v1.2 §2.3.3/§4.2；`src/weatherlib/models.py`（fit_mlp）、`analysis/r3_mlp.py`
> 运行：`"/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/r3_mlp.py`

## 问题定义
① MLP（容量上限）相对线性模型能否带来超出噪声的提升？② 多站合并（pooled）能否优于各站独立建模？（Δ 空间站点气候偏移抵消，理论上可干净合并。）

## 方法与配置
- 输入 = all163 levels+deltas（326），train/val/test = 2920/366/366。
- MLP：GELU+Dropout+Huber(δ=1×train σ_Δ)+Adam(lr 1e-3)+batch 64+EarlyStopping(val MAE, patience 30, max 300)，种子 {0,1,2}；**按 mean val MAE 选配置**，test 仅最终评估。
- 网格：hidden {32}/{64,32}/{128,64} × dropout {0.2,0.4} × wd {1e-4,1e-3} = 12 配置。
- Pooled：18 站 × 3652 = 65736 station-days（test 18×366=6588）；目标=各站自身 ΔT_max(t+1)，特征=同一全局 326 场；对照 pooled zero / per-station L2 均值 / pooled L2 / pooled MLP((64,32), do0.3, wd1e-4, batch256, ≤100 ep)。

## 结果

**① MLP 网格（BASEL，选定配置按 mean val MAE）**
最佳：h=(64,32) do=0.4 wd=1e-4（val 1.797）
- per-seed test MAE：1.795 / 1.758 / 1.757
- **ensemble test MAE 1.751 [1.546, 1.946]**，R² 0.457，skill **+26.8%**
- vs 线性 1.827 / Lasso 1.773 —— 提升 0.076 °C，CI 与线性 [1.623,2.014] **重叠** → **在噪声内**。
- 网格整体平（test ens 均 1.70–1.75）→ 模型对该任务不敏感，容量不是瓶颈。

**② 多站合并（负结果，讨论素材）**

| 模型 | test MAE [CI] | skill vs pooled zero |
|---|---|---|
| pooled zero | 2.186 | — |
| **per-station L2 均值** | **1.705**（min 1.299 / max 1.998）| **+22.0%** |
| pooled L2（单一全局模型）| 2.066 [2.016,2.115] | +5.5% |
| pooled MLP | 2.038 [1.986,2.088] | +6.8% |

- **单一全局模型显著劣于 18 个独立模型**（1.705 vs 2.066，差 0.36 °C 远超 CI）。原因：各站对同一环流场的响应关系不同（本地地形/方位），线性类模型无法用一个权重向量表达；MLP 靠非线性交互缓解部分（2.038），仍不及独立。
- 副产物：pooled zero 2.186 < BASEL zero 2.393 → BASEL 变温幅度高于站点均值，per-station 的 BASEL 1.827 高于均值 1.705 同理。

## 遇到的问题与解法
1. **首个脚本版本用 ensemble test MAE 选配置（违反"test 仅评一次"）** → 改为 mean val MAE 选参；per-station 循环索引混乱 → 用 enumerate 重写。属编码审查时自查修正，非实验失败。
2. 控制台 `±` 在 cp936 下乱码（`1.738??0.032`）——仅显示问题，CSV（UTF-8）数字正确，不影响复现。

## 结论与下一步
- **能力阶梯封顶**：zero 2.393 → 线性 1.827 → Lasso 1.773 → MLP 1.751，后三者 CI 全部重叠 → 大气可预报性 ~24-27% 是上限，模型复杂度不再加分（样本量-容量权衡坐实）。
- **pooled 负结果**：多站合并降级为 discussion（站点特异性 vs 全局共享表示的权衡）。
- 下一步 **R4（Wave 5）**：block bootstrap 差异显著性表（模型间配对 CI）+ 全消融汇总 + 图表；**R5**：报告成文。
