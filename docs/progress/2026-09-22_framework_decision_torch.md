# 2026-09-22 框架决策：100% PyTorch（砍树模型，运行时切 Windows Python 3.12）

> 类型：设计决策 + R1 复算记录
> 关联：spec v1.2（§2.2/§2.3）、`src/weatherlib/models.py`、`analysis/r1_baselines.py`、AGENTS.md §2

## 问题定义
用户明确偏好 PyTorch（更熟），要求建模以 torch 为主、不再用 sklearn。树模型（RF/HistGBM）在 torch 中不存在，需决定去留；同时 sklearn 的 `RidgeCV`/`LassoCV` 内部做 K 折/LOO CV，对时序数据有轻微的跨时间褶皱，不完全符合"val 选参、严禁时间洗牌"协议。

## 决策
1. **砍掉树模型，100% PyTorch**（用户选定）：模型阶梯 = 基线 → 线性(torch) → MLP(torch) → 多站合并。非线性对照由"线性系数物理解释 + 特征组消融"替代。
2. **α 选择改用 val 2008**（而非 sklearn 内部 CV）——所有模型统一"训练集训练、验证集选参、测试集评估一次"，协议叙事更干净。
3. **运行时 = Windows Python 3.12**（`D:\Program Files\Pythons\python3.12`，torch 2.13+cu126、CUDA 可用、numpy 2.4、pandas 3.0），从 WSL 经 `/mnt/d/Program Files/Pythons/python3.12/python.exe` 调用；废弃 uv 方案。
4. **线性 L2 = 封闭解**（`torch.linalg.solve`，精确且秒级），L1 Lasso 留 R2 用梯度；MLP 留 R3。

## 遇到的问题与解法
- **首版 fit_linear 用"25 个 α × 6 组 × 500 epoch Adam 小批量梯度"，运行数分钟被用户中止**。解法：L2 改为 `torch.linalg.solve` 封闭解 `w=(X^T X + αI)^{-1}X^T y`，整轮 R1 秒级完成。
- `torch.linalg.solve` 要求右侧 2-D → `b=(X^T y).unsqueeze(1)`。
- Windows 进程内相对路径解析正常（cwd 由 WSL 自动翻译为 E:\...）；`__file__` 为 Windows 路径，`Path.resolve()` 正确。
- pandas 3.0（CoW 默认）：`clean()` 先 `astype(float)` 再置 NaN，未触发 LossySetitemError。

## 结果（torch 封闭解版，test 366 天，与 sklearn 版对比）
| 模型 | torch MAE | sklearn MAE | 差异 |
|---|---|---|---|
| zero_change（门槛） | 2.393 | 2.393 | 0 |
| yesterday_Δ | 3.447 | 3.447 | 0 |
| monthly_Δ 气候 | 2.393 | 2.393 | 0 |
| **linear_lvl+d_all163** | **1.827 [1.623,2.014]** | 1.851 | −0.024 |
| linear_up7_lvl+dtmax | 2.056 | 2.058 | −0.002 |
| linear_up7_dtmax | 2.115 | 2.115 | 0 |
| linear_own_lvl+dtmax | 2.392 | 2.422 | −0.030 |

- val 选出的 α=177.8（sklearn LOO 为 3.16）——更重正则；测试期表现略好（1.827），CI 与 zero 仍不重叠 → **+23.6% skill 保持**。
- 一致性成立：封闭解 = 相同目标函数，差异仅来自 α 选择机制。

## 结论与下一步
- 框架切换无性能损失，且协议更严格（val 选参、无内部 CV、无洗牌）。
- 后续：R2 = Lasso(torch, L1 梯度) 变量选择 + 特征组消融成表（09-25）；R3 = MLP 网格 + 3 种子（CUDA 可用）。
