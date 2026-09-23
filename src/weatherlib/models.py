"""Model factory -- 100% PyTorch (no sklearn, no trees).

Framework decision 2026-09-22 (see docs/progress/2026-09-22_framework_decision_torch.md):
- all learned models are PyTorch; the tree rung (RF/HistGBM) is dropped
- alpha (L2/L1 penalty) is selected on the VALIDATION years (2008) instead of
  sklearn's internal CV -> no temporal shuffle anywhere in model selection
- inputs are standardized with train-only mean/std (computed inside fit)

Performance note: the linear L2 baseline is solved in CLOSED FORM with
torch.linalg.solve (== the ridge objective, exact and instant). Gradient
training (with L1/Lasso and, later, the MLP) lands in R2/R3 with far fewer
epochs. A per-alpha gradient loop over 25 alphas x 500 epochs was too slow.

Runtime: Windows Python 3.12 (torch 2.13 + CUDA), invoked from WSL via
  "/mnt/d/Program Files/Pythons/python3.12/python.exe" <script>
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

ALPHAS = np.logspace(-2, 4, 25)
DEFAULT_SEED = 0


@dataclass
class Fitted:
    """A trained torch model plus its train-only standardizer."""

    net: nn.Module
    mean: np.ndarray
    std: np.ndarray
    info: dict

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = torch.tensor((X - self.mean) / self.std, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            return self.net(Xs).squeeze(1).numpy()


def _ridge_solve(Xt: torch.Tensor, yt: torch.Tensor, alpha: float) -> torch.Tensor:
    """Closed-form ridge weights on standardized (zero-mean) inputs.

    w = (X^T X + alpha I)^-1 X^T y   (torch.linalg.solve, exact, instant)
    """
    d = Xt.shape[1]
    A = Xt.t() @ Xt + alpha * torch.eye(d)
    b = (Xt.t() @ yt).unsqueeze(1)
    return torch.linalg.solve(A, b).squeeze(1)


def _l1_grad(
    Xt: torch.Tensor,
    yt: torch.Tensor,
    alpha: float,
    epochs: int = 300,
    lr: float = 1e-2,
    batch: int = 256,
    seed: int = DEFAULT_SEED,
) -> torch.Tensor:
    """Lasso via full/mini-batch Adam with L1 penalty on the weights."""
    torch.manual_seed(seed)
    d = Xt.shape[1]
    w = torch.zeros(d, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    n = len(yt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = torch.mean((Xt[idx] @ w - yt[idx]) ** 2) + alpha * w.abs().sum()
            loss.backward()
            opt.step()
    return w.detach()


def fit_linear(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    penalty: str = "l2",
    alphas=ALPHAS,
    epochs: int = 300,
    seed: int = DEFAULT_SEED,
) -> Fitted:
    """Linear model with L1/L2 penalty; alpha chosen by val MAE on the 2008 years.

    - penalty="l2": closed-form ridge (torch.linalg.solve) -- instant, exact.
    - penalty="l1": gradient Lasso (Adam + L1), intended for R2 with a small grid.
    """
    mean = Xtr.mean(0)
    std = Xtr.std(0)
    std[std == 0] = 1.0
    Xs_tr = (Xtr - mean) / std
    Xs_va = (Xva - mean) / std
    b0 = float(ytr.mean())

    Xt = torch.tensor(Xs_tr, dtype=torch.float32)
    yt = torch.tensor(ytr - b0, dtype=torch.float32)
    Xv = torch.tensor(Xs_va, dtype=torch.float32)

    best_mae = np.inf
    best_alpha, best_w = None, None
    for a in alphas:
        if penalty == "l2":
            w = _ridge_solve(Xt, yt, float(a))
        else:
            w = _l1_grad(Xt, yt, float(a), epochs=epochs, seed=seed)
        with torch.no_grad():
            pred = (Xv @ w + b0).numpy()
        va_mae = float(np.mean(np.abs(pred - yva)))
        if va_mae < best_mae:
            best_mae, best_alpha, best_w = va_mae, float(a), w

    net = nn.Linear(Xtr.shape[1], 1)
    with torch.no_grad():
        net.weight.copy_(best_w.view(1, -1))
        net.bias.fill_(b0)
    return Fitted(
        net=net,
        mean=mean,
        std=std,
        info={"alpha": best_alpha, "val_mae": best_mae, "penalty": penalty},
    )
