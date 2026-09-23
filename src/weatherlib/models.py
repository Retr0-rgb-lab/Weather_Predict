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
    """A trained torch model plus its train-only standardizer.

    y is de-standardized in predict() when y_mean/y_std are set (MLP trains on
    a standardized target; linear models set bias internally and leave these 0/1).
    """

    net: nn.Module
    mean: np.ndarray
    std: np.ndarray
    info: dict
    y_mean: float = 0.0
    y_std: float = 1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = torch.tensor((X - self.mean) / self.std, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            out = self.net(Xs).squeeze(1).numpy()
        return out * self.y_std + self.y_mean


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


def _build_mlp(d: int, hidden: tuple[int, ...], dropout: float) -> nn.Module:
    layers: list[nn.Module] = []
    in_dim = d
    for h in hidden:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def fit_mlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    hidden: tuple[int, ...] = (64, 32),
    dropout: float = 0.3,
    wd: float = 1e-4,
    lr: float = 1e-3,
    batch: int = 64,
    epochs: int = 300,
    patience: int = 30,
    loss: str = "huber",
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> Fitted:
    """Small MLP with GELU + Dropout, Huber loss on a train-standardized target.

    Early stopping on val MAE (raw degrees C), best weights restored.
    Returns a CPU model; predict() is device-agnostic.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mean = Xtr.mean(0)
    std = Xtr.std(0)
    std[std == 0] = 1.0
    Xs_tr = torch.tensor((Xtr - mean) / std, dtype=torch.float32)
    Xs_va = torch.tensor((Xva - mean) / std, dtype=torch.float32)
    y_mean = float(ytr.mean())
    y_std = float(ytr.std())
    ys_tr = torch.tensor((ytr - y_mean) / y_std, dtype=torch.float32)
    ys_va = torch.tensor((yva - y_mean) / y_std, dtype=torch.float32)

    torch.manual_seed(seed)
    net = _build_mlp(Xtr.shape[1], hidden, dropout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.HuberLoss(delta=1.0) if loss == "huber" else nn.MSELoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xs_tr.to(dev), ys_tr.to(dev)),
        batch_size=batch, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    Xv, yv = Xs_va.to(dev), ys_va.to(dev)

    def val_mae() -> float:
        net.eval()
        with torch.no_grad():
            pred = (net(Xv).squeeze(1) * y_std + y_mean).cpu().numpy()
        return float(np.mean(np.abs(pred - yva)))

    best_mae = np.inf
    best_epoch, best_state, stall = 0, None, 0
    for ep in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            lossv = lossf(net(xb).squeeze(1), yb)
            lossv.backward()
            opt.step()
        vm = val_mae()
        if vm < best_mae:
            best_mae, best_epoch, stall = vm, ep, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stall += 1
            if stall >= patience:
                break
    net.load_state_dict(best_state)
    return Fitted(
        net=net.to("cpu"), mean=mean, std=std, y_mean=y_mean, y_std=y_std,
        info={"hidden": hidden, "dropout": dropout, "wd": wd, "seed": seed,
              "loss": loss, "val_mae": best_mae, "best_epoch": best_epoch},
    )


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
