"""LSTM forecaster and the loss functions compared in the study (PyTorch)."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn

LOSS_NAMES = ("mse", "mae", "huber", "focal_mse", "focal_mse_norm", "target_weighted_mse")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WaveLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dense=(64, 32)):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        blocks, width = [], hidden
        for d in dense:
            blocks += [nn.Linear(width, d), nn.ReLU()]
            width = d
        blocks.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*blocks)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def focal_mse(pred, target, gamma: float = 2.0, alpha: float = 1.0, normalise: bool = False):
    """Residual-weighted MSE.

    Each squared residual is multiplied by (|r| / max|r|)^gamma, with the maximum taken over
    the batch and detached from the graph. All weights are <= 1, so the loss (and the gradient)
    is smaller than plain MSE. With normalise=True the weights are rescaled to a batch mean of 1,
    which removes that difference in effective step size.
    """
    r = target - pred
    w = (r.abs() / (r.abs().max().detach() + 1e-12)) ** gamma
    if normalise:
        w = w / (w.mean().detach() + 1e-12)
    return alpha * torch.mean(w.detach() * r**2)


def target_weighted_mse(pred, target, kappa: float = 1.0):
    """MSE with weights that grow with the observed (standardised) target.

    Unlike the residual-based focal weights, this puts the emphasis on high waves
    themselves, not on whichever samples happen to be badly predicted.
    """
    w = 1.0 + kappa * torch.clamp(target, min=0.0)
    w = w / w.mean().detach()
    return torch.mean(w.detach() * (target - pred) ** 2)


def get_loss(name: str, huber_delta: float = 1.0, gamma: float = 2.0, kappa: float = 1.0):
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=huber_delta)
    if name == "focal_mse":
        return lambda p, t: focal_mse(p, t, gamma=gamma)
    if name == "focal_mse_norm":
        return lambda p, t: focal_mse(p, t, gamma=gamma, normalise=True)
    if name == "target_weighted_mse":
        return lambda p, t: target_weighted_mse(p, t, kappa=kappa)
    raise ValueError(f"unknown loss '{name}', choose from {LOSS_NAMES}")


def _batches(n, batch_size, rng):
    order = rng.permutation(n)
    for i in range(0, n, batch_size):
        yield order[i : i + batch_size]


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    loss_name: str,
    seed: int,
    *,
    max_epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    stop_patience: int = 10,
    device: str | None = None,
    verbose: bool = False,
):
    """Train one model and return (model, history). Best validation checkpoint is restored.

    Validation loss uses the same loss function as training, computed over the whole
    validation set, so models are selected by the objective they were trained with.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = WaveLSTM(X_train.shape[-1]).to(device)
    loss_fn = get_loss(loss_name)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience
    )
    Xt, yt = torch.tensor(X_train, device=device), torch.tensor(y_train, device=device)
    Xv, yv = torch.tensor(X_val, device=device), torch.tensor(y_val, device=device)
    best, best_state, wait = np.inf, None, 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        for idx in _batches(len(Xt), batch_size, rng):
            idx_t = torch.as_tensor(idx, device=device)
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx_t]), yt[idx_t])
            loss.backward()
            opt.step()
            running += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(Xv), yv))
        sched.step(val)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running / len(Xt),
                "val_loss": val,
                "lr": opt.param_groups[0]["lr"],
            }
        )
        if verbose:
            print(f"epoch {epoch + 1:02d}  train {history[-1]['train_loss']:.5f}  val {val:.5f}")
        if val < best - 1e-9:
            best, best_state, wait = val, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= stop_patience:
                break
    model.load_state_dict(best_state)
    return model, history


def predict(model: WaveLSTM, X, device: str | None = None, batch_size: int = 2048) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.tensor(X[i : i + batch_size], device=device)).cpu().numpy())
    return np.concatenate(out)
