#!/usr/bin/env python
"""Benchmark: neuraloperator FNO on full Burgers trajectory (nu=0.1).

FNO 2D: u0(x) -> s(x, t) for all t in [0, T].
Input grid: (N, 3, Nx, Nt) — channels: [u0 repeated, x-coord, t-coord]
Output:     (N, 1, Nx, Nt)

Compare with our POD-DeepONet (nu=0.1, n_train=9000): mean=0.0480
"""
import os, sys, pathlib, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from utils.datasets import load_stacked

from neuralop.models import FNO

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.expanduser("~/data/1D/Burgers/Train")
NU          = 0.1
N_SAMPLES   = 9500
N_TEST      = 1000
N_MODES_X   = 16      # Fourier modes along x
N_MODES_T   = 16      # Fourier modes along t
HIDDEN      = 32      # hidden channels in FNO
N_LAYERS    = 4
EPOCHS      = 200
BATCH_SIZE  = 32
LR          = 1e-3
SEED        = 42

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"device={DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 1. Load data ──────────────────────────────────────────────────────────────
entries = [(NU, os.path.join(DATA_DIR, f"1D_Burgers_Sols_Nu{NU}.hdf5"))]
s_np, kappa_np, x_np, t_np, Nx, Nt = load_stacked(entries, n_samples=N_SAMPLES)

# s: (N, Nt*Nx) — reshape to (N, Nx, Nt)
s_grid = s_np.reshape(-1, Nt, Nx).transpose(0, 2, 1)  # (N, Nx, Nt)
u0     = s_grid[:, :, 0]                                # (N, Nx) — t=0 slice

print(f"s_grid={s_grid.shape}  u0={u0.shape}  Nx={Nx}  Nt={Nt}")

# ── 2. Build coordinate grids ─────────────────────────────────────────────────
# x_grid: (Nx, Nt) — x coordinate repeated along t
# t_grid: (Nx, Nt) — t coordinate repeated along x
xx = np.tile(x_np[:, None], (1, Nt)).astype(np.float32)   # (Nx, Nt)
tt = np.tile(t_np[None, :], (Nx, 1)).astype(np.float32)   # (Nx, Nt)

# Normalise coords to [-1, 1]
xx = (xx - xx.min()) / (xx.max() - xx.min()) * 2 - 1
tt = (tt - tt.min()) / (tt.max() - tt.min()) * 2 - 1

# ── 3. Build FNO input: (N, 3, Nx, Nt) ───────────────────────────────────────
# channel 0: u0 repeated along t
# channel 1: x coordinate
# channel 2: t coordinate
u0_expanded = np.repeat(u0[:, :, None], Nt, axis=2).astype(np.float32)  # (N, Nx, Nt)
xx_batch    = np.broadcast_to(xx[None], (len(s_grid), Nx, Nt)).copy().astype(np.float32)
tt_batch    = np.broadcast_to(tt[None], (len(s_grid), Nx, Nt)).copy().astype(np.float32)

X = np.stack([u0_expanded, xx_batch, tt_batch], axis=1)  # (N, 3, Nx, Nt)
Y = s_grid[:, None, :, :].astype(np.float32)             # (N, 1, Nx, Nt)

print(f"FNO input X={X.shape}  output Y={Y.shape}")

# ── 4. Train/test split ───────────────────────────────────────────────────────
rng  = np.random.default_rng(SEED)
idx  = rng.permutation(N_SAMPLES)
tr, te = idx[N_TEST:], idx[:N_TEST]

X_tr, X_te = torch.tensor(X[tr]), torch.tensor(X[te])
Y_tr, Y_te = torch.tensor(Y[tr]), torch.tensor(Y[te])
print(f"train={len(X_tr)}  test={len(X_te)}")

dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True)

# ── 5. Build FNO model ────────────────────────────────────────────────────────
model = FNO(
    n_modes=(N_MODES_X, N_MODES_T),
    in_channels=3,
    out_channels=1,
    hidden_channels=HIDDEN,
    n_layers=N_LAYERS,
    use_channel_mlp=True,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"FNO params: {n_params:,}")

opt   = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

# ── 6. Train ──────────────────────────────────────────────────────────────────
def rel_l2_batch(true, pred):
    """true, pred: (N, Nx*Nt)"""
    return (np.linalg.norm(true - pred, axis=1) /
            np.linalg.norm(true, axis=1)).mean()

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(xb)
    sched.step()

    if epoch % 20 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            # evaluate on test in batches
            preds = []
            for i in range(0, len(X_te), 128):
                preds.append(model(X_te[i:i+128].to(DEVICE)).cpu())
            pred_np = torch.cat(preds).numpy()  # (N_test, 1, Nx, Nt)

        # flatten to (N_test, Nx*Nt) for rel L2
        pred_flat = pred_np[:, 0, :, :].reshape(len(X_te), -1)
        true_flat = Y_te[:, 0, :, :].reshape(len(X_te), -1).numpy()
        err = rel_l2_batch(true_flat, pred_flat)

        elapsed = time.time() - t0
        train_loss = total / len(X_tr)
        print(f"epoch {epoch:3d} | train_mse={train_loss:.4e} | test_rel_l2={err:.4f} | {elapsed:.0f}s")

# ── 7. Final evaluation ───────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds = []
    for i in range(0, len(X_te), 128):
        preds.append(model(X_te[i:i+128].to(DEVICE)).cpu())
    pred_np = torch.cat(preds).numpy()

pred_flat = pred_np[:, 0, :, :].reshape(len(X_te), -1)
true_flat = Y_te[:, 0, :, :].reshape(len(X_te), -1).numpy()

per_sample = (np.linalg.norm(true_flat - pred_flat, axis=1) /
              np.linalg.norm(true_flat, axis=1))

print(f"\n=== Results (nu={NU}) ===")
print(f"  mean   rel L2 = {per_sample.mean():.4f}")
print(f"  median rel L2 = {np.median(per_sample):.4f}")
print(f"  std           = {per_sample.std():.4f}")
print(f"\nOur POD-DeepONet (nu=0.1, n_train=8500): mean=0.0480  median=0.0324")
print(f"Difference: {per_sample.mean() - 0.0480:+.4f}")
print(f"Total time: {time.time()-t0:.0f}s")
