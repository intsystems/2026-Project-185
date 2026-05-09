#!/usr/bin/env python
"""Quick benchmark: deepxde PODDeepONet vs our POD-DeepONet on Burgers nu=0.1."""
import os, sys, pathlib
import numpy as np

os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
from deepxde.nn.pytorch import PODDeepONet

_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from utils.datasets import load_stacked

DATA_DIR    = os.path.expanduser("~/data/1D/Burgers/Train")
NU          = 0.1
N_SAMPLES   = 9500
N_TEST      = 1000
N_MODES     = 32
HIDDEN      = [128, 128, 128, 128, N_MODES]  # branch net layer sizes
EPOCHS      = 50000
LR          = 1e-3
BATCH_SIZE  = None   # CartesianProd — use full batch or mini-batch on branch dim

# ── 1. Load data ──────────────────────────────────────────────────────────────
entries = [(NU, os.path.join(DATA_DIR, f"1D_Burgers_Sols_Nu{NU}.hdf5"))]
s_np, kappa_np, x_np, t_np, Nx, Nt = load_stacked(entries, n_samples=N_SAMPLES)

# s: (N, Nx*Nt) — flattened space-time trajectories
# u0: (N, Nx)   — initial condition (sensor values = branch input)
u0 = s_np[:, :Nx]  # initial condition

# Trunk query points: (Nx*Nt, 2) = [x, t] pairs
xx, tt = np.meshgrid(x_np, t_np, indexing="ij")   # (Nx, Nt)
xt_flat = np.stack([xx.flatten(), tt.flatten()], axis=1).astype(np.float32)  # (Nx*Nt, 2)
s_flat  = s_np.astype(np.float32)     # (N, Nx*Nt)
u0      = u0.astype(np.float32)       # (N, Nx)

# Train/test split
idx     = np.random.default_rng(42).permutation(N_SAMPLES)
tr, te  = idx[N_TEST:], idx[:N_TEST]
u0_tr, u0_te = u0[tr], u0[te]
s_tr,  s_te  = s_flat[tr], s_flat[te]

print(f"train={len(u0_tr)}  test={len(u0_te)}  Nx={Nx}  Nt={Nt}  Ny={Nx*Nt}")

# ── 2. Compute POD basis on training data ─────────────────────────────────────
# SVD: s_tr ~ U @ S @ Vt,  POD modes = Vt[:N_MODES].T  (Ny, N_MODES)
s_mean    = s_tr.mean(axis=0)                                           # (Ny,)
U, sv, Vt = np.linalg.svd(s_tr - s_mean, full_matrices=False)
pod_basis = Vt[:N_MODES].T.astype(np.float32)  # (Ny, N_MODES)
energy    = sv**2
cumvar    = np.cumsum(energy) / energy.sum() * 100
print(f"POD: {N_MODES} modes capture {cumvar[N_MODES-1]:.2f}% variance")

# ── 3. Build deepxde dataset & model ─────────────────────────────────────────
# Center targets: branch predicts coefficients in mean-subtracted space.
# Evaluation will add mean back manually.
s_tr_c = (s_tr - s_mean).astype(np.float32)
s_te_c = (s_te - s_mean).astype(np.float32)

data = dde.data.TripleCartesianProd(
    X_train=(u0_tr, xt_flat),
    y_train=s_tr_c,
    X_test=(u0_te, xt_flat),
    y_test=s_te_c,
)

branch_sizes = [Nx] + HIDDEN   # input dim = Nx
net = PODDeepONet(
    pod_basis=pod_basis,
    layer_sizes_branch=branch_sizes,
    activation="tanh",
    kernel_initializer="Glorot normal",
    layer_sizes_trunk=None,   # POD only, no extra trunk net
)

model = dde.Model(data, net)
model.compile("adam", lr=LR, loss="mse", metrics=["mean l2 relative error"])

# ── 4. Train ──────────────────────────────────────────────────────────────────
print("Training...")
losshistory, train_state = model.train(iterations=EPOCHS, batch_size=256, display_every=5000)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
# Model predicts in centered space; add mean back for real-space error
y_pred_c = model.predict((u0_te, xt_flat))   # (N_test, Ny), centered
y_pred   = y_pred_c + s_mean                 # restore mean

def rel_l2(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)

err = rel_l2(s_te, y_pred)
print(f"\n=== Results (nu={NU}) ===")
print(f"  mean   rel L2 = {err.mean():.4f}")
print(f"  median rel L2 = {np.median(err):.4f}")
print(f"  std           = {err.std():.4f}")
print(f"\nOur POD-DeepONet (nu=0.1, n_train=9000): mean=0.0480  median=0.0324")
print(f"Difference: {err.mean() - 0.0480:+.4f}")
