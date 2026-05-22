#!/usr/bin/env python
"""Regenerate basis figures from saved checkpoint (no retraining)."""
import pathlib, sys
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from models.pod import PODTrainer, PODConfig
from models.regime_basis import FourierRegimeBasis

_ROOT = pathlib.Path(__file__).resolve().parents[2]
C0, C1 = plt.cm.tab10(0), plt.cm.tab10(1)

NU        = 1.0
N_TRAIN   = 9000
N_MODES_VIZ = 8
_PROJ     = _ROOT.parent  # project root
OUT_DIR   = _PROJ / "paper" / "figs"
CKPT      = _ROOT / "test_models" / "burgers" / "npod_deeponet.pt"

# ── data ────────────────────────────────────────────────────────────────────
data_path = pathlib.Path.home() / f"data/1D/Burgers/Train/1D_Burgers_Sols_Nu{NU}.hdf5"
with h5py.File(data_path, "r") as f:
    raw  = f["tensor"][:N_TRAIN]
    x_np = f["x-coordinate"][:]
    t_np = f["t-coordinate"][:]
if raw.ndim == 4:
    raw = raw[..., 0]
N, Nt, Nx = raw.shape
Ny = Nt * Nx

t_grid = torch.tensor(t_np[:Nt], dtype=torch.float32)
x_grid = torch.tensor(x_np,      dtype=torch.float32)
tt, xx = torch.meshgrid(t_grid, x_grid, indexing="ij")
x_flat = torch.stack([xx.flatten(), tt.flatten()], dim=1)   # (Ny, 2)
s_traj = torch.tensor(raw.reshape(N, Ny), dtype=torch.float32)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device={device}")

# ── POD (fast SVD) ───────────────────────────────────────────────────────────
print("Computing POD ...")
trainer_pod = PODTrainer(PODConfig(max_modes=32))
trainer_pod.train(s_traj.to(device), x=None, t=None)

# ── load saved FNPOD basis ───────────────────────────────────────────────────
print("Loading FNPOD basis ...")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
basis_sd = ckpt["basis"]
n_modes  = sum(1 for k in basis_sd if k.startswith("modes.") and k.endswith("lambda_ten"))

w = torch.ones(Ny, dtype=torch.float32) / Ny
torch.manual_seed(42)
basis = FourierRegimeBasis(
    d_x=2, M=N, quad_weights=w,
    hidden_dim=256, num_frequencies=96, n_layers=3,
    scales=[0.5, 2.0, 6.0],
).to("cpu")
# load only the network weights (not lambda_ten, which is training-specific)
for _ in range(n_modes):
    basis.add_mode()
net_sd = {k: v for k, v in basis_sd.items() if "lambda_ten" not in k and k != "quad_weights"}
missing, unexpected = basis.load_state_dict(net_sd, strict=False)
print(f"  loaded {n_modes} modes | missing={len(missing)} unexpected={len(unexpected)}")
basis.eval()

# ── compute FNPOD residuals on grid ─────────────────────────────────────────
print("Evaluating residuals ...")
with torch.no_grad():
    x_dev = x_flat  # cpu
    mean_vals = basis.mean_net(x_dev).squeeze()               # (Ny,)
    residuals = []
    r = s_traj - mean_vals.unsqueeze(0)                        # (N, Ny)
    total_var = (r ** 2).sum().item()
    residuals.append(total_var)
    for k in range(min(n_modes, len(basis.modes))):
        phi = basis.modes[k].phi(x_dev).squeeze()             # (Ny,)
        lam = (r * phi.unsqueeze(0)).sum(dim=1, keepdim=True) # (N,1)
        r   = r - lam * phi.unsqueeze(0)
        residuals.append((r ** 2).sum().item())

res_fnpod = np.array(residuals[1:]) / residuals[0]            # normalised

# ── plot_spectrum ────────────────────────────────────────────────────────────
coeffs_np = trainer_pod.basis.coeffs.cpu().numpy()
sigmas    = np.sqrt((coeffs_np ** 2).sum(axis=0))
sigmas   /= sigmas[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
ax = axes[0]
ax.semilogy(range(1, len(sigmas)+1), sigmas, "o-", color=C0, ms=3, lw=1.5)
ax.set_xlabel("Mode"); ax.set_ylabel("Normalised singular value")
ax.set_title("POD — singular value decay", fontweight="bold")
ax.grid(True, ls="--", alpha=0.25)
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
ax.semilogy(range(1, len(res_fnpod)+1), res_fnpod, "o-", color=C1, ms=3, lw=1.5)
ax.set_xlabel("Mode"); ax.set_ylabel("Normalised weighted residual")
ax.set_title("FNPOD — residual decay", fontweight="bold")
ax.grid(True, ls="--", alpha=0.25)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out = OUT_DIR / f"basis_spectrum_nu{NU}.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"saved {out}")

# ── plot_modes ───────────────────────────────────────────────────────────────
K = min(N_MODES_VIZ, trainer_pod.basis.num_modes, len(basis.modes))
pod_modes = trainer_pod.basis.modes.cpu().numpy()   # (Ny, P)

with torch.no_grad():
    fnpod_modes = [basis.modes[k].phi(x_flat).squeeze().numpy() for k in range(K)]

fig, axes = plt.subplots(2, K, figsize=(2.2*K, 4.2))
for k in range(K):
    phi_pod   = pod_modes[:, k].reshape(Nt, Nx)
    phi_fnpod = fnpod_modes[k].reshape(Nt, Nx)
    for row, phi, label, color in [(0, phi_pod, "POD", C0), (1, phi_fnpod, "FNPOD", C1)]:
        ax   = axes[row, k]
        vmax = np.abs(phi).max()
        ax.imshow(phi, aspect="auto", origin="lower", cmap="RdBu",
                  vmin=-vmax, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"$\\phi_{{{k+1}}}$", fontsize=9)
        if k == 0:
            ax.set_ylabel(label, fontsize=9, color=color, fontweight="bold")

fig.text(0.5, -0.01, "space  →", ha="center", fontsize=8)
fig.text(-0.01, 0.5, "← time", va="center", rotation="vertical", fontsize=8)
plt.suptitle("Basis modes (space–time, first K)", fontweight="bold", fontsize=10)
plt.tight_layout()
out = OUT_DIR / f"basis_modes_nu{NU}.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
print(f"saved {out}")
