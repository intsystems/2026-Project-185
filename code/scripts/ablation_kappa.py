#!/usr/bin/env python
"""Ablation: how much does kappa affect TEMPO inference quality?

For each benchmark, runs inference with:
  real   - actual kappa values
  zero   - kappa replaced with zeros
  mean   - kappa replaced with training-set mean

Reports relative L2 error (overall + per kappa value) and mean gating weights
per kappa variant to show routing sensitivity. Supports both POD and NeuralPOD
basis types.

Usage:
  # Burgers POD (server, Nx=1024 data)
  python code/scripts/ablation_kappa.py --benchmark burgers \
      --run_dir TEMPO_results/burgers/tempo_pod_M3_v1

  # Burgers NeuralPOD (local, Nx=101 data; needs valid trainer.pt)
  python code/scripts/ablation_kappa.py --benchmark burgers \
      --run_dir code/TEMPO_results/burgers/tempo_npod_burgers_M3

  # Darcy / NS (server)
  python code/scripts/ablation_kappa.py --benchmark darcy \
      --run_dir TEMPO_results/darcy/tempo_pod_darcy_M5_v1
  python code/scripts/ablation_kappa.py --benchmark ns \
      --run_dir TEMPO_results/navier_stokes/tempo_pod_navier_stokes_M4_v1
"""

import argparse
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from models.tempo_online import TEMPOOnline, GatingNet, _eval_basis, _num_modes
from models.pod_deeponet import BranchNet
from models.pod import PODTrainer


# Helpers

def rel_l2_vec(pred: torch.Tensor, true: torch.Tensor) -> np.ndarray:
    """Per-sample relative L2. Both tensors: (N, D)."""
    return (torch.norm(pred - true, dim=1) / torch.norm(true, dim=1).clamp(min=1e-8)).cpu().numpy()


def _load_model(run_dir: str, device: str):
    """Load TEMPOOnline and frozen sub-trainers from run_dir. Returns (model, sub_trainers, cfg_online)."""
    trainer_path = os.path.join(run_dir, "trainer.pt")
    model_path   = os.path.join(run_dir, "model_online.pt")

    trainer_obj = torch.load(trainer_path, map_location="cpu", weights_only=False)
    ckpt        = torch.load(model_path,   map_location="cpu", weights_only=False)

    cfg_online   = ckpt["cfg_online"]
    sd           = ckpt["model_online"]
    sub_trainers = trainer_obj.trainers
    M            = len(sub_trainers)

    P_list = [_num_modes(t) for t in sub_trainers]

    # Infer architecture from state dict; cfg_online.n_layers can be stale.
    # _mlp(n_layers) builds n_layers+1 Linear layers, so n_layers = n_weight_keys - 1.
    w0 = sd["gating.branch.net.0.weight"]   # (hidden_dim, m_sensors + d_kappa)
    hidden_dim = w0.shape[0]
    d_kappa    = 1   # scalar kappa for all three benchmarks
    m_sensors  = w0.shape[1] - d_kappa
    n_linears  = sum(1 for k in sd if k.startswith("gating.branch.net.") and k.endswith(".weight"))
    n_layers   = n_linears - 1

    gating   = GatingNet(m_sensors, d_kappa, M, hidden_dim, n_layers)
    branches = nn.ModuleList([
        BranchNet(m_sensors, P_list[m], hidden_dim, n_layers, d_kappa=d_kappa)
        for m in range(M)
    ])
    model = TEMPOOnline(gating, branches)
    model.load_state_dict(sd)
    model.eval().to(device)

    return model, sub_trainers, cfg_online


def _compute_bases(sub_trainers, x_flat, device):
    """Evaluate frozen basis (mean, modes) per regime. x_flat required for NeuralPOD, ignored for POD."""
    try:
        from models.fourier_neural_pod import FourierNeuralPODTrainer
        has_npod = True
    except ImportError:
        has_npod = False

    # POD doesn't use x, but _eval_basis needs it for .to(device). Use a dummy if x_flat=None.
    if x_flat is None:
        x_dev = torch.zeros(1, 1, device=device)
    else:
        x_dev = x_flat.to(device)

    means_list, modes_list = [], []
    for t in sub_trainers:
        if has_npod and isinstance(t, FourierNeuralPODTrainer):
            if x_flat is None:
                raise ValueError(
                    "x_flat is required for FourierNeuralPODTrainer but got None. "
                    "The data loader must return a proper coordinate grid."
                )
            # Put basis networks on device in eval mode
            t.basis.to(device).eval()
        with torch.no_grad():
            mean, modes = _eval_basis(t, x_dev)
        means_list.append(mean.float().detach())
        modes_list.append(modes.float().detach())

    return means_list, modes_list


@torch.no_grad()
def _predict_batched(model, u0_sensors, kappa, means, modes, batch=256):
    """Run model.forward in batches to avoid OOM, concat results."""
    preds, ws = [], []
    for i in range(0, len(u0_sensors), batch):
        p, w = model(u0_sensors[i:i+batch], kappa[i:i+batch], means, modes)
        preds.append(p.cpu())
        ws.append(w.cpu())
    return torch.cat(preds), torch.cat(ws)


def _report(variant_label, err_vec, kappa_np, param_label, gating_w=None):
    """Print per-variant summary: overall error + per-kappa breakdown with gates."""
    print(f"  {variant_label}: overall={err_vec.mean():.4f}")
    for v in np.unique(kappa_np):
        mask  = kappa_np == v
        w_str = ""
        if gating_w is not None:
            w_per = gating_w[mask].mean(axis=0)
            w_str = "  gates=[" + ", ".join(f"{w:.3f}" for w in w_per) + "]"
        print(f"    {param_label}={v:.4g}: {err_vec[mask].mean():.4f}{w_str}")


# Benchmark-specific data loaders
# Returns: (s_test, u0_test, kappa_test, kappa_train, x_flat, param_label)
#   x_flat - (Ny, d_x) coordinate grid used during training, or None for NS POD

def _load_burgers(data_dir, nu_values, n_samples, n_test_per_nu):
    from utils.datasets import load_stacked

    entries = [(nu, os.path.join(data_dir, f"1D_Burgers_Sols_Nu{nu}.hdf5"))
               for nu in nu_values]
    s_np, kappa_np, x_np, t_np, Nx, _ = load_stacked(entries, n_samples=n_samples)

    N_per = n_samples
    train_idx = np.concatenate([
        np.arange(i*N_per, (i+1)*N_per - n_test_per_nu) for i in range(len(nu_values))])
    test_idx = np.concatenate([
        np.arange((i+1)*N_per - n_test_per_nu, (i+1)*N_per) for i in range(len(nu_values))])

    s     = torch.from_numpy(s_np)
    kappa = torch.from_numpy(kappa_np[:, None])
    u0    = s[:, :Nx]   # first time step = initial condition

    # Coordinate grid matching training script: x_flat columns = [x, t]
    x_grid = torch.tensor(x_np, dtype=torch.float32)
    t_grid = torch.tensor(t_np, dtype=torch.float32)
    tt, xx = torch.meshgrid(t_grid, x_grid, indexing="ij")
    x_flat = torch.stack([xx.flatten(), tt.flatten()], dim=1)  # (Nt*Nx, 2)

    return s[test_idx], u0[test_idx], kappa[test_idx], kappa[train_idx], x_flat, "nu"


def _load_darcy(data_dir, beta_values, n_samples, n_test_per_beta):
    from utils.datasets import load_darcy_stacked

    entries = []
    for beta in beta_values:
        fname = f"2D_DarcyFlow_beta{beta}_Train.hdf5"
        for root in [data_dir, os.path.expanduser("~/data/2D/DarcyFlow")]:
            if root:
                fpath = os.path.join(root, fname)
                if os.path.exists(fpath):
                    entries.append((beta, fpath))
                    break
        else:
            print(f"  WARNING: beta={beta} file not found, skipping")

    s_np, a_np, kappa_np, xy_np, _, _ = load_darcy_stacked(entries, n_samples=n_samples)
    beta_loaded = [e[0] for e in entries]

    N_per = n_samples
    train_idx = np.concatenate([
        np.arange(i*N_per, (i+1)*N_per - n_test_per_beta) for i in range(len(beta_loaded))])
    test_idx = np.concatenate([
        np.arange((i+1)*N_per - n_test_per_beta, (i+1)*N_per) for i in range(len(beta_loaded))])

    s     = torch.from_numpy(s_np)
    a     = torch.from_numpy(a_np)
    kappa = torch.from_numpy(kappa_np[:, None])
    x_flat = torch.tensor(xy_np, dtype=torch.float32)  # (Nx*Ny, 2)

    # Darcy: sensors = permeability field a, target = pressure s
    return s[test_idx], a[test_idx], kappa[test_idx], kappa[train_idx], x_flat, "beta"


def _load_ns(data_dir, re_values, n_samples, n_test_per_re):
    from utils.datasets import load_ns_stacked

    entries = []
    for re in re_values:
        fpath = os.path.join(data_dir, f"2D_NavierStokes_Incomp_Re{re:05d}.npz")
        if os.path.exists(fpath):
            entries.append((re, fpath))
        else:
            print(f"  WARNING: Re={re} file not found, skipping")

    s_np, u0_np, kappa_np, _, _, _, _ = load_ns_stacked(entries, n_samples=n_samples)

    train_idx_list, test_idx_list = [], []
    for re_val in sorted(np.unique(kappa_np)):
        idx    = np.where(kappa_np == re_val)[0]
        n_test = min(n_test_per_re, len(idx))
        train_idx_list.append(idx[:-n_test])
        test_idx_list.append(idx[-n_test:])
    train_idx = np.concatenate(train_idx_list)
    test_idx  = np.concatenate(test_idx_list)

    s     = torch.from_numpy(s_np)
    u0    = torch.from_numpy(u0_np)
    kappa = torch.from_numpy(kappa_np[:, None])

    # NS training script normalises s; POD bases are scaled the same way,
    # so rel-L2 is invariant, but we must replicate the normalisation for
    # predictions to match the target scale.
    s_scale = float(s[train_idx].std())
    print(f"  NS s_scale={s_scale:.4f}")
    s = s / s_scale

    # x_flat = None: NS POD basis ignores coordinates (_eval_basis uses device only).
    # If NeuralPOD NS support is needed, build the full (Nt*Nx*Ny*2, 4) grid here.
    return s[test_idx], u0[test_idx], kappa[test_idx], kappa[train_idx], None, "Re"


# Main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=["burgers", "darcy", "ns"])
    p.add_argument("--run_dir",   required=True,
                   help="Directory with trainer.pt and model_online.pt")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--burgers_data_dir", default=os.path.expanduser("~/data/1D/Burgers/Train"))
    p.add_argument("--nu_values",        type=float, nargs="+", default=[0.001, 0.1, 1.0])
    p.add_argument("--n_samples_burgers", type=int, default=9500)
    p.add_argument("--n_test_burgers",    type=int, default=1000)

    p.add_argument("--darcy_data_dir",   default=None)
    p.add_argument("--beta_values",      type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--n_samples_darcy",  type=int, default=10000)
    p.add_argument("--n_test_darcy",     type=int, default=1000)

    p.add_argument("--ns_data_dir",      default=os.path.expanduser("~/data/2D/Navier_Stokes"))
    p.add_argument("--re_values",        type=int, nargs="+", default=[100, 1000, 3600, 10000])
    p.add_argument("--n_samples_ns",     type=int, default=5000)
    p.add_argument("--n_test_ns",        type=int, default=1000)

    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device
    print(f"device={device}  benchmark={args.benchmark}  run_dir={args.run_dir}")

    model, sub_trainers, cfg_online = _load_model(args.run_dir, device)
    M      = model.M
    stride = cfg_online.sensor_stride
    basis_types = [type(t).__name__ for t in sub_trainers]
    n_linears   = sum(1 for k in model.state_dict()
                      if k.startswith("gating.branch.net.") and k.endswith(".weight"))
    print(f"M={M}  basis={basis_types[0]}  sensor_stride={stride}  "
          f"hidden_dim={cfg_online.hidden_dim}  n_layers={n_linears - 1}")

    if args.benchmark == "burgers":
        s_test, u0_test, kappa_test, kappa_train, x_flat, param_label = _load_burgers(
            args.burgers_data_dir, args.nu_values, args.n_samples_burgers, args.n_test_burgers)
    elif args.benchmark == "darcy":
        s_test, u0_test, kappa_test, kappa_train, x_flat, param_label = _load_darcy(
            args.darcy_data_dir, args.beta_values, args.n_samples_darcy, args.n_test_darcy)
    else:
        s_test, u0_test, kappa_test, kappa_train, x_flat, param_label = _load_ns(
            args.ns_data_dir, args.re_values, args.n_samples_ns, args.n_test_ns)

    print(f"test: s={s_test.shape}  u0={u0_test.shape}  kappa={kappa_test.shape}")

    means, modes = _compute_bases(sub_trainers, x_flat, device)

    u0_sensors = u0_test[:, ::stride].to(device)
    s_true     = s_test.to(device)

    w0 = model.state_dict()["gating.branch.net.0.weight"]
    expected_sensors = w0.shape[1] - 1
    if u0_sensors.shape[1] != expected_sensors:
        raise RuntimeError(
            f"Sensor dimension mismatch: model expects {expected_sensors} sensors but got "
            f"{u0_sensors.shape[1]} (u0 has {u0_test.shape[1]} points, stride={stride}). "
            f"Check --{args.benchmark}_data_dir."
        )

    kappa_np         = kappa_test[:, 0].numpy()
    kappa_train_mean = float(kappa_train[:, 0].mean())
    print(f"kappa_train_mean={kappa_train_mean:.4g}")

    kappa_real = kappa_test.to(device)
    kappa_zero = torch.zeros_like(kappa_real)
    kappa_mean = torch.full_like(kappa_real, kappa_train_mean)

    print("\n--- Inference ---")
    pred_real, w_real = _predict_batched(model, u0_sensors, kappa_real, means, modes)
    pred_zero, w_zero = _predict_batched(model, u0_sensors, kappa_zero, means, modes)
    pred_mean, w_mean = _predict_batched(model, u0_sensors, kappa_mean, means, modes)

    s_true_cpu = s_true.cpu()
    err_real   = rel_l2_vec(pred_real, s_true_cpu)
    err_zero   = rel_l2_vec(pred_zero, s_true_cpu)
    err_mean   = rel_l2_vec(pred_mean, s_true_cpu)

    print(f"\n=== Kappa ablation: {args.benchmark.upper()} ===")
    print(f"  N_test={len(s_test)}, M={M}, param={param_label}\n")

    _report("kappa_real", err_real, kappa_np, param_label, w_real.numpy())
    print()
    _report("kappa_zero", err_zero, kappa_np, param_label, w_zero.numpy())
    print()
    _report("kappa_mean", err_mean, kappa_np, param_label, w_mean.numpy())

    print("\n--- Sensitivity ---")
    print(f"  rel-L2 degradation  kappa=0    vs real: {err_zero.mean() - err_real.mean():+.4f}")
    print(f"  rel-L2 degradation  kappa=mean vs real: {err_mean.mean() - err_real.mean():+.4f}")
    print(f"  mean |gate_zero - gate_real|: {np.abs(w_zero.numpy() - w_real.numpy()).mean():.4f}")
    print(f"  mean |gate_mean - gate_real|: {np.abs(w_mean.numpy() - w_real.numpy()).mean():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
