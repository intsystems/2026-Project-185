#!/usr/bin/env python
"""Train FNO on 2D Darcy Flow — specialist (one beta) or joint (multiple beta).

Single --beta_values  → specialist: cross-beta generalization eval, cross_beta.png.
Multiple --beta_values → joint: per-beta error bar, error_per_beta.png.

FNO 2D: a(x,y) + beta → u(x,y) for all (x,y) simultaneously.
Input:  (N, 4, Nx, Ny) — [permeability a, log10(beta) norm, x-coord, y-coord]
Output: (N, 1, Nx, Ny)
"""
import argparse
import json
import os
import pathlib
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from neuralop.models import FNO
from utils.datasets import load_darcy_stacked, DATA_DIR, measure_inference_time

_ALL_BETA = [0.01, 0.1, 1.0, 10.0, 100.0]
_LOG10_BETA_MIN = np.log10(min(_ALL_BETA))   # -2
_LOG10_BETA_MAX = np.log10(max(_ALL_BETA))   #  2


def rel_l2(true, pred):
    """true, pred: (N, D) — returns (N,) per-sample relative L2."""
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


def _data_path(beta: float, data_dir: str = None) -> str:
    filename = f"2D_DarcyFlow_beta{beta}_Train.hdf5"
    if data_dir:
        return os.path.join(data_dir, filename)
    local  = pathlib.Path(DATA_DIR) / filename
    server = pathlib.Path(os.path.expanduser("~/data/2D/DarcyFlow")) / filename
    return str(local) if local.exists() else str(server)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beta_values",      type=float, nargs="+", required=True)
    p.add_argument("--run_name",         type=str,   default=None)
    p.add_argument("--results_dir",      type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "darcy"))
    p.add_argument("--n_samples",        type=int,   default=10000)
    p.add_argument("--n_test_per_beta",  type=int,   default=1000)
    p.add_argument("--data_dir",         type=str,   default=None)
    # FNO architecture
    p.add_argument("--n_modes",          type=int,   default=16,
                   help="Fourier modes per spatial dimension (same for x and y)")
    p.add_argument("--hidden_dim",       type=int,   default=32)
    p.add_argument("--n_layers",         type=int,   default=4)
    # Training
    p.add_argument("--n_epochs",         type=int,   default=105)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--lr_step",          type=int,   default=80)
    p.add_argument("--lr_gamma",         type=float, default=0.5)
    p.add_argument("--log_every",        type=int,   default=10)
    # Misc
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--n_viz",            type=int,   default=3)
    return p.parse_args()


def build_fno_input(a_np, kappa_np, xy_np, Nx, Ny):
    """Build FNO input (N, 4, Nx, Ny).

    Channel 0: permeability a(x,y)
    Channel 1: log10(beta) normalised to [-1, 1]
    Channel 2: x-coordinate normalised to [-1, 1]
    Channel 3: y-coordinate normalised to [-1, 1]
    """
    N = len(a_np)
    a_grid = a_np.reshape(N, Nx, Ny).astype(np.float32)

    xx = xy_np[:, 0].reshape(Nx, Ny).astype(np.float32)
    yy = xy_np[:, 1].reshape(Nx, Ny).astype(np.float32)
    xx = (xx - xx.min()) / (xx.max() - xx.min()) * 2 - 1
    yy = (yy - yy.min()) / (yy.max() - yy.min()) * 2 - 1
    xx_b = np.broadcast_to(xx[None], (N, Nx, Ny)).copy()
    yy_b = np.broadcast_to(yy[None], (N, Nx, Ny)).copy()

    log_kappa  = np.log10(kappa_np.astype(np.float64)).astype(np.float32)
    norm_kappa = (log_kappa - _LOG10_BETA_MIN) / (_LOG10_BETA_MAX - _LOG10_BETA_MIN) * 2 - 1
    kappa_ch   = np.broadcast_to(norm_kappa[:, None, None], (N, Nx, Ny)).copy()

    return np.stack([a_grid, kappa_ch, xx_b, yy_b], axis=1).astype(np.float32)  # (N, 4, Nx, Ny)


def predict_batched(model, X_tensor, device, batch_size=64):
    """X_tensor: (N, 4, Nx, Ny) → returns (N, Nx*Ny) on CPU."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            out = model(X_tensor[i:i + batch_size].to(device))  # (B, 1, Nx, Ny)
            parts.append(out[:, 0].reshape(len(out), -1).cpu())
    return torch.cat(parts, dim=0).numpy()


def main():
    args = parse_args()

    beta_values = args.beta_values
    is_joint    = len(beta_values) > 1

    if args.run_name:
        RUN_NAME = args.run_name
    elif is_joint:
        RUN_NAME = "fno_joint_darcy_v1"
    else:
        RUN_NAME = f"fno_darcy_beta{beta_values[0]}_v1"

    RUN_DIR = os.path.join(args.results_dir, RUN_NAME)
    os.makedirs(RUN_DIR, exist_ok=True)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"device={DEVICE}  run_dir={os.path.abspath(RUN_DIR)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    C0, C1, C2 = plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2)

    # --- Data loading ---
    entries = []
    for beta in beta_values:
        fpath = _data_path(beta, args.data_dir)
        if not os.path.exists(fpath):
            print(f"  beta={beta}: file not found, skipping")
            continue
        entries.append((beta, fpath))

    if not entries:
        raise RuntimeError("No data files found. Check --data_dir.")

    beta_loaded = [e[0] for e in entries]
    s_np, a_np, kappa_np, xy_np, Nx, Ny = load_darcy_stacked(entries, n_samples=args.n_samples)
    Nxy = Nx * Ny

    x_np_1d = xy_np[::Ny, 0]   # unique x coords (Nx,)
    y_np_1d = xy_np[:Ny,  1]   # unique y coords (Ny,)

    n_beta = len(beta_loaded)
    N_per  = args.n_samples
    n_test = args.n_test_per_beta

    train_idx = np.concatenate([
        np.arange(i * N_per, (i + 1) * N_per - n_test) for i in range(n_beta)
    ])
    test_idx = np.concatenate([
        np.arange((i + 1) * N_per - n_test, (i + 1) * N_per) for i in range(n_beta)
    ])

    s_train     = s_np[train_idx];     s_test     = s_np[test_idx]
    a_train     = a_np[train_idx];     a_test     = a_np[test_idx]
    kappa_train = kappa_np[train_idx]; kappa_test = kappa_np[test_idx]
    del s_np, a_np, kappa_np

    N_train = len(train_idx)
    N_test  = len(test_idx)

    X_train = torch.tensor(build_fno_input(a_train, kappa_train, xy_np, Nx, Ny)).float()
    X_test  = torch.tensor(build_fno_input(a_test,  kappa_test,  xy_np, Nx, Ny)).float()
    Y_train = torch.tensor(s_train.reshape(-1, Nx, Ny)[:, None, :, :]).float()  # (N, 1, Nx, Ny)
    Y_test  = torch.tensor(s_test.reshape(-1,  Nx, Ny)[:, None, :, :]).float()

    print(f"FNO input: {tuple(X_train.shape)}  output: {tuple(Y_train.shape)}")

    # GPU pre-load
    _pin = False
    if DEVICE == "cuda":
        _data_gb = (X_train.numel() + Y_train.numel() + X_test.numel()) * 4 / 1e9
        _free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if _data_gb < _free_gb * 0.45:
            print(f"Pre-loading {_data_gb:.1f} GB data to GPU")
            X_train = X_train.to(DEVICE)
            Y_train = Y_train.to(DEVICE)
            X_test  = X_test.to(DEVICE)
        else:
            print(f"Data {_data_gb:.1f} GB > {_free_gb*0.45:.1f} GB threshold, using pin_memory")
            _pin = True

    # --- Model ---
    model = FNO(
        n_modes=(args.n_modes, args.n_modes),
        in_channels=4,
        out_channels=1,
        hidden_channels=args.hidden_dim,
        n_layers=args.n_layers,
        use_channel_mlp=True,
        positional_embedding=None,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"FNO params: {n_params:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    dl    = DataLoader(TensorDataset(X_train, Y_train),
                       batch_size=args.batch_size, shuffle=True,
                       pin_memory=_pin, num_workers=0)

    # --- Training ---
    mode_str = "joint" if is_joint else f"specialist beta={beta_values[0]}"
    print(f"=== Training FNO | {mode_str} | N_train={N_train} | epochs={args.n_epochs} ===")
    t0 = time.time()
    history_train, history_val = [], []

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=_pin)
            yb = yb.to(DEVICE, non_blocking=_pin)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        sched.step()
        avg = total / N_train
        history_train.append(avg)

        elapsed = time.time() - t0
        if epoch % args.log_every == 0 or epoch == 1:
            pred_val = predict_batched(model, X_test, DEVICE)
            err_val  = rel_l2(s_test, pred_val).mean()
            history_val.append((epoch, err_val))
            print(f"  epoch {epoch:4d} | train_mse={avg:.4e} | val_rel_l2={err_val:.4f} | {elapsed:.0f}s")
        else:
            print(f"  epoch {epoch:4d} | train_mse={avg:.4e} | {elapsed:.0f}s")

    print(f"  done: total time {time.time() - t0:.0f}s")

    # --- Training dynamics plot ---
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.semilogy(range(1, len(history_train) + 1), history_train, color=C0, lw=1.5, label="Train MSE")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE", color=C0)
    ax1.tick_params(axis="y", labelcolor=C0)
    ax1.grid(True, ls="--", alpha=0.25)
    ax1.spines[["top"]].set_visible(False)
    if history_val:
        ve, vl = zip(*history_val)
        ax2 = ax1.twinx()
        ax2.semilogy(ve, vl, color=C1, lw=1.5, ls="--", label="Val rel-L2")
        ax2.set_ylabel("Val rel-L2", color=C1)
        ax2.tick_params(axis="y", labelcolor=C1)
        ax2.spines[["top"]].set_visible(False)
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, framealpha=0.7)
    ax1.set_title("FNO Darcy training dynamics", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "training_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Final evaluation ---
    pred_train = predict_batched(model, X_train, DEVICE)
    pred_test  = predict_batched(model, X_test,  DEVICE)

    err_train = rel_l2(s_train, pred_train)
    err_test  = rel_l2(s_test,  pred_test)

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test, 95):.4f}")

    if is_joint:
        metrics = {
            "run_name":       RUN_NAME,
            "n_params":       n_params,
            "n_train":        N_train,
            "n_test":         N_test,
            "overall_mean":   float(err_test.mean()),
            "overall_median": float(np.median(err_test)),
        }
        for i, beta in enumerate(beta_loaded):
            sl  = slice(i * n_test, (i + 1) * n_test)
            e_b = err_test[sl]
            key = f"beta{beta:.4g}"
            metrics[f"{key}_mean"]   = float(e_b.mean())
            metrics[f"{key}_median"] = float(np.median(e_b))
            metrics[f"{key}_std"]    = float(e_b.std())
            print(f"  beta={beta:.4g}: mean={e_b.mean():.4f}  median={np.median(e_b):.4f}")

        beta_labels = [f"β={b:.4g}" for b in beta_loaded]
        means_      = [metrics[f"beta{b:.4g}_mean"]   for b in beta_loaded]
        medians_    = [metrics[f"beta{b:.4g}_median"] for b in beta_loaded]
        x_pos = np.arange(len(beta_labels))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x_pos - 0.2,  means_,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
        ax.bar(x_pos + 0.15, medians_, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
        ax.set_xticks(x_pos); ax.set_xticklabels(beta_labels)
        ax.set_ylabel("Relative L2 error")
        ax.set_title("FNO - per-beta test errors (joint, Darcy)", fontweight="bold")
        ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "error_per_beta.png"), dpi=150, bbox_inches="tight")
        plt.close()

    else:
        TRAIN_BETA = beta_values[0]
        metrics = {
            "run_name":     RUN_NAME,
            "n_params":     n_params,
            "n_train":      N_train,
            "n_test":       N_test,
            "train_mean":   float(err_train.mean()),
            "train_median": float(np.median(err_train)),
            "train_std":    float(err_train.std()),
            "test_mean":    float(err_test.mean()),
            "test_median":  float(np.median(err_test)),
            "test_std":     float(err_test.std()),
            "test_p95":     float(np.percentile(err_test, 95)),
        }

        cross_beta_metrics = {}
        for beta in _ALL_BETA:
            fpath = _data_path(beta, args.data_dir)
            if not os.path.exists(fpath):
                print(f"  beta={beta:.4g}: file not found, skipping")
                continue
            with h5py.File(fpath, "r") as f:
                raw_s = f["tensor"][args.n_samples - n_test : args.n_samples, 0]  # (n_test, Nx, Ny)
                raw_a = f["nu"][args.n_samples - n_test : args.n_samples]          # (n_test, Nx, Ny)
            s_b   = raw_s.reshape(n_test, Nxy).astype(np.float32)
            a_b   = raw_a.reshape(n_test, Nxy).astype(np.float32)
            kap_b = np.full(n_test, beta, dtype=np.float32)
            X_b   = torch.tensor(build_fno_input(a_b, kap_b, xy_np, Nx, Ny))
            pred_b = predict_batched(model, X_b, DEVICE)
            err_b  = rel_l2(s_b, pred_b)
            cross_beta_metrics[beta] = {
                "mean":   float(err_b.mean()),
                "median": float(np.median(err_b)),
                "std":    float(err_b.std()),
                "p95":    float(np.percentile(err_b, 95)),
            }
            tag = " (trained)" if beta == TRAIN_BETA else ""
            print(f"  beta={beta:.4g}{tag}: mean={err_b.mean():.4f}  median={np.median(err_b):.4f}")

        metrics["cross_beta"] = cross_beta_metrics

        if cross_beta_metrics:
            beta_keys   = list(cross_beta_metrics.keys())
            beta_labels = [f"β={b:.4g}" for b in beta_keys]
            means_      = [cross_beta_metrics[b]["mean"]   for b in beta_keys]
            medians_    = [cross_beta_metrics[b]["median"] for b in beta_keys]
            x_pos = np.arange(len(beta_labels))
            _, ax = plt.subplots(figsize=(9, 4))
            ax.bar(x_pos - 0.2,  means_,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
            ax.bar(x_pos + 0.15, medians_, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
            if TRAIN_BETA in beta_keys:
                ax.axvline(beta_keys.index(TRAIN_BETA), color="gray", ls="--", lw=1.2,
                           alpha=0.7, label="Trained on")
            ax.set_xticks(x_pos); ax.set_xticklabels(beta_labels)
            ax.set_ylabel("Relative L2 error")
            ax.set_title("FNO - cross-beta generalization (Darcy)", fontweight="bold")
            ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(RUN_DIR, "cross_beta.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # --- Reconstruction examples ---
    rng  = np.random.default_rng(args.seed)
    idxs = rng.choice(n_test, size=args.n_viz, replace=False)

    fig, axes = plt.subplots(args.n_viz, 4, figsize=(16, 3.5 * args.n_viz))
    if args.n_viz == 1:
        axes = axes[None, :]
    for row, idx in enumerate(idxs):
        a_i    = a_test[idx].reshape(Nx, Ny)
        true   = s_test[idx].reshape(Nx, Ny)
        pred_i = pred_test[idx].reshape(Nx, Ny)
        err_i  = np.abs(true - pred_i)
        vmax_u = np.abs(true).max()
        rl2    = float(np.linalg.norm(true - pred_i) / np.linalg.norm(true))
        ext    = [y_np_1d.min(), y_np_1d.max(), x_np_1d.min(), x_np_1d.max()]
        for col, (arr, title, cmap, vmin, vm) in enumerate([
            (a_i,   "Input a(x,y)", "viridis",  a_i.min(),  a_i.max()),
            (true,  "True u(x,y)",  "RdBu_r",   -vmax_u,    vmax_u),
            (pred_i,"FNO pred",     "RdBu_r",   -vmax_u,    vmax_u),
            (err_i, "Abs. Error",   "Oranges",   0,          err_i.max()),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
                           extent=ext, vmin=vmin, vmax=vm)
            if row == 0:
                ax.set_title(title, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"x   (rel L2={rl2:.3f})")
            if row == args.n_viz - 1:
                ax.set_xlabel("y")
            ax.spines[["top", "right"]].set_visible(False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("FNO Darcy: reconstruction examples", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "fno_reconstruction.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Error distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(err_test, bins=30, color=C0, alpha=0.8, linewidth=0)
    ax.axvline(err_test.mean(),     color=C1, ls="--", lw=1.5, label=f"Mean {err_test.mean():.4f}")
    ax.axvline(np.median(err_test), color=C2, ls="--", lw=1.5, label=f"Median {np.median(err_test):.4f}")
    ax.set_xlabel("Relative L2 error"); ax.set_ylabel("Count")
    ax.set_title("Test error distribution", fontweight="bold")
    ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(np.sort(err_test), color=C0, lw=1.5)
    ax.axhline(err_test.mean(), color=C1, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Sample rank"); ax.set_ylabel("Relative L2 error")
    ax.set_title("Sorted test errors", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("FNO Darcy - test set errors", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "fno_err_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Checkpoint ---
    torch.save({
        "model_state": model.state_dict(),
        "metrics":     metrics,
        "run_name":    RUN_NAME,
        "hparams":     vars(args),
    }, os.path.join(RUN_DIR, "model.pt"))

    # Inference time
    _inf_ms = measure_inference_time(
        lambda: predict_batched(model, X_test, DEVICE),
        device=DEVICE
    )
    metrics["inference_ms_total"] = _inf_ms
    metrics["inference_ms_per_sample"] = _inf_ms / len(X_test)

    metrics["hparams"] = vars(args)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
