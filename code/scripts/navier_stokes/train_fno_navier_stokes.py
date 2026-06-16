#!/usr/bin/env python
"""Train FNO on 2D incompressible Navier-Stokes - specialist (one Re) or joint (multiple Re).

Input:  (N, 2, 64, 64) - [u0, v0]  (no Re channel - faithful to original FNO)
Output: (N, 2*Nt, 64, 64) - full velocity trajectory

Usage:
  Specialist: python code/scripts/train_fno_navier_stokes.py --re_values 1000
  Joint:      python code/scripts/train_fno_navier_stokes.py --re_values 100 1000 3600 10000
"""
import argparse
import json
import os
import pathlib
import sys
import time

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
from utils.datasets import load_ns_stacked, measure_inference_time

_ALL_RE = [100, 1000, 3600, 10000]


def rel_l2(true, pred):
    """true, pred: (N, D) → (N,) per-sample relative L2."""
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


def build_fno_input(u0_np, kappa_np, Nx, Ny):
    """u0_np: (N, Nx*Ny*2), kappa_np: (N,) → (N, 2, Nx, Ny)  (no Re channel)"""
    N = len(u0_np)
    u0g = u0_np.reshape(N, Nx, Ny, 2).transpose(0, 3, 1, 2).astype(np.float32)
    return u0g  # (N, 2, Nx, Ny)


def build_target(s_np, Nt, Nx, Ny):
    """s_np: (N, Nt*Nx*Ny*2) C-order → (N, 2*Nt, Nx, Ny)"""
    N = len(s_np)
    g = s_np.reshape(N, Nt, Nx, Ny, 2)   # (N, Nt, Nx, Ny, 2)
    g = g.transpose(0, 4, 1, 2, 3)        # (N, 2, Nt, Nx, Ny)
    return g.reshape(N, 2 * Nt, Nx, Ny).astype(np.float32)


def pred_to_flat(pred_np, Nt, Nx, Ny):
    """(N, 2*Nt, Nx, Ny) → (N, Nt*Nx*Ny*2) - same flat layout as s_np."""
    N = len(pred_np)
    g = pred_np.reshape(N, 2, Nt, Nx, Ny)   # (N, 2, Nt, Nx, Ny)
    g = g.transpose(0, 2, 3, 4, 1)           # (N, Nt, Nx, Ny, 2)
    return g.reshape(N, -1)


def predict_batched(model, X_tensor, device, batch_size=32):
    """(N, 3, Nx, Ny) → (N, 2*Nt, Nx, Ny) on CPU."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            parts.append(model(X_tensor[i:i + batch_size].to(device)).cpu())
    return torch.cat(parts, dim=0).numpy()


def _data_path(re, data_dir):
    return os.path.join(data_dir, f"2D_NavierStokes_Incomp_Re{re:05d}.npz")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--re_values",     type=int,   nargs="+", required=True)
    p.add_argument("--run_name",      type=str,   default=None)
    p.add_argument("--results_dir",   type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "navier_stokes"))
    p.add_argument("--n_samples",     type=int,   default=5000)
    p.add_argument("--n_test_per_re", type=int,   default=1000)
    p.add_argument("--data_dir",      type=str,   default=os.path.expanduser("~/data/2D/Navier_Stokes"))
    # FNO arch
    p.add_argument("--n_modes",       type=int,   default=12)
    p.add_argument("--hidden_dim",    type=int,   default=64)
    p.add_argument("--n_layers",      type=int,   default=4)
    # Training
    p.add_argument("--n_epochs",      type=int,   default=70)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--lr_step",       type=int,   default=150)
    p.add_argument("--lr_gamma",      type=float, default=0.5)
    p.add_argument("--log_every",     type=int,   default=20)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--n_viz",         type=int,   default=2)
    return p.parse_args()


def main():
    args = parse_args()
    is_joint = len(args.re_values) > 1

    if args.run_name:
        RUN_NAME = args.run_name
    elif is_joint:
        RUN_NAME = "fno_joint_navier_stokes_v1"
    else:
        RUN_NAME = f"fno_navier_stokes_re{args.re_values[0]}_v1"

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

    C0, C1 = plt.cm.tab10(0), plt.cm.tab10(1)

    entries = []
    for re in args.re_values:
        fpath = _data_path(re, args.data_dir)
        if not os.path.exists(fpath):
            print(f"  Re={re}: file not found, skipping")
            continue
        entries.append((re, fpath))
    if not entries:
        raise RuntimeError("No data files found. Check --data_dir.")

    re_loaded = [e[0] for e in entries]
    s_np, u0_np, kappa_np, _, Nx, Ny, Nt = load_ns_stacked(entries, n_samples=args.n_samples)
    out_ch = 2 * Nt
    print(f"Loaded: N={len(s_np)}  Nx={Nx}  Ny={Ny}  Nt={Nt}  Re={re_loaded}")

    # kappa-based train/test split (robust after NaN filtering)
    train_idx, test_idx = [], []
    for re_val in sorted(np.unique(kappa_np)):
        idx = np.where(kappa_np == re_val)[0]
        n_test = min(args.n_test_per_re, len(idx))
        train_idx.append(idx[:-n_test])
        test_idx.append(idx[-n_test:])
    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)
    N_train, N_test = len(train_idx), len(test_idx)

    kappa_train = kappa_np[train_idx]
    kappa_test  = kappa_np[test_idx]

    X_train = torch.tensor(build_fno_input(u0_np[train_idx], kappa_train, Nx, Ny))
    X_test  = torch.tensor(build_fno_input(u0_np[test_idx],  kappa_test,  Nx, Ny))
    Y_train = torch.tensor(build_target(s_np[train_idx], Nt, Nx, Ny))
    Y_test  = torch.tensor(build_target(s_np[test_idx],  Nt, Nx, Ny))

    s_train_flat = s_np[train_idx].copy()
    s_test_flat  = s_np[test_idx].copy()
    del u0_np, s_np

    print(f"FNO input: {tuple(X_train.shape)}  output: {tuple(Y_train.shape)}")

    # GPU pre-load if data fits
    _pin = False
    if DEVICE == "cuda":
        _data_gb = (X_train.numel() + Y_train.numel() + X_test.numel()) * 4 / 1e9
        _free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if _data_gb < _free_gb * 0.45:
            print(f"Pre-loading {_data_gb:.1f} GB to GPU")
            X_train, Y_train, X_test = X_train.to(DEVICE), Y_train.to(DEVICE), X_test.to(DEVICE)
        else:
            print(f"Data {_data_gb:.1f} GB > threshold, using pin_memory")
            _pin = True

    model = FNO(
        n_modes=(args.n_modes, args.n_modes),
        in_channels=2,
        out_channels=out_ch,
        hidden_channels=args.hidden_dim,
        n_layers=args.n_layers,
        use_channel_mlp=True,
        positional_embedding=None,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"FNO params: {n_params:,}  out_channels={out_ch}")

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    dl    = DataLoader(TensorDataset(X_train, Y_train),
                       batch_size=args.batch_size, shuffle=True,
                       pin_memory=_pin, num_workers=0)

    mode_str = "joint" if is_joint else f"specialist Re={args.re_values[0]}"
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

        if epoch % args.log_every == 0 or epoch == 1:
            p_val = predict_batched(model, X_test, DEVICE)
            err_val = rel_l2(s_test_flat, pred_to_flat(p_val, Nt, Nx, Ny)).mean()
            history_val.append((epoch, err_val))
            print(f"  epoch {epoch:4d} | train_mse={avg:.4e} | val_rel_l2={err_val:.4f} | {time.time()-t0:.0f}s")
        else:
            print(f"  epoch {epoch:4d} | train_mse={avg:.4e} | {time.time()-t0:.0f}s")

    print(f"  done: {time.time()-t0:.0f}s total")

    # Training dynamics plot
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.semilogy(range(1, len(history_train)+1), history_train, color=C0, lw=1.5, label="Train MSE")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE", color=C0)
    ax1.tick_params(axis="y", labelcolor=C0); ax1.grid(True, ls="--", alpha=0.25)
    ax1.spines[["top"]].set_visible(False)
    if history_val:
        ve, vl = zip(*history_val)
        ax2 = ax1.twinx()
        ax2.semilogy(ve, vl, color=C1, lw=1.5, ls="--", label="Val rel-L2")
        ax2.set_ylabel("Val rel-L2", color=C1); ax2.tick_params(axis="y", labelcolor=C1)
        ax2.spines[["top"]].set_visible(False)
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, framealpha=0.7)
    ax1.set_title("FNO NS training dynamics", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "training_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    pred_test_flat = pred_to_flat(predict_batched(model, X_test, DEVICE), Nt, Nx, Ny)
    err_test  = rel_l2(s_test_flat, pred_test_flat)

    rng = np.random.default_rng(args.seed)
    idx_s = rng.choice(N_train, size=min(2000, N_train), replace=False)
    X_s = X_train[idx_s] if isinstance(X_train, torch.Tensor) else torch.tensor(X_train[idx_s])
    pred_train_flat = pred_to_flat(predict_batched(model, X_s, DEVICE), Nt, Nx, Ny)
    err_train = rel_l2(s_train_flat[idx_s], pred_train_flat)

    print(f"Train | mean={err_train.mean():.4f}  median={np.median(err_train):.4f}  std={err_train.std():.4f}")
    print(f"Test  | mean={err_test.mean():.4f}  median={np.median(err_test):.4f}  std={err_test.std():.4f}  p95={np.percentile(err_test,95):.4f}")

    metrics = {
        "run_name": RUN_NAME, "n_params": n_params,
        "n_train": N_train, "n_test": N_test,
        "overall_mean": float(err_test.mean()), "overall_median": float(np.median(err_test)),
    }

    # Per-Re metrics on test set
    cross_re_metrics = {}
    print("Per-Re test errors:")
    for re in sorted(np.unique(kappa_test)):
        mask = kappa_test == re
        e = err_test[mask]
        cross_re_metrics[int(re)] = {
            "mean": float(e.mean()), "median": float(np.median(e)),
            "std": float(e.std()), "p95": float(np.percentile(e, 95)),
        }
        print(f"  Re={int(re):5d}: mean={e.mean():.4f}  median={np.median(e):.4f}")
    metrics["cross_re"] = cross_re_metrics

    # Cross-Re generalization (specialist only)
    if not is_joint:
        print("Cross-Re generalization:")
        for re_eval in _ALL_RE:
            fpath = _data_path(re_eval, args.data_dir)
            if not os.path.exists(fpath):
                continue
            s_re, u0_re, _, _, _, _, _ = load_ns_stacked([(re_eval, fpath)], n_samples=args.n_samples)
            s_re_test  = s_re[-args.n_test_per_re:]
            u0_re_test = u0_re[-args.n_test_per_re:]
            kap_re     = np.full(len(u0_re_test), float(re_eval), dtype=np.float32)
            X_re = torch.tensor(build_fno_input(u0_re_test, kap_re, Nx, Ny))
            err_re = rel_l2(s_re_test, pred_to_flat(predict_batched(model, X_re, DEVICE), Nt, Nx, Ny))
            tag = " (trained)" if re_eval == args.re_values[0] else ""
            cross_re_metrics[re_eval] = {
                "mean": float(err_re.mean()), "median": float(np.median(err_re)),
                "std": float(err_re.std()), "p95": float(np.percentile(err_re, 95)),
            }
            print(f"  Re={re_eval:5d}{tag}: mean={err_re.mean():.4f}  median={np.median(err_re):.4f}")
        metrics["cross_re"] = cross_re_metrics

    # Per-Re error bar plot
    re_keys = sorted(cross_re_metrics.keys())
    if is_joint and len(re_keys) > 1:
        re_labels = [f"Re={r}" for r in re_keys]
        means_   = [cross_re_metrics[r]["mean"]   for r in re_keys]
        medians_ = [cross_re_metrics[r]["median"] for r in re_keys]
        x_pos = np.arange(len(re_labels))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x_pos - 0.2,  means_,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
        ax.bar(x_pos + 0.15, medians_, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
        ax.set_xticks(x_pos); ax.set_xticklabels(re_labels)
        ax.set_ylabel("Relative L2 error")
        ax.set_title("FNO - per-Re test errors (joint, NS)", fontweight="bold")
        ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "error_per_re.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Error distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(err_test, bins=40, color=C0, alpha=0.8, linewidth=0)
    ax.axvline(err_test.mean(), color=C1, ls="--", lw=1.5, label=f"Mean {err_test.mean():.4f}")
    ax.axvline(np.median(err_test), color="green", ls="--", lw=1.5, label=f"Median {np.median(err_test):.4f}")
    ax.set_xlabel("Relative L2 error"); ax.set_ylabel("Count")
    ax.set_title("Test error distribution"); ax.legend(framealpha=0.7)
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(np.sort(err_test), color=C0, lw=1.5)
    ax.axhline(err_test.mean(), color=C1, ls="--", lw=1.2)
    ax.set_xlabel("Sample rank"); ax.set_ylabel("Relative L2 error")
    ax.set_title("Sorted test errors"); ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("FNO Navier-Stokes - test errors", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "err_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Reconstruction examples
    idxs = rng.choice(N_test, size=min(args.n_viz, N_test), replace=False)
    time_show = [0, Nt // 2, Nt - 1]

    for si, idx in enumerate(idxs):
        s_true = s_test_flat[idx].reshape(Nt, Nx, Ny, 2)
        s_pred = pred_test_flat[idx].reshape(Nt, Nx, Ny, 2)
        re_i   = int(kappa_test[idx])
        rl2    = float(err_test[idx])

        fig, axes_g = plt.subplots(4, len(time_show), figsize=(4 * len(time_show), 10))
        fig.suptitle(f"FNO NS  Re={re_i}  rel-L2={rl2:.4f}", fontweight="bold")
        for ti, t in enumerate(time_show):
            for ci, label in enumerate(["u", "v"]):
                tr = s_true[t, :, :, ci]
                pr = s_pred[t, :, :, ci]
                vm = np.abs(tr).max()
                r  = 2 * ci
                axes_g[r,   ti].imshow(tr, cmap="RdBu_r", vmin=-vm, vmax=vm, origin="lower")
                axes_g[r,   ti].set_title(f"{label}_true t={t}", fontsize=8)
                axes_g[r,   ti].axis("off")
                axes_g[r+1, ti].imshow(pr, cmap="RdBu_r", vmin=-vm, vmax=vm, origin="lower")
                axes_g[r+1, ti].set_title(f"{label}_pred t={t}", fontsize=8)
                axes_g[r+1, ti].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, f"reconstruction_{si}.png"), dpi=100, bbox_inches="tight")
        plt.close()

    # Checkpoint
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
