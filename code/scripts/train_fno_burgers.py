#!/usr/bin/env python
"""Train FNO on 1D Burgers — specialist (one nu) or joint (multiple nu).

Single --nu_values  → specialist mode: cross-nu generalization eval, cross_nu.png.
Multiple --nu_values → joint mode: per-nu error bar, error_per_nu.png.

FNO 2D: (u0, nu, x, t) -> s(x, t) for all (x, t) simultaneously.
Input:  (N, 4, Nx, Nt)  — channels: [u0 broadcast, log10(nu) norm, x-coord, t-coord]
Output: (N, 1, Nx, Nt)
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
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from neuralop.models import FNO
from utils.datasets import load_stacked


_ALL_NU = [0.001, 0.01, 0.1, 1.0]
_LOG10_NU_MIN = np.log10(min(_ALL_NU))   # -3
_LOG10_NU_MAX = np.log10(max(_ALL_NU))   #  0


def rel_l2(true, pred):
    """true, pred: (N, D) — returns (N,) per-sample relative L2."""
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nu_values",     type=float, nargs="+", required=True)
    p.add_argument("--run_name",      type=str,   default=None)
    p.add_argument("--results_dir",   type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "burgers"))
    p.add_argument("--n_samples",     type=int,   default=9500)
    p.add_argument("--n_test_per_nu", type=int,   default=1000)
    p.add_argument("--data_dir",      type=str,   default=os.path.expanduser("~/data/1D/Burgers/Train"))
    # FNO architecture
    p.add_argument("--n_modes_x",     type=int,   default=16)
    p.add_argument("--n_modes_t",     type=int,   default=16)
    p.add_argument("--hidden_dim",    type=int,   default=32)
    p.add_argument("--n_layers",      type=int,   default=4)
    # Training
    p.add_argument("--n_epochs",      type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--lr_step",       type=int,   default=40)
    p.add_argument("--lr_gamma",      type=float, default=0.5)
    p.add_argument("--log_every",     type=int,   default=5)
    # Misc
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--n_viz",         type=int,   default=3)
    return p.parse_args()


def build_fno_input(u0_np, kappa_np, x_np, t_np, Nx, Nt):
    """Build FNO input (N, 4, Nx, Nt).

    Channel 0: u0 broadcast along t
    Channel 1: log10(nu) normalised to [-1, 1], broadcast over (Nx, Nt)
    Channel 2: x-coordinate normalised to [-1, 1]
    Channel 3: t-coordinate normalised to [-1, 1]
    """
    N = len(u0_np)
    xx = np.tile(x_np[:, None], (1, Nt)).astype(np.float32)   # (Nx, Nt)
    tt = np.tile(t_np[None, :], (Nx, 1)).astype(np.float32)   # (Nx, Nt)
    xx = (xx - xx.min()) / (xx.max() - xx.min()) * 2 - 1
    tt = (tt - tt.min()) / (tt.max() - tt.min()) * 2 - 1

    u0_exp = np.repeat(u0_np[:, :, None], Nt, axis=2).astype(np.float32)         # (N, Nx, Nt)
    xx_b   = np.broadcast_to(xx[None], (N, Nx, Nt)).copy().astype(np.float32)
    tt_b   = np.broadcast_to(tt[None], (N, Nx, Nt)).copy().astype(np.float32)

    log_nu = np.log10(kappa_np.astype(np.float64)).astype(np.float32)
    norm_nu = (log_nu - _LOG10_NU_MIN) / (_LOG10_NU_MAX - _LOG10_NU_MIN) * 2 - 1  # (N,)
    nu_ch  = np.broadcast_to(norm_nu[:, None, None], (N, Nx, Nt)).copy().astype(np.float32)

    return np.stack([u0_exp, nu_ch, xx_b, tt_b], axis=1)  # (N, 4, Nx, Nt)


def predict_batched(model, X_tensor, device, batch_size=128):
    """Run model on X_tensor (N, 4, Nx, Nt), return (N, Nt*Nx) on CPU."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            out = model(X_tensor[i:i + batch_size].to(device))  # (B, 1, Nx, Nt)
            # transpose to (B, Nt, Nx) then flatten to (B, Nt*Nx) — matches s convention
            parts.append(out[:, 0].permute(0, 2, 1).reshape(len(out), -1).cpu())
    return torch.cat(parts, dim=0).numpy()


def main():
    args = parse_args()

    nu_values = args.nu_values
    is_joint  = len(nu_values) > 1

    if args.run_name:
        RUN_NAME = args.run_name
    elif is_joint:
        RUN_NAME = "fno_joint_burgers_v1"
    else:
        RUN_NAME = f"fno_burgers_nu{nu_values[0]}_v1"

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
    data_dir  = pathlib.Path(args.data_dir)
    local_dir = _PROJECT_ROOT / "data"

    def _find_path(nu):
        local  = local_dir / f"Burgers_Nu{nu}.hdf5"
        server = data_dir  / f"1D_Burgers_Sols_Nu{nu}.hdf5"
        return str(local) if local.exists() else str(server)

    entries = [(nu, _find_path(nu)) for nu in nu_values]
    s, kappa, x_np, t_np, Nx, Nt = load_stacked(entries, n_samples=args.n_samples)
    # s: (N_total, Nt*Nx),  kappa: (N_total,)

    n_nu  = len(nu_values)
    N_per = args.n_samples
    n_test = args.n_test_per_nu

    train_idx = np.concatenate([
        np.arange(i * N_per, (i + 1) * N_per - n_test) for i in range(n_nu)
    ])
    test_idx = np.concatenate([
        np.arange((i + 1) * N_per - n_test, (i + 1) * N_per) for i in range(n_nu)
    ])

    s_train     = s[train_idx]       # (N_train, Nt*Nx)
    s_test      = s[test_idx]        # (N_test,  Nt*Nx)
    kappa_train = kappa[train_idx]   # (N_train,)
    kappa_test  = kappa[test_idx]    # (N_test,)
    del s, kappa

    u0_train = s_train[:, :Nx]   # (N_train, Nx)  — t=0 slice
    u0_test  = s_test[:,  :Nx]   # (N_test,  Nx)

    # FNO input / target tensors
    X_train = torch.tensor(build_fno_input(u0_train, kappa_train, x_np, t_np, Nx, Nt))
    X_test  = torch.tensor(build_fno_input(u0_test,  kappa_test,  x_np, t_np, Nx, Nt))

    # Target: (N, 1, Nx, Nt) from (N, Nt*Nx) → (N, Nt, Nx) → transpose → add channel
    Y_train = torch.tensor(
        s_train.reshape(-1, Nt, Nx).transpose(0, 2, 1)[:, None, :, :])  # (N, 1, Nx, Nt)
    Y_test  = torch.tensor(
        s_test.reshape(-1, Nt, Nx).transpose(0, 2, 1)[:, None, :, :])

    N_train = len(train_idx)
    N_test  = len(test_idx)
    print(f"FNO input: {tuple(X_train.shape)}  output: {tuple(Y_train.shape)}")

    # Pre-load to GPU if data fits — eliminates per-batch CPU→GPU transfers
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

    # --- Build model ---
    model = FNO(
        n_modes=(args.n_modes_x, args.n_modes_t),
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
    mode_str = "joint" if is_joint else f"specialist nu={nu_values[0]}"
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
    ax1.set_title("FNO training dynamics", fontweight="bold")
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
        # --- Joint metrics: per-nu breakdown ---
        metrics: dict = {
            "run_name":       RUN_NAME,
            "n_params":       n_params,
            "n_train":        N_train,
            "n_test":         N_test,
            "overall_mean":   float(err_test.mean()),
            "overall_median": float(np.median(err_test)),
        }
        for i, nu in enumerate(nu_values):
            nu_sl = slice(i * n_test, (i + 1) * n_test)
            e_nu  = err_test[nu_sl]
            key   = f"nu{nu:.3f}"
            metrics[f"{key}_mean"]   = float(e_nu.mean())
            metrics[f"{key}_median"] = float(np.median(e_nu))
            metrics[f"{key}_std"]    = float(e_nu.std())
            print(f"  nu={nu:.3f}: mean={e_nu.mean():.4f}  median={np.median(e_nu):.4f}")

        # Per-nu bar chart
        nu_labels = [f"nu={nu:.3f}" for nu in nu_values]
        means_   = [metrics[f"nu{nu:.3f}_mean"]   for nu in nu_values]
        medians_ = [metrics[f"nu{nu:.3f}_median"] for nu in nu_values]
        x_pos    = np.arange(len(nu_labels))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x_pos - 0.2,  means_,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
        ax.bar(x_pos + 0.15, medians_, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
        ax.set_xticks(x_pos); ax.set_xticklabels(nu_labels)
        ax.set_ylabel("Relative L2 error")
        ax.set_title("FNO - per-nu test errors (joint)", fontweight="bold")
        ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "error_per_nu.png"), dpi=150, bbox_inches="tight")
        plt.close()

    else:
        # --- Specialist metrics + cross-nu generalization ---
        TRAIN_NU = nu_values[0]
        metrics: dict = {
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

        cross_nu_metrics: dict = {}
        for nu in _ALL_NU:
            local  = local_dir / f"Burgers_Nu{nu}.hdf5"
            server = data_dir  / f"1D_Burgers_Sols_Nu{nu}.hdf5"
            fpath  = local if local.exists() else server
            if not fpath.exists():
                print(f"  nu={nu:.3f}: file not found, skipping")
                continue
            with h5py.File(fpath, "r") as f:
                raw_nu = f["tensor"][args.n_samples - n_test : args.n_samples]
                if raw_nu.ndim == 4:
                    raw_nu = raw_nu[..., 0]
            # raw_nu: (n_test, Nt, Nx)
            u0_nu   = raw_nu[:, 0, :].astype(np.float32)
            s_nu    = raw_nu.reshape(n_test, -1).astype(np.float32)
            kap_nu  = np.full(n_test, nu, dtype=np.float32)
            X_nu    = torch.tensor(build_fno_input(u0_nu, kap_nu, x_np, t_np, Nx, Nt))
            pred_nu = predict_batched(model, X_nu, DEVICE)
            err_nu  = rel_l2(s_nu, pred_nu)
            cross_nu_metrics[nu] = {
                "mean":   float(err_nu.mean()),
                "median": float(np.median(err_nu)),
                "std":    float(err_nu.std()),
                "p95":    float(np.percentile(err_nu, 95)),
            }
            tag = " (trained)" if nu == TRAIN_NU else ""
            print(f"  nu={nu:.3f}{tag}: mean={err_nu.mean():.4f}  median={np.median(err_nu):.4f}")

        metrics["cross_nu"] = cross_nu_metrics

        # Cross-nu bar chart
        if cross_nu_metrics:
            nu_keys   = list(cross_nu_metrics.keys())
            nu_labels = [f"nu={nu:.3f}" for nu in nu_keys]
            means_    = [cross_nu_metrics[nu]["mean"]   for nu in nu_keys]
            medians_  = [cross_nu_metrics[nu]["median"] for nu in nu_keys]
            x_pos     = np.arange(len(nu_labels))

            _, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x_pos - 0.2,  means_,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
            ax.bar(x_pos + 0.15, medians_, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
            if TRAIN_NU in nu_keys:
                ax.axvline(nu_keys.index(TRAIN_NU), color="gray", ls="--", lw=1.2,
                           alpha=0.7, label="Trained on")
            ax.set_xticks(x_pos); ax.set_xticklabels(nu_labels)
            ax.set_ylabel("Relative L2 error")
            ax.set_title("FNO - cross-nu generalization", fontweight="bold")
            ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(os.path.join(RUN_DIR, "cross_nu.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # --- Reconstruction examples (from test set of first nu) ---
    rng  = np.random.default_rng(args.seed)
    idxs = rng.choice(n_test, size=args.n_viz, replace=False)

    fig, axes = plt.subplots(args.n_viz, 3, figsize=(14, 3 * args.n_viz))
    if args.n_viz == 1:
        axes = axes[None, :]
    for row, idx in enumerate(idxs):
        true   = s_test[idx].reshape(Nt, Nx)                             # (Nt, Nx)
        pred_i = predict_batched(model, X_test[idx:idx + 1], DEVICE
                                 ).reshape(Nt, Nx)
        err_i  = np.abs(true - pred_i)
        vmax   = np.abs(true).max()
        rl2    = float(np.linalg.norm(true - pred_i) / np.linalg.norm(true))
        for col, (arr, title, cmap, vmin, vm) in enumerate([
            (true,   "Ground Truth", "RdBu_r",  -vmax, vmax),
            (pred_i, "FNO",          "RdBu_r",  -vmax, vmax),
            (err_i,  "Abs. Error",   "Oranges",     0, err_i.max()),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
                           extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
                           vmin=vmin, vmax=vm)
            if row == 0:
                ax.set_title(title, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"t   (rel L2={rl2:.3f})")
            if row == args.n_viz - 1:
                ax.set_xlabel("x")
            ax.spines[["top", "right"]].set_visible(False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("FNO: reconstruction examples", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "fno_reconstruction.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Error distribution (test set) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(err_test, bins=30, color=C0, alpha=0.8, linewidth=0)
    ax.axvline(err_test.mean(),     color=C1, ls="--", lw=1.5,
               label=f"Mean {err_test.mean():.4f}")
    ax.axvline(np.median(err_test), color=C2, ls="--", lw=1.5,
               label=f"Median {np.median(err_test):.4f}")
    ax.set_xlabel("Relative L2 error"); ax.set_ylabel("Count")
    ax.set_title("Test error distribution", fontweight="bold")
    ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(np.sort(err_test), color=C0, lw=1.5)
    ax.axhline(err_test.mean(), color=C1, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Trajectory rank"); ax.set_ylabel("Relative L2 error")
    ax.set_title("Sorted test errors", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("FNO - test set errors", fontweight="bold")
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

    metrics["hparams"] = vars(args)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")


if __name__ == "__main__":
    main()
