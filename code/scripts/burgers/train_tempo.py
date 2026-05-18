#!/usr/bin/env python
"""Train TEMPO (offline EM + online gating) on multiple viscosity values."""
import argparse
import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]

sys.path.insert(0, str(_SCRIPT_DIR.parents[1]))
from models.tempo import TEMPOTrainer, TEMPOConfig, pod_factory, fourier_pod_factory
from models.pod import PODConfig
from models.fourier_neural_pod import FourierNeuralPODConfig
from models.tempo_online import build_tempo_online, TEMPOOnlineConfig, _num_modes
from utils.datasets import load_stacked, measure_inference_time


def rel_l2_vec(true, pred):
    return np.linalg.norm(true - pred, axis=1) / np.linalg.norm(true, axis=1)


def parse_args():
    p = argparse.ArgumentParser()

    # Run
    p.add_argument("--run_name",      type=str,   default=None)
    p.add_argument("--results_dir",   type=str,   default=str(_PROJECT_ROOT / "TEMPO_results" / "burgers"))

    # Data
    p.add_argument("--nu_values",     type=float, nargs="+", default=[0.001, 0.1, 1.0])
    p.add_argument("--n_samples",     type=int,   default=9500)
    p.add_argument("--n_test_per_nu", type=int,   default=1000)
    p.add_argument("--data_dir",      type=str,
                   default=os.path.expanduser("~/data/1D/Burgers/Train"))

    # EM (TEMPOConfig)
    p.add_argument("--M",             type=int,   default=3,    help="Number of regimes")
    p.add_argument("--P_global",      type=int,   default=25,   help="Global POD modes for GMM init")
    p.add_argument("--sigma2",        type=float, default=0.1)
    p.add_argument("--max_em_iters",  type=int,   default=30)
    p.add_argument("--eps_skip",      type=float, default=0.01)
    p.add_argument("--eps_large",     type=float, default=0.1)
    p.add_argument("--eps_conv",      type=float, default=0.005)
    p.add_argument("--heteroscedastic", action="store_true",
                   help="Use relative error in EM E-step (robust for multi-scale data)")

    # Regime basis
    p.add_argument("--basis_type",    type=str,   default="pod",
                   choices=["pod", "fourier"],    help="Basis type per regime")
    p.add_argument("--basis_max_modes", type=int, default=32,
                   help="Max modes per regime. Overridden by --total_modes if set.")
    p.add_argument("--total_modes",   type=int,   default=None,
                   help="If set, basis_max_modes = total_modes // M (fair comparison)")
    # Fourier basis extra args (used only when basis_type=fourier)
    p.add_argument("--n_epochs_mean",        type=int,   default=165)
    p.add_argument("--n_epochs_mode",        type=int,   default=50)
    p.add_argument("--fourier_epoch_subset", type=float, default=None,
                   help="Fraction of N to use per mode epoch. Default: 1/M")

    # Online phase (TEMPOOnlineConfig)
    p.add_argument("--online_lr",         type=float, default=3e-4)
    p.add_argument("--online_epochs",     type=int,   default=170)
    p.add_argument("--online_batch",      type=int,   default=32)
    p.add_argument("--online_hidden_dim", type=int,   default=128)
    p.add_argument("--online_n_layers",   type=int,   default=4)
    p.add_argument("--sensor_stride",     type=int,   default=1)
    p.add_argument("--lambda_kl",         type=float, default=0.1)
    p.add_argument("--lambda_ent",        type=float, default=0.1)
    p.add_argument("--log_every",         type=int,   default=20)

    # Misc
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--n_viz",         type=int,   default=3)
    p.add_argument("--n_umap",        type=int,   default=5000)
    p.add_argument("--skip_umap",     action="store_true",
                   help="Skip UMAP visualization (faster, no umap-learn needed)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.total_modes is not None:
        args.basis_max_modes = max(1, args.total_modes // args.M)
        print(f"fair mode: total_modes={args.total_modes}, basis_max_modes={args.basis_max_modes} per regime")

    nu_tag = "_".join(str(nu) for nu in args.nu_values)
    RUN_NAME = args.run_name or f"tempo_{args.basis_type}_M{args.M}_v1"
    RUN_DIR  = os.path.join(args.results_dir, RUN_NAME)
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

    REGIME_COLORS = [plt.cm.tab10(i) for i in range(10)]
    regime_colors = REGIME_COLORS[:args.M]
    _D2 = plt.cm.Dark2

    # --- Data loading ---
    entries = [
        (nu, os.path.join(args.data_dir, f"1D_Burgers_Sols_Nu{nu}.hdf5"))
        for nu in args.nu_values
    ]
    s_np, kappa_np, x_np, t_np, Nx, Nt = load_stacked(entries, n_samples=args.n_samples)
    Ny = Nt * Nx

    x_grid = torch.tensor(x_np, dtype=torch.float32)
    t_grid = torch.tensor(t_np, dtype=torch.float32)
    tt, xx = torch.meshgrid(t_grid, x_grid, indexing="ij")
    x_flat = torch.stack([xx.flatten(), tt.flatten()], dim=1)

    s     = torch.from_numpy(s_np);           del s_np
    kappa = torch.from_numpy(kappa_np[:, None]); del kappa_np
    x     = x_flat.to(DEVICE)

    print(f"s={s.shape}, x={x.shape}, kappa={kappa.shape}")

    # --- Phase 1: TEMPO EM ---
    print("=== Phase 1: TEMPO EM ===")

    if args.basis_type == "pod":
        basis_cfg     = PODConfig(max_modes=args.basis_max_modes)
        basis_factory = pod_factory
    else:
        fourier_epoch_subset = args.fourier_epoch_subset if args.fourier_epoch_subset is not None else 1.0 / args.M
        basis_cfg = FourierNeuralPODConfig(
            max_modes=args.basis_max_modes,
            n_epochs_mean=args.n_epochs_mean,
            n_epochs_mode=args.n_epochs_mode,
            epoch_subset=fourier_epoch_subset,
        )
        basis_factory = fourier_pod_factory

    cfg = TEMPOConfig(
        M=args.M,
        P_global=args.P_global,
        sigma2=args.sigma2,
        max_em_iters=args.max_em_iters,
        eps_skip=args.eps_skip,
        eps_large=args.eps_large,
        eps_conv=args.eps_conv,
        heteroscedastic=args.heteroscedastic,
        basis_config=basis_cfg,
        basis_factory=basis_factory,
    )

    # Split before EM so basis is fitted on training data only
    N_per_nu  = args.n_samples
    nu_list   = args.nu_values
    train_idx = torch.cat([
        torch.arange(i * N_per_nu, (i + 1) * N_per_nu - args.n_test_per_nu)
        for i in range(len(nu_list))
    ])
    test_idx = torch.cat([
        torch.arange((i + 1) * N_per_nu - args.n_test_per_nu, (i + 1) * N_per_nu)
        for i in range(len(nu_list))
    ])

    trainer = TEMPOTrainer(cfg)
    trainer.train(s[train_idx], x, t=None, kappa=kappa[train_idx])

    # EM convergence plot
    log = trainer.history_phase1
    if log:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(log["iter"], log["ll"], "o-", color=plt.cm.tab10(0), lw=1.5, markersize=4)
        axes[0].set_xlabel("EM iteration"); axes[0].set_ylabel("Log-likelihood")
        axes[0].set_title("EM: log-likelihood", fontweight="bold")

        axes[1].plot(log["iter"], log["entropy"], "o-", color=plt.cm.tab10(1), lw=1.5, markersize=4)
        axes[1].set_xlabel("EM iteration"); axes[1].set_ylabel("Entropy")
        axes[1].set_title("EM: assignment entropy", fontweight="bold")

        for m in range(args.M):
            delta_m = [d[m] for d in log["delta"]]
            axes[2].plot(log["iter"], delta_m, "o-", color=regime_colors[m],
                         lw=1.5, markersize=4, label=f"Regime {m+1}")
        axes[2].axhline(args.eps_conv, color="gray", ls="--", lw=1.2, label="eps_conv")
        axes[2].set_xlabel("EM iteration"); axes[2].set_ylabel("Delta")
        axes[2].set_title("EM: distribution shift per regime", fontweight="bold")
        axes[2].legend(fontsize=8, framealpha=0.7)

        for ax in axes:
            ax.grid(True, ls="--", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "em_convergence.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # --- Phase 2: Online gated operator ---
    print("=== Phase 2: Online gated operator ===")

    u0 = s[:, :Nx]
    s_train     = s[train_idx];         s_test     = s[test_idx]
    u0_train    = u0[train_idx];        u0_test    = u0[test_idx]
    kappa_train = kappa[train_idx];     kappa_test = kappa[test_idx]
    gamma_train = trainer.gamma  # trainer was fitted on train_idx only

    print(f"train: {s_train.shape}, test: {s_test.shape}")
    for m, t in enumerate(trainer.trainers):
        print(f"  Regime {m+1}: P={_num_modes(t)} modes")

    cfg_online = TEMPOOnlineConfig(
        lr=args.online_lr,
        n_epochs=args.online_epochs,
        batch_size=args.online_batch,
        hidden_dim=args.online_hidden_dim,
        n_layers=args.online_n_layers,
        sensor_stride=args.sensor_stride,
        lambda_kl=args.lambda_kl,
        lambda_ent=args.lambda_ent,
        log_every=args.log_every,
    )

    model_online, online_trainer = build_tempo_online(
        trainers=trainer.trainers,
        d_kappa=kappa.shape[1],
        Nx=Nx,
        cfg=cfg_online,
    )
    model_online = model_online.to(DEVICE)

    history = online_trainer.train(
        s=s_train, u0=u0_train, kappa=kappa_train,
        x_flat=x, gamma_star=gamma_train,
        trainers=trainer.trainers,
        val_s=s_test, val_u0=u0_test, val_kappa=kappa_test,
    )

    # Training dynamics plot
    epochs = range(len(history["total"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    c0, c1 = plt.cm.tab10(0), plt.cm.tab10(1)

    axes[0].plot(epochs, history["total"], color=c0, label="total", lw=2)
    axes[0].plot(epochs, history["data"],  color=c1, label="data",  lw=1.5, ls="--")
    if online_trainer.val_history:
        ve, vl = zip(*online_trainer.val_history)
        axes[0].plot(ve, vl, color=plt.cm.tab10(4), lw=1.5, ls=":", label="val data")
    axes[0].set_title("Phase 2: branch network", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(framealpha=0.7); axes[0].set_yscale("symlog")

    axes[1].plot(epochs, history["kl"],  color=plt.cm.tab10(2), label="KL",  lw=1.5)
    axes[1].plot(epochs, history["ent"], color=plt.cm.tab10(3), label="ent", lw=1.5, ls="--")
    axes[1].set_title("Regularisation terms", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(framealpha=0.7)

    for ax in axes:
        ax.grid(True, ls="--", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "training_dynamics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Evaluation ---
    s_pred, w_pred = online_trainer.predict(
        u0_new=u0_test, kappa_new=kappa_test,
        x_flat=x, trainers=trainer.trainers,
    )
    s_pred_np  = s_pred.cpu().numpy()
    s_test_np  = s_test.numpy()
    w_np       = w_pred.cpu().numpy()
    kappa_t_np = kappa_test[:, 0].numpy()
    nu_unique_t = np.unique(kappa_t_np)

    rel_l2 = rel_l2_vec(s_test_np, s_pred_np)
    print("Mean rel L2 error:")
    for nu in nu_unique_t:
        mask = kappa_t_np == nu
        print(f"  nu={nu:.3f}: {rel_l2[mask].mean():.4f} +/- {rel_l2[mask].std():.4f}")

    # Gating + error plots
    nu_to_col_t = {nu: _D2(i / 7) for i, nu in enumerate(nu_unique_t)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    x_pos = np.arange(len(nu_unique_t))
    width = 0.8 / args.M
    for m in range(args.M):
        w_m = [w_np[kappa_t_np == nu, m].mean() for nu in nu_unique_t]
        axes[0].bar(x_pos + m * width, w_m, width,
                    color=regime_colors[m], label=f"Regime {m+1}", alpha=0.85, linewidth=0)
    axes[0].set_xticks(x_pos + width * (args.M - 1) / 2)
    axes[0].set_xticklabels([f"nu={nu:.3f}" for nu in nu_unique_t])
    axes[0].set_ylabel("Mean gating weight")
    axes[0].set_title("Gating weights per viscosity", fontweight="bold")
    axes[0].legend(fontsize=9, framealpha=0.7); axes[0].set_ylim(0, 1)

    bp_data = [rel_l2[kappa_t_np == nu] for nu in nu_unique_t]
    bp = axes[1].boxplot(bp_data, patch_artist=True, widths=0.45,
                         medianprops=dict(color="black", lw=1.5),
                         whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                         flierprops=dict(marker="o", markersize=3, alpha=0.4, linestyle="none"))
    for patch, nu in zip(bp["boxes"], nu_unique_t):
        patch.set_facecolor(nu_to_col_t[nu]); patch.set_alpha(0.8); patch.set_linewidth(0)
    axes[1].set_xticklabels([f"nu={nu:.3f}" for nu in nu_unique_t])
    axes[1].set_ylabel("Relative L2 error")
    axes[1].set_title("Reconstruction error by viscosity", fontweight="bold")
    axes[1].set_ylim(0, 1)

    for ax in axes:
        ax.grid(True, ls="--", alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle("TEMPO Phase 2 - test set results", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "TEMPO_phase2.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Reconstruction examples
    n_rows = len(nu_unique_t)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = ["Ground Truth", "Prediction", "Absolute Error"]
    for row, nu in enumerate(nu_unique_t):
        idx_nu  = np.where(kappa_t_np == nu)[0][0]
        s_true  = s_test_np[idx_nu].reshape(Nt, Nx)
        s_hat   = s_pred_np[idx_nu].reshape(Nt, Nx)
        err     = np.abs(s_true - s_hat)
        rl2     = float(np.linalg.norm(s_true - s_hat) / np.linalg.norm(s_true))
        vmax    = np.abs(s_true).max()
        kw = dict(aspect="auto", origin="lower",
                  extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()])
        label = f"$\\nu$={nu:.3f},  rel L2={rl2:.3f}"
        for col, (arr, cmap, vmin, vm) in enumerate([
            (s_true, "RdBu_r", -vmax, vmax),
            (s_hat,  "RdBu_r", -vmax, vmax),
            (err,    "Oranges",    0, err.max()),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, vmin=vmin, vmax=vm, cmap=cmap, **kw)
            if row == 0:
                ax.set_title(col_titles[col], fontweight="bold")
            if col == 0:
                ax.set_ylabel("t", fontsize=9)
                ax.text(0.02, 0.98, label, transform=ax.transAxes,
                        ha="left", va="top", fontsize=8, fontweight="bold",
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
            if row == n_rows - 1:
                ax.set_xlabel("x")
            ax.spines[["top", "right"]].set_visible(False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"TEMPO({args.basis_type.upper()}): reconstruction examples",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "reconstruct.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Metrics ---
    n_params = sum(p.numel() for p in model_online.parameters())
    n_params_basis = 0
    if args.basis_type == "fourier":
        for t in trainer.trainers:
            n_params_basis += sum(p.numel() for p in t.basis.parameters())
    metrics = {
        "run_name":        RUN_NAME,
        "M":               int(args.M),
        "basis_type":      args.basis_type,
        "n_params":        n_params,
        "n_params_basis":  n_params_basis,
        "n_train":        int(len(s_train)),
        "n_test":         int(len(s_test)),
        "overall_mean":   float(rel_l2.mean()),
        "overall_median": float(np.median(rel_l2)),
    }
    for nu in nu_unique_t:
        mask = kappa_t_np == nu
        metrics[f"nu{nu:.3f}_mean"]   = float(rel_l2[mask].mean())
        metrics[f"nu{nu:.3f}_median"] = float(np.median(rel_l2[mask]))
        metrics[f"nu{nu:.3f}_std"]    = float(rel_l2[mask].std())

    metrics["phase1_log"] = trainer.history_phase1
    metrics["phase2_log"] = history

    # --- Checkpoint ---
    torch.save({
        "model_online": model_online.state_dict(),
        "cfg":          cfg,
        "cfg_online":   cfg_online,
        "metrics":      metrics,
        "run_name":     RUN_NAME,
    }, os.path.join(RUN_DIR, "model_online.pt"))

    torch.save(trainer, os.path.join(RUN_DIR, "trainer.pt"))

    # Inference time
    _inf_ms = measure_inference_time(
        lambda: online_trainer.predict(
            u0_new=u0_test, kappa_new=kappa_test,
            x_flat=x, trainers=trainer.trainers,
        ),
        device=DEVICE
    )
    metrics["inference_ms_total"] = _inf_ms
    metrics["inference_ms_per_sample"] = _inf_ms / len(u0_test)

    metrics["hparams"] = vars(args)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved to {os.path.abspath(RUN_DIR)}")

    # UMAP visualization (after all results saved)
    if not args.skip_umap:
        try:
            from utils.plotting import plot_umap_regimes
            rng_umap = np.random.default_rng(args.seed)
            idx_umap = rng_umap.choice(len(train_idx), min(args.n_umap, len(train_idx)), replace=False)
            plot_umap_regimes(
                alpha=trainer.alpha[idx_umap].cpu().numpy(),
                mu=trainer.mu.cpu().numpy(),
                Sigma=trainer.Sigma.cpu().numpy(),
                hard_labels=trainer.gamma[idx_umap].argmax(dim=1).cpu().numpy(),
                param_vals=kappa[train_idx][idx_umap, 0].cpu().numpy(),
                regime_colors=regime_colors,
                param_label="nu",
                title="Burgers trajectories - TEMPO EM regime structure",
                save_path=os.path.join(RUN_DIR, "TEMPO_phase1.png"),
                seed=args.seed,
            )
        except ImportError:
            print("umap-learn not installed, skipping UMAP plot")
        except Exception as e:
            print(f"UMAP failed: {e}")


if __name__ == "__main__":
    main()
