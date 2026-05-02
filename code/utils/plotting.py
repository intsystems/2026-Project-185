"""Unified plotting utilities."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

try:
    import seaborn as sns
    PALETTE = sns.color_palette("Set2")
except ImportError:
    PALETTE = [plt.cm.tab10(i) for i in range(10)]

COLOR_MEAN     = PALETTE[2]
COLOR_RESIDUAL = PALETTE[1]
C0, C1, C2     = plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2)


def plot_train_history(trainer) -> plt.Figure:
    """Visualize loss and residual evolution during training.

    Args:
        trainer: any trainer with history attribute

    Returns:
        matplotlib figure with 4 subplots
    """
    h = trainer.history
    cfg = trainer.cfg
    n_modes = len(h.mode_losses)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Mean network loss
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(h.mean_loss, color=COLOR_MEAN, lw=2)
    ax0.set_title("Mean network loss", fontweight="bold")
    ax0.set_xlabel(f"Epoch (x{cfg.log_every})")
    ax0.set_ylabel("MSE")
    ax0.grid(True, ls="--", alpha=0.4)

    # Mode losses
    ax1 = fig.add_subplot(gs[0, 1])
    for p, losses in enumerate(h.mode_losses):
        ax1.plot(losses, color=PALETTE[p % len(PALETTE)], lw=1.5, label=f"mode {p + 1}")
    ax1.set_title("Mode losses", fontweight="bold")
    ax1.set_xlabel(f"Epoch (x{cfg.log_every})")
    ax1.set_ylabel("MSE")
    if n_modes:
        ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, ls="--", alpha=0.4)

    # Residual MSE per mode
    ax2 = fig.add_subplot(gs[1, 0])
    if h.residual_norms:
        idx = list(range(1, len(h.residual_norms) + 1))
        ax2.semilogy(idx, h.residual_norms, "o-", color=COLOR_RESIDUAL, lw=2, ms=8)
        ax2.set_xticks(idx)
        ax2.set_title("Residual MSE per mode", fontweight="bold")
        ax2.set_xlabel("Mode index")
        ax2.set_ylabel("MSE")
    ax2.grid(True, ls="--", alpha=0.4)

    # Residual drop per mode
    ax3 = fig.add_subplot(gs[1, 1])
    if h.residual_norms:
        norms = h.residual_norms
        drops = [norms[0]] + [norms[i - 1] - norms[i] for i in range(1, len(norms))]
        ax3.bar(
            range(1, len(drops) + 1),
            drops,
            color=[PALETTE[p % len(PALETTE)] for p in range(len(drops))],
            edgecolor="white",
            linewidth=0.5,
        )
        ax3.set_xticks(range(1, len(drops) + 1))
        ax3.set_title("Residual drop per mode", fontweight="bold")
        ax3.set_xlabel("Mode index")
        ax3.set_ylabel("MSE reduction")
    ax3.grid(True, ls="--", alpha=0.4, axis="y")

    return fig


def plot_training_loss(trainer) -> plt.Figure:
    """Plot mean loss decay on log scale."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(
        trainer.history.mean_loss,
        "o-",
        linewidth=2,
        markersize=6,
        color=COLOR_MEAN,
    )
    ax.set_xlabel("Logging Step", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    return fig


def plot_space_time_heatmaps(obj, s_true: torch.Tensor,
                              x: torch.Tensor, t: torch.Tensor, title: str = "NeuralPOD") -> plt.Figure:
    """Plot truth, prediction, and error in space-time.

    Args:
        obj: basis, trainer, or predictor object
        s_true: (N, Ny) snapshot matrix
        x: (Ny, d_x) spatial grid
        t: (N,) time vector
        title: plot title

    Returns:
        matplotlib figure with 3 subplots
    """
    with torch.no_grad():
        if hasattr(obj, 'predict'):
            s_pred = obj.predict(x, t)
        elif hasattr(obj, 'basis'):
            obj.basis.eval()
            s_pred = obj.basis(x, t, None)
        else:
            obj.eval()
            batch_size = len(t)
            kappa_dummy = torch.ones(batch_size, 1, device=x.device)
            s_pred = obj(x, t, kappa_dummy)

    s_true_np = s_true.detach().cpu().numpy()
    s_pred_np = s_pred.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy().squeeze()
    t_np = t.detach().cpu().numpy()
    error_np = np.abs(s_true_np - s_pred_np)

    vmax = np.abs(s_true_np).max()
    emax = error_np.max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(
        s_true_np, aspect='auto', origin='lower', cmap='RdBu_r',
        extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
        vmin=-vmax, vmax=vmax
    )
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('t', fontsize=11)
    axes[0].set_title(r'Ground Truth $u(x,t)$', fontsize=12, fontweight='bold')
    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.set_label('u', fontsize=10)

    im1 = axes[1].imshow(
        s_pred_np, aspect='auto', origin='lower', cmap='RdBu_r',
        extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
        vmin=-vmax, vmax=vmax
    )
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('t', fontsize=11)
    axes[1].set_title(rf'{title} Prediction $\hat{{u}}(x,t)$', fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.set_label(r'$\hat{u}$', fontsize=10)

    im2 = axes[2].imshow(
        error_np, aspect='auto', origin='lower', cmap='hot',
        extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
        vmin=0, vmax=emax
    )
    axes[2].set_xlabel('x', fontsize=11)
    axes[2].set_ylabel('t', fontsize=11)
    axes[2].set_title(r'Absolute Error $|u - \hat{u}|$', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[2])
    cbar2.set_label('error', fontsize=10)

    rel_err = np.linalg.norm(s_true_np - s_pred_np) / np.linalg.norm(s_true_np)
    mean_error = error_np.mean()

    fig.suptitle(
        rf'{title} Reconstruction | Relative L2 Error: {rel_err:.4f} | Mean Error: {mean_error:.4f}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    return fig


def plot_pod_phase1(sigmas: np.ndarray, save_path: str) -> None:
    energy = sigmas ** 2
    cumvar = np.cumsum(energy) / energy.sum() * 100
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.semilogy(range(1, len(sigmas) + 1), sigmas, "o-", color=C0, markersize=4, lw=1.5)
    ax.set_xlabel("Mode index"); ax.set_ylabel("Singular value")
    ax.set_title("Phase 1: singular value decay", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", color=C0, markersize=4, lw=1.5)
    ax.axhline(99.99, color=C1, ls="--", lw=1.2, label="99.99%")
    ax.set_xlabel("Mode index"); ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Phase 1: cumulative variance explained", fontweight="bold")
    ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_error_dist(errors: np.ndarray, title: str, save_path: str) -> None:
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(errors, bins=30, color=C0, alpha=0.8, linewidth=0)
    ax.axvline(errors.mean(),     color=C1, ls="--", lw=1.5, label=f"Mean {errors.mean():.4f}")
    ax.axvline(np.median(errors), color=C2, ls="--", lw=1.5, label=f"Median {np.median(errors):.4f}")
    ax.set_xlabel("Relative L2 error"); ax.set_ylabel("Count")
    ax.set_title("Test error distribution", fontweight="bold")
    ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(np.sort(errors), color=C0, lw=1.5)
    ax.axhline(errors.mean(), color=C1, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Sample rank"); ax.set_ylabel("Relative L2 error")
    ax.set_title("Sorted test errors", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_cross_param_bar(
    cross_metrics: dict,
    trained_val,
    param_label: str,
    title: str,
    save_path: str,
    ylim: float = 1.0,
) -> None:
    """Bar chart of mean/median cross-parameter errors."""
    keys    = list(cross_metrics.keys())
    labels  = [f"{param_label}={k:.4g}" if isinstance(k, float) else f"{param_label}={k}" for k in keys]
    means   = [cross_metrics[k]["mean"]   for k in keys]
    medians = [cross_metrics[k]["median"] for k in keys]
    x_pos   = np.arange(len(labels))
    _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_pos - 0.2,  means,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
    ax.bar(x_pos + 0.15, medians, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
    if trained_val in keys:
        ax.axvline(keys.index(trained_val), color="gray", ls="--", lw=1.2, alpha=0.7, label="Trained on")
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Relative L2 error")
    ax.set_title(title, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_reconstruction_xt(
    true_list: list,
    pred_list: list,
    rl2_list: list,
    x_np: np.ndarray,
    t_np: np.ndarray,
    model_name: str,
    save_path: str,
) -> None:
    """Space-time reconstruction panels for 1D time-dependent PDEs."""
    n = len(true_list)
    _, axes = plt.subplots(n, 3, figsize=(14, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for row, (true, pred, rl2) in enumerate(zip(true_list, pred_list, rl2_list)):
        err  = np.abs(true - pred)
        vmax = np.abs(true).max()
        for col, (arr, ttl, cmap, vmin, vm) in enumerate([
            (true, "Ground Truth",   "RdBu_r",  -vmax, vmax),
            (pred, model_name,       "RdBu_r",  -vmax, vmax),
            (err,  "Absolute Error", "Oranges",  0,    err.max()),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap,
                           extent=[x_np.min(), x_np.max(), t_np.min(), t_np.max()],
                           vmin=vmin, vmax=vm)
            if row == 0: ax.set_title(ttl, fontweight="bold")
            if col == 0: ax.set_ylabel(f"t   (rel L2={rl2:.3f})")
            if row == n - 1: ax.set_xlabel("x")
            ax.spines[["top", "right"]].set_visible(False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"{model_name}: reconstruction examples", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_umap_regimes(
    alpha: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    hard_labels: np.ndarray,
    param_vals: np.ndarray,
    regime_colors: list,
    param_label: str,
    title: str,
    save_path: str,
    seed: int = 0,
) -> None:
    """3-panel UMAP: regime assignment, param coloring, POD coefficient space.

    Args:
        alpha:        (N, P) POD coefficients
        mu:           (M, P) regime centroids
        Sigma:        (M, P, P) regime covariances
        hard_labels:  (N,) integer regime assignments
        param_vals:   (N,) scalar parameter per sample (nu, beta, ...)
        regime_colors: list of M colors
        param_label:  axis/legend label for the parameter
        title:        figure suptitle
        save_path:    output file path
    """
    from umap import UMAP
    from matplotlib.patches import Ellipse

    def _cov_ellipse(ax, mean2d, cov2d, color, n_std=2.0, **kw):
        vals, vecs = np.linalg.eigh(cov2d)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * n_std * np.sqrt(np.abs(vals))
        ax.add_patch(Ellipse(mean2d, w, h, angle=angle,
                             edgecolor=color, facecolor="none", lw=2, **kw))

    reducer   = UMAP(n_neighbors=30, min_dist=0.0, random_state=seed)
    embedding = reducer.fit_transform(alpha)
    mu_umap   = reducer.transform(mu)

    M           = mu.shape[0]
    param_unique = np.unique(param_vals)
    PCMAP        = plt.cm.plasma
    param_to_col = {v: PCMAP(i / max(len(param_unique) - 1, 1))
                    for i, v in enumerate(param_unique)}
    point_colors_reg   = [regime_colors[m] for m in hard_labels]
    point_colors_param = [param_to_col[float(v)] for v in param_vals]
    centroid_kw = dict(s=120, marker="D", zorder=10, edgecolors="white", linewidths=0.8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(embedding[:, 0], embedding[:, 1],
                    c=point_colors_reg, s=3, alpha=0.3, rasterized=True, linewidths=0)
    for m in range(M):
        axes[0].scatter(mu_umap[m, 0], mu_umap[m, 1], color=regime_colors[m], **centroid_kw)
    axes[0].set_title("UMAP — regime assignment", fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=regime_colors[m], markersize=7,
                          label=f"Regime {m+1}") for m in range(M)]
    axes[0].legend(handles=handles, fontsize=9, framealpha=0.7)

    axes[1].scatter(embedding[:, 0], embedding[:, 1],
                    c=point_colors_param, s=3, alpha=0.3, rasterized=True, linewidths=0)
    for m in range(M):
        axes[1].scatter(mu_umap[m, 0], mu_umap[m, 1], color=regime_colors[m], **centroid_kw)
    axes[1].set_title(f"UMAP — colored by {param_label}", fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=param_to_col[v], markersize=7,
                          label=f"{param_label}={v:.4g}") for v in param_unique]
    axes[1].legend(handles=handles, fontsize=9, framealpha=0.7)

    axes[2].scatter(alpha[:, 0], alpha[:, 1],
                    c=point_colors_param, s=3, alpha=0.3, rasterized=True, linewidths=0)
    for m in range(M):
        _cov_ellipse(axes[2], mu[m, :2], Sigma[m, :2, :2], color=regime_colors[m], linestyle="--")
        axes[2].scatter(*mu[m, :2], color=regime_colors[m], **centroid_kw, label=f"Regime {m+1}")
    axes[2].set_title(r"POD space: $\alpha_1$ vs $\alpha_2$", fontweight="bold")
    axes[2].set_xlabel(r"$\alpha_1$"); axes[2].set_ylabel(r"$\alpha_2$")
    axes[2].legend(fontsize=9, framealpha=0.7)

    for ax in axes[:2]:
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    for ax in axes:
        ax.grid(True, ls="--", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_reconstruction_xy(
    true_list: list,
    pred_list: list,
    rl2_list: list,
    x_np: np.ndarray,
    y_np: np.ndarray,
    model_name: str,
    save_path: str,
    row_labels: list = None,
) -> None:
    """Spatial reconstruction panels for 2D steady-state PDEs."""
    n = len(true_list)
    _, axes = plt.subplots(n, 3, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[None, :]
    for row, (true, pred, rl2) in enumerate(zip(true_list, pred_list, rl2_list)):
        err  = np.abs(true - pred)
        vmax = np.abs(true).max()
        label = f"{row_labels[row]},  rel L2={rl2:.3f}" if row_labels else f"rel L2={rl2:.3f}"
        for col, (arr, ttl, cmap, vmin, vm) in enumerate([
            (true, "Ground Truth",   "viridis", 0,    vmax),
            (pred, "Prediction",     "viridis", 0,    vmax),
            (err,  "Absolute Error", "Oranges", 0,    err.max()),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, aspect="equal", origin="lower", cmap=cmap,
                           extent=[x_np.min(), x_np.max(), y_np.min(), y_np.max()],
                           vmin=vmin, vmax=vm)
            if row == 0: ax.set_title(ttl, fontweight="bold")
            if col == 0:
                ax.set_ylabel("y", fontsize=9)
                ax.text(0.02, 0.98, label, transform=ax.transAxes,
                        ha="left", va="top", fontsize=8, fontweight="bold",
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
            if row == n - 1: ax.set_xlabel("x")
            ax.spines[["top", "right"]].set_visible(False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"{model_name}: reconstruction examples", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
