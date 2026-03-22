"""Unified plotting utilities with Set2 color palette."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import torch


PALETTE = sns.color_palette("Set2")
COLOR_MEAN = PALETTE[2]
COLOR_RESIDUAL = PALETTE[1]
CMAP = plt.cm.get_cmap("tab10")


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
