"""Utilities for NeuralPOD trainers: plotting and training history."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field


@dataclass
class TrainHistory:
    """Training metrics: mean loss, mode losses, residual norms."""
    mean_loss: list[float] = field(default_factory=list)
    mode_losses: list[list[float]] = field(default_factory=list)
    residual_norms: list[float] = field(default_factory=list)


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
    cmap = plt.cm.tab10

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Mean network loss
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(h.mean_loss, color="steelblue", lw=2)
    ax0.set_title("Mean network loss", fontweight="bold")
    ax0.set_xlabel(f"Epoch (x{cfg.log_every})")
    ax0.set_ylabel("MSE")
    ax0.grid(True, ls="--", alpha=0.4)

    # Mode losses
    ax1 = fig.add_subplot(gs[0, 1])
    for p, losses in enumerate(h.mode_losses):
        ax1.plot(losses, color=cmap(p % 10), lw=1.5, label=f"mode {p + 1}")
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
        ax2.semilogy(idx, h.residual_norms, "o-", color="darkorange", lw=2, ms=8)
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
            color=[cmap(p % 10) for p in range(len(drops))],
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
        color="steelblue",
    )
    ax.set_xlabel("Logging Step", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=14, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    return fig
