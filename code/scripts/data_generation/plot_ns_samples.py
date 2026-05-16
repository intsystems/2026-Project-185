#!/usr/bin/env python3
"""
Visualise samples from generated NS .npz files.
Saves plot to ns_samples.png (no display needed).

Usage:
    python code/scripts/data_generation/plot_ns_samples.py
    python code/scripts/data_generation/plot_ns_samples.py --re 1000 --n 4
"""
import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = pathlib.Path.home() / "data" / "2D" / "Navier_Stokes"
RE_LIST  = [100, 1000, 3600, 10000]


def load_samples(re, n):
    path = DATA_DIR / f"2D_NavierStokes_Incomp_Re{re:05d}.npz"
    if not path.exists():
        return None, path
    data = np.load(path)
    vel = data["velocity"]          # (N, T+1, 64, 64, 2)
    idx = np.random.default_rng(0).choice(len(vel), size=min(n, len(vel)), replace=False)
    return vel[idx], path


def vorticity(vel):
    """Finite-diff curl of (u, v) -> ω = dv/dx - du/dy."""
    u, v = vel[..., 0], vel[..., 1]
    dvdx = np.gradient(v, axis=-1)
    dudy = np.gradient(u, axis=-2)
    return dvdx - dudy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=int, nargs="*", default=None)
    parser.add_argument("--n",  type=int, default=3, help="samples per Re")
    parser.add_argument("--t",  type=int, default=-1, help="timestep index (-1 = last)")
    parser.add_argument("--out", default="ns_samples.png")
    args = parser.parse_args()

    re_list = args.re or RE_LIST
    re_list = [r for r in re_list if (DATA_DIR / f"2D_NavierStokes_Incomp_Re{r:05d}.npz").exists()]
    if not re_list:
        print(f"No .npz files found in {DATA_DIR}"); return

    n_rows = len(re_list) * args.n
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis]

    row = 0
    for re in re_list:
        samples, path = load_samples(re, args.n)
        if samples is None:
            print(f"  missing {path}"); continue
        print(f"Re={re}: loaded {len(samples)} samples from {path.name}")

        for s in range(len(samples)):
            vel_t = samples[s, args.t]          # (64, 64, 2)
            speed = np.hypot(vel_t[..., 0], vel_t[..., 1])
            omg   = vorticity(vel_t)

            ax0, ax1, ax2 = axes[row]

            im0 = ax0.imshow(vel_t[..., 0], cmap="RdBu_r", origin="lower")
            ax0.set_title(f"Re={re}  u  (s={s})", fontsize=8)
            plt.colorbar(im0, ax=ax0, fraction=0.046)

            im1 = ax1.imshow(vel_t[..., 1], cmap="RdBu_r", origin="lower")
            ax1.set_title(f"Re={re}  v  (s={s})", fontsize=8)
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            im2 = ax2.imshow(omg, cmap="bwr", origin="lower",
                             vmin=-np.percentile(np.abs(omg), 98),
                             vmax= np.percentile(np.abs(omg), 98))
            ax2.set_title(f"Re={re}  ω=dv/dx-du/dy  (s={s})", fontsize=8)
            plt.colorbar(im2, ax=ax2, fraction=0.046)

            for ax in (ax0, ax1, ax2):
                ax.axis("off")
            row += 1

    # stats summary
    print("\n--- Stats (last timestep, all loaded samples) ---")
    for re in re_list:
        path = DATA_DIR / f"2D_NavierStokes_Incomp_Re{re:05d}.npz"
        if not path.exists(): continue
        vel = np.load(path)["velocity"][:, -1]   # (N, 64, 64, 2)
        speed = np.hypot(vel[..., 0], vel[..., 1])
        nan_frac = np.isnan(vel).mean()
        print(f"  Re={re:5d}: |u|_max={vel[...,0].max():.2f}  speed_mean={speed.mean():.3f}"
              f"  NaN={nan_frac:.1%}  shape={vel.shape}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
