#!/usr/bin/env python3
"""
Generate 2D incompressible Navier-Stokes data using pseudo-spectral solver.

Vorticity-streamfunction formulation with Kolmogorov forcing:
  dω/dt + u·∇ω = (1/Re)∇²ω + f
  f(x,y) = 0.1 * (sin(2π*(x+y)) + cos(2π*(x+y)))   [Kolmogorov-like]

Method: Crank-Nicolson diffusion + Adams-Bashforth advection, dealiased with 2/3 rule.

Grid: 64x64, domain [0,1]x[0,1] periodic
T_burn: 5.0 (spin-up before saving)
T_save: 20 timesteps at dt=0.1

Usage:
    python generate_navier_stokes_spectral.py
"""

import os
import sys
import pathlib
import numpy as np
from datetime import datetime

REYNOLDS_CONFIGS = {
    100:   {"nu": 1e-2, "n_samples": 10000, "T_burn": 5.0},
    1000:  {"nu": 1e-3, "n_samples": 10000, "T_burn": 5.0},
    3600:  {"nu": 2.78e-4, "n_samples": 10000, "T_burn": 5.0},
    10000: {"nu": 1e-4, "n_samples": 10000, "T_burn": 5.0},
}

N_TIMESTEPS = 20   # saved timesteps after burn-in
DT_SAVE     = 0.1  # physical time between saved steps
N_INNER     = 10   # sub-steps per saved step (dt = DT_SAVE / N_INNER)
N_GRID      = 64

OUTPUT_DIR = pathlib.Path.home() / "data" / "2D" / "Navier_Stokes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- spectral helpers ----------

def _build_wavenumbers(N):
    k = np.fft.fftfreq(N, d=1.0 / N).astype(np.float64)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0   # avoid division by zero for k=0 mode
    return kx, ky, k2


def _dealias_mask(N):
    k = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    kmax = N // 3
    return (np.abs(kx) <= kmax) & (np.abs(ky) <= kmax)


def _forcing(N):
    x = np.linspace(0, 1, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = 0.1 * (np.sin(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X + Y)))
    return np.fft.rfft2(f)


def _omega_to_vel(omega_hat, kx, ky, k2):
    """Return (u, v) in physical space from vorticity Fourier coefficients."""
    # ψ = ω / (-k²),  u = ∂ψ/∂y,  v = -∂ψ/∂x
    psi_hat = -omega_hat / k2
    u = np.fft.irfft2(1j * ky[:, :omega_hat.shape[1]] * psi_hat)
    v = np.fft.irfft2(-1j * kx[:, :omega_hat.shape[1]] * psi_hat)
    return u, v


def _nonlinear(omega_hat, kx, ky, k2, mask):
    """Compute -u·∇ω in Fourier space (dealiased)."""
    Nk = omega_hat.shape[1]
    u, v = _omega_to_vel(omega_hat, kx[:, :Nk], ky[:, :Nk], k2[:, :Nk])
    omega = np.fft.irfft2(omega_hat)
    dw_dx = np.fft.irfft2(1j * kx[:, :Nk] * omega_hat)
    dw_dy = np.fft.irfft2(1j * ky[:, :Nk] * omega_hat)
    nl = -(u * dw_dx + v * dw_dy)
    nl_hat = np.fft.rfft2(nl)
    nl_hat *= mask[:, :Nk]
    return nl_hat


def simulate_one(omega0, nu, dt, n_steps_burn, n_steps_save, n_inner, N):
    """
    Run NS from initial vorticity omega0.
    Returns: trajectory (n_steps_save+1, N, N, 2) or None if NaN detected.
    """
    kx, ky, k2 = _build_wavenumbers(N)
    mask = _dealias_mask(N)
    f_hat = _forcing(N)
    Nk = N // 2 + 1
    k2r = k2[:, :Nk]
    kxr = kx[:, :Nk]
    kyr = ky[:, :Nk]

    # Crank-Nicolson diffusion factor
    cn_denom = 1.0 + 0.5 * nu * dt * k2r
    cn_num   = 1.0 - 0.5 * nu * dt * k2r

    omega_hat = np.fft.rfft2(omega0)
    omega_hat *= mask[:, :Nk]

    nl_prev = None

    def step(omega_hat, nl_prev):
        nl = _nonlinear(omega_hat, kx, ky, k2, mask)
        if nl_prev is None:
            nl_prev = nl
        # Adams-Bashforth 2nd order for advection
        rhs = (cn_num * omega_hat
               + dt * (1.5 * nl - 0.5 * nl_prev)
               + dt * f_hat[:, :Nk])
        omega_hat_new = rhs / cn_denom
        omega_hat_new[0, 0] = 0.0  # zero mean
        return omega_hat_new, nl

    # Burn-in
    for _ in range(n_steps_burn):
        omega_hat, nl_prev = step(omega_hat, nl_prev)
        if not np.isfinite(omega_hat).all():
            return None

    # Save trajectory
    traj = np.zeros((n_steps_save + 1, N, N, 2), dtype=np.float32)
    u, v = _omega_to_vel(omega_hat, kxr, kyr, k2r)
    traj[0, :, :, 0] = u.astype(np.float32)
    traj[0, :, :, 1] = v.astype(np.float32)

    for t in range(n_steps_save):
        for _ in range(n_inner):
            omega_hat, nl_prev = step(omega_hat, nl_prev)
        if not np.isfinite(omega_hat).all():
            return None
        u, v = _omega_to_vel(omega_hat, kxr, kyr, k2r)
        traj[t + 1, :, :, 0] = u.astype(np.float32)
        traj[t + 1, :, :, 1] = v.astype(np.float32)

    return traj


def generate_ns_data(re, nu, n_samples, T_burn):
    N = N_GRID
    dt = DT_SAVE / N_INNER
    n_steps_burn = int(T_burn / dt)
    n_steps_save = N_TIMESTEPS

    filename = OUTPUT_DIR / f"2D_NavierStokes_Incomp_Re{re:05d}.npz"

    print(f"\n[Re={re}] nu={nu:.2e}  samples={n_samples}  burn={T_burn}s  save={n_steps_save}×{DT_SAVE}s")

    all_data = np.zeros((n_samples, n_steps_save + 1, N, N, 2), dtype=np.float32)

    x = np.linspace(0, 1, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")

    n_retries = 0
    for i in range(n_samples):
        traj = None
        attempt = 0
        while traj is None:
            rng = np.random.default_rng(seed=i + re * 100000 + attempt * 99999983)
            omega0 = np.zeros((N, N))
            for k in range(1, 8):
                amp = rng.standard_normal() / (1 + k)
                phi = rng.uniform(0, 2 * np.pi)
                omega0 += amp * np.sin(2 * np.pi * k * X + phi)
                omega0 += amp * np.cos(2 * np.pi * k * Y + phi)
            traj = simulate_one(omega0, nu, dt,
                                n_steps_burn=n_steps_burn,
                                n_steps_save=n_steps_save,
                                n_inner=N_INNER, N=N)
            if traj is None:
                n_retries += 1
                attempt += 1
        all_data[i] = traj

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_samples}  (retries so far: {n_retries})")

    np.savez_compressed(filename,
        velocity=all_data,
        reynolds_number=re,
        kinematic_viscosity=nu,
        n_samples=n_samples,
        time_steps=n_steps_save,
        dt=DT_SAVE,
        grid_size=N,
        generated_at=str(datetime.now().isoformat())
    )
    size_mb = filename.stat().st_size / (1024**2)
    print(f"  Saved: {filename.name} ({size_mb:.0f} MB)")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=int, nargs="*", default=None,
                        help="Reynolds numbers to generate (default: all)")
    args = parser.parse_args()

    target_res = args.re if args.re else sorted(REYNOLDS_CONFIGS.keys())

    print("=" * 70)
    print("2D Navier-Stokes Pseudo-Spectral Generator")
    print("=" * 70)
    print(f"Grid: {N_GRID}x{N_GRID}  T_save={N_TIMESTEPS}×{DT_SAVE}s")
    print(f"Generating Re: {target_res}")
    print(f"Output: {OUTPUT_DIR}\n")

    success = 0
    for re in target_res:
        if re not in REYNOLDS_CONFIGS:
            print(f"  WARNING: Re={re} not in REYNOLDS_CONFIGS, skipping")
            continue
        cfg = REYNOLDS_CONFIGS[re]
        try:
            generate_ns_data(re, cfg["nu"], cfg["n_samples"], cfg["T_burn"])
            success += 1
        except Exception as e:
            import traceback
            print(f"  ERROR Re={re}: {e}")
            traceback.print_exc()

    print(f"\nDone: {success}/{len(target_res)} Reynolds numbers generated")
    print(f"Files in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.glob("*.npz")):
        print(f"  {f.name}  ({f.stat().st_size/1e6:.0f} MB)")
    return 0 if success == len(target_res) else 1


if __name__ == "__main__":
    sys.exit(main())
