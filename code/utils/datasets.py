import os
import argparse
import numpy as np
import h5py

DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)

DARCY_DATASETS = {
    "DarcyFlow_beta0.01":  (0.01,  "2D_DarcyFlow_beta0.01_Train.hdf5"),
    "DarcyFlow_beta0.1":   (0.1,   "2D_DarcyFlow_beta0.1_Train.hdf5"),
    "DarcyFlow_beta1.0":   (1.0,   "2D_DarcyFlow_beta1.0_Train.hdf5"),
    "DarcyFlow_beta10.0":  (10.0,  "2D_DarcyFlow_beta10.0_Train.hdf5"),
    "DarcyFlow_beta100.0": (100.0, "2D_DarcyFlow_beta100.0_Train.hdf5"),
}

HF_REPO = "erbacher/PDEBench-1D"

# (hf_config, split, x_range, t_range, n_channels)
#  n_channels: 1 for scalar fields, 3 for CFD (rho/v/p), 2 for ReacDiff (u/v)
DATASETS = {
    # Burgers: Nx=1024, Nt=201, x in (-1,1), t in (0,2), V=1
    "Burgers_Nu0.001": ("Burgers_Sols_Nu0.001", "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu0.002": ("Burgers_Sols_Nu0.002", "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu0.004": ("Burgers_Sols_Nu0.004", "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu0.01":  ("Burgers_Sols_Nu0.01",  "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu0.02":  ("Burgers_Sols_Nu0.02",  "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu0.04":  ("Burgers_Sols_Nu0.04",  "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu0.1":   ("Burgers_Sols_Nu0.1",   "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu0.2":   ("Burgers_Sols_Nu0.2",   "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu0.4":   ("Burgers_Sols_Nu0.4",   "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu1.0":   ("Burgers_Sols_Nu1.0",   "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu2.0":   ("Burgers_Sols_Nu2.0",   "train", (-1.0, 1.0), (0.0, 2.0), 1),
    "Burgers_Nu4.0":   ("Burgers_Sols_Nu4.0",   "train", (-1.0, 1.0), (0.0, 2.0), 1),
    # Advection: Nx=1024, Nt=201, x in (0,1), t in (0,2), V=1
    "Advection_beta0.1": ("Advection_Sols_beta0.1", "train", (0.0, 1.0), (0.0, 2.0), 1),
    "Advection_beta0.4": ("Advection_Sols_beta0.4", "train", (0.0, 1.0), (0.0, 2.0), 1),
    "Advection_beta1.0": ("Advection_Sols_beta1.0", "train", (0.0, 1.0), (0.0, 2.0), 1),
    "Advection_beta2.0": ("Advection_Sols_beta2.0", "train", (0.0, 1.0), (0.0, 2.0), 1),
    "Advection_beta4.0": ("Advection_Sols_beta4.0", "train", (0.0, 1.0), (0.0, 2.0), 1),
    # ReacDiff: Nx=1024, Nt=101, x in (0,1), t in (0,5), V=2
    "ReacDiff_Nu1.0_Rho1.0": ("ReacDiff_Nu1.0_Rho1.0", "train", (0.0, 1.0), (0.0, 5.0), 2),
    "ReacDiff_Nu1.0_Rho5.0": ("ReacDiff_Nu1.0_Rho5.0", "train", (0.0, 1.0), (0.0, 5.0), 2),
    # CFD: Nx=1024, Nt=101, x in (0,1), t in (0,1), V=3 (rho, v, p)
    "CFD_Eta0.1_Zeta0.1": ("CFD_Rand_Eta0.1_Zeta0.1_periodic", "train", (0.0, 1.0), (0.0, 1.0), 3),
}


def hdf5_path(name: str) -> str:
    return os.path.join(DATA_DIR, f"{name}.hdf5")


def download(name: str) -> None:
    """Download dataset from HuggingFace and save as HDF5."""
    from datasets import load_dataset  # imported lazily — only needed for download

    path = hdf5_path(name)
    if os.path.exists(path):
        print(f"[skip]     {name}")
        return

    hf_config, split, x_range, t_range, n_channels = DATASETS[name]
    print(f"[download] {name}  ({hf_config}/{split})")

    ds = load_dataset(HF_REPO, hf_config, split=split)

    tensor = np.array(ds["tensor"], dtype=np.float32)

    assert tensor.ndim == 4, f"Unexpected tensor ndim={tensor.ndim}, expected 4"
    N, Nt, Nx, V = tensor.shape
    assert V == n_channels, (
        f"Channel mismatch for {name}: got V={V}, expected {n_channels}"
    )

    x = np.linspace(*x_range, Nx, dtype=np.float32)
    t = np.linspace(*t_range, Nt, dtype=np.float32)

    os.makedirs(DATA_DIR, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("tensor", data=tensor)
        f.create_dataset("x-coordinate", data=x)
        f.create_dataset("t-coordinate", data=t)
        f.attrs["n_channels"] = V
        f.attrs["source"]     = hf_config

    print(f"[saved]    {path}  shape={tensor.shape}")


def load(name: str, path: str = None) -> dict[str, np.ndarray]:
    """Load a full HDF5 dataset into memory (small files only)."""
    if path is None:
        if not os.path.exists(hdf5_path(name)):
            download(name)
        path = hdf5_path(name)

    with h5py.File(path, "r") as f:
        return {key: f[key][:] for key in f.keys()}


def load_stacked(
    entries: list[tuple[float, str]],
    n_samples: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Stack trajectory HDF5 files into a single pre-allocated array.

    Reads only n_samples rows per file via HDF5 slicing — no full-file loads.

    Returns: s (N_total, Nt*Nx), kappa (N_total,), x_np, t_np, Nx, Nt
    """
    # probe first file for grid shape
    _, path0 = entries[0]
    with h5py.File(path0, "r") as f:
        shape = f["tensor"].shape           # (N_file, Nt, Nx) or (N_file, Nt, Nx, 1)
        Nt, Nx = shape[1], shape[2]
        t_np   = f["t-coordinate"][:][:Nt]
        x_np   = f["x-coordinate"][:]
        n_file = shape[0]

    n_per   = min(n_samples, n_file) if n_samples is not None else n_file
    n_total = n_per * len(entries)
    Ny      = Nt * Nx

    print(f"Nt={Nt}, Nx={Nx}, n_per_kappa={n_per}, "
          f"N_total={n_total}, s≈{n_total*Ny*4/1e9:.1f} GB")

    # pre-allocate — avoids concatenation peak
    s     = np.empty((n_total, Ny), dtype=np.float32)
    kappa = np.empty(n_total,       dtype=np.float32)

    for i, (kappa_val, path) in enumerate(entries):
        with h5py.File(path, "r") as f:
            u = f["tensor"][:n_per]         # HDF5 reads only n_per rows
        if u.ndim == 4:
            u = u[..., 0]                   # (N, Nt, Nx, 1) -> (N, Nt, Nx)
        sl        = slice(i * n_per, (i + 1) * n_per)
        s[sl]     = u.reshape(n_per, Ny).astype(np.float32, copy=False)
        kappa[sl] = kappa_val
        del u
        print(f"  kappa={kappa_val}: {n_per} trajectories loaded")

    return s, kappa, x_np, t_np, Nx, Nt


def load_darcy_stacked(
    entries: list[tuple[float, str]],
    n_samples: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Stack Darcy Flow HDF5 files into pre-allocated arrays.

    Each file: nu (N, Nx, Ny) permeability input, tensor (N, 1, Nx, Ny) pressure output.

    Returns: s (N_total, Nx*Ny), a (N_total, Nx*Ny), kappa (N_total,), xy (Nx*Ny, 2), Nx, Ny
      s     - output pressure field u(x,y), flattened
      a     - input permeability field a(x,y), flattened
      kappa - beta value per sample
      xy    - grid coordinates (Nx*Ny, 2) for NeuralPOD mode networks
    """
    _, path0 = entries[0]
    with h5py.File(path0, "r") as f:
        shape = f["tensor"].shape          # (N_file, 1, Nx, Ny)
        Nx, Ny = shape[2], shape[3]
        x_np = f["x-coordinate"][:]        # (Nx,)
        y_np = f["y-coordinate"][:]        # (Ny,)
        n_file = shape[0]

    n_per   = min(n_samples, n_file) if n_samples is not None else n_file
    n_total = n_per * len(entries)
    Nxy     = Nx * Ny

    print(f"Darcy: Nx={Nx}, Ny={Ny}, n_per_beta={n_per}, "
          f"N_total={n_total}, s≈{n_total*Nxy*4/1e9:.1f} GB")

    s     = np.empty((n_total, Nxy), dtype=np.float32)
    a     = np.empty((n_total, Nxy), dtype=np.float32)
    kappa = np.empty(n_total,        dtype=np.float32)

    for i, (beta_val, path) in enumerate(entries):
        with h5py.File(path, "r") as f:
            u = f["tensor"][:n_per, 0]     # (n_per, Nx, Ny)
            v = f["nu"][:n_per]            # (n_per, Nx, Ny)
        sl        = slice(i * n_per, (i + 1) * n_per)
        s[sl]     = u.reshape(n_per, Nxy).astype(np.float32, copy=False)
        a[sl]     = v.reshape(n_per, Nxy).astype(np.float32, copy=False)
        kappa[sl] = beta_val
        del u, v
        print(f"  beta={beta_val}: {n_per} samples loaded")

    xx, yy = np.meshgrid(x_np, y_np, indexing="ij")  # (Nx, Ny) each
    xy_np  = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)  # (Nxy, 2)

    return s, a, kappa, xy_np, Nx, Ny


def load_ns_stacked(
    entries: list[tuple[int, str]],
    n_samples: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Stack 2D Navier-Stokes NPZ files into pre-allocated arrays.

    Each file: velocity (N, Nt, Nx, Ny, 2) trajectories with 2 velocity components.

    Returns:
      s     (N_total, Nt*Nx*Ny*2) - full velocity trajectory flattened
      u0    (N_total, Nx*Ny*2) - initial velocity field flattened
      kappa (N_total,) - Reynolds number per sample
      xy    (Nx*Ny, 2) - spatial grid coordinates
      Nx, Ny, Nt - grid and time dimensions
    """
    _MAX_PER_RE = 9500  # hard cap per Reynolds number

    _, path0 = entries[0]
    data0 = np.load(path0)
    vel0 = data0["velocity"]  # (N_file, Nt, Nx, Ny, 2)
    shape = vel0.shape
    Nt, Nx, Ny = shape[1], shape[2], shape[3]
    n_file = shape[0]

    n_want  = min(n_samples, _MAX_PER_RE, n_file) if n_samples is not None else min(_MAX_PER_RE, n_file)
    # Load a few extra rows to ensure n_want clean samples after filtering
    n_load  = min(n_want + 50, n_file)
    n_total = n_want * len(entries)
    Nxy     = Nx * Ny
    Nxyt    = Nt * Nxy

    print(f"NS: Nx={Nx}, Ny={Ny}, Nt={Nt}, n_per_re={n_want}, "
          f"N_total={n_total}, s≈{n_total*Nxyt*2*4/1e9:.1f} GB")

    s     = np.empty((n_total, Nxyt * 2), dtype=np.float32)
    u0    = np.empty((n_total, Nxy * 2), dtype=np.float32)
    kappa = np.empty(n_total, dtype=np.float32)

    pos = 0
    for re_val, path in entries:
        data = np.load(path)
        vel = data["velocity"][:n_load]  # load a buffer

        # Drop NaN / inf / diverged samples (rare spectral-solver instability)
        finite_mask = np.isfinite(vel).all(axis=(1, 2, 3, 4))
        mag_mask    = np.abs(vel).max(axis=(1, 2, 3, 4)) < 1e3
        valid = finite_mask & mag_mask
        if not valid.all():
            n_bad = int((~valid).sum())
            print(f"  Re={re_val}: dropping {n_bad} bad samples "
                  f"({(~finite_mask).sum()} non-finite, {(~mag_mask & finite_mask).sum()} diverged)")
            vel = vel[valid]

        # Keep exactly n_want samples
        vel = vel[:n_want]
        n_loaded = len(vel)

        u0_batch = vel[:, 0, :, :, :].reshape(n_loaded, Nxy * 2).astype(np.float32)
        s_batch  = vel.reshape(n_loaded, Nxyt * 2).astype(np.float32)

        s[pos:pos + n_loaded]  = s_batch
        u0[pos:pos + n_loaded] = u0_batch
        kappa[pos:pos + n_loaded] = float(re_val)
        pos += n_loaded
        del vel, u0_batch, s_batch
        print(f"  Re={re_val}: {n_loaded} trajectories loaded")

    # Trim pre-allocated arrays to actual size
    s     = s[:pos]
    u0    = u0[:pos]
    kappa = kappa[:pos]

    # Create spatial grid
    x_np = np.linspace(0, 1, Nx, dtype=np.float32)
    y_np = np.linspace(0, 1, Ny, dtype=np.float32)
    xx, yy = np.meshgrid(x_np, y_np, indexing="ij")  # (Nx, Ny) each
    xy_np = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)  # (Nxy, 2)

    return s, u0, kappa, xy_np, Nx, Ny, Nt


def load_ns_1d_stacked(
    entries: list[tuple[int, str]],
    n_samples: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Stack 1D Navier-Stokes NPZ files into pre-allocated arrays.

    Each file: velocity (N, Nt, Nx) trajectories.

    Returns:
      s     (N_total, Nt*Nx) - full velocity trajectory flattened
      u0    (N_total, Nx) - initial velocity field
      kappa (N_total,) - Reynolds number per sample
      x_np  (Nx,) - spatial grid coordinates
      Nx, Nt - grid and time dimensions
    """
    _, path0 = entries[0]
    data0 = np.load(path0)
    vel0 = data0["velocity"]  # (N_file, Nt, Nx)
    shape = vel0.shape
    Nt, Nx = shape[1], shape[2]
    n_file = shape[0]

    n_per = min(n_samples, n_file) if n_samples is not None else n_file
    n_total = n_per * len(entries)
    Nxt = Nt * Nx

    print(f"NS-1D: Nx={Nx}, Nt={Nt}, n_per_re={n_per}, "
          f"N_total={n_total}, s≈{n_total*Nxt*4/1e9:.1f} GB")

    s = np.empty((n_total, Nxt), dtype=np.float32)
    u0 = np.empty((n_total, Nx), dtype=np.float32)
    kappa = np.empty(n_total, dtype=np.float32)

    for i, (re_val, path) in enumerate(entries):
        data = np.load(path)
        vel = data["velocity"][:n_per]  # (n_per, Nt, Nx)

        # Extract initial condition u0
        u0_batch = vel[:, 0, :]  # (n_per, Nx)
        u0_batch = u0_batch.astype(np.float32)

        # Flatten full trajectory
        s_batch = vel.reshape(n_per, Nxt).astype(np.float32)

        sl = slice(i * n_per, (i + 1) * n_per)
        s[sl] = s_batch
        u0[sl] = u0_batch
        kappa[sl] = float(re_val)
        del vel, u0_batch, s_batch
        print(f"  Re={re_val}: {n_per} trajectories loaded")

    # Create spatial grid
    x_np = np.linspace(0, 1, Nx, dtype=np.float32)

    return s, u0, kappa, x_np, Nx, Nt