import os
import argparse
import numpy as np
import h5py

DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)

HF_REPO = "erbacher/PDEBench-1D"

# (hf_config, split, x_range, t_range, n_channels)
#  n_channels: 1 for scalar fields, 3 for CFD (rho/v/p), 2 for ReacDiff (u/v)
DATASETS = {
    # Burgers: Nx=1024, Nt=101, x in (-1,1), t in (0,2), V=1
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