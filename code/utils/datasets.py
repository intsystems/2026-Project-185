import os
import argparse
import numpy as np
import h5py
from datasets import load_dataset

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
    """Download dataset"""
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


def load(name: str) -> dict[str, np.ndarray]:
    if not os.path.exists(hdf5_path(name)):
        download(name)

    with h5py.File(hdf5_path(name), "r") as f:
        return {key: f[key][:] for key in f.keys()}
