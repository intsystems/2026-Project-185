#!/usr/bin/env python
"""Kappa prediction ablation for NeuralPOD M=3 on Burgers'.

Trains an MLP classifier to predict nu from u0, then runs TEMPO(NeuralPOD)
inference with: kappa_real, kappa_zero, kappa_predicted.

Usage:
  python code/scripts/ablation_kappa_npod_predicted.py \
      --run_dir code/results/TEMPO_RES_FNO_BURGERS_TEMPO/burgers/tempo_npod_burgers_M3
"""

import argparse
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "code"))

from models.tempo_online import TEMPOOnline, GatingNet, _eval_basis, _num_modes
from models.pod_deeponet import BranchNet
from utils.datasets import load_stacked


# ---------------------------------------------------------------------------
# Data loading — matches training script exactly
# ---------------------------------------------------------------------------

def load_burgers(data_dir, nu_values, n_samples, n_test_per_nu):
    entries = [(nu, os.path.join(data_dir, f"1D_Burgers_Sols_Nu{nu}.hdf5"))
               for nu in nu_values]
    s_np, kappa_np, x_np, t_np, Nx, Nt = load_stacked(entries, n_samples=n_samples)

    N_per = n_samples
    train_idx = np.concatenate([
        np.arange(i * N_per, (i + 1) * N_per - n_test_per_nu)
        for i in range(len(nu_values))])
    test_idx = np.concatenate([
        np.arange((i + 1) * N_per - n_test_per_nu, (i + 1) * N_per)
        for i in range(len(nu_values))])

    s = torch.from_numpy(s_np)
    kappa = torch.from_numpy(kappa_np[:, None])
    u0 = s[:, :Nx]  # first time step = initial condition

    # Coordinate grid: x_flat columns = [x, t]
    x_grid = torch.tensor(x_np, dtype=torch.float32)
    t_grid = torch.tensor(t_np, dtype=torch.float32)
    tt, xx = torch.meshgrid(t_grid, x_grid, indexing="ij")
    x_flat = torch.stack([xx.flatten(), tt.flatten()], dim=1)  # (Nt*Nx, 2)

    return (s[train_idx], u0[train_idx], kappa[train_idx],
            s[test_idx],  u0[test_idx],  kappa[test_idx],
            x_flat, Nx)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(run_dir, device):
    from models.fourier_neural_pod import FourierNeuralPODTrainer

    trainer_path = os.path.join(run_dir, "trainer.pt")
    model_path   = os.path.join(run_dir, "model_online.pt")

    trainer_obj = torch.load(trainer_path, map_location="cpu", weights_only=False)
    ckpt        = torch.load(model_path,   map_location="cpu", weights_only=False)

    cfg_online   = ckpt["cfg_online"]
    sd           = ckpt["model_online"]
    sub_trainers = trainer_obj.trainers
    M            = len(sub_trainers)

    P_list = [_num_modes(t) for t in sub_trainers]

    w0 = sd["gating.branch.net.0.weight"]
    hidden_dim = w0.shape[0]
    d_kappa    = 1
    m_sensors  = w0.shape[1] - d_kappa
    n_linears  = sum(1 for k in sd
                     if k.startswith("gating.branch.net.") and k.endswith(".weight"))
    n_layers   = n_linears - 1

    gating   = GatingNet(m_sensors, d_kappa, M, hidden_dim, n_layers)
    branches = nn.ModuleList([
        BranchNet(m_sensors, P_list[m], hidden_dim, n_layers, d_kappa=d_kappa)
        for m in range(M)
    ])
    model = TEMPOOnline(gating, branches)
    model.load_state_dict(sd)
    model.eval().to(device)

    stride = cfg_online.sensor_stride
    return model, sub_trainers, stride, M


def compute_bases(sub_trainers, x_flat, device):
    from models.fourier_neural_pod import FourierNeuralPODTrainer

    x_dev = x_flat.to(device)
    means_list, modes_list = [], []
    for t in sub_trainers:
        if isinstance(t, FourierNeuralPODTrainer):
            t.basis.to(device).eval()
        with torch.no_grad():
            mean, modes = _eval_basis(t, x_dev)
        means_list.append(mean.float().detach())
        modes_list.append(modes.float().detach())
    return means_list, modes_list


# ---------------------------------------------------------------------------
# kappa classifier: predicts nu class from u0
# ---------------------------------------------------------------------------

def build_classifier(in_dim, n_classes, hidden=256):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, 128),    nn.ReLU(),
        nn.Linear(128, n_classes),
    )


def _labels_to_idx(nu_labels, nu_values):
    """Map float kappa tensor to class indices, robust to float32 precision."""
    nu_arr  = np.array(nu_values, dtype=np.float64)
    vals    = nu_labels.squeeze().numpy().astype(np.float64)
    indices = np.argmin(np.abs(vals[:, None] - nu_arr[None, :]), axis=1)
    return torch.from_numpy(indices).long()


def train_classifier(u0_train, nu_labels_train, nu_values, device, epochs=30, batch=256):
    """Train MLP to classify nu from u0.

    nu_labels_train: (N, 1) float tensor of nu values — converted to class indices.
    Returns: trained model (on device).
    """
    y_train = _labels_to_idx(nu_labels_train, nu_values)

    in_dim    = u0_train.shape[1]
    n_classes = len(nu_values)
    clf       = build_classifier(in_dim, n_classes).to(device)
    opt       = torch.optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    ds     = TensorDataset(u0_train.to(device), y_train.to(device))
    loader = DataLoader(ds, batch_size=batch, shuffle=True)

    print(f"\n--- Training nu classifier (in={in_dim}, classes={n_classes}) ---")
    for epoch in range(1, epochs + 1):
        clf.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for xb, yb in loader:
            logits = clf(xb)
            loss   = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(xb)
            n_correct  += (logits.argmax(1) == yb).sum().item()
            n_total    += len(xb)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}: loss={total_loss/n_total:.4f}  "
                  f"train_acc={n_correct/n_total*100:.1f}%")

    return clf


@torch.no_grad()
def eval_classifier(clf, u0_test, nu_labels_test, nu_values, device):
    """Predict nu for each test sample. Returns kappa_pred tensor (N, 1)."""
    y_true = _labels_to_idx(nu_labels_test, nu_values)

    clf.eval()
    all_preds = []
    for i in range(0, len(u0_test), 512):
        logits = clf(u0_test[i:i+512].to(device))
        all_preds.append(logits.argmax(1).cpu())
    y_pred = torch.cat(all_preds)

    acc = (y_pred == y_true).float().mean().item()
    print(f"\n--- Classifier evaluation ---")
    print(f"  overall accuracy: {acc*100:.1f}%")
    for i, v in enumerate(nu_values):
        mask = (y_true == i)
        per_acc = (y_pred[mask] == y_true[mask]).float().mean().item()
        print(f"  nu={v}: {per_acc*100:.1f}%  ({mask.sum()} samples)")

    # Map predicted class back to nu value (float)
    idx_to_nu = {i: float(v) for i, v in enumerate(nu_values)}
    kappa_pred = torch.tensor(
        [idx_to_nu[int(c)] for c in y_pred.tolist()],
        dtype=torch.float32
    ).unsqueeze(1)  # (N, 1)

    return kappa_pred, acc


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batched(model, u0_sensors, kappa, means, modes, batch=256):
    preds, ws = [], []
    for i in range(0, len(u0_sensors), batch):
        p, w = model(u0_sensors[i:i+batch], kappa[i:i+batch], means, modes)
        preds.append(p.cpu())
        ws.append(w.cpu())
    return torch.cat(preds), torch.cat(ws)


def rel_l2_vec(pred, true):
    return (torch.norm(pred - true, dim=1) /
            torch.norm(true, dim=1).clamp(min=1e-8)).numpy()


def report(label, err_vec, kappa_np, nu_values):
    print(f"  {label}: overall={err_vec.mean():.4f}")
    for v in nu_values:
        mask = np.abs(kappa_np - v) < 1e-6
        print(f"    nu={v}: {err_vec[mask].mean():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir",
                   default=os.path.expanduser("~/data/1D/Burgers/Train"))
    p.add_argument("--nu_values", type=float, nargs="+",
                   default=[0.001, 0.1, 1.0])
    p.add_argument("--n_samples",     type=int, default=5000)
    p.add_argument("--n_test_per_nu", type=int, default=1000)
    p.add_argument("--clf_epochs",    type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device
    print(f"device={device}  run_dir={args.run_dir}")

    # --- Load model ---
    model, sub_trainers, stride, M = load_model(args.run_dir, device)
    basis_type = type(sub_trainers[0]).__name__
    print(f"M={M}  basis={basis_type}  stride={stride}")

    # --- Load data ---
    (s_train, u0_train, kappa_train,
     s_test,  u0_test,  kappa_test,
     x_flat, Nx) = load_burgers(
        args.data_dir, args.nu_values, args.n_samples, args.n_test_per_nu)

    print(f"train: {s_train.shape}  test: {s_test.shape}")

    nu_np = kappa_test[:, 0].numpy()

    # --- Compute frozen bases ---
    means, modes = compute_bases(sub_trainers, x_flat, device)

    # --- Sensor inputs ---
    u0_sensors_test  = u0_test[:, ::stride].to(device)
    s_true           = s_test.to(device)

    # Sanity-check sensor dim
    expected = model.state_dict()["gating.branch.net.0.weight"].shape[1] - 1
    actual   = u0_sensors_test.shape[1]
    if actual != expected:
        raise RuntimeError(
            f"Sensor mismatch: model expects {expected}, got {actual} "
            f"(Nx={Nx}, stride={stride})"
        )

    # --- Train nu classifier on FULL u0 (no stride) ---
    # Classifier uses full-resolution u0 for better accuracy
    clf = train_classifier(u0_train, kappa_train, args.nu_values, device, epochs=args.clf_epochs)

    # --- Predict nu for test set ---
    kappa_pred, clf_acc = eval_classifier(clf, u0_test, kappa_test, args.nu_values, device)
    kappa_pred = kappa_pred.to(device)

    # --- Three kappa variants ---
    kappa_real = kappa_test.to(device)
    kappa_zero = torch.zeros_like(kappa_real)

    print("\n--- Inference ---")
    pred_real, _ = predict_batched(model, u0_sensors_test, kappa_real, means, modes)
    pred_zero, _ = predict_batched(model, u0_sensors_test, kappa_zero, means, modes)
    pred_pred, _ = predict_batched(model, u0_sensors_test, kappa_pred, means, modes)

    s_cpu    = s_true.cpu()
    err_real = rel_l2_vec(pred_real, s_cpu)
    err_zero = rel_l2_vec(pred_zero, s_cpu)
    err_pred = rel_l2_vec(pred_pred, s_cpu)

    # --- Results ---
    print(f"\n{'='*60}")
    print(f"=== KAPPA ABLATION: NeuralPOD M={M} Burgers' ===")
    print(f"    N_test={len(s_test)}, clf_acc={clf_acc*100:.1f}%")
    print(f"{'='*60}")

    report("kappa_real",      err_real, nu_np, args.nu_values)
    print()
    report("kappa_zero",      err_zero, nu_np, args.nu_values)
    print()
    report("kappa_predicted", err_pred, nu_np, args.nu_values)

    print(f"\n--- Degradation vs kappa_real ---")
    d_zero = err_zero.mean() - err_real.mean()
    d_pred = err_pred.mean() - err_real.mean()
    print(f"  kappa=0      : {d_zero:+.4f} ({d_zero/err_real.mean()*100:+.1f}%)")
    print(f"  kappa_predict: {d_pred:+.4f} ({d_pred/err_real.mean()*100:+.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
