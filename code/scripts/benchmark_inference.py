"""
Inference-time benchmark across all neural operator baselines.

Loads checkpoints from code/test_models/, constructs dummy inputs of the correct
shape, and measures per-sample latency using measure_inference_time().
Results are written to code/test_models/timing_results.json.

Usage (from project root):
    python code/scripts/benchmark_inference.py [--benchmarks burgers darcy ns] [--dry-run] [--summary] [--quick]
"""

import argparse
import json
import os
import pathlib
import sys

import torch
import torch.nn as nn

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_CODE = _ROOT / "code"
sys.path.insert(0, str(_CODE))

# CNO2d_simplified can live in several places; CNO_PATH env var takes priority.
_CNO_SEARCH = [
    os.environ.get("CNO_PATH", ""),
    str(_ROOT.parent / "CNO" / "CNO2d_simplified"),
    str(_CODE / "CNO2d_simplified"),
    str(pathlib.Path.home() / "CNO" / "CNO2d_simplified"),
    str(pathlib.Path.home() / "ConvolutionalNeuralOperator" / "CNO2d_simplified"),
]
_CNO_PATH = None
for _p in _CNO_SEARCH:
    if _p and pathlib.Path(_p).exists():
        sys.path.insert(0, _p)
        _CNO_PATH = _p
        break

from utils.datasets import measure_inference_time  # noqa: E402

TEST_MODELS_DIR = _CODE / "test_models"
RESULTS_FILE    = TEST_MODELS_DIR / "timing_results.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_WARMUP = 5
N_REP    = 20
QUICK    = False



def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(d):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(d, f, indent=2)


def run_and_record(key, predict_fn, n_samples, results, dry_run=False):
    print(f"  {key} ...", end=" ", flush=True)
    try:
        ms = measure_inference_time(predict_fn, device=DEVICE,
                                    n_warmup=N_WARMUP, n_rep=N_REP)
        per_sample = ms / n_samples
        print(f"{ms:.2f} ms/call  ->  {per_sample:.4f} ms/sample")
        if not dry_run:
            results[key] = {"inference_ms_per_call": ms,
                            "inference_ms_per_sample": per_sample}
    except Exception as e:
        print(f"FAILED: {e}")


def _count_branch_weights(sd, prefix="branch.net."):
    return sum(1 for k in sd if k.startswith(prefix) and k.endswith(".weight"))



def bench_pod_deeponet(ckpt_path, n_samples, key, results, dry_run):
    if not ckpt_path.exists():
        print(f"  [missing] {ckpt_path.name}")
        return
    try:
        from models.pod import PODBasis
        from models.pod_deeponet import BranchNet, PODDeepONet

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd   = ckpt.get("model", ckpt.get("model_state"))

        in_dim  = sd["branch.net.0.weight"].shape[1]
        hidden  = sd["branch.net.0.weight"].shape[0]
        n_modes = sd["basis.modes"].shape[1]
        n_w     = _count_branch_weights(sd)

        # PODBasis buffers are set via initialize(), not __init__; must call before load_state_dict.
        basis = PODBasis()
        basis.initialize(
            mean=sd["basis.mean"].cpu(),
            modes=sd["basis.modes"].cpu(),
            coeffs=sd["basis.coeffs"].cpu(),
        )
        branch = BranchNet(m=in_dim, P=n_modes, hidden_dim=hidden,
                           n_layers=n_w - 1, d_kappa=0)
        model = PODDeepONet(basis, branch).to(DEVICE)
        model.load_state_dict(sd)
        model.eval()

        u0 = torch.randn(n_samples, in_dim, device=DEVICE)

        def predict():
            with torch.no_grad():
                return model(u0)

        run_and_record(key, predict, n_samples, results, dry_run)

    except Exception as e:
        print(f"  {key}  FAILED: {e}")



def bench_npod_deeponet(ckpt_path, n_samples, Ny, d_x, key, results, dry_run):
    if not ckpt_path.exists():
        print(f"  [missing] {ckpt_path.name}")
        return
    try:
        from models.regime_basis import FourierRegimeBasis
        from models.pod_deeponet import BranchNet
        from models.neural_pod_deeponet import NeuralPODDeepONet

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd_model  = ckpt.get("model")
        sd_branch = ckpt.get("branch")
        sd_basis  = ckpt.get("basis")

        # Older checkpoints store branch and basis under separate keys;
        # newer ones store the full combined state dict under "model".
        if sd_model is not None and any(k.startswith("basis.") for k in sd_model):
            sd = sd_model
        else:
            branch_sd = sd_branch or sd_model or {}
            sd = {}
            if branch_sd:
                sd.update({f"branch.{k}": v for k, v in branch_sd.items()})
            if sd_basis:
                sd.update({f"basis.{k}": v for k, v in sd_basis.items()})

        if not any(k.endswith(".lambda_ten") for k in sd):
            print(f"  {key}  SKIPPED: RegimeBasis checkpoint - FourierRegimeBasis expected")
            return

        K         = sum(1 for k in sd if k.startswith("basis.modes.") and k.endswith(".lambda_ten"))
        M_train   = sd["basis.modes.0.lambda_ten"].shape[0]
        num_freq  = sd["basis.mean_net.net.0.weight"].shape[1] // 2
        basis_h   = sd["basis.mean_net.net.0.weight"].shape[0]
        n_phi_w   = _count_branch_weights(sd, "basis.modes.0.phi.net.")
        branch_in = sd["branch.net.0.weight"].shape[1]
        branch_h  = sd["branch.net.0.weight"].shape[0]
        branch_nw = _count_branch_weights(sd)

        # Infer Ny from the saved quad_weights - its size may differ from the benchmark default.
        if "basis.quad_weights" in sd:
            Ny = sd["basis.quad_weights"].shape[0]

        basis = FourierRegimeBasis(
            d_x=d_x, M=M_train,
            quad_weights=torch.ones(Ny, device=DEVICE) / Ny,
            hidden_dim=basis_h,
            num_frequencies=num_freq,
            n_layers=n_phi_w - 1,
        ).to(DEVICE)
        for _ in range(K):
            basis.add_mode()

        branch = BranchNet(m=branch_in, P=K, hidden_dim=branch_h,
                           n_layers=branch_nw - 1, d_kappa=0).to(DEVICE)
        model = NeuralPODDeepONet(basis, branch).to(DEVICE)
        # strict=False: the Fourier matrix B is a non-persistent buffer, absent from the checkpoint.
        model.load_state_dict(sd, strict=False)
        model.eval()

        u0     = torch.randn(n_samples, branch_in, device=DEVICE)
        x_flat = torch.randn(Ny, d_x, device=DEVICE)

        def predict():
            with torch.no_grad():
                return model(u0, x_flat)

        run_and_record(key, predict, n_samples, results, dry_run)

    except Exception as e:
        print(f"  {key}  FAILED: {e}")



def _mlp(in_dim, out_dim, hidden_dim, n_layers):
    layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def bench_deeponet(n_samples, m, Ny, trunk_coord_dim, n_components,
                   key, results, dry_run, hidden_dim=256, n_layers=4):
    """Random-weight DeepONet - architecture matches training scripts exactly.

    n_components=1: Burgers/Darcy (scalar output per query point)
    n_components=2: NS (two velocity components per query point)
    """
    try:
        d = hidden_dim

        class _Branch(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = _mlp(m + 1, d, hidden_dim, n_layers)
            def forward(self, u0, kappa):
                return self.net(torch.cat([u0, kappa], dim=-1))

        class _Trunk(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = _mlp(trunk_coord_dim, n_components * d, hidden_dim, n_layers)
            def forward(self, x):
                return self.net(x)

        class _DeepONet(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch = _Branch()
                self.trunk  = _Trunk()
                self.bias   = nn.Parameter(torch.zeros(1))
            def forward(self, u0, kappa, x):
                b = self.branch(u0, kappa)           # (N, d)
                t = self.trunk(x)                    # (Ny, n_components*d)
                if n_components == 1:
                    return torch.einsum("nd,md->nm", b, t) + self.bias
                t = t.reshape(Ny, n_components, d)
                out = torch.einsum("nd,mcd->nmc", b, t) + self.bias
                return out.reshape(out.shape[0], -1)

        model = _DeepONet().to(DEVICE)
        model.eval()

        u0    = torch.randn(n_samples, m, device=DEVICE)
        kappa = torch.randn(n_samples, 1, device=DEVICE)
        x     = torch.randn(Ny, trunk_coord_dim, device=DEVICE)

        def predict():
            with torch.no_grad():
                return model(u0, kappa, x)

        run_and_record(key, predict, n_samples, results, dry_run)

    except Exception as e:
        print(f"  {key}  FAILED: {e}")



def bench_fno(ckpt_path, x_dummy, n_samples, key, results, dry_run,
              in_channels, out_channels, n_modes_tuple, batch_size=64):
    if not ckpt_path.exists():
        print(f"  [missing] {ckpt_path.name}")
        return
    try:
        from neuralop.models import FNO

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        hp   = ckpt.get("hparams", {})
        sd   = ckpt.get("model_state", ckpt.get("model"))

        nm = hp.get("n_modes", None)
        if nm is not None and not isinstance(nm, (list, tuple)):
            n_modes_tuple = (nm, nm)
        hd = hp.get("hidden_dim", 32)
        nl = hp.get("n_layers", 4)

        model = FNO(
            n_modes=n_modes_tuple,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hd,
            n_layers=nl,
            use_channel_mlp=True,
            positional_embedding=None,
        ).to(DEVICE)
        model.load_state_dict(sd)
        model.eval()

        x_d = x_dummy.to(DEVICE)

        def predict():
            with torch.no_grad():
                out = []
                for i in range(0, len(x_d), batch_size):
                    out.append(model(x_d[i:i + batch_size]))
                return torch.cat(out)

        run_and_record(key, predict, n_samples, results, dry_run)

    except Exception as e:
        print(f"  {key}  FAILED: {e}")



def bench_cno(ckpt_path, x_dummy, n_samples, key, results, dry_run,
              in_dim, out_dim, size, batch_size=64):
    if not ckpt_path.exists():
        print(f"  [missing] {ckpt_path.name}")
        return
    if _CNO_PATH is None:
        print(f"  {key}  SKIPPED: CNO2d_simplified not found "
              f"(set CNO_PATH or clone camlab-ethz/ConvolutionalNeuralOperator)")
        return
    try:
        from CNO2d import CNO2d

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        hp   = ckpt.get("hparams", {})
        sd   = ckpt.get("model_state", ckpt.get("model"))

        model = CNO2d(
            in_dim=in_dim,
            out_dim=out_dim,
            size=size,
            N_layers=hp.get("n_layers", 4),
            N_res=hp.get("n_res", 4),
            N_res_neck=hp.get("n_res_neck", 4),
            channel_multiplier=hp.get("channel_multiplier", 16),
            use_bn=hp.get("use_bn", False),
        ).to(DEVICE)
        model.load_state_dict(sd)
        model.eval()

        x_d = x_dummy.to(DEVICE)

        def predict():
            with torch.no_grad():
                out = []
                for i in range(0, len(x_d), batch_size):
                    out.append(model(x_d[i:i + batch_size]))
                return torch.cat(out)

        run_and_record(key, predict, n_samples, results, dry_run)

    except Exception as e:
        print(f"  {key}  FAILED: {e}")



def bench_tempo(ckpt_path, n_samples, Ny, key, results, dry_run):
    if not ckpt_path.exists():
        print(f"  [missing] {ckpt_path.name}")
        return
    try:
        from models.tempo_online import TEMPOOnline, GatingNet
        from models.pod_deeponet import BranchNet

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        mo   = ckpt["model_online"]

        d_kappa    = 1
        gating_in  = mo["gating.branch.net.0.weight"].shape[1]
        m_sensors  = gating_in - d_kappa
        hidden_dim = mo["gating.branch.net.0.weight"].shape[0]
        n_gating_w = _count_branch_weights(mo, "gating.branch.net.")
        M = sum(1 for k in mo if k.startswith("branches.") and ".net.0.weight" in k)

        P_list = []
        for m in range(M):
            last_key = sorted(
                [k for k in mo if k.startswith(f"branches.{m}.net.") and k.endswith(".weight")],
                key=lambda x: int(x.split(".")[3]),
            )[-1]
            P_list.append(mo[last_key].shape[0])

        gating   = GatingNet(m_sensors, d_kappa, M, hidden_dim, n_gating_w - 1)
        branches = nn.ModuleList([
            BranchNet(m_sensors, P_list[m], hidden_dim, n_gating_w - 1, d_kappa=d_kappa)
            for m in range(M)
        ])
        model = TEMPOOnline(gating, branches).to(DEVICE)
        model.load_state_dict(mo)
        model.eval()

        u0_s  = torch.randn(n_samples, m_sensors, device=DEVICE)
        kap   = torch.randn(n_samples, d_kappa, device=DEVICE)
        means = [torch.zeros(Ny, device=DEVICE) for _ in range(M)]
        modes = [torch.zeros(Ny, P_list[m], device=DEVICE) for m in range(M)]

        def predict():
            with torch.no_grad():
                return model(u0_s, kap, means, modes)

        run_and_record(key, predict, n_samples, results, dry_run)

    except Exception as e:
        print(f"  {key}  FAILED: {e}")



def benchmark_burgers(results, dry_run=False):
    print("\n" + "=" * 65)
    print("BURGERS  (Nx=1024, Nt=201)")
    print("=" * 65)
    bdir   = TEST_MODELS_DIR / "burgers"
    Ny_pod = 1024 * 201

    q = QUICK
    N_pod   = 4  if q else 256
    N_npod  = 2  if q else 64
    N_fno   = 2  if q else 32   # input shape (N, 4, 1024, 201) is large; keep N small on CPU
    N_cno   = 4  if q else 500
    N_tempo = 2  if q else 64

    bench_pod_deeponet(bdir / "pod_deeponet.pt", N_pod, "burgers/pod_deeponet", results, dry_run)
    bench_npod_deeponet(bdir / "npod_deeponet.pt", N_npod, Ny_pod, d_x=2,
                        key="burgers/npod_deeponet", results=results, dry_run=dry_run)
    bench_fno(bdir / "fno.pt", x_dummy=torch.randn(N_fno, 4, 1024, 201),
              n_samples=N_fno, key="burgers/fno", results=results, dry_run=dry_run,
              in_channels=4, out_channels=1, n_modes_tuple=(16, 16))
    bench_cno(bdir / "cno.pt", x_dummy=torch.randn(N_cno, 4, 128, 128),
              n_samples=N_cno, key="burgers/cno", results=results, dry_run=dry_run,
              in_dim=4, out_dim=1, size=128)
    bench_tempo(bdir / "tempo_pod.pt",  N_tempo, Ny_pod, "burgers/tempo_pod",  results, dry_run)
    bench_tempo(bdir / "tempo_npod.pt", N_tempo, Ny_pod, "burgers/tempo_npod", results, dry_run)

    N_don = 4 if q else 64
    # Burgers DeepONet data: Nx=101, Nt=256 (different resolution from FNO/POD benchmarks)
    # confirmed from HDF5 shape (9500, 256, 101, 1) and n_params=553473
    bench_deeponet(N_don, m=101, Ny=256*101, trunk_coord_dim=2, n_components=1,
                   key="burgers/deeponet", results=results, dry_run=dry_run)


def benchmark_darcy(results, dry_run=False):
    print("\n" + "=" * 65)
    print("DARCY  (128x128, steady-state)")
    print("=" * 65)
    bdir   = TEST_MODELS_DIR / "darcy"
    Ny_pod = 128 * 128

    N_pod   = 1000 if not QUICK else 4
    N_npod  = 200  if not QUICK else 2
    N_fno   = 1000 if not QUICK else 2
    N_cno   = 1000 if not QUICK else 4
    N_tempo = 256  if not QUICK else 2

    bench_pod_deeponet(bdir / "pod_deeponet.pt", N_pod, "darcy/pod_deeponet", results, dry_run)
    bench_npod_deeponet(bdir / "npod_deeponet.pt", N_npod, Ny_pod, d_x=2,
                        key="darcy/npod_deeponet", results=results, dry_run=dry_run)
    bench_fno(bdir / "fno.pt", x_dummy=torch.randn(N_fno, 4, 128, 128),
              n_samples=N_fno, key="darcy/fno", results=results, dry_run=dry_run,
              in_channels=4, out_channels=1, n_modes_tuple=(16, 16))
    bench_cno(bdir / "cno.pt", x_dummy=torch.randn(N_cno, 4, 128, 128),
              n_samples=N_cno, key="darcy/cno", results=results, dry_run=dry_run,
              in_dim=4, out_dim=1, size=128)
    bench_tempo(bdir / "tempo_pod.pt",  N_tempo, Ny_pod, "darcy/tempo_pod",  results, dry_run)
    bench_tempo(bdir / "tempo_npod.pt", N_tempo, Ny_pod, "darcy/tempo_npod", results, dry_run)

    N_don = 4 if QUICK else 64
    # m=16384 (all sensors, stride=1), Ny=Nx*Ny, trunk takes (x,y)
    bench_deeponet(N_don, m=16384, Ny=Ny_pod, trunk_coord_dim=2, n_components=1,
                   key="darcy/deeponet", results=results, dry_run=dry_run)


def benchmark_ns(results, dry_run=False):
    print("\n" + "=" * 65)
    print("NAVIER-STOKES  (64x64, Nt=21)")
    print("=" * 65)
    bdir   = TEST_MODELS_DIR / "ns"
    Ny_pod = 64 * 64 * 21

    N_pod  = 256 if not QUICK else 4
    N_npod = 64  if not QUICK else 2
    N_fno  = 500 if not QUICK else 2
    N_cno  = 500 if not QUICK else 4

    bench_pod_deeponet(bdir / "pod_deeponet.pt", N_pod, "ns/pod_deeponet", results, dry_run)
    bench_npod_deeponet(bdir / "npod_deeponet.pt", N_npod, Ny_pod, d_x=2,
                        key="ns/npod_deeponet", results=results, dry_run=dry_run)
    bench_fno(bdir / "fno.pt", x_dummy=torch.randn(N_fno, 2, 64, 64),
              n_samples=N_fno, key="ns/fno", results=results, dry_run=dry_run,
              in_channels=2, out_channels=42, n_modes_tuple=(12, 12))
    bench_cno(bdir / "cno.pt", x_dummy=torch.randn(N_cno, 2, 64, 64),
              n_samples=N_cno, key="ns/cno", results=results, dry_run=dry_run,
              in_dim=2, out_dim=42, size=64)

    N_tempo = 256 if not QUICK else 2
    bench_tempo(bdir / "tempo_pod.pt",  N_tempo, Ny_pod, "ns/tempo_pod",  results, dry_run)
    bench_tempo(bdir / "tempo_npod.pt", N_tempo, Ny_pod, "ns/tempo_npod", results, dry_run)

    N_don = 4 if QUICK else 64
    # m=8192 (64x64x2 flattened velocity), Ny=Nt*Nx*Ny=86016, trunk takes (x,y,t), 2 components
    bench_deeponet(N_don, m=8192, Ny=86016, trunk_coord_dim=3, n_components=2,
                   key="ns/deeponet", results=results, dry_run=dry_run)



def print_summary(results):
    print("\n" + "=" * 70)
    print(f"{'INFERENCE TIME SUMMARY':^70}")
    print("=" * 70)
    print(f"  {'Model':45s} {'ms/sample':>10}")
    print("-" * 70)
    for key in sorted(results):
        val = results[key].get("inference_ms_per_sample")
        if val is not None:
            print(f"  {key:45s} {val:10.4f}")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", nargs="+", default=["burgers", "darcy", "ns"],
                   choices=["burgers", "darcy", "ns"])
    p.add_argument("--dry-run",  action="store_true", help="Measure but do not save results")
    p.add_argument("--summary",  action="store_true", help="Print saved results without re-running")
    p.add_argument("--quick",    action="store_true", help="1 warmup + 1 rep, tiny N (smoke test)")
    args = p.parse_args()

    if args.quick:
        global N_WARMUP, N_REP, QUICK
        N_WARMUP, N_REP, QUICK = 1, 1, True

    print(f"Device: {DEVICE}  |  warmup={N_WARMUP}  reps={N_REP}")
    print(f"Models: {TEST_MODELS_DIR}")
    print(f"CNO:    {_CNO_PATH or 'not found - CNO benchmarks will be skipped'}")

    results = load_results()

    if args.summary:
        print_summary(results)
        return

    if "burgers" in args.benchmarks:
        benchmark_burgers(results, args.dry_run)
    if "darcy" in args.benchmarks:
        benchmark_darcy(results, args.dry_run)
    if "ns" in args.benchmarks:
        benchmark_ns(results, args.dry_run)

    if not args.dry_run:
        save_results(results)
        print(f"\nResults saved -> {RESULTS_FILE}")

    print_summary(results)


if __name__ == "__main__":
    main()
