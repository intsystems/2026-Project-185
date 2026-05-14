# Code

Source code for the paper. Models and training scripts for three benchmarks: 1D Burgers, 2D Darcy flow, and 2D Navier-Stokes.

## Installation

```bash
pip install -r requirements.txt
```

## Structure

```
code/
  models/                 core model implementations
  scripts/                training scripts (run from project root)
    data_generation/      NS data generators and verification
  slurm/                  SLURM job submission scripts
  notebooks/              exploration and visualization
  utils/                  data loading utilities
  results/                experiment outputs (checkpoints, metrics, plots)
```

## Training

Run all scripts from the **project root**. Each script supports `--help` for full argument list.

### Burgers (1D, multi-viscosity)

```bash
python code/scripts/train_tempo.py --nu_values 0.001 0.01 0.1 1.0 --M 4 --basis_type pod
python code/scripts/train_pod.py --nu_values 0.001 0.01 0.1 1.0
python code/scripts/train_neural_pod.py --nu_values 0.001 0.01 0.1 1.0
```

### Darcy Flow (2D, multi-beta)

```bash
python code/scripts/train_tempo_darcy.py --M 4 --basis_type pod
python code/scripts/train_pod_darcy.py
python code/scripts/train_neural_pod_darcy.py
```

### Navier-Stokes (2D, multi-Reynolds)

```bash
# Generate data first (~3-4 hours on CPU)
python code/scripts/data_generation/generate_navier_stokes_spectral.py

python code/scripts/train_tempo_navier_stokes.py --re_values 100 1000 3600 10000 --M 4
python code/scripts/train_pod_navier_stokes.py --re_values 1000
python code/scripts/train_neural_pod_navier_stokes.py --re_values 1000
```

