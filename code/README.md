# Code

Source code for the paper. Models, training scripts, and notebooks for Burgers equation experiments.

## Installation

```bash
pip install -r requirements.txt
```

## Structure

```
code/
  models/       core model implementations
  scripts/      training scripts (run on cluster or locally)
  slurm/        SLURM job submission scripts
  notebooks/    exploration and visualization
  utils/        data loading utilities
```

## Training a single model

Run from the **project root**:

```bash
# POD-DeepONet, trained on nu=0.01
python code/scripts/train_pod.py --nu 0.01

# NeuralPOD-DeepONet, trained on nu=0.1
python code/scripts/train_neural_pod.py --nu 0.1

# TEMPO with M=3 regimes, POD basis
python code/scripts/train_tempo.py --M 3 --basis_type pod
```

Results (checkpoints, metrics, plots) are saved to `TEMPO_results/<run_name>/`.

### Key arguments: train_pod.py / train_neural_pod.py

| Argument        | Default      | Description          |
|-----------------|--------------|----------------------|
| `--nu`          | required     | Viscosity            |
| `--run_name`    | auto         | Output directory name |
| `--n_train`     | 9000 / 2500  | Training samples     |
| `--n_epochs`    | 5000 / 80000 | Training epochs      |
| `--max_modes`   | 32 / 20      | Basis modes          |

### Key arguments: train_tempo.py

| Argument           | Default         | Description                        |
|--------------------|-----------------|------------------------------------|
| `--M`              | 3               | Number of regimes                  |
| `--basis_type`     | pod             | Regime basis: `pod` or `fourier`   |
| `--nu_values`      | 0.001 0.1 1.0   | Viscosities to train on            |
| `--n_samples`      | 5000            | Samples per viscosity              |
| `--basis_max_modes`| 32              | Modes per regime basis             |
| `--max_em_iters`   | 30              | Max EM iterations                  |
| `--online_epochs`  | 170             | Online phase training epochs       |

## Running experiments on SLURM

### POD-DeepONet and NeuralPOD-DeepONet

Submits 8 jobs in parallel: 4 viscosity values x 2 models, each on a separate GPU.

```bash
bash code/slurm/submit_all.sh
```

Logs: `logs/pod_nu<nu>_<jobid>.out`, `logs/npod_nu<nu>_<jobid>.out`.

### TEMPO

Submits a sweep over number of regimes `M` (and optionally basis type).
Edit `M_VALUES` and `BASIS_VALUES` in the script before running.

```bash
bash code/slurm/submit_tempo.sh
```

Logs: `logs/tempo_<basis>_M<M>_<jobid>.out`.

## Notebooks

| Notebook | Description |
|---|---|
| `POD-DeepONet.ipynb` | POD basis + branch network |
| `NeuralPOD-DeepONet.ipynb` | Fourier neural basis + branch network |
| `TEMPO.ipynb` | EM offline phase + online gating |
| `Hankel.ipynb` | Hankel matrix analysis of Burgers dynamics |
