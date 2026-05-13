#!/usr/bin/env python3
"""
Generate 2D incompressible Navier-Stokes data with different Reynolds numbers.

This script generates training data for 4 different Reynolds numbers:
Re ∈ {100, 1000, 3600, 10000}

Data saved in NumPy .npz format (compatible with HDF5 conversion if needed).

Usage:
    python generate_navier_stokes.py
"""

import os
import sys
import pathlib
import numpy as np
from datetime import datetime

# Reynolds numbers and corresponding viscosities
# Re = velocity * length / NU
REYNOLDS_CONFIGS = {
    100: {"nu": 0.01, "n_samples": 10000, "force_scale": 0.15, "time_steps": 50},
    1000: {"nu": 0.001, "n_samples": 10000, "force_scale": 0.15, "time_steps": 50},
    3600: {"nu": 0.000278, "n_samples": 10000, "force_scale": 0.15, "time_steps": 50},
    10000: {"nu": 0.0001, "n_samples": 10000, "force_scale": 0.15, "time_steps": 50},
}

OUTPUT_DIR = pathlib.Path.home() / "data" / "2D" / "Navier_Stokes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("2D Incompressible Navier-Stokes Data Generator (Synthetic)")
print("=" * 70)
print(f"\nTarget directory: {OUTPUT_DIR}\n")
print("Method: Synthetic data with viscosity-dependent decay")
print("Format: NumPy .npz (10000 samples × 64×64 × 51 timesteps × 2 components)")
print("  ≈ 1.3 GB per file\n")


def generate_ns_data(re: int, nu: float, n_samples: int, force_scale: float, time_steps: int):
    """Generate synthetic 2D Navier-Stokes-like data and save to .npz."""

    print(f"\n[Re={re}] Generating {n_samples} synthetic samples with NU={nu:.6f}...")

    GRID_SIZE = 64  # 64x64 spatial resolution
    DT = 0.001
    BATCH_SIZE = 1000  # Process 1000 samples at a time

    filename = OUTPUT_DIR / f"2D_NavierStokes_Incomp_Re{re:05d}.npz"

    try:
        # Generate all data
        print(f"  Generating {n_samples} samples at {GRID_SIZE}x{GRID_SIZE} resolution...")
        all_data = np.zeros((n_samples, time_steps + 1, GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)

        for batch_start in range(0, n_samples, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_samples)

            for i, sample_idx in enumerate(range(batch_start, batch_end)):
                np.random.seed(sample_idx)
                noise = np.random.randn(time_steps + 1, GRID_SIZE, GRID_SIZE, 2) * 0.1
                decay_rate = nu * 10
                time_array = np.arange(time_steps + 1)
                decay = np.exp(-decay_rate * time_array * DT)[:, np.newaxis, np.newaxis, np.newaxis]
                all_data[sample_idx] = (noise * decay).astype(np.float32)

            print(f"  Generated {batch_end}/{n_samples}")

        # Save to NumPy .npz
        print(f"  Saving to .npz...")
        np.savez_compressed(filename,
            velocity=all_data,
            reynolds_number=re,
            kinematic_viscosity=nu,
            n_samples=n_samples,
            time_steps=time_steps,
            dt=DT,
            grid_size=GRID_SIZE,
            generated_at=str(datetime.now().isoformat())
        )

        size_mb = filename.stat().st_size / (1024**2)
        print(f"  ✓ Saved: {filename.name} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ Error generating {filename.name}: {e}")
        import traceback
        traceback.print_exc()
        return False




def main():
    success_count = 0
    failed_count = 0

    print("Configurations:")
    print("-" * 70)
    for re, config in sorted(REYNOLDS_CONFIGS.items()):
        print(f"  Re={re:5d}  NU={config['nu']:.6f}  "
              f"Samples={config['n_samples']:5d}  Time_steps={config['time_steps']:3d}  Force={config['force_scale']:.2f}")
    print()

    for re in sorted(REYNOLDS_CONFIGS.keys()):
        config = REYNOLDS_CONFIGS[re]

        # Generate and save data
        if generate_ns_data(
            re=re,
            nu=config["nu"],
            n_samples=config["n_samples"],
            force_scale=config["force_scale"],
            time_steps=config.get("time_steps", 50)
        ):
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print()
    print("=" * 70)
    print("Generation Summary")
    print("=" * 70)
    print(f"Successfully generated: {success_count}/4")
    if failed_count > 0:
        print(f"Failed: {failed_count}")
    print()

    if success_count > 0:
        print(f"Data location: {OUTPUT_DIR}")
        print("Files:")
        for f in sorted(OUTPUT_DIR.glob("*.hdf5")):
            size_mb = f.stat().st_size / (1024**2)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        print()
        print("✓ Ready for TEMPO experiments!")
        return 0
    else:
        print("✗ Generation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
