#!/usr/bin/env python3
"""
Verify generated Navier-Stokes HDF5 files for correctness.

Checks:
- File exists and is readable
- Correct shape (n_samples, time_steps+1, Nx, Ny, 2)
- Correct data types and attributes
- No NaN/Inf values
- File size is reasonable
"""
import h5py
import numpy as np
from pathlib import Path
import sys


def verify_ns_file(filepath):
    """Verify a single NS HDF5 file."""
    print(f"\nChecking: {filepath.name}")

    try:
        with h5py.File(filepath, 'r') as f:
            if 'velocity' not in f:
                print("  ✗ No 'velocity' dataset found")
                return False

            vel = f['velocity']
            shape = vel.shape
            dtype = vel.dtype

            print(f"  Shape: {shape}")
            print(f"  Dtype: {dtype}")

            # Check shape
            if len(shape) != 5:
                print(f"  ✗ Wrong number of dimensions: {len(shape)} (expected 5)")
                return False

            n_samples, nt, nx, ny, n_comp = shape

            if n_comp != 2:
                print(f"  ✗ Wrong number of components: {n_comp} (expected 2)")
                return False

            if nx != 256 or ny != 256:
                print(f"  ✗ Wrong spatial resolution: {nx}x{ny} (expected 256x256)")
                return False

            # Check data type
            if dtype != np.float32:
                print(f"  ✗ Wrong data type: {dtype} (expected float32)")
                return False

            # Sample check for NaN/Inf
            sample = vel[0, :, :, :, :]
            if np.isnan(sample).any():
                print("  ✗ Contains NaN values")
                return False
            if np.isinf(sample).any():
                print("  ✗ Contains Inf values")
                return False

            # Check attributes
            attrs = dict(f.attrs)
            required_attrs = ['reynolds_number', 'kinematic_viscosity', 'n_samples', 'time_steps']
            missing = [a for a in required_attrs if a not in attrs]
            if missing:
                print(f"  ✗ Missing attributes: {missing}")
                return False

            print(f"  Reynolds: {attrs['reynolds_number']}")
            print(f"  Viscosity: {attrs['kinematic_viscosity']:.6f}")
            print(f"  Samples: {attrs['n_samples']}")
            print(f"  Time steps: {attrs['time_steps']}")

            # File size
            size_gb = filepath.stat().st_size / (1024**3)
            expected_gb = (n_samples * nt * nx * ny * 2 * 4) / (1024**3)
            print(f"  File size: {size_gb:.2f} GB (expected ≈ {expected_gb:.2f} GB)")

            if size_gb > expected_gb * 1.5:
                print(f"  ⚠ File larger than expected (fragmentation?)")

            print(f"  ✓ File OK")
            return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    data_dir = Path.home() / "data" / "2D" / "Navier_Stokes"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    files = sorted(data_dir.glob("*.hdf5"))

    if not files:
        print(f"No HDF5 files found in {data_dir}")
        return 1

    print("=" * 70)
    print("Navier-Stokes Data Verification")
    print("=" * 70)

    results = []
    for fpath in files:
        ok = verify_ns_file(fpath)
        results.append((fpath.name, ok))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    all_ok = all(ok for _, ok in results)
    if all_ok:
        print("\n✓ All files verified successfully!")
        return 0
    else:
        print("\n✗ Some files have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
