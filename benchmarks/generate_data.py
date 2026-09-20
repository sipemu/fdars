"""
Seeded curve-set generator for the fdars benchmark harness.

Produces a deterministic (n, m) matrix of sinusoidal curves with noise,
written as a full-precision CSV consumed bit-identically by Python, R, and
Rust runners.

Usage:
    python generate_data.py --n 60 --m 50 --seed 42
    python generate_data.py --n 100 --m 100 --seed 42
"""

import argparse
import hashlib
import os
import pathlib

import numpy as np
import pandas as pd


def generate_curves(n: int, m: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate a deterministic (n, m) matrix of sinusoidal curves with noise.

    Each curve: random_amplitude * sin(2*pi*t + phase) + noise_scale * N(0,1)

    The numpy.random.default_rng(seed) RNG ensures bit-identical output across
    Python versions and platforms (PCG-64 DXSM with seed=42 by default).

    Args:
        n: Number of curves (observations).
        m: Number of evaluation points per curve.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with columns: curve_id, t_0, t_1, ..., t_{m-1}.
        Values written with float_format="%.17g" to preserve full float64 precision.
    """
    rng = np.random.default_rng(seed)
    argvals = np.linspace(0, 1, m)

    # Random curve parameters — drawn once in this order for reproducibility
    amplitudes = rng.uniform(0.5, 2.0, size=n)
    phases = rng.uniform(0, 2 * np.pi, size=n)
    noise_scale = rng.uniform(0.0, 0.2, size=n)

    curves = np.array([
        amplitudes[i] * np.sin(2 * np.pi * argvals + phases[i])
        + noise_scale[i] * rng.standard_normal(m)
        for i in range(n)
    ])

    df = pd.DataFrame(curves, columns=[f"t_{j}" for j in range(m)])
    df.insert(0, "curve_id", range(n))
    return df


def write_curves(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Write curves to CSV with full float64 precision."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.17g")


def csv_sha256(path: pathlib.Path) -> str:
    """Return the SHA-256 hex digest of the CSV file bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seeded benchmark CSV data")
    parser.add_argument("--n", type=int, default=60, help="Number of curves")
    parser.add_argument("--m", type=int, default=50, help="Number of evaluation points")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: benchmarks/data/ relative to script)")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent
    data_dir = pathlib.Path(args.out_dir) if args.out_dir else script_dir / "data"
    out_path = data_dir / f"curves_n{args.n}_m{args.m}.csv"

    df = generate_curves(n=args.n, m=args.m, seed=args.seed)
    write_curves(df, out_path)

    digest = csv_sha256(out_path)
    print(f"Generated: {out_path} ({args.n} curves, {args.m} eval points, seed={args.seed})")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
