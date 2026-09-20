"""
Seeded curve-set generator for the fdars benchmark harness.

Produces a deterministic (n, m) matrix of sinusoidal curves with noise,
written as a full-precision CSV consumed bit-identically by Python, R, and
Rust runners.

Also emits:
  - y_n{n}.csv      : seeded scalar response vector for FPCR benchmarks
  - elastic_n2_m{m}.csv : two-curve CSV for elastic pairwise alignment benchmarks

Usage:
    python generate_data.py --n 60 --m 50 --seed 42
    python generate_data.py --n 100 --m 100 --seed 42
"""

import argparse
import hashlib
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


def generate_response(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate a deterministic scalar response vector for FPCR benchmarks.

    Uses seed+100 to ensure the response is independent of but reproducible
    alongside the curve data generated with the same seed.

    Args:
        n: Number of observations (must match the curve matrix row count).
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with a single column 'y' (n rows), written as y_n{n}.csv.
    """
    rng = np.random.default_rng(seed + 100)
    # Simple linear model with noise: y ~ N(0, 1), independent of curve phases
    y = rng.standard_normal(n)
    return pd.DataFrame({"y": y})


def generate_elastic_pair(m: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate a two-curve dataset for elastic pairwise alignment benchmarks.

    Produces curves f1 and f2 on [0, 1] with m evaluation points. f1 is a
    pure sinusoid; f2 is a phase-shifted version with slight amplitude
    variation, giving a non-trivial but deterministic alignment problem.

    Args:
        m: Number of evaluation points.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with columns: curve_id, t_0, ..., t_{m-1} (2 rows).
        Written as elastic_n2_m{m}.csv.
    """
    rng = np.random.default_rng(seed + 200)
    argvals = np.linspace(0, 1, m)

    phase_shift = rng.uniform(0.2, 0.5)  # deterministic phase shift
    amp2 = rng.uniform(0.8, 1.2)          # slight amplitude variation

    f1 = np.sin(2 * np.pi * argvals)
    f2 = amp2 * np.sin(2 * np.pi * argvals + phase_shift)

    curves = np.vstack([f1, f2])
    df = pd.DataFrame(curves, columns=[f"t_{j}" for j in range(m)])
    df.insert(0, "curve_id", [0, 1])
    return df


def write_csv(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Write a DataFrame to CSV with full float64 precision."""
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
    parser.add_argument("--m", type=int, default=50, help="Number of eval points")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: benchmarks/data/ relative to script)")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent
    data_dir = pathlib.Path(args.out_dir) if args.out_dir else script_dir / "data"

    # 1. Curve matrix
    curves_path = data_dir / f"curves_n{args.n}_m{args.m}.csv"
    df_curves = generate_curves(n=args.n, m=args.m, seed=args.seed)
    write_csv(df_curves, curves_path)
    print(f"Generated: {curves_path} ({args.n} curves, {args.m} eval points, seed={args.seed})")
    print(f"  SHA-256: {csv_sha256(curves_path)}")

    # 2. Scalar response vector (FPCR fixtures)
    y_path = data_dir / f"y_n{args.n}.csv"
    df_y = generate_response(n=args.n, seed=args.seed)
    write_csv(df_y, y_path)
    print(f"Generated: {y_path} ({args.n} response values, seed={args.seed}+100)")
    print(f"  SHA-256: {csv_sha256(y_path)}")

    # 3. Elastic pair fixture (2 curves, same m)
    elastic_path = data_dir / f"elastic_n2_m{args.m}.csv"
    df_elastic = generate_elastic_pair(m=args.m, seed=args.seed)
    write_csv(df_elastic, elastic_path)
    print(f"Generated: {elastic_path} (2-curve elastic fixture, m={args.m}, seed={args.seed}+200)")
    print(f"  SHA-256: {csv_sha256(elastic_path)}")


if __name__ == "__main__":
    main()
