"""
Python benchmark runner for the fdars cross-language harness.

Reads the seeded CSV from generate_data.py, runs scikit-fda FPCA with n_components=5,
and emits timing + result_fingerprint to benchmarks/results/python.json.

Usage:
    python python_runner.py --op fpca [--smoke] [--n 60] [--m 50] [--seed 42]
    python python_runner.py --op fpca --smoke
"""

import argparse
import json
import pathlib
import timeit

import numpy as np
import pandas as pd

# --- scikit-fda imports (inside functions so import errors are descriptive) ---


def _import_skfda():
    try:
        from skfda import FDataGrid
        from skfda.preprocessing.dim_reduction import FPCA
        return FDataGrid, FPCA
    except ImportError as e:
        raise ImportError(
            "scikit-fda is not installed. Activate the venv: "
            "source benchmarks/.venv/bin/activate"
        ) from e


def load_csv(csv_path: pathlib.Path) -> tuple[np.ndarray, int, int]:
    """Load curve CSV produced by generate_data.py. Returns (matrix, n, m)."""
    df = pd.read_csv(csv_path)
    # First column is curve_id — drop it
    mat = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    n, m = mat.shape
    return mat, n, m


def run_fpca(
    csv_path: pathlib.Path,
    n_components: int = 5,
    n_repeats: int = 50,
    smoke: bool = False,
) -> dict:
    """
    Time scikit-fda FPCA on the curves in csv_path.

    Returns a record dict with keys: op, library, n, m, mean_ns, result_fingerprint.
    result_fingerprint carries {'loadings': [[...], ...]} with shape (n_components, m)
    (loadings matrix, one row per component — transposed from the (m, ncomp) convention
    so it serialises cleanly as a list-of-rows).
    """
    FDataGrid, FPCA = _import_skfda()

    mat, n, m = load_csv(csv_path)
    argvals = np.linspace(0, 1, m)

    # Build FDataGrid once outside the timing loop
    fd = FDataGrid(data_matrix=mat, grid_points=[argvals])

    fpca_estimator = FPCA(n_components=n_components)

    if smoke:
        # Smoke run: single iteration to verify correctness
        fpca_estimator.fit(fd)
        n_repeats = 1

    # Warm-up pass (not timed)
    fpca_estimator.fit(fd)
    result_obj = fpca_estimator  # keep fitted result for fingerprint

    if not smoke:
        # Time with timeit for consistency (wraps repeated fits)
        _fd = fd
        _FPCA = FPCA
        _ncomp = n_components

        def _run():
            est = _FPCA(n_components=_ncomp)
            est.fit(_fd)
            return est

        total_s = timeit.timeit(lambda: _run(), number=n_repeats)
        mean_ns = (total_s / n_repeats) * 1e9
        # Re-fit for fingerprint (ensure it uses fresh estimator)
        result_obj = _run()
    else:
        mean_ns = 0.0

    # Extract loadings: FDataGrid data_matrix is (n_components, m, 1)
    # components_.data_matrix shape is (n_components, n_eval_points, 1)
    raw = fpca_estimator.components_.data_matrix
    # Shape: (n_components, m, 1) → squeeze to (n_components, m)
    loadings_ncomp_m = raw[:, :, 0]  # shape (n_components, m)

    # Convert to list-of-lists for JSON serialisation
    loadings_list = loadings_ncomp_m.tolist()

    return {
        "op": "fpca",
        "library": "python",
        "n": n,
        "m": m,
        "mean_ns": mean_ns,
        "result_fingerprint": {
            "loadings": loadings_list,  # n_components rows, each of length m
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Python scikit-fda benchmark runner")
    parser.add_argument("--op", required=True, choices=["fpca"],
                        help="Operation to benchmark")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: verify import and one pass only")
    parser.add_argument("--n", type=int, default=60, help="Number of curves")
    parser.add_argument("--m", type=int, default=50, help="Number of eval points")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n-repeats", type=int, default=50,
                        help="Number of timing repetitions")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent
    csv_path = script_dir / "data" / f"curves_n{args.n}_m{args.m}.csv"
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "python.json"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. "
            "Run: python generate_data.py first."
        )

    if args.op == "fpca":
        record = run_fpca(csv_path, n_components=5, n_repeats=args.n_repeats,
                          smoke=args.smoke)
    else:
        raise ValueError(f"Unknown op: {args.op}")

    with out_path.open("w") as f:
        json.dump([record], f, indent=2)

    print(f"Python runner [{args.op}]: mean_ns={record['mean_ns']:.1f}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
