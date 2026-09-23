"""
Python benchmark runner for the fdars cross-language harness.

Reads the seeded CSV from generate_data.py, runs scikit-fda operations, and
emits timing + result_fingerprint to benchmarks/results/python.json.

Supported ops (--op):
  fpca            - FPCA with n_components=5
  basis_smoothing - B-spline basis smoothing with lambda=0 (interpolation)
  depth           - Functional depth via ModifiedBandDepth (deterministic)
  functional_boxplot - Functional boxplot via Boxplot+ModifiedBandDepth
  fpcr            - FPCR regression: FPCA scores -> OLS on seeded y
  elastic         - Elastic alignment / Karcher mean via FisherRaoElasticRegistration

Elastic class (scikit-fda 0.10.1):
  skfda.preprocessing.registration.FisherRaoElasticRegistration (penalty=0.0)
  Verified at runtime against installed 0.10.1 package.

Usage:
    python python_runner.py --op fpca [--smoke] [--n 60] [--m 50] [--seed 42]
    python python_runner.py --op basis_smoothing --smoke
    python python_runner.py --smoke  # runs all ops once each
"""

import argparse
import json
import pathlib
import timeit

import numpy as np
import pandas as pd

# --- scikit-fda imports (inside functions so import errors are descriptive) ---

ALL_OPS = ["fpca", "basis_smoothing", "depth", "functional_boxplot", "fpcr", "elastic"]

SCRIPT_DIR = pathlib.Path(__file__).parent


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


def load_response(y_path: pathlib.Path) -> np.ndarray:
    """Load scalar response vector produced by generate_data.py. Returns (n,) array."""
    df = pd.read_csv(y_path)
    return df["y"].to_numpy(dtype=np.float64)


def _time_fn(fn, n_repeats: int, smoke: bool) -> tuple[float, object]:
    """Run fn() n_repeats times (or once if smoke), return (mean_ns, result)."""
    result = fn()
    if smoke:
        return 0.0, result
    # Warm-up already done via fn() above
    total_s = timeit.timeit(fn, number=n_repeats)
    mean_ns = (total_s / n_repeats) * 1e9
    result = fn()  # re-run for a fresh result object
    return mean_ns, result


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
    _FPCA = FPCA
    _ncomp = n_components

    def _run():
        est = _FPCA(n_components=_ncomp)
        est.fit(fd)
        return est

    mean_ns, result_obj = _time_fn(_run, n_repeats, smoke)

    # Extract loadings: FDataGrid data_matrix is (n_components, m, 1)
    # components_.data_matrix shape is (n_components, n_eval_points, 1)
    raw = result_obj.components_.data_matrix
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


def run_basis_smoothing(
    csv_path: pathlib.Path,
    n_basis: int = 10,
    n_repeats: int = 50,
    smoke: bool = False,
) -> dict:
    """
    Time scikit-fda B-spline basis smoothing (lambda=0 = interpolation).

    Uses BSplineBasis + BasisSmoother (lambda=0, no penalty) so it is
    comparable to the R side (smooth.basis with lambda=0). The fingerprint
    is the smoothed curve values on the evaluation grid for all n curves.
    """
    FDataGrid, _ = _import_skfda()
    from skfda.representation.basis import BSplineBasis
    from skfda.preprocessing.smoothing import BasisSmoother

    mat, n, m = load_csv(csv_path)
    argvals = np.linspace(0, 1, m)

    fd = FDataGrid(data_matrix=mat, grid_points=[argvals])
    basis = BSplineBasis(domain_range=(0.0, 1.0), n_basis=n_basis, order=4)
    # smoothing_parameter=0 → pure basis interpolation (no penalty)
    smoother = BasisSmoother(basis=basis, smoothing_parameter=0.0)

    def _run():
        return smoother.fit_transform(fd)

    mean_ns, result_fd = _time_fn(_run, n_repeats, smoke)

    # Evaluate the smoothed functions on the original grid for fingerprint
    smoothed_vals = result_fd.data_matrix[:, :, 0]  # shape (n, m)

    return {
        "op": "basis_smoothing",
        "library": "python",
        "n": n,
        "m": m,
        "mean_ns": mean_ns,
        "result_fingerprint": {
            "smoothed_values": smoothed_vals.tolist(),  # n rows, each length m
        },
    }


def run_depth(
    csv_path: pathlib.Path,
    n_repeats: int = 50,
    smoke: bool = False,
) -> dict:
    """
    Time scikit-fda functional depth via ModifiedBandDepth (deterministic).

    ModifiedBandDepth is chosen over RandomProjectionDepth for the gated
    path because it is deterministic (per RESEARCH Pitfall 5), giving a
    reproducible result_fingerprint that the Plan-03 gate can compare.
    """
    FDataGrid, _ = _import_skfda()
    from skfda.exploratory.depth import ModifiedBandDepth

    mat, n, m = load_csv(csv_path)
    argvals = np.linspace(0, 1, m)
    fd = FDataGrid(data_matrix=mat, grid_points=[argvals])

    depth_fn = ModifiedBandDepth()

    def _run():
        return depth_fn(fd)

    mean_ns, depths = _time_fn(_run, n_repeats, smoke)

    return {
        "op": "depth",
        "library": "python",
        "n": n,
        "m": m,
        "mean_ns": mean_ns,
        "result_fingerprint": {
            "depths": depths.tolist(),  # length-n depth vector
        },
    }


def run_functional_boxplot(
    csv_path: pathlib.Path,
    n_repeats: int = 50,
    smoke: bool = False,
) -> dict:
    """
    Time scikit-fda functional boxplot via Boxplot + ModifiedBandDepth.

    The fingerprint is the MBD depth vector (matches the R side which uses
    roahd::fbplot(Depths="MBD") and returns the Depth vector).

    scikit-fda 0.10.1 API:
        skfda.exploratory.visualization.Boxplot (with depth_method=ModifiedBandDepth())
        skfda.exploratory.depth.ModifiedBandDepth
    """
    FDataGrid, _ = _import_skfda()
    from skfda.exploratory.depth import ModifiedBandDepth
    from skfda.exploratory.visualization import Boxplot

    mat, n, m = load_csv(csv_path)
    argvals = np.linspace(0, 1, m)
    fd = FDataGrid(data_matrix=mat, grid_points=[argvals])

    depth_fn = ModifiedBandDepth()

    def _run():
        # WR-02: previous version called depth_fn(fd) separately after Boxplot(),
        # causing MBD to be computed twice per timing iteration (~2x inflation vs R).
        # Boxplot() already calls depth_method internally; we time only the Boxplot
        # construction (single MBD call), matching R's single fbplot() call.
        bp = Boxplot(fd, depth_method=depth_fn, factor=1.5)
        return bp

    mean_ns, bp = _time_fn(_run, n_repeats, smoke)
    # Compute depth outside the timing loop for the fingerprint only (one extra call,
    # not counted in mean_ns).
    depths = depth_fn(fd)

    return {
        "op": "functional_boxplot",
        "library": "python",
        "n": n,
        "m": m,
        "mean_ns": mean_ns,
        "result_fingerprint": {
            # Depth vector (MBD) — same quantity as roahd::fbplot()$Depth on the R side
            "depths": depths.tolist(),
        },
    }


def run_fpcr(
    csv_path: pathlib.Path,
    y_path: pathlib.Path,
    n_components: int = 5,
    n_repeats: int = 50,
    smoke: bool = False,
) -> dict:
    """
    Time FPCR regression: FPCA scores -> OLS on the seeded response vector.

    Approach: compute FPCA (5 components), project curves to scores, fit OLS
    via numpy.linalg.lstsq. This matches fda.usc::fregre.pc on the R side
    (FPCA scores -> OLS).
    """
    FDataGrid, _ = _import_skfda()
    from skfda.preprocessing.dim_reduction import FPCA

    mat, n, m = load_csv(csv_path)
    y = load_response(y_path)
    argvals = np.linspace(0, 1, m)
    fd = FDataGrid(data_matrix=mat, grid_points=[argvals])

    def _run():
        fpca = FPCA(n_components=n_components)
        fpca.fit(fd)
        scores = fpca.transform(fd)   # shape (n, n_components)
        # OLS: add intercept column
        X = np.column_stack([np.ones(n), scores])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ coeffs
        return fitted

    mean_ns, fitted = _time_fn(_run, n_repeats, smoke)

    return {
        "op": "fpcr",
        "library": "python",
        "n": n,
        "m": m,
        "mean_ns": mean_ns,
        "result_fingerprint": {
            "fitted_values": fitted.tolist(),  # length-n fitted response vector
        },
    }


def run_elastic(
    csv_path: pathlib.Path,
    elastic_csv_path: pathlib.Path,
    n_repeats: int = 10,
    smoke: bool = False,
    pairwise_only: bool = False,
) -> dict:
    """
    Time elastic alignment via FisherRaoElasticRegistration.

    Sub-operations:
    1. Pairwise alignment: align curve 1 to curve 0 using FisherRaoElasticRegistration
       with penalty=0.0 (no warp penalty = pure Fisher-Rao metric). N-independent, fast.
    2. Karcher mean: compute the Fréchet mean of all n curves via
       fisher_rao_karcher_mean. N-dependent, slow (~minutes at m=200).

    Phase 113 Plan 02 (user-approved 2026-09-23): when `pairwise_only=True`
    (grid baseline mode), ONLY the fast, N-independent pairwise alignment is
    TIMED — the slow Karcher mean is EXCLUDED from the grid timing (it is
    noise-gated out of the ranking at RSD 29–46% and takes ~10 min/cell at m=200).
    The Karcher-mean fingerprint is still computed ONCE (outside the timed loop)
    so the agreement gate's secondary karcher check remains available; only its
    TIMING is dropped. When `pairwise_only=False`, the legacy behavior (Karcher
    inside the timed loop) is preserved for the single-cell path.

    Elastic class (scikit-fda 0.10.1):
        skfda.preprocessing.registration.FisherRaoElasticRegistration (penalty=0.0)
    Karcher mean function: skfda.exploratory.stats.fisher_rao_karcher_mean

    Fingerprint: warping function values from pairwise alignment (length m),
                 plus first curve values of the Karcher mean (length m).
    """
    FDataGrid, _ = _import_skfda()
    from skfda.preprocessing.registration import FisherRaoElasticRegistration
    from skfda.exploratory.stats import fisher_rao_karcher_mean

    # Load full curve set for Karcher mean
    mat, n, m = load_csv(csv_path)
    argvals = np.linspace(0, 1, m)
    fd_all = FDataGrid(data_matrix=mat, grid_points=[argvals])

    # Load 2-curve pair for pairwise alignment
    mat_pair, n_pair, m_pair = load_csv(elastic_csv_path)
    assert n_pair == 2 and m_pair == m, f"Expected 2x{m} elastic fixture, got {n_pair}x{m_pair}"
    fd_ref = FDataGrid(data_matrix=mat_pair[[0]], grid_points=[argvals])   # reference (template)
    fd_target = FDataGrid(data_matrix=mat_pair[[1]], grid_points=[argvals])  # curve to align

    if pairwise_only:
        # Grid baseline: TIME only the pairwise alignment (N-independent).
        def _run_pairwise():
            reg = FisherRaoElasticRegistration(template=fd_ref, penalty=0.0)
            reg.fit_transform(fd_target)
            return reg.warping_.data_matrix[0, :, 0]  # shape (m,)

        mean_ns, warping_vals = _time_fn(_run_pairwise, n_repeats, smoke)

        # Compute the Karcher fingerprint ONCE, untimed, so the gate's secondary
        # karcher check still has data (its timing is intentionally excluded).
        karcher = fisher_rao_karcher_mean(fd_all)
        karcher_vals = karcher.data_matrix[0, :, 0]

        return {
            "op": "elastic",
            "library": "python",
            "n": n,
            "m": m,
            "mean_ns": mean_ns,
            "elastic_timing_scope": "pairwise_only",
            "result_fingerprint": {
                "warping_values": warping_vals.tolist(),
                "karcher_mean": karcher_vals.tolist(),
            },
        }

    # Legacy single-cell path: time pairwise + Karcher together.
    def _run():
        reg = FisherRaoElasticRegistration(template=fd_ref, penalty=0.0)
        reg.fit_transform(fd_target)
        warping_vals = reg.warping_.data_matrix[0, :, 0]  # shape (m,)
        karcher = fisher_rao_karcher_mean(fd_all)
        karcher_vals = karcher.data_matrix[0, :, 0]    # shape (m,)
        return warping_vals, karcher_vals

    mean_ns, (warping_vals, karcher_vals) = _time_fn(_run, n_repeats, smoke)

    return {
        "op": "elastic",
        "library": "python",
        "n": n,
        "m": m,
        "mean_ns": mean_ns,
        "elastic_timing_scope": "pairwise_plus_karcher",
        "result_fingerprint": {
            "warping_values": warping_vals.tolist(),
            "karcher_mean": karcher_vals.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Python scikit-fda benchmark runner")
    parser.add_argument(
        "--op",
        choices=ALL_OPS,
        default=None,
        help="Operation to benchmark (omit to run all ops in --smoke mode)",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: verify import and one pass only")
    parser.add_argument("--n", type=int, default=60, help="Number of curves")
    parser.add_argument("--m", type=int, default=50, help="Number of eval points")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n-repeats", type=int, default=50,
                        help="Number of timing repetitions")
    parser.add_argument("--pairwise-only", action="store_true",
                        help="Elastic: time ONLY the fast pairwise alignment, "
                             "exclude the slow N-dependent Karcher mean from timing "
                             "(grid baseline mode, Phase 113 Plan 02)")
    args = parser.parse_args()

    # When --smoke is passed without --op, run all ops
    if args.op is None:
        if not args.smoke:
            parser.error("--op is required unless --smoke is used (which runs all ops)")
        ops_to_run = ALL_OPS
    else:
        ops_to_run = [args.op]

    data_dir = SCRIPT_DIR / "data"
    csv_path = data_dir / f"curves_n{args.n}_m{args.m}.csv"
    y_path = data_dir / f"y_n{args.n}.csv"
    elastic_csv_path = data_dir / f"elastic_n2_m{args.m}.csv"

    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "python.json"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. "
            "Run: python generate_data.py first."
        )
    if "fpcr" in ops_to_run and not y_path.exists():
        raise FileNotFoundError(
            f"Response CSV not found: {y_path}. "
            "Run: python generate_data.py first."
        )
    if "elastic" in ops_to_run and not elastic_csv_path.exists():
        raise FileNotFoundError(
            f"Elastic pair CSV not found: {elastic_csv_path}. "
            "Run: python generate_data.py first."
        )

    records = []
    for op in ops_to_run:
        if op == "fpca":
            record = run_fpca(csv_path, n_components=5, n_repeats=args.n_repeats,
                              smoke=args.smoke)
        elif op == "basis_smoothing":
            record = run_basis_smoothing(csv_path, n_basis=10, n_repeats=args.n_repeats,
                                         smoke=args.smoke)
        elif op == "depth":
            record = run_depth(csv_path, n_repeats=args.n_repeats, smoke=args.smoke)
        elif op == "functional_boxplot":
            record = run_functional_boxplot(csv_path, n_repeats=args.n_repeats,
                                            smoke=args.smoke)
        elif op == "fpcr":
            record = run_fpcr(csv_path, y_path, n_components=5, n_repeats=args.n_repeats,
                               smoke=args.smoke)
        elif op == "elastic":
            record = run_elastic(csv_path, elastic_csv_path, n_repeats=args.n_repeats // 5,
                                 smoke=args.smoke, pairwise_only=args.pairwise_only)
        else:
            raise ValueError(f"Unknown op: {op}")

        records.append(record)
        print(f"Python runner [{op}]: mean_ns={record['mean_ns']:.1f}")

    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
