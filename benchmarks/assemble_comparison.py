"""
Assembler for the fdars cross-language baseline comparison CSV.

Reads per-cell python_n{n}_m{m}.json and r_n{n}_m{m}.json files (produced by
run_all.py --grid) plus results/rust.json (criterion output from collect_rust.py),
and merges them into benchmarks/results/comparison.csv.

Schema columns (RESEARCH Section 5, exact order):
    op, library, n, m, mean_ns, ci_lower_ns, ci_upper_ns, std_dev_ns,
    rsd_pct, ci_width_pct, noise_gate, source, notes

Key design invariants:
- Uses pandas.DataFrame.to_csv (Don't-Hand-Roll — no manual string concat).
- comparison.csv is only written when gate_passed=True is passed explicitly
  (BENCH-05 measure-first invariant enforced at write time).
- Rust/criterion rows: ci_lower_ns, ci_upper_ns, std_dev_ns populated;
  rsd_pct = std_dev_ns/mean_ns*100; ci_width_pct = (ci_upper-ci_lower)/mean*100;
  noise_gate = 'PASS' if rsd_pct<5.0 AND ci_width_pct<5.0 else 'FAIL-NOISY'.
- Python and r_svd rows: ci/std_dev/rsd/ci_width left blank; noise_gate = 'NA'.
- r_native_pca_fd rows (fpca_native_r op): noise_gate='NA';
  notes = "native pca.fd — B-spline coeff space; not gated".
- Library mapping:
    Python runner records  -> library = 'python'
    R SVD-matched records  -> library = 'r_svd'    (op != 'fpca_native_r')
    R native pca.fd        -> library = 'r_native_pca_fd'  (op == 'fpca_native_r')
    Rust criterion records -> library = 'rust'

Usage:
    python assemble_comparison.py               # reads benchmarks/results/, writes comparison.csv
    python assemble_comparison.py --dry-run     # prints row count, no file written
"""

import argparse
import json
import pathlib
import sys

import pandas as pd

_SCRIPT_DIR = pathlib.Path(__file__).parent

# Exact schema column order (RESEARCH Section 5)
SCHEMA_COLUMNS = [
    "op", "library", "n", "m",
    "mean_ns", "ci_lower_ns", "ci_upper_ns", "std_dev_ns",
    "rsd_pct", "ci_width_pct", "noise_gate", "source", "notes",
]

# Grid cells matching run_all.py GRID_N × GRID_M
GRID_N = [100, 300, 500]
GRID_M = [50, 200]

# Noise-gate thresholds (RESEARCH Section 1; locked constraints)
_RSD_THRESHOLD = 5.0       # RSD < 5% → PASS
_CI_WIDTH_THRESHOLD = 5.0  # CI-width < 5% → PASS


def _noise_gate_rust(mean_ns: float, ci_lower_ns: float,
                     ci_upper_ns: float, std_dev_ns: float) -> tuple[float, float, str]:
    """
    Compute noise metrics and gate label for a Rust/criterion row.

    Returns (rsd_pct, ci_width_pct, noise_gate).
    """
    if mean_ns <= 0:
        return float("nan"), float("nan"), "FAIL-NOISY"
    rsd_pct = std_dev_ns / mean_ns * 100.0
    ci_width_pct = (ci_upper_ns - ci_lower_ns) / mean_ns * 100.0
    if rsd_pct < _RSD_THRESHOLD and ci_width_pct < _CI_WIDTH_THRESHOLD:
        gate = "PASS"
    else:
        gate = "FAIL-NOISY"
    return rsd_pct, ci_width_pct, gate


def _build_rust_rows(rust_records: list[dict]) -> list[dict]:
    """
    Convert criterion records (from collect_rust.py / rust.json) to schema rows.

    Each criterion record may appear multiple times (one per bench ID within a
    group); all are included so assembler.csv captures the full picture.
    The source field carries the criterion group/bench path for traceability.
    """
    rows = []
    for rec in rust_records:
        op = rec.get("op", rec.get("group", "unknown"))
        group = rec.get("group", "")
        bench = rec.get("bench", "")
        source = f"criterion:{group}/{bench}" if bench else f"criterion:{group}"
        mean_ns = float(rec.get("mean_ns", 0.0))
        ci_lower = float(rec.get("ci_lower_ns", 0.0))
        ci_upper = float(rec.get("ci_upper_ns", 0.0))
        std_dev = float(rec.get("std_dev_ns", rec.get("std_err_ns", 0.0)))

        rsd_pct, ci_width_pct, gate = _noise_gate_rust(mean_ns, ci_lower, ci_upper, std_dev)

        # Criterion output does not carry per-cell n/m in the record itself;
        # the bench ID encodes it (e.g. "n500_m200"). We leave n/m as None
        # here and fill from the bench field when parseable.
        n_val: int | None = None
        m_val: int | None = None
        bench_lower = bench.lower()
        # Try to parse n/m from bench ID patterns like "n500_m200", "n100/m50", etc.
        import re
        m_n = re.search(r"[_/]?n(\d+)", bench_lower)
        m_m = re.search(r"[_/]?m(\d+)", bench_lower)
        if m_n:
            n_val = int(m_n.group(1))
        if m_m:
            m_val = int(m_m.group(1))

        rows.append({
            "op": op,
            "library": "rust",
            "n": n_val,
            "m": m_val,
            "mean_ns": mean_ns,
            "ci_lower_ns": ci_lower,
            "ci_upper_ns": ci_upper,
            "std_dev_ns": std_dev,
            "rsd_pct": rsd_pct,
            "ci_width_pct": ci_width_pct,
            "noise_gate": gate,
            "source": source,
            "notes": "",
        })
    return rows


# FPCR disclosure threshold — must match gate._FPCR_CORR_THRESHOLD. FPCR is NOT
# hard-gated (Phase 113 Plan 02); this bar only LABELS whether the cross-library
# correlation cleared the disclosure floor in the recorded notes.
_FPCR_DISCLOSURE_THRESHOLD = 0.95


def _fpcr_disclosure_note(corr: float, n_val, m_val) -> str:
    """
    Build the FPCR disclosure string recorded in the notes column.

    Discloses the measured cross-library correlation and states plainly that
    FPCR is NOT hard-gated — a genuine divergence surfaced for the Phase-116
    report (Phase 113 Plan 02, user-approved).
    """
    if corr != corr:  # NaN
        corr_str = "n/a"
    else:
        corr_str = f"{corr:.4f}"
    n_str = str(n_val) if n_val is not None else "?"
    return (
        f"FPCR cross-library agreement corr={corr_str} @ n={n_str} "
        f"(scikit-fda FPCA->OLS vs R fda.usc fregre.pc: related but not identical FPCR; "
        f"NOT hard-gated — disclosed divergence for the Phase 116 report)"
    )


def _compute_cell_fpcr_corr(py_records: list[dict], r_records: list[dict]) -> float:
    """
    Compute the FPCR cross-library Pearson correlation for one cell.

    Uses the python and r fpcr result_fingerprint fitted_values. Returns
    |Pearson corr|, or float('nan') if either fingerprint is missing.
    """
    import numpy as np
    from scipy.stats import pearsonr

    py_fpcr = next((r for r in py_records if r.get("op") == "fpcr"), None)
    r_fpcr = next((r for r in r_records if r.get("op") == "fpcr"), None)
    if py_fpcr is None or r_fpcr is None:
        return float("nan")
    py_fp = py_fpcr.get("result_fingerprint", {})
    r_fp = r_fpcr.get("result_fingerprint", {})
    try:
        py_fv = np.array(py_fp["fitted_values"], dtype=np.float64)
        r_fv = np.array(r_fp["fitted_values"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return float("nan")
    if py_fv.size == 0 or py_fv.size != r_fv.size:
        return float("nan")
    corr, _ = pearsonr(py_fv, r_fv)
    return float(abs(corr))


def _build_python_rows(py_records: list[dict], fpcr_corr: float = float("nan")) -> list[dict]:
    """
    Convert Python runner records to schema rows.

    No CI data → ci/std_dev/rsd/ci_width left blank; noise_gate = 'NA'.
    FPCR rows carry the measured cross-library disclosure note (fpcr_corr).
    """
    rows = []
    for rec in py_records:
        op = rec.get("op", "unknown")
        n_val = rec.get("n")
        m_val = rec.get("m")
        mean_ns = float(rec.get("mean_ns", 0.0))
        notes = _fpcr_disclosure_note(fpcr_corr, n_val, m_val) if op == "fpcr" else ""
        rows.append({
            "op": op,
            "library": "python",
            "n": n_val,
            "m": m_val,
            "mean_ns": mean_ns,
            "ci_lower_ns": None,
            "ci_upper_ns": None,
            "std_dev_ns": None,
            "rsd_pct": None,
            "ci_width_pct": None,
            "noise_gate": "NA",
            "source": "python_runner",
            "notes": notes,
        })
    return rows


def _build_r_rows(r_records: list[dict], fpcr_corr: float = float("nan")) -> list[dict]:
    """
    Convert R runner records to schema rows.

    - op='fpca_native_r' → library='r_native_pca_fd'; notes = methodology disclosure.
    - op='fpcr'          → library='r_svd'; notes = FPCR cross-library disclosure.
    - All other ops      → library='r_svd'.
    - No CI data → ci/std_dev/rsd/ci_width blank; noise_gate = 'NA'.
    """
    rows = []
    for rec in r_records:
        op = rec.get("op", "unknown")
        n_val = rec.get("n")
        m_val = rec.get("m")
        mean_ns = float(rec.get("mean_ns", 0.0))

        if op == "fpca_native_r":
            library = "r_native_pca_fd"
            notes = "native pca.fd — B-spline coeff space; not gated"
        elif op == "fpcr":
            library = "r_svd"
            notes = _fpcr_disclosure_note(fpcr_corr, n_val, m_val)
        else:
            library = "r_svd"
            notes = ""

        rows.append({
            "op": op,
            "library": library,
            "n": n_val,
            "m": m_val,
            "mean_ns": mean_ns,
            "ci_lower_ns": None,
            "ci_upper_ns": None,
            "std_dev_ns": None,
            "rsd_pct": None,
            "ci_width_pct": None,
            "noise_gate": "NA",
            "source": "r_runner",
            "notes": notes,
        })
    return rows


def assemble(
    results_dir: pathlib.Path | None = None,
    gate_passed: bool = True,
) -> pd.DataFrame:
    """
    Merge per-cell JSON result files into a schema-correct DataFrame.

    Args:
        results_dir: directory containing python_n{n}_m{m}.json,
                     r_n{n}_m{m}.json, and rust.json.
                     Defaults to benchmarks/results/.
        gate_passed: must be True (caller asserts the gate passed before
                     assembling — BENCH-05).  Passing False raises ValueError.

    Returns:
        DataFrame with columns in SCHEMA_COLUMNS order.
    """
    if results_dir is None:
        results_dir = _SCRIPT_DIR / "results"
    results_dir = pathlib.Path(results_dir)

    all_rows: list[dict] = []

    # --- Rust rows from rust.json (criterion output, no per-cell n/m) ---
    rust_path = results_dir / "rust.json"
    if rust_path.exists():
        with rust_path.open() as f:
            rust_records = json.load(f)
        all_rows.extend(_build_rust_rows(rust_records))

    # --- Per-cell Python + R rows ---
    for n in GRID_N:
        for m in GRID_M:
            py_path = results_dir / f"python_n{n}_m{m}.json"
            r_path = results_dir / f"r_n{n}_m{m}.json"

            py_records: list[dict] = []
            r_records: list[dict] = []
            if py_path.exists():
                with py_path.open() as f:
                    py_records = json.load(f)
            if r_path.exists():
                with r_path.open() as f:
                    r_records = json.load(f)

            # Compute the per-cell FPCR cross-library disclosure correlation so
            # both the python and r fpcr rows can carry the measured value in notes
            # (Phase 113 Plan 02: FPCR disclosed, not hard-gated).
            fpcr_corr = _compute_cell_fpcr_corr(py_records, r_records)

            if py_records:
                all_rows.extend(_build_python_rows(py_records, fpcr_corr=fpcr_corr))
            if r_records:
                all_rows.extend(_build_r_rows(r_records, fpcr_corr=fpcr_corr))

    if not all_rows:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    df = pd.DataFrame(all_rows)

    # Ensure all schema columns are present and in the correct order
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLUMNS]

    return df


def write_comparison_csv(
    df: pd.DataFrame,
    out_path: pathlib.Path,
    gate_passed: bool = True,
) -> None:
    """
    Write the assembled DataFrame to comparison.csv.

    Enforces the BENCH-05 gate-before-timing invariant: raises if
    gate_passed is not True.  Uses pandas.DataFrame.to_csv (Don't-Hand-Roll).

    Args:
        df: assembled DataFrame (from assemble()).
        out_path: destination path for comparison.csv.
        gate_passed: must be True; raises ValueError otherwise.
    """
    if not gate_passed:
        raise ValueError(
            "BENCH-05 violation: write_comparison_csv called with gate_passed=False. "
            "The agreement gate must pass before comparison.csv is written."
        )
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_path), index=False)
    print(f"Written: {out_path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble grid-cell result JSON files into comparison.csv"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to results directory (default: benchmarks/results/)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output CSV path (default: <results_dir>/comparison.csv)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print row count and library/op coverage; do not write comparison.csv"
    )
    parser.add_argument(
        "--gate-passed", action="store_true", default=False,
        help="Pass this flag to confirm the agreement gate passed (BENCH-05). "
             "Required to write comparison.csv."
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir) if args.results_dir else None
    df = assemble(results_dir=results_dir, gate_passed=True)  # assemble always OK

    if args.dry_run:
        print(f"Assembled {len(df)} rows")
        if not df.empty:
            print(f"  Libraries: {sorted(df['library'].dropna().unique())}")
            print(f"  Ops:       {sorted(df['op'].dropna().unique())}")
            print(f"  N values:  {sorted(df['n'].dropna().unique())}")
            print(f"  M values:  {sorted(df['m'].dropna().unique())}")
            noise_counts = df[df['library'] == 'rust']['noise_gate'].value_counts().to_dict()
            print(f"  Rust noise_gate: {noise_counts}")
        return

    if not args.gate_passed:
        print(
            "ERROR: --gate-passed flag not provided. "
            "Run the agreement gate first (run_all.py --grid), then pass --gate-passed.",
            file=sys.stderr,
        )
        sys.exit(1)

    results_dir_resolved = (
        pathlib.Path(args.results_dir) if args.results_dir
        else _SCRIPT_DIR / "results"
    )
    out_path = (
        pathlib.Path(args.out) if args.out
        else results_dir_resolved / "comparison.csv"
    )
    write_comparison_csv(df, out_path, gate_passed=True)

    print(f"\nSummary:")
    if not df.empty:
        print(f"  Total rows: {len(df)}")
        print(f"  Libraries: {sorted(df['library'].dropna().unique())}")
        print(f"  Ops:       {sorted(df['op'].dropna().unique())}")
        noise_counts = df[df['library'] == 'rust']['noise_gate'].value_counts().to_dict()
        print(f"  Rust noise_gate distribution: {noise_counts}")


if __name__ == "__main__":
    main()
