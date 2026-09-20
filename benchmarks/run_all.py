"""
Master orchestrator for the fdars cross-language benchmark harness.

Sets OMP/OPENBLAS/MKL/RAYON/NUMEXPR/VECLIB thread env vars to "1" in the
subprocess environment before forking any runner. The measure-first ordering
(numeric agreement gate BEFORE timing comparison is written) is the
load-bearing invariant of BENCH-05.

Full-op orchestration sequence:
  1. Generate all shared CSV fixtures (seeded, bit-identical)
  2. Run Python runner (all ops or specified op)
  3. Run R runner (all ops or specified op)
  4. Run cargo bench (full bench suite or specified bench files) — skipped in --dry-run
  5. Collect criterion output into results/rust.json
  6. Capture environment metadata into results/env.json
  7. GATE: run full-op agreement gate BEFORE writing any timing comparison
     - For FPCA: compare fdars (via fpca_dump), Python, R
     - For all other ops: compare Python vs R (fdars fingerprints via fpca_dump
       extension for smoke runs; criterion output has timing only, not fingerprints)
     - Abort (exit non-zero) on any op mismatch; no timing recorded on failure
  8. Write merged per-op timing comparison (only on gate pass)

Usage:
    python run_all.py --op fpca [--dry-run] [--env-only]
    python run_all.py --env-only                   # capture env.json only
    python run_all.py --dry-run                    # all ops, no cargo bench
    python run_all.py                              # full run including cargo bench

All subprocess calls use list-form only (no shell=True) per T-112-02.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

# -------------------------------------------------------------------
# Thread-pinning environment (fairness contract, BENCH-04)
# All six thread env vars pinned to "1" before ANY subprocess fork.
# -------------------------------------------------------------------

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

_SCRIPT_DIR = pathlib.Path(__file__).parent
_WORKSPACE_ROOT = _SCRIPT_DIR.parent
_VENV_PYTHON = _SCRIPT_DIR / ".venv" / "bin" / "python"

# Fallback: if venv python not found, use the running interpreter
_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

ALL_OPS = ["fpca", "basis_smoothing", "depth", "functional_boxplot", "fpcr", "elastic"]

# Bench files that cover the overlapping op set
BENCH_FILES = [
    "regression_benchmarks",   # fregre_lm, functional_logistic
    "depth_benchmarks",        # fraiman_muniz, modified_band, functional_boxplot
    "alignment_benchmarks",    # elastic_align_pair, karcher_mean
    "basis_benchmarks",        # smooth_basis, bspline_basis
]


def _build_env() -> dict:
    """Build the pinned-thread environment dict (BENCH-04, Pitfall 1)."""
    return {**os.environ, **THREAD_ENV}


def _run(cmd: list[str], env: dict, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """
    Run a subprocess with the pinned-thread environment.

    Uses list-form only (no shell=True) per T-112-02.
    """
    return subprocess.run(cmd, env=env, check=check, **kwargs)


# -------------------------------------------------------------------
# Step functions
# -------------------------------------------------------------------


def step_generate(n: int, m: int, seed: int, env: dict) -> None:
    """Run the seeded CSV generator (all fixture types)."""
    print(f"[1] Generating CSV fixtures (n={n}, m={m}, seed={seed})...")
    _run(
        [_PYTHON, str(_SCRIPT_DIR / "generate_data.py"),
         "--n", str(n), "--m", str(m), "--seed", str(seed)],
        env=env,
    )


def step_python_runner(op: str | None, n: int, m: int, seed: int, env: dict,
                       smoke: bool = False) -> None:
    """Run the Python (scikit-fda) runner for one op or all ops."""
    if op is None:
        print(f"[2] Running Python runner (all ops, smoke={smoke})...")
        _run(
            [_PYTHON, str(_SCRIPT_DIR / "python_runner.py"),
             "--smoke" if smoke else "--op", "fpca",
             "--n", str(n), "--m", str(m), "--seed", str(seed)],
            env=env,
        ) if smoke else _run_all_ops_python(n, m, seed, env)
    else:
        print(f"[2] Running Python runner (op={op})...")
        _run(
            [_PYTHON, str(_SCRIPT_DIR / "python_runner.py"),
             "--op", op, "--n", str(n), "--m", str(m), "--seed", str(seed)],
            env=env,
        )


def _run_all_ops_python(n: int, m: int, seed: int, env: dict) -> None:
    """Run all ops via python_runner using --smoke to get fingerprints for all ops at once."""
    _run(
        [_PYTHON, str(_SCRIPT_DIR / "python_runner.py"),
         "--smoke",
         "--n", str(n), "--m", str(m), "--seed", str(seed)],
        env=env,
    )


def step_r_runner(op: str | None, n: int, m: int, seed: int, env: dict,
                  smoke: bool = False) -> None:
    """Run the R runner for one op or all ops."""
    if op is None:
        print(f"[3] Running R runner (all ops, smoke={smoke})...")
        _run(
            ["Rscript", str(_SCRIPT_DIR / "r_runner.R"),
             "--smoke",
             "--n", str(n), "--m", str(m), "--seed", str(seed)],
            env=env,
        )
    else:
        print(f"[3] Running R runner (op={op})...")
        _run(
            ["Rscript", str(_SCRIPT_DIR / "r_runner.R"),
             "--op", op, "--n", str(n), "--m", str(m), "--seed", str(seed)],
            env=env,
        )


def step_fdars_loadings(n: int, m: int, env: dict) -> list[list[float]]:
    """
    Run fpca_dump to obtain fdars FPCA loadings from the generated CSV.

    The fpca_dump binary is a standalone Rust crate in benchmarks/fpca_dump/
    that calls fdars-core fdata_to_pc and emits loadings as JSON.
    Returns the loadings as a list of ncomp rows (each of length m).
    """
    dump_bin = _SCRIPT_DIR / "fpca_dump" / "target" / "release" / "fpca_dump"
    if not dump_bin.exists():
        print("[fdars_dump] Building fpca_dump binary...")
        _run(
            ["cargo", "build", "--release"],
            env=env,
            cwd=str(_SCRIPT_DIR / "fpca_dump"),
        )

    csv_path = _SCRIPT_DIR / "data" / f"curves_n{n}_m{m}.csv"
    result = _run(
        [str(dump_bin), "--csv", str(csv_path), "--ncomp", "5"],
        env=env,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout.strip())
    return data["loadings"]  # list of ncomp rows, each of length m


def step_cargo_bench(bench_files: list[str], env: dict) -> None:
    """Run the full set of criterion bench files (skipped in --dry-run mode)."""
    for bench_name in bench_files:
        print(f"[4] Running cargo bench ({bench_name})...")
        _run(
            ["cargo", "bench",
             "-p", "fdars-core",
             "--features", "linalg,parallel",
             "--bench", bench_name],
            env=env,
            cwd=str(_WORKSPACE_ROOT),
        )


def step_collect_rust(env: dict) -> None:
    """Collect criterion output into results/rust.json."""
    print("[5] Collecting criterion output...")
    _run(
        [_PYTHON, str(_SCRIPT_DIR / "collect_rust.py")],
        env=env,
    )


def step_capture_env(env: dict) -> None:
    """Capture environment metadata to results/env.json."""
    print("[E] Capturing environment metadata...")
    _run(
        [_PYTHON, str(_SCRIPT_DIR / "capture_env.py")],
        env=env,
    )


def step_run_gate(n: int, m: int, env: dict,
                  py_path: pathlib.Path, r_path: pathlib.Path) -> bool:
    """
    Run the full-op numeric-agreement gate BEFORE recording any timing.

    Loads fingerprints from python.json and r.json (both runners). For FPCA,
    also loads fdars loadings via fpca_dump and checks fdars vs Python and
    fdars vs R. For all other ops, checks Python vs R.

    Returns True if all ops pass. Aborts (sys.exit 1) on gate failure so
    no timing comparison is written (BENCH-05 measure-first invariant).
    """
    import numpy as np
    sys.path.insert(0, str(_SCRIPT_DIR))
    from gate import run_gate, check_fpca_agreement, fingerprint_to_array

    with py_path.open() as f:
        py_records = json.load(f)
    with r_path.open() as f:
        r_records = json.load(f)

    # Step 6a: For FPCA, compare fdars vs Python and fdars vs R
    print("[6a] Agreement gate: fdars FPCA vs Python + R...")
    py_fpca = next((r for r in py_records if r["op"] == "fpca"), None)
    r_fpca = next((r for r in r_records if r["op"] == "fpca"), None)

    fpca_passed = True
    if py_fpca and r_fpca:
        fdars_rows = step_fdars_loadings(n=n, m=m, env=env)
        fdars_load = np.array(fdars_rows, dtype=np.float64).T  # (m, ncomp)

        p1, mc1, _ = check_fpca_agreement(fdars_load,
                                           fingerprint_to_array(py_fpca["result_fingerprint"]),
                                           threshold=0.99, label="fdars-vs-python")
        print(f"    fdars vs python: {'PASS' if p1 else 'FAIL'} (min |corr|={mc1:.6f})")

        p2, mc2, _ = check_fpca_agreement(fdars_load,
                                           fingerprint_to_array(r_fpca["result_fingerprint"]),
                                           threshold=0.99, label="fdars-vs-r")
        print(f"    fdars vs r:      {'PASS' if p2 else 'FAIL'} (min |corr|={mc2:.6f})")

        if not p1 or not p2:
            fpca_passed = False

    # Step 6b: Full-op gate — Python vs R for all gated ops
    print("[6b] Agreement gate: Python vs R (all overlapping ops)...")

    # Build a merged record set for run_gate:
    # Use python records as "rust" (fdars reference) and r records as "r"
    # so run_gate compares them. Then also add the real fdars FPCA record.
    gate_records = []

    # For each op, add python record as the "reference" (library=rust stand-in)
    # and r record as the comparison target
    py_by_op = {r["op"]: r for r in py_records}
    r_by_op = {r["op"]: r for r in r_records}

    for op in ["fpca", "basis_smoothing", "depth", "functional_boxplot", "fpcr", "elastic"]:
        py_rec = py_by_op.get(op)
        r_rec = r_by_op.get(op)
        if py_rec and r_rec:
            # Python acts as reference (library=rust) for the gate comparison
            ref_rec = dict(py_rec)
            ref_rec["library"] = "rust"
            gate_records.append(ref_rec)
            gate_records.append(r_rec)

    all_passed_py_r = run_gate(gate_records)

    gate_all_passed = fpca_passed and all_passed_py_r

    if not gate_all_passed:
        print(
            "\nGATE FAIL — aborting. No timing comparison written.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n[GATE PASS] All overlapping ops verified — proceeding to timing comparison.")
    return True


def write_timing_comparison(n: int, m: int) -> None:
    """
    Write the merged per-op timing comparison record.

    Only called AFTER the gate passes (BENCH-05 measure-first invariant).
    Records timing from all three libraries (python.json, r.json, rust.json)
    for every overlapping op.
    """
    py_path = _SCRIPT_DIR / "results" / "python.json"
    r_path = _SCRIPT_DIR / "results" / "r.json"
    rust_path = _SCRIPT_DIR / "results" / "rust.json"
    summary_path = _SCRIPT_DIR / "results" / "timing_comparison.json"

    with py_path.open() as f:
        py_records = json.load(f)
    with r_path.open() as f:
        r_records = json.load(f)

    rust_records = []
    if rust_path.exists():
        with rust_path.open() as f:
            rust_records = json.load(f)

    py_by_op = {r["op"]: r for r in py_records}
    r_by_op = {r["op"]: r for r in r_records}

    # Build timing summary per op
    timing_rows = []
    for op in ALL_OPS:
        py_rec = py_by_op.get(op, {})
        r_rec = r_by_op.get(op, {})
        # Get fdars timing: find the best matching criterion record
        # (prefer criterion groups that have already been run)
        rust_rec = next(
            (r for r in rust_records if r.get("op") == op),
            {}
        )
        timing_rows.append({
            "op": op,
            "n": n,
            "m": m,
            "gate_passed": True,
            "python_mean_ns": py_rec.get("mean_ns"),
            "r_mean_ns": r_rec.get("mean_ns"),
            "rust_criterion_group": rust_rec.get("group"),
            "rust_criterion_bench": rust_rec.get("bench"),
            "rust_mean_ns": rust_rec.get("mean_ns"),
        })

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(timing_rows, f, indent=2)

    print(f"\nTiming comparison written: {summary_path}")
    for row in timing_rows:
        py_t = f"{row['python_mean_ns']:.0f}" if row["python_mean_ns"] else "N/A"
        r_t = f"{row['r_mean_ns']:.0f}" if row["r_mean_ns"] else "N/A"
        ru_t = f"{row['rust_mean_ns']:.0f}" if row["rust_mean_ns"] else "N/A (run cargo bench)"
        print(f"  {row['op']}: python={py_t}ns  r={r_t}ns  rust={ru_t}ns")

    # Also write the legacy fpca_timing.json for backward compat
    fpca_row = next((r for r in timing_rows if r["op"] == "fpca"), None)
    if fpca_row:
        fpca_path = _SCRIPT_DIR / "results" / "fpca_timing.json"
        with fpca_path.open("w") as f:
            json.dump({
                "op": "fpca", "n": n, "m": m, "gate_passed": True,
                "timing_ns": {
                    "python": fpca_row["python_mean_ns"],
                    "r": fpca_row["r_mean_ns"],
                    "rust_criterion_group": fpca_row["rust_criterion_group"],
                    "rust_criterion_bench": fpca_row["rust_criterion_bench"],
                    "rust": fpca_row["rust_mean_ns"],
                },
            }, f, indent=2)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fdars cross-language benchmark orchestrator (full-op)"
    )
    parser.add_argument(
        "--op", choices=ALL_OPS + ["all"], default="all",
        help="Operation to benchmark (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Generator + runners + gate only; skip cargo bench")
    parser.add_argument("--env-only", action="store_true",
                        help="Capture env.json only; skip all runners")
    parser.add_argument("--n", type=int, default=60, help="Number of curves")
    parser.add_argument("--m", type=int, default=50, help="Number of eval points")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    # Build pinned-thread environment FIRST (Pitfall 1: set before any fork)
    env = _build_env()

    if args.env_only:
        step_capture_env(env)
        return

    op = None if args.op == "all" else args.op

    # Step 1: Generate shared CSV fixtures (all fixture types)
    step_generate(n=args.n, m=args.m, seed=args.seed, env=env)

    # Step 2: Python runner
    step_python_runner(op=op, n=args.n, m=args.m, seed=args.seed, env=env)

    # Step 3: R runner
    step_r_runner(op=op, n=args.n, m=args.m, seed=args.seed, env=env)

    # Step 4: Cargo bench (skipped in --dry-run)
    if not args.dry_run:
        step_cargo_bench(bench_files=BENCH_FILES, env=env)

    # Step 5: Collect criterion output
    step_collect_rust(env=env)

    # Step 6: Capture environment metadata
    step_capture_env(env)

    # Step 7: Run agreement gate BEFORE writing timing comparison (BENCH-05)
    # This will sys.exit(1) on failure — no timing written on mismatch.
    py_path = _SCRIPT_DIR / "results" / "python.json"
    r_path = _SCRIPT_DIR / "results" / "r.json"

    if not py_path.exists() or not r_path.exists():
        print(
            "ERROR: Missing results/python.json or results/r.json — "
            "runners must complete before gate.",
            file=sys.stderr,
        )
        sys.exit(1)

    step_run_gate(n=args.n, m=args.m, env=env, py_path=py_path, r_path=r_path)

    # Step 8: Write timing comparison (ONLY after gate passes)
    write_timing_comparison(n=args.n, m=args.m)

    print("\nDone.")


if __name__ == "__main__":
    main()
