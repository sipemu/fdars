"""
Master orchestrator for the fdars cross-language benchmark harness.

Sets OMP/OPENBLAS/MKL/RAYON/NUMEXPR/VECLIB thread env vars to "1" in the
subprocess environment before forking any runner. The measure-first ordering
(numeric agreement gate BEFORE timing comparison is written) is the
load-bearing invariant of BENCH-05.

Usage:
    python run_all.py --op fpca [--dry-run] [--env-only]
    python run_all.py --env-only                   # capture env.json only
    python run_all.py --op fpca --dry-run          # generator + runners + gate, no cargo bench
    python run_all.py --op fpca                    # full run including cargo bench

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


def _build_env() -> dict:
    """Build the pinned-thread environment dict."""
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
    """Run the seeded CSV generator."""
    print(f"[1/N] Generating CSV data (n={n}, m={m}, seed={seed})...")
    _run(
        [_PYTHON, str(_SCRIPT_DIR / "generate_data.py"),
         "--n", str(n), "--m", str(m), "--seed", str(seed)],
        env=env,
    )


def step_python_runner(op: str, n: int, m: int, seed: int, env: dict) -> None:
    """Run the Python (scikit-fda) runner."""
    print(f"[2/N] Running Python runner (op={op})...")
    _run(
        [_PYTHON, str(_SCRIPT_DIR / "python_runner.py"),
         "--op", op, "--n", str(n), "--m", str(m), "--seed", str(seed)],
        env=env,
    )


def step_r_runner(op: str, n: int, m: int, seed: int, env: dict) -> None:
    """Run the R runner."""
    print(f"[3/N] Running R runner (op={op})...")
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
        # Build if not already built
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


def step_cargo_bench(bench_name: str, env: dict) -> None:
    """Run the criterion bench (skipped in --dry-run mode)."""
    print(f"[4/N] Running cargo bench ({bench_name})...")
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
    print("[5/N] Collecting criterion output...")
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


def step_gate_fpca(n: int, m: int, env: dict) -> bool:
    """
    Run the FPCA numeric-agreement gate.

    Loads loadings from Python, R, AND fdars, sign-aligns, and checks
    Pearson correlation >= 0.99. Returns True if all pairs pass.
    ABORTS with a non-zero exit and 'GATE FAIL' message on any failure.
    No timing comparison is written if the gate fails.
    """
    from gate import check_fpca_agreement, fingerprint_to_array

    py_path = _SCRIPT_DIR / "results" / "python.json"
    r_path = _SCRIPT_DIR / "results" / "r.json"

    with py_path.open() as f:
        py_records = json.load(f)
    with r_path.open() as f:
        r_records = json.load(f)

    py_fpca = next((r for r in py_records if r["op"] == "fpca"), None)
    r_fpca = next((r for r in r_records if r["op"] == "fpca"), None)

    if py_fpca is None or r_fpca is None:
        print("GATE FAIL: no FPCA records found in results files", file=sys.stderr)
        sys.exit(1)

    py_load = fingerprint_to_array(py_fpca["result_fingerprint"])  # (m, ncomp)
    r_load = fingerprint_to_array(r_fpca["result_fingerprint"])   # (m, ncomp)

    # Obtain fdars loadings via the fpca_dump binary
    print("[6/N] Obtaining fdars FPCA loadings for agreement gate...")
    fdars_rows = step_fdars_loadings(n=n, m=m, env=env)
    import numpy as np
    fdars_load = np.array(fdars_rows, dtype=np.float64).T  # (m, ncomp)

    # Gate 1: fdars vs Python
    print("[6a/N] Agreement gate: fdars vs Python...", end=" ")
    passed1, min_corr1, details1 = check_fpca_agreement(
        fdars_load, py_load, threshold=0.99, label="fdars-vs-python"
    )
    print(f"{'PASS' if passed1 else 'FAIL'} (min |corr| = {min_corr1:.6f})")
    for d in details1:
        print(f"  PC{d['component']+1}: corr={d['correlation']:.6f}")

    # Gate 2: fdars vs R
    print("[6b/N] Agreement gate: fdars vs R...", end=" ")
    passed2, min_corr2, details2 = check_fpca_agreement(
        fdars_load, r_load, threshold=0.99, label="fdars-vs-r"
    )
    print(f"{'PASS' if passed2 else 'FAIL'} (min |corr| = {min_corr2:.6f})")
    for d in details2:
        print(f"  PC{d['component']+1}: corr={d['correlation']:.6f}")

    all_passed = passed1 and passed2
    if not all_passed:
        print("\nGATE FAIL — aborting. No timing comparison written.", file=sys.stderr)
        sys.exit(1)

    print("\n[GATE PASS] FPCA agreement verified — proceeding to timing comparison.")
    return True


def write_timing_summary(n: int, m: int) -> None:
    """
    Write the merged FPCA timing comparison record.

    Only called AFTER the gate passes (BENCH-05 measure-first invariant).
    """
    py_path = _SCRIPT_DIR / "results" / "python.json"
    r_path = _SCRIPT_DIR / "results" / "r.json"
    rust_path = _SCRIPT_DIR / "results" / "rust.json"
    summary_path = _SCRIPT_DIR / "results" / "fpca_timing.json"

    with py_path.open() as f:
        py_records = json.load(f)
    with r_path.open() as f:
        r_records = json.load(f)

    py_fpca = next((r for r in py_records if r["op"] == "fpca"), {})
    r_fpca = next((r for r in r_records if r["op"] == "fpca"), {})

    # Get fdars timing from rust.json (audit_fpca group, n500_m200 or closest)
    rust_records = []
    if rust_path.exists():
        with rust_path.open() as f:
            rust_records = json.load(f)

    rust_fpca = next(
        (r for r in rust_records if "fpca" in r.get("op", "").lower()),
        {}
    )

    summary = {
        "op": "fpca",
        "n": n,
        "m": m,
        "gate_passed": True,
        "timing_ns": {
            "python": py_fpca.get("mean_ns"),
            "r": r_fpca.get("mean_ns"),
            "rust_criterion_group": rust_fpca.get("op"),
            "rust_criterion_bench": rust_fpca.get("bench"),
            "rust": rust_fpca.get("mean_ns"),
        },
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTiming summary written: {summary_path}")
    print(f"  Python:  {py_fpca.get('mean_ns', 'N/A'):.0f} ns" if py_fpca.get('mean_ns') else "  Python:  N/A")
    print(f"  R:       {r_fpca.get('mean_ns', 'N/A'):.0f} ns" if r_fpca.get('mean_ns') else "  R:       N/A")
    if rust_fpca.get('mean_ns'):
        print(f"  Rust:    {rust_fpca['mean_ns']:.0f} ns  (group={rust_fpca.get('op')}, bench={rust_fpca.get('bench')})")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fdars cross-language benchmark orchestrator"
    )
    parser.add_argument("--op", choices=["fpca"], default=None,
                        help="Operation to benchmark")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generator + runners + gate only; skip cargo bench")
    parser.add_argument("--env-only", action="store_true",
                        help="Capture env.json only; skip all runners")
    parser.add_argument("--n", type=int, default=60, help="Number of curves")
    parser.add_argument("--m", type=int, default=50, help="Number of eval points")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    # Build pinned-thread environment
    env = _build_env()

    if args.env_only:
        step_capture_env(env)
        return

    if args.op is None:
        parser.error("--op is required (unless --env-only)")

    if args.op == "fpca":
        # Step 1: Generate data
        step_generate(n=args.n, m=args.m, seed=args.seed, env=env)

        # Step 2: Python runner
        step_python_runner(op="fpca", n=args.n, m=args.m, seed=args.seed, env=env)

        # Step 3: R runner
        step_r_runner(op="fpca", n=args.n, m=args.m, seed=args.seed, env=env)

        # Step 4: Cargo bench (skipped in --dry-run)
        if not args.dry_run:
            step_cargo_bench("regression_benchmarks", env=env)
            step_collect_rust(env=env)

        # Step 5: Capture environment
        step_capture_env(env)

        # Step 6: FPCA agreement gate (BENCH-05: BEFORE writing timing comparison)
        # This will sys.exit(1) on failure — no timing written on mismatch.
        step_gate_fpca(n=args.n, m=args.m, env=env)

        # Step 7: Write timing summary (ONLY after gate passes)
        write_timing_summary(n=args.n, m=args.m)

    print("\nDone.")


if __name__ == "__main__":
    main()
