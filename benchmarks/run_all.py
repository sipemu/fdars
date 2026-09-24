"""
Master orchestrator for the fdars cross-language benchmark harness.

Sets OMP/OPENBLAS/MKL/RAYON/NUMEXPR/VECLIB thread env vars to "1" in the
subprocess environment before forking any runner. The measure-first ordering
(numeric agreement gate BEFORE timing comparison is written) is the
load-bearing invariant of BENCH-05.

Full-op orchestration sequence:
  1. Generate all shared CSV fixtures (seeded, bit-identical)
  2. Run Python runner (all ops or specified op)
  3. Run R runner (all ops or specified op) — in --grid mode, fpca_native_r is
     also invoked explicitly (not in ALL_OPS) for the ecosystem-realistic timing.
  4. Run cargo bench (full bench suite or specified bench files) — skipped in --dry-run
  5. Collect criterion output into results/rust.json
  6. Capture environment metadata into results/env.json
  7. GATE: run full-op agreement gate BEFORE writing any timing comparison
     - For FPCA: compare fdars (via fpca_dump), Python, R
     - For all other ops: compare Python vs R (fdars fingerprints via fpca_dump
       extension for smoke runs; criterion output has timing only, not fingerprints)
     - Abort (exit non-zero) on any op mismatch; no timing recorded on failure
  8. Write merged per-op timing comparison (only on gate pass)

Grid mode (--grid):
  Iterates the locked N×M baseline grid (N ∈ {100,300,500} × M ∈ {50,200}) —
  6 cells total.  Each cell runs the full sequence above SEQUENTIALLY (never
  concurrent — shared CSV filenames per Pitfall 2 in RESEARCH.md).  Rust
  criterion output is NOT re-run per cell; it is collected from whatever is
  already on disk (Plan 01 populated it).  Per-cell result files are written to
  results/python_n{n}_m{m}.json and results/r_n{n}_m{m}.json so that the
  assembler (assemble_comparison.py) can merge them without re-running.

Usage:
    python run_all.py --op fpca [--dry-run] [--env-only]
    python run_all.py --env-only                   # capture env.json only
    python run_all.py --dry-run                    # all ops, no cargo bench
    python run_all.py                              # full run including cargo bench
    python run_all.py --grid                       # full N×M grid baseline run
    python run_all.py --grid --dry-run             # grid wiring check, no cargo bench

All subprocess calls use list-form only (no shell=True) per T-112-02.
"""

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

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

# Locked N×M grid for the --grid baseline run (BASE-01).
# Cells run sequentially (never concurrent — shared CSV filenames; Pitfall 2).
# N=300 exists only on the Python/R side; Rust criterion reuses N=100/N=500 output.
GRID_N = [100, 300, 500]
GRID_M = [50, 200]

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
        if smoke:
            # Smoke path: --smoke flag only (no --op argument; python_runner.py
            # treats --smoke without --op as "run all ops once for import/fingerprint
            # verification"). CR-02: do NOT splice --op value into a --smoke command.
            _run(
                [_PYTHON, str(_SCRIPT_DIR / "python_runner.py"),
                 "--smoke",
                 "--n", str(n), "--m", str(m), "--seed", str(seed)],
                env=env,
            )
        else:
            _run_all_ops_python(n, m, seed, env)
    else:
        print(f"[2] Running Python runner (op={op})...")
        _run(
            [_PYTHON, str(_SCRIPT_DIR / "python_runner.py"),
             "--op", op, "--n", str(n), "--m", str(m), "--seed", str(seed)],
            env=env,
        )


def _run_all_ops_python(n: int, m: int, seed: int, env: dict,
                        pairwise_only: bool = False,
                        ops: list[str] | None = None) -> None:
    """Run ops via python_runner, timing each op individually.

    WR-04: the previous implementation passed --smoke, which sets mean_ns=0.0
    for every op. The non-smoke path invokes --op <name> per op so that real
    timings are recorded. Because python_runner.py overwrites results/python.json
    on each call, we accumulate records from each per-op invocation and write the
    merged list back to python.json after all ops complete.

    Phase 113 Plan 02: when pairwise_only=True (grid baseline), the elastic op is
    run with --pairwise-only so only the fast pairwise alignment is timed (the
    slow N-dependent Karcher mean is excluded from grid timing).

    Phase 116 Plan 01: `ops` restricts which ops are run (default: ALL_OPS), so a
    single-op re-run (e.g. `--op fpca`) does not re-time the other five ops.
    """
    ops = ops if ops is not None else ALL_OPS
    py_out = _SCRIPT_DIR / "results" / "python.json"
    accumulated: list[dict] = []
    for single_op in ops:
        cmd = [_PYTHON, str(_SCRIPT_DIR / "python_runner.py"),
               "--op", single_op,
               "--n", str(n), "--m", str(m), "--seed", str(seed)]
        if pairwise_only and single_op == "elastic":
            cmd.append("--pairwise-only")
        _run(cmd, env=env)
        # python_runner.py writes a 1-element list; read it back and accumulate
        if py_out.exists():
            with py_out.open() as f:
                records = json.load(f)
            accumulated.extend(records)
    # Write the merged all-ops result so downstream steps see all 6 records
    py_out.parent.mkdir(parents=True, exist_ok=True)
    with py_out.open("w") as f:
        json.dump(accumulated, f, indent=2)


def step_r_runner(op: str | None, n: int, m: int, seed: int, env: dict,
                  smoke: bool = False) -> None:
    """Run the R runner for one op or all ops."""
    if op is None:
        print(f"[3] Running R runner (all ops, smoke={smoke})...")
        if smoke:
            # Smoke path: run all ops at once via --smoke (mean_ns=0.0 is expected).
            _run(
                ["Rscript", str(_SCRIPT_DIR / "r_runner.R"),
                 "--smoke",
                 "--n", str(n), "--m", str(m), "--seed", str(seed)],
                env=env,
            )
        else:
            # Non-smoke path: time each op individually so results/r.json has real
            # timings. WR-04: the previous implementation always passed --smoke,
            # producing mean_ns=0.0 for every op in the default run.
            # r_runner.R overwrites r.json on each call, so accumulate and merge.
            r_out = _SCRIPT_DIR / "results" / "r.json"
            accumulated: list[dict] = []
            for single_op in ALL_OPS:
                _run(
                    ["Rscript", str(_SCRIPT_DIR / "r_runner.R"),
                     "--op", single_op,
                     "--n", str(n), "--m", str(m), "--seed", str(seed)],
                    env=env,
                )
                if r_out.exists():
                    with r_out.open() as f:
                        records = json.load(f)
                    accumulated.extend(records)
            r_out.parent.mkdir(parents=True, exist_ok=True)
            with r_out.open("w") as f:
                json.dump(accumulated, f, indent=2)
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

        # fdars-vs-R is the HARD gate (both use Simpson's-family quadrature →
        # they must span the identical FPCA subspace; this proves fdars correct).
        p2, mc2, _ = check_fpca_agreement(fdars_load,
                                           fingerprint_to_array(r_fpca["result_fingerprint"]),
                                           threshold=0.99, label="fdars-vs-r")
        print(f"    fdars vs r:      {'PASS (HARD-GATE)' if p2 else 'FAIL (HARD-GATE)'} (subspace cos={mc2:.6f})")

        # fdars-vs-Python is DISCLOSED, NOT hard-gated (user-approved 2026-09-23):
        # scikit-fda uses trapezoidal quadrature vs fdars/R Simpson's, so its
        # trailing near-degenerate PCs tilt the subspace at coarse grids. Record
        # the subspace cosine per cell; do NOT abort on it.
        p1, mc1, _ = check_fpca_agreement(fdars_load,
                                           fingerprint_to_array(py_fpca["result_fingerprint"]),
                                           threshold=0.99, label="fdars-vs-python")
        _tag = "AGREES" if p1 else "DISCLOSED-BELOW"
        print(f"    fdars vs python: {mc1:.6f} (disclosure bar 0.99; {_tag}; NOT hard-gated — trapezoidal vs Simpson's quadrature)")

        # Persist the disclosed FPCA agreement into the python FPCA record so the
        # assembler can surface it in the comparison.csv notes column.
        py_fpca["fpca_disclosure"] = {
            "fdars_vs_python_subspace_cos": float(mc1),
            "fdars_vs_r_subspace_cos": float(mc2),
        }
        with py_path.open("w") as f:
            json.dump(py_records, f, indent=2)

        if not p2:
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

    # NOTE: fpca is NOT in this Python-vs-R list. FPCA agreement is established
    # authoritatively in step 6a (fdars-vs-R hard-gated at 1.0; fdars-vs-python
    # disclosed). A Python-vs-R FPCA check here would re-introduce the trapezoidal-
    # vs-Simpson's quadrature divergence we agreed to DISCLOSE, not hard-gate
    # (user-approved 2026-09-23) — so it is intentionally excluded.
    for op in ["basis_smoothing", "depth", "functional_boxplot", "fpcr", "elastic"]:
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


def _match_rust_rec(rust_records: list, op: str, n: int, m: int) -> dict:
    """
    Return the best-matching Rust record for (op, n, m), or {} if none.

    Prefers an exact n/m match (parsed from the record's bench field) so that
    write_timing_comparison does not associate Python/R timings for one grid cell
    with Rust timings from a different cell when multiple sizes exist on disk.
    Falls back to op-only match for single-size runs.
    """
    # Prefer exact n/m match (parsed from bench field, e.g. "n100_m50")
    exact = [
        r for r in rust_records
        if r.get("op") == op and r.get("n") == n and r.get("m") == m
    ]
    if exact:
        return exact[0]
    # Fall back to op-only match (single-size runs or n/m not parsed)
    return next((r for r in rust_records if r.get("op") == op), {})


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
        # Get fdars timing: find the best matching criterion record by op AND n/m.
        # rust.json may contain multiple sizes for the same op; match by n and m
        # to avoid associating Python/R timings for one cell with Rust from another.
        rust_rec = _match_rust_rec(rust_records, op, n, m)
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
# Grid support — per-cell result-file persistence for assembler
# -------------------------------------------------------------------


def _merge_cell_records(existing: list[dict], new: list[dict], ops: list[str]) -> list[dict]:
    """
    Merge freshly-run per-op records into an existing per-cell record list.

    Drops any existing record whose op is in `ops` (it is being replaced by the
    fresh run), plus `fpca_native_r` when "fpca" is in `ops` (fpca_native_r is
    fpca's ecosystem-realistic companion record, re-run together), then appends
    `new`. Used when a grid re-run covers only a strict subset of ALL_OPS
    (e.g. `--op fpca`) so a prior full-grid cell file is not silently truncated
    down to just the freshly-run op (Phase 116 Plan 01).
    """
    drop_ops = set(ops)
    if "fpca" in drop_ops:
        drop_ops.add("fpca_native_r")
    kept = [rec for rec in existing if rec.get("op") not in drop_ops]
    return kept + new


def _write_cell_results(n: int, m: int, py_records: list, r_records: list,
                        ops: list[str] | None = None) -> None:
    """
    Persist per-cell Python and R timing records for the assembler.

    Writes:
        results/python_n{n}_m{m}.json  — all Python op records for this cell
        results/r_n{n}_m{m}.json       — all R op records for this cell
           (including fpca_native_r if present)

    These files are consumed by assemble_comparison.py to build comparison.csv
    without re-running the benchmarks.

    Phase 116 Plan 01: when `ops` is a STRICT SUBSET of ALL_OPS, merge into
    whatever per-cell files already exist on disk (via _merge_cell_records)
    instead of overwriting them — a `--op fpca` re-run must not erase the other
    five ops' records from a prior full-grid run. When `ops` is None or the
    full ALL_OPS set, behavior is unchanged (full overwrite).
    """
    ops = ops if ops is not None else ALL_OPS
    results_dir = _SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    py_out = results_dir / f"python_n{n}_m{m}.json"
    r_out = results_dir / f"r_n{n}_m{m}.json"

    if set(ops) < set(ALL_OPS):
        existing_py: list[dict] = []
        if py_out.exists():
            with py_out.open() as f:
                existing_py = json.load(f)
        existing_r: list[dict] = []
        if r_out.exists():
            with r_out.open() as f:
                existing_r = json.load(f)
        py_records = _merge_cell_records(existing_py, py_records, ops)
        r_records = _merge_cell_records(existing_r, r_records, ops)

    with py_out.open("w") as f:
        json.dump(py_records, f, indent=2)
    with r_out.open("w") as f:
        json.dump(r_records, f, indent=2)

    print(f"  Wrote {py_out.name} ({len(py_records)} records)")
    print(f"  Wrote {r_out.name} ({len(r_records)} records)")


def _run_r_with_native_fpca(n: int, m: int, seed: int, env: dict,
                            pairwise_only: bool = False,
                            ops: list[str] | None = None) -> list:
    """
    Run R runner for `ops` and (when "fpca" is in `ops`) also fpca_native_r;
    return merged records.

    fpca_native_r is not in ALL_OPS (keeps smoke/default contract unchanged per
    Plan 01), so it must be invoked explicitly via --op fpca_native_r.

    Phase 113 Plan 02: when pairwise_only=True (grid baseline), the elastic op is
    run with --pairwise-only so only the fast pairwise alignment is timed (the
    slow fdasrvf time_warping Karcher mean is excluded from grid timing).

    Phase 116 Plan 01: `ops` restricts which ops are run (default: ALL_OPS).
    fpca_native_r is invoked only when "fpca" is in `ops` — it is fpca's
    ecosystem-realistic companion timing, not an independent op.
    """
    ops = ops if ops is not None else ALL_OPS
    r_out = _SCRIPT_DIR / "results" / "r.json"
    accumulated: list[dict] = []

    # Requested overlapping ops
    for single_op in ops:
        cmd = ["Rscript", str(_SCRIPT_DIR / "r_runner.R"),
               "--op", single_op,
               "--n", str(n), "--m", str(m), "--seed", str(seed)]
        if pairwise_only and single_op == "elastic":
            cmd.append("--pairwise-only")
        _run(cmd, env=env)
        if r_out.exists():
            with r_out.open() as f:
                records = json.load(f)
            accumulated.extend(records)

    # Explicitly invoke fpca_native_r — ecosystem-realistic pca.fd timing —
    # only when fpca itself is being (re-)run this call.
    if "fpca" in ops:
        print(f"[3b] Running R fpca_native_r (n={n}, m={m})...")
        _run(
            ["Rscript", str(_SCRIPT_DIR / "r_runner.R"),
             "--op", "fpca_native_r",
             "--n", str(n), "--m", str(m), "--seed", str(seed)],
            env=env,
        )
        if r_out.exists():
            with r_out.open() as f:
                native_records = json.load(f)
            accumulated.extend(native_records)

    # Write merged list back to r.json
    r_out.parent.mkdir(parents=True, exist_ok=True)
    with r_out.open("w") as f:
        json.dump(accumulated, f, indent=2)

    return accumulated


def run_grid(dry_run: bool = False, seed: int = 42,
            cells: list[tuple[int, int]] | None = None,
            ops: list[str] | None = None) -> None:
    """
    Execute the N×M baseline grid run (BASE-01), or a caller-selected subset.

    Iterates `cells` (default: the full GRID_N × GRID_M cross product, 6 cells)
    strictly sequentially (never concurrent — shared CSV filenames per Pitfall 2
    in RESEARCH.md).  For each cell:

      1. Generate CSV fixtures (seeded, bit-identical across ecosystems)
      2. Run Python runner (`ops`, per-op timing)
      3. Run R runner (`ops` + fpca_native_r when "fpca" is in `ops`)
      4. Collect Rust criterion output (existing on disk; no new cargo bench)
      5. Run gate — ABORT the whole grid run on any failure (BENCH-05)
      6. Persist per-cell result files (python_n{n}_m{m}.json, r_n{n}_m{m}.json);
         merges into any existing files when `ops` is a strict subset of
         ALL_OPS (Phase 116 Plan 01 — see _write_cell_results).

    In --dry-run mode: steps 1–3 are skipped entirely (proves wiring without
    the heavy Python/R runs); gate and persist are also skipped; the loop
    still iterates every requested cell to verify the grid structure.

    Note: N=300 cells on the Rust criterion side reuse whatever criterion
    output is on disk (typically N=100/N=500 cells); no new cargo bench is
    run for N=300.  See RESEARCH.md open-question 2.

    Args:
        dry_run: skip the heavy runner/gate/persist steps (wiring check only).
        seed: RNG seed passed through to the CSV generator.
        cells: explicit (n, m) cells to run (default: all 6 grid cells).
        ops: which harness ops to run per cell (default: ALL_OPS).
    """
    env = _build_env()
    cells = cells if cells is not None else [(n, m) for n in GRID_N for m in GRID_M]
    ops = ops if ops is not None else ALL_OPS
    cells_run = 0
    failed_cell = None

    print(f"[GRID] Starting grid run: {len(cells)} cell(s) {cells}, ops={ops}, "
          f"dry_run={dry_run}")

    for n, m in cells:
        print(f"\n[GRID] Cell n={n} m={m} ({'DRY RUN — skipping runners' if dry_run else 'running'})")

        if dry_run:
            # Prove the loop wiring without heavy runs
            print(f"  [dry] would run: generate(n={n}, m={m}, seed={seed})")
            print(f"  [dry] would run: python_runner (ops={ops}, elastic pairwise-only, n={n}, m={m})")
            print(f"  [dry] would run: r_runner (ops={ops} + fpca_native_r if fpca in ops, elastic pairwise-only, n={n}, m={m})")
            print(f"  [dry] would run: collect_rust + gate")
            print(f"  [dry] would write: python_n{n}_m{m}.json, r_n{n}_m{m}.json")
            cells_run += 1
            continue

        # Step 1: Generate CSV fixtures for this cell
        step_generate(n=n, m=m, seed=seed, env=env)

        # Step 2: Python runner — requested ops, per-op real timings.
        # pairwise_only=True: elastic times only the fast pairwise alignment
        # (Karcher mean excluded from grid timing — Phase 113 Plan 02).
        _run_all_ops_python(n=n, m=m, seed=seed, env=env, pairwise_only=True, ops=ops)
        py_path = _SCRIPT_DIR / "results" / "python.json"
        with py_path.open() as f:
            py_records = json.load(f)

        # Step 3: R runner — requested ops (+ fpca_native_r if fpca in ops),
        # elastic pairwise-only.
        r_records = _run_r_with_native_fpca(n=n, m=m, seed=seed, env=env,
                                            pairwise_only=True, ops=ops)
        r_path = _SCRIPT_DIR / "results" / "r.json"

        # Step 4: Collect Rust criterion output (existing on disk)
        step_collect_rust(env=env)

        # Step 5: Gate — ABORT on any failure (BENCH-05 measure-first)
        print(f"[GRID] Running agreement gate for n={n} m={m}...")
        gate_passed = step_run_gate(n=n, m=m, env=env,
                                    py_path=py_path, r_path=r_path)
        if not gate_passed:
            # step_run_gate calls sys.exit(1) on failure; reaching here
            # means it returned False rather than exiting (defensive).
            failed_cell = (n, m)
            print(
                f"\nGATE FAIL for cell n={n} m={m} — aborting grid run.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Step 6: Persist per-cell result files for assembler.
        # Re-load py_records from disk: step_run_gate patched python.json in
        # place with the FPCA fdars-vs-python disclosure (fpca_disclosure),
        # so the persisted per-cell file must reflect that patch, not the
        # stale pre-gate in-memory records.
        with py_path.open() as f:
            py_records = json.load(f)
        _write_cell_results(n=n, m=m, py_records=py_records, r_records=r_records, ops=ops)
        cells_run += 1
        print(f"[GRID] Cell n={n} m={m}: DONE (gate passed, results written)")

    if dry_run:
        print(f"\n[GRID] Dry-run complete: verified loop structure for "
              f"{cells_run}/{len(cells)} cells.")
    else:
        print(f"\n[GRID] Grid run complete: {cells_run} cells processed, "
              f"gate passed for all cells.")


# -------------------------------------------------------------------
# Cell-spec parsing + timeout-bounded Rust bench invocation (Phase 116 Plan 01)
# -------------------------------------------------------------------


def _parse_cell(text: str) -> tuple[int, int]:
    """
    Parse a "NxM" cell spec (e.g. "500x200") into (n, m).

    Raises:
        ValueError: if `text` is not of the form "<int>x<int>", or if the
            parsed (n, m) is not a member of the locked GRID_N × GRID_M grid.
    """
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Malformed cell spec {text!r}: expected 'NxM' (e.g. '500x200')")
    try:
        n, m = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Malformed cell spec {text!r}: N and M must be integers") from exc
    if n not in GRID_N or m not in GRID_M:
        raise ValueError(
            f"Cell ({n}, {m}) is not in the locked grid: N in {GRID_N}, M in {GRID_M}"
        )
    return (n, m)


def _run_with_timeout(cmd: list[str], env: dict, timeout_s: int | None,
                      cwd: str | None = None) -> tuple[str, float, int]:
    """
    Run a subprocess with an optional wall-clock timeout (T-116-02).

    The child is started in its own session (`start_new_session=True`) so that
    on timeout the WHOLE process group can be SIGKILLed via `os.killpg` — a
    stuck `cargo bench` invocation may itself have spawned the actual bench
    binary as a grandchild, and killing only the immediate `cargo` process
    would leave that grandchild running and skewing every later timing.
    List-form argv only (no shell=True) per T-112-02.

    Returns:
        (status, wall_s, returncode):
          - status: "ok" (returncode == 0), "timeout" (killed after
            timeout_s seconds), or "error" (non-zero exit, no timeout).
          - wall_s: elapsed wall-clock seconds.
          - returncode: the process's exit code, or -9 on timeout-kill.
    """
    start = time.monotonic()
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, start_new_session=True)
    try:
        returncode = proc.wait(timeout=timeout_s)
        wall_s = time.monotonic() - start
        status = "ok" if returncode == 0 else "error"
        return status, wall_s, returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()  # reap the child so it does not linger as a zombie
        wall_s = time.monotonic() - start
        return "timeout", wall_s, -9


def run_rust_bench(bench: str, filter_regex: str, env: dict,
                   timeout_s: int | None) -> tuple[str, float]:
    """
    Run a single criterion bench file, filtered to `filter_regex`, under an
    optional wall-clock timeout (Phase 116 Plan 01 — thread-pinned single
    criterion invocation for the tracer/grid-aligned crosslang benches).

    Validates that `fdars-core/benches/<bench>.rs` exists before spawning
    (exits 2 with a clear message otherwise — a typo'd bench name should fail
    fast here, not surface as cargo's own opaque "no bench target" error deep
    inside a subprocess).

    The env is the thread-pinned `_build_env()`-shaped dict passed in by the
    caller, plus `FDARS_CROSSLANG_DATA_DIR` (absolute `benchmarks/data`) and
    `FDARS_CROSSLANG_REQUIRE_CSV=1` so the bench never silently falls back to
    its synthetic-data path during a real harness run.

    Prints `RUST_BENCH_STATUS=<status> wall_s=<seconds>` for the caller (and
    for whichever process reads this script's stdout) to key off of.

    Returns:
        (status, wall_s) — see `_run_with_timeout`.
    """
    bench_path = _SCRIPT_DIR.parent / "fdars-core" / "benches" / f"{bench}.rs"
    if not bench_path.exists():
        print(f"ERROR: bench file not found: {bench_path}", file=sys.stderr)
        sys.exit(2)

    run_env = {
        **env,
        "FDARS_CROSSLANG_DATA_DIR": str((_SCRIPT_DIR / "data").resolve()),
        "FDARS_CROSSLANG_REQUIRE_CSV": "1",
    }
    cmd = [
        "cargo", "bench",
        "-p", "fdars-core",
        "--features", "linalg,parallel",
        "--bench", bench,
        "--",
        filter_regex,
    ]
    status, wall_s, _returncode = _run_with_timeout(
        cmd, env=run_env, timeout_s=timeout_s, cwd=str(_WORKSPACE_ROOT)
    )
    print(f"RUST_BENCH_STATUS={status} wall_s={wall_s:.2f}")
    return status, wall_s


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
                        help="Generator + runners + gate only; skip cargo bench; "
                             "in --grid mode: proves loop wiring without any runner calls")
    parser.add_argument("--env-only", action="store_true",
                        help="Capture env.json only; skip all runners")
    parser.add_argument("--grid", action="store_true",
                        help="Run the full N×M baseline grid "
                             "(N in {100,300,500} × M in {50,200}, 6 cells) "
                             "sequentially with gate-before-timing per cell (BASE-01). "
                             "Combine with --dry-run to verify loop wiring only.")
    parser.add_argument("--n", type=int, default=60, help="Number of curves")
    parser.add_argument("--m", type=int, default=50, help="Number of eval points")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--cell", action="append", default=None, metavar="NxM",
        help="Grid cell as 'NxM' (e.g. '500x200'); repeatable. Only valid with "
             "--grid. Must be a member of GRID_N x GRID_M. Default: all 6 cells.",
    )
    parser.add_argument(
        "--rust-bench", type=str, default=None,
        help="Run a single Rust criterion bench file (fdars-core/benches/<name>.rs) "
             "under an optional wall-clock timeout, thread-pinned, instead of the "
             "Python/R grid. Requires --rust-filter.",
    )
    parser.add_argument(
        "--rust-filter", type=str, default=None,
        help="Criterion filter regex for --rust-bench (required with --rust-bench).",
    )
    parser.add_argument(
        "--timeout-s", type=int, default=None,
        help="Wall-clock timeout in seconds for --rust-bench (default: no limit).",
    )
    args = parser.parse_args()

    # Build pinned-thread environment FIRST (Pitfall 1: set before any fork)
    env = _build_env()

    # --rust-bench: single thread-pinned criterion invocation under a timeout
    # (Phase 116 Plan 01). Exits 0 only when the bench completes successfully.
    if args.rust_bench is not None:
        if args.rust_filter is None:
            print("ERROR: --rust-filter is required with --rust-bench.", file=sys.stderr)
            sys.exit(2)
        status, _wall_s = run_rust_bench(args.rust_bench, args.rust_filter, env, args.timeout_s)
        sys.exit(0 if status == "ok" else 1)

    if args.cell is not None and not args.grid:
        print("ERROR: --cell is only valid with --grid.", file=sys.stderr)
        sys.exit(2)

    if args.env_only:
        step_capture_env(env)
        return

    # --grid: run the N×M baseline grid (BASE-01), or a --cell/--op subset of it
    if args.grid:
        cells: list[tuple[int, int]] | None = None
        if args.cell is not None:
            try:
                cells = [_parse_cell(c) for c in args.cell]
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(2)
        grid_ops = ALL_OPS if args.op == "all" else [args.op]
        run_grid(dry_run=args.dry_run, seed=args.seed, cells=cells, ops=grid_ops)
        if not args.dry_run:
            # Re-capture env.json after the grid run so it reflects this run
            step_capture_env(env)
            print("\nGrid baseline run complete. Run assemble_comparison.py to build comparison.csv.")
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
