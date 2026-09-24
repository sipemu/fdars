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

    # Phase 116 Plan 03 — resumable fdars criterion re-run:
    python run_all.py --list-rust-chunks           # print the 67-chunk manifest
    python run_all.py --rerun-init                 # initialize results/rerun_log.json
    python run_all.py --rust-next --budget-s 540   # run pending chunks within one call's budget
    python run_all.py --rust-status                # print per-status chunk counts
    python run_all.py --rust-requeue 12             # re-queue chunk 12 (last status must be error)

All subprocess calls use list-form only (no shell=True) per T-112-02.
"""

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

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

# -------------------------------------------------------------------
# Phase 116 Plan 03 — resumable, budgeted fdars criterion re-run (D-09).
# -------------------------------------------------------------------

# Six baseline groups whose `benchmark_group("...")` literal no longer exists
# in fdars-core/benches (renamed or removed since the Phase 113 baseline was
# captured). They cannot be re-run; the report lists them as stale rows.
ORPHAN_GROUPS = frozenset({
    "face_covariance",         # renamed to perf_face_covariance / mface_covariance
    "coclustering_co_cluster",  # superseded by coclustering_co_cluster_select
    "fem_smooth",               # superseded by fem_smooth_gcv
    "fpca_variants_fsvd",       # superseded by fpca_fsvd (fpca_variants_benchmarks.rs)
    "frechet_anova",            # superseded by frechet_global_reg
    "fts_long_run_covariance",  # superseded by fts_ftsm
})

# The 67-chunk re-run manifest (D-09 FULL re-run). Execution order = list
# order. `baseline_mean_ns` is the Phase 113 committed criterion mean (ns),
# informational for group-level chunks (sum of the group's baseline ids),
# `None` for the six new crosslang_<op> groups (no Phase 113 baseline exists
# for them), and the ordering/inference key for per-id chunks. Filters are
# anchored regexes matched against the full criterion id `group/function[/param]`.
RUST_RERUN_CHUNKS: list[dict] = [
    # --- Tier P0: grid-aligned head-to-head (bench crosslang_grid), group-level,
    # run first so the head-to-head report has numbers even if later tiers
    # time out. ---
    {"tier": "P0", "bench": "crosslang_grid", "group": "crosslang_fpca",
     "filter": r"^crosslang_fpca/", "baseline_mean_ns": None},
    {"tier": "P0", "bench": "crosslang_grid", "group": "crosslang_basis_smoothing",
     "filter": r"^crosslang_basis_smoothing/", "baseline_mean_ns": None},
    {"tier": "P0", "bench": "crosslang_grid", "group": "crosslang_depth",
     "filter": r"^crosslang_depth/", "baseline_mean_ns": None},
    {"tier": "P0", "bench": "crosslang_grid", "group": "crosslang_functional_boxplot",
     "filter": r"^crosslang_functional_boxplot/", "baseline_mean_ns": None},
    {"tier": "P0", "bench": "crosslang_grid", "group": "crosslang_fpcr",
     "filter": r"^crosslang_fpcr/", "baseline_mean_ns": None},
    {"tier": "P0", "bench": "crosslang_grid", "group": "crosslang_elastic",
     "filter": r"^crosslang_elastic/", "baseline_mean_ns": None},

    # --- Tier P1: optimization targets (Phase 114/115 anchors + SVD paths). ---
    {"tier": "P1", "bench": "perf_hotpaths", "group": "perf_face_covariance",
     "filter": r"^perf_face_covariance/", "baseline_mean_ns": 207488577.11420387},
    {"tier": "P1", "bench": "audit_hotpaths", "group": "audit_p3_elastic_cross_banded",
     "filter": r"^audit_p3_elastic_cross_banded/n100_m50$", "baseline_mean_ns": 324136666.3875},
    {"tier": "P1", "bench": "audit_hotpaths", "group": "audit_p3_elastic_cross_banded",
     "filter": r"^audit_p3_elastic_cross_banded/n100_m200$", "baseline_mean_ns": 6177604457.0},
    {"tier": "P1", "bench": "audit_hotpaths", "group": "audit_p3_elastic_cross_banded",
     "filter": r"^audit_p3_elastic_cross_banded/n500_m50$", "baseline_mean_ns": 7841426505.6},
    {"tier": "P1", "bench": "audit_hotpaths", "group": "audit_p4_fpca",
     "filter": r"^audit_p4_fpca/", "baseline_mean_ns": 61215967.34755423},
    {"tier": "P1", "bench": "audit_hotpaths", "group": "audit_fpca",
     "filter": r"^audit_fpca/", "baseline_mean_ns": 16390770.837457303},
    {"tier": "P1", "bench": "audit_hotpaths", "group": "audit_p6_svd_nalgebra",
     "filter": r"^audit_p6_svd_nalgebra/", "baseline_mean_ns": 453085512.05575854},
    {"tier": "P1", "bench": "audit_hotpaths", "group": "audit_p6_svd_faer_seq",
     "filter": r"^audit_p6_svd_faer_seq/", "baseline_mean_ns": 155945201.9209864},
    {"tier": "P1", "bench": "audit_hotpaths", "group": "audit_p6_svd_conversion",
     "filter": r"^audit_p6_svd_conversion/", "baseline_mean_ns": 31.840410426222668},

    # --- Tier P2: legacy cross-language comparators (group-level). ---
    {"tier": "P2", "bench": "depth_benchmarks", "group": "functional_boxplot",
     "filter": r"^functional_boxplot/", "baseline_mean_ns": 65401577.5430609},
    {"tier": "P2", "bench": "regression_benchmarks", "group": "fregre_lm",
     "filter": r"^fregre_lm/", "baseline_mean_ns": 2408889.977191288},
    {"tier": "P2", "bench": "alignment_benchmarks", "group": "elastic_align_pair",
     "filter": r"^elastic_align_pair/", "baseline_mean_ns": 94410976.25283563},
    {"tier": "P2", "bench": "audit_hotpaths", "group": "audit_depth",
     "filter": r"^audit_depth/", "baseline_mean_ns": 468855.425524497},
    {"tier": "P2", "bench": "audit_hotpaths", "group": "audit_streaming_depth",
     "filter": r"^audit_streaming_depth/", "baseline_mean_ns": 525657.6310235932},
    {"tier": "P2", "bench": "audit_hotpaths", "group": "audit_smooth",
     "filter": r"^audit_smooth/", "baseline_mean_ns": 110417.04703894928},
    {"tier": "P2", "bench": "alignment_benchmarks", "group": "karcher_mean",
     "filter": r"^karcher_mean/", "baseline_mean_ns": 527996492.22},
    {"tier": "P2", "bench": "audit_hotpaths", "group": "audit_karcher",
     "filter": r"^audit_karcher/", "baseline_mean_ns": 162282532.0618295},
    {"tier": "P2", "bench": "audit_hotpaths", "group": "audit_elastic",
     "filter": r"^audit_elastic/", "baseline_mean_ns": 816797380.75},
    {"tier": "P2", "bench": "audit_hotpaths", "group": "audit_p4_elastic_fpca",
     "filter": r"^audit_p4_elastic_fpca/", "baseline_mean_ns": 2185260.145023933},

    # --- Tier P3: remainder of the baseline suite. Per-id chunks, ascending
    # baseline mean within each group. ---
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_elastic_cross",
     "filter": r"^audit_p3_elastic_cross/n100_m50$", "baseline_mean_ns": 1565454445.875},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_elastic_cross",
     "filter": r"^audit_p3_elastic_cross/n100_m200$", "baseline_mean_ns": 28852158307.3},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_elastic_cross",
     "filter": r"^audit_p3_elastic_cross/n500_m50$", "baseline_mean_ns": 37964661211.7},

    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_elastic_self",
     "filter": r"^audit_p3_elastic_self/n100_m50$", "baseline_mean_ns": 1479459642.0},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_elastic_self",
     "filter": r"^audit_p3_elastic_self/n100_m200$", "baseline_mean_ns": 16755681683.1},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_elastic_self",
     "filter": r"^audit_p3_elastic_self/n500_m50$", "baseline_mean_ns": 26554376669.6},

    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_elastic_self_banded",
     "filter": r"^audit_p3_elastic_self_banded/n100_m50$", "baseline_mean_ns": 406670174.4333334},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_elastic_self_banded",
     "filter": r"^audit_p3_elastic_self_banded/n100_m200$", "baseline_mean_ns": 3423428252.0},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_elastic_self_banded",
     "filter": r"^audit_p3_elastic_self_banded/n500_m50$", "baseline_mean_ns": 4223192730.1},

    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_karcher",
     "filter": r"^audit_p3_karcher/n100_m50$", "baseline_mean_ns": 644766211.15},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_karcher",
     "filter": r"^audit_p3_karcher/n500_m50$", "baseline_mean_ns": 3979197269.1},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_karcher",
     "filter": r"^audit_p3_karcher/n100_m200$", "baseline_mean_ns": 6483448642.25},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_karcher",
     "filter": r"^audit_p3_karcher/n500_m200$", "baseline_mean_ns": 28809597402.0},

    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_karcher_banded",
     "filter": r"^audit_p3_karcher_banded/n100_m50$", "baseline_mean_ns": 75448867.86900131},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_karcher_banded",
     "filter": r"^audit_p3_karcher_banded/n500_m50$", "baseline_mean_ns": 421493296.34103173},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_karcher_banded",
     "filter": r"^audit_p3_karcher_banded/n100_m200$", "baseline_mean_ns": 754891204.4673413},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p3_karcher_banded",
     "filter": r"^audit_p3_karcher_banded/n500_m200$", "baseline_mean_ns": 4872578977.05},

    {"tier": "P3", "bench": "alignment_benchmarks", "group": "elastic_self_distance_matrix",
     "filter": r"^elastic_self_distance_matrix/n_band0\.1/10$", "baseline_mean_ns": 14834670.41},
    {"tier": "P3", "bench": "alignment_benchmarks", "group": "elastic_self_distance_matrix",
     "filter": r"^elastic_self_distance_matrix/n/10$", "baseline_mean_ns": 48865427.1},
    {"tier": "P3", "bench": "alignment_benchmarks", "group": "elastic_self_distance_matrix",
     "filter": r"^elastic_self_distance_matrix/n_band0\.1/30$", "baseline_mean_ns": 87870493.04},
    {"tier": "P3", "bench": "alignment_benchmarks", "group": "elastic_self_distance_matrix",
     "filter": r"^elastic_self_distance_matrix/n_band0\.1/50$", "baseline_mean_ns": 186781587.59},
    {"tier": "P3", "bench": "alignment_benchmarks", "group": "elastic_self_distance_matrix",
     "filter": r"^elastic_self_distance_matrix/n/30$", "baseline_mean_ns": 356063270.38},
    {"tier": "P3", "bench": "alignment_benchmarks", "group": "elastic_self_distance_matrix",
     "filter": r"^elastic_self_distance_matrix/n/50$", "baseline_mean_ns": 757224142.5},

    # --- Group-level (remainder). ---
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_cv",
     "filter": r"^audit_cv/", "baseline_mean_ns": 950779.1294042313},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p5_karcher_paybackN",
     "filter": r"^audit_p5_karcher_paybackN/", "baseline_mean_ns": 720493960.0433267},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p5_karcher_threads",
     "filter": r"^audit_p5_karcher_threads/", "baseline_mean_ns": 331243388.1997222},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p5_streaming_paybackN",
     "filter": r"^audit_p5_streaming_paybackN/", "baseline_mean_ns": 887702.9204060981},
    {"tier": "P3", "bench": "audit_hotpaths", "group": "audit_p5_streaming_threads",
     "filter": r"^audit_p5_streaming_threads/", "baseline_mean_ns": 622756.1811937515},
    {"tier": "P3", "bench": "boosting_regression_benchmarks", "group": "boosting_boost_fosr",
     "filter": r"^boosting_boost_fosr/", "baseline_mean_ns": 55377631.722367585},
    {"tier": "P3", "bench": "coclustering_benchmarks", "group": "coclustering_co_cluster_select",
     "filter": r"^coclustering_co_cluster_select/", "baseline_mean_ns": 229331682.25},
    {"tier": "P3", "bench": "density_fda_benchmarks", "group": "density_lqd_fpca",
     "filter": r"^density_lqd_fpca/", "baseline_mean_ns": 3013224.49609375},
    {"tier": "P3", "bench": "density_fda_benchmarks", "group": "density_wasserstein_barycenter",
     "filter": r"^density_wasserstein_barycenter/", "baseline_mean_ns": 965152.9234933712},
    {"tier": "P3", "bench": "fem_smoothing_benchmarks", "group": "fem_smooth_gcv",
     "filter": r"^fem_smooth_gcv/", "baseline_mean_ns": 305012030.34375},
    {"tier": "P3", "bench": "fpca_variants_benchmarks", "group": "fpca_der",
     "filter": r"^fpca_der/", "baseline_mean_ns": 10272408.164794922},
    {"tier": "P3", "bench": "fpca_variants_benchmarks", "group": "fpca_fsvd",
     "filter": r"^fpca_fsvd/", "baseline_mean_ns": 1253884.69921875},
    {"tier": "P3", "bench": "frechet_benchmarks", "group": "frechet_global_reg",
     "filter": r"^frechet_global_reg/", "baseline_mean_ns": 260465.47412109375},
    {"tier": "P3", "bench": "fts_benchmarks", "group": "fts_ftsm",
     "filter": r"^fts_ftsm/", "baseline_mean_ns": 11648729.756103516},
    {"tier": "P3", "bench": "inference_benchmarks", "group": "inference_t_perm_test",
     "filter": r"^inference_t_perm_test/", "baseline_mean_ns": 5373165.424126053},
    {"tier": "P3", "bench": "face_benchmarks", "group": "mface_covariance",
     "filter": r"^mface_covariance/", "baseline_mean_ns": 1251420505.1875},
    {"tier": "P3", "bench": "perf_hotpaths", "group": "perf_dpca",
     "filter": r"^perf_dpca/", "baseline_mean_ns": 68211941.21308571},
    {"tier": "P3", "bench": "perf_parallelism", "group": "perf_parallelism_co_cluster",
     "filter": r"^perf_parallelism_co_cluster/", "baseline_mean_ns": 53947527.33652596},
    {"tier": "P3", "bench": "perf_parallelism", "group": "perf_parallelism_frechet_anova",
     "filter": r"^perf_parallelism_frechet_anova/", "baseline_mean_ns": 32948954.186678696},
]

RERUN_LOG_PATH = _SCRIPT_DIR / "results" / "rerun_log.json"

# Final (terminal-for-this-call) rust_chunk statuses. A chunk with the most
# recent status in this set is NOT re-scheduled by run_rust_next unless a
# --rust-requeue record appears after it.
_FINAL_CHUNK_STATUSES = frozenset({"ok", "timeout", "error", "skipped-inferred-timeout"})


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
# Rerun log — atomic read-modify-write persistence (Phase 116 Plan 03, T-116-11)
# -------------------------------------------------------------------


def _utc_now() -> str:
    """Return the current UTC time in capture_env.py's env.json timestamp format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_rerun_log(path: pathlib.Path) -> dict | None:
    """
    Load the rerun log at `path`.

    Returns None when the file does not exist. Exits (SystemExit) with a
    message naming the file when the file exists but is not valid JSON
    (fail-closed, T-116-11) — a corrupted log must never be silently treated
    as empty.
    """
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise SystemExit(f"ERROR: malformed rerun log {path}: {exc}")


def _append_rerun_log(path: pathlib.Path, record: dict) -> None:
    """
    Append `record` to the rerun log at `path` (read-modify-write).

    Writes to a temp file in the same directory, then `os.replace` for an
    atomic update (T-116-11) — a crash mid-write never leaves a truncated log.
    Requires the log to already exist (run --rerun-init first).
    """
    log = _load_rerun_log(path)
    if log is None:
        raise SystemExit(
            f"ERROR: rerun log {path} does not exist — run --rerun-init first."
        )
    log.setdefault("records", []).append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".rerun_log_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(log, f, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def _records_for_chunk(log: dict, chunk_idx: int) -> list[dict]:
    """Return all `rust_chunk` records for `chunk_idx`, in log order."""
    return [
        r for r in log.get("records", [])
        if r.get("kind") == "rust_chunk" and r.get("chunk_idx") == chunk_idx
    ]


def _final_status(log: dict, chunk_idx: int) -> str | None:
    """Return the LAST final status recorded for `chunk_idx`, or None."""
    result = None
    for r in _records_for_chunk(log, chunk_idx):
        if r.get("status") in _FINAL_CHUNK_STATUSES:
            result = r["status"]
    return result


def _chunk_is_pending(log: dict, chunk_idx: int) -> bool:
    """
    Return True when `chunk_idx` still needs to run.

    No records at all -> pending. Most recent record status is a final
    status (ok/timeout/error/skipped-inferred-timeout) -> not pending. Most
    recent record status is non-final (interrupted/requeued) -> pending
    (retried as the first chunk of the next call, per interrupted's contract).
    """
    recs = _records_for_chunk(log, chunk_idx)
    if not recs:
        return True
    return recs[-1].get("status") not in _FINAL_CHUNK_STATUSES


def rerun_init(log_path: pathlib.Path = RERUN_LOG_PATH,
               criterion_dir: pathlib.Path | None = None) -> int:
    """
    Initialize the rerun log (Phase 116 Plan 03).

    Refuses (exit 2) when the log already exists — re-initializing would
    discard resumable state. Archives any stale `target/criterion` output by
    renaming it to a UTC-timestamped sibling (T-116-09: rename, never delete),
    so only fresh post-optimization measurements are collected.
    """
    if criterion_dir is None:
        criterion_dir = _WORKSPACE_ROOT / "target" / "criterion"
    if log_path.exists():
        print(
            f"ERROR: rerun log {log_path} already exists; refusing to re-init "
            "(would discard resumable state). Delete it manually if a fresh "
            "start is really intended.",
            file=sys.stderr,
        )
        sys.exit(2)

    archive_path: pathlib.Path | None = None
    if criterion_dir.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_path = criterion_dir.parent / f"criterion.pre116-{ts}"
        criterion_dir.rename(archive_path)
        print(f"Archived stale criterion output: {criterion_dir} -> {archive_path}")

    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, cwd=str(_WORKSPACE_ROOT),
    ).stdout.strip()

    log = {
        "meta": {
            "started_utc": _utc_now(),
            "head_sha": head_sha,
            "features": "linalg,parallel",
            "thread_env": dict(THREAD_ENV),
            "criterion_archive": str(archive_path) if archive_path else None,
            "chunk_budget_s": None,
        },
        "records": [],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        json.dump(log, f, indent=2)
    print(f"Initialized rerun log: {log_path}")
    return 0


def inferred_timeout_reason(chunk: dict, log: dict) -> str | None:
    """
    Return a reason string when `chunk` should be skipped as an inferred
    timeout, or None otherwise.

    A per-id chunk is inferred-timeout when another chunk of the SAME
    (bench, group) with baseline_mean_ns <= this chunk's has final status
    timeout — a costlier sibling has no chance of finishing either.
    Group-level chunks (no sibling sharing the same group, or
    baseline_mean_ns is None) are never inferred.
    """
    if chunk.get("baseline_mean_ns") is None:
        return None
    bench, group = chunk["bench"], chunk["group"]
    siblings = [
        (idx, c) for idx, c in enumerate(RUST_RERUN_CHUNKS)
        if c["bench"] == bench and c["group"] == group
    ]
    if len(siblings) <= 1:
        return None
    for idx, c in siblings:
        if c is chunk or c.get("baseline_mean_ns") is None:
            continue
        if c["baseline_mean_ns"] <= chunk["baseline_mean_ns"] and _final_status(log, idx) == "timeout":
            return (
                f"cheaper sibling chunk {idx} ({group}, baseline "
                f"{c['baseline_mean_ns'] / 1e9:.3f}s) already timed out"
            )
    return None


def run_rust_next(budget_s: int, env: dict, runner=run_rust_bench,
                  clock=time.monotonic, log_path: pathlib.Path = RERUN_LOG_PATH) -> int:
    """
    Run pending fdars criterion chunks within a bounded wall-clock budget.

    Requires an initialized rerun log (exit 2 otherwise). Iterates
    RUST_RERUN_CHUNKS in manifest order:
      - An inferred-timeout candidate is recorded skipped-inferred-timeout
        and consumes no budget.
      - The FIRST chunk actually executed this call gets timeout = budget_s.
      - Every later chunk starts only if remaining budget >= 60s, with
        timeout = remaining.
      - status ok -> record ok, continue.
      - status error -> record error, print failure tail, return 3.
      - timeout on the FIRST executed chunk -> final timeout record
        (limit_s = budget_s).
      - timeout on a LATER chunk -> interrupted record (non-final); the call
        stops.
    Prints RUST_CHUNKS_REMAINING=<pending count> and returns 0 (unless an
    error status forced an early return 3).
    """
    log = _load_rerun_log(log_path)
    if log is None:
        print(
            f"ERROR: rerun log {log_path} not found; run --rerun-init first.",
            file=sys.stderr,
        )
        sys.exit(2)

    remaining_budget = float(budget_s)
    first_chunk_executed = False

    for idx, chunk in enumerate(RUST_RERUN_CHUNKS):
        log = _load_rerun_log(log_path)
        if not _chunk_is_pending(log, idx):
            continue

        reason = inferred_timeout_reason(chunk, log)
        if reason is not None:
            _append_rerun_log(log_path, {
                "kind": "rust_chunk", "started_utc": _utc_now(), "wall_s": 0.0,
                "status": "skipped-inferred-timeout", "chunk_idx": idx,
                "tier": chunk["tier"], "bench": chunk["bench"], "group": chunk["group"],
                "filter": chunk["filter"], "baseline_mean_ns": chunk["baseline_mean_ns"],
                "limit_s": None, "note": reason,
            })
            continue

        # Capture BEFORE flipping first_chunk_executed — this distinguishes
        # "the chunk actually executed first this call" (final timeout) from
        # a later chunk in the same call (interrupted, non-final).
        is_first_chunk_this_call = not first_chunk_executed
        if is_first_chunk_this_call:
            this_timeout = int(budget_s)
        else:
            if remaining_budget < 60:
                break
            this_timeout = int(remaining_budget)

        call_start = clock()
        status, wall_s = runner(chunk["bench"], chunk["filter"], env, this_timeout)
        elapsed = clock() - call_start
        remaining_budget -= elapsed
        first_chunk_executed = True

        base_record = {
            "kind": "rust_chunk", "started_utc": _utc_now(), "wall_s": wall_s,
            "chunk_idx": idx, "tier": chunk["tier"], "bench": chunk["bench"],
            "group": chunk["group"], "filter": chunk["filter"],
            "baseline_mean_ns": chunk["baseline_mean_ns"],
        }

        if status == "ok":
            _append_rerun_log(log_path, {**base_record, "status": "ok", "limit_s": this_timeout})
            continue
        if status == "error":
            _append_rerun_log(log_path, {**base_record, "status": "error", "limit_s": this_timeout})
            print(
                f"ERROR: chunk {idx} ({chunk['group']} {chunk['filter']}) failed "
                "(non-zero exit) — see cargo output above.",
                file=sys.stderr,
            )
            return 3
        # status == "timeout": the FIRST chunk executed this call gets a final
        # timeout record (limit_s = budget_s) and the loop CONTINUES (an
        # inferred-timeout skip for a costlier sibling costs no budget, and
        # any further chunk still obeys the >=60s remaining-budget check). A
        # LATER chunk timing out gets a non-final interrupted record and the
        # call stops immediately — only that case is a hard stop.
        if is_first_chunk_this_call:
            _append_rerun_log(log_path, {**base_record, "status": "timeout", "limit_s": budget_s})
            continue
        _append_rerun_log(log_path, {**base_record, "status": "interrupted", "limit_s": this_timeout})
        break

    log = _load_rerun_log(log_path)
    pending = sum(1 for i in range(len(RUST_RERUN_CHUNKS)) if _chunk_is_pending(log, i))
    print(f"RUST_CHUNKS_REMAINING={pending}")
    return 0


def rust_status(log_path: pathlib.Path = RERUN_LOG_PATH) -> int:
    """
    Print per-status chunk counts plus the list of non-ok chunks.

    Returns 0 only when nothing is pending, 1 otherwise.
    """
    log = _load_rerun_log(log_path)
    if log is None:
        print(
            f"ERROR: rerun log {log_path} not found; run --rerun-init first.",
            file=sys.stderr,
        )
        sys.exit(2)

    counts: dict[str, int] = {}
    non_ok: list[tuple[int, str, str, str]] = []
    for idx, chunk in enumerate(RUST_RERUN_CHUNKS):
        if _chunk_is_pending(log, idx):
            counts["pending"] = counts.get("pending", 0) + 1
            non_ok.append((idx, chunk["group"], chunk["filter"], "pending"))
            continue
        status = _final_status(log, idx) or "unknown"
        counts[status] = counts.get(status, 0) + 1
        if status != "ok":
            non_ok.append((idx, chunk["group"], chunk["filter"], status))

    print("Chunk status counts:")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    if non_ok:
        print("\nNon-ok chunks (idx, group, filter, status):")
        for idx, group, filt, status in non_ok:
            print(f"  [{idx}] {group} {filt} -> {status}")
    return 0 if counts.get("pending", 0) == 0 else 1


def rust_requeue(idx: int, log_path: pathlib.Path = RERUN_LOG_PATH) -> int:
    """
    Append a non-final `requeued` record for chunk `idx`.

    Only valid when the chunk's last FINAL status is error (after a
    bench-code fix). Refuses (exit 2) for timeout, ok, or
    skipped-inferred-timeout — a genuine timeout must never be erased.
    """
    log = _load_rerun_log(log_path)
    if log is None:
        print(
            f"ERROR: rerun log {log_path} not found; run --rerun-init first.",
            file=sys.stderr,
        )
        sys.exit(2)
    if idx < 0 or idx >= len(RUST_RERUN_CHUNKS):
        print(
            f"ERROR: chunk index {idx} out of range (0..{len(RUST_RERUN_CHUNKS) - 1}).",
            file=sys.stderr,
        )
        sys.exit(2)

    status = _final_status(log, idx)
    if status != "error":
        print(
            f"ERROR: chunk {idx}'s last final status is {status!r}, not 'error' — "
            "refusing to requeue (a genuine timeout, ok, or inferred-skip must "
            "never be erased).",
            file=sys.stderr,
        )
        sys.exit(2)

    chunk = RUST_RERUN_CHUNKS[idx]
    _append_rerun_log(log_path, {
        "kind": "rust_chunk", "started_utc": _utc_now(), "wall_s": 0.0,
        "status": "requeued", "chunk_idx": idx, "tier": chunk["tier"],
        "bench": chunk["bench"], "group": chunk["group"], "filter": chunk["filter"],
        "baseline_mean_ns": chunk["baseline_mean_ns"], "limit_s": None,
        "note": "requeued after a bench-code fix (--rust-requeue)",
    })
    print(f"Chunk {idx} ({chunk['group']} {chunk['filter']}) requeued.")
    return 0


def list_rust_chunks() -> None:
    """Print one line per manifest chunk: idx, tier, bench, filter, baseline_s."""
    print("idx tier bench filter baseline_s")
    for idx, chunk in enumerate(RUST_RERUN_CHUNKS):
        baseline_ns = chunk["baseline_mean_ns"]
        baseline_s = "-" if baseline_ns is None else f"{baseline_ns / 1e9:.3f}"
        print(f"{idx} {chunk['tier']} {chunk['bench']} {chunk['filter']} {baseline_s}")


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
    parser.add_argument(
        "--list-rust-chunks", action="store_true",
        help="Print the 67-entry RUST_RERUN_CHUNKS manifest (idx tier bench filter baseline_s).",
    )
    parser.add_argument(
        "--rerun-init", action="store_true",
        help="Initialize results/rerun_log.json (archives stale target/criterion). "
             "Refuses if the log already exists.",
    )
    parser.add_argument(
        "--rust-next", action="store_true",
        help="Run pending fdars criterion chunks within one call's --budget-s "
             "wall-clock budget (resumable). Requires --rerun-init first.",
    )
    parser.add_argument(
        "--budget-s", type=int, default=540,
        help="Wall-clock budget in seconds for one --rust-next call (default: 540).",
    )
    parser.add_argument(
        "--rust-status", action="store_true",
        help="Print per-status chunk counts and the list of non-ok chunks.",
    )
    parser.add_argument(
        "--rust-requeue", type=int, default=None, metavar="IDX",
        help="Re-queue chunk IDX after a bench-code fix. Refuses unless the "
             "chunk's last final status is error.",
    )
    args = parser.parse_args()

    # Build pinned-thread environment FIRST (Pitfall 1: set before any fork)
    env = _build_env()

    if args.list_rust_chunks:
        list_rust_chunks()
        return

    if args.rerun_init:
        sys.exit(rerun_init())

    if args.rust_next:
        sys.exit(run_rust_next(args.budget_s, env))

    if args.rust_status:
        sys.exit(rust_status(RERUN_LOG_PATH))

    if args.rust_requeue is not None:
        sys.exit(rust_requeue(args.rust_requeue))

    # --rust-bench: single thread-pinned criterion invocation under a timeout
    # (Phase 116 Plan 01). Exits 0 only when the bench completes successfully.
    # Appends a `remediation` record when a rerun log exists (Plan 03).
    if args.rust_bench is not None:
        if args.rust_filter is None:
            print("ERROR: --rust-filter is required with --rust-bench.", file=sys.stderr)
            sys.exit(2)
        status, wall_s = run_rust_bench(args.rust_bench, args.rust_filter, env, args.timeout_s)
        if RERUN_LOG_PATH.exists():
            _append_rerun_log(RERUN_LOG_PATH, {
                "kind": "remediation", "started_utc": _utc_now(), "wall_s": wall_s,
                "status": status, "bench": args.rust_bench, "filter": args.rust_filter,
                "limit_s": args.timeout_s,
            })
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
