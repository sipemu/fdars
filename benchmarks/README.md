# benchmarks/

External cross-language benchmark harness for fdars vs scikit-fda (Python) and the R FDA
ecosystem. This directory is **NOT** a Cargo workspace member and has **no fdars-core
dependency** — it consumes only the criterion output files under `target/criterion/`.

## Architecture

```
generate_data.py   -> data/curves_n{n}_m{m}.csv   (seeded, bit-identical across runners)
python_runner.py   -> results/python.json
r_runner.R         -> results/r.json
collect_rust.py    -> results/rust.json            (reads target/criterion/*/new/estimates.json)
gate.py            -> validates numeric agreement BEFORE timing is recorded
run_all.py         -> orchestrator; sets OMP/OPENBLAS/MKL/RAYON_NUM_THREADS=1
capture_env.py     -> results/env.json             (version + BLAS + CPU + thread-env metadata)
verify_inputs.py   -> bit-identical CSV spot-check
```

## Setup

### Python (one-time)

```bash
# From the workspace root:
python3 -m venv benchmarks/.venv
benchmarks/.venv/bin/pip install -r benchmarks/requirements.lock
```

Activate the venv before running scripts directly:
```bash
source benchmarks/.venv/bin/activate
```

Or use the venv Python directly:
```bash
benchmarks/.venv/bin/python benchmarks/run_all.py --op fpca
```

### R (one-time)

Required packages (already installed on this machine as of 2026-09-20):
- `fda` 6.3.0, `fda.usc` 2.2.0, `roahd` 1.4.3, `fdasrvf` 2.5.0
- `bench` 1.1.4, `refund` 0.1-40, `fdapace` 0.6.0, `jsonlite` 2.0.0

To restore from `renv.lock`:
```bash
Rscript -e "renv::restore()"
```

Note: `fdapace` is installed (0.6.0); its `FPCA()` function provides an alternative FPCA
implementation. The **gated** R FPCA path (`r_svd` in `comparison.csv`) is a hand-rolled,
Simpson-weighted SVD port that agrees with fdars at subspace cosine 1.0 — not `fda::pca.fd`.
R's own native `fda::pca.fd` (6.3.0) is timed separately as `fpca_native_r`, an
ecosystem-realistic number shown alongside `r_svd` but never gated against fdars (it operates
in B-spline coefficient space and diverges numerically from fdars at PC3–PC5).

On a clean clone, `renv::restore()` above installs into the project's `renv` library, but
`.Rprofile` and `renv/` are both gitignored, so `renv` is **not auto-activated** by simply
`cd`-ing into the clone — the `renv::restore()` invocation must be run explicitly (as shown)
before any R runner script. This restore procedure is documented here for reproduction; it was
not re-executed in Phase 116 (the R packages were already installed on the machine that
produced the Phase 116 numbers, from the original Phase 112 setup).

After restoring, verify the installed package versions match `renv.lock`:
```bash
Rscript benchmarks/r_deps_check.R
```

### Rust / criterion

No setup needed — timings are read from `target/criterion/` which is populated by running
`cargo bench -p fdars-core` from the workspace root.

## Running

```bash
# Full FPCA tracer (generate + Python/R runners + criterion collect + gate + env):
benchmarks/.venv/bin/python benchmarks/run_all.py --op fpca

# Dry-run (no cargo bench):
benchmarks/.venv/bin/python benchmarks/run_all.py --op fpca --dry-run

# Capture environment metadata only:
benchmarks/.venv/bin/python benchmarks/run_all.py --env-only

# Gate self-test (proves pass-on-agreement and fail-on-mismatch):
benchmarks/.venv/bin/python benchmarks/gate.py --self-test
```

## Full re-run (Phase 116)

The single end-to-end reproduction command, thread-pinned, all three ecosystems, the full
N×M grid:

```bash
benchmarks/.venv/bin/python benchmarks/run_all.py --full
```

`--full` wraps the exact resumable sequence below into one call. The published Phase 116 run
used the sequence directly (not one blocking `--full` call) so an interrupted step could
resume without losing prior progress — this is the same sequence, run manually:

```bash
# 1. Initialize the resumable rerun log (archives stale target/criterion/ output).
benchmarks/.venv/bin/python benchmarks/run_all.py --rerun-init

# 2. Generate the seed-42 CSV fixtures for all six grid cells.
benchmarks/.venv/bin/python benchmarks/run_all.py --generate-grid-data

# 3. Pre-build every criterion bench binary (resumable; cargo itself resumes an
#    interrupted build on retry).
benchmarks/.venv/bin/python benchmarks/run_all.py --prebuild

# 4. Run pending fdars criterion chunks in bounded calls until none remain.
#    Repeat until `--rust-status` reports 0 remaining (a chunk's first-executed
#    timeout is final; a later chunk's timeout is "interrupted" and retried).
benchmarks/.venv/bin/python benchmarks/run_all.py --rust-next --budget-s 540
benchmarks/.venv/bin/python benchmarks/run_all.py --rust-status

# 5. Run the scikit-fda + R grid, one cell at a time, gate-before-timing per cell.
for cell in 100x50 100x200 300x50 300x200 500x50 500x200; do
  benchmarks/.venv/bin/python benchmarks/run_all.py --grid --cell "$cell" --cell-timeout-s 540
done

# 6. Refresh the environment snapshot for the after-run.
benchmarks/.venv/bin/python benchmarks/run_all.py --env-only

# 7. Collect the fdars criterion output into results/rust.json.
benchmarks/.venv/bin/python benchmarks/collect_rust.py

# 8. Assemble the gate-passed comparison.csv from all three ecosystems' results.
benchmarks/.venv/bin/python benchmarks/assemble_comparison.py --gate-passed

# 9. Record target/ disk usage into the rerun log.
benchmarks/.venv/bin/python benchmarks/run_all.py --record-footprint
```

Then render and verify the report:

```bash
benchmarks/.venv/bin/python benchmarks/render_report.py
benchmarks/.venv/bin/python benchmarks/render_report.py --check
benchmarks/.venv/bin/python benchmarks/render_report.py --check-prose
```

## Fairness Controls

All runner processes inherit `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1` — set by `run_all.py` in the subprocess environment
before forking any runner. This pins both Python (numpy/OpenBLAS) and Rust (rayon)
to single-threaded execution for a fair comparison.

## Result Schema

Each runner emits a list of records to its `results/*.json`:

```json
[{
  "op": "fpca",
  "library": "python",
  "n": 60,
  "m": 50,
  "mean_ns": 1234567.0,
  "result_fingerprint": {
    "loadings": [[...], ...]
  }
}]
```

The `result_fingerprint` carries the numeric output the agreement gate checks — FPCA loadings
matrix (ncomp x m) for FPCA ops.

`assemble_comparison.py --gate-passed` merges the three runners' JSON output into
`results/comparison.csv`, schema `op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,
rsd_pct,ci_width_pct,noise_gate,source,notes`. `results/comparison_baseline.csv` uses the
identical schema — it is the Phase 113 baseline, frozen byte-identically before the Phase 116
re-run overwrote `comparison.csv`, and is the "before" side of every before/after table in
`documentation/PERFORMANCE.md`. `benchmarks/results/hotspot_ranking.md` is the Phase 113
leverage ranking computed from what is now `comparison_baseline.csv` — it was never
re-generated against the after-run numbers, since its purpose (ranking optimization targets)
is now historical.

`results/rerun_log.json` (`{meta, records}`) is the resumable scheduler's audit trail: `meta`
carries `head_sha`/`started_utc`/`thread_env`/the archived pre-run criterion directory;
`records` is an append-only list of `rust_chunk`/`grid_cell`/`prebuild`/`remediation`/
`footprint` entries, each with a `status` (`ok`/`timeout`/`skipped-inferred-timeout`/
`interrupted`/`error`) and a `wall_s` duration. `render_report.py`'s `not-rerun` and `runtime`
blocks are both derived from this file.

`results/ab/manifest.json` (`{entries: [...]}`) is the controlled-A/B evidence register: each
`kind: "ab"` entry names a `before`/`after` criterion `estimates.json` pair plus the SHAs and
thread configuration under which they were measured; `kind: "rebench"` entries (no before/after
pair) are a tightened re-measurement, not an A/B. `render_report.py`'s `ab-evidence` and
`key-findings` blocks read this file directly.

## Numeric Agreement Gate

The gate runs sign-aligned Pearson correlation on FPCA loadings before any timing is recorded.
Threshold: all components must achieve |correlation| > 0.99.

The sign convention matches fdars `fix_svd_signs` (regression.rs): for each loading vector,
the element with the largest absolute value is made strictly positive.

### Python 3.14 multimethod patch

scikit-fda's dependency chain pulls in `multimethod==2.1`, whose `__init__.py` calls
`type.__new__(cls, str(tp), bases, namespace)` inside a `typing`-derived metaclass helper.
Python 3.14 tightened metaclass-compatibility checks in `type.__new__`, so this call now raises
`TypeError: multiple bases have instance lay-out conflict`-style errors on import — scikit-fda
fails to import under Python 3.14 with an unpatched `multimethod` 2.1. The fix (patched
in-place, only inside the pinned `.venv`, never upstreamed) wraps the `type.__new__` call in a
`try`/`except TypeError` that falls back to reconciling the bases' metaclasses, and finally to
`object` if no reconciliation is possible.

The patch is committed at `benchmarks/patches/multimethod-2.1-py314.patch`, generated by
diffing the pristine PyPI wheel (verified against the venv's own
`multimethod-2.1.dist-info/RECORD` hash — proving it is the same file `pip install` originally
placed there) against the venv's patched copy. Apply it after installing from
`requirements.lock`, on Python 3.14 only:

```bash
patch -p1 -d benchmarks/.venv/lib/python3.14/site-packages \
    < benchmarks/patches/multimethod-2.1-py314.patch
```

## Notes

- `data/` is gitignored (generated locally, never committed).
- `results/*` is gitignored by default, with explicit `!` negations re-including every
  committed evidence artifact: `comparison.csv`, `comparison_baseline.csv`, `env.json`,
  `env_baseline.json`, `hotspot_ranking.md`, `rust.json`, `rerun_log.json`, the per-cell
  `python_n*_m*.json`/`r_n*_m*.json` files, and everything under `ab/`. A bare `results/`
  ignore rule cannot be selectively re-included (a pattern matching an ignored directory's
  contents is itself ignored), which is why the rule is `results/*` plus explicit negations
  rather than a directory-level ignore. Scratch outputs not in this list (`python.json`, `r.json`,
  `tracer_comparison.csv`, ad hoc single-op runs) stay gitignored.
- `.venv/` is gitignored.
- `fdasrvf` version: 2.5.0 is installed (research noted 2.4.4 was originally planned; 2.5.0
  installed by renv's resolver — this is expected and fine per execution notes).
