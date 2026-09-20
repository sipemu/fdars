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
implementation. The primary R FPCA path uses `fda::pca.fd` (6.3.0).

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

## Numeric Agreement Gate

The gate runs sign-aligned Pearson correlation on FPCA loadings before any timing is recorded.
Threshold: all components must achieve |correlation| > 0.99.

The sign convention matches fdars `fix_svd_signs` (regression.rs): for each loading vector,
the element with the largest absolute value is made strictly positive.

## Notes

- `data/` and `results/` are gitignored (generated); commit selectively when publishing baselines.
- `.venv/` is gitignored.
- `fdasrvf` version: 2.5.0 is installed (research noted 2.4.4 was originally planned; 2.5.0
  installed by renv's resolver — this is expected and fine per execution notes).
