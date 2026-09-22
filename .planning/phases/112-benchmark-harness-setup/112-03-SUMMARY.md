---
phase: 112-benchmark-harness-setup
plan: "03"
subsystem: benchmarks
tags: [benchmark-harness, numeric-gate, criterion-collector, tdd, full-op]
status: complete

dependency_graph:
  requires: [112-01, 112-02]
  provides:
    - benchmarks/gate.py (full-op, 6 ops, per-op tolerances, run_gate)
    - benchmarks/collect_rust.py (full-op criterion parse + OP_MAP)
    - benchmarks/run_all.py (full-op orchestration, gate-before-timing)
    - benchmarks/verify_inputs.py (3 fixture types checked)
    - benchmarks/test_gate.py (18 pytest unit tests)
    - benchmarks/test_collect_rust.py (7 pytest unit tests)
  affects: []

tech_stack:
  added: []
  patterns:
    - TDD (RED → GREEN) for Tasks 1 and 2
    - Per-op dispatch gate with fail-closed run_gate
    - OP_MAP criterion group → shared op label mapping
    - All subprocess calls list-form (no shell=True, T-112-07)
    - Seed-determinism regen check in verify_inputs.py

key_files:
  created:
    - benchmarks/test_collect_rust.py
    - benchmarks/test_gate.py
  modified:
    - benchmarks/collect_rust.py
    - benchmarks/gate.py
    - benchmarks/run_all.py
    - benchmarks/verify_inputs.py

decisions:
  - "Per-op fingerprint approach: fdars vs Python+R for FPCA (via fpca_dump); Python vs R for other ops (both runners emit fingerprints, criterion has timing only)"
  - "Karcher mean avg-L2 not gated (1.06 observed for n=60) — gate only on pairwise warping L-inf (1.11e-16, PASS) per RESEARCH recommendation"
  - "Depth/fpcr failures are implementation divergences for Phase 113 investigation, not harness bugs"

metrics:
  duration_minutes: 8
  completed: 2026-09-20
  tasks_completed: 3
  tasks_total: 4
  commits: 3
  plan_head_before: 28d4b44816656abff54ce83921442acce562ec46

actuals:
  tokens: 20108
  tasks: 3
  commits: 3
---

# Phase 112 Plan 03: Full-Op Gate + Collector + Orchestration Summary

Tasks 1–4 complete. Task 4 (blocking-human) APPROVED 2026-09-22 — user accepted the harness, bit-identity, and env.json fairness fields, and approved deferring per-op tolerance calibration for depth/functional_boxplot/FPCR to Phase 113 (see hand-off below).

## Phase 113 hand-off — tolerance/metric calibration (approved deferral)

The full-op agreement gate is wired and fails closed (BENCH-05 met). Empirical run flagged three op/pairs whose abs-diff tolerance is the wrong instrument for a cross-library convention difference (NOT harness bugs):
- **depth** (python vs r): max|diff| 6.52e-3 vs 1e-3 — scikit-fda `ModifiedBandDepth` and `roahd::MBD` differ ~0.65% by normalization convention; rank/Pearson correlation is the right metric. Proposed: correlation ≥ 0.99 or widen abs tol to ~1e-2.
- **functional_boxplot** (python vs r): max|diff| 6.52e-3 vs 1e-3 — same MBD-normalization root cause; same fix.
- **fpcr** (python vs r): max|diff| 4.83e-1 vs 1e-3, but Pearson corr 0.96 — Python raw-lstsq vs R `fda.usc::fregre.pc` differ in FPCA weights/scale normalization. Proposed: switch to the correlation metric used by the FPCA gate.
Phase 113 MUST calibrate these before those three ops produce a committed baseline (the gate is fail-closed, so an uncalibrated 113 run aborts on them). FPCA, basis smoothing, and elastic already pass and need no change.

Criterion collector generalized to full op set (OP_MAP), agreement gate extended to all 6 overlapping ops with per-op tolerances, and orchestrator updated to full-op sequence with gate enforced before timing.

## Tasks Completed

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | Criterion collector — full op-set parse + unit tests | 129ae7bb | DONE |
| 2 | Agreement gate — full op set, per-op tolerances + unit tests | 28879284 | DONE |
| 3 | Full-op orchestration + finalized env capture | ab35a52b | DONE |
| 4 | Human verification — full harness end-to-end | — | APPROVED 2026-09-22 |

## Task Details

### Task 1: Criterion Collector (BENCH-03)

`collect_rust.py` now exports `OP_MAP` mapping every criterion group name to a shared op label:

- FPCA: `audit_fpca`, `audit_p4_fpca`, `fdata_to_pc` → `fpca`
- Basis smoothing: `smooth_basis`, `bspline_basis` → `basis_smoothing`
- Depth: `fraiman_muniz`, `modified_band`, `audit_depth`, `audit_streaming_depth` → `depth`
- Functional boxplot: `functional_boxplot` → `functional_boxplot`
- FPCR: `fregre_lm` → `fpcr`
- Elastic: `elastic_align_pair`, `karcher_mean`, `audit_karcher`, `audit_p3_karcher`, etc. → `elastic`

Pitfall 3 (base/ ignored), Pitfall 4 (BenchmarkId slash paths), T-112-06 (path-traversal guard) all unit-tested. 7 pytest tests, all green.

### Task 2: Agreement Gate (BENCH-05)

`gate.py` now exports `check_agreement(op, fdars_fp, other_fp)` with per-op dispatch:

| Op | Metric | Tolerance | Pass criterion |
|----|--------|-----------|----------------|
| fpca | sign-aligned Pearson correlation | 0.99 | min corr > 0.99 |
| basis_smoothing | max abs diff on smoothed_values | 1e-3 | max_diff < 1e-3 |
| depth | max abs diff on depths vector | 1e-3 | max_diff < 1e-3 |
| functional_boxplot | max abs diff on depths vector | 1e-3 | max_diff < 1e-3 |
| fpcr | max abs diff on fitted_values | 1e-3 | max_diff < 1e-3 |
| elastic | L-inf on warping_values | 1e-2 | L-inf < 1e-2 |

`run_gate(results)` iterates all (op, lib) pairs excluding rust, returns False fail-closed with `GATE FAIL [<lib>] <op>` output. Pitfall 2 (sign-flip) handled: sign-flipped FPCA PASSES (verified in test). Pitfall 5: random-projection depth excluded from `_GATED_OPS`. 18 pytest tests, `--self-test` all green.

### Task 3: Full-Op Orchestration (BENCH-04)

`run_all.py` now orchestrates all 6 ops:

1. Generate shared CSV fixtures (all types)
2. Python runner (all ops or single op)
3. R runner (all ops or single op)
4. cargo bench for regression/depth/alignment/basis bench files (skipped in `--dry-run`)
5. Collect criterion output (collect_rust.py)
6. Capture env.json (capture_env.py) — thread_env all "1" verified
7. **Gate** — FPCA: fdars vs Python + R; other ops: Python vs R — abort on fail
8. Write timing_comparison.json (only after gate passes)

`verify_inputs.py` now checks all 3 fixture files:
- `curves_n60_m50.csv` — seed-deterministic, Python + R bit-identical
- `y_n60.csv` — seed-deterministic, Python + R bit-identical
- `elastic_n2_m50.csv` — seed-deterministic, Python + R bit-identical

## Task 4 Evidence (for Human Review)

### 1. verify_inputs.py Output

```
Verifying [curves_n60_m50]: curves_n60_m50.csv
  File SHA-256:   52dd197401496f77...
  Python content: 6a6a29f24fdb7b40...
  R content:      a5 36 42 3e 95 6...
  Regen hash:     52dd197401496f77...
  Regen matches: YES (seed-deterministic)
  [curves_n60_m50]: OK

Verifying [y_n60]: y_n60.csv
  File SHA-256:   575afc217160c60d...
  Regen matches: YES (seed-deterministic)
  [y_n60]: OK

Verifying [elastic_n2_m50]: elastic_n2_m50.csv
  File SHA-256:   e67cab4e6159abfc...
  Regen matches: YES (seed-deterministic)
  [elastic_n2_m50]: OK

BIT-IDENTITY CHECK: PASSED
```

### 2. Full-Op run_gate Per-Op Agreement Metrics (n=60, m=50, smoke run)

| Op | Library pair | Metric | Value | Tolerance | Status |
|----|-------------|--------|-------|-----------|--------|
| fpca | fdars vs python | min\|corr\| | 0.998271 | >0.99 | **PASS** |
| fpca | fdars vs r | min\|corr\| | 1.000000 | >0.99 | **PASS** |
| basis_smoothing | python vs r | max\|diff\| | 5.55e-15 | <1e-3 | **PASS** |
| depth | python vs r | max\|diff\| | 6.52e-03 | <1e-3 | **FAIL — see below** |
| functional_boxplot | python vs r | max\|diff\| | 6.52e-03 | <1e-3 | **FAIL — see below** |
| fpcr | python vs r | max\|diff\| | 4.83e-01 | <1e-3 | **FAIL — see below** |
| elastic (warp) | python vs r | L-inf | 1.11e-16 | <1e-2 | **PASS** |

**FPCA per-component details (fdars vs python):**

| Component | Correlation | Sign-flipped |
|-----------|-------------|--------------|
| PC1 | 1.000000 | No |
| PC2 | 1.000000 | No |
| PC3 | 0.999602 | No |
| PC4 | 0.998817 | No |
| PC5 | 0.998271 | No |

**FPCA per-component details (fdars vs r):**

| Component | Correlation | Sign-flipped |
|-----------|-------------|--------------|
| PC1 | 1.000000 | No |
| PC2 | 1.000000 | No |
| PC3 | 1.000000 | No |
| PC4 | 1.000000 | No |
| PC5 | 1.000000 | No |

**Failing ops — investigation findings:**

**depth/functional_boxplot (max|diff|=6.52e-3):** scikit-fda `ModifiedBandDepth` and `roahd::MBD` compute MBD with slightly different normalizations. Both libraries produce depth vectors in the range 0.22–0.51 (correct order, high rank correlation), but the absolute values differ by up to 0.65%. This is an inter-library normalization divergence, not a harness error. The 1e-3 tolerance is too tight for cross-library MBD comparison. **Phase 113 action: widen tolerance to 1e-2 or switch to rank-correlation metric for depth.**

**fpcr (max|diff|=0.483, corr=0.96):** Python uses scikit-fda FPCA (trapezoid weights) + numpy lstsq; R uses `fda.usc::fregre.pc` (different FPCA weight scheme, different scale normalization). The fitted values are highly correlated (0.96) but differ in scale. **Phase 113 action: use sign-aligned Pearson correlation metric (>0.99) for FPCR fitted values, matching the FPCA gate approach.**

**elastic Karcher mean avg L2 = 1.06:** Not gated (only pairwise warping is gated). The large Karcher mean value difference reflects both libraries converging to different representations of the mean in the original function space (not the warp space). Pairwise warping L-inf = 1.11e-16 (essentially identical).

### 3. benchmarks/results/env.json

```json
{
  "timestamp": "2026-09-20T21:06:09Z",
  "cpu": {"model": "13th Gen Intel(R) Core(TM) i9-13900H", "cores_logical": 20},
  "rust": {"version": "rustc 1.97.0 (2d8144b78 2026-07-07)", "features": "linalg,parallel"},
  "python": {"version": "3.14.7", "numpy_version": "2.5.3",
             "numpy_blas": "\"name\": \"scipy-openblas\","},
  "scikit_fda": {"version": "0.10.1"},
  "r": {"version": "4.6.1", "blas": "/usr/lib/libblas.so.3.12.0",
        "lapack_version": "/usr/lib/liblapack.so.3.12.0"},
  "r_packages": {"fda": "6.3.0", "fda.usc": "2.2.0", "roahd": "1.4.3",
                 "fdasrvf": "2.5.0", "bench": "1.1.4", "refund": "0.1.40",
                 "fdapace": "0.6.0", "jsonlite": "2.0.0"},
  "thread_env": {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"
  },
  "criterion_version": "0.5"
}
```

All six thread env vars are "1". BLAS/version metadata is complete. scikit-fda 0.10.1 confirmed.

## Deviations from Plan

### Auto-fixed Issues

None — plan executed as written.

### Flagged for Phase 113 Investigation

**1. [Rule 1 - Bug flag] depth/functional_boxplot tolerance too tight**
- **Found during:** Task 4 evidence collection
- **Issue:** scikit-fda `ModifiedBandDepth` and `roahd::MBD` differ by 6.52e-3 (max abs diff). The 1e-3 tolerance fails on a known inter-library normalization difference.
- **Phase 113 action:** Widen depth/functional_boxplot tolerance to 1e-2, or use rank-correlation metric
- **Not a harness bug** — the gate correctly flags this; the tolerance needs calibration

**2. [Rule 1 - Bug flag] fpcr gate metric mismatch**
- **Found during:** Task 4 evidence collection
- **Issue:** Python uses raw lstsq fitted values; R `fregre.pc` uses different scale normalization. Max abs diff = 0.483 but Pearson corr = 0.96.
- **Phase 113 action:** Switch fpcr gate metric to sign-aligned Pearson correlation >0.99 (same as FPCA gate approach)
- **Not a harness bug** — the gate correctly flags this; the metric needs adjustment

**3. Karcher mean avg L2 large but not gated**
- **Found during:** Task 4 evidence collection
- **Value:** avg L2 = 1.06 between scikit-fda and fdasrvf Karcher means
- **Status:** Not gated per RESEARCH "Practical recommendation" (gate only pairwise warp L-inf)
- **Phase 113 action:** Investigate whether this is a convergence difference or a different reference frame

## Known Stubs

None.

## Threat Flags

None beyond the T-112-06/07/08 already in the plan's threat register.

## Self-Check: PASSED

- benchmarks/collect_rust.py: EXISTS
- benchmarks/gate.py: EXISTS
- benchmarks/run_all.py: EXISTS
- benchmarks/verify_inputs.py: EXISTS
- benchmarks/test_collect_rust.py: EXISTS
- benchmarks/test_gate.py: EXISTS
- Task 1 commit 129ae7bb: EXISTS
- Task 2 commit 28879284: EXISTS
- Task 3 commit ab35a52b: EXISTS
- pytest test_collect_rust.py: 7 passed
- pytest test_gate.py: 18 passed
- gate.py --self-test: PASSED
- verify_inputs.py: PASSED (3 fixtures)
- run_all.py --env-only + env.json assertions: PASSED
