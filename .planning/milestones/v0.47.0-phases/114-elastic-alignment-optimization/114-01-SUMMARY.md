---
phase: 114-elastic-alignment-optimization
plan: "01"
subsystem: alignment
tags: [benchmark, golden-fixture, elastic, baseline, numeric-identity]
dependency_graph:
  requires: []
  provides:
    - fdars-core/tests/equivalence_phase114.rs
    - .planning/phases/114-elastic-alignment-optimization/114-baseline-criterion/audit_p3_elastic_cross_banded__n500_m50/estimates.json
    - .planning/phases/114-elastic-alignment-optimization/114-baseline-criterion/audit_p3_elastic_cross_banded__n500_m50/BASELINE.md
  affects:
    - Plan 114-02 (consumes baseline CI + golden fixture for OPT-01 non-overlapping-CI check)
tech_stack:
  added: []
  patterns:
    - golden-value fixture (equivalence_phase47.rs pattern verbatim)
    - criterion BEFORE-baseline commit (D-Measure-first, Anti-Pattern 3)
key_files:
  created:
    - fdars-core/tests/equivalence_phase114.rs
    - .planning/phases/114-elastic-alignment-optimization/114-baseline-criterion/audit_p3_elastic_cross_banded__n500_m50/estimates.json
    - .planning/phases/114-elastic-alignment-optimization/114-baseline-criterion/audit_p3_elastic_cross_banded__n500_m50/benchmark.json
    - .planning/phases/114-elastic-alignment-optimization/114-baseline-criterion/audit_p3_elastic_cross_banded__n500_m50/BASELINE.md
  modified: []
decisions:
  - "Golden constants captured at generate_curves(20,50) matching the bench generator exactly (amplitude/phase formula byte-identical to audit_hotpaths.rs:generate_curves), ensuring bench numbers and golden constants are commensurable."
  - "Task 2 committed before benchmark completed (probe test confirms values from un-optimized code; bench only needed for Task 1 baseline CI)."
  - "Baseline SHA is 8c0b1d7d (post golden-fixture commit) — src/ is unchanged from a9d6318f so the measurement is on un-optimized code."
metrics:
  duration_seconds: 161
  completed: "2026-09-23"
  tasks_completed: 2
  files_changed: 4
status: complete
actuals:
  tokens: 5500
  tasks: 2
  commits: 2
  plan_head_before: a9d6318f58760ae893f00f07cf9e33e1fb6b1bcc
---

# Phase 114 Plan 01: BEFORE-Baseline + Golden Fixture Summary

Captured the committed BEFORE-optimization evidence for the elastic-alignment band-iteration
optimization (OPT-01): a criterion baseline on the ranked banded target cell and a golden-value
test locking the current elastic self-distance output at n=20,m=50 to literal 1e-10 constants.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | BEFORE-baseline criterion (tracer) | `13be5201` | estimates.json, benchmark.json, BASELINE.md |
| 2 | Golden-value fixture (auto/tdd) | `8c0b1d7d` | fdars-core/tests/equivalence_phase114.rs |

## BEFORE-Baseline Numbers (for Plan 02)

| Metric | Value |
|--------|-------|
| **mean** | **7,310,073,752 ns (≈ 7.310 s)** |
| CI lower (95%) | 7,072,703,467 ns (≈ **7.073 s**) |
| CI upper (95%) | 7,552,415,933 ns (≈ 7.552 s) |
| git SHA | `8c0b1d7d` (un-optimized src — identical to plan-start SHA `a9d6318f`) |
| features | `linalg,parallel` |
| bench target | `audit_p3_elastic_cross_banded/n500_m50` |

**Plan 02 success gate:** after-optimization CI upper bound must be < 7,072,703,467 ns (CI lower
of this baseline) to prove a non-overlapping-CI speedup for OPT-01.

## Golden Fixture Values (for reference)

### Unbanded path (`elastic_self_distance_matrix`, lambda=0.0)

| Cell | Value |
|------|-------|
| `(0,1)` | `1.17458127979673499e0` |
| `(0,19)` | `4.85253341940299066e-1` |
| `(9,10)` | `8.69320275292602251e-1` |
| `(19,18)` | `1.56594861489696280e0` |
| `(5,15)` | `6.69029915915371198e-1` |

### Banded path (`elastic_self_distance_matrix_banded`, lambda=0.0, band_frac=0.1)

| Cell | Value |
|------|-------|
| `(0,1)` | `2.15003440465124251e0` |
| `(0,19)` | `4.85253341940299066e-1` |
| `(9,10)` | `1.35275846614977113e0` |
| `(19,18)` | `2.90036187866837514e0` |
| `(5,15)` | `8.10038567393554620e-1` |

## Deviations from Plan

None — plan executed exactly as written.

## Tracer Feedback Gate

Both `<verify>` commands re-run end-to-end after commits:
- Task 1: `estimates.json` present, mean=7,310,073,752 ns, CI=[7,072,703,467, 7,552,415,933] (positive, valid) — PASS
- Task 2: `cargo test --test equivalence_phase114`: 2 passed, 0 failed — PASS

## Self-Check: PASSED

- [x] `fdars-core/tests/equivalence_phase114.rs` exists and has 136 lines
- [x] `114-baseline-criterion/audit_p3_elastic_cross_banded__n500_m50/estimates.json` exists with valid mean/CI
- [x] `BASELINE.md` exists with provenance (SHA, features, machine)
- [x] 2 commits from plan-head `a9d6318f` to HEAD (`git rev-list --count` = 2)
- [x] Zero `fdars-core/src/` files modified (`git diff --name-only a9d6318f..HEAD | grep src`: none)
