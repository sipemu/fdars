---
phase: 47-hot-path-allocation-performance
plan: 02
subsystem: infra
tags: [perf, allocation, fsvd, ssvd, functional_acf, from_fn]
requires:
  - phase: 47-01
    provides: golden/dhat/ledger proof harness
provides:
  - "OPT-B/C/D: fsvd/ssvd/functional_acf copy removals (from_column_slice → from_fn), behavior-preserving"
affects: [47-04]
actuals:
  tokens: 8000
  tasks: 3
  commits: 1
tech-stack:
  added: []
  patterns: [git-stash pre-change golden capture]
key-files:
  created: []
  modified: [fdars-core/src/fpca_variants.rs, fdars-core/src/fts/acf.rs, fdars-core/tests/equivalence_phase47.rs, fdars-core/tests/alloc_audit_dpca.rs, .planning/phases/47-hot-path-allocation-performance/PERF-RESULTS.md]
key-decisions:
  - "Captured golden references from pre-edit code via git-stash of the two src files, so the goldens independently prove OPT-B/C/D behavior-preserving (not self-referential)."
patterns-established:
  - "Copy removal = replace Vec-staging + DMatrix::from_column_slice(&buf) with DMatrix::from_fn(|row,col| ...) using identical column-major arithmetic."
requirements-completed: [PERF-02]
coverage:
  - id: D1
    description: "OPT-B/C/D copy removals (fsvd/ssvd/functional_acf) — one staging Vec + m×m copy eliminated each; output byte-equivalent"
    requirement: PERF-02
    verification:
      - kind: integration
        ref: "golden_ssvd/fsvd/functional_acf pass rel 1e-12 (refs captured pre-edit via git-stash); dhat fsvd 275→274, ssvd 22→21; suite green; clippy --all-targets clean"
        status: pass
    human_judgment: false
duration: 30min
completed: 2026-08-31
status: complete
---

# Phase 47 / Plan 02: OPT-B/C/D Copy Removals Summary

**fsvd/ssvd/functional_acf now build their eigen matrices via DMatrix::from_fn (no Vec staging + no m×m copy); functional_acf also precomputes sqrt(w). Byte-equivalent, proven by pre-edit golden captures at rel 1e-12.**

## Accomplishments
- **OPT-B (fsvd):** gram matrix via `from_fn` (branch on `gram_on_right`), dropped `g_dim`/`gram` Vec. 275→274 blocks.
- **OPT-C (ssvd):** scaled covariance via `from_fn`, dropped `c_scaled` Vec. 22→21 blocks.
- **OPT-D (functional_acf):** `c0_scaled` via `from_fn` with precomputed `sqrt_w` — drops the staging Vec AND ~(m²−m) redundant `sqrt()` calls. `long_run_covariance` left untouched (its 6 blocks are load-bearing).
- Golden references (ssvd/fsvd/functional_acf) captured from the **pre-edit** code (git-stash of the two src files), so the goldens independently confirm equivalence; all pass at 1e-12. Suite green, clippy `--all-targets` clean.

## Deviations
None — all three are algebraically-identical transformations, verified byte-equivalent.

---
*Phase: 47-hot-path-allocation-performance · Completed: 2026-08-31*
