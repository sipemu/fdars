---
phase: 47-hot-path-allocation-performance
plan: 01
subsystem: infra
tags: [perf, allocation, dpca, fts, criterion, dhat, tracer]
requires:
  - phase: 46
    provides: PROF-01 ranked hot-path targets + before-numbers
provides:
  - Permanent proof pipeline (perf_hotpaths bench, equivalence_phase47 golden, alloc_audit_dpca dhat, PERF-RESULTS.md)
  - OPT-A: fts::dpca allocation −54% (17,739→8,139 blocks), behavior-preserving
affects: [47-02, 47-03, 47-04, 51-benchmark-coverage-regression-guards]
actuals:
  tokens: 11000
  tasks: 4
  commits: 1
tech-stack:
  added: []
  patterns: [golden-equivalence capture-then-assert, permanent perf bench, dhat hard-assert regression gate]
key-files:
  created: [fdars-core/benches/perf_hotpaths.rs, fdars-core/tests/equivalence_phase47.rs, fdars-core/tests/alloc_audit_dpca.rs, .planning/phases/47-hot-path-allocation-performance/PERF-RESULTS.md]
  modified: [fdars-core/Cargo.toml, fdars-core/src/fts/spectral.rs]
key-decisions:
  - "OPT-A index-sort refactor materializes only ncomp eigenvectors (was all m) + from_fn (drops scaled Vec)."
  - "DEVIATION: <1000-block target was optimistic; residual ~8k blocks are spectral_density + nalgebra SymmetricEigen internals (out of OPT-A scope). Achieved −54% clears the ≥25% bar."
patterns-established:
  - "Golden reference captured pre-change (print {:.17e}, paste consts); assert_rel_close at 1e-12."
requirements-completed: [PERF-01, PERF-02]
coverage:
  - id: D1
    description: "OPT-A dpca allocation reduction (17,739→8,139 blocks, −54%) proven behavior-preserving"
    requirement: PERF-02
    verification:
      - kind: integration
        ref: "cargo test --features dhat-heap,linalg count_dpca_allocations_n200_m50 => <9000 (8,139); golden_dpca_n50_m10 => pass rel 1e-12"
        status: pass
    human_judgment: false
  - id: D2
    description: "Permanent proof pipeline scaffolded (bench + golden + dhat + ledger), wired end-to-end"
    requirement: PERF-01
    verification:
      - kind: integration
        ref: "cargo build --benches; perf_hotpaths registered; suite green; clippy --all-targets clean"
        status: pass
    human_judgment: false
duration: 40min
completed: 2026-08-31
status: complete
---

# Phase 47 / Plan 01: OPT-A Tracer Summary

**fts::dpca allocation cut 54% (17,739→8,139 blocks) via index-sort eigenvector materialization, behavior-preserving (golden 1e-12), with the permanent proof pipeline (bench + golden + dhat + ledger) established end-to-end.**

## Accomplishments
- Scaffolded `benches/perf_hotpaths.rs` (permanent `[[bench]]`, Phase 51 BENCH-02 guard; dpca/face_covariance/fem_smooth cells), `tests/equivalence_phase47.rs` (golden helpers), `tests/alloc_audit_dpca.rs` (dhat), `PERF-RESULTS.md` (ledger + environment).
- **OPT-A:** refactored `eigen_at_frequency` to index-sort eigenvalues and collect only the retained `ncomp` eigenvectors (was all `m` per frequency) + `DMatrix::from_fn` (drops `scaled` Vec). dpca @ n200_m50: **17,739 → 8,139 blocks (−54%)**, output equivalent (golden rel 1e-12). dpca wall-time baseline ~62ms (powersave, informational).

## Deviations
- **OPT-A block target `<1000` not reached (achieved 8,139).** The estimate assumed the eigenvector collection was ~all of dpca's allocation; in fact ~8k blocks come from `spectral_density` (called inside dpca) + nalgebra `SymmetricEigen` per-frequency internals, both outside OPT-A's `eigen_at_frequency` scope. The −54% achieved clears the locked ≥25% bar; dhat guard set to `<9000`. Documented in PERF-RESULTS.md. Further reduction needs a workspace-reusing eigensolver (risky, out of scope).

## Next
Wave 2 (Plan 02) reuses the golden/dhat/ledger harness for OPT-B/C/D.

---
*Phase: 47-hot-path-allocation-performance · Completed: 2026-08-31*
