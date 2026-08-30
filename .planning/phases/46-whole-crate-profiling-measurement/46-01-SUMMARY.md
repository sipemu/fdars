---
phase: 46-whole-crate-profiling-measurement
plan: 01
subsystem: infra
tags: [profiling, criterion, dhat, benchmarking, fpca_variants, tracer]
requires: []
provides:
  - Proven throwaway-probe pipeline (author → temp-register → run → dhat → PROF-01 row)
  - PROF-01 skeleton with Environment section + first grounded fpca_variants::fsvd row
  - Reusable Environment values (governor powersave, 20 cores) for Plan 02
affects: [46-02]
actuals:
  tokens: 9000
  tasks: 3
  commits: 1
tech-stack:
  added: []
  patterns: [throwaway criterion probe bench, temp [[bench]] registration, dhat integration-test probe]
key-files:
  created:
    - fdars-core/benches/probe_fpca_variants.rs
    - fdars-core/tests/alloc_audit_new_subsystems.rs
    - .planning/phases/46-whole-crate-profiling-measurement/PROF-01-hotpath-targets.md
  modified:
    - fdars-core/Cargo.toml
key-decisions:
  - "Environment: governor=powersave (LOW-CONFIDENCE), 20 logical cores, features linalg,parallel — reused by Plan 02."
  - "fpca_variants::fsvd is strongly M-dominated (O(M³) gram eigendecomposition at src/fpca_variants.rs:488)."
patterns-established:
  - "Probe bench = verbatim copy of audit_hotpaths.rs structure; temp [[bench]] entry carries THROWAWAY comment; removed in Plan 02."
requirements-completed: [PROF-01]
coverage:
  - id: D1
    description: "End-to-end profiling tracer proven on fpca_variants: probe bench compiles+runs (4 cells), dhat probe runs, PROF-01 skeleton written with Environment + measured row"
    requirement: PROF-01
    verification:
      - kind: other
        ref: "cargo bench --bench probe_fpca_variants (4 n*_m* cells with time:); cargo test --features dhat-heap,linalg count_fsvd_allocations_n200_m50 => 1 passed; PROF-01 doc contains src/fpca_variants.rs:488"
        status: pass
    human_judgment: false
duration: 25min
completed: 2026-08-30
status: complete
---

# Phase 46 / Plan 01: Profiling Tracer Summary

**End-to-end throwaway-probe pipeline proven on fpca_variants::fsvd — 4 criterion cells + a dhat allocation probe feed the first grounded row of PROF-01, with baseline suite green and zero src edits.**

## Performance
- **Duration:** ~25 min
- **Tasks:** 3
- **Files modified:** 4 (2 created dev-harness, 1 doc, Cargo.toml temp entry)

## Accomplishments
- **Task 1 (Wave 0):** TMPDIR cache created, `target/debug/{incremental,examples}` freed, environment captured (governor `powersave`, 20 cores), baseline suite **green** (13 test groups, 0 failed) — the phase's zero-behavior-change anchor.
- **Task 2:** Authored `benches/probe_fpca_variants.rs` (verbatim `audit_hotpaths.rs` structure), temp-registered it in Cargo.toml with the THROWAWAY marker, compiled + ran 4 N×M cells:
  - n50_m50 = 601 µs · n200_m50 = 999 µs · n50_m200 = **12.09 ms** · n1000_m50 = 2.88 ms
- **Task 3:** Authored `tests/alloc_audit_new_subsystems.rs` dhat probe for `fsvd` (n200_m50 → 275 blocks / 600 KB total / 411 KB peak); wrote `PROF-01-hotpath-targets.md` skeleton with Environment + Measured-Cells table + provisional #1 target + M-dominated scaling note.

## Task Commits
1. **Tasks 1–3 (tracer)** — committed together with `--no-verify` (long-cargo hook; gates run out-of-band: baseline suite green, bench compiles clean)

## Files Created/Modified
- `fdars-core/benches/probe_fpca_variants.rs` — throwaway fsvd probe (removed in Plan 02)
- `fdars-core/tests/alloc_audit_new_subsystems.rs` — throwaway dhat probe (removed in Plan 02)
- `fdars-core/Cargo.toml` — temp `[[bench]] probe_fpca_variants` (removed in Plan 02)
- `.planning/phases/46-whole-crate-profiling-measurement/PROF-01-hotpath-targets.md` — inventory skeleton

## Decisions Made
- Ran all 4 cells (small cell was <1 ms, so n1000_m50 was tractable) — gives an immediate N-vs-M scaling read even in the tracer.

## Deviations from Plan
None - plan executed exactly as written. Committed with `--no-verify` per the project's long-cargo-hook memory pointer (dev-harness + doc only; baseline suite verified green out-of-band).

## Issues Encountered
None.

## Next Phase Readiness
- Pipeline proven; Plan 02 copies `probe_fpca_variants.rs` to the other 8 subsystems, reuses the Environment block, and removes all throwaway probes at the end.

---
*Phase: 46-whole-crate-profiling-measurement*
*Completed: 2026-08-30*
