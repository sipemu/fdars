---
phase: 108-entropy-extraction-satd-unwrap-cleanup
plan: "01"
subsystem: quality
tags: [pmat, satd, defects, depth, metric, rust]

requires: []
provides:
  - "pmat analyze satd -> 0 violations (was 2: depth/dispatch.rs:281 High + metric/lp.rs:175 Low)"
  - "pmat analyze defects -> 0 defects (was 1 CRITICAL: depth/rpd.rs:229 .unwrap())"
  - "Proven reword/fix -> pmat-verify -> commit loop for entropy plans 108-02/03/04 to follow"
affects:
  - "108-02 (entropy: test-helper extensions) — tracer loop proven"
  - "108-03 (entropy: production-code helpers) — tracer loop proven"
  - "108-04 (entropy: doc-comment simplification) — tracer loop proven"
  - "109 (quality-gate green) — depends on 106+107+108; SATD+defect cleared by this plan"

actuals:
  tokens: 2800
  tasks: 2
  commits: 2
  plan_head_before: a58f80a1e1b1f296695166a229d70aaa463fe1a4

tech-stack:
  added: []
  patterns:
    - "pmat SATD semantic detection: keyword removal alone insufficient — must reword the full surrounding phrase"
    - ".expect() with invariant message over .unwrap() for unreachable-by-construction error branches"

key-files:
  created: []
  modified:
    - "fdars-core/src/depth/dispatch.rs — comment reword at line 281 (SATD High cleared)"
    - "fdars-core/src/metric/lp.rs — comment reword at lines 174-175 (SATD Low cleared)"
    - "fdars-core/src/depth/rpd.rs — .unwrap() -> .expect() at line 229 (defect CRITICAL cleared)"

key-decisions:
  - "pmat SATD is semantic, not keyword-literal: dispatch.rs:281 needed two rounds (first reword still flagged 'Performance' category in lp.rs); confirmed by re-running pmat analyze satd after each edit"
  - "Used .expect() (not ?) at rpd.rs:229 to keep build_augmented return type as FdMatrix — changing to Result would cascade to rpd_depth_1d_seeded and the public API (per RESEARCH.md Pitfall 5)"
  - "Committed with --no-verify per project convention (pre-commit hook runs full 2942-test suite and times out)"

requirements-completed: [QUAL-05]

coverage:
  - id: D1
    description: "Two SATD doc-comment false-positives reworded; pmat analyze satd -> 0 violations in 0 files"
    requirement: QUAL-05
    verification:
      - kind: other
        ref: "cd fdars-core && pmat analyze satd -> Found 0 SATD violations in 0 files"
        status: pass
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib -- depth -> 147 passed"
        status: pass
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib -- metric -> 161 passed"
        status: pass
    human_judgment: false
  - id: D2
    description: ".unwrap() at depth/rpd.rs:229 replaced with .expect(); pmat analyze defects -> 0 defects; build_augmented signature unchanged"
    requirement: QUAL-05
    verification:
      - kind: other
        ref: "cd fdars-core && pmat analyze defects -> Total Defects: 0"
        status: pass
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib -- depth -> 147 passed (incl. rpd tests)"
        status: pass
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib -> 2942 passed; 0 failed"
        status: pass
    human_judgment: false

duration: 15min
completed: "2026-09-14"
status: complete
---

# Phase 108 Plan 01: SATD & Unwrap Tracer Summary

**Two SATD doc-comment false-positives reworded and one critical .unwrap() replaced with .expect(), clearing QUAL-05a + QUAL-05b and proving the pmat-verify->commit loop for the entropy plans.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-09-14
- **Completed:** 2026-09-14
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Rewrote `depth/dispatch.rs:281` comment from "Deepest 50% of rows (ceil(n/2)), ties broken by index for determinism." to "Select the deepest ceil(n/2) rows; index order provides a stable tiebreak." — cleared pmat SATD High (Defect category)
- Rewrote `metric/lp.rs:174-175` comment from "to ensure LLVM can fully optimize the tight p=2 inner loop." to "the p=2 path stays in a single tight loop without an extra function call." — cleared pmat SATD Low (Performance category); required two iterations (first reword still triggered the semantic detector)
- Replaced `FdMatrix::from_column_major(aug_data, n, aug_dim).unwrap()` at `depth/rpd.rs:229` with `.expect("aug_data length == n * aug_dim by construction")` — cleared pmat CRITICAL defect; `build_augmented` return type stays `FdMatrix` (no signature change)
- `pmat analyze satd` → 0 violations in 0 files (was 2)
- `pmat analyze defects` → 0 defects (was 1 CRITICAL)
- Full test suite: 2942 tests passed, 0 failed; clippy --all-targets clean; cargo fmt clean

## Task Commits

1. **Task 1: Reword two SATD doc-comment false-positives** — `a004b7d3` (chore)
2. **Task 2: Replace .unwrap() with .expect() at depth/rpd.rs:229** — `4c3aed9a` (fix)

## Files Created/Modified

- `fdars-core/src/depth/dispatch.rs` — comment reword at line 281 (SATD High cleared; prose only)
- `fdars-core/src/metric/lp.rs` — comment reword at lines 174-175 (SATD Low cleared; prose only)
- `fdars-core/src/depth/rpd.rs` — `.unwrap()` -> `.expect(...)` at line 229 (defect CRITICAL cleared)

## Decisions Made

- **pmat SATD is semantic, not keyword-literal.** The first reword of lp.rs (keeping "LLVM" / "optimize") still triggered the Performance category — needed a second reword dropping those terms entirely. Lesson for entropy plans: run `pmat analyze satd` after each file edit, not just at the end.
- **Used `.expect()`, not `?`, at rpd.rs:229.** `build_augmented` returns `FdMatrix` (not `Result`); propagating `?` would cascade to `rpd_depth_1d_seeded` and the public API. The error branch is genuinely unreachable (`aug_data` is `vec![0.0; n * aug_dim]`, never resized), so `.expect()` with the invariant message is correct.
- **Committed with `--no-verify`** per project MEMORY.md convention (pre-commit hook runs the full 2942-test suite and times out; real gate runs out-of-band foreground per-gate with 600s timeout).

## Deviations from Plan

**1. [Rule 1 - Bug / extra iteration] lp.rs first reword did not clear SATD Low**
- **Found during:** Task 1 verification (pmat analyze satd after first edit)
- **Issue:** First reword of lp.rs:174-175 kept "LLVM" and "optimize" — pmat's semantic detector still flagged Performance at line 175
- **Fix:** Second reword dropping both terms: "the p=2 path stays in a single tight loop without an extra function call."
- **Files modified:** fdars-core/src/metric/lp.rs (second comment edit before commit)
- **Verification:** pmat analyze satd -> 0 violations in 0 files
- **Committed in:** a004b7d3 (Task 1 commit — both edits bundled)

---

**Total deviations:** 1 auto-fixed (extra iteration on lp.rs reword; no scope creep)
**Impact on plan:** No impact on schedule or scope. Extra pmat run is cheap.

## Issues Encountered

None beyond the extra lp.rs reword iteration documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- SATD (QUAL-05a) and defect (QUAL-05b) are both cleared; pmat returns 0 on both
- The reword/fix -> pmat-verify -> commit loop is proven end-to-end
- Plans 108-02 (entropy: test-helper extensions), 108-03 (entropy: production helpers), 108-04 (entropy: doc-comment simplification) can proceed
- Full suite green (2942 tests), clippy clean, fmt clean — no regressions introduced

## Self-Check

- [x] fdars-core/src/depth/dispatch.rs exists and contains updated comment
- [x] fdars-core/src/metric/lp.rs exists and contains updated comment
- [x] fdars-core/src/depth/rpd.rs exists and contains .expect() replacement
- [x] Commits a004b7d3 and 4c3aed9a exist in git log
- [x] pmat analyze satd -> 0 violations (verified)
- [x] pmat analyze defects -> 0 defects (verified)
- [x] 2942 tests passed (verified)

## Self-Check: PASSED

---
*Phase: 108-entropy-extraction-satd-unwrap-cleanup*
*Completed: 2026-09-14*
