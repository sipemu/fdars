---
phase: 46-whole-crate-profiling-measurement
plan: 04
subsystem: api
tags: [profiling, api-consistency, static-analysis, deprecation, config-default]
requires:
  - phase: 46-01
    provides: (none — independent static-analysis plan)
provides:
  - PROF-03 ranked API-inconsistency inventory with additive-safe canonical forms (drives Phase 50)
affects: [50-additive-api-surface-consolidation]
actuals:
  tokens: 6000
  tasks: 2
  commits: 1
tech-stack:
  added: []
  patterns: [additive-safe vs breaking classification, impact+breadth ranking]
key-files:
  created: [.planning/phases/46-whole-crate-profiling-measurement/PROF-03-api-inventory.md]
  modified: []
key-decisions:
  - "Exactly 4 Config structs miss Default (BoostingConfig/BayesianConfig/StabilityConfig/StlConfig) — confirmed 56 structs / 52 impls."
  - "fanova lacks a seed param (non-reproducible) — HIGH-impact additive fix via fanova_seeded."
  - "Result-field renames and bulk _1d/_2d deprecation are BREAKING → deferred to APIB-01; _nd algorithms must NOT be deprecated."
patterns-established:
  - "Every API item classified additive-safe vs breaking to protect R/WASM bindings + 28 examples."
requirements-completed: [PROF-03]
coverage:
  - id: D1
    description: "PROF-03 ranked API-inconsistency inventory with proposed canonical forms + additive-safe/breaking classification + API-01/02/03 tags"
    requirement: PROF-03
    verification:
      - kind: other
        ref: "grep -Eic 'src/…rs:N|BoostingConfig|StlConfig' => 11 (>=6); distinct API-0[123] tags => 3 (>=3); git status src/ => 0"
        status: pass
    human_judgment: true
    rationale: "Whether canonical forms are genuinely additive-safe and useful to Phase 50 is a human judgement; automation only confirms anchors + tags."
duration: 15min
completed: 2026-08-30
status: complete
---

# Phase 46 / Plan 04: PROF-03 API-Inconsistency Inventory Summary

**Ranked API-inconsistency inventory: 4 configs missing Default and a non-seedable fanova are the top additive-safe targets; field renames and bulk _1d/_2d unification classified breaking and deferred.**

## Performance
- **Duration:** ~15 min (ran in parallel with the 46-01 baseline suite)
- **Tasks:** 2
- **Files modified:** 1 (doc only)

## Accomplishments
- Confirmed exactly **4 Config structs lack `Default`** (56 `pub struct *Config` / 52 `impl Default`): `BoostingConfig` (mod.rs:44), `BayesianConfig` (mod.rs:76), `StabilityConfig` (mod.rs:103), `StlConfig` (stl.rs:49) — matches RESEARCH. → API-01, top target.
- Surfaced a reproducibility gap: `fanova` (`function_on_scalar.rs:791`) takes `(data, groups, n_perm)` with **no seed**, while every sibling permutation test (`t_perm_test`, `f_perm_test`, `frechet_anova`, `generic_permutation_importance`) is `(…, n_perm, seed)`. → additive `fanova_seeded`, API-01/API-02.
- Grouped 13 `_1d`/`_2d` paired families; flagged genuinely-different `_nd` algorithms (`pca_nd`, `karcher_mean_nd`, `karcher_covariance_nd`, `srsf_transform_nd`) as **do-not-deprecate**.
- Classified every item additive-safe vs breaking; result-field renames deferred to APIB-01 (breaking) — Phase 50 does documentation only there.

## Task Commits
1. **Task 1: grep-verify API-inconsistency candidates** — analysis (no file)
2. **Task 2: write PROF-03 inventory** — committed with plan docs

## Files Created/Modified
- `.planning/phases/46-whole-crate-profiling-measurement/PROF-03-api-inventory.md` — ranked inventory

## Decisions Made
- Ranked by user-facing impact + breadth: the 4 missing-Default configs (HIGH, small, self-contained) above the wide-but-lower-impact `_1d`/`_2d` family sweep.
- Correction to RESEARCH: none — the assumed 4-missing count is exactly right.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## Next Phase Readiness
- PROF-03 is ready for Phase 50 (API-01/02/03): each in-scope item is additive (add + `#[deprecated]`) so the 28 examples + R/WASM bindings keep compiling with deprecation warnings only.

---
*Phase: 46-whole-crate-profiling-measurement*
*Completed: 2026-08-30*
