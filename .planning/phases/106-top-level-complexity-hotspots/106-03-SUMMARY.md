---
phase: 106-top-level-complexity-hotspots
plan: "03"
subsystem: boosting_regression
tags: [pmat, complexity-refactor, boost-fofr, bayesian-fosr, gibbs-sampler]

requires:
  - "106-01 (proven extract->test->pmat->commit loop)"

provides:
  - "boost_fofr decomposed into 7 private helpers; all below cog 15 / cyc 9"
  - "bayesian_fosr decomposed into 6 private helpers; all below cog 16 / cyc 7"
  - "Both target functions clear pmat error thresholds (cog <=25, cyc <=30)"
  - "Gibbs RNG draw order preserved — bayesian_fosr_is_deterministic_under_seed passes"

affects: [106-04]

actuals:
  tokens: 42000
  tasks: 2
  commits: 2
  plan_head_before: dcfdcd234efe844f8a409ec695c488ff5cc53d78

tech-stack:
  added: []
  patterns:
    - "Split-validate-predictors: extra level of split for the per-predictor loop in boost_fofr_validate_dims — predictor dim checks extracted to boost_fofr_validate_predictors"
    - "Residual-helper: common Y-Fhat residual matrix computation extracted to boost_fofr_residuals; reused for both within-iteration and final residuals"
    - "Precompute-return-xtx: bayesian_fosr_precompute returns xtx (not xbar) so the Gibbs step receives X'X without a redundant compute_xtx call in the parent"
    - "allow(clippy::too_many_arguments) on bayesian_fosr_gibbs_step: 13-param Gibbs step is inherently argument-rich; all args necessary for the step's computation"

key-files:
  created: []
  modified:
    - fdars-core/src/boosting_regression/boost_fofr.rs
    - fdars-core/src/boosting_regression/bayesian.rs

key-decisions:
  - "boost_fofr_validate needed 3 helpers (predictors + dims + config), not 2: the dims validator alone scored cog 26 after extracting the per-predictor loop; added boost_fofr_validate_predictors to bring dims to cog 15"
  - "boost_fofr_residuals extracted as a shared helper: called both in the boosting iteration loop and for the final residual computation — avoids code duplication"
  - "bayesian_fosr_precompute returns (fpca, xc, xtx, xt_xi): includes xtx in the return tuple rather than forcing the parent to call compute_xtx separately; xbar is used internally within precompute only"
  - "bayesian_fosr_gibbs_step receives all required state by &mut reference: b_state, sigma2, rng — single-owner sequential access, no aliasing; inner kk loop preserved in parent calling the step"
  - "RNG draw order: outer iter loop remains in bayesian_fosr parent; inner kk loop calls bayesian_fosr_gibbs_step for each kk — exact same draw order as original (z draw then gamma sample per kk per iter)"

requirements-completed: [QUAL-01]

coverage:
  - id: D1
    description: "boost_fofr cognitive complexity reduced from 98 to 15 (pmat 3.18.0); cyclomatic from 39 to 9; all 7 new helpers below cog 15 / cyc 9; 6 module tests confirm numeric-identical output"
    requirement: QUAL-01
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib -- boosting_regression::boost_fofr::tests (6 tests)"
        status: pass
      - kind: other
        ref: "pmat analyze complexity --path fdars-core/src/boosting_regression/boost_fofr.rs: boost_fofr cog=15 cyc=9; max helper cog=15 cyc=7"
        status: pass
    human_judgment: false
  - id: D2
    description: "bayesian_fosr cognitive complexity reduced from 84 to 7 (pmat 3.18.0); cyclomatic from 36 to 5; all 6 new helpers below cog 16 / cyc 7; 6 module tests confirm numeric-identical output including RNG-order-sensitive bayesian_fosr_is_deterministic_under_seed"
    requirement: QUAL-01
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib -- boosting_regression::bayesian::tests (6 tests)"
        status: pass
      - kind: other
        ref: "pmat analyze complexity --path fdars-core/src/boosting_regression/bayesian.rs: bayesian_fosr cog=7 cyc=5; max helper cog=16 cyc=7"
        status: pass
    human_judgment: false
  - id: D3
    description: "Full test suite (2942 lib tests) green after both refactors; clippy --all-targets clean; cargo fmt --check clean"
    requirement: QUAL-01
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib: 2942 passed, 0 failed"
        status: pass
      - kind: other
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings: Finished (no errors); cargo fmt --check: clean"
        status: pass
    human_judgment: false

duration: 21min
completed: "2026-09-14"
status: complete
---

# Phase 106 Plan 03: Top-Level Complexity Hotspots Summary

**Private-helper extraction of boost_fofr (cog 98→15, cyc 39→9) and bayesian_fosr (cog 84→7, cyc 36→5) clears the two boosting-regression QUAL-01 hotspots; Gibbs RNG order preserved; 2942 tests green**

## Performance

- **Duration:** 21 min
- **Started:** 2026-09-14T05:49:33Z
- **Completed:** 2026-09-14T06:10:33Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `boost_fofr` (cog 98 → 15, cyc 39 → 9): extracted `boost_fofr_validate_predictors`, `boost_fofr_validate_dims`, `boost_fofr_validate_config`, `boost_fofr_build_learners`, `boost_fofr_best_learner`, `boost_fofr_residuals`, `boost_fofr_build_beta_surfaces`; all helpers at or below cog 15 / cyc 9
- `bayesian_fosr` (cog 84 → 7, cyc 36 → 5): extracted `bayesian_fosr_validate_dims`, `bayesian_fosr_validate_config`, `bayesian_fosr_precompute`, `bayesian_fosr_gibbs_step`, `bayesian_fosr_record_draw`, `bayesian_fosr_summaries`; all helpers at or below cog 16 / cyc 7
- Gibbs RNG draw order preserved: outer `iter` loop and inner `kk` loop remain in the parent; `bayesian_fosr_gibbs_step` called per-kk inside the unchanged outer loop; `bayesian_fosr_is_deterministic_under_seed` passes with exact bit-identical posterior draws
- Full suite 2942 lib tests green; clippy --all-targets clean; cargo fmt --check clean
- No public API / signature / visibility change — `pub fn boost_fofr`, `pub fn bayesian_fosr` signatures and visibility untouched

## Task Commits

1. **Task 1: decompose boost_fofr** — `323af4b3` (refactor)
2. **Task 2: decompose bayesian_fosr** — `b2451db8` (refactor)

## Files Created/Modified

- `fdars-core/src/boosting_regression/boost_fofr.rs` — 7 private helpers added; parent simplified; net +275 lines (helpers) / -223 lines (inline bodies removed); no public API change
- `fdars-core/src/boosting_regression/bayesian.rs` — 6 private helpers added; parent simplified; net +238 lines (helpers) / -118 lines (inline bodies removed); no public API change

## Decisions Made

- **boost_fofr_validate split 3 ways**: The RESEARCH.md proposed `boost_fofr_validate` (1 helper), but after first extraction `boost_fofr_validate_dims` scored cog 26 (above threshold). Added `boost_fofr_validate_predictors` to extract the per-predictor loop, bringing `_dims` to cog 15.
- **boost_fofr_residuals as shared helper**: The residual computation (Y − F̂) appears both in the boosting loop and for the final residual output. Extracted once, called twice.
- **bayesian_fosr_precompute returns xtx**: The RESEARCH.md signature included xbar in the return, but the parent only needs xtx (xbar is internal to centering). Changed return to `(fpca, xc, xtx, xt_xi)` to avoid a redundant `compute_xtx` call in the parent.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] boost_fofr_validate needed 3 helpers, not 2**
- **Found during:** Task 1 — pmat check after first extraction
- **Issue:** After extracting `boost_fofr_validate_dims` (all dims/argvals/per-predictor guards), pmat reported cognitive 26 — still above the 25 threshold. The RESEARCH.md seam proposed a single `boost_fofr_validate` helper, but the per-predictor loop with nested ifs contributed enough nesting to keep it above threshold.
- **Fix:** Extracted `boost_fofr_validate_predictors` (per-predictor row-count + argvals-length loop) from within `boost_fofr_validate_dims`. The dims validator then calls `boost_fofr_validate_predictors?` at the end, scoring cog 15.
- **Files modified:** fdars-core/src/boosting_regression/boost_fofr.rs
- **Verification:** pmat reports boost_fofr_validate_dims cog=15, boost_fofr_validate_predictors cog=11; both below threshold. 6 tests pass.
- **Committed in:** 323af4b3 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 — additional helper level needed to clear pmat threshold)
**Impact on plan:** Necessary to achieve the complexity-drop mandate. No scope creep. All helpers remain private, no API surface change.

## pmat Before → After

| Function | Baseline Cog | Final Cog | Baseline Cyc | Final Cyc |
|----------|-------------|-----------|-------------|-----------|
| `boost_fofr` | 98 | **15** | 39 | 9 |
| `boost_fofr_validate_predictors` (new) | — | 11 | — | 4 |
| `boost_fofr_validate_dims` (new) | — | 15 | — | 6 |
| `boost_fofr_validate_config` (new) | — | 10 | — | 5 |
| `boost_fofr_build_learners` (new) | — | 12 | — | 6 |
| `boost_fofr_best_learner` (new) | — | 15 | — | 7 |
| `boost_fofr_residuals` (new) | — | 3 | — | 3 |
| `boost_fofr_build_beta_surfaces` (new) | — | 6 | — | 4 |
| `bayesian_fosr` | 84 | **7** | 36 | 5 |
| `bayesian_fosr_validate_dims` (new) | — | 11 | — | 6 |
| `bayesian_fosr_validate_config` (new) | — | 16 | — | 7 |
| `bayesian_fosr_precompute` (new) | — | 10 | — | 7 |
| `bayesian_fosr_gibbs_step` (new) | — | 7 | — | 7 |
| `bayesian_fosr_record_draw` (new) | — | 6 | — | 4 |
| `bayesian_fosr_summaries` (new) | — | 9 | — | 6 |

All functions at or below thresholds (cog ≤ 25, cyc ≤ 30). No new violations introduced.

## Known Stubs

None. All helper extractions are complete, production-quality private functions wired directly into their parents. No placeholder behavior.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Pure internal refactor — no new security surface.

## Self-Check: PASSED

- File exists: `fdars-core/src/boosting_regression/boost_fofr.rs` — FOUND
- File exists: `fdars-core/src/boosting_regression/bayesian.rs` — FOUND
- Commit 323af4b3 exists — FOUND (Task 1: boost_fofr)
- Commit b2451db8 exists — FOUND (Task 2: bayesian_fosr)
- 2 commits measured from ledger (dcfdcd23..HEAD)

---
*Phase: 106-top-level-complexity-hotspots*
*Completed: 2026-09-14*
