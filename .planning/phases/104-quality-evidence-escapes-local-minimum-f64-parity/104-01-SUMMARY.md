---
phase: 104-quality-evidence-escapes-local-minimum-f64-parity
plan: 01
subsystem: testing
tags: [soft-dtw, barycenter, multi-restart, local-minimum, evidence, rust, fdars-core]

# Dependency graph
requires:
  - phase: 103-sdtw-03-barycenter-config-entry-point
    provides: soft_dtw_barycenter_with_config, SoftDtwBarycenterConfig, legacy 4-arg soft_dtw_barycenter delegating to multirestart
  - phase: 102-sdtw-02-multi-restart-barycenter
    provides: soft_dtw_barycenter_multirestart, perturbed_init, per_col_std
provides:
  - "Committed deterministic test proving multi-restart escapes a genuine local minimum (367% cost improvement on bimodal Gaussian-pulse MM-stall dataset)"
  - "Committed deterministic test proving legacy 4-arg soft_dtw_barycenter returns finite, length-m, per-column-convex-hull-bounded barycenter"
  - "SDTW-04 quality-evidence gate cleared"
affects: [phase-105-v045-release, ROADMAP-TO-1.0-SDTW-O1-checkoff]

# Actuals (#2632) — pairs with the plan's `estimate` to calibrate future estimates.
actuals:
  tokens: 2774        # chars/4 over the realized diff (11098 bytes / 4)
  tasks: 3            # tasks completed (T1 tracer + T2 auto + T3 guard — no code in T3)
  commits: 1          # MEASURED: git rev-list --count 6f257854..HEAD
  plan_head_before: 6f257854aac3cc96ae44dc44cfe7c41e1fbb9928

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SDTW-04 evidence form: deterministic inline #[cfg(test)] test modules, fixed dataset, fixed seed, absolute-improvement margin assertion (robust to negative soft-DTW costs)"
    - "Bimodal Gaussian-pulse MM-stall dataset: two clusters at t=0.12 and t=0.88, σ=0.04, γ=0.01 — reliably traps single-init L-BFGS while multi-restart escapes"
    - "Margin assertion uses improvement = single_init_total - multi_restart_total >= 1e-3 * |single_init_total| — sign-correct for negative soft-DTW values"

key-files:
  created: []
  modified:
    - fdars-core/src/metric/soft_dtw.rs

key-decisions:
  - "Assertion uses absolute-improvement form (improvement >= required_margin) rather than multiplicative form (multi < single * (1-eps)) — the multiplicative form is vacuously true for negative costs, leading to a false green on the first dataset attempt."
  - "Struct literal syntax for SoftDtwBarycenterConfig inside crate tests (not default()+field-assign) — avoids clippy::field_reassign_with_default; only external callers need the idiom."
  - "Dataset tuning: Gaussian pulses at 0.12/0.88 with γ=0.01 and n_restarts=16 produce a 367% improvement gap vs the 0.1% required — far from knife-edge, robust to cross-platform rounding."
  - "Legacy_bounded_convex_hull uses sinusoidal data (not the escape dataset) for independence — tests a distinct representative data class."

patterns-established:
  - "MM-stall evidence test pattern: bimodal Gaussian-pulse dataset + absolute-improvement assertion + eprintln!-verified gap before finalizing"
  - "Negative-cost awareness: always verify assertion formula is sign-correct before accepting a passing test"

requirements-completed: [SDTW-04]

coverage:
  - id: D1
    description: "escape_from_local_minimum test: multi-restart total cost beats single-init total cost by >0.1% on a bimodal Gaussian-pulse MM-stall dataset (single=+0.46, multi=-3.21, 367% improvement)"
    requirement: SDTW-04
    verification:
      - kind: unit
        ref: "fdars-core/src/metric/soft_dtw.rs#barycenter_escape_evidence_tests::escape_from_local_minimum"
        status: pass
      - kind: unit
        ref: "RAYON_NUM_THREADS=1 fdars-core/src/metric/soft_dtw.rs#barycenter_escape_evidence_tests::escape_from_local_minimum"
        status: pass
    human_judgment: false
  - id: D2
    description: "legacy_bounded_convex_hull test: legacy 4-arg soft_dtw_barycenter returns finite, length-m barycenter with all coordinates within per-column [min-eps, max+eps] on 6-row sinusoidal dataset"
    requirement: SDTW-04
    verification:
      - kind: unit
        ref: "fdars-core/src/metric/soft_dtw.rs#barycenter_escape_evidence_tests::legacy_bounded_convex_hull"
        status: pass
      - kind: unit
        ref: "RAYON_NUM_THREADS=1 fdars-core/src/metric/soft_dtw.rs#barycenter_escape_evidence_tests::legacy_bounded_convex_hull"
        status: pass
    human_judgment: false
  - id: D3
    description: "All 36 soft_dtw tests pass including existing test_soft_dtw_barycenter_vs_tslearn regression guard, under default threading and RAYON_NUM_THREADS=1; clippy --all-targets --features linalg,parallel -D warnings is clean"
    requirement: SDTW-04
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel soft_dtw → 36 passed"
        status: pass
      - kind: unit
        ref: "RAYON_NUM_THREADS=1 cargo test -p fdars-core --features linalg,parallel soft_dtw → 36 passed"
        status: pass
      - kind: other
        ref: "cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings → Finished (0 errors)"
        status: pass
    human_judgment: false

# Metrics
duration: 25min
completed: 2026-09-13
status: complete
---

# Phase 104 Plan 01: Quality Evidence — Escapes-Local-Minimum & f64 Parity Summary

**Two deterministic SDTW-04 evidence tests committed: multi-restart escapes a bimodal Gaussian-pulse local minimum by 367% improvement; legacy path returns finite, convex-hull-bounded barycenter on sinusoidal data**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-09-13T00:00:00Z
- **Completed:** 2026-09-13
- **Tasks:** 3 (T1 tracer + T2 auto + T3 guard verification)
- **Files modified:** 1 (`fdars-core/src/metric/soft_dtw.rs`)

## Accomplishments
- `escape_from_local_minimum`: bimodal Gaussian-pulse dataset (cluster A at t=0.12, cluster B at t=0.88, σ=0.04, γ=0.01) traps single-init L-BFGS at total cost +0.460 while multi-restart (n_restarts=16, seed=7) escapes to −3.214 — a 367% cost improvement, far exceeding the required 0.1% margin
- `legacy_bounded_convex_hull`: 6-row offset-sinusoid dataset, 30 evaluation points; asserts finite, length-30, per-column convex-hull-bounded barycenter from the legacy 4-arg path
- Both tests deterministic, pass under default parallelism and RAYON_NUM_THREADS=1; clippy --all-targets clean; all 36 soft_dtw tests green (including existing `test_soft_dtw_barycenter_vs_tslearn` regression guard)

## Task Commits

1. **Tasks 1+2: escape_from_local_minimum + legacy_bounded_convex_hull** - `f03ee900` (test)
2. **Task 3: Guard — verified all gates pass; no code change needed** — no additional commit

**Plan metadata commit:** (skipped — docs disabled or .planning gitignored per SDK response)

## Files Created/Modified
- `fdars-core/src/metric/soft_dtw.rs` — added `barycenter_escape_evidence_tests` module with two `#[cfg(test)]` tests (228 lines inserted); no production symbol changes

## Decisions Made
- **Assertion formula fix (critical):** First draft used multiplicative form `multi < single * (1 - 1e-3)`. This is vacuously true when `single` is negative (a more-negative multi satisfies it even when both costs are equal). Replaced with absolute-improvement form: `improvement = single - multi >= 1e-3 * |single|`.
- **Dataset iteration:** Step-function dataset (transition indices 6 vs 14) showed 0% gap — both paths converged identically. Switched to bimodal Gaussian-pulse dataset with widely separated centers (0.12 vs 0.88) and γ=0.01, which creates a genuinely flat region at the mean that traps single-init.
- **Struct literal in tests:** Inside-crate tests use `SoftDtwBarycenterConfig { .. }` struct literal (clippy-clean) instead of `default()+field-assign` (which clippy::field_reassign_with_default flags). External callers still use the idiom per SEAL-02.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Incorrect assertion formula for negative soft-DTW costs**
- **Found during:** Task 1 (escape_from_local_minimum initial implementation)
- **Issue:** First assertion `multi_restart_total < single_init_total * (1.0 - 1e-3)` is vacuously true when costs are negative — more-negative multi satisfies it even when identical to single. The test passed with 0% gap, giving a false green for the first step-function dataset.
- **Fix:** Replaced with `improvement = single_init_total - multi_restart_total >= 1e-3 * single_init_total.abs().max(1.0)`. This is sign-correct regardless of cost sign.
- **Files modified:** fdars-core/src/metric/soft_dtw.rs
- **Verification:** Test fails correctly when costs are identical (0% gap), passes correctly when multi beats single by 367%.
- **Committed in:** f03ee900

**2. [Rule 1 - Bug] Step-function dataset did not exhibit genuine MM-stall**
- **Found during:** Task 1 (first dataset attempt, step transitions at indices 6 vs 14)
- **Issue:** Both single-init and multi-restart converged to identical barycenters (0% gap). The step-function dataset with balanced clusters does not create a local minimum that traps single-init — L-BFGS from the mean finds the global optimum.
- **Fix:** Replaced with bimodal Gaussian-pulse dataset (peak centers at t=0.12 and t=0.88, σ=0.04, γ=0.01, n_restarts=16, seed=7), which creates a genuine flat region at the pointwise mean that traps single-init while multi-restart escapes to one cluster's deep basin.
- **Files modified:** fdars-core/src/metric/soft_dtw.rs
- **Verification:** Observed cost values via eprintln! — single_init=+0.460696, multi_restart=−3.213728, improvement=367.4%. eprintln! removed before commit.
- **Committed in:** f03ee900

---

**Total deviations:** 2 auto-fixed (2 Rule 1 bugs — incorrect assertion formula, non-stalling dataset)
**Impact on plan:** Both fixes essential for the evidence to be genuine. The plan explicitly required tuning the dataset until single-init demonstrably stalls; the deviation rule applies because a false-green test would invalidate the SDTW-04 gate.

## Issues Encountered
- Negative soft-DTW costs (due to log-sum-exp softmin construction) make percentage-of-reference assertions require care — the sign-correct form must use absolute value of the reference.
- Step-function datasets with balanced bimodal clusters are NOT reliable MM-stall exhibitors; Gaussian-pulse datasets with widely separated centers and small γ are.

## Known Stubs
None — all assertions are verified against real computed values.

## Threat Flags
None — test-only phase, no new network endpoints, auth paths, file access patterns, or schema changes.

## Next Phase Readiness
- SDTW-04 cleared; Phase 105 (release prep: version bump 0.44.0→0.45.0, CHANGELOG, ROADMAP-TO-1.0.md SDTW-O1 checkoff, full gate battery, cargo package) is unblocked.
- The 28-example count is unchanged (no new runnable example added, per decision D-Evidence-form).

## Self-Check: PASSED
- `fdars-core/src/metric/soft_dtw.rs` modified and committed at f03ee900 — FOUND
- Commit f03ee900 exists — FOUND
- Both new tests in `barycenter_escape_evidence_tests` module — FOUND
- No production symbol changes — VERIFIED (only `#[cfg(test)]` block modified)
- No Cargo.toml change — VERIFIED
- 36 soft_dtw tests pass, 0 failed — VERIFIED
- clippy --all-targets clean — VERIFIED

---
*Phase: 104-quality-evidence-escapes-local-minimum-f64-parity*
*Completed: 2026-09-13*
