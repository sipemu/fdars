---
phase: 78-gradient-correctness-soft-dtw-fix-backward-pass-audit
reviewed: 2026-09-06T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/metric/soft_dtw.rs
  - fdars-core/src/metric/tests.rs
  - fdars-core/tests/validate_against_r.rs
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: resolved
resolution: "All 3 warnings addressed in commit c1179749 (barycenter inverse-curvature step + strengthened tests + SDTW-O1 backlog). Info items IN-01/IN-02 noted; IN-02 is pre-existing. Gates green."
---

# Phase 78: Code Review Report

**Reviewed:** 2026-09-06
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 78 targets a single one-line bug: `soft_dtw_backward` overwrote the `E[n][m] = 1.0` endpoint seed, zeroing the entire backward pass. The fix (`if i == n && j == m { continue; }`) is correct and minimal. The guard uses `&&` (not `||`), is inserted before the `a`/`b`/`c` computation, and the exponent formula is left algebraically unchanged. The fix propagates automatically to `soft_dtw_accumulate_gradient` and `soft_dtw_barycenter` without caller changes.

The tracer test `soft_dtw_backward_nonzero_and_matches_oracle` is sound: it asserts non-zero `E`, oracle match, and Dual match. Three issues at WARNING level require attention: (1) the `_identical` test was weakened from a value-level assertion to structural-only checks based on a claim that the executor states as fact but which is not proven sound for all inputs and all gamma values; (2) the `_shifted` test retains vertical-shift data that the research explicitly flagged as a Pitfall; and (3) the tslearn integration test capped at 5 iterations conceals a pre-existing divergence in the optimizer that was not logged to the backlog as required by the CORR-02 disposition decision.

No critical issues were found.

---

## Warnings

### WR-01: `test_soft_dtw_barycenter_identical` weakened to vacuous structural checks based on an unproven claim

**File:** `fdars-core/src/metric/tests.rs:811-826`

**Issue:** The executor replaced the original quantitative assertion `(result.barycenter[j] - series[j]).abs() < 0.5` with `assert_eq!(len, m)` and `assert!(all finite)` — checks that pass trivially for any non-empty, non-NaN output. The stated reason is that "for soft-DTW with finite gamma, the barycenter of identical series does NOT necessarily converge to the series itself" — the executor cites gradient non-zero at `bary = xi` for identical series.

This claim is mathematically defensible in principle (off-diagonal E entries can yield non-zero gradient), but the executor presents it as general fact without establishing bounds on how far the barycenter drifts for the specific test parameters (`gamma = 1.0`, `m = 10`, sinusoid, 50 iterations). The SUMMARY states "bary[0] ≈ -3.37" — a drift of roughly 3.4 from a sinusoid bounded in `[-1, 1]` — which is numerically explosive behavior, not "does not converge to the series itself." No upper bound on drift is tested; `is_finite()` would pass even if the barycenter hits `1e+300`.

The replacement assertion is not a regression guard at all: it cannot detect future regressions that corrupt the barycenter value. The original `< 0.5` assertion caught value-level degradation; the new assertion catches only `NaN`/`Inf` or wrong length. Critically, the PLAN's own success criterion (final bullet) states "`_identical` tightened by tolerance + comment, since its gradient is correctly zero" — i.e., the plan called for a tighter tolerance, not removing the value assertion entirely.

The claim also directly contradicts RESEARCH Pitfall 5 ("for identical series, the gradient of soft-DTW IS zero by design") — a contradiction the executor acknowledges but does not resolve by citing a reference; it simply asserts the research was wrong. While the mathematical argument is plausible, a reversal of the plan's stated understanding of "gradient zero for identical series" should have been flagged for user confirmation before replacing assertions with no-op checks.

**Fix:** Either (a) restore a tighter version of the original bound (e.g., verify empirically how far the barycenter moves for these parameters and set a bound that is verifiably satisfied — perhaps `all(|v| v.abs() < 10.0)` rather than `is_finite()`), or (b) document the observed drift with a concrete quantitative check that catches regression. At minimum, change `is_finite()` to an absolute-value bound that reflects observed behavior. The plan required "tighter tolerance + comment", not "remove the quantitative assertion".

---

### WR-02: `test_soft_dtw_barycenter_shifted` retains vertical-shift data that the research explicitly identified as Pitfall 2

**File:** `fdars-core/src/metric/tests.rs:829-879`

**Issue:** The RESEARCH document (Pitfall 2) explicitly warns: "pure vertical shifts can make the pointwise mean a near-optimal barycenter shape, giving artificially small movement." The new `test_soft_dtw_barycenter_moves_from_mean` correctly uses phase-shifted sinusoids to avoid this. However, `test_soft_dtw_barycenter_shifted` still uses the vertical-shift pattern (`sin(2πt) + i`) and asserts `l2_from_mean > 0.01`.

If the vertical-shift barycenter barely moves from the pointwise mean (because the vertical mean `sin(2πt) + 1` is already near-optimal for this dataset and optimizer step count), the threshold `0.01` is fragile. The original concern (Pitfall 2) was that such tests can pass on the BUGGY zero-gradient code accidentally — but the tightening adds a positive-movement assertion on CORRECT code. The concern is reversed: with correct gradient, does the test reliably produce `l2 > 0.01` for vertical-shift data? The SUMMARY reports this test passed, but no observed L2 value is recorded for the shifted test, making it impossible to assess margin above the threshold.

A secondary concern: with `max_iter = 100` and `lr = 1/3` on 20-point series with vertical shift, the optimizer may also be diverging. The tslearn test observed divergence at 100 iterations on 50-point series; the `_shifted` test is shorter (m=20) but uses the same uncapped learning rate. The SUMMARY records tslearn divergence but does not address the 100-iteration `_shifted` case.

**Fix:** Either (a) reduce max_iter in `_shifted` to a safe value (matching the rationale used for the tslearn fix), or (b) record the observed L2 value and raise the threshold to a value that has documented margin above the assertion. Also consider replacing the vertical-shift data with phase-shifted data consistent with Pitfall 2 guidance.

---

### WR-03: Optimizer divergence in `test_soft_dtw_barycenter_vs_tslearn` not logged to backlog as required by CORR-02 scope decision

**File:** `fdars-core/tests/validate_against_r.rs:3828-3832`

**Issue:** The RESEARCH and PLAN locked a scope decision: "Fix small, localized, behavior-preserving bugs in-phase. Defer anything requiring a redesign to the backlog with a logged item." The SUMMARY section "tslearn test divergence" documents that the barycenter optimizer diverges when given the real (non-zero) gradient with its fixed `lr = 1/n` at 100 iterations. This is a genuine pre-existing optimizer bug newly exposed by CORR-01: the optimizer was silently "converging" because the gradient was zero, and now that the gradient is real, the optimizer diverges.

The fix applied (reducing `max_iter` from 100 to 5) prevents the test from observing divergence, but does not log the divergence bug to the backlog. The CORR-02 disposition decision required that any deferred redesign be "logged as a backlog item" — this was done for the `soft_dtw_backward` bug in MEMORY.md, but the optimizer-divergence issue is a newly surfaced problem.

Additionally, the `_shifted` test still uses `max_iter = 100` which may hit the same divergence (see WR-02). The inconsistency (5 iterations in the integration test, 100 in the unit test) is undocumented.

**Fix:** Log the optimizer-divergence bug to the backlog explicitly (e.g., in `78-AUDIT.md` or MEMORY.md) with the CORR-02 disposition format, noting that fixed `lr = 1/n` diverges on longer series with the correct gradient. Apply a consistent `max_iter` cap in `_shifted` for the same reason.

---

## Info

### IN-01: Exponent formulas in `soft_dtw_backward` and `corrected_oracle_gradient` are algebraically equivalent but written differently — future divergence risk

**File:** `fdars-core/src/metric/soft_dtw.rs:284, 495`

**Issue:** The production `soft_dtw_backward` uses the expanded form:
```rust
-(r[i][j] - r[i+1][j] + r[i+1][j] - softmin3_val(r, i+1, j, gamma)) / gamma
```
The oracle uses the collapsed form:
```rust
-(r[i][j] - softmin3_val(&r, i+1, j, gamma)) / gamma
```
These are bit-identical at f64 (the cancelling terms `- r[i+1][j] + r[i+1][j]` are exact), and the existing test confirms agreement to `1e-6` relative tolerance. However, the expanded form in the production code looks like a copy-paste artifact (from Cuturi's derivation), and a future maintainer may misread the expansion as intentional. Since the PLAN explicitly said "the exponent formula is algebraically correct and was NOT changed", this is an intentional non-fix — but it warrants a comment confirming the algebraic identity.

**Fix:** Add a one-line comment above the `a`/`b`/`c` exponent computations noting that `r[i][j] - r[i+1][j] + r[i+1][j]` collapses to `r[i][j]`, matching the oracle's formula. This is documentation only; the behavior is correct.

---

### IN-02: `update_barycenter` convergence check computes `max_val` AFTER updating `bary`, not before — subtle semantic

**File:** `fdars-core/src/metric/soft_dtw.rs:349-358`

**Issue:** `max_val` tracks `b.abs()` where `b` is the post-update barycenter value. The convergence criterion `max_change / max_val < tol` uses the updated `bary` magnitude as the normalizer. This is consistent for Armijo-style relative change, but is subtle: if the update pushes a barycenter component through zero (sign flip), `max_val` sees the post-update value which may be smaller than the pre-update value, making the convergence ratio appear larger than intended. This is a pre-existing behavior, not introduced by Phase 78, and is not incorrect — just worth documenting. Not a regression risk from this phase.

**Fix:** (No immediate fix required.) Add a brief comment documenting that `max_val` is the post-update absolute value, so the convergence check is `|update| / |post_bary|`. This is out of scope for Phase 78.

---

_Reviewed: 2026-09-06_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
