---
phase: 11-performance-wins-parallel-cv-folds-faer-fpca-svd
reviewed: 2026-08-11T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/classification/cv.rs
  - fdars-core/src/regression.rs
  - fdars-core/src/spm/tests.rs
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: needs-attention
---

# Phase 11: Code Review Report

**Reviewed:** 2026-08-11
**Depth:** standard
**Files Reviewed:** 3
**Status:** needs-attention

## Summary

Phase 11 introduced two performance-oriented changes — parallelizing the `fclassif_cv` fold loop via `iter_maybe_parallel!` and replacing the nalgebra FPCA SVD backend with faer `Svd::new_thin` under the `linalg` feature — plus a compensating test guard in `spm/tests.rs`. The implementation is structurally sound: column-major `MatRef` dimensions are correct, `cfg`-gating is correct, `fix_svd_signs` edge-case behavior is safe, and the parallel fold loop preserves result ordering. No critical bugs or security issues were found.

Two warnings require attention before this code ships to users who rely on the MEWMA monitor's SPE behavior, or who run the test suite with altered test data that happens to trigger the degenerate regime differently.

---

## Warnings

### WR-01: SPM test guard makes the MEWMA IC SPE-alarm assertion permanently unexecutable on current test data

**File:** `fdars-core/src/spm/tests.rs:3291-3298`

**Issue:** The change replaces an unconditional `assert!(n_spe_alarm < 10)` with a runtime guard `if max_spe > 1e-20`. On the current test data (`generate_ic_data` with seed 42/99, ncomp=3), `max_spe` is ~1e-28 — so the alarm-count assertion is never executed. The test now verifies only structural properties (correct vector lengths, `spe_limit > 0`). A bug in `spm_mewma_monitor` that zeroes all SPE values, or sets every `spe_alarm` to `true`, would pass undetected because `max_spe == 0.0 <= 1e-20` satisfies the guard.

No other test in `spm/tests.rs` verifies `spm_mewma_monitor`'s IC SPE-alarm count on non-degenerate data. The tests at lines 285-309 cover `spm_monitor` (not MEWMA), and `test_mewma_asymptotic_ic_data` (line 1606) does not assert SPE alarm counts.

**Fix:** Add a companion test that uses data with residual variance above the machine-noise floor, ensuring the SPE assertion exercises real signal. The simplest approach is to reduce `ncomp` below the data's intrinsic rank so reconstruction is imperfect, or add a noise floor large enough that SPE values exceed 1e-20:

```rust
#[test]
fn test_mewma_ic_spe_alarms_with_nontrivial_residual() {
    use crate::spm::mewma::{spm_mewma_monitor, MewmaConfig};
    let m = 20;
    let argvals = uniform_grid(m);
    let data = generate_ic_data(40, m, 42);
    let config = SpmConfig {
        ncomp: 1,  // deliberately under-fit so SPE > 0 above machine noise
        alpha: 0.05,
        ..SpmConfig::default()
    };
    let chart = spm_phase1(&data, &argvals, &config).unwrap();
    let monitor_data = generate_ic_data(20, m, 99);
    let mewma_config = MewmaConfig { lambda: 0.2, ncomp: 1, alpha: 0.05, asymptotic: true };
    let result = spm_mewma_monitor(&chart, &monitor_data, &argvals, &mewma_config).unwrap();
    // With ncomp=1, reconstruction error is above machine noise
    let max_spe = result.spe.iter().cloned().fold(0.0_f64, f64::max);
    assert!(max_spe > 1e-10, "SPE should be above machine noise with ncomp=1");
    let n_spe_alarm = result.spe_alarm.iter().filter(|&&a| a).count();
    assert!(n_spe_alarm < 15, "IC data should have few SPE alarms: got {n_spe_alarm}/20");
}
```

---

### WR-02: `fix_svd_signs` silent no-op on all-NaN rotation column; NaN propagation not guarded

**File:** `fdars-core/src/regression.rs:184-200`

**Issue:** When a rotation column contains NaN values (which faer or nalgebra could emit for degenerate numerical inputs), `partial_cmp` returns `None` for any comparison involving NaN. The `unwrap_or(std::cmp::Ordering::Equal)` in `max_by` treats all NaN comparisons as ties, so `j_max` lands on the last index by `max_by`'s tie-breaking behavior. If `rotation[(j_max, k)]` is NaN, the comparison `rotation[(j_max, k)] < 0.0` evaluates to `false` (NaN comparisons are always false in IEEE 754), and the sign flip is skipped — silently. The downstream unscaling loop and FpcaResult then propagate NaN into the public output without any `FdarError` being returned.

This is not a new path (NaN inputs to SVD are pre-existing), but `fix_svd_signs` introduces a new silent-failure point: a rotation column that is all-NaN or mixed-NaN will produce a result that appears successful but is corrupted.

**Fix:** Add a NaN guard in `fix_svd_signs` and return an error, or at minimum add a debug assertion. Because `fix_svd_signs` is a private `fn` with an infallible signature, the simplest fix is a `debug_assert!` that fires in test builds:

```rust
fn fix_svd_signs(rotation: &mut FdMatrix, scores: &mut FdMatrix, ncomp: usize) {
    let m = rotation.nrows();
    let n = scores.nrows();
    for k in 0..ncomp {
        let j_max = (0..m)
            .max_by(|&a, &b| {
                rotation[(a, k)]
                    .abs()
                    .partial_cmp(&rotation[(b, k)].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        debug_assert!(
            rotation[(j_max, k)].is_finite(),
            "fix_svd_signs: rotation[{j_max},{k}] is not finite — SVD produced NaN/Inf"
        );
        if rotation[(j_max, k)] < 0.0 {
            for j in 0..m {
                rotation[(j, k)] = -rotation[(j, k)];
            }
            for i in 0..n {
                scores[(i, k)] = -scores[(i, k)];
            }
        }
    }
}
```

Alternatively, change the function signature to `Result<(), FdarError>` and return `ComputationFailed` on NaN detection, which is consistent with the project convention of never silently corrupting output.

---

## Info

### IN-01: Equivalence test proves call-to-call determinism but not sequential-vs-parallel equivalence

**File:** `fdars-core/src/classification/cv.rs:367-399`

**Issue:** `test_fclassif_cv_parallel_matches_sequential` calls `fclassif_cv` twice with the same seed and asserts bit-for-bit equality of the two results. Under the `parallel` feature, both calls use Rayon — so the test proves that Rayon's indexed parallel collect is deterministic across identical runs. It does not compare the `parallel` output against the `sequential` output, which is the cross-mode equivalence contract implied by the test name.

This is a minor naming/scope issue. The test is correct for what it does (determinism), and the indexed `collect()` guarantees identical order between sequential and parallel results because each fold value is a scalar and there is no floating-point reduction that could reorder. Cross-mode inequality is therefore structurally impossible here, but the test does not verify it explicitly.

**Fix:** Either rename the test to `test_fclassif_cv_deterministic_across_runs` to accurately describe what it asserts, or add a cfg-conditional inner block that compares against a sequentially-forced reference:

```rust
// Under parallel feature, also run a sequential reference for cross-mode verification
#[cfg(feature = "parallel")]
{
    let res_seq = {
        // Force sequential by calling the underlying logic directly if accessible,
        // or document that cross-mode equivalence is guaranteed by indexed collect
    };
}
```

If cross-mode equivalence is accepted as structurally guaranteed (which it is, given each fold returns a deterministic f64 scalar), a code comment explaining this in the test body is sufficient.

---

_Reviewed: 2026-08-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
