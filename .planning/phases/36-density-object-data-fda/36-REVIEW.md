---
phase: 36-density-object-data-fda
reviewed: 2026-08-21T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - fdars-core/src/density_fda.rs
  - fdars-core/src/lib.rs
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 36: Code Review Report

**Reviewed:** 2026-08-21
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

`density_fda.rs` is a clean, well-structured module with good validation
coverage and correct algorithm structure matching the fdadensity R reference.
The LQD transform/inverse, FVE computation, column-major indexing in
`lqd_fpca` and `wasserstein_barycenter`, the θ_ψ rescaling step, the
`dedup_adjacent` guard, and the post-hoc non-finite ψ check are all
implemented correctly.

Four warnings and three info-level items remain. No critical (data-loss /
security / crash-on-valid-input) defects were found. The most impactful
finding is the density spike from an under-clamped `eps` in
`quantile_density_from_q` (WR-01) which degrades the numerical quality of
`wasserstein_barycenter` at distribution tails. WR-02 documents that the
barycenter uses a noisier finite-difference inversion while the singleton test
tolerance is 100× looser than the research spec. WR-03 and WR-04 are
correctness-adjacent quality items.

`lib.rs` modification is minimal and correct: only `pub mod density_fda;` and
the six crate-root re-exports were added; no existing signatures were changed.

---

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: `quantile_density_from_q` — eps clamp too small, produces tail spikes

**File:** `fdars-core/src/density_fda.rs:642-643`
**Issue:** The quantile density is approximated by finite differences on
`q_scaled`, then inverted as `1.0 / dq.max(eps)` with `eps = 1e-12`. At
distribution tails, where the quantile function is very steep (e.g. a
Gaussian near its support boundary), the central-difference derivative `dq`
computed over a coarse t_grid can be legitimately small — smaller than `1e-12`
even for well-behaved densities. The clamp then produces values up to
`1/1e-12 = 1e12`, creating extreme density spikes at tail grid points. These
spikes are passed to `linear_interp` and `trapz` for renormalization. While
renormalization suppresses the spike's contribution to the integral, the
spike dominates `linear_interp`'s boundary clamping: any `argvals` query
outside the spike zone inherits the extreme boundary value, yielding a
visibly distorted barycenter density near the support edges.

The effect is subtle for smooth, well-separated Gaussians on wide grids but
becomes pronounced for densities with sharp tails on narrower grids (< 101
points), and for any barycenter of densities with different support widths.

**Fix:** Raise the clamp to a physically meaningful floor. The quantile
density `q(t) = dQ/dt` is the reciprocal of the density `f(Q(t))`. For a
density bounded above by some `f_max`, the minimum quantile density is
`1/f_max`. A practical floor of `1e-6` (implying a maximum density spike of
`1e6`) is still wide for any real continuous density, and eliminates the
`1e12` pathology while preserving the algorithm's intent:

```rust
// density_fda.rs line 642
let eps = 1e-6_f64;   // was 1e-12; floor prevents 1e12 tail spikes
qd.iter().map(|&dq| 1.0 / dq.max(eps)).collect()
```

Additionally consider clamping the pre-inversion derivative to a maximum
(e.g. `dq.clamp(eps, 1.0 / eps)`) so that implausibly large quantile
densities (from non-monotone numerical noise) are also bounded.

---

### WR-02: `wasserstein_barycenter` uses finite-difference inversion — singleton test tolerance 100× looser than spec

**File:** `fdars-core/src/density_fda.rs:503-504, 886`
**Issue:** Two related problems:

1. `wasserstein_barycenter` inverts the mean quantile function `q_bar` to a
   density via `quantile_density_from_q` (finite central differences), while
   `inverse_lqd` inverts via the analytic `exp(-psi)` formula. For `n=1`
   (singleton barycenter), the quantile average `q_bar = Q_1`, and inverting
   it should recover the input density to the same accuracy as a direct
   round-trip. But the finite-difference path introduces additional O(h²)
   noise that the `exp(-psi)` path avoids. The research doc (Pattern 3, line
   250-253) describes an alternative: for the barycenter, one can reuse the
   inverse_lqd-style back-map by noting `q_bar` is the quantile function and
   using `linear_interp(q_bar_rescaled, t_grid, x)` (Appendix of Pattern 3).

2. The test `barycenter_singleton_reduction` (line 886) asserts `max_err <
   1e-2`. The research validation map (line 528) specifies `L∞ < 1e-4` for
   this test. The implementation is 100× looser than spec, masking the
   degradation introduced by the finite-difference path.

**Fix (option A — minimal, keeps finite differences):** Raise the eps clamp
per WR-01 and tighten the test to `1e-3` (achievable with `eps=1e-6`).
Acknowledge in rustdoc that the barycenter uses finite differences.

**Fix (option B — preferred, aligns with research spec):** Implement the
density inversion for `wasserstein_barycenter` using the same
`interpolate-and-normalize` back-map as `inverse_lqd`, without going through
`quantile_density_from_q`:

```rust
// Instead of quantile_density_from_q:
// q_bar is the quantile function on t_grid; recover density by
// interpolating target_argvals onto (q_scaled, t_grid) and computing
// the reciprocal derivative, then normalize.
// Cleanest: factor out a private `invert_quantile_fn` shared by both.
```

Tighten the singleton test to `1e-3` or better after the fix.

---

### WR-03: `lqd_fpca` does not reject `ncomp == 0`

**File:** `fdars-core/src/density_fda.rs:548-603`
**Issue:** `ncomp = 0` is passed directly to `fdata_to_pc_1d`. If
`fdata_to_pc_1d` accepts 0 and returns an `FpcaResult` with an empty
`singular_values` vec, then `sv_sq` is empty, `total = 0.0`, the `if total >
0.0` branch is never taken, and `fve` becomes an empty `Vec<f64>`. This is
not a panic, but it silently returns a degenerate `LqdFpcaResult` with an
empty `fve`. Callers who `unwrap()` `result.fve.last()` (as in test line 1056)
would panic on the empty vec. The existing full-rank test uses `ncomp=4` so
it does not trigger this.

**Fix:** Add a guard immediately after the dimension checks:

```rust
// density_fda.rs, after line 567
if ncomp == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "ncomp",
        message: "ncomp must be at least 1".to_string(),
    });
}
```

---

### WR-04: `dedup_adjacent` silently drops non-monotone points rather than detecting them

**File:** `fdars-core/src/density_fda.rs:611-620`
**Issue:** `dedup_adjacent` keeps a point only when `xi > xd.last()`, i.e.
it silently skips any `xi <= xd.last()`. This handles exact duplicates
correctly, but if `q_scaled` were ever non-monotone (e.g., `q_scaled[i] <
q_scaled[i-1]` due to numerical noise in a future code path that reuses this
helper), the function silently discards the non-monotone point. The caller
then passes a truncated x-array to `linear_interp`, which uses binary search
and assumes strict monotonicity. Binary search on a non-monotone slice
produces silently incorrect interpolated values.

For the current callers (`inverse_lqd` and `wasserstein_barycenter`),
`q_scaled` is provably monotone because `q_range > 0` is guarded and the
rescaling is a positive linear map. So this is not an active bug. But as a
private helper the name "dedup" does not signal that it also silently eats
non-monotone values.

**Fix:** Either rename to `dedup_and_skip_nondecreasing` and document the
behavior, or add a debug assertion:

```rust
fn dedup_adjacent(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut xd = Vec::with_capacity(x.len());
    let mut yd = Vec::with_capacity(y.len());
    for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
        if i == 0 || xi > xd[xd.len() - 1] {
            xd.push(xi);
            yd.push(yi);
        } else {
            // In debug builds, flag non-monotone values rather than silently discarding
            debug_assert!(
                xi >= xd[xd.len() - 1],
                "dedup_adjacent: non-monotone x value {xi} at index {i}"
            );
        }
    }
    (xd, yd)
}
```

---

## Info

### IN-01: `lqd_transform` non-finite guard error message does not attribute the cause

**File:** `fdars-core/src/density_fda.rs:256-260`
**Issue:** The `ComputationFailed` error reports "non-finite ψ values produced;
check for zero/near-zero density values". In practice the most likely cause is
that `dens_norm[i]` underflowed to exactly 0.0 after dividing a very small
density by a large integral, producing `(-0.0_f64).ln() = -inf` and then ψ =
`+inf`. The message is slightly misleading because the validation at line 222
already rejects `density[i] <= 0.0` before normalization. After normalization
a previously-positive value could underflow. Improving the message would aid
debugging.

**Fix:** Extend the message:

```rust
detail: "non-finite ψ values produced; possible cause: a density value \
         underflowed to 0 after normalization (input density too small \
         relative to its maximum on this grid)".to_string(),
```

---

### IN-02: `wasserstein_barycenter` `argvals.len() != m` error message when `m == 0` is confusing

**File:** `fdars-core/src/density_fda.rs:413-419`
**Issue:** When `density_matrix` has `m == 0` columns, the error is:
```
InvalidDimension { parameter: "argvals", expected: "0 elements (matching density_matrix columns)", actual: "N elements" }
```
This is accurate but unhelpful — the root cause is a zero-column matrix, not
an argvals mismatch. The same pattern appears in `lqd_fpca` at line 562.

**Fix:** Separate the zero-column check:

```rust
if m == 0 {
    return Err(FdarError::InvalidDimension {
        parameter: "density_matrix",
        expected: "at least 1 column".to_string(),
        actual: "0 columns".to_string(),
    });
}
if argvals.len() != m {
    return Err(FdarError::InvalidDimension { ... });
}
```

---

### IN-03: Module doc comment list does not mention `LqdFpcaResult` in the re-export note

**File:** `fdars-core/src/density_fda.rs:1` (module doc)
**Issue:** The module doc comment references the five R analogues
(`dens2lqd`, `lqd2dens`, `FPCAdens`, `getWFmean`) and four public entry
points, but does not mention `LqdFpcaResult` as part of the public API
surface. Callers discovering the module through rustdoc would need to navigate
to the struct definition to find it.

**Fix:** Add `LqdFpcaResult` to the "# R baseline" or a new "# Types" section:

```rust
//! # Types
//! - [`LqdFpcaResult`] — output of [`lqd_fpca`], embedding the LQD-space
//!   [`crate::regression::FpcaResult`] plus fraction of variance explained.
```

---

_Reviewed: 2026-08-21_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
