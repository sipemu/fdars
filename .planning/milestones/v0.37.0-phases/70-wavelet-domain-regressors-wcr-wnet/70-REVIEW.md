---
phase: 70-wavelet-domain-regressors-wcr-wnet
reviewed: 2026-09-04T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - fdars-core/src/wavelet/regression.rs
  - fdars-core/src/wavelet/mod.rs
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: resolved
resolved: 2026-09-04T00:00:00Z
resolution_commit: 6778e525
---

# Phase 70: Code Review Report

**Reviewed:** 2026-09-04
**Depth:** standard
**Files Reviewed:** 2
**Status:** resolved

## Resolution (2026-09-04)

All actionable findings fixed in commit `6778e525` (only
`fdars-core/src/wavelet/regression.rs` + its inline tests changed):

- **CR-01 — RESOLVED.** ncomp clamp changed to
  `config.ncomp.min(n.saturating_sub(1)).min(p)`, keeping the `[1, scores]` OLS
  design overdetermined (ncomp + 1 <= n). The `n < 3` guard already rejects the
  degenerate `n <= 1` case. New regression test
  `wcr_small_n_default_config_succeeds` fits the default config (ncomp = 5) at
  n = 4 and n = 3 and asserts Ok + finite β(t)/fitted/residuals.
- **WR-01 — RESOLVED.** `n_folds > n` now returns `InvalidParameter` in both
  `wnet` and `wnet_cv_lambda`; test `wnet_rejects_too_many_folds`.
- **WR-02 — RESOLVED.** Negative / NaN `tol` rejected in `elastic_net_cd` and at
  the `wnet` entry; test `wnet_rejects_negative_or_nan_tol`.
- **WR-03 — RESOLVED.** `max_iter == 0` rejected in `elastic_net_cd` and at the
  `wnet` entry; test `wnet_rejects_zero_max_iter`.
- **IN-01 / IN-02 — DEFERRED (Info).** Not fixed. IN-01 (near-duplicate
  `compute_fitted` / `compute_fitted_affine`) is a cosmetic dedup with no
  behavior change; IN-02 (ridge-nudge strength for `n < P`) would alter numerics
  and risks perturbing the tight β(t) recovery — both intentionally left for a
  future cleanup rather than forced here. Doc comments for the affected error
  paths were clarified.

Gates: 78 wavelet lib tests pass (was 73, +5), clippy `--all-targets
--features linalg,parallel` clean, `cargo fmt --check` clean. All pre-existing
recovery / determinism tests unchanged and passing.

## Summary

Phase 70 introduces `wcr` (wavelet-coefficient PCR/PLS regressor) and `wnet`
(per-coefficient elastic-net regressor with deterministic CV-λ) in a single new
file, `fdars-core/src/wavelet/regression.rs`. The only other change is adding
`pub mod regression;` to `wavelet/mod.rs` — verified, nothing else changed there.

The overall design is sound: the shared seams (`curves_to_coeff_design`,
`coeff_weights_to_beta_t`), the coordinate-descent engine, the soft-threshold
formula, and the CV machinery are all numerically correct. Determinism holds —
`create_folds` is seeded via `config.seed` and no HashMap iteration or unseeded
RNG touches the λ-selection path. No new crate dependencies were added, no
crate-root/prelude re-exports were made, `additive.rs` was not modified, and
`variable_selection` is not called. Clippy with `--all-targets --features
linalg,parallel` is clean.

One blocker was found: the ncomp-vs-n clamping strategy makes `wcr` return a
spurious `InvalidDimension` error for any call where `n <= config.ncomp` — this
includes the default configuration with `n` in `{3, 4, 5}`, all of which are
within the documented valid range.

---

## Critical Issues

### CR-01: `wcr` returns `InvalidDimension` when `ncomp >= n` (default config breaks for n ≤ 5) — RESOLVED

**File:** `fdars-core/src/wavelet/regression.rs:500,516-517`

**Issue:** `ncomp` is clamped at line 500 to `config.ncomp.min(n).min(p)`. When
`n <= config.ncomp` (e.g., `n = 5` with the default `ncomp = 5`, or any
`n <= ncomp`), the effective `ncomp` equals `n`. The OLS design matrix
`ols_design` is then `n × (1 + ncomp) = n × (n + 1)`. `ols_solve` checks
`n < p` at line 287 (where `p = 1 + ncomp = n + 1`) and immediately returns
`Err(FdarError::InvalidDimension { ... })`. This bubbles up as a
`ComputationFailed`/`InvalidDimension` even though the input is valid.

With the default `WcrConfig` (`ncomp = 5`), every call with `n ∈ {3, 4, 5}`
fails — and `n = 3` is the documented minimum. Users cannot use `wcr` with small
samples without explicitly setting `ncomp < n`, which is undocumented.

The same defect affects the case where a user explicitly sets `ncomp ≥ n` with
any dataset size.

```rust
// regression.rs line 500 — fix: clamp to at most n - 1
// (OLS with intercept needs n > 1 + ncomp, i.e., ncomp <= n - 2 for overdetermined;
// conservatively clamp to n - 1 to prevent the n < p rejection, but at minimum
// the intercept column makes p = 1 + ncomp, so ncomp must be <= n - 1)
let ncomp = config.ncomp.min(n.saturating_sub(1)).min(p);
```

Note: `saturating_sub(1)` gives 0 when `n = 0`, but `n < 3` is already rejected
above. With `n >= 3`: `ncomp <= n - 1` ensures `ols_design` is `n × (1 + ncomp)`
with `1 + ncomp <= n`, satisfying `ols_solve`'s precondition.

---

## Warnings

### WR-01: `n_folds > n` is not validated — silent partial CV — RESOLVED

**File:** `fdars-core/src/wavelet/regression.rs:869,891,894`

**Issue:** `wnet_cv_lambda` and `wnet` validate `n_folds >= 2` but do not check
`n_folds <= n`. `create_folds(n, n_folds, seed)` assigns fold labels `0..n_folds`
via `rank % n_folds` over only `n` observations. When `n_folds > n`, fold labels
`n..n_folds` receive zero observations; `fold_sets` has `n_folds` entries but
only `min(n, n_folds)` have non-empty test sets. The CV loop silently runs on
only `min(n, n_folds)` folds and skips the rest, producing misleading "K-fold"
semantics. With a user-set `n_folds = 1000` and `n = 10`, memory for 1000 fold
pairs (each O(n)) is allocated unnecessarily, and the caller sees no warning.

```rust
// Add this check in wnet_cv_lambda and wnet after the n_folds >= 2 check:
if config.n_folds > n {
    return Err(FdarError::InvalidParameter {
        parameter: "n_folds",
        message: format!(
            "n_folds ({}) must not exceed the number of observations ({n})",
            config.n_folds
        ),
    });
}
```

### WR-02: `tol` is not validated in `elastic_net_cd` — negative or NaN tol silently disables convergence — RESOLVED

**File:** `fdars-core/src/wavelet/regression.rs:702-703,790`

**Issue:** `elastic_net_cd` validates `alpha` and `lambda` but not `tol`. A
negative `tol` makes `max_delta < tol` always `false` (since `max_delta >= 0`),
so the loop always runs the full `max_iter` sweeps regardless of actual
convergence — silently degrading performance. A NaN `tol` causes the same
infinite-loop effect because `f64::NAN < f64::NAN` is `false`. Neither path
panics, but both produce silent unexpected behavior that contradict the documented
semantics.

```rust
// Add after the lambda validation block (around line 723):
if !tol.is_finite() || tol < 0.0 {
    return Err(FdarError::InvalidParameter {
        parameter: "tol",
        message: format!("tol must be finite and >= 0, got {tol}"),
    });
}
```

### WR-03: `max_iter = 0` is not validated — silently returns zero-coefficient model — RESOLVED

**File:** `fdars-core/src/wavelet/regression.rs:755`

**Issue:** `elastic_net_cd` does not validate `max_iter`. When `max_iter = 0` the
loop body at line 755 never executes, `beta` remains all-zeros, and the function
returns `(mu_y, vec![0.0; p])` — a trivially constant predictor that ignores all
features. The caller receives `Ok(...)` with no indication that the fit produced a
zero model. The `WnetConfig` default of `1000` protects most users, but a user
setting `max_iter: 0` gets silent wrong output. A minimum value of 1 should be
enforced.

```rust
// Add to elastic_net_cd validation block:
if max_iter == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "max_iter",
        message: "max_iter must be >= 1".to_string(),
    });
}
```

---

## Info

### IN-01: Near-duplicate `compute_fitted` / `compute_fitted_affine` helpers — DEFERRED

**File:** `fdars-core/src/wavelet/regression.rs:422,1069`

**Issue:** Two private helpers compute OLS predictions: `compute_fitted` (used by
`wcr`, intercept is column 0 of the design) and `compute_fitted_affine` (used by
`wnet`, intercept is a separate argument). The bodies differ only in where the
intercept comes from. This is not wrong but is a minor duplication — `wcr` could
use `compute_fitted_affine` if the OLS coefficients were unpacked first, or
`compute_fitted_affine` could call `compute_fitted` with a prepended intercept
column. No change required; left for future cleanup.

### IN-02: Ridge nudge in `recover_coeff_weights` may be numerically weak for very small `n` — DEFERRED

**File:** `fdars-core/src/wavelet/regression.rs:412-415`

**Issue:** The ridge stabilizer is `eps = 1e-10 * (trace / p).max(1e-12)`. When
`n < P` (e.g., `n = 3`, `P = m = 32`), the centered design `X_c` is `3 × 32`
and `X_c^T X_c` is rank ≤ 2. The `1e-10`-scale nudge is far too small to
regularize a near-null eigenvalue adequately; the Cholesky solve may still
succeed (the nudge prevents exactly-zero diagonals) but produces a numerically
unstable `beta_coeff` with enormous norm. In that regime `beta_t = IDWT(beta_coeff)`
may be non-meaningful. This is masked by the fact that the public API already
constrains `n >= 3` and typical usage has `n >> P`. No correctness contract is
violated — the doc says nothing about `n < P` stability. However, a larger
ridge or an explicit `n >= P` guard in `wcr` (post CR-01 fix) would make
behavior more predictable. Document the current limitation.

---

_Reviewed: 2026-09-04_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
