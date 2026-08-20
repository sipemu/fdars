---
phase: 32-flexible-mixed-effects-regression
reviewed: 2026-08-20T00:00:00Z
depth: deep
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/famm.rs
  - fdars-core/src/fof_regression.rs
  - fdars-core/src/lib.rs
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 32: Code Review Report

**Reviewed:** 2026-08-20
**Depth:** deep
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 32 adds `dense_flmm`, `multi_famm`, `fast_fmm` to `famm.rs` and
`fof_re_regression`/`predict_fof_re` to `fof_regression.rs`. The structural
design is sound — correct reuse of existing REML-EM helpers, correct
`pub(crate)` promotion, correct column-major indexing, correct random-effect
transpose in `fof_re_regression`, and sound panic safety throughout the public
API. The primary critical finding is that `fast_fmm` silently ignores its own
`config.max_iter` and `config.tol` fields, rendering two of three documented
configuration knobs inert. Additionally, the running-mean smoother produces
incorrect window widths for even `smooth_window` values, several required test
cases are absent, and the `random_slopes = true` code path is silently ignored
without error or diagnostic — a contract gap the rustdoc acknowledges but does
not guard against misuse.

---

## Critical Issues

### CR-01: `fast_fmm` ignores `config.max_iter` and `config.tol` — configuration fields are silently inert

**File:** `fdars-core/src/famm.rs:1543-1553`

**Issue:** `fast_fmm` calls `fit_scalar_mixed_model` (the original, hardcoded
variant at line 438) rather than `fit_scalar_mixed_model_tracked`.
`fit_scalar_mixed_model` hard-codes `50` iterations and `1e-10` relative
tolerance. As a result `FastFmmConfig::max_iter` and `FastFmmConfig::tol` are
completely ignored at runtime — a user who sets `max_iter: 5` to limit compute
on a large grid gets exactly the same 50-iteration run as the default. There is
no warning, no error, and no documentation of this divergence from the
declared config semantics.

The RESEARCH.md (Pitfall 3) describes the `PointwiseResult` / collect pattern
but says nothing about keeping iteration limits configurable; the struct was
designed with `max_iter` and `tol` for this purpose.

**Fix:** Replace the call on line 1546 with `fit_scalar_mixed_model_tracked`,
passing `config.max_iter` and `config.tol`:

```rust
// In fast_fmm, Step 1 loop (line 1543-1553):
let per_point: Vec<PointwiseResult> = iter_maybe_parallel!(0..m)
    .map(|t| {
        let y_t: Vec<f64> = data.column(t).to_vec();
        let r = fit_scalar_mixed_model_tracked(
            &y_t,
            &subject_map,
            n_subjects,
            covariates,
            p,
            config.max_iter,
            config.tol,
        );
        PointwiseResult {
            gamma: r.result.gamma,
            sigma2_u: r.result.sigma2_u,
            sigma2_eps: r.result.sigma2_eps,
        }
    })
    .collect();
```

`fit_scalar_mixed_model_tracked` is already `fn` (module-private) so no
visibility change is needed — it is in the same file.

---

## Warnings

### WR-01: Running-mean smoother produces incorrect window size for even `smooth_window` values

**File:** `fdars-core/src/famm.rs:1573-1583`

**Issue:** The half-width computation is `half = w / 2` (integer division) and
the actual window used is `[t - half, t + half + 1)` — a span of
`2*half + 1 = w + 1` elements for even `w`.

```
w=2: half=1 → actual count 3 (not 2)
w=4: half=2 → actual count 5 (not 4)
```

For odd `w` (including the default of 3) the count equals `w` exactly —
correct. Users who pass an even window get a silently wider smoother and no
diagnostic. The discrepancy compounds with edge truncation so the actual
effective window at the boundary is also wrong.

**Fix:** Validate at the entry point that `smooth_window` is odd, or document
the rounding behaviour. The simplest robust fix is to force the window to the
nearest odd value:

```rust
// After the smooth_window == 0 validation in fast_fmm:
let w = if config.smooth_window % 2 == 0 {
    config.smooth_window + 1  // round up to odd
} else {
    config.smooth_window
};
// ... then use `w` throughout Step 2
```

Alternatively, add `InvalidParameter` if even and document that only odd
window widths are supported.

---

### WR-02: `DenseFlmmConfig::random_slopes = true` is silently ignored — no error, no diagnostic

**File:** `fdars-core/src/famm.rs:930-937` (config doc), `famm.rs:1143`

**Issue:** The rustdoc on `random_slopes` states it is "accepted (no error) but
silently falls back to intercept-only estimation." A caller who sets
`random_slopes: true` expecting two-random-effect estimation gets a
single-intercept model with identically zero `sigma2_slope`, no warning, and
a result type that misleadingly carries a `sigma2_slope` field implying slope
estimates were produced.

This is not merely a deferred feature: the public API takes the field, stores
it, and the result struct exposes `sigma2_slope` whose semantics are unclear
without reading the struct-level rustdoc. Users of programmatic pipelines
(serde round-trips, configs loaded from files) can silently get wrong models.

**Fix:** Either (a) return `FdarError::InvalidParameter` when
`random_slopes = true` until the feature is implemented, or (b) emit a
`#[cfg(debug_assertions)] eprintln!` warning. Option (a) is strictly safer:

```rust
if config.random_slopes {
    return Err(FdarError::InvalidParameter {
        parameter: "random_slopes",
        message: "random slope estimation is not yet implemented; \
                  use random_slopes: false".to_string(),
    });
}
```

This converts a silent correctness trap into a loud API signal and can be
removed when the feature ships.

---

### WR-03: `dense_flmm` does not validate `config.max_iter == 0`

**File:** `fdars-core/src/famm.rs:1031-1161` (`dense_flmm`)

**Issue:** When `config.max_iter = 0`, `fit_scalar_mixed_model_tracked` runs
`for _iter in 0..0` — the REML-EM loop body never executes. Gamma stays at
the OLS initialisation, BLUPs are computed from OLS residuals with the
initial variance estimates, `n_iter` is reported as 0, and `converged` is
reported as `false`. The result is silently numerically incorrect (no REML
update was applied) with no error returned. `MultiFammConfig` has the same gap
(propagated through `DenseFlmmConfig`). `FastFmmConfig.max_iter = 0` has the
same formal gap but is moot until CR-01 is fixed.

**Fix:**

```rust
// In dense_flmm, after the ncomp == 0 check:
if config.max_iter == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "max_iter",
        message: "must be >= 1".to_string(),
    });
}
```

Add the equivalent guard in `fast_fmm` when CR-01 is fixed.

---

### WR-04: Test plan gaps — REG-05-G and REG-05-K missing; REG-05-L incomplete

**File:** `fdars-core/src/famm.rs` (test module, lines 2212-2376)

**Issue:** Three required test cases from the phase validation architecture are
absent or incomplete:

- **REG-05-G** (`test_dense_flmm_converged`): No test asserts that
  `DenseFlmmResult.converged` is `true` for well-conditioned synthetic data or
  `false` when `max_iter` is forced to 1. The `converged` field is computed
  but never checked in any test.

- **REG-05-K** (`test_fast_fmm_detects_effect`): No test verifies that
  `fast_fmm` detects a known non-zero fixed effect — the `beta_matrix` norm is
  never asserted positive. The existing `test_fast_fmm_basic` only checks
  shapes and p-value range.

- **REG-05-L** (`test_fast_fmm_empty_error`): `test_fast_fmm_invalid_inputs`
  checks `smooth_window = 0` and subject-ID mismatch, but does not pass an
  empty (`0 × 0`) matrix, which is a distinct code path validated at line 1510.

These gaps reduce confidence that the new paths are correct without execution
evidence.

**Fix:** Add the three missing tests following the existing test-helper
pattern:

```rust
#[test]
fn test_dense_flmm_converged() {
    let (data, subject_ids, covariates, _t) = generate_fmm_data(10, 3, 20);
    let cfg = DenseFlmmConfig { max_iter: 100, ..Default::default() };
    let result = dense_flmm(&data, &subject_ids, Some(&covariates), &cfg).unwrap();
    assert!(result.converged, "should converge with 100 iterations");

    // Force 1 iteration — converged must be false for realistic data
    let tight_cfg = DenseFlmmConfig { max_iter: 1, tol: 1e-30, ..Default::default() };
    let result2 = dense_flmm(&data, &subject_ids, Some(&covariates), &tight_cfg).unwrap();
    // n_iter == 1, converged likely false
    assert_eq!(result2.n_iter, 1);
}

#[test]
fn test_fast_fmm_detects_effect() {
    let (data, subject_ids, covariates, _t) = generate_fmm_data(10, 3, 20);
    let cfg = FastFmmConfig { compute_inference: true, ..Default::default() };
    let result = fast_fmm(&data, &subject_ids, Some(&covariates), &cfg).unwrap();
    let norm_sq: f64 = (0..result.beta_matrix.ncols())
        .map(|t| result.beta_matrix[(0, t)].powi(2))
        .sum();
    assert!(norm_sq > 0.0, "beta_matrix row 0 should be non-zero for data with covariate effect");
}

#[test]
fn test_fast_fmm_empty_data_error() {
    let empty = FdMatrix::zeros(0, 0);
    let cfg = FastFmmConfig::default();
    assert!(fast_fmm(&empty, &[], None, &cfg).is_err());
}
```

---

## Info

### IN-01: `build_subject_map` uses O(n_subjects) linear search per observation

**File:** `fdars-core/src/famm.rs:191-194`

**Issue:** The mapping from subject ID to compact index uses
`.iter().position(|u| u == id)` — O(n_subjects) per observation, O(n *
n_subjects) total. For FDA datasets with many subjects this is harmless, but
for `fast_fmm` this runs once before the O(m) per-gridpoint loop, so the cost
is paid only once. The fallback `.unwrap_or(0)` on a failed position is
unreachable because `unique_ids` contains all IDs by construction — the
`unwrap_or(0)` silently maps unknown IDs to subject 0, which would be wrong if
the input were somehow corrupted, but validation upstream prevents this.

**Fix (optional):** Use a `HashMap<usize, usize>` for O(1) average lookup,
worth considering if n_subjects is ever large (>1000). Not urgent given current
use cases.

---

### IN-02: `DenseFlmmResult` missing `#[cfg_attr(feature = "serde", ...)]` on the struct attribute line

**File:** `fdars-core/src/famm.rs:962-994`

**Issue:** `DenseFlmmResult`, `MultiFammResult`, and `FastFmmResult` all carry
`#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`
which is correct. However `FofReResult` in `fof_regression.rs` (line 565-600)
also has serde gating. All four result structs embed `FdMatrix` which itself
has serde gating, so the conditional compilation chain is consistent. No
action required — this is an informational note confirming correctness.

---

### IN-03: `sigma2_eps` in `DenseFlmmResult` is the mean across FPC components, not a per-observation residual variance

**File:** `fdars-core/src/famm.rs:978-979`

**Issue:** The field doc says "Average residual variance across FPC components"
which is accurate, but users familiar with R's `lmer` output may expect a
single pooled residual variance. The averaging across k components with
different score scales (before back-projection) means `sigma2_eps` is not
directly comparable to `sigma2_eps` from a single-component `fmm` or to R's
`sigma(lmer_fit)^2`. The RESEARCH.md documents the FPC-score parametrization
divergence but the field-level doc does not cross-reference it.

**Fix:** Expand the rustdoc on `sigma2_eps` to note the parametrization:

```rust
/// Mean residual variance averaged across FPC-score component models.
///
/// Each per-component model operates on L²-normalized scores; this average
/// is on the normalized scale and is not directly comparable to the marginal
/// residual variance `σ²_ε` from R's `lmer()`. See the struct-level
/// parametrization note.
pub sigma2_eps: f64,
```

---

_Reviewed: 2026-08-20_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
