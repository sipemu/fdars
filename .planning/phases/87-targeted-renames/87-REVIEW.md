---
phase: 87-targeted-renames
reviewed: 2026-09-08T22:15:00Z
depth: deep
files_reviewed: 11
files_reviewed_list:
  - fdars-core/src/depth/spatial.rs
  - fdars-core/src/depth/mod.rs
  - fdars-core/src/depth/tests.rs
  - fdars-core/src/metric/hausdorff.rs
  - fdars-core/src/metric/mod.rs
  - fdars-core/src/metric/tests.rs
  - fdars-core/src/fdata.rs
  - fdars-core/src/peer.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
  - fdars-core/src/explain/helpers/kernel.rs
  - fdars-core/benches/depth_benchmarks.rs
  - fdars-core/examples/02_functional_operations/main.rs
  - fdars-core/examples/05_depth_measures/main.rs
  - fdars-core/examples/06_distances_and_metrics/main.rs
  - fdars-core/tests/validate_against_r.rs
findings:
  critical: 0
  warning: 1
  info: 0
  total: 1
status: issues_found
---

# Phase 87: Code Review Report

**Reviewed:** 2026-09-08T22:15:00Z
**Depth:** deep (cross-file body-identity verification)
**Files Reviewed:** 16
**Status:** issues_found

## Summary

Reviewed commits c315dc6b (consolidation + LpeerResult rename) and f6b7cf81 (external call-site migration).

**Body identity — CONFIRMED.** All `_impl` bodies are byte-identical to the pre-refactor `_1d`/`_2d` bodies. Verified against pre-refactor state for each family:

- `functional_spatial_impl` ≡ old `functional_spatial_1d` body ✓
- `kernel_functional_spatial_1d_impl` ≡ old `kernel_functional_spatial_1d` body ✓
- `kernel_functional_spatial_2d_impl` ≡ old `kernel_functional_spatial_2d` body ✓ (uniform `vec![1.0; n_points]`, not Simpson's — the critical pitfall is correct)
- `hausdorff_self_1d_impl` / `hausdorff_cross_1d_impl` / `hausdorff_self_2d_impl` / `hausdorff_cross_2d_impl` ≡ old public bodies ✓
- `geometric_median_1d_impl` / `geometric_median_2d_impl` ≡ old public bodies ✓

**Dispatcher argument mapping — CONFIRMED.** All Dim branches forward the correct args:
- `geometric_median` Dim::Two correctly passes `argvals_t.unwrap_or(&[])` to `_2d_impl`; `_2d_impl` guards `m != expected_cols` so an empty argvals_t yields an empty result rather than a panic ✓
- `hausdorff_self` / `hausdorff_cross` Dim::Two correctly pass `argvals_t.unwrap_or(&[])` ✓
- `functional_spatial` Dim::Two passes `None` (matches old `_2d = _1d(.., None)`) ✓

**Call-site migrations — CONFIRMED.** All external callers in examples, benches, and validate_against_r.rs correctly wrap argvals in `Some()` for Dim::One, pass `None` for Dim::Two where appropriate, and select the correct Dim variant. Dim mismatches would produce wrong results; none found.

**hausdorff_3d** — still public, untouched ✓

**`#[must_use]`** — New `functional_spatial` and `kernel_functional_spatial` dispatchers carry `#[must_use]` (matching the removed attributes on `_1d`/`_2d`). `hausdorff_self`/`hausdorff_cross` lack `#[must_use]` — but the pre-refactor functions also lacked it, so no regression. `geometric_median` lacks `#[must_use]` — pre-refactor `geometric_median_1d/_2d` also lacked it ✓

**LpeerResult → LocalPeerResult** — 15 in-crate refs updated including struct definition, impl block, `lpeer` return type, doctests, lib.rs and prelude.rs re-exports. Historical CHANGELOG v0.36.0 entry correctly left intact ✓

One warning follows.

## Warnings

### WR-01: `kernel_functional_spatial` with `Dim::One` and `argvals: None` panics on non-empty data

**File:** `fdars-core/src/depth/spatial.rs:233-234`

**Issue:** The dispatcher calls `kernel_functional_spatial_1d_impl(data_obj, data_ori, argvals.unwrap_or(&[]), h)`. When `argvals` is `None` and the matrices are non-empty, `argvals.unwrap_or(&[])` produces an empty slice. Inside `kernel_functional_spatial_1d_impl`, the guard only checks `n_points == 0` (line 253); it does not check `argvals.len() == 0`. `simpsons_weights(&[])` returns an empty `Vec<f64>`. Then in `kfsd_weighted`, the inner loop `for t in 0..n_points { ... weights[t] * ... }` (line 168, 194) accesses `weights[t]` with `t >= 0` and `weights` empty — index-out-of-bounds panic.

The old `kernel_functional_spatial_1d` required `argvals: &[f64]` (non-optional), so callers always provided it. The new API silently accepts `None` for Dim::One and converts it to an empty slice rather than the uniform grid that `functional_spatial_impl` uses for its `None` path.

This differs from `functional_spatial`, which handles `None` correctly for Dim::One by building a uniform grid inside the impl. The `kernel_functional_spatial` dispatcher does not have an equivalent fallback.

**Fix:** Either validate in the dispatcher:

```rust
pub fn kernel_functional_spatial(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    argvals: Option<&[f64]>,
    h: f64,
    dim: Dim,
) -> Vec<f64> {
    match dim {
        Dim::One => {
            let n_points = data_obj.ncols();
            let default_av: Vec<f64>;
            let av = if let Some(av) = argvals {
                av
            } else {
                default_av = (0..n_points)
                    .map(|i| i as f64 / (n_points - 1).max(1) as f64)
                    .collect();
                &default_av
            };
            kernel_functional_spatial_1d_impl(data_obj, data_ori, av, h)
        }
        Dim::Two => kernel_functional_spatial_2d_impl(data_obj, data_ori, h),
    }
}
```

Or add a guard in `kernel_functional_spatial_1d_impl`:

```rust
if nobj == 0 || nori == 0 || n_points == 0 || argvals.len() != n_points {
    return Vec::new();
}
```

The second option is simpler and consistent with how `geometric_median_1d_impl` guards `argvals.len() != m`. If keeping the current `unwrap_or(&[])` approach, the guard is the minimum fix. If `None` for Dim::One should be a user error (not silently handled), update the doc comment to say "panics if `None` is passed with `Dim::One`" — but that would be a poor API contract for a public function.

---

_Reviewed: 2026-09-08T22:15:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
