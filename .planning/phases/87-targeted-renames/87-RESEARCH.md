# Phase 87: Targeted Renames - Research

**Researched:** 2026-09-08
**Domain:** Rust API refactoring — Dim-dispatch consolidation and type rename
**Confidence:** HIGH

## Summary

Phase 87 is a pure API-shape change for `fdars-core` v0.42.0: four `_1d`/`_2d` function families collapse onto single `Dim`-dispatched public signatures, and `LpeerResult` is renamed `LocalPeerResult`. No numeric or behavioral change. The v0.41.0 `Dim`-dispatch pattern is already established in the codebase (4 precedents: `mean`, `fraiman_muniz`, `modal`, `random_projection`, `random_tukey`) and is unambiguously simple — the public dispatcher `match dim { Dim::One | Dim::Two => impl_fn(...) }` — no enum-carried data, no `Option` parameters.

The key architectural decision for this phase concerns the Hausdorff family: `hausdorff_self_2d` and `hausdorff_cross_2d` take *different* argument types from their `_1d` counterparts (`argvals_s: &[f64], argvals_t: &[f64]` instead of `argvals: &[f64]`). This is unlike the other three families, where `_2d` merely delegates to `_1d` with no additional arguments. The Hausdorff `_2d` variants have genuinely divergent signatures; they CANNOT be folded into a `Dim`-dispatched signature using the same `Dim::One | Dim::Two => same_impl(...)` pattern without introducing a superset argument list or Option wrappers. The established codebase pattern provides no precedent for carrying extra per-Dim arguments — `Dim` is a plain enum with no data.

**Primary recommendation:** Keep `hausdorff_3d` separate (different data type entirely). For the `hausdorff_self`/`hausdorff_cross` pair: use the superset signature pattern — the consolidated `hausdorff_self(dim, data, argvals, argvals_t: Option<&[f64]>)` / `hausdorff_cross(dim, data1, data2, argvals, argvals_t: Option<&[f64]>)` where `Dim::One` ignores `argvals_t` and `Dim::Two` requires it (error if `None`). This is the only approach consistent with keeping `Dim` data-free and matching the project's existing `Option<&[f64]>` idiom already present in `functional_spatial_1d`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
None stated — all choices at Claude's discretion.

### Claude's Discretion
Pure infrastructure/refactor phase — all choices at Claude's discretion, guided by:
- The v0.41.0 `Dim`-dispatch pattern (`src/dim.rs` defines `enum Dim`). Replicate whatever an already-consolidated v0.41.0 function does.
- Byte-identical `_impl` bodies: move existing 1d/2d bodies verbatim into private `_impl` fns; the public fn only dispatches. A code-review gate confirms zero numeric drift.

Key constraints (from REQUIREMENTS / STATE):
- Old lone `_1d`/`_2d` public forms are REMOVED (breaking — intended for v0.42.0).
- All external construction/call sites (28 examples + doctests + `tests/`) updated to the new signatures (compile-time proof).
- Sequenced BEFORE Phase 88 (the large high-blast-radius suffix batch).

### Deferred Ideas (OUT OF SCOPE)
- NAME-04 (the large lone-`_1d`/`_2d` suffix batch across the rest of the crate) — Phase 88.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NAME-01 | `geometric_median` consolidated from `_1d`/`_2d` onto single `Dim`-dispatched signature with byte-identical `_impl` bodies | Bodies enumerated; dispatch pattern established; all callers listed |
| NAME-02 | `hausdorff_*` family consolidated onto `Dim`-dispatched signatures; `hausdorff_3d` decision required | All signatures read; `_3d` keep-separate decision justified; superset-arg approach for `self`/`cross` recommended |
| NAME-03 | `functional_spatial_*` and `kernel_functional_spatial_*` consolidated onto `Dim` dispatch | All signatures read; dispatch pattern fits existing precedent; all callers listed |
| NAME-05 | `LpeerResult` renamed to `LocalPeerResult` everywhere | All 15 references enumerated; no external call sites in examples/tests |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Dim-dispatch consolidation | fdars-core (src/) | — | Pure in-crate API reshape, no external dependency |
| Caller updates (examples, tests, benches) | fdars-core (external surface) | — | Compile-time gate proves coverage; benches are external crates too |
| Type rename (LpeerResult) | fdars-core src/peer.rs | lib.rs, prelude.rs | Definition + re-exports travel together |

---

## 1. The v0.41.0 `Dim`-Dispatch Pattern (CRITICAL)

### `enum Dim` — verbatim definition

[VERIFIED: fdars-core/src/dim.rs:19-26]

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Dim {
    /// 1D functional data (curves).
    One,
    /// 2D functional data (surfaces, flattened column-major).
    Two,
}
```

`Dim` carries **no data**. It is a plain C-style enum. Both arms currently delegate to the same `_1d` primitive; the `dim` argument makes caller intent explicit.

### Established consolidation pattern — verbatim examples

**Example 1: `fdata::mean`** [VERIFIED: fdars-core/src/fdata.rs:182-197]

```rust
/// Compute the mean function for 1D or 2D functional data via a unified [`Dim`] dispatch.
///
/// The 2D path never diverged from the 1D one (both compute the pointwise mean
/// over the flattened column-major grid), so both [`Dim`] arms forward to
/// [`mean_1d`]. The `dim` argument makes caller intent explicit and provides a
/// single future seam should a real 2D specialization ever be needed.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `dim` - Dimensionality selector ([`Dim::One`] or [`Dim::Two`])
#[must_use = "expensive computation whose result should not be discarded"]
pub fn mean(data: &FdMatrix, dim: Dim) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => mean_1d(data),
    }
}
```

The underlying `mean_1d` remains **public** in this existing consolidation. Its signature is:
[VERIFIED: fdars-core/src/fdata.rs:168]
```rust
pub fn mean_1d(data: &FdMatrix) -> Vec<f64> {
```

**Example 2: `depth::fraiman_muniz`** [VERIFIED: fdars-core/src/depth/fraiman_muniz.rs:42-58]

```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fraiman_muniz(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool, dim: Dim) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => fraiman_muniz_1d(data_obj, data_ori, scale),
    }
}
```

The underlying `fraiman_muniz_1d` is also `pub` [VERIFIED: fdars-core/src/depth/fraiman_muniz.rs:33].

**Key pattern observation (critical for Phase 87):** In the v0.41.0 consolidations, the old `_1d` function is **NOT renamed** to `_impl` — it stays `pub fn ..._1d` and the new dispatcher delegates to it. However, Phase 87 REMOVES the old `_1d`/`_2d` public names. This means Phase 87 must:

1. Rename existing `pub fn name_1d(...)` → `fn name_impl(...)` (private)
2. Remove `pub fn name_2d(...)` entirely (or fold into `_impl`)
3. Add `pub fn name(dim: Dim, ...) { match dim { Dim::One | Dim::Two => name_impl(...) } }`

This is a slight variation from the v0.41.0 precedents (which kept `_1d` public alongside the dispatcher). Phase 87 is doing the v0.42.0 **removal** step.

### How consolidated signatures handle 1D vs 2D argument differences

For functions where `_2d` had **identical semantics** and merely forwarded to `_1d` (i.e., no extra args), the consolidated signature matches the `_1d` signature exactly, and `Dim` is appended as the final argument. Examples: `mean`, `fraiman_muniz`, `modal`, `random_projection`, `random_tukey` — all `_2d` variants took the same args and forwarded directly.

For `functional_spatial_1d` vs `functional_spatial_2d`:
- `_1d` takes `(data_obj, data_ori, argvals: Option<&[f64]>)`
- `_2d` takes `(data_obj, data_ori)` and calls `_1d(..., None)`
- Both reduce to the SAME `_1d` body. Consolidated signature: `functional_spatial(data_obj, data_ori, argvals: Option<&[f64]>, dim: Dim)` routing to `functional_spatial_impl`.

For `kernel_functional_spatial_1d` vs `kernel_functional_spatial_2d`:
- `_1d` takes `(data_obj, data_ori, argvals: &[f64], h: f64)` — uses `simpsons_weights(argvals)`
- `_2d` takes `(data_obj, data_ori, h: f64)` — uses `vec![1.0; n_points]`
- Bodies differ (different weights). Must route to separate `_impl` fns. The consolidated signature: `kernel_functional_spatial(data_obj, data_ori, argvals: Option<&[f64]>, h: f64, dim: Dim)` where `Dim::One` requires `argvals` (returns error or uniform if None), `Dim::Two` uses uniform weights.

For Hausdorff: see Section 3 below.

---

## 2. NAME-01: `geometric_median` Consolidation

### Current public signatures [VERIFIED: fdars-core/src/fdata.rs:1076-1115]

```rust
pub fn geometric_median_1d(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64>
```

```rust
pub fn geometric_median_2d(
    data: &FdMatrix,
    argvals_s: &[f64],
    argvals_t: &[f64],
    max_iter: usize,
    tol: f64,
) -> Vec<f64>
```

### Bodies

`geometric_median_1d` body [VERIFIED: fdars-core/src/fdata.rs:1082-1089]:
```rust
let (n, m) = data.shape();
if n == 0 || m == 0 || argvals.len() != m {
    return Vec::new();
}
let weights = simpsons_weights(argvals);
weiszfeld_iteration(data, &weights, max_iter, tol)
```

`geometric_median_2d` body [VERIFIED: fdars-core/src/fdata.rs:1107-1116]:
```rust
let (n, m) = data.shape();
let expected_cols = argvals_s.len() * argvals_t.len();
if n == 0 || m == 0 || m != expected_cols {
    return Vec::new();
}
let weights = simpsons_weights_2d(argvals_s, argvals_t);
weiszfeld_iteration(data, &weights, max_iter, tol)
```

### Analysis

The bodies differ only in weight computation (`simpsons_weights` vs `simpsons_weights_2d`). Both call `weiszfeld_iteration`. The consolidated approach:

- Private `fn geometric_median_impl(data, weights, max_iter, tol)` — the shared `weiszfeld_iteration` call (already private; just a thin rename is sufficient, or simply keep `weiszfeld_iteration` as-is)
- Public `fn geometric_median(data, argvals, argvals_t: Option<&[f64]>, max_iter, tol, dim: Dim)`
  - `Dim::One`: validates `argvals.len() == m`, computes `simpsons_weights(argvals)`, calls `weiszfeld_iteration`
  - `Dim::Two`: requires `argvals_t` (error or empty-guard if `None`), computes `simpsons_weights_2d(argvals, argvals_t)`, calls `weiszfeld_iteration`

Alternative simpler split: two private `_impl` fns (`geometric_median_1d_impl`, `geometric_median_2d_impl`) with the bodies moved verbatim; the public dispatcher branches on `dim`.

### External callers (blast radius) [VERIFIED: grep this session]

| File | Line | Old call | Must update to |
|------|------|----------|---------------|
| `examples/02_functional_operations/main.rs` | 8 | `use ... geometric_median_1d` (import) | `use ... geometric_median` |
| `examples/02_functional_operations/main.rs` | 127 | `geometric_median_1d(&mat, &t, 100, 1e-6)` | `geometric_median(&mat, &t, None, 100, 1e-6, Dim::One)` |
| `tests/validate_against_r.rs` | 2478 | `fdars_core::fdata::geometric_median_1d(&mat, &d.argvals, 100, 1e-6)` | `fdars_core::fdata::geometric_median(&mat, &d.argvals, None, 100, 1e-6, Dim::One)` |
| `tests/validate_against_r.rs` | 2498 | `fdars_core::fdata::geometric_median_1d(&mat, &d.argvals, 100, 1e-6)` | same |
| `src/lib.rs` | 675 | `geometric_median_1d, geometric_median_2d` (re-export) | `geometric_median` |

No `geometric_median_2d` external callers found in examples/tests/benches. [VERIFIED: grep this session]

---

## 3. NAME-02: `hausdorff_*` Family

### Current public signatures [VERIFIED: fdars-core/src/metric/hausdorff.rs:46-164]

```rust
// 1D self-distance — argvals: &[f64] (single grid)
pub fn hausdorff_self_1d(data: &FdMatrix, argvals: &[f64]) -> FdMatrix

// 1D cross-distance — argvals: &[f64] (single grid)
pub fn hausdorff_cross_1d(data1: &FdMatrix, data2: &FdMatrix, argvals: &[f64]) -> FdMatrix

// 3D point-cloud — ENTIRELY DIFFERENT type: &[(f64,f64,f64)]
pub fn hausdorff_3d(points1: &[(f64, f64, f64)], points2: &[(f64, f64, f64)]) -> f64

// 2D self-distance — TWO separate grids: argvals_s, argvals_t
pub fn hausdorff_self_2d(data: &FdMatrix, argvals_s: &[f64], argvals_t: &[f64]) -> FdMatrix

// 2D cross-distance — TWO separate grids
pub fn hausdorff_cross_2d(data1: &FdMatrix, data2: &FdMatrix, argvals_s: &[f64], argvals_t: &[f64]) -> FdMatrix
```

### Key architectural decision: `hausdorff_3d`

`hausdorff_3d` operates on `&[(f64, f64, f64)]` point clouds — a completely different type from `FdMatrix`. It is NOT a `_1d`/`_2d` pair member. It MUST remain as a separate public function `hausdorff_3d`. [VERIFIED: fdars-core/src/metric/hausdorff.rs:80-111]

Additionally, `hausdorff_self_2d` internally calls `hausdorff_3d` as a helper [VERIFIED: fdars-core/src/metric/hausdorff.rs:144]:
```rust
self_distance_matrix(n, |i, j| hausdorff_3d(&surfaces[i], &surfaces[j]))
```

### The argument-list divergence problem

`hausdorff_self_1d(data, argvals)` vs `hausdorff_self_2d(data, argvals_s, argvals_t)`: the `_2d` variant needs TWO grids, but `Dim` carries no data. This is unlike the other families.

**Recommendation: superset argument + `Option` for the second grid**

```rust
pub fn hausdorff_self(
    data: &FdMatrix,
    argvals: &[f64],
    argvals_t: Option<&[f64]>,
    dim: Dim,
) -> FdMatrix {
    match dim {
        Dim::One => hausdorff_self_1d_impl(data, argvals),
        Dim::Two => {
            let at = argvals_t.unwrap_or(&[]);
            hausdorff_self_2d_impl(data, argvals, at)
        }
    }
}

pub fn hausdorff_cross(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    argvals_t: Option<&[f64]>,
    dim: Dim,
) -> FdMatrix {
    match dim {
        Dim::One => hausdorff_cross_1d_impl(data1, data2, argvals),
        Dim::Two => {
            let at = argvals_t.unwrap_or(&[]);
            hausdorff_cross_2d_impl(data1, data2, argvals, at)
        }
    }
}
```

This pattern is consistent with `functional_spatial_1d`'s existing `argvals: Option<&[f64]>` idiom already in the codebase. The `_1d_impl` and `_2d_impl` bodies are byte-identical copies of the current `_1d`/`_2d` function bodies.

**Why not `Dim::Two(m1, m2)` or data-carrying `Dim`?** The `Dim` enum is `#[non_exhaustive]` and data-free by design. Adding a data-carrying variant is a breaking change to `Dim` itself and contradicts the crate's established pattern for all existing consolidations. Do not do this.

### External callers (blast radius) [VERIFIED: grep this session]

| File | Line | Old call | Must update to |
|------|------|----------|---------------|
| `examples/06_distances_and_metrics/main.rs` | 10 | `use ... hausdorff_self_1d` | `use ... hausdorff_self` |
| `examples/06_distances_and_metrics/main.rs` | 83 | `hausdorff_self_1d(&data, &t)` | `hausdorff_self(&data, &t, None, Dim::One)` |
| `tests/validate_against_r.rs` | 2789 | `fdars_core::metric::hausdorff_self_1d(&mat, &d.argvals)` | `fdars_core::metric::hausdorff_self(&mat, &d.argvals, None, Dim::One)` |
| `tests/validate_against_r.rs` | 2829 | `fdars_core::metric::hausdorff_cross_1d(&m1, &m2, &d.argvals)` | `fdars_core::metric::hausdorff_cross(&m1, &m2, &d.argvals, None, Dim::One)` |
| `tests/validate_against_r.rs` | 5263 | `fdars_core::metric::hausdorff_self_1d(&sub, &d.argvals)` | same pattern |
| `benches/depth_benchmarks.rs` | 14 | `use fdars_core::metric::{..., hausdorff_self_1d, ...}` | `hausdorff_self` |
| `benches/depth_benchmarks.rs` | 152 | group name string `"hausdorff_self_1d"` | update string |
| `benches/depth_benchmarks.rs` | 158 | `hausdorff_self_1d(black_box(data), black_box(&argvals))` | `hausdorff_self(black_box(data), black_box(&argvals), None, Dim::One)` |
| `src/metric/tests.rs` | 127 | `hausdorff_self_1d(&data, &argvals)` | `hausdorff_self(&data, &argvals, None, Dim::One)` |
| `src/metric/tests.rs` | 145 | `hausdorff_self_1d(&data, &argvals)` | same |
| `src/metric/tests.rs` | 154 | `hausdorff_self_1d(&empty, &[])` | `hausdorff_self(&empty, &[], None, Dim::One)` |
| `src/metric/tests.rs` | 308 | `hausdorff_self_2d(&data, &argvals_s, &argvals_t)` | `hausdorff_self(&data, &argvals_s, Some(&argvals_t), Dim::Two)` |
| `src/metric/tests.rs` | 322 | `hausdorff_self_2d(&empty, &[], &[])` | `hausdorff_self(&empty, &[], Some(&[]), Dim::Two)` |
| `src/metric/tests.rs` | 343 | `hausdorff_cross_1d(&data1, &data2, &argvals)` | `hausdorff_cross(&data1, &data2, &argvals, None, Dim::One)` |
| `src/metric/tests.rs` | 358–359 | `hausdorff_self_1d` + `hausdorff_cross_1d` | both updated |
| `src/metric/tests.rs` | 502 | `hausdorff_self_2d(&data, &argvals_s, &argvals_t)` | `hausdorff_self(&data, &argvals_s, Some(&argvals_t), Dim::Two)` |
| `src/metric/tests.rs` | 544 | `hausdorff_cross_2d(&data1, &data2, &argvals_s, &argvals_t)` | `hausdorff_cross(...)` with `Some(&argvals_t)` |
| `src/metric/tests.rs` | 691 | `hausdorff_self_1d(&data, &argvals)` | updated |
| `src/lib.rs` | 634–635 | `hausdorff_cross_1d, hausdorff_cross_2d, hausdorff_self_1d, hausdorff_self_2d` | `hausdorff_self, hausdorff_cross` |
| `src/metric/mod.rs` | 145–147 | `hausdorff_cross_1d, hausdorff_cross_2d, hausdorff_self_1d, hausdorff_self_2d` | `hausdorff_self, hausdorff_cross` |

Note: `hausdorff_3d` is NOT renamed and remains in all re-exports.

---

## 4. NAME-03: `functional_spatial_*` Consolidation

### Current public signatures [VERIFIED: fdars-core/src/depth/spatial.rs:17-230]

```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn functional_spatial_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    argvals: Option<&[f64]>,
) -> Vec<f64>

#[must_use = "expensive computation whose result should not be discarded"]
pub fn functional_spatial_2d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64>

#[must_use = "expensive computation whose result should not be discarded"]
pub fn kernel_functional_spatial_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    argvals: &[f64],
    h: f64,
) -> Vec<f64>

#[must_use = "expensive computation whose result should not be discarded"]
pub fn kernel_functional_spatial_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64) -> Vec<f64>
```

### Bodies

`functional_spatial_2d` [VERIFIED: fdars-core/src/depth/spatial.rs:77-79]:
```rust
pub fn functional_spatial_2d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    functional_spatial_1d(data_obj, data_ori, None)
}
```
Pure forward to `_1d` with `None`. Bodies are byte-identical at the semantic level — the `_2d` form simply passes `None` for `argvals`.

`kernel_functional_spatial_2d` [VERIFIED: fdars-core/src/depth/spatial.rs:219-230]:
```rust
pub fn kernel_functional_spatial_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64) -> Vec<f64> {
    // ...guard...
    let weights = vec![1.0; n_points];
    kfsd_weighted(data_obj, data_ori, h, &weights)
}
```
The `_2d` form uses uniform weights (`vec![1.0; n_points]`) while `_1d` uses `simpsons_weights(argvals)`. These have divergent bodies. The consolidated signature needs `argvals: Option<&[f64]>` where `Dim::Two` → uniform weights, `Dim::One` → Simpson weights.

### Consolidated signatures

```rust
// functional_spatial: _2d just calls _1d(None), so consolidated signature is:
pub fn functional_spatial(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    argvals: Option<&[f64]>,
    dim: Dim,
) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => functional_spatial_impl(data_obj, data_ori, argvals),
    }
}
// (Dim::Two callers pass None; Dim::One callers pass Some(argvals) or None for uniform)

// kernel_functional_spatial: bodies differ (weights differ)
pub fn kernel_functional_spatial(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    argvals: Option<&[f64]>,
    h: f64,
    dim: Dim,
) -> Vec<f64> {
    match dim {
        Dim::One => kernel_functional_spatial_1d_impl(data_obj, data_ori,
                       argvals.unwrap_or(&[]), h),
        Dim::Two => kernel_functional_spatial_2d_impl(data_obj, data_ori, h),
    }
}
```

### Internal callers (must also update)

`src/explain/helpers/kernel.rs` calls `functional_spatial_1d` at lines 133 and 173 [VERIFIED]. These are internal `pub(crate)` callers — must be updated to `functional_spatial(scores, scores, None, Dim::One)` / `functional_spatial(row, reference, None, Dim::One)`.

### External callers (blast radius) [VERIFIED: grep this session]

| File | Line | Old call | Must update to |
|------|------|----------|---------------|
| `examples/05_depth_measures/main.rs` | 9 | `use ... functional_spatial_1d` | `use ... functional_spatial` (+ `Dim`) |
| `examples/05_depth_measures/main.rs` | 114 | `functional_spatial_1d(&mat, &mat, None)` | `functional_spatial(&mat, &mat, None, Dim::One)` |
| `benches/depth_benchmarks.rs` | 8 | `use fdars_core::depth::{..., functional_spatial_1d, kernel_functional_spatial_1d, ...}` | updated names |
| `benches/depth_benchmarks.rs` | 116 | group name string `"functional_spatial_1d"` | update string |
| `benches/depth_benchmarks.rs` | 121 | `functional_spatial_1d(black_box(data), black_box(data), None)` | `functional_spatial(..., Dim::One)` |
| `benches/depth_benchmarks.rs` | 221–227 | `kernel_functional_spatial_1d(data, data, &argvals, 0.5)` | `kernel_functional_spatial(data, data, Some(&argvals), 0.5, Dim::One)` |
| `tests/validate_against_r.rs` | 840 | `fdars_core::depth::functional_spatial_1d(&mat, &mat, Some(&argvals))` | `fdars_core::depth::functional_spatial(&mat, &mat, Some(&argvals), Dim::One)` |
| `tests/validate_against_r.rs` | 853 | `fdars_core::depth::kernel_functional_spatial_1d(&mat, &mat, &argvals, h)` | `fdars_core::depth::kernel_functional_spatial(&mat, &mat, Some(&argvals), h, Dim::One)` |
| `tests/validate_against_r.rs` | 2290 | `fdars_core::depth::functional_spatial_1d(&mat, &mat, None)` | updated |
| `tests/validate_against_r.rs` | 2291 | `fdars_core::depth::functional_spatial_2d(&mat, &mat)` | `fdars_core::depth::functional_spatial(&mat, &mat, None, Dim::Two)` |
| `src/depth/tests.rs` | 143 | `functional_spatial_1d(&data, &data, None)` | `functional_spatial(&data, &data, None, Dim::One)` |
| `src/depth/tests.rs` | 153 | `functional_spatial_1d(&empty, &empty, None)` | same |
| `src/depth/tests.rs` | 242 | `kernel_functional_spatial_1d(&data, &data, &argvals, 0.5)` | `kernel_functional_spatial(&data, &data, Some(&argvals), 0.5, Dim::One)` |
| `src/depth/tests.rs` | 278 | same | same |
| `src/depth/tests.rs` | 293 | same | same |
| `src/depth/tests.rs` | 295 | same | same |
| `src/depth/tests.rs` | 305 | `kernel_functional_spatial_2d(&data, &data, 0.5)` | `kernel_functional_spatial(&data, &data, None, 0.5, Dim::Two)` |
| `src/depth/tests.rs` | 325 | `functional_spatial_1d(&data, &data, None)` | updated |
| `src/depth/tests.rs` | 326 | `functional_spatial_2d(&data, &data)` | `functional_spatial(&data, &data, None, Dim::Two)` |
| `src/depth/tests.rs` | 424 | `functional_spatial_1d(&data, &data, None)` | updated |
| `src/depth/tests.rs` | 446 | `functional_spatial_1d(&data, &data, None)` | updated |
| `src/depth/mod.rs` | 43–45 | all four old names | two new names |
| `src/lib.rs` | 648–650 | `functional_spatial_1d, functional_spatial_2d, kernel_functional_spatial_1d, kernel_functional_spatial_2d` | `functional_spatial, kernel_functional_spatial` |
| `src/prelude.rs` | 41 | `functional_spatial_1d, functional_spatial_2d` | `functional_spatial` |
| `src/explain/helpers/kernel.rs` | 133 | `depth::functional_spatial_1d(scores, scores, None)` | `depth::functional_spatial(scores, scores, None, Dim::One)` |
| `src/explain/helpers/kernel.rs` | 173 | `depth::functional_spatial_1d(row, reference, None)` | same |

---

## 5. NAME-05: `LpeerResult` → `LocalPeerResult`

### Definition site [VERIFIED: fdars-core/src/peer.rs:191-222]

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use = "expensive computation whose result should not be discarded"]
pub struct LpeerResult { ... }
```

### Function returning it [VERIFIED: fdars-core/src/peer.rs:460-466]

```rust
pub fn lpeer(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    subject_map: &[usize],
    config: &PeerConfig,
) -> Result<LpeerResult, FdarError>
```

The function name `lpeer` stays as-is; only the return type `LpeerResult` is renamed.

### `impl` block [VERIFIED: fdars-core/src/peer.rs:709]

```rust
impl LpeerResult { ... }
```

### Sibling for naming consistency

`PeerResult` is the sibling type [VERIFIED: fdars-core/src/lib.rs:615]. The rename `LpeerResult` → `LocalPeerResult` matches the full-name convention of `PeerResult` (peer = PeerResult, local-peer = LocalPeerResult).

### All 15 references [VERIFIED: fdars-core/src/peer.rs, src/lib.rs, src/prelude.rs]

| File | Line | Content |
|------|------|---------|
| `src/peer.rs` | 195 | `pub struct LpeerResult {` — **definition** |
| `src/peer.rs` | 466 | `-> Result<LpeerResult, FdarError>` — return type of `lpeer()` |
| `src/peer.rs` | 694 | `Ok(LpeerResult {` — construction |
| `src/peer.rs` | 709 | `impl LpeerResult {` — impl block |
| `src/peer.rs` | 1163 | doc comment: `for both \`PeerResult\` and \`LpeerResult\`` |
| `src/peer.rs` | 2097 | doc comment: `// LpeerResult::predict on training data...` |
| `src/peer.rs` | 2107 | `.expect("LpeerResult::predict on training data should succeed")` |
| `src/peer.rs` | 2112 | `"LpeerResult::predict vs fitted_values mismatch..."` |
| `src/peer.rs` | 2139 | `-> Result<LpeerResult, FdarError> = crate::peer::lpeer;` (doctest type alias) |
| `src/peer.rs` | 2149 | doc comment: `// Verify LpeerResult and PeerResult are constructible` |
| `src/peer.rs` | 2151 | `let _ = std::mem::size_of::<LpeerResult>();` |
| `src/peer.rs` | 2169 | `-> Result<LpeerResult, FdarError> = lpeer;` (doctest) |
| `src/peer.rs` | 2179 | `let _ = std::mem::size_of::<LpeerResult>();` |
| `src/lib.rs` | 615 | `lpeer, peer, LambdaChoice, LambdaMethod, LpeerResult, PeerConfig, ...` — re-export |
| `src/prelude.rs` | 117 | `lpeer, peer, LambdaChoice, LambdaMethod, LpeerResult, PeerConfig, ...` — re-export |

**No external callers** (examples, tests, benches) reference `LpeerResult` directly. [VERIFIED: grep this session returned empty]

Total references: **15 in `src/`**, all within `fdars-core`. Pure mechanical rename — no caller outside `src/` needs updating.

---

## 6. Re-export Lines to Update

### `src/lib.rs` changes required [VERIFIED: fdars-core/src/lib.rs:634-635, 648-650, 675]

```rust
// Line 634-635 (metric block) — remove: hausdorff_cross_1d, hausdorff_cross_2d, hausdorff_self_1d, hausdorff_self_2d
// Add: hausdorff_self, hausdorff_cross  (hausdorff_3d stays)

// Line 648-650 (depth block) — remove: functional_spatial_1d, functional_spatial_2d,
//   kernel_functional_spatial_1d, kernel_functional_spatial_2d
// Add: functional_spatial, kernel_functional_spatial

// Line 675 (fdata block) — remove: geometric_median_1d, geometric_median_2d
// Add: geometric_median

// Line 615 (peer block) — LpeerResult → LocalPeerResult
```

### `src/prelude.rs` changes required [VERIFIED: fdars-core/src/prelude.rs:41, 117]

```rust
// Line 41 (depth block) — remove: functional_spatial_1d, functional_spatial_2d
// Add: functional_spatial

// Line 117 (peer block) — LpeerResult → LocalPeerResult
```

### `src/depth/mod.rs` [VERIFIED: fdars-core/src/depth/mod.rs:42-45]

```rust
// Remove: functional_spatial_1d, functional_spatial_2d, kernel_functional_spatial_1d, kernel_functional_spatial_2d
// Add: functional_spatial, kernel_functional_spatial
```

### `src/metric/mod.rs` [VERIFIED: fdars-core/src/metric/mod.rs:145-147]

```rust
// Remove: hausdorff_cross_1d, hausdorff_cross_2d, hausdorff_self_1d, hausdorff_self_2d
// Add: hausdorff_self, hausdorff_cross  (hausdorff_3d stays)
```

---

## Architecture Patterns

### Recommended Pattern for Each Consolidation

**Case A — identical semantics (functional_spatial, geometric_median dispatch arms):**
```rust
// Private impl (byte-identical body of old _1d)
fn geometric_median_1d_impl(data: &FdMatrix, argvals: &[f64], max_iter: usize, tol: f64) -> Vec<f64> {
    // ... verbatim body of old geometric_median_1d ...
}
fn geometric_median_2d_impl(data: &FdMatrix, argvals_s: &[f64], argvals_t: &[f64], max_iter: usize, tol: f64) -> Vec<f64> {
    // ... verbatim body of old geometric_median_2d ...
}

// Public dispatcher
pub fn geometric_median(
    data: &FdMatrix,
    argvals: &[f64],
    argvals_t: Option<&[f64]>,
    max_iter: usize,
    tol: f64,
    dim: Dim,
) -> Vec<f64> {
    match dim {
        Dim::One => geometric_median_1d_impl(data, argvals, max_iter, tol),
        Dim::Two => geometric_median_2d_impl(
            data, argvals, argvals_t.unwrap_or(&[]), max_iter, tol
        ),
    }
}
```

**Case B — `_2d` is a pure forward (functional_spatial):**
```rust
fn functional_spatial_impl(data_obj: &FdMatrix, data_ori: &FdMatrix, argvals: Option<&[f64]>) -> Vec<f64> {
    // ... verbatim body of old functional_spatial_1d ...
}

pub fn functional_spatial(data_obj: &FdMatrix, data_ori: &FdMatrix, argvals: Option<&[f64]>, dim: Dim) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => functional_spatial_impl(data_obj, data_ori, argvals),
    }
}
```

### Recommended Project Structure (no change needed)

The consolidation happens entirely within existing files. No new files required.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Enum-carried dim args | `Dim::Two(&[f64])` variant | superset public signature with `Option<&[f64]>` | `Dim` is `#[non_exhaustive]` data-free by design; adding data is a breaking `Dim` change |
| New dispatch mechanism | new trait/dispatch infrastructure | plain `match dim { ... }` | consistent with all 5 existing codebase precedents |

---

## Common Pitfalls

### Pitfall 1: Forgetting bench callers
**What goes wrong:** `benches/depth_benchmarks.rs` imports `hausdorff_self_1d` and `functional_spatial_1d`/`kernel_functional_spatial_1d`. Benches compile as external crates. Updating `src/` only while forgetting benches causes a bench-compile failure.
**How to avoid:** The blast-radius table above lists bench lines explicitly. The gate `cargo build --benches` (or `cargo bench --no-run`) must be part of the verification plan.

### Pitfall 2: `kernel_functional_spatial_2d` uses uniform weights, not Simpson's
**What goes wrong:** Treating `kernel_functional_spatial_2d` as a "pure forward" to `_1d` would change its numeric behavior. The `_2d` body uses `vec![1.0; n_points]` while `_1d` uses `simpsons_weights(argvals)`.
**How to avoid:** The `_2d_impl` body must be preserved verbatim with uniform weights. A code-review gate compares before/after bodies byte-by-byte.

### Pitfall 3: `hausdorff_3d` getting accidentally removed
**What goes wrong:** Treating `hausdorff_3d` as a `_1d`/`_2d` sibling and removing it.
**How to avoid:** `hausdorff_3d` is NOT a `_1d`/`_2d` pair member — it operates on `&[(f64,f64,f64)]`, a different type. It STAYS public and unchanged. Additionally, `hausdorff_self_2d` and `hausdorff_cross_2d` delegate TO `hausdorff_3d` internally.

### Pitfall 4: `src/explain/helpers/kernel.rs` internal caller
**What goes wrong:** Internal `functional_spatial_1d` callers inside `src/explain/helpers/kernel.rs` (lines 133, 173) are not examples or tests — they're in `src/`. Updating only external callers leaves broken internal references.
**How to avoid:** Explicitly update these two lines as part of the `functional_spatial` consolidation task.

### Pitfall 5: `LpeerResult` in doctests within `src/peer.rs`
**What goes wrong:** The doctests at lines 2139, 2151, 2169, 2179 reference `LpeerResult` as a type in inline Rust code. These compile as external crate code and will fail if `LpeerResult` is removed from the public API before the doctest is updated.
**How to avoid:** The rename in the struct definition must propagate to all 15 locations atomically (single commit), including the re-exports in `lib.rs` and `prelude.rs`.

---

## Validation Architecture

### Compile-time gates (primary correctness proof)

The "byte-identical `_impl` bodies" guarantee means no behavioral test is needed — the existing test suite proves numeric correctness. The primary gate is **compilation**:

```
cargo build --features linalg,parallel
cargo build --examples
cargo build --benches
cargo test --features linalg,parallel  (all existing tests pass = no drift)
cargo test --doc                        (doctests including LpeerResult renaming)
```

### Full gate set (from CLAUDE.md)

```bash
cargo fmt --check
cargo clippy --all-targets --features linalg,parallel -- -D warnings
cargo test --features linalg,parallel
cargo build --features serde
cargo build --examples
cargo test --doc
```

### Phase verification strategy

| Check | What it proves |
|-------|---------------|
| `cargo build` green | All symbol renames are consistent |
| `cargo build --examples` | All 28 examples updated to new API surface |
| `cargo build --benches` | `depth_benchmarks.rs` updated |
| `cargo test --features linalg,parallel` | Byte-identical `_impl` bodies — no numeric drift |
| `cargo test --doc` | All doctests (including `LpeerResult`→`LocalPeerResult` in `peer.rs`) updated |
| Code-review gate (human) | `_impl` bodies compared to removed `_1d`/`_2d` bodies — confirms byte-identical |
| `cargo build --features serde` | `LocalPeerResult` serde derive still works |

### Wave 0 Gaps

None — no new test files needed. All coverage comes from the existing suite.

---

## Environment Availability

Step 2.6: SKIPPED — this phase is purely code/config changes within an existing Rust crate with no new external dependencies.

---

## Security Domain

Not applicable — this is a rename/API-shape phase with no new input-handling, cryptography, authentication, or network paths.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `geometric_median_2d` has no callers in examples/tests/benches | NAME-01 blast radius | Would miss a caller update — compile error would catch it |
| A2 | No `LpeerResult` references exist outside `fdars-core/src/` | NAME-05 | Would miss an external caller update — compile error would catch it |
| A3 | The `kernel_functional_spatial_2d` consolidated `argvals: Option<&[f64]>` where `None` → uniform weights is the right ergonomics | NAME-03 | Could pick a different convention (e.g., always require argvals and ignore for Dim::Two), but the Option idiom is already present in the codebase |

All three assumptions are protected by compile-time gates — even if wrong, the build fails before merge.

---

## Sources

### Primary (HIGH confidence)
- [VERIFIED: fdars-core/src/dim.rs:19-26] — `enum Dim` verbatim definition
- [VERIFIED: fdars-core/src/fdata.rs:182-197] — `mean` Dim-dispatch template
- [VERIFIED: fdars-core/src/depth/fraiman_muniz.rs:42-58] — `fraiman_muniz` Dim-dispatch template
- [VERIFIED: fdars-core/src/fdata.rs:1076-1116] — `geometric_median_1d/2d` signatures and bodies
- [VERIFIED: fdars-core/src/metric/hausdorff.rs:46-164] — all hausdorff signatures and bodies
- [VERIFIED: fdars-core/src/depth/spatial.rs:17-230] — all spatial depth signatures and bodies
- [VERIFIED: fdars-core/src/peer.rs:191-222, 460-466, 694, 709] — `LpeerResult` definition and usage
- [VERIFIED: fdars-core/src/lib.rs:615, 634-635, 648-650, 675] — lib.rs re-export lines
- [VERIFIED: fdars-core/src/prelude.rs:41, 117] — prelude.rs re-export lines
- [VERIFIED: grep this session] — all external caller enumeration

### Secondary (MEDIUM confidence)
- [ASSUMED] — `geometric_median_2d` has no callers outside definition files (grep returned empty; could miss a caller in an uncommitted file)

## Metadata

**Confidence breakdown:**
- Dim-dispatch pattern: HIGH — 5 existing examples read verbatim from source
- Function signatures: HIGH — all read directly from source files
- External caller enumeration: HIGH — grep confirmed, all lines cited
- LpeerResult references: HIGH — 15 references confirmed, all in src/

**Research date:** 2026-09-08
**Valid until:** Until any of the enumerated files change (stable codebase, no time-sensitivity)
