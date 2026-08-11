# Phase 10: Capability Gaps — Spline Interpolation & Functional Summary Statistics — Research

**Researched:** 2026-08-10
**Domain:** Rust FDA library — additive public API additions to `fdars-core`
**Confidence:** HIGH

---

## Summary

This phase adds two independent capability clusters to `fdars-core`. Both are purely additive (new public functions, no removals or signature changes) and all required building blocks already exist in the codebase.

**FEAT-01 (spline_interpolate):** The existing B-spline basis system in `basis/bspline.rs` provides everything needed: `construct_bspline_knots`, `bspline_basis`, and `bspline_basis_from_knots`. The interpolation pattern is already demonstrated by `pspline_evaluate` in `basis/pspline.rs`, which (1) constructs knots and a basis matrix on `argvals`, (2) solves for coefficients per curve, (3) evaluates at arbitrary `query_points` via `bspline_basis_from_knots`. `spline_interpolate` follows the same three-step fit-then-evaluate pattern, written directly in `helpers.rs` without using P-spline smoothing.

**FEAT-02 (functional summary statistics):** The five functions split cleanly into two families. The variance/std/covariance trio operates purely pointwise across rows of `FdMatrix` — O(n·m) for variance/std and O(n·m²) for covariance — with no depth machinery needed. The depth-based duo (`depth_based_median`, `trim_mean`) delegates to `fraiman_muniz_1d` in `depth/fraiman_muniz.rs`, which already returns a per-curve `Vec<f64>` depth score. The BACKLOG confirms these belong in `fdata.rs` and `helpers.rs` (EXPL-02 area note: "`fdata.rs`, `covariance.rs`") but `fdata.rs` is the natural home given `mean_1d` and `center_1d` already live there.

**Primary recommendation:** Implement `spline_interpolate` in `helpers.rs` (~80 lines) following the `pspline_evaluate` coefficient-solve pattern; implement all five summary-statistics functions in `fdata.rs` (~120 lines total), calling `fraiman_muniz_1d` for the depth-based pair.

---

## Project Constraints (from CLAUDE.md)

- **Audit-only milestone context completed:** v0.15.0 is the first *implementation* milestone — real `fdars-core/src/` changes.
- All public functions return `Result<T, FdarError>` (never panic on input). [VERIFIED: fdars-core/src/error.rs:1-51]
- Inline `#[cfg(test)] mod tests` pattern (no separate test files for unit tests). [VERIFIED: fdars-core/src/helpers.rs:590-844]
- `#[must_use]` on expensive computations. [VERIFIED: fdars-core/src/depth/fraiman_muniz.rs:31]
- Feature-gated parallelism via `iter_maybe_parallel!` macro. [VERIFIED: fdars-core/src/parallel.rs:42-55]
- Column-major `FdMatrix` — element `(row, col)` at index `row + col * nrows`. [VERIFIED: fdars-core/src/matrix.rs:10-12]
- No new external dependencies: "All four items carry no new crates." [VERIFIED: fdars-core/src/basis/bspline.rs — all needed B-spline code exists]
- MSRV 1.81.0 (no language features beyond 2021 edition stable-at-1.81).
- GSD workflow enforcement: work through `/gsd-execute-phase`, not direct edits.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FEAT-01 | `spline_interpolate(data, argvals, query_points, order) -> Result<FdMatrix, FdarError>` in `helpers.rs`; fits B-spline per curve (reusing `basis/`), evaluates at query_points; tests: reproduces input exactly at argvals (≤1e-10), cubic-spline known values at off-grid points | `construct_bspline_knots` + `bspline_basis` + `bspline_basis_from_knots` + `svd_pseudoinverse` (or direct solve) enable fit-then-evaluate; `pspline_evaluate` demonstrates the exact pattern |
| FEAT-02 | Five public functions: `trim_mean`, `depth_based_median`, `functional_covariance`, `functional_variance`, `functional_std`; accept `&FdMatrix`; return `Result<_, FdarError>`; inline tests verify cross-consistency (var = std²; cov diagonal = var; depth_based_median is argmax-depth index) | `fraiman_muniz_1d` returns per-curve depths; `mean_1d`/`center_1d` provide the mean subtraction needed for covariance; `FdMatrix::column(j)` gives O(1) zero-copy column access for pointwise passes |
</phase_requirements>

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Spline interpolation | Library (helpers.rs) | basis/ (reused) | Interpolation is a numeric utility; `helpers.rs` already owns `fdata_interpolate`, `linear_interp`, `cubic_hermite_interp` |
| Depth-based statistics | Library (fdata.rs) | depth/ (reused) | `trim_mean`/`depth_based_median` are FDA operations alongside existing `mean_1d`/`center_1d`; depth scores computed by `depth/` |
| Pointwise statistics | Library (fdata.rs) | — | `functional_variance`, `functional_std`, `functional_covariance` are purely pointwise matrix operations; same tier as `mean_1d` |

---

## Standard Stack

No new external crates are required. All building blocks are already present in the codebase.

### Core Building Blocks (all verified in this session)

| Symbol | File | Role |
|--------|------|------|
| `construct_bspline_knots(t_min, t_max, nknots, order) -> Vec<f64>` | `basis/bspline.rs:4-17` | Produces extended-boundary knot vector |
| `bspline_basis(t, nknots, order) -> Vec<f64>` | `basis/bspline.rs:100-125` | Basis matrix (column-major, `n*nbasis`) on original grid |
| `bspline_basis_from_knots(t, knots, order) -> Vec<f64>` | `basis/bspline.rs:62-83` | Evaluate B-spline basis using a **pre-built** knot vector at arbitrary points — the key function for evaluate-at-query-points |
| `svd_pseudoinverse(mat) -> Option<DMatrix<f64>>` | `basis/helpers.rs:9-36` | `pub(super)` — not directly callable from `helpers.rs`; must re-use `nalgebra` SVD inline or via a local solve |
| `pspline_evaluate(result, new_argvals) -> FdMatrix` | `basis/pspline.rs:163-187` | **Reference implementation** for the fit-then-evaluate pattern using `bspline_basis_from_knots` |
| `fraiman_muniz_1d(data_obj, data_ori, scale) -> Vec<f64>` | `depth/fraiman_muniz.rs:32-39` | Returns per-curve depth scores for the depth-based functions |
| `mean_1d(data) -> Vec<f64>` | `fdata.rs:166-178` | Pointwise mean across curves |
| `center_1d(data) -> FdMatrix` | `fdata.rs:211-236` | Mean-centered matrix (needed for sample covariance) |
| `FdMatrix::zeros(nrows, ncols)` | `matrix.rs:84-90` | Construct output matrix |
| `FdMatrix::column(j) -> &[f64]` | `matrix.rs:127-130` | Zero-copy column slice for pointwise passes |
| `FdMatrix::from_column_major(data, n, m) -> Result` | `matrix.rs:50-63` | Build output from flat buffer |

---

## Architecture Patterns

### Recommended Project Structure (no new files needed)

```
fdars-core/src/
├── helpers.rs        # ADD: spline_interpolate (FEAT-01)
└── fdata.rs          # ADD: functional_variance, functional_std,
                      #       functional_covariance, depth_based_median, trim_mean (FEAT-02)
```

Both files already have extensive inline test blocks (`#[cfg(test)] mod tests`) at their end. New tests append to those blocks.

---

### Pattern 1: Spline Interpolation via Fit-then-Evaluate

**What:** Build a B-spline basis on `argvals` → solve for coefficients per curve (least-squares via nalgebra) → evaluate at `query_points` using `bspline_basis_from_knots` with the same stored knot vector.

**When to use:** This is the only pattern for FEAT-01. It directly mirrors `pspline_evaluate` in `basis/pspline.rs` but without the P-spline smoothing penalty (interpolation, not smoothing).

**Key insight:** `bspline_basis_from_knots` is what makes the evaluate-on-different-grid step possible. It accepts the same knot vector built from `argvals` but evaluates at `query_points`. [VERIFIED: fdars-core/src/basis/bspline.rs:62-83]

**Coefficient solve:** Since there is no smoothing penalty, the coefficient solve is `coefs = (B^T B)^{-1} B^T y` where `B` is the `bspline_basis(argvals, nknots, order)` matrix. The pseudoinverse via nalgebra SVD is the safest approach (handles near-singular Gram matrices). `svd_pseudoinverse` in `basis/helpers.rs` is `pub(super)` and only visible within the `basis` module. For `helpers.rs`, implement the same pattern inline using `nalgebra::SVD::new` directly (3–4 lines).

**nknots choice:** For interpolation with `m` evaluation points and order `k`, use `nknots = m.saturating_sub(k).max(2)` so that `nbasis = nknots + order = m` — an interpolating system. The `pspline_fit_1d` uses `nknots = nbasis.saturating_sub(4).max(2)` for a preset `nbasis`; `spline_interpolate` should auto-compute `nknots` from `m` and `order`. [VERIFIED: fdars-core/src/basis/pspline.rs:79-80]

**Skeleton:**

```rust
// Source: derived from pspline_evaluate pattern (basis/pspline.rs:163-187) [VERIFIED]
pub fn spline_interpolate(
    data: &crate::matrix::FdMatrix,
    argvals: &[f64],
    query_points: &[f64],
    order: usize,
) -> Result<crate::matrix::FdMatrix, crate::FdarError> {
    let (n, m) = data.shape();
    // Input validation: order, sizes, query range
    // ...
    let nknots = m.saturating_sub(order).max(2);
    let t_min = argvals[0];
    let t_max = argvals[m - 1];
    // Build knot vector from argvals domain
    let knots = crate::basis::bspline::construct_bspline_knots(t_min, t_max, nknots, order);
    // Basis matrix on argvals: shape (m x nbasis) stored as flat column-major vec
    let basis_vals = crate::basis::bspline::bspline_basis(argvals, nknots, order);
    let nbasis = basis_vals.len() / m;
    // Form B^T B and its pseudoinverse (via nalgebra SVD)
    // Per curve: coefs = pinv(B) * y  ->  evaluate at query_points
    let basis_query = crate::basis::bspline::bspline_basis_from_knots(query_points, &knots, order);
    let m_q = query_points.len();
    let mut out = crate::matrix::FdMatrix::zeros(n, m_q);
    for i in 0..n {
        // solve for coefs_i, then sum coefs_i[k] * basis_query[j + k*m_q]
    }
    Ok(out)
}
```

---

### Pattern 2: Pointwise Functional Statistics (variance / std / covariance)

**What:** Single pass over columns of `FdMatrix` to compute per-evaluation-point statistics across `n` curves.

**Definitions (plain sample statistics, not integration-weighted):** The phase spec says "pointwise" which matches scikit-fda's `FDataGrid.var()` — a plain (unweighted) sample variance at each grid point, not a functional inner-product norm.

- `functional_variance(data)[j] = (1/n) * sum_i (data[(i,j)] - mean[j])^2` (or 1/(n-1) Bessel-corrected; spec is silent — use Bessel correction to match scikit-fda, which uses numpy's `ddof=1` by default) [ASSUMED — scikit-fda default ddof not confirmed by direct doc read this session]
- `functional_std(data)[j] = sqrt(functional_variance(data)[j])`
- `functional_covariance(data)[j1, j2] = (1/(n-1)) * sum_i (data[(i,j1)] - mean[j1]) * (data[(i,j2)] - mean[j2])` — M×M matrix stored as `FdMatrix(M, M)` in column-major order

**Why no integration weights here:** scikit-fda's `covariance()` returns the sample covariance *function* evaluated on a grid — it is not integrated; callers who want a weighted L2 inner product compose it separately. [ASSUMED based on scikit-fda API convention — not directly verified]

**Efficiency note:** `functional_covariance` is O(n·m²) — iterate over all `(j1, j2)` pairs using mean-centered data from `center_1d`. For large `m` this is expensive; the spec says "m×m sample covariance" which confirms the full matrix is required.

**Skeleton (variance):**

```rust
// Source: derived from mean_1d pattern (fdata.rs:166-178) [VERIFIED]
pub fn functional_variance(data: &FdMatrix) -> Result<Vec<f64>, FdarError> {
    let (n, m) = data.shape();
    if n < 2 { return Err(FdarError::InvalidDimension { ... }); }
    let means = mean_1d(data);
    let var: Vec<f64> = (0..m).map(|j| {
        let col = data.column(j);
        let mu = means[j];
        col.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / (n - 1) as f64
    }).collect();
    Ok(var)
}
```

---

### Pattern 3: Depth-Based Statistics

**What:** Compute per-curve FM depth scores → sort/rank → select or aggregate.

**Key function:** `fraiman_muniz_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64>` [VERIFIED: fdars-core/src/depth/fraiman_muniz.rs:32-39]

- Returns a `Vec<f64>` of length `data_obj.nrows()` with one depth per curve.
- When called with `data_obj == data_ori`, gives all curves' depths against the full sample — the standard self-depth call.

**`depth_based_median`:** Returns the index `i*` of the curve with the maximum FM depth. Not the curve itself — the phase spec says "index of deepest curve."

**`trim_mean`:** Depth-trimmed mean. Parameter `alpha ∈ [0, 1)` — exclude the `floor(alpha * n)` least-deep curves, average the remaining. Returns a `Vec<f64>` of length `m` (the mean curve values at each evaluation point).

**Import path:** `use crate::depth::fraiman_muniz_1d;` (re-exported at `depth/mod.rs:24`). [VERIFIED: fdars-core/src/depth/mod.rs:24]

**Skeleton (depth_based_median):**

```rust
// Source: derived from fraiman_muniz_1d usage (depth/fraiman_muniz.rs:32-39) [VERIFIED]
pub fn depth_based_median(data: &FdMatrix) -> Result<usize, FdarError> {
    let (n, _) = data.shape();
    if n == 0 { return Err(FdarError::InvalidDimension { ... }); }
    let depths = crate::depth::fraiman_muniz_1d(data, data, true);
    let idx = depths.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .ok_or_else(|| FdarError::ComputationFailed { ... })?;
    Ok(idx)
}
```

---

### Anti-Patterns to Avoid

- **Calling `svd_pseudoinverse` from outside `basis/`:** It is `pub(super)` — not visible from `helpers.rs`. Reproduce the same 3-line nalgebra SVD inline, or expose it. Prefer inline to avoid changing `basis/helpers.rs` visibility.
- **Using `pspline_fit_1d` for interpolation:** P-spline smoothing adds a ridge penalty — the system will not interpolate the original data exactly. Implement the un-penalized `B^T B` solve directly.
- **Allocating full row vectors per curve in covariance:** Use `data[(i, j)]` indexing or `data.column(j)` slices, not `data.row(i)` allocation in the inner j1/j2 loop.
- **Using integration weights in functional_variance/std:** The spec says "pointwise" — these are plain sample statistics, not L2-norm-weighted inner products. Using Simpson's weights would change the semantics.
- **Returning NaN instead of `FdarError` on bad input:** All public functions must validate inputs and return `FdarError`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| B-spline basis evaluation | Cox–de Boor from scratch | `bspline_basis` + `bspline_basis_from_knots` | Already correct, tested, handles boundary cases |
| Knot vector construction | Custom spacing | `construct_bspline_knots` | Extended-boundary convention already handled |
| Per-curve depth scores | Custom FM depth loop | `fraiman_muniz_1d` | Streaming implementation with parallel support |
| Pseudoinverse | Custom SVD | `nalgebra::SVD::new` with threshold | Already present via `basis/helpers.rs` pattern |
| FdMatrix construction | Raw Vec manipulation | `FdMatrix::zeros` + indexed assignment, or `FdMatrix::from_column_major` | Dimension-safe; returns `FdarError` on mismatch |
| Pointwise mean | Manual column loop | `mean_1d` | Already exists in `fdata.rs:166` |

---

## Common Pitfalls

### Pitfall 1: Query Points Outside `argvals` Domain

**What goes wrong:** `bspline_basis_from_knots` will evaluate at any point, but the knot vector's extended-boundary construction only guarantees partition-of-unity inside `[t_min, t_max]`. Evaluating outside produces mathematically valid but extrapolated (not interpolated) values that can grow unboundedly.

**Why it happens:** The phase spec says query points may be "off-grid" but does not explicitly restrict them to the original domain. The success criteria implicitly assume interpolation, not extrapolation.

**How to avoid:** Validate that all `query_points` lie within `[argvals[0], argvals[argvals.len()-1]]`. Return `FdarError::InvalidParameter` if any query point is outside. (Extrapolation support is REPR-03, a deferred backlog item.) [VERIFIED: BACKLOG.md ranks REPR-03 as deferred]

**Warning signs:** Test case with query point at exact min/max of argvals passes; query 0.001 outside fails silently.

### Pitfall 2: B-Spline System Rank Deficiency with Small `nknots`

**What goes wrong:** When `order > m/2` or `m` is very small, `nknots = m.saturating_sub(order).max(2)` gives `nknots = 2`, producing a severely underdetermined basis system. The pseudoinverse will return near-zero coefficients and the interpolation will be poor.

**Why it happens:** The `max(2)` floor in nknots computation is a safety guard, not a correctness guarantee. With `order = 4` and `m = 3`, `nknots = max(3-4, 2) = 2`, `nbasis = 6 > m = 3` — the system is overdetermined in the wrong direction.

**How to avoid:** Validate `order < m` and return `FdarError::InvalidParameter { parameter: "order", message: "must be less than number of evaluation points" }` if violated.

**Warning signs:** Interpolation test passes at argvals but diverges at query points.

### Pitfall 3: Column-Major vs Row-Major Confusion in Basis Matrix

**What goes wrong:** `bspline_basis` returns a flat vector in column-major order where the basis-function index is the "column" and evaluation-point index is the "row": element `basis[t + k * m]` is basis function `k` evaluated at `argvals[t]`. This is NOT the same layout as `FdMatrix` (where "columns" are evaluation points and "rows" are observations).

**Why it happens:** The basis matrix is `m × nbasis` (m evaluation points, nbasis functions) stored column-major. FdMatrix is `n × m` (n curves, m evaluation points). Confusing the two leads to transposed coefficient solves.

**How to avoid:** Read `pspline_fit_1d` carefully: `DMatrix::from_column_slice(m, actual_nbasis, &basis)` produces an `m × nbasis` matrix `B`. Then `B^T * B` is `nbasis × nbasis`, and `B^T * y` is `nbasis × 1`. [VERIFIED: fdars-core/src/basis/pspline.rs:86-87]

**Warning signs:** The "reproduces input at argvals within 1e-10" test fails.

### Pitfall 4: `n < 2` for Bessel-Corrected Variance

**What goes wrong:** `functional_variance` divides by `n-1`. With `n=1`, this panics (integer underflow if `n` is `usize`) or divides by zero.

**How to avoid:** Validate `n >= 2` at function entry; return `FdarError::InvalidDimension` for `n < 2`.

### Pitfall 5: `trim_mean` With `alpha = 0` or `alpha >= 1`

**What goes wrong:** `alpha = 0` excludes 0 curves (valid, equals regular mean). `alpha >= 1` would exclude all curves, leaving an empty average.

**How to avoid:** Validate `alpha ∈ [0, 1)` per the spec; return `FdarError::InvalidParameter` for `alpha >= 1.0` or `alpha < 0.0`.

---

## Code Examples

### B-Spline Basis Matrix Layout (Verified)

```rust
// Source: basis/bspline.rs:100-125 [VERIFIED]
// bspline_basis(t, nknots, order) returns Vec<f64> of length t.len() * nbasis
// Layout: basis[ti + k * n] = B_k(t[ti])   (column k = basis function k)
// That is: FdMatrix-style column-major WHERE column = basis function, row = eval point
let t: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
let basis = bspline_basis(&t, 5, 4);
// nbasis = nknots + order = 5 + 4 = 9; basis.len() == 20 * 9
```

### Evaluate at Different Grid Using Same Knots (Verified)

```rust
// Source: basis/bspline.rs:62-83 and pspline.rs:163-187 [VERIFIED]
// Step 1: build knot vector from original domain
let knots = construct_bspline_knots(t_min, t_max, nknots, order);
// Step 2: basis on argvals (for coefficient solve)
let basis_on_argvals = bspline_basis(argvals, nknots, order);
// Step 3: basis on query_points using SAME knots
let basis_on_query = bspline_basis_from_knots(query_points, &knots, order);
// basis_on_query[j + k * m_q] = B_k(query_points[j])
```

### Fraiman-Muniz Depth Self-Depth Call (Verified)

```rust
// Source: depth/fraiman_muniz.rs:32-39 [VERIFIED]
// Self-depth: each curve's depth against the full sample
let depths: Vec<f64> = fraiman_muniz_1d(&data, &data, true);
// depths[i] ∈ [0, 1] for curve i; larger = more central
```

### FdMatrix Column Access for Pointwise Pass (Verified)

```rust
// Source: matrix.rs:127-130 [VERIFIED]
// data.column(j) returns &[f64] of length n — contiguous, zero-copy
for j in 0..m {
    let col = data.column(j);  // all n curve values at evaluation point j
    let mean_j = col.iter().sum::<f64>() / n as f64;
}
```

### Error Validation Pattern (Verified)

```rust
// Source: error.rs:1-25 [VERIFIED] — exact variant syntax
if query_points.is_empty() {
    return Err(FdarError::InvalidDimension {
        parameter: "query_points",
        expected: ">= 1".to_string(),
        actual: "0".to_string(),
    });
}
if order == 0 || order >= m {
    return Err(FdarError::InvalidParameter {
        parameter: "order",
        message: format!("must be in [1, {}), got {}", m, order),
    });
}
```

### Test Pattern (Verified)

```rust
// Source: helpers.rs:590 pattern [VERIFIED]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    #[test]
    fn spline_interpolate_reproduces_argvals() {
        let t = uniform_grid(20);
        let vals: Vec<f64> = t.iter().map(|&x| x.powi(3)).collect();
        let data = crate::matrix::FdMatrix::from_column_major(vals, 1, 20).unwrap();
        let result = spline_interpolate(&data, &t, &t, 4).unwrap();
        for j in 0..20 {
            assert!((result[(0, j)] - data[(0, j)]).abs() < 1e-10,
                "at j={j}: got {}, expected {}", result[(0, j)], data[(0, j)]);
        }
    }
}
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Linear interpolation only (`InterpolationMethod::Linear`) | Add B-spline order-k (`spline_interpolate`) as a separate additive function | Off-grid accuracy improves from O(h²) to O(h^{2k}) for smooth curves |
| No functional summary statistics | Add `functional_variance`, `functional_std`, `functional_covariance`, `depth_based_median`, `trim_mean` | Closes EXPL-02 gap vs scikit-fda's `FDataGrid.var()`, `.std()`, `.cov()`, `depth_median()`, `trim_mean()` |

**Deprecated/outdated:** Nothing is deprecated. The existing `fdata_interpolate` + `InterpolationMethod` API (linear and cubic Hermite) remains fully available. `spline_interpolate` is additive.

---

## scikit-fda Baseline Reference

The following describes scikit-fda's semantics to ensure alignment:

- `SplineInterpolation(interpolation_order=k)`: Fits an order-k B-spline interpolant per curve. The `spline_interpolate` API matches this in spirit — same parameter name semantics (`order`). [ASSUMED — scikit-fda docs not fetched this session]
- `FDataGrid.var()`: Returns pointwise sample variance (plain, not L2-weighted). Matches the planned `functional_variance`. [ASSUMED]
- `FDataGrid.cov()`: Returns M×M sample covariance. Matches `functional_covariance`. [ASSUMED]
- `depth.depth_median()` (via `ModifiedBandDepth` or `Fraiman-Muniz`): Returns the deepest curve. The fdars implementation uses FM depth, which is specified in the success criteria. [ASSUMED — default depth choice]
- `trim_mean(data, proportiontocut=alpha)`: Trims `alpha` fraction of least-deep curves. Matches the planned `trim_mean(data, alpha)`. [ASSUMED]

---

## Package Legitimacy Audit

No new external packages are introduced in this phase. All implementation uses existing crate dependencies.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

---

## Validation Architecture

Framework: Rust built-in test harness (`#[test]`), criterion 0.5 for benchmarks (not needed here).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` |
| Config file | none (uses cargo test) |
| Quick run command | `cargo test -p fdars-core --features linalg` |
| Full suite command | `cargo test -p fdars-core --features linalg` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FEAT-01-a | `spline_interpolate` reproduces input at argvals within 1e-10 | unit | `cargo test -p fdars-core --features linalg spline_interpolate_reproduces_argvals` | ❌ Wave 0 |
| FEAT-01-b | Known cubic-spline values at off-grid points within 1e-10 | unit | `cargo test -p fdars-core --features linalg spline_interpolate_cubic_offgrid` | ❌ Wave 0 |
| FEAT-01-c | Error on out-of-range query points | unit | `cargo test -p fdars-core --features linalg spline_interpolate_rejects_out_of_range` | ❌ Wave 0 |
| FEAT-01-d | Error on order too large for grid | unit | `cargo test -p fdars-core --features linalg spline_interpolate_rejects_bad_order` | ❌ Wave 0 |
| FEAT-02-a | `functional_variance` = `functional_std`² pointwise | unit | `cargo test -p fdars-core --features linalg functional_variance_equals_std_squared` | ❌ Wave 0 |
| FEAT-02-b | `functional_covariance` diagonal = `functional_variance` | unit | `cargo test -p fdars-core --features linalg functional_covariance_diagonal_matches_variance` | ❌ Wave 0 |
| FEAT-02-c | `depth_based_median` returns argmax-depth curve index | unit | `cargo test -p fdars-core --features linalg depth_based_median_argmax` | ❌ Wave 0 |
| FEAT-02-d | `trim_mean` with alpha=0 equals `mean_1d` | unit | `cargo test -p fdars-core --features linalg trim_mean_alpha_zero_equals_mean` | ❌ Wave 0 |
| FEAT-02-e | Error on alpha outside [0,1) and n<2 for variance | unit | `cargo test -p fdars-core --features linalg functional_stats_input_validation` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg`
- **Per wave merge:** `cargo test -p fdars-core --features linalg && cargo clippy -p fdars-core --features linalg`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

All test functions listed above are new — they do not exist yet. The test infrastructure (cargo test harness, `test_helpers::uniform_grid`) is already present. Wave 0 of the plan must create the test stubs before or alongside the implementations.

---

## Security Domain

Phase adds pure-Rust numeric computation functions with no I/O, no auth, no sessions, no network calls, and no user-supplied strings processed as code.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (limited) | `FdarError::InvalidParameter` / `InvalidDimension` on all inputs |
| V6 Cryptography | no | — |

**Threat pattern:** Integer overflow in dimension arithmetic (`n * m` in covariance matrix allocation). Mitigation: check `m.checked_mul(m)` before allocating the M×M covariance matrix; return `FdarError::InvalidParameter` if `m` is implausibly large.

---

## Environment Availability

All dependencies are compile-time Rust crates; no external services or CLI tools are needed beyond the already-available Rust toolchain.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | Building fdars-core | ✓ | 1.97.0 (CLAUDE.md) | — |
| cargo test | Running tests | ✓ | bundled with Rust 1.97.0 | — |
| nalgebra | Coefficient solve in spline_interpolate | ✓ | 0.33 (Cargo.toml) | — |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `functional_variance` uses Bessel correction (ddof=1) to match scikit-fda | Architecture Patterns – Pattern 2 | Tests comparing against hand-computed references would pass either way (specify ddof in test); but cross-validation with scikit-fda would fail if convention differs |
| A2 | `functional_covariance` is plain sample covariance (not integration-weighted) | Architecture Patterns – Pattern 2 | If scikit-fda uses L2-weighted covariance, the `covariance diagonal = variance` identity would still hold (both change together), but values would differ numerically |
| A3 | `depth_based_median` uses Fraiman-Muniz depth (not Modified Band Depth) as default | Architecture Patterns – Pattern 3 | Phase spec says "depth-based median" without specifying depth function; FM is standard and already well-tested; if MBD is intended, signature stays the same but inner call changes |
| A4 | scikit-fda `trim_mean` trims the `floor(alpha * n)` least-deep curves | Architecture Patterns – Pattern 3 | If scikit-fda uses ceiling or rounds differently, numerical tests against hand-computed references would still pass but cross-validation values would shift |

---

## Open Questions

1. **ddof for functional_variance/covariance**
   - What we know: phase spec says "sample covariance" which conventionally means Bessel-corrected (n-1)
   - What's unclear: scikit-fda's exact default (not verified this session)
   - Recommendation: implement with `n-1` (Bessel correction) and document in rustdoc; the hand-computed test in the spec will be explicit about which formula it uses

2. **Placement of summary statistics: `fdata.rs` vs new `summary.rs`**
   - What we know: BACKLOG EXPL-02 area note says "`fdata.rs`, `covariance.rs`"; `mean_1d`/`center_1d` are in `fdata.rs`
   - What's unclear: whether the planner wants a new file for cleanliness
   - Recommendation: add to `fdata.rs` — the module already owns functional data operations and keeping related functions co-located avoids adding a new public module; if the file grows too large this is a refactor concern for a later milestone

3. **Re-export locations in `lib.rs`**
   - What we know: `lib.rs` re-exports `fdata_interpolate`, `linear_interp`, and `InterpolationMethod` from `helpers.rs` at lines 171-174
   - What's unclear: whether `spline_interpolate` should be re-exported at crate root or only via `fdars_core::helpers::spline_interpolate`
   - Recommendation: follow the existing `fdata_interpolate` re-export pattern — add `spline_interpolate` to the same re-export block in `lib.rs`

---

## Sources

### Primary (HIGH confidence)

- `fdars-core/src/basis/bspline.rs:1-125` — exact function signatures and layout for `construct_bspline_knots`, `bspline_basis`, `bspline_basis_from_knots` [VERIFIED this session]
- `fdars-core/src/basis/pspline.rs:66-187` — reference fit-then-evaluate pattern via `pspline_fit_1d` / `pspline_evaluate` [VERIFIED this session]
- `fdars-core/src/depth/fraiman_muniz.rs:1-46` — `fraiman_muniz_1d` signature and return type [VERIFIED this session]
- `fdars-core/src/depth/mod.rs:1-34` — re-export of `fraiman_muniz_1d` at module level [VERIFIED this session]
- `fdars-core/src/fdata.rs:166-236` — `mean_1d`, `center_1d` patterns [VERIFIED this session]
- `fdars-core/src/matrix.rs:1-213` — `FdMatrix` API: `zeros`, `from_column_major`, `column`, `shape`, `nrows`, `ncols`, indexed access `[(i,j)]` [VERIFIED this session]
- `fdars-core/src/error.rs:1-51` — `FdarError` variants verbatim [VERIFIED this session]
- `fdars-core/src/helpers.rs:1-844` — existing `linear_interp`, `fdata_interpolate`, `InterpolationMethod`; inline test pattern [VERIFIED this session]
- `fdars-core/src/test_helpers.rs:1-8` — `uniform_grid(n)` signature [VERIFIED this session]
- `.planning/research/BACKLOG.md` — EXPL-02 location note, REPR-03 deferred status [VERIFIED this session]

### Secondary (MEDIUM confidence)

- `.planning/REQUIREMENTS.md` — FEAT-01/FEAT-02 specs verbatim [VERIFIED this session]
- `.planning/STATE.md` — implementation conventions and constraints [VERIFIED this session]
- `.claude/CLAUDE.md` (project) — architectural constraints, naming conventions [VERIFIED via system-reminder]

### Tertiary (LOW confidence / assumed)

- scikit-fda covariance/variance semantics [ASSUMED — not fetched this session]

---

## Metadata

**Confidence breakdown:**
- B-spline API: HIGH — read every relevant file directly
- Depth API: HIGH — read `fraiman_muniz.rs` and `depth/mod.rs` directly
- FdMatrix API: HIGH — read `matrix.rs` directly
- Error handling: HIGH — read `error.rs` directly
- Functional statistics math (ddof choice, scikit-fda alignment): LOW — assumed from convention

**Research date:** 2026-08-10
**Valid until:** Stable (library internals do not change; valid until next codebase refactor affecting `basis/` or `depth/`)
