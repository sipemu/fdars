# Phase 29: Outlier-Detector Suite - Pattern Map

**Mapped:** 2026-08-19
**Files analyzed:** 2 (outliers.rs additive blocks + lib.rs re-export extension)
**Analogs found:** all symbols covered by existing codebase patterns

---

## File Classification

| New/Modified Symbol | Role | Data Flow | Closest Analog | Match Quality |
|---------------------|------|-----------|----------------|---------------|
| `TvdMssOutliers` struct | result struct | — | `OutligramResult` (`outliers.rs:248-264`) | exact |
| `MuodResult` struct | result struct | — | `MagnitudeShapeResult` (`outliers.rs:334-342`) | exact |
| `SeqTransformOutliers` struct | result struct | — | `OutligramResult` (`outliers.rs:248-264`) | exact |
| `DepthgramResult` struct | result struct | — | `OutligramResult` (`outliers.rs:248-264`) | exact |
| `tvdmss` fn | outlier detector | FdMatrix → indices/scores | `outliergram` (`outliers.rs:278-332`) | exact |
| `muod` fn | outlier detector | FdMatrix → indices/scores | `magnitude_shape_outlyingness` (`outliers.rs:352+`) | exact |
| `sequential_transform_outliers` fn | outlier detector | FdMatrix → indices | `outliergram` (`outliers.rs:278-332`) | role-match |
| `depthgram` fn | outlier detector | FdMatrix → indices/scores | `outliergram` (`outliers.rs:278-332`) | exact |
| `iqr_fence` private helper | numeric helper | `&[f64]` → `(f64, f64)` | inline IQR block (`outliers.rs:314-319`) + `quantile_sorted` (`helpers.rs:283-298`) | partial |
| `pub use outliers::{…}` extension | re-export | — | `pub use outliers::{…}` (`lib.rs:434-437`) | exact |

---

## Pattern Assignments

### Result Structs: `TvdMssOutliers`, `MuodResult`, `SeqTransformOutliers`, `DepthgramResult`

**Analog:** `src/outliers.rs` lines 248-264 (`OutligramResult`) and lines 334-342 (`MagnitudeShapeResult`)

**Derive + attribute pattern** (lines 249-251, replicated at 335-336):
```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OutligramResult {
    pub mei: Vec<f64>,
    pub mbd: Vec<f64>,
    pub a0: f64,
    pub a1: f64,
    pub a2: f64,
    pub threshold: f64,
    pub outlier_flags: Vec<bool>,
}
```

Apply verbatim to all four new structs — `#[derive(Debug, Clone, PartialEq)]`, then `#[non_exhaustive]`, then conditional serde, in that exact order. All public fields. `Vec<usize>` for index sets, `Vec<f64>` for score vectors, matching the locked field names in CONTEXT.md.

---

### `tvdmss` — Two-Stage TVD+MSS Outlier Detector

**Analog:** `src/outliers.rs` lines 278-332 (`outliergram`)

**Entry validation pattern** (lines 279-286):
```rust
pub fn outliergram(data: &FdMatrix, factor: f64) -> Result<OutligramResult, FdarError> {
    let n = data.nrows();
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
```

Copy this exact guard shape for `tvdmss` — guard `n < 3` (inherited from `total_variation_depth_1d`), return `FdarError::InvalidDimension` with `parameter: "data"`, string-formatted `expected`/`actual`.

**Depth call pattern** (lines 288-289 — analog for consuming an existing depth fn):
```rust
let mei = modified_epigraph_index_1d(data, data);
let mbd = modified_band_1d(data, data);
```

For `tvdmss`, replace with:
```rust
use crate::depth::tvd::total_variation_depth_1d;
let depth = total_variation_depth_1d(data, data)?;  // TvdMssResult { tvd, mss }
```

Note the `?` propagation — `total_variation_depth_1d` returns `Result`, unlike the bare `Vec<f64>` returns of `modified_band_1d` / `modified_epigraph_index_1d`.

**IQR fence inline pattern** (lines 313-319 — the existing inline block to extract into a helper):
```rust
let mut sorted_resid = residuals.clone();
sorted_resid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
let q1 = sorted_resid[n / 4];
let q3 = sorted_resid[3 * n / 4];
let iqr = q3 - q1;
let threshold = q1 - factor * iqr;
```

The new `iqr_fence` private helper **replaces** this inline block everywhere in the new code, using `quantile_sorted` instead of floor-index quartiles (see Shared Patterns below).

**functional_boxplot call pattern** (`src/depth/dispatch.rs` lines 243-247):
```rust
pub fn functional_boxplot(
    data: &FdMatrix,
    method: DepthMethod,
    factor: f64,
) -> Result<FunctionalBoxplotResult, FdarError>
```

Call signature for Stage 2 of `tvdmss`: `functional_boxplot(&reduced_data, DepthMethod::ModifiedBand, config.emp_factor_tvd)?`. Result field `.outliers: Vec<usize>` gives the magnitude outlier indices into the reduced dataset — re-map to original indices via the `keep` index vec.

**Result construction pattern** (lines 323-331):
```rust
Ok(OutligramResult {
    mei,
    mbd,
    a0,
    a1,
    a2,
    threshold,
    outlier_flags,
})
```

Copy this `Ok(StructName { field, field, … })` tail for each new detector.

---

### `muod` — Massive Unsupervised Outlier Detection

**Analog:** `src/outliers.rs` lines 352-410 (`magnitude_shape_outlyingness`)

**Entry validation pattern** (lines 353-360):
```rust
pub fn magnitude_shape_outlyingness(data: &FdMatrix) -> Result<MagnitudeShapeResult, FdarError> {
    let (n, m) = data.shape();
    if n < 2 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows and 1 column".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }
```

For `muod`, guard `n < 3` (R baseline requires n ≥ 3) and `m < 2` (need at least 2 points for covariance), using same `FdarError::InvalidDimension` shape.

**iter_maybe_parallel! over curves** (outliers.rs lines 32-50, helper `compute_trimmed_stats`):
```rust
let results: Vec<(f64, f64)> = iter_maybe_parallel!(0..m)
    .map(|j| {
        // per-column computation
        (mean_j, var_j)
    })
    .collect();
```

Apply to muod's per-curve index computation: `iter_maybe_parallel!(0..n).map(|i| { … }).collect()`. The parallel macro requires the closure to be `Send` — all captured data (`data`, `mu` slice) is `&[f64]` / `&FdMatrix`, which are `Sync`; no `RefCell` or interior mutability.

**Column-mean computation pattern** (lines 369-375 in `magnitude_shape_outlyingness`):
```rust
let mut col_means = vec![0.0; m];
for j in 0..m {
    for i in 0..n {
        col_means[j] += data[(i, j)];
    }
    col_means[j] /= n as f64;
}
```

Use `data[(i, j)]` operator throughout — never raw pointer arithmetic. For `muod`'s pointwise mean `mu_t`, this is the exact pattern: iterate `j` (evaluation points) in outer loop, `i` (curves) in inner loop.

**Degenerate std guard pattern** (lines 384-386):
```rust
let norm = norm_sq.sqrt().max(1e-15);
```

For muod, analogous guard on `xi_std` and `mu_var`: `if xi_std < 1e-15 { shape_index = 0.0 }` — a constant curve has perfect shape relative to any reference.

---

### `sequential_transform_outliers` — Sequential-Transformation Detection

**Analog:** `src/outliers.rs` lines 278-332 (`outliergram`) for function skeleton; `src/depth/dispatch.rs` lines 243-341 (`functional_boxplot`) as the base detector called per-transform step.

**Entry validation pattern** — same `FdarError::InvalidDimension` with `"data"` parameter. Guard `n < 2` (functional_boxplot requires n ≥ 2); also guard `m < 2` before any D1 differencing step.

**functional_boxplot call** (`dispatch.rs:243`):
```rust
use crate::depth::dispatch::{functional_boxplot, DepthMethod};
// per transform step:
let fbp = functional_boxplot(&current_data, config.depth_method, config.emp_factor)?;
// fbp.outliers: Vec<usize> — indices of outlier curves in current_data
```

**Column-major FdMatrix construction for transformed data** — use `FdMatrix::from_column_major(data_vec, n, m_new)` where `m_new = m - 1` after D1. If `from_column_major` returns `Err`, propagate as `FdarError::InvalidDimension`.

**T2 zero-norm guard** — same pattern as `let norm = norm_sq.sqrt().max(1e-15)` but strict: return `FdarError::ComputationFailed` if any curve L2 norm is below `1e-15` before division.

**Union construction** — collect all per-transform outlier sets, flatten, sort, dedup:
```rust
let mut union_outliers: Vec<usize> = per_transform_outliers
    .iter()
    .flat_map(|(_, v)| v.iter().copied())
    .collect();
union_outliers.sort_unstable();
union_outliers.dedup();
```

---

### `depthgram` — Depthgram Index Statistic

**Analog:** `src/outliers.rs` lines 278-332 (`outliergram`) — the parabola+IQR outlier detection formula is the exact same outliergram parabola logic, reused here.

**Depth calls** (`band.rs:43-50` and `band.rs:57-89`):
```rust
use crate::depth::band::{modified_band_1d, modified_epigraph_index_1d};
// These return Vec<f64>, not Result — no ? needed:
let mbd = modified_band_1d(data, data);   // length n, O(n²·m)
let mei = modified_epigraph_index_1d(data, data); // length n, O(n·m)
```

**FdMatrix column-vector wrapping for second-level depths** — wrap the `mbd` / `mei` `Vec<f64>` as `n×1` FdMatrix:
```rust
let mei_mat = FdMatrix::from_column_major(mei.clone(), n, 1)?;
let mbd_mat = FdMatrix::from_column_major(mbd.clone(), n, 1)?;
let mbd_mei = modified_band_1d(&mei_mat, &mei_mat);   // MBD of MEI
let mei_mbd = modified_epigraph_index_1d(&mbd_mat, &mbd_mat); // MEI of MBD
```

**Outliergram parabola + IQR fence pattern** (outliers.rs lines 291-321 — the exact formula to reuse):
```rust
// Parabola coefficients (from roahd formula, matches outliergram):
let a2 = -2.0 / (n as f64 * (n as f64 - 1.0));
let a0 = a2;
let a1 = 2.0 * (n as f64 + 1.0) / (n as f64 - 1.0);
// Distance from parabola (positive = above, negative = below → shape outlier):
let dist: Vec<f64> = (0..n)
    .map(|i| (a0 + a1 * mei[i] + a2 * (n as f64).powi(2) * mei[i].powi(2)) - mbd[i])
    .collect();
// Upper IQR fence on dist (dist > upper → shape outlier):
let (_, upper_dist) = iqr_fence(&dist, config.outliergram_factor);
let shape_outliers: Vec<usize> = (0..n).filter(|&i| dist[i] > upper_dist).collect();
```

Note: `outliergram` uses a **lower** fence on residuals (`r < threshold`); `depthgram` uses an **upper** fence on `dist` (`dist[i] > upper_dist`) per R source. The sign convention differs — the parabola formula is the same, the fence direction is opposite.

**functional_boxplot for magnitude outliers** — same call as `tvdmss` Stage 2, but on the `mbd` column vector:
```rust
let mbd_mat2 = FdMatrix::from_column_major(mbd.clone(), n, 1)?;
let fbp = functional_boxplot(&mbd_mat2, DepthMethod::ModifiedBand, config.boxplot_factor)?;
let magnitude_outliers = fbp.outliers;
```

**For p=1: all three representations are identical** — assign `mbd_mei_d`, `mbd_mei_t`, `mbd_mei_t2` all to `mbd_mei.clone()` (and similarly for `mei_mbd_*`). Document in rustdoc: "For p=1, all three depthgram representations are equivalent; this implementation handles univariate functional data only."

---

### `iqr_fence` — Private IQR Helper

**Analog:** `src/outliers.rs` lines 313-319 (inline IQR block — to be superseded by the helper) + `src/helpers.rs` lines 280-298 (`quantile_sorted` + `sort_nan_safe`)

**sort_nan_safe** (`helpers.rs:10-12`):
```rust
pub fn sort_nan_safe(slice: &mut [f64]) {
    slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}
```

**quantile_sorted** (`helpers.rs:283-298`):
```rust
pub fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    if sorted.len() == 1 || p <= 0.0 { return sorted[0]; }
    if p >= 1.0 { return sorted[sorted.len() - 1]; }
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}
```

**New private helper** (no existing single-function equivalent; compose from above):
```rust
/// Classical boxplot IQR fence.
///
/// Returns `(lower_fence, upper_fence)` = `(Q1 − factor × IQR, Q3 + factor × IQR)`.
/// Uses linear-interpolation quantiles (same as `quantile_sorted`) rather than
/// R's floor-index quartiles; produces the same result for `n > 20`.
fn iqr_fence(values: &[f64], factor: f64) -> (f64, f64) {
    let mut sorted = values.to_vec();
    crate::helpers::sort_nan_safe(&mut sorted);
    let q1 = crate::helpers::quantile_sorted(&sorted, 0.25);
    let q3 = crate::helpers::quantile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;
    (q1 - factor * iqr, q3 + factor * iqr)
}
```

Place this private fn **before** the first public detector that uses it in `outliers.rs`.

---

### `pub use outliers::{…}` — Re-Export Extension

**Analog:** `src/lib.rs` lines 434-437

**Exact existing block to extend:**
```rust
// Re-export outlier detection functions
pub use outliers::{
    detect_outliers_lrt, magnitude_shape_outlyingness, outliergram, outliers_threshold_lrt,
    outliers_threshold_lrt_with_dist, MagnitudeShapeResult, OutligramResult,
};
```

**Extended block** — add all new public symbols in alphabetical order within the list, following the existing comma-separated style:
```rust
pub use outliers::{
    depthgram, detect_outliers_lrt, magnitude_shape_outlyingness, muod, outliergram,
    outliers_threshold_lrt, outliers_threshold_lrt_with_dist, sequential_transform_outliers,
    tvdmss, DepthgramResult, MagnitudeShapeResult, MuodResult, OutligramResult,
    SeqTransformOutliers, TvdMssOutliers,
};
```

---

## Shared Patterns

### Error Handling
**Source:** `src/outliers.rs` lines 279-286 (`outliergram`), lines 353-360 (`magnitude_shape_outlyingness`)
**Apply to:** All four new detector functions
```rust
return Err(FdarError::InvalidDimension {
    parameter: "data",
    expected: "at least N rows".to_string(),
    actual: format!("{n} rows"),
});
```
For T2 zero-norm and D1/m-too-small cases, use `FdarError::ComputationFailed`:
```rust
return Err(FdarError::ComputationFailed {
    operation: "T2 normalization",
    detail: "curve at index {i} has zero L2 norm".to_string(),
});
```

### iter_maybe_parallel! Import
**Source:** `src/outliers.rs` line 8
```rust
use crate::iter_maybe_parallel;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
```
Both lines required together. The `#[cfg(feature = "parallel")]` import is needed for `.collect()` to resolve on parallel iterators.

### Column-Major Access
**Source:** All modules — `data[(i, j)]` throughout. Never `data.data()[i + j * n]`.

### #[must_use] on Expensive Fns
**Source:** `src/depth/band.rs` lines 29, 42, 56; 74+ instances codebase-wide.
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn tvdmss(...) -> Result<TvdMssOutliers, FdarError> {
```
Apply to all four new public detector functions.

### Inline Tests
**Source:** `src/outliers.rs` (existing `#[cfg(test)] mod tests { … }` block at end of file)
**Pattern:** Append to the existing `tests` module in `outliers.rs` — do **not** create a new `mod tests` block. Use `uniform_grid` from `src/test_helpers.rs` if argvals are needed. Build synthetic FdMatrix via `FdMatrix::from_column_major(data_vec, n, m)`.

---

## No Analog Found

No new symbols are without analog. All patterns are covered by existing code in `outliers.rs`, `depth/band.rs`, `depth/dispatch.rs`, `depth/tvd.rs`, and `helpers.rs`.

---

## Metadata

**Analog search scope:** `fdars-core/src/outliers.rs`, `fdars-core/src/depth/dispatch.rs`, `fdars-core/src/depth/band.rs`, `fdars-core/src/depth/tvd.rs`, `fdars-core/src/helpers.rs`, `fdars-core/src/lib.rs`
**Files scanned:** 6
**Pattern extraction date:** 2026-08-19
