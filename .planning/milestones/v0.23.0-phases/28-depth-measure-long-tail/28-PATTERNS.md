# Phase 28: Depth-Measure Long Tail - Pattern Map

**Mapped:** 2026-08-19
**Files analyzed:** 9 (6 new + 3 modified)
**Analogs found:** 9 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/depth/half_region.rs` | depth-measure module | CRUD (FdMatrix → Result<Vec<f64>>) | `src/depth/band.rs` (MEI) | exact |
| `src/depth/hypo_epi.rs` | depth-measure module | CRUD (FdMatrix → Result<Vec<f64>>) | `src/depth/band.rs` (MEI) | exact |
| `src/depth/extremal.rs` | depth-measure module | transform (rank + sort) | `src/depth/dispatch.rs` sort pattern | role-match |
| `src/depth/erl.rs` | depth-measure module | CRUD O(n²·m) | `src/depth/band.rs` MEI + `mod.rs` random_depth_core | role-match |
| `src/depth/linf.rs` | depth-measure module | CRUD O(n²·m) | `src/depth/band.rs` (MEI iter pattern) | exact |
| `src/depth/tvd.rs` | depth-measure module + result struct | transform (rank → TVD/MSS) | `src/depth/dispatch.rs` (FunctionalBoxplotResult struct) | role-match |
| `src/depth/mod.rs` | module barrel | — | itself (additive edit) | exact |
| `src/depth/dispatch.rs` | dispatcher enum + match | request-response | itself (additive edit) | exact |
| `src/lib.rs` | crate root re-export | — | itself (additive edit) | exact |

---

## Pattern Assignments

### `src/depth/half_region.rs` (depth-measure module, CRUD)

**Analog:** `src/depth/band.rs` — `modified_epigraph_index_1d` (lines 52–89)

**Imports pattern** (band.rs lines 1–9):
```rust
//! Half-region depth and modified half-region depth (HRD, MHRD).

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::error::FdarError;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
```
(Note: `band.rs` imports `iter_maybe_parallel` and conditionally `rayon::iter::ParallelIterator`; new files do the same.)

**Function signature pattern** (band.rs line 57):
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn modified_epigraph_index_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
```
New functions use `Result<Vec<f64>, FdarError>` return (per CONTEXT.md decision) and same two-matrix signature:
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn half_region_depth_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
) -> Result<Vec<f64>, FdarError> {
```

**Validation pattern** (band.rs lines 62–64, returns empty Vec; new functions return Err):
```rust
// Existing (returns empty Vec):
if nobj == 0 || nori == 0 || n_points == 0 {
    return Vec::new();
}

// New pattern (returns Err):
let (nobj, nori, m) = (data_obj.nrows(), data_ori.nrows(), data_obj.ncols());
if nobj == 0 || nori == 0 || m == 0 {
    return Err(FdarError::InvalidDimension {
        parameter: "data_obj",
        expected: "non-empty matrix".to_string(),
        actual: format!("{}x{}", nobj, m),
    });
}
if nobj < 2 {
    return Err(FdarError::InvalidDimension {
        parameter: "data_obj",
        expected: "at least 2 curves for half-region depth".to_string(),
        actual: format!("{nobj}"),
    });
}
```

**Core O(n²·m) loop pattern** (band.rs lines 66–88 — MEI, the direct template):
```rust
iter_maybe_parallel!(0..nobj)
    .map(|i| {
        let mut total = 0.0;
        for j in 0..nori {
            let mut count = 0.0;
            for t in 0..n_points {
                let xi = data_obj[(i, t)];
                let xj = data_ori[(j, t)];
                if xi <= xj {   // MEI epigraph condition
                    count += 1.0;
                }
            }
            total += count / n_points as f64;
        }
        total / nori as f64
    })
    .collect()
```
HRD/MHRD implementation computes EI and HI (or MEI and MHI) in a single fused pass to avoid nested parallelism:
```rust
// Fused single pass for HRD = min(EI, HI):
iter_maybe_parallel!(0..nobj)
    .map(|i| {
        let mut ei_count = 0.0_f64;  // global indicator: X_j >= X_i for all t
        let mut hi_count = 0.0_f64;  // global indicator: X_j <= X_i for all t
        'outer: for j in 0..nori {
            let mut j_above = true;
            let mut j_below = true;
            for t in 0..m {
                let xi = data_obj[(i, t)];
                let xj = data_ori[(j, t)];
                if xj > xi { j_below = false; }
                if xj < xi { j_above = false; }
                if !j_above && !j_below { continue 'outer; }
            }
            if j_above { ei_count += 1.0; }
            if j_below { hi_count += 1.0; }
        }
        f64::min(ei_count, hi_count) / nori as f64
    })
    .collect::<Vec<f64>>()
```

**Test pattern** (dispatch.rs lines 240–400 — inline `#[cfg(test)]`):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;

    fn sample(n: usize, m: usize) -> FdMatrix {
        let col_major: Vec<f64> = (0..n * m).map(|k| {
            let i = k % n;
            let t = k / n;
            (t as f64 / (m as f64 - 1.0) * std::f64::consts::PI).sin() + 0.05 * i as f64
        }).collect();
        FdMatrix::from_column_major(col_major, n, m).unwrap()
    }

    #[test]
    fn central_curve_deepest() { ... }

    #[test]
    fn empty_matrix_returns_err() {
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(half_region_depth_1d(&empty, &empty).is_err());
    }
}
```

---

### `src/depth/hypo_epi.rs` (depth-measure module, CRUD)

**Analog:** `src/depth/band.rs` — `modified_epigraph_index_1d` (lines 52–89)

**Imports pattern**: identical to `half_region.rs` above.

**HI — global indicator** (condition flipped vs MEI — `continue 'outer` on first violation):
```rust
// HI: count j where data_ori[(j,t)] <= data_obj[(i,t)] for ALL t
iter_maybe_parallel!(0..nobj)
    .map(|i| {
        let mut count = 0.0_f64;
        'outer: for j in 0..nori {
            for t in 0..m {
                if data_ori[(j, t)] > data_obj[(i, t)] {
                    continue 'outer;  // X_j crosses above X_i — not in hypograph
                }
            }
            count += 1.0;
        }
        count / nori as f64
    })
    .collect::<Vec<f64>>()
```
HI values are always multiples of `1/nori` (integers / nori) — verified in tests.

**EI — global indicator** (complement of HI):
```rust
// EI: count j where data_ori[(j,t)] >= data_obj[(i,t)] for ALL t
// Inner condition: if data_ori[(j,t)] < data_obj[(i,t)] { continue 'outer; }
```

**MHI — pointwise average** (mirrors MEI exactly, condition `>=` instead of `<=`):
```rust
// MHI: fraction of t where X_i(t) >= X_j(t) — matches band.rs MEI with >= instead of <=
iter_maybe_parallel!(0..nobj)
    .map(|i| {
        let mut total = 0.0;
        for j in 0..nori {
            let mut count = 0.0;
            for t in 0..m {
                if data_obj[(i, t)] >= data_ori[(j, t)] {  // MHI condition
                    count += 1.0;
                }
            }
            total += count / m as f64;
        }
        total / nori as f64
    })
    .collect::<Vec<f64>>()
```

**Validation pattern**: same `FdarError::InvalidDimension` as `half_region.rs`; HI/EI add n≥2 guard; MHI allows n≥1.

---

### `src/depth/extremal.rs` (depth-measure module, transform)

**Analog:** `src/depth/dispatch.rs` — sort with tie-breaker (lines 181–186) + `band.rs` struct pattern

**Sort with tie-breaker pattern** (dispatch.rs lines 181–186):
```rust
order.sort_by(|&a, &b| {
    depths[b]
        .partial_cmp(&depths[a])
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.cmp(&b))
});
```
Extremal depth uses the same `partial_cmp(...).unwrap_or(Equal).then(idx_cmp)` pattern for stable tie-breaking:
```rust
// Sort by (d_level asc, mass desc), then original index for stability
indices.sort_by(|&a, &b| {
    d_level[a].partial_cmp(&d_level[b])
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(mass[b].partial_cmp(&mass[a]).unwrap_or(std::cmp::Ordering::Equal))
        .then(a.cmp(&b))
});
```

**Sequential O(n·m) pattern** (no `iter_maybe_parallel!` — extremal depth is O(n·m + n log n)):
```rust
pub fn extremal_depth_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
) -> Result<Vec<f64>, FdarError> {
    let (n, m) = (data_ori.nrows(), data_ori.ncols());
    // Step 1: pointwise depth D[i,t] = 1 - |2*rank(col_t)[i] - n - 1| / n as f64
    // Step 2: d_level[i] = D[i,:].min(); mass[i] = #{t: D[i,t]==d_level[i]} / m
    // Step 3: sort indices by (d_level asc, mass desc, idx asc)
    // Step 4: depth[original_idx] = (position + 1) as f64 / n as f64
    Ok(depths)
}
```
Uses inline `column_ranks` helper (from RESEARCH.md Pattern 6) — defined as `fn column_ranks(data: &FdMatrix, col: usize) -> Vec<f64>` in the same file.

**n≥3 guard** (mirrors dispatch.rs Band guard at lines 69–76):
```rust
if n < 3 {
    return Err(FdarError::InvalidDimension {
        parameter: "data_ori",
        expected: "at least 3 curves for extremal depth".to_string(),
        actual: format!("{n}"),
    });
}
```

---

### `src/depth/erl.rs` (depth-measure module, CRUD O(n²·m))

**Analog:** `src/depth/band.rs` MEI loop + `src/depth/mod.rs` `random_depth_core` for `iter_maybe_parallel!` over outer i

**O(n²) parallel outer loop** (same structure as MEI but over pairwise lex comparison):
```rust
// Step 4: for each i, count j where sorted_R[j] lexicographically < sorted_R[i]
// i.e., curve j is more extreme than curve i
let depths: Vec<f64> = iter_maybe_parallel!(0..n)
    .map(|i| {
        let mut not_more_extreme = 0usize;  // count of j where i is NOT more extreme than j
        for j in 0..n {
            // sorted_R[i] < sorted_R[j] lex means i is more extreme
            let i_more_extreme = sorted_r[i].iter().zip(sorted_r[j].iter())
                .find(|(&ri, &rj)| (ri - rj).abs() > f64::EPSILON)
                .map(|(&ri, &rj)| ri < rj)
                .unwrap_or(false);
            if !i_more_extreme {
                not_more_extreme += 1;
            }
        }
        not_more_extreme as f64 / n as f64
    })
    .collect();
```

**`column_ranks` helper** shared with `extremal.rs` — define in each file (or factor to `mod.rs` as `pub(super)`). Prefer per-file to avoid coupling.

---

### `src/depth/linf.rs` (depth-measure module, CRUD O(n²·m))

**Analog:** `src/depth/band.rs` MEI — identical loop structure, different inner computation

**Core pattern** (pairwise L∞ distance, then invert):
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn linfinity_depth_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
) -> Result<Vec<f64>, FdarError> {
    let (nobj, nori, m) = (data_obj.nrows(), data_ori.nrows(), data_obj.ncols());
    // validation ...
    let depths: Vec<f64> = iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut mean_dist = 0.0_f64;
            for j in 0..nori {
                // L∞ = max_t |X_i(t) - X_j(t)|
                let mut max_diff = 0.0_f64;
                for t in 0..m {
                    let diff = (data_obj[(i, t)] - data_ori[(j, t)]).abs();
                    if diff > max_diff { max_diff = diff; }
                }
                mean_dist += max_diff;
            }
            mean_dist /= nori as f64;
            1.0 / (1.0 + mean_dist)
        })
        .collect();
    Ok(depths)
}
```

**n≥1 guard** (meaningful from n=1; depth=1.0 when nori=1 and i==0):
```rust
if nobj == 0 || nori == 0 || m == 0 {
    return Err(FdarError::InvalidDimension { ... });
}
```
No n≥2 guard — L∞ is well-defined for n=1 (depth = 1.0).

---

### `src/depth/tvd.rs` (depth-measure module + result struct)

**Analog for struct:** `src/depth/dispatch.rs` — `FunctionalBoxplotResult` (lines 107–124)

**Result struct pattern** (dispatch.rs lines 107–124):
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FunctionalBoxplotResult {
    pub median: Vec<f64>,
    // ...
}
```
Apply to `TvdMssResult`:
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TvdMssResult {
    /// Total variation depth (magnitude component). Higher = more central. Range (0, 0.25].
    pub tvd: Vec<f64>,
    /// Modified shape similarity index (shape component). Higher = more shape-central.
    pub mss: Vec<f64>,
}
```

**TVD rank pass** (reuses `column_ranks` helper, same MEI loop structure for parallel rank accumulation):
```rust
pub fn total_variation_depth_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
) -> Result<TvdMssResult, FdarError> {
    let (n, m) = (data_ori.nrows(), data_ori.ncols());
    // Step 1: for each column t, compute normalized ranks p[i,t] = rank(col_t)[i] / n
    // Step 2: TVD[i] = (1/m) * sum_t p[i,t] * (1 - p[i,t])
    // Step 3: MSS: compute first differences, rank per interval, same p*(1-p) weighted by v_weights
    // Guard: flat curve (total variation = 0) → MSS[i] = 0.0
    // Returns Ok(TvdMssResult { tvd, mss })
}
```
n≥3 guard (mirrors dispatch.rs Band guard pattern).

**Dispatcher usage** (in dispatch.rs — returns tvd field only):
```rust
DepthMethod::TotalVariation => {
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 curves for total variation depth".to_string(),
            actual: format!("{n}"),
        });
    }
    total_variation_depth_1d(data, data)?.tvd
}
```

---

### `src/depth/mod.rs` (barrel edit)

**Analog:** itself — current state (lines 11–36)

**Current pub mod block** (mod.rs lines 11–22):
```rust
pub mod band;
pub mod dispatch;
pub mod fraiman_muniz;
pub mod modal;
pub mod random_projection;
pub mod random_tukey;
pub mod rpd;
pub mod spatial;
```
Add after line 18 (`pub mod spatial;`):
```rust
pub mod erl;
pub mod extremal;
pub mod half_region;
pub mod hypo_epi;
pub mod linf;
pub mod tvd;
```

**Current pub use block** (mod.rs lines 24–36):
```rust
pub use band::{band_1d, modified_band_1d, modified_epigraph_index_1d};
pub use dispatch::{functional_boxplot, functional_depth, DepthMethod, FunctionalBoxplotResult};
// ...
```
Add after `pub use band::{...};`:
```rust
pub use erl::extreme_rank_length_depth_1d;
pub use extremal::extremal_depth_1d;
pub use half_region::{half_region_depth_1d, modified_half_region_depth_1d};
pub use hypo_epi::{epigraph_index_1d, hypograph_index_1d, modified_hypograph_index_1d};
pub use linf::linfinity_depth_1d;
pub use tvd::{total_variation_depth_1d, TvdMssResult};
```

---

### `src/depth/dispatch.rs` (enum + match edit)

**Analog:** itself — current `DepthMethod` enum (lines 26–43) and `functional_depth` match (lines 66–97)

**Enum extension pattern** (after `RandomProjection` variant at line 42):
```rust
// Existing last variant:
RandomProjection {
    nproj: usize,
    seed: u64,
},
// Add 9 new parameter-free variants:
/// Half-region depth (HRD = min(EI, HI)). Requires n ≥ 2.
HalfRegion,
/// Modified half-region depth (MHRD = min(MEI, MHI)). Requires n ≥ 2.
ModifiedHalfRegion,
/// Hypograph index (HI). Requires n ≥ 2.
HypographIndex,
/// Modified hypograph index (MHI).
ModifiedHypographIndex,
/// Epigraph index (EI, un-modified). Requires n ≥ 2.
EpigraphIndex,
/// Extremal depth (Narisetty & Nair 2016). Requires n ≥ 3.
Extremal,
/// Extreme rank length depth (Myllymäki et al. 2017). Requires n ≥ 2.
ExtremeRankLength,
/// L-infinity depth (fdaoutlier). Valid for n ≥ 1.
LInfinity,
/// Total variation depth + MSSI (Huang & Sun 2019). Returns TVD component. Requires n ≥ 3.
TotalVariation,
```

**Import addition** (dispatch.rs top, after existing `use crate::depth::{...}`):
```rust
use crate::depth::{
    epigraph_index_1d, extremal_depth_1d, extreme_rank_length_depth_1d,
    half_region_depth_1d, hypograph_index_1d, linfinity_depth_1d,
    modified_half_region_depth_1d, modified_hypograph_index_1d,
    total_variation_depth_1d,
};
```

**Match arm pattern** (mirrors Band arm at dispatch.rs lines 68–77):
```rust
// Existing Band arm (template):
DepthMethod::Band => {
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 curves for band depth".to_string(),
            actual: format!("{n}"),
        });
    }
    band_1d(data, data)
}
// New arms follow identical structure; TotalVariation uses .tvd field accessor:
DepthMethod::TotalVariation => {
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 curves for total variation depth".to_string(),
            actual: format!("{n}"),
        });
    }
    total_variation_depth_1d(data, data)?.tvd
}
```

---

### `src/lib.rs` (re-export edit)

**Analog:** itself — current `pub use depth::{...}` block (lines 422–428)

**Current block** (lib.rs lines 422–428):
```rust
pub use depth::{
    band_1d, fraiman_muniz_1d, fraiman_muniz_2d, functional_boxplot, functional_depth,
    functional_spatial_1d, functional_spatial_2d, kernel_functional_spatial_1d,
    kernel_functional_spatial_2d, modal_1d, modal_2d, modified_band_1d, modified_epigraph_index_1d,
    random_projection_1d, random_projection_1d_seeded, random_projection_2d, random_tukey_1d,
    random_tukey_1d_seeded, random_tukey_2d, DepthMethod, FunctionalBoxplotResult,
};
```
Extend by inserting after `modified_epigraph_index_1d,`:
```rust
    epigraph_index_1d, extremal_depth_1d, extreme_rank_length_depth_1d,
    half_region_depth_1d, hypograph_index_1d, linfinity_depth_1d,
    modified_half_region_depth_1d, modified_hypograph_index_1d,
    total_variation_depth_1d, TvdMssResult,
```

---

## Shared Patterns

### Parallel iteration (O(n²) measures)
**Source:** `src/depth/band.rs` lines 3, 9, 66 + `src/parallel.rs` lines 41–57
**Apply to:** `half_region.rs`, `hypo_epi.rs`, `erl.rs`, `linf.rs`, `tvd.rs`
```rust
use crate::iter_maybe_parallel;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

let result: Vec<f64> = iter_maybe_parallel!(0..nobj)
    .map(|i| { /* per-curve computation */ })
    .collect();
```

### FdMatrix column-major indexing
**Source:** `src/depth/dispatch.rs` lines 176, 193–200 + `src/depth/band.rs` lines 74–79
**Apply to:** all 6 new files
```rust
// Element (row i, time t) via 2D index:
let val = data[(i, t)];            // data_obj[(i, t)]
// Dimensions:
let n = data.nrows();              // number of curves
let m = data.ncols();              // number of evaluation points
// Column-major raw index (when needed): i + t * n
```

### Error handling: InvalidDimension
**Source:** `src/depth/dispatch.rs` lines 59–64 and 70–75
**Apply to:** all 6 new files at function entry
```rust
if nobj == 0 || nori == 0 || m == 0 {
    return Err(FdarError::InvalidDimension {
        parameter: "data_obj",
        expected: "non-empty matrix (nrows > 0 and ncols > 0)".to_string(),
        actual: format!("{nobj}x{m}"),
    });
}
// Per-measure n≥k guard:
if n < K {
    return Err(FdarError::InvalidDimension {
        parameter: "data_ori",
        expected: "at least K curves for <measure> depth".to_string(),
        actual: format!("{n}"),
    });
}
```

### Sort with NaN-safe tie-breaker
**Source:** `src/depth/dispatch.rs` lines 181–186
**Apply to:** `extremal.rs`, `erl.rs`
```rust
indices.sort_by(|&a, &b| {
    key_a.partial_cmp(&key_b)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.cmp(&b))          // index tie-breaker for determinism
});
```

### Result struct derive pattern
**Source:** `src/depth/dispatch.rs` lines 107–109
**Apply to:** `tvd.rs` (`TvdMssResult`)
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TvdMssResult { ... }
```

### Inline test module structure
**Source:** `src/depth/dispatch.rs` lines 240–400
**Apply to:** all 6 new files
```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixture builder
    fn sample(n: usize, m: usize) -> FdMatrix { ... }

    #[test]
    fn central_curve_deepest() { ... }

    #[test]
    fn empty_matrix_returns_err() { ... }

    #[test]
    fn too_few_curves_returns_err() { ... }
}
```
Dispatch.rs test fixture pattern (lines 245–255):
```rust
fn sample(n: usize, m: usize) -> FdMatrix {
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for t in 0..m {
            let x = t as f64 / (m as f64 - 1.0);
            col_major[i + t * n] = (x * std::f64::consts::PI).sin() + 0.05 * i as f64;
        }
    }
    FdMatrix::from_column_major(col_major, n, m).unwrap()
}
```

### Pointwise rank with tie-averaging helper
**Source:** RESEARCH.md Pattern 6 (no existing analog in codebase — new helper)
**Apply to:** `extremal.rs`, `erl.rs`, `tvd.rs`
```rust
/// Average-tie ranks for one column, matching R's rank(ties.method="average").
/// Returns 1-based ranks as f64.
fn column_ranks(data: &FdMatrix, col: usize) -> Vec<f64> {
    let n = data.nrows();
    let mut indexed: Vec<(f64, usize)> = (0..n).map(|i| (data[(i, col)], i)).collect();
    indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let mut k = 0;
    while k < n {
        let mut j = k + 1;
        while j < n && (indexed[j].0 - indexed[k].0).abs() < f64::EPSILON { j += 1; }
        let avg_rank = (k as f64 + 1.0 + j as f64) / 2.0;
        for item in &indexed[k..j] { ranks[item.1] = avg_rank; }
        k = j;
    }
    ranks
}
```
Define as `fn column_ranks(...)` (private) in each file that needs it, or promote to `pub(super)` in `mod.rs` if shared across ≥2 files.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|---|---|---|---|
| `column_ranks` helper | utility | transform | No pointwise rank helper exists in codebase; hand-roll per RESEARCH.md Pattern 6 |

---

## Metadata

**Analog search scope:** `fdars-core/src/depth/` (all 8 existing files read)
**Files scanned:** 5 (band.rs, dispatch.rs, mod.rs, fraiman_muniz.rs, parallel.rs) + lib.rs integration point
**Pattern extraction date:** 2026-08-19
