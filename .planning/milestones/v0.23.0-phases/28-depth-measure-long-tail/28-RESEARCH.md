# Phase 28: Depth-Measure Long Tail - Research

**Researched:** 2026-08-19
**Domain:** Functional data depth measures — Rust implementation in `fdars-core/src/depth/`
**Confidence:** MEDIUM (algorithms verified from authoritative R source code; math from official
package documentation and published papers)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **API**: Each of the 9 new measures is a `Result<Vec<f64>, FdarError>`-returning public function
  with dimension/parameter validation at entry. Signature: `fn <measure>_1d(data_obj: &FdMatrix,
  data_ori: &FdMatrix) -> Result<Vec<f64>, FdarError>`. Dispatcher passes `(data, data)`.
- **File grouping**: `half_region.rs` (HRD/MHRD), `hypo_epi.rs` (HI/MHI/EI), `extremal.rs`,
  `erl.rs`, `linf.rs`, `tvd.rs`.
- **DepthMethod variants** (parameter-free): `HalfRegion`, `ModifiedHalfRegion`,
  `HypographIndex`, `ModifiedHypographIndex`, `EpigraphIndex`, `Extremal`, `ExtremeRankLength`,
  `LInfinity`, `TotalVariation`. Existing variants and dispatcher signature **untouched**.
- **Re-exports**: all 9 functions at crate root (`lib.rs` `pub use depth::{…}`).
- **R baselines**: HRD/MHRD/HI/MHI/EI/ERL → `roahd`; extremal/L∞/TVD+MSSI → `fdaoutlier`.
- **Tie handling**: MEI's `<=` + 0.5 convention uniformly for index measures.
- **TVD+MSSI** follows Huang & Sun (2019); MSSI is intrinsic (no tuning knob). Stable interface
  for Phase 29's `tvdmss` — the function must return both TVD and MSSI as separate per-curve vecs.
- **Testing**: Synthetic fixtures — central curve deepest, outliers shallowest. Hand-computed
  small-case values within `1e-9`. Error-path coverage per entry point.
- **Parallelism**: O(n²) measures use `iter_maybe_parallel!`; O(n·m) stay sequential.
- **Constraints**: no new crate dependency; zero changes to existing public signatures; additive
  only; `Result<T, FdarError>` everywhere; inline `#[cfg(test)] mod tests`; crate-root re-exports.

### Claude's Discretion

- Internal helper structure within each new file (e.g., whether to factor sub-routines).

### Deferred Ideas (OUT OF SCOPE)

- Streaming/online depth variants.
- Plotting/rendering of depth regions or boxplots.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DEPTH-01 | Add missing univariate functional depth measures to `depth/` — HRD/MHRD, HI/MHI/EI, extremal depth, ERL depth, L∞ depth, TVD+MSSI. Each `Result`-returning over column-major `FdMatrix`, registered in `DepthMethod` dispatcher. Batch measures only. | Algorithms pinned below for all 9 measures from R source + official docs. Integration points identified from codebase read. |
</phase_requirements>

---

## Summary

Phase 28 adds nine canonical batch univariate functional depth measures to `fdars-core/src/depth/`,
grouped into six new files. The algorithms have been verified from the `roahd` and `fdaoutlier` R
package documentation and source code. All nine measures follow the existing two-matrix
`(data_obj, data_ori)` convention and return `Result<Vec<f64>, FdarError>`.

The three families have distinct computational structures: (a) the index measures (HI/MHI/EI and
their composites HRD/MHRD) are pure pointwise comparisons, either global-indicator or
pointwise-average; (b) extremal depth and ERL depth are rank-ordering measures that assign depths
as normalized positions in a total ordering; (c) L∞ depth is a pairwise-distance inversion; and
(d) TVD+MSSI is a two-component measure returning separate per-curve TVD and MSS vectors.

The most architecturally significant decision: TVD+MSSI must return **two** `Vec<f64>` values
(TVD and MSS separately) because Phase 29's `tvdmss` outlier detector consumes both independently.
The function signature should be a dedicated result struct or a tuple, not just `Vec<f64>`.

**Primary recommendation:** Define a `TvdMssResult { tvd: Vec<f64>, mss: Vec<f64> }` struct in
`tvd.rs`; expose it at crate root; make `DepthMethod::TotalVariation` dispatch return the `tvd`
field (depth = TVD component). Phase 29 calls `total_variation_depth_1d` directly for the full
struct.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| HRD, MHRD | `depth/half_region.rs` | `depth/dispatch.rs` | Composite of HI/EI and MHI/MEI; self-contained |
| HI, MHI, EI | `depth/hypo_epi.rs` | `depth/dispatch.rs` | Three related index measures in one file |
| Extremal depth | `depth/extremal.rs` | `depth/dispatch.rs` | Rank-ordering measure, standalone |
| ERL depth | `depth/erl.rs` | `depth/dispatch.rs` | Lexicographic rank-vector ordering, standalone |
| L∞ depth | `depth/linf.rs` | `depth/dispatch.rs` | Pairwise max-norm distance inversion |
| TVD + MSSI | `depth/tvd.rs` | `depth/dispatch.rs` + `outliers.rs` (Phase 29) | Two-component; Phase 29 consumes `TvdMssResult` |
| DepthMethod dispatch | `depth/dispatch.rs` | — | Extend enum + match arm only; no logic |
| Crate-root re-exports | `src/lib.rs` line 422 | — | Extend existing `pub use depth::{…}` block |

---

## Standard Stack

### Core (no new dependencies — all measures use existing primitives)

| Asset | Version | Purpose | Notes |
|-------|---------|---------|-------|
| `FdMatrix` | in-repo | Column-major data storage + access | `data[(i,t)]`, `nrows()`, `ncols()` |
| `FdarError` | in-repo | `InvalidDimension` / `InvalidParameter` / `ComputationFailed` | All new functions return `Result` |
| `iter_maybe_parallel!` | in-repo macro | Feature-gated rayon parallelism | O(n²) measures only |
| `crate::error::FdarError` | in-repo | Error type | Import path in new files |

No new crate dependencies. [VERIFIED: fdars-core/src/depth/mod.rs:1-37, fdars-core/src/lib.rs:422-428]

---

## Package Legitimacy Audit

No external packages are added in this phase. All implementation uses in-repo infrastructure.

**Packages added:** None.

---

## Architecture Patterns

### System Architecture Diagram

```
External caller
    │
    ▼
functional_depth(data, DepthMethod::X)          ← dispatch.rs (entry point)
    │
    ├─ HalfRegion         → half_region_depth_1d(data, data)    → half_region.rs
    ├─ ModifiedHalfRegion → modified_half_region_depth_1d(...)  → half_region.rs
    ├─ HypographIndex     → hypograph_index_1d(...)              → hypo_epi.rs
    ├─ ModifiedHypographIndex → modified_hypograph_index_1d(...) → hypo_epi.rs
    ├─ EpigraphIndex      → epigraph_index_1d(...)               → hypo_epi.rs
    ├─ Extremal           → extremal_depth_1d(...)               → extremal.rs
    ├─ ExtremeRankLength  → extreme_rank_length_depth_1d(...)    → erl.rs
    ├─ LInfinity          → linfinity_depth_1d(...)              → linf.rs
    └─ TotalVariation     → total_variation_depth_1d(...).map(|r| r.tvd) → tvd.rs
                                                    ↑
                        Phase 29 consumes total_variation_depth_1d directly
                        for TvdMssResult { tvd, mss }
```

### Recommended Project Structure

```
fdars-core/src/depth/
├── band.rs              # existing: BD, MBD, MEI
├── dispatch.rs          # existing: DepthMethod + functional_depth — EXTEND
├── extremal.rs          # NEW: extremal_depth_1d
├── erl.rs               # NEW: extreme_rank_length_depth_1d
├── fraiman_muniz.rs     # existing
├── half_region.rs       # NEW: half_region_depth_1d, modified_half_region_depth_1d
├── hypo_epi.rs          # NEW: hypograph_index_1d, modified_hypograph_index_1d, epigraph_index_1d
├── linf.rs              # NEW: linfinity_depth_1d
├── modal.rs             # existing
├── mod.rs               # existing — ADD pub mod + pub use for 6 new files
├── random_projection.rs # existing
├── random_tukey.rs      # existing
├── rpd.rs               # existing
├── spatial.rs           # existing
├── tests.rs             # existing integration tests
└── tvd.rs               # NEW: total_variation_depth_1d → TvdMssResult
```

---

## Measure Definitions (the core research deliverable)

All definitions verified against `roahd` and `fdaoutlier` R source code and official
package documentation. [CITED: rdrr.io/cran/roahd/man/, rdrr.io/cran/fdaoutlier/man/]

### 1. Hypograph Index (HI) — `hypo_epi.rs`

**Formula (global indicator):**

```
HI(X) = (1/N) · ∑_{i=1}^{N} 𝟙[ X_i(t) ≤ X(t) for all t ∈ [a,b] ]
```

Discrete grid: for each reference curve `j`, check whether `data_ori[(j,t)] <= data_obj[(i,t)]`
holds at **every** time point `t = 0..m`. If so, count that pair. HI = count / N.

**Tie convention**: `<=` (consistent with MEI, roahd convention). [CITED: rdrr.io/cran/roahd/man/HI.html]

**Properties**: Range [0,1]. The deepest curve (one above all others) has HI=1. HI requires all
points to satisfy the condition — one crossing destroys membership. Requires n≥1; meaningful n≥2.
O(n·N·m) = O(n²·m) but the inner loop is a simple `&&`-short-circuit, no parallelism needed
unless n is large.

**Monotonicity**: Curves at the top of the sample (magnitude outliers above) have HI→0; curves
below all others have HI→1. NOT a centrality depth — the *deepest* in HI sense is the topmost
curve. HRD = min(EI,HI) resolves this by taking the minimum.

---

### 2. Epigraph Index (EI) — `hypo_epi.rs`

**Formula (global indicator):**

```
EI(X) = (1/N) · ∑_{i=1}^{N} 𝟙[ X_i(t) ≥ X(t) for all t ∈ [a,b] ]
```

Discrete grid: check `data_ori[(j,t)] >= data_obj[(i,t)]` for every t. Count / N.

**Tie convention**: `>=`, i.e., `<=` in the reversed sense. [CITED: rdrr.io/cran/roahd/man/EI.html]

**Properties**: EI is the complement of HI in the symmetric sense. Curves at the bottom of the
sample have EI→1 (all others above them). Requires n≥1; meaningful n≥2. O(n²·m) with short-circuit.

---

### 3. Modified Hypograph Index (MHI) — `hypo_epi.rs`

**Formula (pointwise average):**

```
MHI(X) = (1/N) · ∑_{i=1}^{N} λ̃( X(t) ≥ X_i(t) )
         = (1/N) · ∑_{i=1}^{N} (1/m) · #{t : X(t) ≥ X_i(t)}
```

Discrete approximation of normalized Lebesgue measure: count the fraction of grid points where
`data_obj[(i,t)] >= data_ori[(j,t)]`.

**Tie convention**: `>=` uses `<=` on the other side; in implementation: count `data_obj[(i,t)]
>= data_ori[(j,t)]`. [CITED: rdrr.io/cran/roahd/man/MHI.html]

**Relation to MEI**: MHI(X) = average fraction of grid where X dominates X_i; MEI(X) = average
fraction where X is dominated by X_i. Both are pointwise-average versions. MHI(X) + MEI(X) = 1
when no ties; with ties and `<=`/`>=` both counting ties, they can sum to >1.

**Properties**: Range [0,1]. O(n²·m). Central curve gets MHI ≈ 0.5.

---

### 4. Modified Epigraph Index (MEI) — already shipped

`modified_epigraph_index_1d` in `band.rs`. Formula: `(1/N)∑(1/m)#{t: X(t) <= X_i(t)}`.
[VERIFIED: fdars-core/src/depth/band.rs:57-89]

Verbatim from source (lines 66-88):
```rust
iter_maybe_parallel!(0..nobj).map(|i| {
    let mut total = 0.0;
    for j in 0..nori {
        let mut count = 0.0;
        for t in 0..n_points {
            let xi = data_obj[(i, t)];
            let xj = data_ori[(j, t)];
            if xi <= xj {  // R's roahd::MEI uses <= for the epigraph condition
                count += 1.0;
            }
        }
        total += count / n_points as f64;
    }
    total / nori as f64
}).collect()
```

**New HI/EI/MHI must mirror this pattern exactly** (same loop structure, `iter_maybe_parallel!`,
same tie convention).

---

### 5. Half-Region Depth (HRD) — `half_region.rs`

**Formula:**

```
HRD(X) = min( EI(X), HI(X) )
```

[CITED: rdrr.io/cran/roahd/reference/HRD.html — "HRD = min(EI, HI)"]

**Properties**: Symmetric; a curve neither globally above nor globally below the sample scores
highest. HRD is zero if the curve is not globally contained in either half-region relative to
any other curve — very strict. Use case: clean, non-crossing curves.
Requires n≥2 (need at least 2 reference curves for meaningful comparison). O(n²·m).

---

### 6. Modified Half-Region Depth (MHRD) — `half_region.rs`

**Formula:**

```
MHRD(X) = min( MEI(X), MHI(X) )
```

[CITED: rdrr.io/cran/roahd/man/MHRD.html — "MHRD = min(MEI, MHI)"]

**Properties**: Pointwise-average version of HRD; tolerates crossing curves. Recommended over
HRD for real data. Central curve gets MHRD ≈ 0.5. Requires n≥2. O(n²·m).

**Implementation note**: Since MEI is already computed in `band.rs`, the `half_region.rs`
implementation can call `modified_epigraph_index_1d(data_obj, data_ori)` and
`modified_hypograph_index_1d(data_obj, data_ori)` and zip them with `f64::min`. Similarly,
`half_region_depth_1d` calls `epigraph_index_1d` and `hypograph_index_1d`.

---

### 7. Extremal Depth — `extremal.rs`

**Algorithm** (verified from `fdaoutlier` R source code): [CITED: rdrr.io/cran/fdaoutlier/src/R/extremal_depth.R]

Step 1 — Pointwise depth at each grid point `t`:
```
D(i, t) = 1 - |2·rank(X_i(t)) - n - 1| / n
```
where `rank` is 1-based (ties → average). This maps the most extreme (rank 1 or n) to depth ≈ 0
and the median-ranked curve to depth ≈ 1.

Step 2 — For each curve `i`, extract:
- `d_level(i)` = minimum pointwise depth across all grid points = `min_t D(i,t)`
- `mass(i)` = fraction of grid points where `D(i,t) == d_level(i)`
  (i.e., how long the curve attains its minimum depth)

Step 3 — Order all n curves by (ascending `d_level`, then descending `mass`):
- Curves with smaller minimum depth are more extreme (ordered first).
- Ties in minimum depth are broken by larger mass (more time at that depth → more extreme).

Step 4 — Assign depths: `ED(i) = rank_in_ordering(i) / n` where rank goes 1..n
(curve ordered first = rank 1 → depth = 1/n; deepest curve → depth = n/n = 1).

**Rust implementation note**: The ordering produces a permutation; use `argsort` on pairs
`(d_level[i], -mass[i])` to get the order, then invert the permutation to get depths.

**Properties**: Range (0,1]. Requires n≥3 (R source checks). O(n·m) for step 1, O(n·log n) for
sort. Sequential (no inner n² loop).

**Monotonicity**: Central curve has high minimum pointwise depth → small d_level → ordered last →
highest depth. Magnitude/shape outliers have low minimum depth → ordered first → lowest depth.

---

### 8. Extreme Rank Length (ERL) Depth — `erl.rs`

**Algorithm** (verified from `fdaoutlier` R source): [CITED: rdrr.io/cran/fdaoutlier/src/R/extreme_rank_length.R]

Step 1 — Pointwise ranks: for each grid point `t`, rank all n curves by value (ties → average).
`r[i,t]` = rank of curve i at point t, in 1..n.

Step 2 — Two-sided transform (default, matching `roahd` convention):
```
R[i,t] = min(r[i,t], n + 1 - r[i,t])
```
This maps ranks near 1 or n (extremes) to small values, and the median rank to large values.

Step 3 — For each curve i, sort its transformed rank vector `R[i,:]` in ascending order:
`sorted_R[i]` = sorted array of length p (grid size).

Step 4 — Total ordering by reverse-lexicographic comparison of sorted rank vectors:
Curve i is "more extreme" than curve j if `sorted_R[i]` precedes `sorted_R[j]`
lexicographically (first position where they differ has `sorted_R[i] < sorted_R[j]`).

Step 5 — `ERL(i) = #{j : curve i is NOT more extreme than j} / n`.
Equivalently: depth = 1 - (rank_in_ordering - 1) / n, where rank_in_ordering is
i's position in the total order from most extreme (position 1) to least extreme (position n).

**Properties**: Range (0,1]. Requires n≥2. O(n²·p) for pairwise comparisons. O(n²) inner loop
→ parallelize with `iter_maybe_parallel!` over the outer curve index.

**Monotonicity**: The least extreme curve (central) has no curve j for which it is more extreme
→ high ERL. An outlier is more extreme than almost all others → ERL near 0.

**Note on tie handling**: In step 1, ties use averaging (not `<=` + 0.5). ERL is a rank-based
measure and averaging is the standard R `rank` default. [ASSUMED — confirmed by R source `rank(x,
ties.method="average")` but not by a roahd-specific convention document]

---

### 9. L∞ Depth — `linf.rs`

**Formula** (verified from `fdaoutlier` R source): [CITED: rdrr.io/cran/fdaoutlier/src/R/linfinity_depth.R]

```
L∞_depth(X_i) = 1 / (1 + mean_j( max_t |X_i(t) - X_j(t)| ))
```

Step 1 — For each pair (i, j), compute the L∞ (sup-norm) distance:
`d(i,j) = max_{t=0..m-1} |data[(i,t)] - data[(j,t)]|`

Step 2 — For each curve i:
`L∞_depth(i) = 1 / (1 + (1/n) · ∑_j d(i,j))`

**Properties**: Range (0,1]. Monotonically decreasing in average L∞ distance from the sample.
Central curve (smallest mean L∞ distance) gets highest depth. Bounded: depth → 0 as distance → ∞.
Requires n≥1; meaningful n≥2. O(n²·m) — parallelize with `iter_maybe_parallel!` over i.

**Note**: The fdaoutlier implementation uses `dist(dt, method="maximum")` which computes
pairwise L∞ norm. This is NOT the L∞ norm to the pointwise median — it is average L∞ distance
to all other curves. [CITED: rdrr.io/cran/fdaoutlier/src/R/linfinity_depth.R]

---

### 10. Total Variation Depth + MSSI — `tvd.rs`

**Formulas** (verified from `fdaoutlier` R source code): [CITED: rdrr.io/cran/fdaoutlier/src/R/total_variation_depth.R]

**TVD component:**

Step 1 — Normalized pointwise rank: for each grid point `t`, rank all n curves.
`p[i,t] = rank(X_i(t)) / n` — normalized rank in (1/n .. 1].
(Ties: average, then divide by n.)

Step 2 — Pointwise variation score:
`TV[i,t] = p[i,t] · (1 - p[i,t])` — this is the pointwise variance of a Bernoulli(p) r.v.

Step 3 — TVD:
`TVD[i] = (1/m) · ∑_{t=0}^{m-1} TV[i,t]` — average over grid.

**Properties**: Range (0, 0.25]. The median-ranked curve at every t gets p=0.5 → TV=0.25 →
TVD→0.25 (maximum depth). Outliers with extreme ranks get TV→0 → TVD→0.

**MSS component (shape variation):**

The R source delegates to a C++ function `C_totalVariationDepth` which computes `shape_variation`
(a matrix of size (m-1) × n). Based on Huang & Sun (2019), the shape variation at interval k
between t_k and t_{k+1} for curve i depends on the conditional ranks given the adjacent column.
The R side then weights by:
```
v_weights[k,i] = |X_i(t_{k+1}) - X_i(t_k)| / ∑_k |X_i(t_{k+1}) - X_i(t_k)|
```
(normalized total variation of the curve itself — the relative contribution of interval k to
the curve's total variation).

`MSS[i] = ∑_{k=0}^{m-2} shape_variation[k,i] · v_weights[k,i]`

**C++ internals note**: The `shape_variation[k,i]` is the pointwise depth analog for the
derivative at position k. Specifically, using the notation of the paper: for each interval k,
compute the rank of `(X_i(t_{k+1}) - X_i(t_k))` among all curves' first differences at k, then
apply the same `rank*(1-rank)` transform. [ASSUMED — derived from paper description; C++ source
not directly read]

**Return type — critical for Phase 29 stability:**

The Rust function must return a struct, not a plain `Vec<f64>`:

```rust
/// Result of total variation depth + MSSI computation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TvdMssResult {
    /// Total variation depth (magnitude component). Length = nrows of data.
    pub tvd: Vec<f64>,
    /// Modified shape similarity index (shape component). Length = nrows of data.
    pub mss: Vec<f64>,
}
```

The `DepthMethod::TotalVariation` dispatcher returns `tvd` only (as the depth value).
Phase 29 calls `total_variation_depth_1d(data, data)?` directly for the full struct.

**Properties**: TVD range (0,0.25]; MSS range [0,0.25] (same transform applied to derivative
ranks). Requires n≥3 (R source checks). O(n²·m) — two rank passes.

**Monotonicity**: High TVD → central in magnitude. High MSS → shape-similar to the sample.
Magnitude outliers: low TVD, normal MSS. Shape outliers: normal TVD, low MSS.

---

## Complexity, Parallelism, and Minimum-n Guards

| Measure | Complexity | Parallelize? | Min n Guard |
|---------|-----------|-------------|-------------|
| HI | O(n²·m) short-circuit | Optional — short-circuit limits gain | ≥2 (for meaningful result) |
| EI | O(n²·m) short-circuit | Optional | ≥2 |
| MHI | O(n²·m) | Yes — `iter_maybe_parallel!` (mirrors MEI) | ≥1 (empty → empty) |
| MEI | O(n²·m) | Already parallel | ≥1 |
| HRD | O(n²·m) | Via HI+EI calls | ≥2 |
| MHRD | O(n²·m) | Via MHI+MEI calls | ≥2 |
| Extremal | O(n·m + n log n) | Sequential | ≥3 |
| ERL | O(n²·p + n log p) | `iter_maybe_parallel!` over outer i | ≥2 |
| L∞ | O(n²·m) | `iter_maybe_parallel!` over i | ≥2 (meaningful) |
| TVD+MSS | O(n²·m) rank pass | `iter_maybe_parallel!` for rank pass | ≥3 |

**Dispatcher guard pattern** (from `dispatch.rs` lines 69-86): [VERIFIED: fdars-core/src/depth/dispatch.rs:56-99]
```rust
DepthMethod::HalfRegion => {
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 curves for half-region depth".to_string(),
            actual: format!("{n}"),
        });
    }
    half_region_depth_1d(data, data)?
}
```
Each function also validates internally (per CONTEXT.md decision).

---

## Integration Points (exact edit locations)

### `depth/mod.rs` — add module declarations and re-exports

Current state [VERIFIED: fdars-core/src/depth/mod.rs:11-36]:
```rust
pub mod band;
pub mod dispatch;
pub mod fraiman_muniz;
// ... (8 existing mod declarations)

pub use band::{band_1d, modified_band_1d, modified_epigraph_index_1d};
pub use dispatch::{functional_boxplot, functional_depth, DepthMethod, FunctionalBoxplotResult};
// ... (6 more pub use lines)
```

Add after existing `pub mod` block:
```rust
pub mod extremal;
pub mod erl;
pub mod half_region;
pub mod hypo_epi;
pub mod linf;
pub mod tvd;
```

Add after existing `pub use` block:
```rust
pub use extremal::extremal_depth_1d;
pub use erl::extreme_rank_length_depth_1d;
pub use half_region::{half_region_depth_1d, modified_half_region_depth_1d};
pub use hypo_epi::{epigraph_index_1d, hypograph_index_1d, modified_hypograph_index_1d};
pub use linf::linfinity_depth_1d;
pub use tvd::{total_variation_depth_1d, TvdMssResult};
```

### `depth/dispatch.rs` — extend `DepthMethod` enum and match

Current enum ends at `RandomProjection { nproj, seed }` (line 42). Add 9 variants after it:
```rust
/// Half-region depth (HRD = min(EI, HI)).
HalfRegion,
/// Modified half-region depth (MHRD = min(MEI, MHI)).
ModifiedHalfRegion,
/// Hypograph index (HI).
HypographIndex,
/// Modified hypograph index (MHI).
ModifiedHypographIndex,
/// Epigraph index (EI, un-modified).
EpigraphIndex,
/// Extremal depth (Narisetty & Nair 2016).
Extremal,
/// Extreme rank length depth (Myllymäki et al. 2017).
ExtremeRankLength,
/// L-infinity depth (fdaoutlier).
LInfinity,
/// Total variation depth (Huang & Sun 2019). Returns TVD component.
TotalVariation,
```

Add import to dispatch.rs top:
```rust
use crate::depth::{
    epigraph_index_1d, extremal_depth_1d, extreme_rank_length_depth_1d,
    half_region_depth_1d, hypograph_index_1d, linfinity_depth_1d,
    modified_half_region_depth_1d, modified_hypograph_index_1d,
    total_variation_depth_1d,
};
```

Match arms in `functional_depth` — add after `RandomProjection` arm:
```rust
DepthMethod::HalfRegion => {
    if n < 2 { /* InvalidDimension */ }
    half_region_depth_1d(data, data)?
}
DepthMethod::ModifiedHalfRegion => {
    if n < 2 { /* InvalidDimension */ }
    modified_half_region_depth_1d(data, data)?
}
DepthMethod::HypographIndex => {
    if n < 2 { /* InvalidDimension */ }
    hypograph_index_1d(data, data)?
}
DepthMethod::ModifiedHypographIndex => {
    modified_hypograph_index_1d(data, data)?
}
DepthMethod::EpigraphIndex => {
    if n < 2 { /* InvalidDimension */ }
    epigraph_index_1d(data, data)?
}
DepthMethod::Extremal => {
    if n < 3 { /* InvalidDimension */ }
    extremal_depth_1d(data, data)?
}
DepthMethod::ExtremeRankLength => {
    if n < 2 { /* InvalidDimension */ }
    extreme_rank_length_depth_1d(data, data)?
}
DepthMethod::LInfinity => {
    linfinity_depth_1d(data, data)?
}
DepthMethod::TotalVariation => {
    if n < 3 { /* InvalidDimension */ }
    total_variation_depth_1d(data, data)?.tvd
}
```

### `lib.rs` line 422 — extend `pub use depth::{…}`

Current block [VERIFIED: fdars-core/src/lib.rs:422-428]:
```rust
pub use depth::{
    band_1d, fraiman_muniz_1d, fraiman_muniz_2d, functional_boxplot, functional_depth,
    functional_spatial_1d, functional_spatial_2d, kernel_functional_spatial_1d,
    kernel_functional_spatial_2d, modal_1d, modal_2d, modified_band_1d, modified_epigraph_index_1d,
    random_projection_1d, random_projection_1d_seeded, random_projection_2d, random_tukey_1d,
    random_tukey_1d_seeded, random_tukey_2d, DepthMethod, FunctionalBoxplotResult,
};
```

Extend by adding after `modified_epigraph_index_1d,`:
```rust
    epigraph_index_1d, extremal_depth_1d, extreme_rank_length_depth_1d,
    half_region_depth_1d, hypograph_index_1d, linfinity_depth_1d,
    modified_half_region_depth_1d, modified_hypograph_index_1d,
    modified_hypograph_index_1d, total_variation_depth_1d, TvdMssResult,
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pointwise ranks with ties | Custom rank function | Standard tie-averaging algorithm: sort indices, identify tie groups, assign average | This is a 10-line loop; hand-rolling is fine, but must match R's `rank(ties.method="average")` |
| Parallel iteration | Raw `rayon::par_iter()` calls | `iter_maybe_parallel!(0..n)` macro | Feature-gated — removes rayon dep when `parallel` feature disabled |
| Sort with custom comparator | Sort structs | `sort_unstable_by` with `partial_cmp` + `unwrap_or(Equal)` for f64 | Pattern in existing code (dispatch.rs line 185) |
| Result propagation in dispatcher | Nested match | `?` operator on `Result<Vec<f64>, FdarError>` | Existing pattern |
| MSS shape_variation | Port C++ from fdaoutlier | Implement in Rust directly using finite differences + rank | No FFI needed — the C++ is compact math, Rust can express it cleanly |

---

## Common Pitfalls

### Pitfall 1: HI/EI Global vs. MHI/MEI Pointwise Confusion

**What goes wrong**: Implementing HI as a pointwise-average (like MHI) instead of a global indicator.
**Why it happens**: The two families look similar; "hypograph" appears in both names.
**How to avoid**: HI uses `&&`-over-all-t (one crossing destroys membership). MHI uses
`count / n_points` (average fraction). The test for HI: a curve that crosses the reference curve
once must get HI=0 from that pair.
**Warning signs**: HI values are not 0 or multiples of 1/N (they should only be k/N for integer k).

### Pitfall 2: HRD/MHRD Composite Calls Causing Double Parallelism

**What goes wrong**: `half_region_depth_1d` calls two O(n²) sub-functions each with their own
`iter_maybe_parallel!` — when rayon is enabled, this nests parallel iterators.
**Why it happens**: Rayon forbids nested parallelism by default.
**How to avoid**: HRD computes EI and HI sequentially in a single pass over pairs:
for each pair (i,j), check both the global-above and global-below conditions together.
MHRD can call MEI and MHI separately (they are already parallel internally) because the outer
`zip()` is not itself parallel.

### Pitfall 3: Extremal Depth Ordering — Stable vs Unstable Sort

**What goes wrong**: Using unstable sort for the (d_level, -mass) sort; different runs may
produce different orderings for tied depth values.
**Why it happens**: f64 `partial_cmp` with NaN produces `Ordering::Equal`, unstable sort
does not preserve input order for Equal.
**How to avoid**: Use `sort_unstable_by` with a tie-breaker on the original index
(`then(a.cmp(&b))`). This matches R's behavior (uses first occurrence in ties). Confirm in
test with known tied inputs.

### Pitfall 4: ERL Lexicographic Comparison — Off-by-One in Depth Assignment

**What goes wrong**: Computing `depth = count_more_extreme / n` (0-indexed) vs `(count + 1) / n`.
**Why it happens**: The R source uses `depths / n` after C_extremeRank returns counts in 1..n.
**How to avoid**: The C_extremeRank returns the total ordering position (1-based rank). Divide
by n. In Rust: after sorting curves by their sorted-rank vectors lexicographically, assign
depth[curve_in_position_k] = k / n_curves as f64 (0-indexed k from 0..n, so depth = (k+1)/n).

### Pitfall 5: TVD Rank Normalization — Divide by n or n+1

**What goes wrong**: Using `rank / (n + 1)` instead of `rank / n`.
**Why it happens**: Some depth formulas normalize by n+1 to avoid 0/1 boundary values.
**How to avoid**: The fdaoutlier source explicitly uses `/ n_curves`. In Rust: use
`rank_f64 / n as f64`. Verify with a 3-curve toy case: ranks [1,2,3] → p = [1/3, 2/3, 1.0] →
TV = [2/9, 2/9, 0] → TVD = 4/27 ≈ 0.148 for the top/bottom curves.

### Pitfall 6: MSS — Missing the Curve-Level Normalization

**What goes wrong**: `shape_variation` values are summed without the `v_weights` normalization
(the relative total-variation weight per interval).
**Why it happens**: The formula's two-step computation is easy to collapse incorrectly.
**How to avoid**: A curve with `rowSums(diff_data) == 0` (flat curve, no variation) produces a
division by zero in `v_weights`. Guard: if total variation of curve i is 0, set MSS[i] = 0.

### Pitfall 7: Clippy `--all-targets` on Test/Bench Code

**What goes wrong**: Clippy passes on `--lib` but fails on test/bench code.
**Why it happens**: CI uses `--all-targets` (MEMORY.md); test helpers may trigger dead_code or
unused-variable lints.
**How to avoid**: Run `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
locally before committing. [VERIFIED: MEMORY.md ci-clippy-all-targets-gate.md pointer]

---

## Code Examples

### Pattern 1: HI Implementation (global indicator, mirrors MEI structure)

```rust
// Source: derived from fdars-core/src/depth/band.rs:66-88 (MEI) + roahd::HI definition
#[must_use = "expensive computation whose result should not be discarded"]
pub fn hypograph_index_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
) -> Result<Vec<f64>, FdarError> {
    let (nobj, nori, m) = (data_obj.nrows(), data_ori.nrows(), data_obj.ncols());
    if nobj == 0 || nori == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data_obj",
            expected: "non-empty matrix".to_string(),
            actual: format!("{}x{}", nobj, m),
        });
    }
    // HI uses global indicator — no iter_maybe_parallel! needed for small n;
    // use it anyway for consistency with MEI.
    use crate::iter_maybe_parallel;
    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;

    let depths = iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut count = 0.0_f64;
            'outer: for j in 0..nori {
                for t in 0..m {
                    if data_ori[(j, t)] > data_obj[(i, t)] {
                        continue 'outer; // X_j not globally ≤ X_i at this point
                    }
                }
                count += 1.0;
            }
            count / nori as f64
        })
        .collect();
    Ok(depths)
}
```

### Pattern 2: MHI Implementation (pointwise average, exact mirror of MEI)

```rust
// Source: derived from fdars-core/src/depth/band.rs:66-88 (MEI) + roahd::MHI definition
pub fn modified_hypograph_index_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
) -> Result<Vec<f64>, FdarError> {
    let (nobj, nori, m) = (data_obj.nrows(), data_ori.nrows(), data_obj.ncols());
    // ... validation ...
    let depths = iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut total = 0.0;
            for j in 0..nori {
                let mut count = 0.0;
                for t in 0..m {
                    if data_obj[(i, t)] >= data_ori[(j, t)] {  // MHI: X(t) >= X_i(t)
                        count += 1.0;
                    }
                }
                total += count / m as f64;
            }
            total / nori as f64
        })
        .collect();
    Ok(depths)
}
```

### Pattern 3: Extremal Depth (rank-based, O(n·m + n log n))

```rust
// Source: translated from fdaoutlier/R/extremal_depth.R pwise_depth + ordering
pub fn extremal_depth_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,  // for self-depth: same as data_obj
) -> Result<Vec<f64>, FdarError> {
    let (n, m) = (data_ori.nrows(), data_ori.ncols());
    // Step 1: pointwise depth D[i,t] = 1 - |2*rank(x[t])[i] - n - 1| / n
    // Step 2: d_level[i] = min_t D[i,t];  mass[i] = #{t: D[i,t]==d_level[i]} / m
    // Step 3: sort indices by (d_level asc, mass desc) — stable to break ties by index
    // Step 4: depth[original_idx] = (position_in_order + 1) as f64 / n as f64
    // Returns Result<Vec<f64>, FdarError>
    todo!()
}
```

### Pattern 4: ERL Depth (lexicographic rank-vector ordering)

```rust
// Source: translated from fdaoutlier/R/extreme_rank_length.R
pub fn extreme_rank_length_depth_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
) -> Result<Vec<f64>, FdarError> {
    let (n, m) = (data_ori.nrows(), data_ori.ncols());
    // Step 1: rank[i,t] = rank of data[(i,t)] among column t (ties → average)
    // Step 2: R[i,t] = min(rank[i,t], n+1-rank[i,t])  (two_sided)
    // Step 3: sorted_R[i] = sort(R[i,:]) ascending — a Vec<f64> of length m
    // Step 4: for each i, count j where sorted_R[j] < sorted_R[i] lexicographically
    //         depth[i] = count / n
    // Step 4 is O(n²·m) — parallelize with iter_maybe_parallel! over i
    todo!()
}
```

### Pattern 5: TvdMssResult Return Struct

```rust
// Source: design derived from fdaoutlier::total_variation_depth return list + CONTEXT.md
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct TvdMssResult {
    /// Total variation depth (magnitude component). Higher = more central.
    pub tvd: Vec<f64>,
    /// Modified shape similarity index (shape component). Higher = more shape-central.
    pub mss: Vec<f64>,
}

/// Dispatcher usage (returns TVD as the depth value):
// DepthMethod::TotalVariation => total_variation_depth_1d(data, data)?.tvd
```

### Pattern 6: Pointwise Rank with Tie-Averaging (reusable helper)

```rust
/// Compute average-tie ranks for one column of a FdMatrix.
/// Returns ranks as f64 in 1..=n (matching R's rank(ties.method="average")).
fn column_ranks(data: &FdMatrix, col: usize) -> Vec<f64> {
    let n = data.nrows();
    let mut indexed: Vec<(f64, usize)> = (0..n).map(|i| (data[(i, col)], i)).collect();
    indexed.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let mut k = 0;
    while k < n {
        let mut j = k + 1;
        while j < n && indexed[j].0 == indexed[k].0 {
            j += 1;
        }
        // Tie group: k..j — assign average rank (k+1 + j) / 2
        let avg_rank = (k as f64 + 1.0 + j as f64) / 2.0;
        for item in &indexed[k..j] {
            ranks[item.1] = avg_rank;
        }
        k = j;
    }
    ranks
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Only 4 depth methods in fdars | 13 methods after Phase 28 | v0.23.0 Phase 28 | Closes roahd/fdaoutlier coverage gap |
| MEI only (epigraph) | MEI + MHI + EI + HI + HRD + MHRD | v0.23.0 | Full roahd index family |
| No extremal/ERL/L∞/TVD | All four added | v0.23.0 | Matches fdaoutlier's depth catalog |

**Deprecated/outdated:** None — this is purely additive.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` / `#[cfg(test)]` |
| Config file | `fdars-core/Cargo.toml` `[[bench]]` entries (for criterion) |
| Quick run command | `cargo test -p fdars-core --features linalg,parallel 2>&1 \| tail -5` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel && cargo clippy --all-targets --features linalg,parallel -- -D warnings` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DEPTH-01 / HI | HI: global indicator — curves above ref get 0 from that pair | unit | `cargo test -p fdars-core --features linalg,parallel hypograph` | ❌ Wave 0 |
| DEPTH-01 / MHI | MHI: pointwise average — values in [0,1], central ≈ 0.5 | unit | same | ❌ Wave 0 |
| DEPTH-01 / EI | EI: global indicator complement | unit | same | ❌ Wave 0 |
| DEPTH-01 / HRD | HRD = min(EI,HI); central deepest | unit | same | ❌ Wave 0 |
| DEPTH-01 / MHRD | MHRD = min(MEI,MHI); central deepest | unit | same | ❌ Wave 0 |
| DEPTH-01 / Extremal | Central curve gets depth n/n; outlier gets 1/n | unit | same | ❌ Wave 0 |
| DEPTH-01 / ERL | Central curve not more extreme than others; high depth | unit | same | ❌ Wave 0 |
| DEPTH-01 / L∞ | Closest-to-centroid curve deepest | unit | same | ❌ Wave 0 |
| DEPTH-01 / TVD | TVD: median-ranked curve gets max TVD; MSS: shape-central gets max | unit | same | ❌ Wave 0 |
| DEPTH-01 / dispatch | All 9 DepthMethod variants round-trip through functional_depth | integration | same | ❌ Wave 0 |
| DEPTH-01 / error paths | Empty matrix, 1-curve, mismatched dims → Err, never panic | unit | same | ❌ Wave 0 |

### Wave 0 Gaps

- [ ] `fdars-core/src/depth/half_region.rs` — covers HRD/MHRD with inline `#[cfg(test)]`
- [ ] `fdars-core/src/depth/hypo_epi.rs` — covers HI/MHI/EI
- [ ] `fdars-core/src/depth/extremal.rs` — covers extremal depth
- [ ] `fdars-core/src/depth/erl.rs` — covers ERL depth
- [ ] `fdars-core/src/depth/linf.rs` — covers L∞ depth
- [ ] `fdars-core/src/depth/tvd.rs` — covers TVD+MSS, TvdMssResult

---

## Security Domain

> `security_enforcement` not set → treated as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — pure library, no auth |
| V3 Session Management | no | — stateless computation |
| V4 Access Control | no | — no user data |
| V5 Input Validation | yes | Dimension checks via `FdarError::InvalidDimension` at function entry |
| V6 Cryptography | no | — no secrets |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow in `i + t * n` indexing | Tampering | `FdMatrix` encapsulates indexing; no raw arithmetic |
| Division by zero in rank normalization | Tampering | Guard `n == 0` at entry; TVD: guard `n_points == 0` |
| NaN propagation in f64 sort | Tampering | `partial_cmp(...).unwrap_or(Equal)` (existing pattern) |
| Panic from `column_ranks` with empty column | Denial of Service | Empty matrix check at function entry |

---

## Environment Availability

Step 2.6: SKIPPED — this phase is pure Rust library code with no external tools beyond the
project's existing toolchain (Cargo, rustc 1.97.0, already verified). No databases, CLI tools,
or external services required.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | ERL `shape_variation` in TVD MSS uses first-differences ranked pointwise (derivative rank) | Measure Definitions §10 | MSS values diverge from fdaoutlier; Phase 29 tvdmss outlier classification may differ |
| A2 | ERL ties use averaging (not ≤+0.5) in the rank pass | Measure Definitions §8 | Small numerical differences vs roahd for tied inputs |
| A3 | HI/EI with n<2 should return Err (not empty Vec) | Integration Points | If returns empty Vec, dispatcher can't tell success from empty matrix |
| A4 | `DepthMethod::LInfinity` does not need n≥2 guard (works with n=1, depth=1.0) | Don't Hand-Roll | If R errors on n<2, our implementation diverges |

---

## Open Questions

1. **MSS shape_variation exact formula**
   - What we know: The R code calls `C_totalVariationDepth` (C++); the output is a (m-1)×n matrix of
     shape_variation values; it is weighted by `v_weights = |diffs| / rowSums(|diffs|)`.
   - What's unclear: Whether shape_variation[k,i] is `rank(Δ_k X_i) / n * (1 - rank(Δ_k X_i) / n)`
     or a conditional rank. The paper says it's the "pointwise variation of the derivative rank."
   - Recommendation: Implement as `TV[i,k] = p_deriv[i,k] * (1 - p_deriv[i,k])` where
     `p_deriv[i,k] = rank(X_i(t_{k+1}) - X_i(t_k)) / n` among all n curves' k-th differences.
     This matches the pattern of TVD itself applied to the derivative. If it diverges from R
     values, add a `// fdars divergence: ...` rustdoc note.

2. **HI/EI tie handling**
   - What we know: roahd uses `<=` for MEI (verified in codebase). HI's condition is
     `X_i(t) <= X(t) for all t` — tie at exact equality means `<=` counts it as inside the
     hypograph.
   - What's unclear: Whether roahd HI also uses strict `<` or `<=` for the global membership.
   - Recommendation: Use `<=` (consistent with MEI tie convention from CONTEXT.md).

---

## Sources

### Primary (MEDIUM confidence — from official R package documentation/source)

- [rdrr.io/cran/roahd/man/HRD.html](https://astamm.github.io/roahd/reference/HRD.html) — HRD formula
- [rdrr.io/cran/roahd/man/MHRD.html](https://rdrr.io/cran/roahd/man/MHRD.html) — MHRD formula
- [rdrr.io/cran/roahd/man/HI.html](https://astamm.github.io/roahd/reference/HI.html) — HI formula
- [rdrr.io/cran/roahd/man/MHI.html](https://astamm.github.io/roahd/reference/MHI.html) — MHI formula
- [rdrr.io/cran/roahd/man/EI.html](https://rdrr.io/cran/roahd/man/EI.html) — EI formula
- [rdrr.io/cran/fdaoutlier/src/R/extremal_depth.R](https://rdrr.io/cran/fdaoutlier/src/R/extremal_depth.R) — extremal depth source
- [rdrr.io/cran/fdaoutlier/src/R/extreme_rank_length.R](https://rdrr.io/cran/fdaoutlier/src/R/extreme_rank_length.R) — ERL source
- [rdrr.io/cran/fdaoutlier/src/R/linfinity_depth.R](https://rdrr.io/cran/fdaoutlier/src/R/linfinity_depth.R) — L∞ source
- [rdrr.io/cran/fdaoutlier/src/R/total_variation_depth.R](https://rdrr.io/cran/fdaoutlier/src/R/total_variation_depth.R) — TVD+MSS source
- [fdars-core/src/depth/band.rs:57-89](VERIFIED) — MEI implementation to mirror
- [fdars-core/src/depth/dispatch.rs:24-99](VERIFIED) — enum + dispatcher pattern
- [fdars-core/src/depth/mod.rs:11-36](VERIFIED) — module layout to extend
- [fdars-core/src/lib.rs:422-428](VERIFIED) — crate-root re-export block

### Secondary (LOW confidence — web search without direct source read)

- Narisetty & Nair (2016) JASA — extremal depth paper (referenced but PDF not readable)
- Huang & Sun (2019) Technometrics — TVD+MSSI paper (referenced from fdaoutlier docs)
- Myllymäki et al. (2017) — ERL original paper (GET package)

---

## Metadata

**Confidence breakdown:**
- HI/MHI/EI formulas: HIGH — verified from official roahd documentation with exact formulas
- HRD/MHRD composites: HIGH — `min(EI,HI)` and `min(MEI,MHI)` confirmed from documentation
- Extremal depth algorithm: MEDIUM — R source code read directly; C functions hidden
- ERL algorithm: MEDIUM — R source read; lexicographic comparison confirmed from sources; C extension hidden
- L∞ formula: HIGH — complete R source read; simple 3-line formula
- TVD formula: HIGH — complete R source read; `p*(1-p)` pattern explicit
- MSS formula: MEDIUM — R side clear; C++ shape_variation internals assumed from paper description
- Integration points (mod.rs, dispatch.rs, lib.rs): HIGH — verified from codebase read

**Research date:** 2026-08-19
**Valid until:** 2026-09-18 (stable R packages — 30 days)
