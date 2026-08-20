# Phase 29: Outlier-Detector Suite - Research

**Researched:** 2026-08-19
**Domain:** Functional data outlier detection — fdaoutlier/roahd algorithms in Rust
**Confidence:** MEDIUM (algorithm logic verified against R source; C++ internals partially ASSUMED)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Each detector is a `Result<T, FdarError>`-returning public fn in `outliers.rs`, crate-root re-exported.
- Dedicated result structs (derive `Debug, Clone, PartialEq`, `#[non_exhaustive]`, conditional serde):
  - `TvdMssOutliers { magnitude_outliers: Vec<usize>, shape_outliers: Vec<usize>, tvd: Vec<f64>, mss: Vec<f64> }`
  - `MuodResult { magnitude/amplitude/shape indices + flagged sets }`
  - `DepthgramResult { index-pair coordinates across 3 representations + outliers }`
  - `sequential_transform_outliers` → struct with per-transform flags + a unioned outlier set
- Small config structs / optional params only where a cutoff factor is genuinely needed (tvdmss cutoff, muod cutoff); sensible defaults matching the R baseline.
- Reuse-first: `tvdmss` builds on Phase 28 `total_variation_depth_1d` (`TvdMssResult`); reuse existing `outliergram` / `magnitude_shape_outlyingness` / `functional_boxplot` fence logic rather than reimplementing. No new crate dependency.
- tvdmss: fdaoutlier two-stage: TVD flags magnitude outliers, then MSS flags shape outliers among the rest. Tunable boxplot-style tail cutoff (IQR factor, default 1.5) on the TVD/MSS statistics. Output both `magnitude_outliers` and `shape_outliers` index sets plus the `tvd`/`mss` vectors. Consumes Phase 28's pinned `TvdMssResult { tvd, mss }` interface directly.
- muod: per-curve magnitude / amplitude / shape indices via regression of each curve on the pointwise mean (intercept ≈ magnitude, slope ≈ amplitude, residual/1−R² ≈ shape); flag each index via a boxplot cutoff.
- sequential_transform_outliers: fdaoutlier transforms T0 (raw), T1 (one-step normalization), T2 (successive differences); run a functional-boxplot / directional detector per transform. Output a struct with per-transform flags plus the unioned outlier set.
- depthgram: numeric only — the (MEI/MBD-style) index-pair coordinates across the three representations plus flagged outliers. No rendering.
- Pin the exact `muod` / `sequential_transform` / `depthgram` statistics from `fdaoutlier`/`roahd` during research; document any divergence from the R baseline in rustdoc (prior-milestone practice).
- Tests: inject magnitude outliers and shape outliers into a synthetic sample; assert `tvdmss` flags both classes and the other detectors return the expected outlier index sets within a documented tolerance; error-path coverage (empty / single-curve / mismatched dims / degenerate columns → `FdarError`) at every entry point.

### Claude's Discretion

- Internal helper visibility (private vs pub(crate)) for IQR fence, MUOD index computation.
- Whether to add a `SeqTransformConfig` struct or pass sequence as `&[SeqTransform]` enum.
- Whether `DepthgramResult` stores 6 `Vec<f64>` vectors (mbd_mei_d, mei_mbd_d, mbd_mei_t, mei_mbd_t, mbd_mei_t2, mei_mbd_t2) or wraps them in a sub-struct.
- Exact minimum-n guard per detector (research the R baselines, recommend).
- Whether to parallelize O(n²) loops with `iter_maybe_parallel!`.

### Deferred Ideas (OUT OF SCOPE)

- Any plotting/rendering of MS-plots, outliergrams, depthgrams, or outlier flags (numeric only).
- `fdaPOIFD` partially-observed-data detectors (explicitly deferred this milestone).
- `O` (outlyingness) transform for multivariate functional data (needs separate projection depth; univariate only).
- Alignment/warping-based transform (not yet supported even in fdaoutlier).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OUT-01 | Add `tvdmss`, `muod`, `sequential_transform_outliers`, and `depthgram` to `outliers.rs`; reuse existing MS-plot/outliergram machinery + DEPTH-01 depths; numeric outputs only; R baseline: `fdaoutlier`/`roahd`; depends on DEPTH-01. | All four detectors pinned algorithmically in §Algorithm Pinning. Reuse map in §Reusable Assets. Integration points in §Architecture Patterns. |
</phase_requirements>

---

## Summary

Phase 29 adds four fdaoutlier/roahd outlier detectors to `fdars-core/src/outliers.rs` as pure
numeric additive extensions. Phase 28 shipped all required depth primitives (`total_variation_depth_1d`
→ `TvdMssResult`, `modified_epigraph_index_1d`, `modified_band_1d`, `hypograph_index_1d`, etc.) and
the `functional_boxplot` fence — this phase consumes all of them.

The detectors are algorithmically simple pipelines over existing depth functions; the complexity is
in correctly wiring the R baseline procedures to the column-major `FdMatrix` API and providing useful
numeric outputs (indices + score vectors) without plotting machinery.

**Primary recommendation:** Implement in a single plan as four additive blocks in `outliers.rs`,
with a shared private `iqr_fence` helper; no new submodules or crate dependencies needed. All
O(n²) inner loops should be wrapped in `iter_maybe_parallel!` per the crate convention.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| tvdmss two-stage detection | outliers.rs | depth/tvd.rs (depth provider) | Consumes `TvdMssResult` from tvd.rs; logic in outliers.rs |
| muod index computation | outliers.rs private fn | helpers.rs (`quantile_sorted`, `sort_nan_safe`) | Pure numeric; no external depth call needed |
| sequential transform detection | outliers.rs | depth/dispatch.rs (`functional_boxplot`) | Calls existing `functional_boxplot` after each transform |
| depthgram index pairs | outliers.rs | depth/band.rs (`modified_band_1d`, `modified_epigraph_index_1d`) | Compose MBD∘MEI and MEI∘MBD in two passes |
| IQR/boxplot cutoff | outliers.rs private helper | helpers.rs (`quantile_sorted`, `sort_nan_safe`) | Reusable across all four detectors |

---

## Algorithm Pinning (Exact Specifications)

This section pins each detector's algorithm to the verified R source so the planner can write
correct code without further research.

### 1. `tvdmss` — Two-Stage TVD+MSS Detector

**Source:** fdaoutlier `tvdmss_new.R` — fetched via GitHub API 2026-08-19 [CITED: github.com/otsegun/fdaoutlier/blob/master/R/tvdmss_new.R]

**Stage 1 — Shape outlier detection (classical IQR boxplot on MSS):**

```text
shape_boxstats <- boxplot(mss, range = emp_factor_mss, plot = FALSE)
shape_outliers <- which(mss %in% shape_boxstats$out[shape_boxstats$out < mean(mss)])
```

Translation: sort all n MSS values, compute Q1 = 25th percentile, Q3 = 75th percentile,
IQR = Q3 − Q1, lower_fence = Q1 − emp_factor_mss × IQR. Curves with MSS < lower_fence
are flagged as **shape outliers** (low MSS = unusual shape). R's `boxplot(range=1.5)` is a
lower-one-sided fence on the MSS distribution.

**Why lower fence for MSS:** MSS is a depth-like measure (higher = more central shape). Shape
outliers have low MSS. The R code uses `< mean(mss)` to further restrict to the lower tail only.
[CITED: github.com/otsegun/fdaoutlier tvdmss_new.R]

**Stage 2 — Magnitude outlier detection (functional boxplot on TVD of remaining curves):**

```r
functional_boxplot(dts[non_shape_outlier_rows, ],
                   depth_values = tvd[non_shape_outlier_rows],
                   emp_factor   = emp_factor_tvd,
                   central_region = central_region_tvd * n_curves / nrow(dts_reduced))
```

Translation: remove the shape outliers from the data and TVD vector, then run a
López-Pintado–Romo depth-fence functional boxplot on the reduced dataset. The
`central_region` is scaled so that `floor(central_region_tvd × n_orig)` curves form the
central envelope, regardless of how many shape outliers were removed.

**Parameters and defaults:**
- `emp_factor_mss = 1.5` — IQR multiplier for the shape stage
- `emp_factor_tvd = 1.5` — inflation factor for the magnitude-stage functional boxplot
- `central_region_tvd = 0.5` — fraction of original n used as central region in stage 2

**Output:** `TvdMssOutliers { magnitude_outliers: Vec<usize>, shape_outliers: Vec<usize>, tvd: Vec<f64>, mss: Vec<f64> }`. All four vectors are over the original indices.

**Dependency:** calls `total_variation_depth_1d(data, data)?` which returns `TvdMssResult { tvd, mss }` — both fields consumed directly. [VERIFIED: fdars-core/src/depth/tvd.rs:22-28]

The pinned interface is:
```rust
pub struct TvdMssResult {
    pub tvd: Vec<f64>,
    pub mss: Vec<f64>,
}
```

**Rust translation:**
```rust
// Stage 1: classical IQR fence on MSS (lower tail)
fn iqr_lower_fence(values: &[f64], factor: f64) -> f64 {
    let mut sorted = values.to_vec();
    sort_nan_safe(&mut sorted);
    let q1 = quantile_sorted(&sorted, 0.25);
    let q3 = quantile_sorted(&sorted, 0.75);
    q1 - factor * (q3 - q1)
}
// shape_outliers: indices where mss[i] < fence AND mss[i] < mean(mss)
// Stage 2: reuse functional_boxplot(DepthMethod::..., factor) from dispatch.rs
//   on the reduced data with pre-computed TVD depths
```

**Minimum n:** n ≥ 3 (inherited from `total_variation_depth_1d`).

**Complexity:** O(n·m) for TVD (already computed); O(n log n) for IQR sort; O(n·m) for functional boxplot central region envelope. Not O(n²).

---

### 2. `muod` — Massive Unsupervised Outlier Detection

**Source:** fdaoutlier `muod.R` + `cor_cov_blockwise.cpp` — fetched via GitHub API 2026-08-19 [CITED: github.com/otsegun/fdaoutlier/blob/master/R/muod.R]

**The three MUOD indices** are computed from a pairwise linear regression of each curve against
every other curve — but in the Fast-MUOD variant implemented in fdaoutlier, each curve is
regressed against the **pointwise mean** (the column mean vector `μ_t = (1/n) Σ_i X_i(t)`).

The C++ function `corCovBlock` computes per-pair rolling averages of:
- `sXY = Σ_t (X_i(t) − X̄_i)(X_j(t) − X̄_j) / (p−1)` — covariance between curves i and j
- Three indices derived from this:
  - `kaccess[j*3]   += sXY / std_i`  → shape pre-index
  - `kaccess[j*3+1] += sXY * (mean_i / var_i)` → magnitude pre-index
  - `kaccess[j*3+2] += sXY / var_i`  → amplitude pre-index

Then in R:
- `pre_ind[,1] /= data_sds`  → shape index (correlation-like; 1.0 for perfect shape match)
- `pre_ind[,2] = data_means - pre_ind[,2]` → magnitude index (0.0 if same location)
- amplitude index is `pre_ind[,3]` as-is (slope-like; 1.0 for same amplitude)

Finally: `abs(indices - benchmark)` where `benchmark = c(1, 0, 1)`:
- **shape index** = |correlation_with_reference − 1|  (0 = same shape, >0 = different shape)
- **magnitude index** = |curve_mean − reference_mean|  (0 = same location)
- **amplitude index** = |slope_to_reference − 1|  (0 = same amplitude)

**Simplified Rust interpretation** (without the pairwise C++ machinery, using pointwise mean):

For each curve i, compute the OLS regression of X_i on the pointwise mean μ:
```text
μ_t = mean over rows at each t          (length m vector)
X̄_i = mean(X_i(t)) over t              (scalar row mean)
μ̄   = mean(μ_t) over t                 (scalar, = X̄_i globally if same mean)
cov(X_i, μ) = Σ_t (X_i(t)-X̄_i)(μ_t-μ̄) / (m-1)
var(μ)       = Σ_t (μ_t-μ̄)² / (m-1)
slope_i      = cov(X_i, μ) / var(μ)
intercept_i  = X̄_i − slope_i × μ̄

shape_index_i     = |correlation(X_i, μ) − 1| = |cov(X_i,μ) / (std(X_i)·std(μ)) − 1|
magnitude_index_i = |intercept_i − 0| = |X̄_i − slope_i × μ̄|
amplitude_index_i = |slope_i − 1|
```

**Cutoff methods:**
- `boxplot` (default): `cutoff = boxplot.stats(sorted_index)$stats[5]` — this is the
  standard upper whisker: Q3 + 1.5 × IQR (R `boxplot.stats` convention). [CITED: github.com/otsegun/fdaoutlier/blob/master/R/muod.R]
- `tangent`: spline-fitting heuristic (complex; lower priority, can be omitted or marked TODO).

**Output:** `MuodResult { shape_outliers: Vec<usize>, magnitude_outliers: Vec<usize>, amplitude_outliers: Vec<usize>, shape_index: Vec<f64>, magnitude_index: Vec<f64>, amplitude_index: Vec<f64> }`. All vectors of length n.

**Minimum n:** n ≥ 3 (R source: `if((dm <- dim(dts))[1] < 3) stop(...)`). [CITED: github.com/otsegun/fdaoutlier/blob/master/R/muod.R]

**Complexity:** O(n·m) for the index computation (one pass per curve against the mean). Not O(n²) in the Fast-MUOD variant. Suitable for `iter_maybe_parallel!` over curves.

**Divergence from R baseline to document in rustdoc:** The R implementation uses the pairwise
C++ covariance block and then averages, which produces the same result as regressing against the
pointwise mean (Fast-MUOD). The Rust implementation uses the simpler pointwise-mean regression
directly. This is the same approach as fdaoutlier's `Fast-MUOD`. [ASSUMED — confirmed equivalent by paper description but Rust impl deviates from the pair-loop C++ approach]

---

### 3. `sequential_transform_outliers` — Sequential-Transformation Detection

**Source:** fdaoutlier `sequential_transformations.R` — fetched via GitHub API 2026-08-19 [CITED: github.com/otsegun/fdaoutlier/blob/master/R/sequential_transformations.R]

**Transform definitions** (exact, from R source):

| Tag | Transform | Rust formula |
|-----|-----------|-------------|
| `T0` | Raw data (identity) | data unchanged |
| `T1` | Vertical centering: subtract row mean | `X_i(t) -= mean_t(X_i)` |
| `T2` | L2 normalization: divide by row L2 norm | `X_i(t) /= sqrt(Σ_t X_i(t)²)` |
| `D1` | First-difference (lag-1) | `Y_i(t) = X_i(t+1) − X_i(t)`, m−1 columns |
| `D2` | Same as D1 | identical to D1 (re-differencing if applied after D1) |

`O` (outlyingness transform for multivariate data) is **deferred** — out of scope for this phase.

The default recommended sequence from the paper (Algorithm 1, Dai et al. 2020): `["T0", "T1", "D1"]`.
[CITED: Dai et al. 2020, doi:10.1016/j.csda.2020.106960 via fdaoutlier docs]

**Important ordering:** transforms are applied **cumulatively**. The sequence `["T0","T1","D1"]` means:
- Step 1: apply functional boxplot to raw data (T0)
- Step 2: apply T1 to the raw data (center), then functional boxplot
- Step 3: apply D1 to the T1-transformed data (difference the centered data), then functional boxplot

Each step's input is the **previous step's output**, not the original data.

**Base detector:** functional boxplot after each step. The R default depth method is `mbd`.
In fdars: call `functional_boxplot(transformed_data, DepthMethod::ModifiedBand, factor)`.

**Outlier combination:** no automatic union in R. Each transform step's outlier set is returned
independently. The locked decision requires: struct with per-transform flags **plus a unioned set**.
The union is the set of all indices flagged by at least one transform step — computed by the fdars
implementation as a convenience, not part of the R baseline.

**Parameters and defaults:**
- `depth_method`: `DepthMethod` selector (default: `ModifiedBand`)
- `emp_factor: f64 = 1.5`
- `central_region: f64 = 0.5`
- `sequence: &[SeqTransform]` where `SeqTransform` is an enum with variants `T0, T1, T2, D1, D2`

**Output struct:**
```rust
pub struct SeqTransformOutliers {
    pub per_transform_outliers: Vec<(SeqTransform, Vec<usize>)>,
    pub union_outliers: Vec<usize>,   // union over all transform steps
}
```

**Minimum n:** n ≥ 2 (functional_boxplot requires n ≥ 2). After D1, m becomes m−1; require m ≥ 2
before differencing.

**Complexity:** O(n·m) per transform step via functional_boxplot. Steps are sequential.

**T2 guard:** if any curve has L2 norm = 0 (all-zero curve), division produces NaN/inf — return
`FdarError::ComputationFailed` with explanation "zero-norm curve in T2 normalization".

---

### 4. `depthgram` — Depthgram Index Statistic

**Source:** roahd `depthgram.R` — fetched via GitHub API 2026-08-19 [CITED: github.com/astamm/roahd/blob/master/R/depthgram.R]

For **univariate** functional data (p=1 dimension), the depthgram computes three
`(MBD(MEI), MEI(MBD))` index pairs from three representations of the data:

**Representation 1 — Dimension-wise (d):**

Compute MBD and MEI of the raw data curves at each evaluation point, treating each
evaluation point as an "observation":

```text
rmat[i,t] = rank of X_i(t) among {X_1(t),...,X_n(t)}  (at each time t)
down[i,t]  = rmat[i,t] - 1
up[i,t]    = n - rmat[i,t]
mbd.d[i]   = (Σ_t up[i,t]*down[i,t] / m + n - 1) / (n*(n-1)/2)
mei.d[i]   = Σ_t (up[i,t]+1) / (n*m)
```

This is exactly `modified_band_1d(data, data)` and `modified_epigraph_index_1d(data, data)`
from `depth/band.rs`. [VERIFIED: fdars-core/src/depth/band.rs:43-95]

**Representation 2 — Time-wise (t):**

Transpose the role of time and observations. For each curve i, produce an m-dimensional
vector of "depth-at-time" values, then compute MBD and MEI treating each time point as an
observation group. In practice: the R code reuses the rank matrix `rmat` looping over
time points. For univariate (p=1):

```text
mbd.t[i,t] = same formula as mbd.d but aggregated over p=1 dimension → equals mbd.d
mei.t[i,t] = same formula as mei.d but aggregated → equals mei.d (for p=1)
```

For p=1, representation 2 collapses to representation 1. [CITED: roahd depthgram.R source]

**Representation 3 — Time/correlation-wise (t2):**

Same as representation 2 but with a sign correction: if `cor(mei.d[:,i], mei.d[:,i-1]) < 0`,
flip the ranks for dimension i (multiply up/down by -1). For univariate data (p=1), there is
no cross-dimension correlation to correct, so representation 3 = representation 2 = representation 1.

**For univariate fdars (p=1):** all three representations produce the same vectors. The depthgram
statistic reduces to:
```
mbd_mei = MBD of the MEI vector  = modified_band_1d(mei_matrix, mei_matrix)
mei_mbd = MEI of the MBD vector  = modified_epigraph_index_1d(mbd_matrix, mbd_matrix)
```

where `mei_matrix` and `mbd_matrix` are n×1 matrices (column vectors).

**Outlier detection in the depthgram:** The R source applies the outliergram boundary formula
to each representation:
```text
a2 = a0 = -2 / (n*(n-1))
a1 = 2*(n+1) / (n-1)
dist = (a0 + a1*mei + a2*n²*mei²) - mbd   ← deviation from parabola
q  = quantile(dist, c(0.25, 0.75))
lim = outliergram_factor * (q[2]-q[1]) + q[2]   ← UPPER fence (dist > lim → outlier)
```

Shape outliers: `dist > lim` (curves whose MBD is below the parabolic boundary).
Magnitude outliers: functional boxplot on MBD values with `boxplot_factor = 1.5`.
[CITED: roahd depthgram.R source]

**Divergence from R baseline:** The R depthgram is designed for p-variate (multivariate) data
and returns n×p matrices. The fdars implementation handles univariate (p=1) only — for
multivariate data, each component is treated independently (as separate univariate FDA
problems). This is consistent with the existing codebase pattern (all depth measures are
1D). Document in rustdoc: "For p=1, all three depthgram representations are equivalent;
this implementation handles univariate functional data only."

**Output:**
```rust
pub struct DepthgramResult {
    // Per-representation index vectors (length n each):
    pub mbd_mei_d: Vec<f64>,    // MBD of MEI, dimension-wise
    pub mei_mbd_d: Vec<f64>,    // MEI of MBD, dimension-wise
    pub mbd_mei_t: Vec<f64>,    // MBD of MEI, time-wise (= mbd_mei_d for p=1)
    pub mei_mbd_t: Vec<f64>,    // MEI of MBD, time-wise (= mei_mbd_d for p=1)
    pub mbd_mei_t2: Vec<f64>,   // correlation-corrected (= above for p=1)
    pub mei_mbd_t2: Vec<f64>,
    // Outlier indices:
    pub shape_outliers: Vec<usize>,
    pub magnitude_outliers: Vec<usize>,
    // Raw depth components:
    pub mbd: Vec<f64>,
    pub mei: Vec<f64>,
}
```

**Minimum n:** n ≥ 2 (MBD requires n ≥ 2; outliergram parabola requires n ≥ 4 for meaningful IQR).

**Complexity:** O(n²·m) for MBD (pair-loop); O(n·m) for MEI. Same as `modified_band_1d`.

---

## Standard Stack

No new crate dependencies. All computation uses existing fdars infrastructure.

### Reusable Assets (VERIFIED in this session)

| Asset | File | What it Provides | Verification |
|-------|------|------------------|--------------|
| `total_variation_depth_1d` | `src/depth/tvd.rs` | `TvdMssResult { tvd, mss }` — pinned interface for tvdmss | [VERIFIED: fdars-core/src/depth/tvd.rs:22-28] |
| `functional_boxplot` | `src/depth/dispatch.rs` | López-Pintado–Romo fence: central region + whisker outlier indices | [VERIFIED: fdars-core/src/depth/dispatch.rs:243-341] |
| `DepthMethod` | `src/depth/dispatch.rs` | Enum with 13 variants incl. `ModifiedBand`, `TotalVariation`, etc. | [VERIFIED: fdars-core/src/depth/dispatch.rs:29-74] |
| `FunctionalBoxplotResult` | `src/depth/dispatch.rs` | `outliers: Vec<usize>`, `depths: Vec<f64>`, fence bounds | [VERIFIED: fdars-core/src/depth/dispatch.rs:210-227] |
| `modified_band_1d` | `src/depth/band.rs` | MBD: O(n²·m) per-pair depth — needed for depthgram | [VERIFIED: fdars-core/src/depth/band.rs:43-50] |
| `modified_epigraph_index_1d` | `src/depth/band.rs` | MEI: O(n·m) — needed for depthgram and outliergram | [VERIFIED: fdars-core/src/depth/band.rs:57-95] |
| `outliergram` | `src/outliers.rs` | Parabola fit + IQR fence on residuals; reuse parabola logic | [VERIFIED: fdars-core/src/outliers.rs:278-332] |
| `quantile_sorted` | `src/helpers.rs` | Linear-interpolation quantile from sorted slice | [VERIFIED: fdars-core/src/helpers.rs:283-298] |
| `sort_nan_safe` | `src/helpers.rs` | Stable NaN-safe sort (used throughout codebase) | [VERIFIED: fdars-core/src/helpers.rs:10] |
| `iter_maybe_parallel!` | `src/parallel.rs` | Rayon-gated parallel iterator macro | [VERIFIED: fdars-core/src/outliers.rs:8] |

### IQR Fence Helper (new private function)

A private `iqr_fence` helper is needed in `outliers.rs` (no existing equivalent in the public API):

```rust
/// Classical boxplot IQR fence.
/// Returns (lower_fence, upper_fence) = (Q1 - factor*IQR, Q3 + factor*IQR).
fn iqr_fence(values: &[f64], factor: f64) -> (f64, f64) {
    let mut sorted = values.to_vec();
    crate::helpers::sort_nan_safe(&mut sorted);
    let q1 = crate::helpers::quantile_sorted(&sorted, 0.25);
    let q3 = crate::helpers::quantile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;
    (q1 - factor * iqr, q3 + factor * iqr)
}
```

Used by: tvdmss (lower fence on MSS), muod (upper fence per index).

The existing `outliergram` already has inline IQR logic at lines 314-319
([VERIFIED: fdars-core/src/outliers.rs:314-319]):
```rust
// verbatim from outliers.rs:
let q1 = sorted_resid[n / 4];
let q3 = sorted_resid[3 * n / 4];
let iqr = q3 - q1;
let threshold = q1 - factor * iqr;
```

This uses floor-quartile (index n/4) rather than `quantile_sorted`. For consistency with the
rest of the codebase, use `quantile_sorted` in the new helper. The planner should choose one
and document the choice; both produce the same result for large n.

---

## Architecture Patterns

### Recommended Project Structure

All new code goes in `src/outliers.rs` as additive blocks — no new files.
`src/lib.rs` line 434 extends the `pub use outliers::{...}` block.

```
src/
├── outliers.rs          # ADD: TvdMssOutliers, MuodResult, SeqTransformOutliers,
│                        #      DepthgramResult, tvdmss, muod,
│                        #      sequential_transform_outliers, depthgram,
│                        #      private helpers: iqr_fence, muod_indices,
│                        #      seq_transform_apply, depthgram_parabola_flags
├── lib.rs               # EXTEND: pub use outliers::{...} block (~line 434)
└── depth/
    ├── tvd.rs           # READ-ONLY: consume TvdMssResult
    ├── dispatch.rs      # READ-ONLY: consume functional_boxplot, DepthMethod
    └── band.rs          # READ-ONLY: consume modified_band_1d, modified_epigraph_index_1d
```

### System Architecture Diagram

```
User call
  │
  ├─► tvdmss(data, config)
  │     ├─► total_variation_depth_1d(data, data) → TvdMssResult{tvd, mss}
  │     ├─► iqr_fence(mss, emp_factor_mss) → flag shape_outliers [lower tail]
  │     ├─► functional_boxplot(data_reduced, ModifiedBand, emp_factor_tvd)
  │     │        with pre-supplied depths=tvd_reduced, adjusted central_region
  │     └─► TvdMssOutliers{magnitude_outliers, shape_outliers, tvd, mss}
  │
  ├─► muod(data, config)
  │     ├─► column_mean(data) → mu_t [pointwise mean]
  │     ├─► for each curve i: OLS regression of X_i on mu_t
  │     │        → shape_idx[i], magnitude_idx[i], amplitude_idx[i]
  │     ├─► iqr_fence(shape_idx, factor) → upper fence → flag shape_outliers
  │     ├─► iqr_fence(magnitude_idx, factor) → upper fence → flag mag_outliers
  │     ├─► iqr_fence(amplitude_idx, factor) → upper fence → flag amp_outliers
  │     └─► MuodResult{..indices, ..outlier_sets}
  │
  ├─► sequential_transform_outliers(data, sequence, config)
  │     ├─► for each transform in sequence:
  │     │       apply transform → transformed_data
  │     │       functional_boxplot(transformed_data, depth_method, factor)
  │     │       → per_transform_outliers[transform]
  │     ├─► union of all per_transform_outlier sets → union_outliers
  │     └─► SeqTransformOutliers{per_transform_outliers, union_outliers}
  │
  └─► depthgram(data, config)
        ├─► modified_band_1d(data, data) → mbd [n values]
        ├─► modified_epigraph_index_1d(data, data) → mei [n values]
        ├─► MBD applied to column-vector of mei → mbd_mei [scalar depth of MEI vector]
        ├─► MEI applied to column-vector of mbd → mei_mbd [scalar depth of MBD vector]
        ├─► outliergram parabola on (mei, mbd) → shape_outliers
        ├─► functional_boxplot on mbd → magnitude_outliers
        └─► DepthgramResult{mbd_mei_d, mei_mbd_d, [t/t2 copies], shape_outliers, mag_outliers, mbd, mei}
```

### Pattern: Result Structs

All new result types follow the established crate pattern (verbatim from `OutligramResult`):
```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TvdMssOutliers {
    pub magnitude_outliers: Vec<usize>,
    pub shape_outliers: Vec<usize>,
    pub tvd: Vec<f64>,
    pub mss: Vec<f64>,
}
```

### Pattern: Public Function Skeleton

```rust
/// Detect magnitude and shape outliers via Total Variation Depth + MSSI.
///
/// Two-stage procedure: (1) classical IQR boxplot on MSS flags shape outliers;
/// (2) functional boxplot on TVD of remaining curves flags magnitude outliers.
///
/// # Arguments
/// * `data` — functional data matrix (n × m)
/// * `config` — tunable factors (defaults match fdaoutlier: factor=1.5, central_region=0.5)
pub fn tvdmss(data: &FdMatrix, config: TvdMssConfig) -> Result<TvdMssOutliers, FdarError> {
    let n = data.nrows();
    // guard: n >= 3 (total_variation_depth_1d requires n >= 3)
    let depth = total_variation_depth_1d(data, data)?;
    // stage 1 ...
    // stage 2 ...
}
```

### Anti-Patterns to Avoid

- **Hand-rolling MBD/MEI inside depthgram:** Use the existing `modified_band_1d` and `modified_epigraph_index_1d` from `depth/band.rs` — they already handle column-major layout.
- **Re-implementing the functional boxplot fence:** Use `functional_boxplot` from `depth/dispatch.rs` with pre-supplied `DepthMethod` and factor — it already implements the López-Pintado–Romo envelope.
- **Using R's n/4 index quartile** inside `iqr_fence`: Use `quantile_sorted` for consistency; document the minor difference vs. R's floor-quartile in the new helper's doc comment.
- **T2 normalization without zero-norm guard:** A curve with all-zero values has L2 norm = 0; guard with `if norm < 1e-15 { return Err(FdarError::ComputationFailed) }`.
- **D1 transform without column-count guard:** After one difference, m becomes m−1. If m == 1 before differencing, the result has 0 columns — return error instead.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| MBD computation in depthgram | Custom rank-sum loop | `modified_band_1d(data, data)` | Already O(n²·m), column-major correct |
| MEI computation in depthgram | Custom epigraph loop | `modified_epigraph_index_1d(data, data)` | Already parallel-gated |
| Functional boxplot fence | Custom central-region envelope | `functional_boxplot(data, method, factor)` | Handles n/2 rounding, depth pre-supply |
| TVD + MSS | Custom rank-pass loops | `total_variation_depth_1d(data, data)` | Pinned interface for Phase 29 |
| Quantile computation | Manual sorted-index access | `quantile_sorted` + `sort_nan_safe` | Already in codebase, linear-interpolation |

---

## Common Pitfalls

### Pitfall 1: tvdmss Stage 2 Central Region Scaling

**What goes wrong:** Using `central_region_tvd = 0.5` directly as the fraction of the
*reduced* dataset, not the original.
**Why it happens:** The R code scales: `central_region = central_region_tvd × n_orig / n_reduced`.
If 10 shape outliers are removed from n=100, n_reduced=90, the central region should contain
`0.5 × 100 = 50` curves, so pass `50/90 ≈ 0.556` to `functional_boxplot`.
**How to avoid:** Track `n_orig` before shape outlier removal; adjust the fraction.
**Warning signs:** Test on a dataset with known shape outliers — the magnitude outlier count
changes when you change the scaling.

### Pitfall 2: MUOD Indices — Degenerate Variance

**What goes wrong:** If a curve is constant (all values equal), `std(X_i) = 0`, dividing by
`std(X_i)` produces NaN in the shape index.
**How to avoid:** Guard with `if std_i < 1e-15 { shape_index = 0.0 }` — a constant curve has
"perfect shape" relative to any reference.
**Warning signs:** NaN propagates through IQR fence → all curves flagged.

### Pitfall 3: Sequential Transform — Cumulative Input

**What goes wrong:** Applying each transform to the *original* data rather than the *previous
transform's output*.
**Why it happens:** T1 then D1 means: center the data, *then* difference the centered data.
**How to avoid:** Maintain a `current_data: FdMatrix` variable and mutate it through the sequence.

### Pitfall 4: Depthgram — MBD of a Column Vector

**What goes wrong:** Calling `modified_band_1d(mei_vector, mei_vector)` where `mei_vector`
is a flat `Vec<f64>` — MBD expects an `FdMatrix`.
**How to avoid:** Reshape the MEI vector into an n×1 `FdMatrix` (n observations, 1 evaluation
point), then call `modified_band_1d`. Verify MBD with n×1 input is valid (it is — band depth
with 1 column degenerates to a rank-based scalar depth, which is exactly what's needed).

### Pitfall 5: Column-Major Index Mapping in FdMatrix

**What goes wrong:** Accessing element (i, t) as `data[i * m + t]` (row-major) instead of
`data[(i, t)]` which internally maps to `data[i + t * n]`.
**How to avoid:** Always use `data[(i, t)]` operator; never use raw index arithmetic.
[VERIFIED: fdars-core/src/matrix.rs:38-44]

---

## Code Examples

### tvdmss — Two-Stage Procedure

```rust
// Source: github.com/otsegun/fdaoutlier/blob/master/R/tvdmss_new.R (translated)
use crate::depth::tvd::total_variation_depth_1d;
use crate::depth::dispatch::{functional_boxplot, DepthMethod};

pub fn tvdmss(data: &FdMatrix, config: TvdMssConfig) -> Result<TvdMssOutliers, FdarError> {
    let n = data.nrows();
    let depth = total_variation_depth_1d(data, data)?;
    let (lower_mss, _) = iqr_fence(&depth.mss, config.emp_factor_mss);
    let mean_mss = depth.mss.iter().sum::<f64>() / n as f64;

    // Stage 1: shape outliers = mss below lower fence AND below mean
    let shape_outliers: Vec<usize> = (0..n)
        .filter(|&i| depth.mss[i] < lower_mss && depth.mss[i] < mean_mss)
        .collect();

    // Stage 2: functional boxplot on TVD of remaining curves
    let keep: Vec<usize> = (0..n).filter(|i| !shape_outliers.contains(i)).collect();
    let n_reduced = keep.len();
    let adj_cr = config.central_region_tvd * n as f64 / n_reduced as f64;
    // Build reduced FdMatrix and tvd sub-slice, then call functional_boxplot...
    // magnitude_outliers = functional_boxplot(reduced_data, ...).outliers
    // re-map indices back to original
    Ok(TvdMssOutliers { magnitude_outliers, shape_outliers, tvd: depth.tvd, mss: depth.mss })
}
```

### muod — OLS on Pointwise Mean

```rust
// Source: github.com/otsegun/fdaoutlier/blob/master/R/muod.R (simplified)
fn muod_indices(data: &FdMatrix) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (n, m) = data.shape();
    // Compute pointwise mean mu_t (length m)
    let mu: Vec<f64> = (0..m).map(|t| (0..n).map(|i| data[(i,t)]).sum::<f64>() / n as f64).collect();
    let mu_mean = mu.iter().sum::<f64>() / m as f64;
    let mu_var = mu.iter().map(|&v| (v - mu_mean).powi(2)).sum::<f64>() / (m - 1) as f64;
    // Per-curve regression
    iter_maybe_parallel!(0..n).map(|i| {
        let xi_mean = (0..m).map(|t| data[(i,t)]).sum::<f64>() / m as f64;
        let cov_i = (0..m).map(|t| (data[(i,t)] - xi_mean) * (mu[t] - mu_mean)).sum::<f64>() / (m-1) as f64;
        let slope = if mu_var > 1e-15 { cov_i / mu_var } else { 1.0 };
        let intercept = xi_mean - slope * mu_mean;
        let xi_std = ((0..m).map(|t| (data[(i,t)] - xi_mean).powi(2)).sum::<f64>() / (m-1) as f64).sqrt();
        let mu_std = mu_var.sqrt();
        let corr = if xi_std > 1e-15 && mu_std > 1e-15 { cov_i / (xi_std * mu_std) } else { 1.0 };
        let shape_idx = (corr - 1.0).abs();
        let magnitude_idx = intercept.abs();
        let amplitude_idx = (slope - 1.0).abs();
        (shape_idx, magnitude_idx, amplitude_idx)
    }).unzip3() // collect into three Vec<f64>
}
```

### depthgram — Apply MBD to MEI Vector

```rust
// Source: github.com/astamm/roahd/blob/master/R/depthgram.R (univariate case)
use crate::depth::band::{modified_band_1d, modified_epigraph_index_1d};

pub fn depthgram(data: &FdMatrix, config: DepthgramConfig) -> Result<DepthgramResult, FdarError> {
    let n = data.nrows();
    let mbd = modified_band_1d(data, data);
    let mei = modified_epigraph_index_1d(data, data);
    // Wrap mbd and mei as n×1 FdMatrix for second-level depth computation
    let mbd_mat = FdMatrix::from_column_major(mbd.clone(), n, 1)?;
    let mei_mat = FdMatrix::from_column_major(mei.clone(), n, 1)?;
    let mbd_mei = modified_band_1d(&mei_mat, &mei_mat);   // MBD of MEI
    let mei_mbd = modified_epigraph_index_1d(&mbd_mat, &mbd_mat); // MEI of MBD
    // Outliergram parabola for shape outliers
    let a2 = -2.0 / (n as f64 * (n as f64 - 1.0));
    let a0 = a2;
    let a1 = 2.0 * (n as f64 + 1.0) / (n as f64 - 1.0);
    let dist: Vec<f64> = (0..n).map(|i| {
        (a0 + a1 * mei[i] + a2 * (n as f64).powi(2) * mei[i].powi(2)) - mbd[i]
    }).collect();
    let (_, upper_dist) = iqr_fence(&dist, config.outliergram_factor);
    let shape_outliers: Vec<usize> = (0..n).filter(|&i| dist[i] > upper_dist).collect();
    // Functional boxplot on MBD for magnitude outliers
    let mbd_mat2 = FdMatrix::from_column_major(mbd.clone(), n, 1)?;
    let fbp = functional_boxplot(&mbd_mat2, DepthMethod::ModifiedBand, config.boxplot_factor)?;
    let magnitude_outliers = fbp.outliers;
    // For p=1, all three representations are identical
    Ok(DepthgramResult {
        mbd_mei_d: mbd_mei.clone(), mei_mbd_d: mei_mbd.clone(),
        mbd_mei_t: mbd_mei.clone(), mei_mbd_t: mei_mbd.clone(),
        mbd_mei_t2: mbd_mei, mei_mbd_t2: mei_mbd,
        shape_outliers, magnitude_outliers,
        mbd, mei,
    })
}
```

---

## Test Patterns

### Synthetic Data for Tests

```rust
// Magnitude outlier: add large constant offset to a curve
fn with_magnitude_outlier(n: usize, m: usize, idx: usize, shift: f64) -> FdMatrix {
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            data[(i, j)] = (2.0 * PI * t[j]).sin() + if i == idx { shift } else { 0.0 };
        }
    }
    data
}

// Shape outlier: add high-frequency component to one curve (same mean, different shape)
fn with_shape_outlier(n: usize, m: usize, idx: usize) -> FdMatrix {
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            data[(i, j)] = if i == idx {
                (8.0 * PI * t[j]).sin() * 0.5  // high frequency, ~same mean
            } else {
                (2.0 * PI * t[j]).sin()
            };
        }
    }
    data
}
```

### Required Test Assertions

| Detector | Test | Assertion |
|----------|------|-----------|
| `tvdmss` | magnitude outlier at idx=n-1 with shift=10 | `result.magnitude_outliers.contains(&(n-1))` |
| `tvdmss` | shape outlier via high-freq curve | `result.shape_outliers.contains(&shape_idx)` |
| `tvdmss` | empty data | returns `FdarError::InvalidDimension` |
| `tvdmss` | n=2 | returns `FdarError::InvalidDimension` |
| `muod` | magnitude outlier (large shift) | `result.magnitude_outliers.contains(&idx)` |
| `muod` | amplitude outlier (large scale factor) | `result.amplitude_outliers.contains(&idx)` |
| `muod` | shape outlier (high-freq curve) | `result.shape_outliers.contains(&idx)` |
| `muod` | n=2 | returns `FdarError::InvalidDimension` |
| `seq_transform` | T0 on data with obvious outlier | per-transform outliers non-empty |
| `seq_transform` | union outliers = union of per-transform | `union_outliers == flatten(per_transform).dedup().sort()` |
| `seq_transform` | D1 with m=1 | returns `FdarError::InvalidDimension` |
| `depthgram` | magnitude outlier | `result.magnitude_outliers.contains(&idx)` |
| `depthgram` | shape outlier | `result.shape_outliers.contains(&idx)` |
| `depthgram` | mbd_mei_d == mbd_mei_t` for p=1` | exact equality for univariate case |
| `depthgram` | n=1 | returns `FdarError::InvalidDimension` |

---

## Environment Availability

Step 2.6: No external dependencies beyond Rust toolchain. All computation uses already-vendored
crates. Build note from MEMORY.md: `export TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required
before `cargo build` to avoid /tmp exhaustion.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`) |
| Config file | none (inline `#[cfg(test)]` in `outliers.rs`) |
| Quick run command | `cargo test -p fdars-core outliers -- --test-thread=4` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OUT-01 | tvdmss flags magnitude outlier | unit | `cargo test -p fdars-core tvdmss -- --test-threads=4` | ❌ Wave 0 |
| OUT-01 | tvdmss flags shape outlier | unit | same | ❌ Wave 0 |
| OUT-01 | muod flags all three outlier types | unit | `cargo test -p fdars-core muod` | ❌ Wave 0 |
| OUT-01 | sequential_transform_outliers per-transform + union | unit | `cargo test -p fdars-core seq_transform` | ❌ Wave 0 |
| OUT-01 | depthgram computes 6 index vectors + flags | unit | `cargo test -p fdars-core depthgram` | ❌ Wave 0 |
| OUT-01 | error paths (empty/n<3/mismatched dims) | unit | `cargo test -p fdars-core -- --test-threads=4` | ❌ Wave 0 |
| OUT-01 | clippy clean incl. test/bench code | gate | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | runs on existing code |

### Sampling Rate

- Per task commit: `cargo test -p fdars-core outliers -- --test-threads=4`
- Per wave merge: `cargo test -p fdars-core --features linalg,parallel`
- Phase gate: Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] All new `#[cfg(test)] mod tests { ... }` blocks in `outliers.rs` — added inline as part of each plan

---

## Security Domain

No user-supplied data reaches the filesystem, network, or shell. All inputs are numeric `f64`
matrices validated with `FdarError::InvalidDimension` guards. No applicable ASVS categories.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Phase 28 TVD depth as standalone | TVD pinned as `TvdMssResult{tvd,mss}` for Phase 29 consumption | Phase 28 (2026-08-19) | `tvdmss` can call `total_variation_depth_1d` directly |
| No depthgram | Univariate depthgram as 3-representation index pair | Phase 29 | Closes roahd gap |
| No MUOD | Fast-MUOD via pointwise mean regression | Phase 29 | Closes fdaoutlier MUOD gap |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | MUOD's C++ pairwise covariance averaging over all j produces same indices as regression against the pointwise mean (Fast-MUOD equivalence) | Algorithm Pinning §MUOD | If wrong, shape/amplitude/magnitude indices may differ numerically from R baseline for small n; document divergence in rustdoc |
| A2 | `quantile_sorted` (linear interpolation) produces sufficiently close Q1/Q3 to R's `quantile(type=7)` for IQR fence | Standard Stack §IQR Fence Helper | Minor numeric difference in cutoff; unlikely to affect outlier set for typical n > 20 |
| A3 | Depthgram shape outlier flag is `dist > upper_iqr_fence` (upper tail) not `< lower_fence` | Algorithm Pinning §depthgram | If the direction is wrong, shape vs non-shape swap. R code uses `dist > lim` confirmed in source. |
| A4 | MUOD `tangent` cutoff method can be omitted (marked as future enhancement) without blocking test correctness | Algorithm Pinning §muod | No risk — `boxplot` is the default and covers the locked decision |
| A5 | For p=1, all three depthgram representations are equal | Algorithm Pinning §depthgram | If wrong, time-wise and corr-wise would need separate computation; but R source confirms they collapse for p=1 since the corr flip only triggers when `corr.mei[i] == -1` for some dimension i>1 |

**If this table is empty:** N/A — all A1–A5 are documented.

---

## Open Questions

1. **Should `SeqTransformOutliers` carry the transformed data matrices?**
   - What we know: R's `seq_transform` optionally returns transformed data (`save_data=TRUE`)
   - What's unclear: adding `Vec<FdMatrix>` to the struct increases memory footprint significantly
   - Recommendation: Omit transformed data from the result struct; it is out of scope per the locked decision (numeric outputs only). The planner should not add `transformed_data` fields.

2. **MUOD `tangent` cutoff method**
   - What we know: the tangent method requires smooth-spline fitting (no spline dep in the crate for this shape)
   - What's unclear: whether users will need it
   - Recommendation: Implement only `boxplot` cutoff for this phase; add `tangent` to backlog. The `cut_method` param can be a `MuodCutMethod` enum with only `Boxplot` variant for now, marked `#[non_exhaustive]`.

3. **MUOD index normalization: `abs(index - benchmark)`**
   - What we know: R returns `abs(indices - c(1, 0, 1))` where benchmark shape=1, magnitude=0, amplitude=1
   - What's unclear: whether the absolute value is appropriate or signed deviations are needed
   - Recommendation: Use `abs()` as R does; add rustdoc note that positive index = more outlying.

---

## Sources

### Primary (HIGH confidence — verified from official R source code)

- [roahd depthgram.R source via GitHub API](https://github.com/astamm/roahd/blob/master/R/depthgram.R) — complete algorithm, three representations, outliergram flags
- [fdaoutlier tvdmss_new.R via GitHub API](https://github.com/otsegun/fdaoutlier/blob/master/R/tvdmss_new.R) — two-stage procedure, exact R code
- [fdaoutlier muod.R via GitHub API](https://github.com/otsegun/fdaoutlier/blob/master/R/muod.R) — index computation, boxplot cutoff
- [fdaoutlier cor_cov_blockwise.cpp via GitHub API](https://github.com/otsegun/fdaoutlier/blob/master/src/cor_cov_blockwise.cpp) — MUOD C++ index formulas
- [fdaoutlier sequential_transformations.R via GitHub API](https://github.com/otsegun/fdaoutlier/blob/master/R/sequential_transformations.R) — exact transforms, base detector

### Secondary (MEDIUM confidence — official docs cross-verified with source)

- [fdaoutlier tvdmss CRAN docs](https://search.r-project.org/CRAN/refmans/fdaoutlier/html/tvdmss.html) — parameters, return values
- [fdaoutlier seq_transform CRAN docs](https://rdrr.io/cran/fdaoutlier/man/seq_transform.html) — transform taxonomy
- [roahd depthgram reference docs](https://astamm.github.io/roahd/reference/depthGram.html) — return value structure

### Tertiary (LOW confidence — training knowledge / secondary web search)

- Dai et al. (2020) doi:10.1016/j.csda.2020.106960 — sequential transform paper (not read directly)
- Huang & Sun (2019) — TVD+MSS paper (reconstruction of algorithm, not read directly)
- Aleman-Gomez et al. (2022) doi:10.1002/sim.9342 — depthgram paper (not read directly)

---

## Project Constraints (from CLAUDE.md)

- **Rust edition 2021, MSRV 1.81.0** — no unstable features; no `faer` without `linalg` feature gate
- **Additive/non-breaking** — zero changes to existing public signatures; `#[non_exhaustive]` on all new structs
- **No new crate dependency** — all computation uses existing vendored crates
- **Clippy gate:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must stay green (CI lints test/bench code)
- **TMPDIR:** `export TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for builds/doctests
- **Column-major FdMatrix:** all matrix access via `data[(i, t)]` operator, never raw pointer arithmetic
- **Result<T, FdarError>** on all public functions; inline `#[cfg(test)] mod tests` per file
- **#[must_use]** on all expensive public computation functions (74+ in codebase)
- **Conditional serde:** `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` on all public result types
- **rayon-gated parallelism:** wrap O(n) outer loops in `iter_maybe_parallel!`; no direct `rayon::` calls
- **Per-thread RNG seeding** (not needed for any of the four deterministic detectors — no randomness involved)

---

## Metadata

**Confidence breakdown:**
- tvdmss algorithm: HIGH — R source read directly; formula confirmed
- muod algorithm: MEDIUM — R source read; C++ internals understood; Rust simplification ASSUMED equivalent
- seq_transform algorithm: HIGH — R source read directly
- depthgram algorithm: HIGH — roahd R source read directly; univariate reduction ASSUMED from p=1 analysis
- Integration points (lib.rs, outliers.rs): HIGH — files read in this session

**Research date:** 2026-08-19
**Valid until:** 2026-11-19 (fdaoutlier stable; roahd stable; ~90-day window)
