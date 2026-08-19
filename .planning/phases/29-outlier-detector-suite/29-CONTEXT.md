# Phase 29: Outlier-Detector Suite - Context

**Gathered:** 2026-08-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Add the four `fdaoutlier`/`roahd` outlier detectors fdars was missing, as **numeric outputs**
(flagged indices and/or per-curve scores — no plotting/rendering), reusing the DEPTH-01 depths
(Phase 28) and the existing MS-plot / outliergram machinery, **without changing any existing
outlier code or public signatures**. The four detectors:

- `tvdmss` — TVD + MSSI two-stage detector (magnitude via TVD, shape via MSS)
- `muod` — Massive Unsupervised Outlier Detection (per-curve magnitude/amplitude/shape indices)
- `sequential_transform_outliers` — sequential-transformation detection (T0/T1/T2)
- `depthgram` — the depthgram statistic (numeric coordinates + flags)

**In scope:** numeric detector outputs, `Result`-returning public fns in `outliers.rs`, crate-root
re-exports, inline `#[cfg(test)]` tests. **Out of scope:** any plotting/rendering of MS-plots /
outliergrams / depthgrams / flags; `fdaPOIFD` partially-observed detectors (deferred); new crate
dependency; changes to existing `magnitude_shape_outlyingness` / `outliergram` / DEPTH-01 signatures.
</domain>

<decisions>
## Implementation Decisions

### API Shape & Return Types
- Each detector is a `Result<T, FdarError>`-returning public fn in `outliers.rs`, crate-root re-exported.
- Dedicated result structs (derive `Debug, Clone, PartialEq`, `#[non_exhaustive]`, conditional serde):
  - `TvdMssOutliers { magnitude_outliers: Vec<usize>, shape_outliers: Vec<usize>, tvd: Vec<f64>, mss: Vec<f64> }`
  - `MuodResult { magnitude/amplitude/shape indices + flagged sets }`
  - `DepthgramResult { index-pair coordinates across 3 representations + outliers }`
  - `sequential_transform_outliers` → struct with per-transform flags + a unioned outlier set
- Small config structs / optional params only where a cutoff factor is genuinely needed
  (tvdmss cutoff, muod cutoff); sensible defaults matching the R baseline.
- Reuse-first: `tvdmss` builds on Phase 28 `total_variation_depth_1d` (`TvdMssResult`); reuse
  existing `outliergram` / `magnitude_shape_outlyingness` / `functional_boxplot` fence logic rather
  than reimplementing. **No new crate dependency.**

### tvdmss Method
- fdaoutlier two-stage: TVD flags magnitude outliers, then MSS flags shape outliers among the rest.
- Tunable boxplot-style tail cutoff (IQR factor, default 1.5) on the TVD/MSS statistics.
- Output both `magnitude_outliers` and `shape_outliers` index sets plus the `tvd`/`mss` vectors.
- Consumes Phase 28's pinned `TvdMssResult { tvd, mss }` interface directly.

### muod & Sequential-Transform
- `muod`: per-curve magnitude / amplitude / shape indices via regression of each curve on the
  pointwise mean (intercept ≈ magnitude, slope ≈ amplitude, residual/1−R² ≈ shape); flag each index
  via a boxplot cutoff.
- `sequential_transform_outliers`: fdaoutlier transforms T0 (raw), T1 (one-step normalization),
  T2 (successive differences); run a functional-boxplot / directional detector per transform.
- Output a struct with per-transform flags plus the unioned outlier set.

### depthgram & Testing
- `depthgram`: numeric only — the (MEI/MBD-style) index-pair coordinates across the three
  representations plus flagged outliers. **No rendering.**
- Pin the exact `muod` / `sequential_transform` / `depthgram` statistics from `fdaoutlier`/`roahd`
  during research; document any divergence from the R baseline in rustdoc (prior-milestone practice).
- Tests: inject magnitude outliers and shape outliers into a synthetic sample; assert `tvdmss`
  flags both classes and the other detectors return the expected outlier index sets within a
  documented tolerance; error-path coverage (empty / single-curve / mismatched dims / degenerate
  columns → `FdarError`) at every entry point.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/outliers.rs` — `outliergram` (`OutligramResult`), `magnitude_shape_outlyingness`
  (`MagnitudeShapeResult`), `detect_outliers_lrt`, `outliers_threshold_lrt*` — the MS-plot /
  outliergram machinery to reuse; the file to extend additively (~995 lines).
- `src/depth/tvd.rs` — `total_variation_depth_1d` → `TvdMssResult { tvd, mss }` (Phase 28, PINNED)
  — the tvdmss dependency.
- `src/depth/dispatch.rs` — `functional_depth` / `functional_boxplot` (fence logic reusable for
  cutoffs) + the 9 new `DepthMethod` variants (for depthgram's depth/epigraph indices).
- `src/matrix.rs` — column-major `FdMatrix`; `src/error.rs` — `FdarError`; `src/parallel.rs` —
  `iter_maybe_parallel!`.

### Established Patterns
- Column-major storage, `Result<T, FdarError>` public API, inline `#[cfg(test)] mod tests`,
  `#[non_exhaustive]` result structs with conditional serde, crate-root re-export in `lib.rs`
  (`pub use outliers::{...}` block ~line 431).

### Integration Points
- `src/outliers.rs` — add the 4 detectors + their result structs.
- `src/lib.rs` (~line 431) — extend the `pub use outliers::{…}` re-export block.

</code_context>

<specifics>
## Specific Ideas

- `tvdmss` is the reason Phase 28 pinned `TvdMssResult`; consume `total_variation_depth_1d`
  directly for both `tvd` and `mss`.
- Match R baselines by **capability**, not R's exact signatures.

</specifics>

<deferred>
## Deferred Ideas

- Any plotting/rendering of MS-plots, outliergrams, depthgrams, or outlier flags (numeric only).
- `fdaPOIFD` partially-observed-data detectors (explicitly deferred this milestone).

</deferred>
