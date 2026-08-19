# Phase 28: Depth-Measure Long Tail - Context

**Gathered:** 2026-08-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Add every canonical **batch univariate functional depth measure** that `roahd`/`fdaoutlier`
expose but fdars was missing, over the column-major `FdMatrix`, each selectable through the
existing unified `DepthMethod` dispatcher — **without changing any existing depth code or
public signatures**. The nine measures:

- Half-region depth (HRD) and modified half-region depth (MHRD)
- Hypograph index (HI), modified-hypograph index (MHI), un-modified epigraph index (EI)
- Extremal depth
- Extreme-rank-length depth (ERL)
- L∞ depth
- Total-variation depth with MSSI (TVD+MSSI)

**In scope:** numeric per-curve depth values; `DepthMethod` variant per measure; crate-root
re-exports; inline `#[cfg(test)]` tests. **Out of scope:** streaming/online depth variants
(fdars strength U-5), any plotting/rendering, new crate dependencies, changes to existing
`DepthMethod` variants or dispatcher signature.

</domain>

<decisions>
## Implementation Decisions

### API Surface & Naming
- Each of the 9 new measures is a `Result<Vec<f64>, FdarError>`-returning public function with
  dimension/parameter validation at entry (satisfies success criteria #1 and #4). This diverges
  from the older bare-`Vec<f64>` `_1d` functions, which validate only in the dispatcher — new
  code carries its own guards.
- Two-matrix self-depth signature `fn <measure>_1d(data_obj: &FdMatrix, data_ori: &FdMatrix)`
  matching the existing `band_1d`/`fraiman_muniz_1d`/`modified_epigraph_index_1d` convention;
  the dispatcher passes `(data, data)`.
- Group measures by family into new files under `src/depth/`: `half_region.rs` (HRD/MHRD),
  `hypo_epi.rs` (HI/MHI/EI), `extremal.rs`, `erl.rs`, `linf.rs`, `tvd.rs`. Avoids 9 tiny files;
  matches per-file module convention.
- Re-export all 9 new functions at the crate root (`lib.rs` `pub use depth::{…}`).

### DepthMethod Dispatcher Integration
- Add one **parameter-free** `DepthMethod` variant per measure: `HalfRegion`,
  `ModifiedHalfRegion`, `HypographIndex`, `ModifiedHypographIndex`, `EpigraphIndex`, `Extremal`,
  `ExtremeRankLength`, `LInfinity`, `TotalVariation`. MSSI is intrinsic to TVD (no tuning knob).
- Existing `DepthMethod` variants and the `functional_depth` / `functional_boxplot` signatures
  stay **untouched** (enum is `#[non_exhaustive]`, so additions are non-breaking).
- Add the **un-modified** epigraph index (EI) plus hypograph index (HI) and modified-hypograph
  index (MHI) to complement the already-shipped `modified_epigraph_index_1d` (MEI).
- Dispatch validation mirrors the existing `n==0 || m==0` guard, adding a `≥2 curves` check for
  any measure whose definition needs a reference band/region (as `Band`/`ModifiedBand` already do).

### Reference-Definition Pinning
- Per-measure R baseline, documented in each function's rustdoc:
  - HRD/MHRD, HI/MHI/EI, ERL → `roahd`
  - Extremal depth, L∞ depth, TVD+MSSI → `fdaoutlier`
- TVD+MSSI follows Huang & Sun (2019): total-variation depth (magnitude) combined with the
  Modified Shape Similarity Index (shape); the exact statistic + MSSI construction is written
  into the rustdoc.
- Tie handling reuses MEI's `<=` comparison with the 0.5 tie adjustment (matches `roahd`),
  uniformly across the index measures.
- Where fdars diverges from the R reference, note it explicitly in rustdoc (prior-milestone
  practice — pin the exact statistic during planning/research).

### Testing & Validation
- Synthetic fixtures: a clearly-central curve plus injected magnitude and shape outliers; assert
  the central curve ranks deepest and the outliers shallowest for each measure (criterion #3).
- Reference matching by rank/ordering assertions plus a few hand-computed small-case values
  within `1e-9` — no R runtime dependency for cross-validation.
- Error-path coverage per entry point: empty matrix, single curve, mismatched argvals vs values,
  degenerate columns → `FdarError`, never panic (criterion #4).
- Parallelism follows existing depth code: parallelize the O(n²) measures with
  `iter_maybe_parallel!` (as band depth does); keep the simple O(n·m) measures sequential.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/depth/dispatch.rs` — `DepthMethod` enum (`#[non_exhaustive]`, `Copy`) + `functional_depth`
  self-depth entry point with `n==0||m==0` and `≥2-curve` guards; the pattern to extend.
- `src/depth/band.rs` — `band_1d`, `modified_band_1d`, `modified_epigraph_index_1d` (MEI, with
  the `<=` + 0.5 tie convention to reuse for HI/MHI/EI).
- `src/depth/fraiman_muniz.rs`, `modal.rs`, `rpd.rs`, `spatial.rs` — existing `_1d`/`_2d`
  measure files showing the two-matrix `(data_obj, data_ori)` convention.
- `src/matrix.rs` — column-major `FdMatrix` with `nrows`/`ncols`, `data[(i,t)]` indexing, and
  efficient row helpers (`row_to_buf`, `row_dot`, `row_l2_sq`).
- `src/parallel.rs` — `iter_maybe_parallel!` etc. for feature-gated rayon.
- `src/error.rs` — `FdarError` with `InvalidDimension` / `InvalidParameter` variants.

### Established Patterns
- Column-major storage, `Result<T, FdarError>` public API, `#[cfg(test)] mod tests` inline,
  `#[non_exhaustive]` on public result enums/structs, per-file module split with explicit
  `pub use` in `depth/mod.rs` and crate-root re-export in `lib.rs`.

### Integration Points
- `src/depth/mod.rs` (line ~24–33) — add `pub use` for each new file.
- `src/depth/dispatch.rs` — add the 9 enum variants + match arms.
- `src/lib.rs` (line ~422) — extend the `pub use depth::{…}` re-export block.

</code_context>

<specifics>
## Specific Ideas

- The TVD+MSSI measure (`TotalVariation`) is the shared dependency for Phase 29's `tvdmss`
  outlier detector — its public interface must be stable and reusable. Pin the TVD/MSSI function
  signature deliberately so Phase 29 consumes it without change.
- Match R baselines by **capability**, not by R's exact function signatures.

</specifics>

<deferred>
## Deferred Ideas

- Streaming/online depth variants (out of scope — fdars strength U-5, batch measures only).
- Any plotting/rendering of depth regions or boxplots (numeric outputs only this milestone).

</deferred>
