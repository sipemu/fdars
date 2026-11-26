---
quick_id: 261126-lj0
issue: 73
status: complete
commits:
  - 31900859 feat(quick-261126-lj0): multivariate DTW family + shape-DTW (refs #73)
  - 083fa1f5 feat(quick-261126-lj0): exact DTW k-NN + LB_Improved / LB_Enhanced (closes #73)
---

# Summary — #73 DTW family gaps (all four parts delivered)

## Public API added (additive; no signature or numeric change to existing items)
1. **Multivariate series** (`d × m` `FdMatrix` per series = aeon `(n_channels, n_timepoints)`;
   collections `&[FdMatrix]`, unequal lengths allowed except k-means):
   - `dtw_distance_multivariate`, `dtw_path_multivariate`, `dtw_pairwise_distance_multivariate`
     (cost `‖a − b‖₂^p`; p = 2 = aeon/tslearn dependent squared Euclidean).
   - `dtw_barycenter_averaging_multivariate` → `DbaMultivariateResult`; `DbaInit::CustomMultivariate`.
   - `dtw_kmeans_multivariate` → `DtwKMeansMultivariateResult` (+ `predict`);
     `DtwKMeansInit::CentroidsMultivariate`.
   - `<ddtw|wdtw|wddtw|adtw|erp|edr|twe|msm|lcss>_{distance,pairwise_distance,cost_matrix,alignment_path}_multivariate`.
2. **Shape DTW**: `shape_dtw_{distance,pairwise_distance,cost_matrix,alignment_path}` (+ `_multivariate`),
   `ShapeDtwDescriptor::{Identity, Paa{num_intervals}, Dwt{num_levels}, Slope{num_intervals}, Derivative}`, `reach`.
3. **k-NN**: `dtw_k_nearest_neighbors` → `DtwKNearestNeighbors`.
4. **Lower bounds**: `lb_improved(x, y, p, window)` (p ≥ 1), `lb_enhanced(x, y, p, window, v)`.

Re-exported from `metric`, crate root and (DTW/DBA/k-means/k-NN entry points) `prelude`.

## Implementation notes
- Elastic recipes moved from `&[f64]` to a channel-interleaved `Ser { v, d }` view; `sqe` / `euc`
  accumulate from `0.0` like aeon's `_univariate_squared_distance`, so `d = 1` performs exactly the
  old floating-point ops. MSM independent = per-channel lattices summed in channel order (aeon);
  dependent = vector split/merge cost. ERP's scalar `g` broadcast to all channels (no `g_arr`).
- DTW/DBA/k-means internals take a channel count `d` (`*_flat` helpers); `d = 1` dispatches to the
  untouched univariate DP. DBA mean init = tslearn `nanmean` (variable lengths).
- Shape DTW reproduces aeon's raw-cost traceback including numpy's `-1` wrap-around at the borders.
- k-NN: `Best` replaced by a `TopK` (k = 1 ≡ old best-so-far); cutoff = k-th best distance.
- LB_Enhanced: `V_eff = min(v, ⌊(min(n,m) − 1)/2⌋)` (reference cap); LB_Keogh when `V_eff = 0`
  (the Java reference double-counts corner cells there). Both bounds use the window's admissible
  rows/cols, so they hold for every `DtwWindow` and unequal lengths.

## Validation
- aeon 1.6.0 (`~/.cache/fdars-golden-venv`): multivariate DTW + all 9 elastic distances
  (distance, cost matrix, path, pairwise self/cross on unequal lengths; window None and 0.3),
  shape DTW (identity natively; PAA/DWT/Slope/Derivative computed with **sktime 1.3.0**
  transformers — installed into the venv — and passed to aeon as `transformed_x/_y`), plus
  descriptor unit goldens. Tolerance 1e-10 (1e-12 descriptors).
- tslearn 0.10.0: multivariate `dtw_path` (incl. Sakoe-Chiba), `dtw_barycenter_averaging`
  (mean init, radius 1, size 9 + weights, variable length), `TimeSeriesKMeans` (fixed init:
  labels, n_iter, centroids). Tolerance 1e-9.
- LB_Improved: Lemire's `lbimproved` C++ (`include/dtw.h`, compiled; p = 1) and Tan et al.'s
  `LbEnhanced` Java `LbImproved` (p = 2); LB_Enhanced: Java `LbEnhanced` (p = 2) — 12 pairs,
  agreement 1e-13. Invariants `LB_Keogh ≤ LB_Improved ≤ DTW`, `LB_Enhanced ≤ DTW` over > 2000
  (pair, window, p) combos incl. unequal lengths, Itakura, aeon windows, p ∈ {0.5,1,1.5,2,3}.
- k-NN: bit-exact vs brute force (stable sort) for k ∈ {1,2,3,7,45}, 10 windows, p ∈ {0.5,1,2},
  ties (duplicates → lowest index); 1-NN indices/distance bits/stats pinned from the pre-change code.
- Bit-identity: every elastic distance, DTW, DBA (4 configs) and k-means (k-means++ restarts and
  explicit init) with `d = 1` vs univariate; existing goldens unchanged.

## Gates
- `cargo fmt --all -- --check`: clean.
- `cargo test -p fdars-core --lib metric::` 214 passed; `--lib dtw_kmeans` 7 passed;
  doc tests `multivariate` (45), `shape_dtw` (8), `dba`, `dtw_kmeans`, `elastic_family` (80),
  `metric::dtw::`, `dtw_lb` (10): all pass; integration `validate_against_r dtw` 12 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`: clean.
- `cargo check -p fdars-core --features serde`: clean.
- pmat cognitive complexity: no new function > 23 (only pre-existing `soft_dtw::corrected_oracle_gradient`).

## Deviations / notes
- aeon's windowed `lcss_alignment_path` does not terminate on our golden data (aeon loops on an
  out-of-band cell, the documented deviation of #57); that one path golden is skipped (distance
  and cost matrix still checked).
- aeon only ships the identity descriptor; the other descriptors follow sktime's `ShapeDTW`
  transformers (HOG1D and compound not implemented).
- `dtw_kmeans_multivariate` requires equal-length series (DBA itself accepts variable lengths).
- ERP multivariate broadcasts the scalar `g` (aeon `g_arr` per-channel vector not exposed).
- `.planning/STATE.md` not touched (orchestrator override); not pushed.
