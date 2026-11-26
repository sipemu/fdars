# Changelog

All notable changes to `fdars-core` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/). Under the SemVer 0.x rule, breaking
changes ship in MINOR bumps. **v0.41.0 is an explicitly breaking API-stabilization
release** — it removes long-deprecated forms, seals a couple of internals, and
tightens the public surface ahead of a future 1.0 (see [0.41.0] below; breaking is
API *shape* only — no numeric/behavioral change). Earlier entries in this span
(v0.40.0 and prior) remained additive and non-breaking with respect to public API
signatures and dependencies, with one intentional exception: the v0.40.0 soft-DTW
barycenter correction changes convergence behavior intentionally (see [0.40.0]
Changed below).

## [Unreleased]

### Added
- **Localised group differences (GH #76).** `alignment::elastic_local_difference_test`
  aligns all curves label-free to one Karcher mean (`karcher_mean_with_config`,
  `WarpOptimConfig`), then runs a local two-sample test on the aligned curves
  (amplitude) and on the warps (phase: `γ(t) − t` or `ψ = √γ'`), returning grid
  regions `[t_start, t_end]` with direction (A above/below B; A earlier/later via
  `LocalDifferenceTest::timing`) and adjusted p-values (ITP, default) or the band
  (SCB). Config `LocalDifferenceConfig`; result `LocalDifferenceResult`.
- `inference::scb_two_sample_band` → `ScbTwoSampleBand` exposes the difference band
  (`estimate`, `lower`, `upper`, `half_width`, `critical_value`) and its regions;
  `scb_two_sample_test` is unchanged (bit-identical, shares the band core).
- `inference::itp_pointwise_pvalues` (`ItpDomainMapping::{Dominant, Support}`) maps
  B-spline ITP adjusted p-values to the grid (fdatest's nearest-component
  convention on fdars's basis); `inference::itp_two_pop_regions`
  (`ItpRegionConfig`, `ItpRegionsResult`) returns directed regions with
  adjusted p < α. ITP closure validated against R `fdatest` 2.1.1 `ITP2bspline`
  (`pval.matrix` and `corrected.pval` reproduced exactly).
- New shared module `regions` (`DifferenceRegion`, `difference_regions`);
  `significant_regions`, `significant_regions_from_se`, `SignificantRegion`,
  `SignificanceDirection` moved there and re-exported unchanged from `explain`
  (these types now also derive serde under the `serde` feature).

### Fixed
- `inference::itp_two_pop` permutation null relabelled only group B (group A kept
  its observed rows), inflating permutation statistics and destroying power; both
  groups are now relabelled as in `fdatest` (`coeff[sample(n), ]`). The B-spline
  components whose basis function vanishes on the grid (two per extended-knot
  basis; identically-zero coefficients) are now excluded from the interval-wise
  closure and reported with adjusted p-value 1.0 — they carried no information
  and diluted every interval containing them. `itp_two_pop` numeric results
  change accordingly (intentional; Fourier ITP is affected only by the
  permutation fix).

## [0.48.0] - 2026-11-26

**Alignment & registration release.** Closes the alignment gap analysis backlog
(GH #42–#62, tracking #63) and its caveat follow-ups (GH #64–#75). Additive and
non-breaking with respect to public API signatures (new fields only on
`#[non_exhaustive]` configs/results; existing defaults produce bit-identical
results, proven by hash tests recorded on the pre-change code); no new crate
dependency. One intentional numeric change: the Bayesian group alignment
template-band centring fix (GH #64, see *Fixed* below).

Highlights:
- **Elastic engine:** Riemannian L-BFGS warp optimiser and in-objective penalties
  (#42/#43), configurable DP neighbourhood / DP2 cost / `scl` refinement (#49),
  exact DP pruning now on by default (~1.4–1.6× faster, bit-identical; #66),
  RBFGS for curve shapes, N-d, landmarks, robust Karcher, closed curves, band
  support and a unified λ scale (#65/#67).
- **Registration methods:** iterative and landmark shift (#46), Ramsay–Li
  continuous registration incl. periodic (#47/#72), affine/dilation warps and
  warp classes in `align_cluster_fd` (#50), exponential-family registration
  with variational-EM and two-step GFPCA (#60), Tang–Müller pairwise
  synchronisation (#61), sparse/irregular elastic curves (#53).
- **Shape analysis:** R^d curve shape space with rotation/scale/closed contours
  (#48), downstream TSRVF/depth/boxplot/outliers/CI (#54), general (a,b)
  elastic metrics incl. closed curves and the exact piecewise-linear distance
  (#59/#68), fdasrvf-compatible model-based shape CI (#71).
- **Inference & UQ:** Bayesian group alignment (pCN + infHMC, parallel
  tempering; #52/#64/#69), split-conformal amplitude/phase sets for partially
  observed curves incl. simultaneous bands, bandwidth tuning and fragments
  (#51/#72), amplitude/phase PERMANOVA test (#62), elastic boxplots and DP-means
  (#55).
- **DTW family:** warping paths, windows, soft-DTW alignment matrix, GAK
  triangular kernel (#44), DBA and DTW k-means (#45), the aeon elastic-distance
  family (#57), lower bounds + EAPrunedDTW + exact k-NN (#58/#73), multivariate
  series and shape-DTW (#73).
- **Quality:** rustdoc `-Dwarnings` clean, pmat CI pin 3.45.1 with 0 violations
  (#75), fdasrvf goldens regenerated with 2.6.0 (#74).

Detailed per-item entries follow.


**fdasrvf golden values regenerated with 2.6.0 (GH #74 ALN-CAV).** Tests only; no
library code or numeric change.

### Changed

- The R fdasrvf goldens of the curve-shape depth/box plot (#54) and SRSF smoothing (#56)
  tests, first generated with fdasrvf 2.5.0, are regenerated with CRAN fdasrvf 2.6.0 (the
  >= 2.5.1 pin). `curvebox_data` and `smooth.data` are bit-identical; `curve_depth` moved
  (amplitude <= 3.5e-4, phase up to 1.2e-2) through 2.5.1's DP changes and unit-norm
  `gam_to_psi`. Phase depths now agree with fdars to 1.0e-2 (was 2.2e-2), so a numeric phase
  bound is added; the amplitude depth bound tightens 3e-3 -> 2.5e-3 and the box plot distance
  bound 1e-12 -> 5e-14. `SrsfSmoothing::SmoothingSpline` is now compared pointwise with the
  real 2.6.0 `f_to_srvf(smooth = TRUE)` (L2 distance 2.7e-3).

**pmat version drift (GH #75 ALN-CAV).** Internal refactor + CI tooling only; no public API
or numeric change.

### Changed

- CI pmat pin bumped 3.18.0 → 3.45.1 so local and CI quality gates agree; the no-op
  `--fail-on-violation` flag is dropped (3.32+ exits non-zero on blocking violations by
  default) and the installed-version check compares only the first `pmat --version` line.
- Decomposed the 9 functions pmat 3.45 flagged for cognitive complexity
  (`ClassifFit::predict_from_scores`, `CorrelationMatrixSpace`/`SphericalSpace`
  `weighted_frechet_mean`, `Lfd::apply`, `MfpcaResult::project`, streaming BD,
  `RollingReference::push`, and a soft-DTW test oracle) into private helpers. Outputs are
  bit-identical: FNV-1a fingerprints of every output bit were recorded on the unmodified code
  before the refactor and are pinned in tests.

**Missing options for continuous registration and partial-curve conformal (GH #72 ALN-CAV).**
Opt-in; default results are bit-identical (FNV-hash tests pinned before the change).

### Added

- `ContinuousRegistrationConfig::periodic` (default `false`): R `register.fd(periodic = TRUE)`.
  A free time shift `δᵢ` is estimated jointly with `Wᵢ` (R's use of the first W coefficient);
  curves are registered as `yᵢ(hᵢ(t) + δᵢ)` on their periodic extension (natural cubic spline
  over three concatenated periods), with the exact shift gradient. The shift is returned in
  the new `ContinuousRegistrationResult::shift`; returned warps are `hᵢ + δᵢ` as R's `warpfd`.
  Golden-tested against R fda 6.3.0 (Fourier `yfd`, both criteria): criterion ≤ R's,
  shifts within 6e-4 of R's.
- `ContinuousRegistrationConfig::r_defaults()`: R's default W basis (2 linear B-splines,
  `lfd_order = 0`, `iterlim = 20`, `crit = 2`); golden-tested against
  `register.fd(..., WfdParobj = NULL)`. `ContinuousRegistrationConfig::default()` is unchanged.
- Partial-curve conformal (`conformal::partial`): **simultaneous amplitude bands**
  (`PartialConformalConfig::band = PartialAmplitudeBand::Simultaneous`) from the max-type score
  `sup_t |Y − Ŷ|/σ(t)` with `σ` from `PartialBandScale::{Constant, TrainingSd, Custom}`; the band
  `ĉ ± Rσ` provably contains the full-conformal set, so `P(Y(t) ∈ band ∀t) ≥ 1 − α` in finite
  samples (`PartialConformalPrediction::amplitude_radius`). Pointwise stays the default.
- **Held-out bandwidth tuning**: `PartialConformalConfig::tuning_fraction` reserves a seeded part of
  the calibration curves, on which `h` minimising the mean leave-one-out band width is selected
  (candidates `bandwidth_candidates` or distance quantiles); `partial_conformal_select_bandwidth`
  exposes the selector and `PartialBandwidthSelection` reports it. Since `h` depends on the tuning
  curves only, coverage stays exact (the paper's same-data Liang–Zhu–Foygel Barber rule has no
  closed form for the affine full-conformal scores). On the simulation model it roughly halves the
  band width (0.30 → 0.16 at α = 0.2, n = 60) at nominal coverage.
- **Fragmented / sparse observation**: `partial_conformal_predict_observed` with
  `PartialObservation::{Prefix, Fragments, Sparse}` — fragments use the paper's length-weighted
  product distance, sparse points the Euclidean distance of (interpolated) values; every fragment
  is located in the template with `elastic_partial_match` (`fragment_registrations`).
- New model fields `tuning_indices`, `tuning_data`, `tuning_aligned`, `training_sd`. Validated by
  exact leave-one-out rotation counts (simultaneous, fragments, sparse), a brute-force containment
  check of the simultaneous bound, a public-API recomputation of the tuning criterion and seeded
  Monte-Carlo coverage (simultaneous 0.81–0.87, sparse/tuned pointwise 0.81–0.83 at 1 − α = 0.8).

**Parallel tempering for Bayesian group alignment (GH #69 ALN-CAV).** Opt-in; default pCN
and infHMC results are bit-identical (FNV-hash tests pinned before the change).

### Added

- `BayesianGroupConfig::n_temperatures` (default 1 = off) and `pt_min_inv_temp` (default 0.1):
  parallel tempering inside each chain of `bayesian_align_group` (either sampler). Replica `k`
  targets `L^{β_k}·prior`, `β_k = β_min^{k/(K−1)}`; every sampler step (pCN ratios, infHMC
  kinetic scale, Gibbs template and σ²) is tempered, adjacent replicas exchange states with
  exact Metropolis–Hastings acceptance (even/odd pairing) and only `β = 1` is reported.
  `BayesianGroupChain::swap_acceptance` reports per-pair exchange rates. On the 5-curve
  problem where plain infHMC leaves a chain in a poor mode (σ² 0.66 vs 0.22, R̂(σ²) 3.4; stuck
  for 4 of 6 seeds) 6 replicas put every chain in the good mode (R̂(σ²) ≤ 1.002) at ≈ 3× the
  run time, with equal or better ESS per draw. Validated against quadrature of the tempered
  warp conditional and against plain infHMC / long pCN posterior moments.

### Documentation

- `bayesian_group` module docs: "Multimodality" — reading `rhat_sigma2`, `mode_spread`,
  `multimodal_curves` and per-chain log-likelihoods, when a chain may be discarded, and how to
  size the tempering ladder from `swap_acceptance`. Independence / pCN jumps mixed into infHMC
  were evaluated and not added (unreliable escape, see the quick-task summary).

**DTW family gaps (GH #73 ALN-CAV).** Additive only: every existing function returns
bit-identical results (the multivariate variants with one channel reproduce the univariate
ones bit-for-bit, pinned by tests).

### Added

- **Multivariate (multichannel) series** for the DTW family — one `d × m` `FdMatrix` per
  series (channels as rows, aeon's `(n_channels, n_timepoints)` layout), collections as
  `&[FdMatrix]`, with aeon / tslearn's dependent semantics (squared Euclidean cost across
  channels): `dtw_distance_multivariate`, `dtw_path_multivariate`,
  `dtw_pairwise_distance_multivariate`; `dtw_barycenter_averaging_multivariate`
  (`DbaMultivariateResult`, new `DbaInit::CustomMultivariate`, variable-length series with
  tslearn's `nanmean` initialisation); `dtw_kmeans_multivariate`
  (`DtwKMeansMultivariateResult` + `predict`, new `DtwKMeansInit::CentroidsMultivariate`);
  and `<name>_{distance,pairwise_distance,cost_matrix,alignment_path}_multivariate` for the
  whole aeon elastic family (DDTW, WDTW, WDDTW, ADTW, ERP, EDR, TWE, MSM with aeon's
  `independent` per-channel sum / dependent vector cost, LCSS). Golden-tested against aeon
  1.6.0 and tslearn 0.10.0 (`dtw_path`, `dtw_barycenter_averaging`, `TimeSeriesKMeans`).
- **Shape DTW** (Zhao & Itti 2018, aeon `shape_dtw_*`): `shape_dtw_distance`,
  `shape_dtw_pairwise_distance`, `shape_dtw_cost_matrix`, `shape_dtw_alignment_path` (+
  `_multivariate`), with `reach` and `ShapeDtwDescriptor` — `Identity` (aeon's default) plus
  sktime's `Paa`, `Dwt`, `Slope` and `Derivative` descriptors. Golden-tested against aeon
  1.6.0 (other descriptors via sktime 1.3.0 transformers fed to aeon as precomputed
  transformations).
- **Exact DTW k-NN**: `dtw_k_nearest_neighbors(queries, references, k, p, window)` →
  `DtwKNearestNeighbors` (per-query indices / distances ordered by `(distance, index)`), the
  LB_Keogh → LB_Kim → LB_Webb → EAPrunedDTW cascade of `dtw_nearest_neighbor` with the k-th
  best distance as cutoff; equals brute force exactly. `dtw_nearest_neighbor` is now its
  `k = 1` case — indices, distance bits and pruning statistics pinned unchanged.
- **DTW lower bounds** `lb_improved` (Lemire 2009, `p ≥ 1`) and `lb_enhanced` (Tan, Petitjean
  & Webb 2019, `v` bands), for every `DtwWindow` and unequal lengths; validated against
  Lemire's `lbimproved` C++ (p = 1) and Tan et al.'s `LbEnhanced` Java code (p = 2), and
  `LB_Keogh ≤ LB_Improved ≤ DTW`, `LB_Enhanced ≤ DTW` on thousands of random pairs.

**fdasrvf-compatible model-based shape CI for R^d curves (GH #71 ALN-CAV).** Additive; the
default `CurveShapeCiMethod::Bootstrap` path of `curve_shape_confidence_interval` is unchanged.

### Added

- `CurveShapeCiMethod { Bootstrap (default), ModelBased }` and the `CurveShapeCiConfig`
  fields `method`, `n_components` (fdasrvf `no`, 5), `n_samples` (`Nsamp`, 100) and
  `n_warp_components` (5). `ModelBased` ports fdasrvf 2.6.0 `shape_CI` / `sample_shapes`:
  Karcher median → zero-mean Gaussian on the leading shape-PCA directions of its shooting
  vectors → geodesic shooting (exact step for open curves, fdasrvf's 4-step projected and
  parallel-translated shooting for closed contours) → random warps from the horizontal FPCA
  of the median's warps (if `reparam`) and von Mises rotations (if `rotation`, planar only)
  → elastic box plot of the posed samples with `alpha = 1 − confidence_level`.
- `CurveShapeCiResult::model: Option<CurveShapeModelCi>` with fdasrvf's `Q1a` / `Q3a`
  (`q1a`, `q3a`), the box plot, the PCA, the sampled scores, unposed / posed samples and
  their SRVFs, warps and rotations. Validated against fdasrvf 2.6.0 on RNG-independent
  quantities (median shape, covariance eigenvalues) and on sample-distance moments.

**(a,b) elastic metrics: closed contours, exact piecewise-linear distance, branch search
(GH #68 ALN-CAV).** Additive; every SRVF result and every open-curve `(a,b)` result with the
default `branch_search = false` is bit-identical (the λ = 1 closure path dispatches to the
existing SRVF projector).

### Added

- `CurveShapeConfig::closed = true` now works with `CurveShapeMetric::Elastic { a, b }` in
  every `curve_shape_*` function (align/distance/matrix, Karcher mean/median, covariance,
  PCA, TSRVF, depth, outliers, box plot, CI). Closure condition in F-space: the curve rebuilt
  from `F` has displacement `(4b²)⁻¹ ∫ |F|² e^{iφ/λ}` (`φ` the lifted `arg F`, λ = a/(2b));
  transforms are projected onto `{∫ |F|² e^{iφ/λ} = 0}` by Gauss–Newton along its L²
  gradients `b_x = 2cos(φ/λ)F − λ⁻¹sin(φ/λ)iF`, `b_y = 2sin(φ/λ)F + λ⁻¹cos(φ/λ)iF`, and the
  Karcher tangent spaces use these normals. Seed search as for the SRVF.
- `ab_closure_residual(f, a, b)` and `ab_project_closed(f, a, b)` (public helpers; `a = 1,
  b = 1/2` = `closure_residual` / `project_closed_srvf`).
- `CurveShapeConfig::branch_search` (default `false`): with the elastic metric and
  `rotation = false`, also try the relative F-branches `e^{±2πiλ}` of `β2` (initial tangents
  on opposite sides of the `±π` cut) without rotating the curve.
- `ab_elastic_distance_pl(c1, c2, a, b)` / `ab_elastic_match_pl` → `AbPlMatching`: exact
  `(a,b)` elastic distance (modulo translation and reparametrisation) between piecewise-linear
  curves in R^d, with the optimal generalised matching — Bauer et al. (J. Nonlinear Sci. 2024)
  §4.1 via the Lahiri–Robinson–Klassen P/N-segment structure: vertex longest-path DP, P-segment
  rays swept by initial slope in straightened coordinates, `O(N1²N2²(N1+N2))`.

**Bayesian group alignment band calibration (GH #64 ALN-CAV).** One fix to the posterior
summaries of `bayesian_align_group`; the chains, warp / σ² summaries and every
`center_draws = false` result are bit-identical (FNV-hash tests pinned on the previous commit).

### Fixed

- `bayesian_align_group` with `center_draws = true` (the default): the re-centred template
  draws (`template_srsf*`, `template*`, `template_srsf_draws`, per-chain `template_srsf_mean` /
  `mode_template_srsf`) warped the *linear interpolant* of the template's grid values by the
  centring warp; its chord error biased every draw and shifted the template bands off the
  truth. The B-spline template is now evaluated exactly at the centring warp. On data simulated
  from the model (40 data sets × 4 chains, infHMC) template-band coverage at nominal 95% went
  from 0.899 ± 0.012 to 0.945 ± 0.008 (SRSF noise 0.01) and from 0.827 ± 0.014 to
  0.915 ± 0.011 (noise 0.1).

### Documentation

- `bayesian_group` module docs: "Calibration of the credible bands" — in-model coverage study
  (warps 0.92–0.96, template 0.91–0.95 for both samplers: the samplers are calibrated), the
  remaining misfit (pointwise interpolated-SRSF pseudo-likelihood under-estimates σ² by
  0.90× at noise 0.1, 0.74× at 0.3; function-space noise is not iid SRSF noise — 0.83–0.87
  warp coverage in the original study), the `beta0` floor `≈ 2β₀/(nT)` on σ², and why raw
  (`center_draws = false`) warp bands over-cover. A function-space likelihood option was not
  added. Coverage tests tightened (short f-noise study > 0.85, long > 0.80) and new in-model
  calibration tests (short + `#[ignore]` long study).

**RBFGS optimiser coverage: N-d, landmarks, robust Karcher, band semantics, λ scale (GH #67
ALN-CAV).** Additive and non-breaking: defaults (and the default RBFGS path) are bit-identical
(FNV-hash tests recorded before the change); no new crate dependency.

### Added

- `_with_config` entry points taking a `WarpOptimConfig` (RBFGS, the DP options `nbhd_dim` /
  `dp_scl` / `dp_cost` / `dp_prune`, `L2Psi`, band, λ conversion):
  `elastic_align_pair_nd_with_config`, `elastic_distance_nd_with_config`,
  `karcher_mean_nd_with_config` (R^d shared warp; RBFGS data term summed over coordinates),
  `elastic_align_pair_constrained_with_config` (landmarks: segmented DP, then per-segment RBFGS
  on the affinely mapped segment with the global penalty restricted to it — landmarks hit exactly),
  `karcher_median_with_config`, `robust_karcher_mean_with_config`, `align_to_target_with_config`,
  `karcher_mean_closed_with_config`.
- `WarpOptimConfig::band_frac` (default 0 = off): Sakoe–Chiba band for every `_with_config` entry
  point. DP: index corridor (equals `*_banded`). RBFGS: banded input-grid DP seed, unconstrained
  RBFGS, pointwise projection `|γ(t) − t| ≤ b` and the better of projected RBFGS / seed — always
  within the band and never worse than the banded DP. Rejected by curve shapes and landmarks.
- `WarpOptimConfig::lambda_scale: WarpLambdaScale` (`Native` default, `Dp`, `Fdasrsf`) and
  `WarpOptimConfig::rbfgs_lambda_factor`: exact DP ↔ RBFGS λ conversion
  `λ_rb = φ·λ_dp`, `φ = scl·‖q1‖‖q2‖/(‖q1ᶠ‖‖q2ᶠ‖)` for 1-D (`φ = 1 + O(1/m)`), `scl/(e1e2)` for
  curve shapes, total-norm form for the R^d shared-warp DP; derivation in the `WarpLambdaScale`
  docs.

### Changed

- The RBFGS coarse-DP seed honours `nbhd_dim` / `dp_scl` / `dp_cost` / `dp_prune` (1-D, R^d,
  curve shapes); with the default options it is unchanged.

**Elastic DP residual floor: diagnosis + refined warp optimisers for curve shapes (GH #65
ALN-CAV).** Additive and non-breaking: defaults are bit-identical (FNV-hash tests recorded before
the change); no new crate dependency.

### Added

- `CurveShapeConfig::warp_optim: WarpOptimConfig` — warp optimiser of every `curve_shape_*`
  function (pairwise, distance matrices, Karcher mean/median/PCA, TSRVF, depth, outliers, box plot,
  CI; open and closed curves, SRVF and `(a,b)` metrics): `WarpOptimMethod::Rbfgs` (Riemannian
  L-BFGS summed over coordinates) or the DP options `nbhd_dim` / `dp_cost` / `dp_scl` / `penalty`.
  Default = the classic 35-move DP. The penalty weight stays `CurveShapeConfig::lambda`
  (`warp_optim.lambda` must be 0).
- `elastic_align_pair_closed_with_config` / `elastic_distance_closed_with_config` — 1-D closed
  (periodic) alignment with a configurable optimiser (shift search and final alignment).
- serde derives for `WarpOptimConfig`, `WarpOptimMethod`, `WarpObjectivePenalty`, `DpEdgeCost`
  (`serde` feature).

### Notes

- Diagnosis: the DP restricts `γ'` to the slopes `{dr/dc : dr, dc ≤ nbhd_dim}`, so `d(β, β∘γ)` has
  a grid-independent floor (SRVF ≈ 0.0186 for m = 81…321; `nbhd_dim = 10` + linear cost ≈ 0.011);
  the resampling/quadrature floor with the exact warp is O(h) (0.0018 → 0.00046). RBFGS with
  `align_iter ≥ 4` reaches that floor (0.0019 → 0.00051); with the default `align_iter = 2` the
  unconverged rotation ↔ warp alternation leaves ≈ 0.002 (hidden below the DP floor).
  Recommended: `warp_optim.method = Rbfgs` + `align_iter = 4`.

**Exact DP pruning on by default (GH #66 ALN-CAV).** Behaviour-preserving and non-breaking: every
result is bit-identical (FNV-hash tests recorded on the unpruned code); no signature change, no new
crate dependency.

### Changed

- `WarpOptimConfig::dp_prune` now defaults to `true` (was `false`). Set it to `false` to run the
  unpruned reference fill.
- The shared elastic DP engine now prunes in the classic paths too — `elastic_align_pair`
  (+ `_banded`, `_penalized`, `_constrained`, `_with_landmarks`, `_multires`), `elastic_distance`,
  the self/cross distance matrices, `karcher_mean` (+ `_with_band`/`_banded`), closed-curve
  alignment, R^d (`elastic_align_pair_nd`, `karcher_mean_nd`), planar `curve_shape_*`, partial
  matching, geodesics, the RBFGS coarse DP start and every downstream user of the DP (elastic
  regression, conformal, FPCA, …). Pruning skips a predecessor whose accumulated cost already
  reaches the target's best; it is exact whenever every edge cost is non-negative, so it is applied
  only when `λ ≥ 0` and the grid is finite and strictly increasing (a new internal gate
  `dp_prune_is_exact`); a negative or NaN `λ` is never pruned.
- Speed (criterion `--quick`, same build, pruned vs `dp_prune = false`): pair alignment m = 100
  6.5 → 4.2 ms (1.5×), m = 200 26.7 → 16.3 ms (1.6×); `karcher_mean` n = 20, m = 50
  87.5 → 60.7 ms (1.4×). The `dp_options` bench rows are renamed (`default` = pruned,
  `*_unpruned` = reference) and `karcher_mean/n20_m50_unpruned` is added.

**Permutation test of group amplitude and phase differences (GH #62 ALN-PHTEST).** Additive and
non-breaking (new modules `alignment::phase_amplitude_test` and `inference::permanova`; existing code
untouched apart from re-exports); no new crate dependency.

### Added

- `alignment::elastic_phase_amplitude_test` (+ `PhaseAmplitudeTestConfig`, `PhaseAmplitudeTestResult`,
  `AmplitudeMetric`) — tests H0 "no group difference" (2+ groups) separately for amplitude, for phase
  and (optionally) jointly in the elastic quotient space. All curves are aligned jointly and
  label-free to their Karcher mean (so permuting labels with fixed distance matrices is an exact test
  under exchangeability); amplitude distances are pairwise elastic distances (default) or L² between
  template-aligned SRSFs, phase distances are Fisher–Rao geodesics `arccos⟨ψᵢ, ψⱼ⟩` between the warps,
  and the joint distance combines both scale-free. Each is tested with a PERMANOVA pseudo-F and a
  seeded permutation p-value (`seed + k`, parallel over permutations, identical with or without the
  `parallel` feature). Returns statistics, p-values, both distance matrices and the alignment.
- `inference::permanova` (+ `PermanovaResult`) — one-way PERMANOVA (Anderson 2001) on any symmetric
  distance matrix; pseudo-F, sums of squares and R² match R `vegan` 2.7.8 `adonis2` to 1e-10, and the
  p-value uses vegan's tie tolerance `F* ≥ F − √ε`.

**Tang–Müller pairwise curve synchronisation (GH #61 ALN-WFDA).** Additive and non-breaking (new
module `alignment::wfda`; existing code untouched apart from re-exports); no new crate dependency.

### Added

- `alignment::pairwise_synchronization(data, argvals, &WfdaConfig) -> WfdaResult` — port of R
  fdapace 0.6.0 `WFDA` (Tang & Müller 2008, Biometrika): per curve, penalised monotone
  piecewise-linear (`nknots` 1–7) or random-search pairwise warps to a seeded random subset of
  partners (`subset_prop`), truncated (cost ≤ 0.9 quantile) or inverse-cost-weighted averaging
  (`WfdaChoice`) into the inverse warp, inversion and registration; fdapace's data-driven default
  λ. `WfdaResult` returns warps, inverse warps, aligned curves, costs, λ and the partners /
  pairwise warps / pairwise costs. `WfdaCostMeasure::FdapaceLegacy` reproduces fdapace's pairwise
  cost, which indexes a single matrix element instead of the partner curve; the default
  `PairwiseFit` uses the intended data-fit cost. Golden-tested against fdapace 0.6.0 (warps within
  1e-4).

**Exponential-family registration with generalized FPCA (GH #60 ALN-EXPFAM).** Additive and
non-breaking (new module `alignment::expfam`; existing code untouched apart from re-exports); no new
crate dependency.

### Added

- `alignment::register_fpca` / `register_fpca_regular` — port of R `registr` 2.2.1
  `register_fpca` (Wrobel et al. 2019 JCGS; Wrobel & Bauer 2021 JOSS, `fpca_type =
  "variationalEM"`): alternating likelihood-based registration with monotone cubic B-spline
  inverse warps (`kh` basis functions per curve) and variational-EM generalized FPCA of the
  registered data, for gaussian, binomial, gamma and poisson families (`ExpFamily`), irregular
  long-format input (`IrregFdata`, curve-specific grids), optional `Y_template`, and incomplete
  curves (`CurveIncompleteness::{Leading, Trailing, Full}` with endpoint penalty `lambda_inc`).
- `alignment::register_expfam` — one registration pass to the spline-GLM population mean
  (registr `registr` without an FPCA object); `ExpFamRegistration` (registered times, `hinv_beta`,
  inner knots, per-curve losses, gamma scales).
- `alignment::gfpca_variational` (+ `GfpcaConfig`, `GfpcaResult`) — registr `fpca_gauss` /
  `bfpca` (Jaakkola–Jordan variational bound for binary data) on irregular data; reproduces R to
  ~1e-7 given the same initial FPC coefficients (`psi_init`, `xi_init`).
- `alignment::evaluate_inverse_warp`, `RegisterFpcaConfig`, `RegisterFpcaResult`.
- `alignment::gfpca_two_step` and `RegisterFpcaConfig::fpca_type = FpcaType::TwoStep` — registr
  `gfpca_twoStep` (Gertheiss, Goldsmith & Staicu 2017), family-specific GFPCA for gaussian,
  binomial, gamma and poisson data, implemented natively (no mgcv / gamm4): P-spline GLM mean with
  GCV/UBRE, cross products smoothed by a tensor-product P-spline (`te(bs = "ps")`, mgcv's `np`
  reparameterisation) with GCV, latent transform, `nearPD`, eigen-decomposition on the rounded
  time grid (`GfpcaConfig::index_significant_digits`), and a P-spline + random-FPC-score GLMM
  (REML for gaussian, two-stage Laplace as lme4 `glmer` otherwise). Matches R to ≈ 1e-6
  (covariance, eigenvalues), 1e-7 (gaussian mean/scores) and ≲ 5e-4 (Laplace fits); usable in
  `register_fpca` end-to-end (re-registration to per-curve latent spline fits).
- `GfpcaConfig::npc_criterion` (`NpcCriterion { var_explained, min_component_share }`) — registr
  `npc_criterion` / `determine_npc` (both elements for the two-step GFPCA; the first, with
  registr's `Kt = 20`, for the variational EM).
- `GfpcaConfig::periodic` — periodic B-spline (`pbs::pbs`) template / FPC basis (registr
  `periodic = TRUE`).
- `RegisterFpcaConfig::warping = WarpingType::PiecewiseLinear2` and `prior_sd` — registr's
  two-knot piecewise-linear inverse warps on the scaled domain with optional normal priors
  (binary data).
- `GfpcaResult` gains `fpca_type`, `evalues_sum`, `variance_components`, `yhat`.
- Deviations (documented on the module): exact monotone reparameterisation + L-BFGS with analytic
  gradients for all families instead of `constrOptim` (losses ≤ R's on every curve where R's
  optimum is feasible); Rust RNG for the EM start; native GCV/UBRE, REML and Laplace optimisers
  instead of mgcv / lme4; the > 10⁷-row subsampling of the template GLM is not ported (memory
  guard, not needed).

**General (a,b) elastic metrics for planar curves (GH #59 ALN-ABMETRIC).** Additive and
non-breaking: new `CurveShapeConfig::metric` field (default `CurveShapeMetric::Srvf`, which keeps
every existing result bit-identical); no new crate dependency.

### Added

- `alignment::CurveShapeMetric` (`Srvf` | `Elastic { a, b }`) and `CurveShapeConfig::metric` —
  the Kurtek–Needham (SIAM J. Imaging Sci. 2020) first-order elastic metric
  `∫ a²⟨D_s h,N⟩⟨D_s k,N⟩ + b²⟨D_s h,T⟩⟨D_s k,T⟩ ds` for open planar curves, through the
  F-transform. All `curve_shape_*` functions (align/distance/distance matrix, Karcher
  mean/median, covariance, PCA, TSRVF, depth, outliers, box plot, bootstrap CI) accept it:
  reparametrisation by the shared DP (F transforms like an SRVF, `(F∘γ)√γ'`), rotation by
  F-space Procrustes (exact quotient over rotations and branches; curve rotation `ψ/λ`),
  scale on the unit sphere. Conventions follow geomstats 2.8.0 `ElasticMetric(a, b)`: SRVF is
  `a = 1, b = 1/2`; `k = 0` branch `θ(0) = atan2 ∈ (−π, π]`. Closed contours are not supported
  with the elastic metric (validated).
- `alignment::ab_transform(beta, a, b)` / `ab_transform_inverse(f, a, b)` — the F-transform
  `2b|c'|^{1/2}(c'/|c'|)^{a/(2b)}` and its inverse (left inverse for `a/(2b) ≤ 1`, right
  inverse for `≥ 1`). Validated against geomstats 2.8.0 (`FTransform`, `ElasticMetric.dist`).

**DTW lower bounds and exact 1-NN search (GH #58 ALN-LB).** Additive and non-breaking: new
`metric::dtw_lb` module; existing DTW code paths and results unchanged; no new crate dependency.

### Added

- `metric::dtw_lb` — `lb_envelope(x, radius)` (tslearn `lb_envelope`, Lemire streaming min/max),
  `lb_envelope_windowed(y, query_len, window)` (envelope under any `DtwWindow` / unequal lengths),
  `lb_keogh(query, &envelope, p)`, `lb_kim(x, y, p)`, `lb_webb(x, y, p, window)` (Webb &
  Petitjean 2021, generalised to every `DtwWindow`; `p ≥ 1`),
  `dtw_distance_early_abandon(x, y, p, window, cutoff)` (EAPrunedDTW, Herrmann & Webb 2021;
  bit-identical to `dtw_distance_windowed` when `≤ cutoff`, else `None`) and
  `dtw_nearest_neighbor(queries, references, p, window)` (exact 1-NN: LB_Keogh-ordered cascade
  LB_Keogh → LB_Kim → LB_Webb → EAPrunedDTW with cumulative LB_Keogh bound, parallel over
  queries; equals brute force in index and distance bits) with `DtwEnvelope`,
  `DtwNearestNeighbors`, `DtwSearchStats`. All bounds use the `dtw_distance_windowed`
  convention `Σ|x_i − y_j|^p` with no root (tslearn/dtaidistance report the square root).
  Golden-tested against tslearn 0.10.0, dtaidistance 2.5.1 and the authors' LB_Webb Java code.
- Bench `dtw_lb_benchmarks` (1-NN vs brute force).

**aeon elastic-distance family (GH #57 ALN-ELDIST).** Additive and non-breaking: new
`metric::elastic_family` module plus two new `DtwWindow` variants (`#[non_exhaustive]` enum);
existing variants, DTW / soft-DTW / DBA code paths and results unchanged; no new crate dependency.

### Added

- `metric::elastic_family` — DDTW, WDTW, WDDTW, ADTW, ERP, EDR, TWE, MSM and LCSS, each with
  aeon's four functions `<name>_distance`, `<name>_pairwise_distance` (self with `other = None`,
  or cross; parallel), `<name>_cost_matrix` and `<name>_alignment_path` (→ `ElasticDistancePath`).
  Parameter names and defaults follow aeon (`g`, `warp_penalty`, `epsilon`, `nu`/`lmbda`, `c`,
  `independent`); DDTW/WDDTW use aeon's Keogh–Pazzani average-of-slopes derivative (interior
  points only, length `n − 2`). Golden-tested against aeon 1.6.0 (distance, cost matrix,
  path; unequal lengths, Sakoe-Chiba fraction and Itakura windows) to 1e-10, cross-checked
  against tsdistances 0.1.9. Shared padded-lattice DP core.
- `DtwWindow::SakoeChibaFraction { window }` and `DtwWindow::ItakuraFraction { max_slope }` —
  exact ports of aeon's `create_bounding_matrix(window=…)` / `(itakura_max_slope=…)`; accepted
  by every windowed DP in the crate.

**Sparse / irregular elastic alignment and means (GH #53 ALN-SPARSE).** Additive and
non-breaking: new `alignment::sparse` module only (new config/result fields default to the
previous behaviour); no existing code path touched; no new crate dependency.

### Added

- `alignment::sparse_elastic_align_pair(&SparseCurve, &SparseCurve, &SparseAlignConfig)` →
  `SparseAlignmentResult` — elastic (SRV) alignment of two curves in R^d (d >= 1) observed at
  their own, differing point sets (R elasdics 1.1.3 `align_curves`, Steyer, Stöcker & Greven,
  Biometrics 2023): curves as polygons with piecewise-constant SRV, warp optimised at the
  observation points by coordinate-wise analytic maximisation; open and closed curves
  (periodic extension + starting-point pre-alignment).
- `alignment::sparse_elastic_mean(&[SparseCurve], &SparseElasticMeanConfig)` →
  `SparseElasticMeanResult` — elastic mean of sparse open curves with a linear
  (`SparseMeanType::Smooth`) or constant (`SparseMeanType::Polygon`) SRV spline on given knots
  (elasdics `compute_elastic_mean`), alternating a least-squares spline fit of the warped SRVs
  with per-curve alignment (parallel under `parallel`); `srv_eval` / `mean_curve` evaluators.
- `SparseCurve` (points `d × m` + optional parametrisation, arc length by default) and
  `sparse_curves_from_irreg(&IrregFdata)` for 1-D irregular functional data.
- Validated against elasdics 1.1.3 goldens (open/closed `align_curves` examples to 1e-12,
  polygon mean coefficients to 1e-9, smooth mean within optimiser tolerance) and invariants
  (collinear-point and reparametrisation invariance, mean of one curve on different grids,
  dense-grid limit vs `karcher_mean`). Deviations (documented): closed-curve distance taken
  before wrapping the warp (elasdics reports an overestimate after wrapping); exact spline
  integrals in the smooth mean.
- Closed sparse elastic means: `SparseElasticMeanConfig { closed, pen_factor, closure_steps }`
  (defaults `false`, `100`, `1`; open results bit-identical) — elasdics `fit_mean_closed`
  generalised from 2-D to any `d`: all-pairs starting-point pre-alignment, periodic SRV spline,
  penalised closedness via one Brent line search per iteration (port of R `optimise`) on the
  analytic closure-penalty gradient, final closure step; `closure_steps > 1` adds Gauss–Newton
  projections that close the mean to round-off. `SparseElasticMeanResult { closed,
  start_indices }` and `closure_residual()`. Validated against elasdics 1.1.3: closed polygon
  means (help-page spirals, noisy closed shapes) match coefficients, warps and closure residual
  to 1e-9; closed smooth means compared on the exact objective; invariants (start-point
  relabelling, reparametrisation, recovery from different grids/starts, 3-D).
- The open smooth-mean golden test now asserts optimiser-independent quantities (objective
  ≤ R's re-scored by the exact integrator, triangle-inequality distance bounds, a 5 %-of-norm
  mean-curve margin from R's `rel.tol = 0.1`) instead of near-limit coefficient tolerances.

**Bayesian group alignment (GH #52 ALN-BAYESGRP).** Additive and non-breaking: new
`alignment::bayesian_group` module only; `bayesian_align_pair` and every existing code path
unchanged; no new crate dependency.

### Added

- `alignment::bayesian_align_group(data, argvals, &BayesianGroupConfig)` →
  `BayesianGroupResult` — joint posterior sampling of all warps, the template SRSF and the
  SRSF noise variance (model of fdasrsf 2.7.5 / fdasrvf 2.6.0 `time_warping_bayes`):
  Fourier tangent-space warps on the Hilbert sphere, quadratic B-spline template, Z-mixture
  pCN updates for warps and template, exact inverse-gamma Gibbs step for σ², periodic
  re-centring. Multi-chain (`n_chains`, chain `k` seeded `seed + k`, parallel under
  `parallel`, bit-identical either way); pooled posterior-mean (ψ-mean) and median warps,
  pointwise credible bands for warps / template SRSF / template function, σ² interval,
  per-curve warp and template acceptance rates, per-chain modes (`BayesianGroupChain`:
  chain-mean warps, max-log-likelihood draw, traces), between-chain `mode_spread`, split-R̂
  for σ², log-likelihood and per-curve warps, full pooled draws. Defaults `init_dp` (start
  from DP alignment) and `center_draws` (summarise exactly re-centred draws) can be switched
  off for fdasrsf behaviour. Validated against fdasrsf 2.7.5 `time_warping_bayes`
  (posterior-mean warps within 0.06), scipy B-spline design matrix, hand-computed split-R̂,
  and distributional checks (recovery, coverage, noise → wider bands, R̂ ≈ 1, determinism).
- `BayesianSampler::InfHmc` (`BayesianGroupConfig::sampler`; default `Pcn` stays
  bit-identical, pinned by a hash test): infinite-dimensional, prior-preconditioned HMC for
  each curve's warp coefficients (fdasrsf 2.7.5 / fdasrvf 2.6.0 `time_warping_bayes_infHMC`,
  Beskos et al. 2011 splitting integrator, `1..=L` leapfrog steps, per-curve dual-averaging
  step sizes during burn-in) with the exact SRSF-likelihood gradient (validated by finite
  differences), plus an exact conjugate Gibbs draw of the template. Config fields
  `hmc_step_size`, `hmc_max_leapfrog`, `hmc_adapt`, `hmc_target_accept`, and
  `injective_warps` (truncate warps to `‖v‖ < π`; always on for infHMC). New diagnostics on
  `BayesianGroupResult`: `ess_sigma2`, `ess_log_likelihood`, `ess_warps` (Geyer ESS summed
  over chains), `warp_bimodality` (Sarle coefficient) and `multimodal_curves`;
  `BayesianGroupChain::warp_step_sizes`. On a two-curve problem at equal iterations infHMC
  reaches ESS/draw ≈ 0.3–0.5 (pCN 0.004–0.03) and R̂ ≈ 1.000 (pCN ≈ 2.5–4); its posterior
  moments match a 10⁶-iteration pCN run and exact 2-D quadrature within MC error.

**Conformal prediction for partially observed curves (GH #51 ALN-CONFPART).** Additive and
non-breaking: new `conformal::partial` module only; no existing code path changed; no new crate
dependency.

### Added

- `conformal::partial_conformal_fit(data, argvals, &PartialConformalConfig)` /
  `partial_conformal_fit_with_template` → `PartialConformalModel` — joint registration
  + split-conformal model of Wang, Kurtek & Zhang (JCGS 2026): random training /
  calibration split, Karcher-mean template on the training part, DP registration of
  the calibration curves (amplitude responses `f_i ∘ γ_i*`, phase responses `γ_i*`).
- `conformal::partial_conformal_predict(&model, observed_prefix, &cfg)` →
  `PartialConformalPrediction` — for a curve observed on `[t_0, s]`: partial registration
  onto the template (`elastic_partial_match`), a pointwise amplitude prediction band and a
  phase (warp) prediction set with finite-sample coverage `1 − α`, using leave-self-out
  Nadaraya–Watson neighbourhood smoothing over L² / SRSF / elastic prefix distances
  (`PartialPredictorDistance`). The amplitude full-conformal interval is computed exactly
  in closed form (affine scores) instead of a trial-value grid; the phase set is exact via
  `PartialConformalPrediction::phase_set_contains`, with a geodesic trial-family envelope
  and radius for display. Validated by the exact rotation (exchangeability) invariant and
  Monte-Carlo coverage at α ∈ {0.1, 0.2, 0.3}. Not implemented: the paper's
  efficiency-oriented bandwidth tuning and fragmented / sparse observation regimes.

**R^d curve downstream analyses (GH #54 ALN-NDDOWN).** Additive and non-breaking: the
`curve_shape` Karcher mean code path is unchanged (same alignment sweep, update and stopping
rule); no new crate dependency.

### Added

- `alignment::curve_shape_karcher_median` — Karcher (geometric) median of R^d curve shapes
  (fdasrvf `multivariate_karcher_mean(ms = "median")`: Weiszfeld tangent step, medoid start).
- `alignment::curve_shape_tsrvf(mean, curves)` → `CurveShapeTsrvfResult` — TSRVF /
  tangent-space coordinates of training or new curves at the Karcher mean (sphere log map,
  closure-normal projection for closed contours; reproduces `shooting_vectors` exactly), and
  `curve_shape_tangent_to_curve(mean, v)` for the inverse map.
- `alignment::curve_shape_depth(curves, cfg)` → `CurveShapeDepthResult` — amplitude / phase
  elastic depth of curves (fdasrvf `curve_depth`); agrees with fdasrvf 2.5.0 to 2.1e-3
  (amplitude) with identical depth rankings.
- `alignment::curve_shape_outlier_detection(curves, &CurveShapeOutlierConfig)` →
  `CurveShapeOutlierResult` — shape-distance outliers around the Karcher median / mean
  (`CurveShapeCentre`), reusing the `OutlierThreshold` Tukey / quantile rules.
- `alignment::curve_shape_boxplot(median, cfg)` / `curve_shape_boxplot_from_aligned` →
  `CurveShapeBoxplot` (`CurveShapeBoxplotConfig { alpha = 0.05, k_a = 1.0 }`) — port of fdasrvf
  `curve_boxplot` (median, quartile and `(1 − alpha)` curves, whiskers, outliers, extremes);
  golden-matched to fdasrvf 2.5.0 `curvebox_data` (distances to 1e-12, all indices).
- `alignment::curve_shape_confidence_interval(curves, shape, &CurveShapeCiConfig)` →
  `CurveShapeCiResult` — bootstrap confidence region of the Karcher mean shape (geodesic
  radius + pointwise bands of the aligned bootstrap means).

**Elastic box plots and DP-means (GH #55 ALN-BOX).** Additive and non-breaking: existing
functions return identical values (`PhaseBoxplot` gains two fields); no new crate dependency.

### Added

- `alignment::amplitude_boxplot_from_aligned` → `AmplitudeBoxplot` and
  `alignment::elastic_phase_boxplot` → `ElasticPhaseBoxplot`: ports of fdasrsf 2.7.5
  `ampbox` / `phbox` (median, quartile and `(1-alpha)` quantile functions chosen by the
  distance-to-median energy criterion, whisker distance `k_a · IQR`, outliers, extremes;
  phase median by the `SqrtMedian` Weiszfeld iteration on the Hilbert sphere). Configured
  by `ElasticBoxplotConfig { alpha = 0.05, k_a = 1.0 }`. Golden-matched to fdasrsf 2.7.5
  (distances, radii, quartile/outlier indices to 1e-10..1e-12).
- New-function classification (fdasrsf `ampbox.classify` / `phbox.classify`):
  `AmplitudeBoxplot::classify_srsfs`, `ElasticPhaseBoxplot::classify_warps` →
  `BoxplotClassification { distances, regions: Vec<BoxplotRegion>, sides: Vec<BoxplotSide> }`
  (`Central50` / `CentralAlpha` / `WithinWhiskers` / `Outlier`, Q1/Q3 side).
- `alignment::elastic_boxplots(data, argvals, lambda, &cfg)` → `ElasticBoxplots`: Karcher
  median alignment then both box plots; `ElasticBoxplots::classify(new_data, argvals)` aligns
  new functions to the fitted template and classifies them in both box plots.
- `PhaseBoxplot::classify_warps` for the band-depth phase box plot, plus the new
  `PhaseBoxplot::fence_lower` / `fence_upper` fields it uses (training warps are flagged
  exactly as in `outlier_indices`).
- `alignment::elastic_dpmeans(data, argvals, &DpMeansConfig)` → `DpMeansResult`: elastic
  DP-means clustering with alignment (fdasrsf 2.7.5 / fdasrvf 2.6.0 `dpmeans_align`), the
  number of clusters learned through the penalty `λ` (`penalty: None` = seeded k-means++
  heuristic). Cluster counts and labels match fdasrsf on a λ grid from 0 to 100.

**Alignment hygiene (GH #56 ALN-HYGIENE).** Additive and non-breaking; all defaults are
bit-identical (proven with pre-change FNV-1a bit-pattern hashes); no new crate dependency.

### Added

- `alignment::OutlierThreshold { TukeyFence { k }, Quantile }` and
  `ElasticOutlierConfig::threshold_method` (default `TukeyFence { k: 1.5 }`, the previous
  fixed fence). `Quantile` finally uses `ElasticOutlierConfig::alpha` (cutoff = empirical
  `(1 − alpha)` quantile of the elastic distances); `alpha` is now documented as having no
  effect under the default Tukey rule.
- `alignment::srsf_transform_smoothed(data, argvals, &SrsfSmoothing)` and
  `alignment::srsf_presmooth` — opt-in pre-smoothing before the SRSF transform:
  `SrsfSmoothing::None` (default, identical to `srsf_transform`), `BoxFilter { passes }`
  (fdasrvf `smooth.data`, golden-matched to fdasrvf 2.5.0) and `SmoothingSpline`
  (semantics of fdasrvf 2.5.1 `f_to_srvf(smooth = TRUE)`: per-curve GCV cubic smoothing
  spline).

### Performance

- Closed-curve rotation search (`elastic_align_pair_closed`, `elastic_distance_closed`,
  `karcher_mean_closed`): the target SRSF and Simpson weights are computed once instead of
  per candidate, and candidate rotations are evaluated in parallel —
  `elastic_align_pair_closed` m=50 37.7 → 17.1 ms, m=100 170.7 → 35.0 ms (new criterion
  bench `elastic_align_pair_closed`). Results unchanged.
- Elastic FPCA: the remaining per-curve loops (ψ conversion, sphere tangent mean,
  covariance build, score projection) dispatch via `iter_maybe_parallel!` above a work
  threshold with deterministic in-order reductions, the covariance is built from
  contiguous centred columns, and joint FPCA's balance search evaluates its two probe SVDs
  concurrently (`vert_fpca` n=100 m=50: 561 → 369 µs). Results bit-identical.
- `elastic_outlier_detection` computes its per-curve distances in parallel.

**Shift / dilation / affine warp registration + linear warp classes in `align_cluster_fd`
(GH #50 ALN-AFFINE).** Additive and non-breaking; `align_cluster_fd` defaults are
bit-identical (default warp class `Elastic` runs the unchanged elastic path); no new crate
dependency.

### Added

- `alignment::affine_registration(data, argvals, LinearWarpClass, ShiftTemplate,
  &AffineRegistrationConfig)` — registration with linear warps `h(t) = a·t + b`
  (`LinearWarpClass::{None, Shift, Dilation, Affine}`) to a re-estimated mean, fixed or
  closure template (reuses `ShiftTemplate` / `ShiftExtrapolation`), minimising
  `AffineMetric::{L2, NormalizedL2, Pearson}` (R fdacluster 0.4.2 definitions) on the full
  grid (`AffineDomain::Full`, default, with extrapolation) or on the warped-domain overlap
  (`AffineDomain::Overlap`, fdacluster); bounds = fdacluster `warping_options` (0.15, 0.15);
  deterministic grid search + golden-section / Nelder–Mead refinement; optional warp
  centring (mean dilation 1, mean shift 0). Returns `AffineRegistrationResult`.
- `alignment::affine_warp_curves` — apply `gᵢ(t) = fᵢ((t − bᵢ)/aᵢ)`.
- `clustering_advanced::AlignClusterWarp::{Elastic (default), None, Shift, Dilation,
  Affine}` with new sealed-config fields `AlignClusterConfig::warp_class` and
  `::linear_config`; `AlignClusterResult::affine_warps: Option<AffineWarps>` (per-curve
  dilations/shifts for linear classes). Linear classes run fdacluster-style k-means
  alignment (`fdakmeans(warping_class, centroid_type = "mean", metric)`).
- Validation: exact recovery of synthetic affine warps, invariants, hand-computed metrics,
  and golden values from R fdacluster 0.4.2 `fdakmeans` (k = 1 affine/dilation/shift,
  k = 2 shift/affine) compared through frame-invariant relative warps (max deviation
  9e-4 / 2.2e-4 / 1.1e-4 for k = 1; 1.6e-4 / 2.6e-3 for k = 2; memberships identical).

**Configurable elastic DP: neighbourhood, `dp_scl` refinement, DP2 cost, pruning (GH #49
ALN-DPACC).** Additive and non-breaking; defaults are bit-identical (existing FNV hash tests
unchanged); no new crate dependency.

### Added

- `WarpOptimConfig` (sealed; `default()` + field assignment) gains DP-only fields:
  - `nbhd_dim` (default `7`, the classic 35-move set; `1..=64`) — coprime move set
    `{(dr, dc) : 1 ≤ dr, dc ≤ nbhd_dim, gcd = 1}` (fdasrvf `nbhd_dim`, fdasrsf `grid_dim`;
    `10` = fdasrsf `DP2`'s 63 moves). A larger set never increases the DP's own optimum.
  - `dp_scl` (default `1`; `1..=64`) — fdasrsf ≥ 2.7.4 `DP` `scl`: SRSFs linearly
    resampled `dp_scl`× finer to evaluate each edge cost; the warp path stays on the input
    grid and `lambda` keeps its scale.
  - `dp_cost: DpEdgeCost` — `PiecewiseConstant` (default, classic) or `PiecewiseLinear`
    (fdasrsf ≥ 2.7.4 `DP2`: exact integral of the linearly interpolated SRSFs per edge).
    With `nbhd_dim = 10` it reproduces fdasrsf 2.7.5 `optimum_reparam(method="DP2")` warps
    to ≤ 2.3e-16 (golden tests, `l2gam` penalty included).
  - `dp_prune` (default `false`) — fdasrsf ≥ 2.7.4 candidate pruning: skip predecessors
    whose cost already reaches the target's best. Exact (bit-identical warps, proved by
    tests) since edge costs are non-negative; ≈ 1.55–1.75× faster DP.
- New public enum `DpEdgeCost` (re-exported from `alignment` and the crate root).
- Bench group `dp_options` in `benches/alignment_benchmarks.rs` (default vs pruned / linear
  / `nbhd_dim = 10` / `dp_scl = 3`).

**R^d curve shape analysis (GH #48 ALN-CURVE).** Additive and non-breaking; no
existing code path changed (the shared DP engine is reused, not modified); no new crate
dependency.

### Added

- New `alignment::curve_shape` module (re-exported from `alignment` and the crate root):
  elastic shape analysis of open curves and closed contours `β : [0,1] → R^d` (curves are
  `d × m` `FdMatrix`, sets are `FdCurveSet`). Sealed `CurveShapeConfig { rotation = true,
  allow_reflection = false, scale = true, include_length = false, closed = false,
  seed_search = true, seed_step = 4, reparam = true, lambda = 0, align_iter = 2 }`.
  - `curve_shape_align_pair` / `curve_shape_distance` / `curve_shape_self_distance_matrix`
    — shape distance modulo translation, rotation (SVD Procrustes / Kabsch over SO(d), or
    O(d) with `allow_reflection`), scale (pre-shape sphere, `arccos⟨q1,q2*⟩`, optional
    log-length term; flat L² size-and-shape distance with `scale = false`),
    reparametrisation (R^d DP on the shared coprime-neighbourhood engine) and, for closed
    contours, starting point (coarse/fine seed search). Rotation and warp are optimised
    by alternation. `CurveShapeAlignment { beta1, beta2_aligned, q1, q2_aligned,
    rotation, gamma, seed, length1, length2, distance }`.
  - `project_closed_srvf` (Srivastava–Klassen closure projection, fdasrsf
    `project_curve` generalised to R^d with the exact Jacobian), `closure_residual`,
    `curve_srvf`, `srvf_to_curve`, `optimal_rotation`, `curve_from_set`, `curves_to_set`.
  - `curve_shape_karcher_mean` (sphere or flat log/exp maps, closure-normal tangent
    projection for closed contours) → `CurveShapeKarcherResult` (mean, mean SRVF,
    aligned curves/SRVFs, rotations, warps, seeds, shooting vectors, distances, lengths,
    geometric mean length, energy); `curve_shape_karcher_covariance`; `curve_shape_pca`
    (L²-weighted tangent PCA via the Gram matrix) → `CurveShapePcaResult` with
    `mode_shape` principal-geodesic reconstruction.
  - Validated against fdasrsf 2.7.5 (`elastic_distance_curve`, `fdacurve.karcher_mean`):
    open 2-D/3-D and closed distances within 5e-3 (fdasrsf's own resolution spread is
    1–2e-3); with rotation fdars reaches a lower distance than fdasrsf's coordinate-space
    rotation; the fdars Karcher mean has lower energy than fdasrsf's. Exact recovery of
    rotations, scales, reflections and sample seed shifts; closure residual < 1e-9.

**Ramsay–Li continuous registration (GH #47 ALN-CC).** Additive and non-breaking;
no existing code path changed; no new crate dependency.

### Added

- `continuous_registration(data, argvals, target, &ContinuousRegistrationConfig)` — port
  of R `fda::register.fd` (non-periodic): monotone warps
  `h(t) = a + (b−a)·∫exp W / ∫ₐᵇexp W` with `W` a clamped B-spline expansion (R
  `create.bspline.basis` convention, first coefficient fixed at 0 as in R) and roughness
  penalty `λ∫(DᴸW)²`. `ContinuousRegistrationCriterion { LeastSquares (crit = 1),
  MinEigenvalue (crit = 2, default) }`; the minimum-eigenvalue criterion is solved by
  variable projection with the exact (envelope-theorem) gradient, both criteria by
  penalised Gauss–Newton with Levenberg–Marquardt step control (monotone criterion
  decrease). Target defaults to the cross-sectional mean; `max_template_iter > 1`
  re-estimates it Procrustes-style. Sealed `ContinuousRegistrationConfig { criterion,
  nbasis = 6, norder = 4, lambda = 0, lfd_order = 2, max_iter = 50, tol = 1e-8, n_fine,
  max_template_iter = 1, template_tol = 1e-4, initial_coefficients }`;
  `ContinuousRegistrationResult { registered_data, warps, w_coefficients, w_knots,
  criterion, target, n_iter, converged, n_template_iter }`. Validated against R fda 6.3.0
  `register.fd` (crit 1 and 2, with and without penalty): fdars reaches a criterion ≤ R's
  on every curve, warps within 1e-2 of R's basis-projected `warpfd`; W basis values and
  penalty matrix match `eval.basis` / `getbasispenalty` to 1e-9.

**Iterative least-squares shift registration + landmark shift registration (GH #46
ALN-SHIFT).** Additive and non-breaking; the existing golden-section
`least_squares_shift_registration` is unchanged; no new crate dependency.

### Added

- `iterative_shift_registration(data, argvals, template, &IterativeShiftConfig)` —
  Ramsay & Silverman (ch. 7) modified Newton–Raphson shift registration with the
  template re-estimated on every iteration, a port of scikit-fda 0.10.1
  `LeastSquaresShiftRegistration`. `ShiftTemplate { Mean (default), Fixed(&[f64]),
  Custom(&dyn Fn(&FdMatrix) -> Vec<f64>) }`; sealed `IterativeShiftConfig { max_iter = 5,
  tol = 1e-2, step_size = 1, restrict_domain = false, extrapolation, initial }` with
  `validate()`; `IterativeShiftResult` reports shifts, template, `n_iter`, `converged`
  and the (optionally restricted) registered grid. Matches scikit-fda golden shifts to
  < 1e-10 (restricted domain: < 1e-12 with SciPy's quadrature matched, < 1e-5 raw).
- `ShiftExtrapolation { Boundary (default), Linear, Periodic, Zeros }` and
  `shift_curves(data, argvals, shifts, extrapolation, restrict_domain)` (scikit-fda
  `FData.shift`).
- Landmark shift registration: `landmark_shift_deltas(landmarks, LandmarkLocation)`
  (`LandmarkLocation { Mean (default), MidRange, Fixed(f64), Custom(&dyn Fn(&[f64]) -> f64) }`),
  `landmark_shift_deltas_multi` (several landmarks per curve, least-squares common
  shift) and `landmark_shift_registration(data, argvals, landmarks, location,
  &LandmarkShiftConfig)` (scikit-fda `landmark_shift_deltas` /
  `landmark_shift_registration`).
- Sign convention: fdars registers `fᵢ(t − δᵢ)`, so all new shifts are the negation
  of scikit-fda's `deltas_`. Documented deviations: Boundary (not linear)
  extrapolation by default; restricted-domain integration recomputed each iteration
  (scikit-fda restricts cumulatively and asserts when the interval re-widens);
  fdars' even-N Simpson rule; zero Newton step for curves with zero derivative energy.

**Riemannian L-BFGS warp optimiser + in-objective warp penalties (GH #42 ALN-RBFGS,
GH #43 ALN-PEN).** Additive and non-breaking; every existing function is
bit-identical (proved against hashes recorded on the previous commit); no new
crate dependency.

### Added

- `WarpOptimMethod { Dp (default), Rbfgs }` — `Rbfgs` is a Riemannian
  limited-memory BFGS over `l = √γ̇` on the unit Hilbert sphere (port of fdasrsf
  2.7.5 `optimum_reparam(method="RBFGS")`, after Huang, Gallivan, Srivastava &
  Absil): exp-map steps, parallel-translated BFGS pairs, cautious update, fading
  barrier, exact discrete gradient; seeded by a coarse (≤ 51 point) DP warp and
  refined on SRSFs resampled `scl` times finer. Reuses the soft-DTW L-BFGS
  two-loop recursion. Typically lower alignment cost than DP with smooth warps;
  matches fdasrsf RBFGS warps to < 0.015 (sup) and its cost to < 5% on golden
  cases.
- `WarpObjectivePenalty { L2Gam (default), L2Psi, Roughness, Geodesic }` —
  warp penalties applied **inside** the objective (fdasrvf/fdasrsf
  `penalty`/`penalty_method`). `L2Gam` (the classic first-order penalty) and
  `L2Psi` are applied pointwise inside the DP; `Roughness` (`∫ γ̈²`) and
  `Geodesic` need `Rbfgs` (DP with `lambda > 0` returns `InvalidParameter`, as
  fdasrsf raises).
- `WarpOptimConfig` (sealed; `method`, `lambda`, `penalty`, `max_iter`, `scl`)
  with `validate()`, and the entry points `elastic_align_pair_with_config`,
  `elastic_distance_with_config`, `elastic_self_distance_matrix_with_config`,
  `elastic_cross_distance_matrix_with_config`, `karcher_mean_with_config`
  (all `Result<_, FdarError>`). The default config equals the classic
  functions bit-for-bit.

### Deprecated (documentation only)

- The post-hoc Tikhonov smoothing behind `WarpPenaltyType::SecondOrder` /
  `Combined` (`elastic_align_pair_penalized`) — prefer the in-objective
  `Rbfgs` + `Roughness` penalty. Behaviour unchanged; no `#[deprecated]`
  attribute.

**DTW-family alignment extensions (GH #44).** Additive and non-breaking;
existing functions are numerically unchanged; no new crate dependency.

### Added

- `dtw_path` / `DtwPath` — optimal hard-DTW warping path via traceback, with
  tslearn `dtw_path` tie-breaking (diagonal, then vertical, then horizontal);
  `cost` equals `dtw_distance(x, y, p, max(n, m))` bit-for-bit.
- `DtwWindow` (`None` / `SakoeChiba { radius }` / `Itakura { max_slope }`) plus
  `DtwWindow::mask` — global warping windows whose admissible cells match
  tslearn's `sakoe_chiba_mask` / `itakura_mask` exactly.
- `dtw_distance_windowed` — O(m)-memory hard-DTW distance under a `DtwWindow`.
- `soft_dtw_distance_windowed` — soft-DTW value under a `DtwWindow` (masked
  cells act as `+inf`).
- `soft_dtw_alignment` / `SoftDtwAlignment` — soft-DTW value plus the `n × m`
  expected alignment matrix `E` (tslearn `soft_dtw_alignment`), with a
  mask-respecting backward pass (masked cells get exactly zero weight).
- `gak_triangular` — normalized GAK with Cuturi's triangular parameter `T`,
  computed in log space; `T = 0` is bit-identical to `gak`.
- `GakConfig::triangular` + `GakConfig::with_triangular` — triangular `T` for
  `gak_gram_matrix` / `gak_gram_train` (and kernel k-means); default `0` keeps
  the previous Gram matrices bit-identical.
- `GakGramTrain::triangular` — the training `T`, reused by `gak_gram_predict`.

tslearn parity: window masks, `dtw_path` tie-breaking and `soft_dtw_alignment`
reproduce tslearn's documented values; tslearn's `gak` has no `T`, so the
triangular parameter follows Cuturi (2011, "Fast Global Alignment Kernels").

**DTW Barycenter Averaging (GH #45).** Additive and non-breaking; no new crate
dependency.

### Added

- `dtw_barycenter_averaging` / `DbaConfig` / `DbaInit` / `DbaResult` — Petitjean
  DTW Barycenter Averaging with mean (default), medoid or custom
  initialisation, an optional `DtwWindow`, per-series weights and a resampled
  `barycenter_size`. Built on `dtw_path`; loop, stopping rule and tie-breaking
  follow tslearn's `dtw_barycenter_averaging`.
- `dtw_kmeans` / `DtwKMeansConfig` / `DtwKMeansInit` / `DtwKMeansResult`
  (+ `predict`) — DTW k-means with DBA centroids warm-started from the previous
  centroid (tslearn `TimeSeriesKMeans(metric="dtw")`), k-means++ seeding on DTW
  costs, `n_init` restarts seeded `seed + r`, and in-place re-seeding of empty
  clusters.

Validation: golden barycenters from tslearn 0.10.0 (mean, Sakoe-Chiba, Itakura,
resampled, weighted, custom and medoid init) and dtaidistance 2.5.1 (`dba`,
`dba_loop`), plus tslearn `TimeSeriesKMeans` labels and centroids from a fixed
initialisation.

## [0.47.0] - 2026-09-25

**Performance — Cross-Language Benchmarks & Optimization Pass.** Strictly
non-breaking and numeric-identical; no public API, signature or visibility
change; no new crate dependency.

A measure-first milestone: an external harness timed `fdars-core` against
scikit-fda and the R FDA ecosystem on identical seeded inputs (single-threaded,
gated on numeric agreement before any timing was trusted), the ranked hot
paths were optimized or confirmed already near-optimal, and every source
change is proven numeric-identical by golden-value fixtures plus the full
existing test suite. Single-threaded, `fdars-core` is faster than
scikit-fda 0.10.1 and R on all six overlapping operations (FPCA, basis
smoothing, depth, functional boxplot, FPCR, elastic alignment) at N=500,
M=200 — see
[documentation/PERFORMANCE.md](documentation/PERFORMANCE.md).

### Changed (internal only)

- **face_covariance covariance surface — upper-triangle computation (OPT-02).**
  A new crate-internal helper `sym_cov_irreg` computes only the upper triangle
  of the symmetric kernel-smoothed covariance surface and bit-copies the
  mirror, sharing a single kernel-weight table (halving the Gaussian-kernel
  evaluations). Output is numeric-identical, locked by the
  `tests/equivalence_phase115.rs` golden fixture in debug and release;
  measured **1.19× faster** single-threaded in a thread-pinned,
  noise-gate-clean controlled A/B (n=200, m=30) — see
  [documentation/PERFORMANCE.md](documentation/PERFORMANCE.md).
- **Elastic alignment and SVD/FPCA — already near-optimal, no speedup
  claimed.** The elastic banded-DP fill (`dp_fill_grid`) now iterates its
  diagonal corridor directly instead of skip-scanning the grid (OPT-01);
  output is bit-identical (golden fixture `tests/equivalence_phase114.rs`)
  and the measured difference is within noise. The faer thin SVD already on
  the `linalg` path was re-benchmarked and left unchanged.

- **dp_fill_grid decomposed (pmat guardrail).** The banded and unbanded DP fill arms of
  the elastic-alignment dynamic program now share one private corridor-bounds helper
  (`dp_band_cols`), bringing the function back under the pmat cognitive-complexity bar so the
  v0.46.0 CI quality-gate guardrail stays green. Visitation order unchanged; output
  bit-identical (golden fixture `tests/equivalence_phase114.rs`, debug + release).

### Added (tooling & docs)

- **`crosslang_grid` criterion bench** (`benches/crosslang_grid.rs`, six
  grid-aligned groups) plus a `functional_boxplot` group in
  `benches/depth_benchmarks.rs` — dev-only, no library API change; reads the
  harness CSVs when present, otherwise falls back to a deterministic seeded
  synthetic dataset, so it compiles and runs from the published crate.
  Golden-value fixtures `tests/equivalence_phase114.rs` /
  `tests/equivalence_phase115.rs`.
- **`documentation/PERFORMANCE.md`** — the published head-to-head report
  (before/after tables, N×M scaling, methodology, reproduction instructions)
  and a README "Performance" speed story.
- **External cross-language harness** in the workspace-root `benchmarks/`
  directory (seeded data generator, scikit-fda + R runners, numeric-agreement
  gate, report generator) — explicitly NOT a Cargo workspace member and NOT
  part of the published crate; no crate dependency added.

## [0.46.0] - 2026-09-14

**Quality Hardening — pmat Complexity Pass & CI Guardrail.** Strictly non-breaking and
numeric-identical. A dedicated quality cycle that drives `pmat quality-gate` (checks:
complexity, satd, dead-code, duplicates) to PASS on `fdars-core` and wires it into CI as a
guardrail. Every code change is pure internal decomposition into private helper functions —
no public API/signature/visibility change, no new crate dependency — and every refactored
body is behavior- and numeric-identical (all loop bodies lifted verbatim; seeded-RNG draw
orders and iterative recurrences preserved bit-for-bit). Proven by the full existing suite
(2942 lib tests + 215 doctests + 28 examples) staying green.

### Changed (internal only)

- **Complexity decomposed to the gate bar (cognitive ≤23 / cyclomatic ≤30).** ~114 functions
  across the crate decomposed into private helpers so `pmat quality-gate` reports 0 complexity
  findings — top-level hotspots (QUAL-01), the `src/spm/` statistical-process-monitoring cluster
  (QUAL-02), and a 97-function tail spanning ~57 files across `alignment/`, `peer.rs`,
  `clustering_advanced.rs`, `scalar_on_function/`, `boosting_regression/`, `gmm/`, and more (QUAL-03).
- **Entropy reduced** (QUAL-04): repeated `DataTransformation` idioms factored into shared
  test/private helpers, numeric-identical.

### Fixed

- The one critical `.unwrap()` (`depth/rpd.rs`) replaced with `.expect(...)` documenting its
  by-construction invariant; two SATD doc-comment false-positives reworded (`depth/dispatch.rs`,
  `metric/lp.rs`) — prose only (QUAL-05).

### Added (CI tooling — not shipped in the crate)

- **pmat quality-gate CI guardrail** (CI-01): a GitHub Actions job installs pmat 3.18.0 and runs
  the curated gate against `fdars-core`, failing the build and gating release on any reintroduced
  complexity/SATD/dead-code/duplicate violation. The stochastic `entropy` analyzer is excluded
  from the gate (non-deterministic sampler → would flake CI); entropy stays an advisory metric.

## [0.45.0] - 2026-09-13

**Soft-DTW Barycenter — Global Optimizer (SDTW-O1).** Strictly additive, non-breaking.
Replaces the fixed-step MM/soft-DBA barycenter descent with an in-crate L-BFGS global
optimizer plus a deterministic multi-restart wrapper, adds a new `SoftDtwBarycenterConfig`
config struct and `soft_dtw_barycenter_with_config` entry point, and provides
quality-evidence tests showing the optimizer escapes local minima. No new crate dependency;
the 4-arg `soft_dtw_barycenter` signature is preserved as a thin non-breaking wrapper.

### Added

- **In-crate L-BFGS optimizer (`SDTW-01`).** Hand-rolled two-loop L-BFGS recursion with a
  VecDeque-based history ring (m=10), Armijo backtracking line search, curvature guard
  (`sᵀy > 1e-20`), non-descent direction fallback, and safeguarded minimum step size.
  Consumes the exact soft-DTW gradient from `soft_dtw_backward`; wasm32-safe, no `linalg`
  or `argmin` dependency. Replaces the MM diagonal-Newton `update_barycenter` step entirely.

- **Deterministic multi-restart wrapper (`SDTW-02`).** Wraps the L-BFGS single-init
  descent with a configurable number of restarts: restart 0 uses the unperturbed pointwise
  mean; restarts k≥1 use per-column population-std Gaussian perturbations seeded via the
  crate-standard `StdRng::seed_from_u64(seed + k)` convention. The restart with the
  strictly lowest total soft-DTW cost is selected (lowest-index tiebreak for determinism).
  Fixed-seed runs are bit-identical across platforms.

- **`SoftDtwBarycenterConfig` + `soft_dtw_barycenter_with_config` (`SDTW-03`).**
  A sealed (`#[non_exhaustive]`), manually-defaulted config struct with fields `gamma`,
  `max_iter`, `tol`, `n_restarts` (default 10), and `seed`. The companion
  `soft_dtw_barycenter_with_config` entry point delegates to the multi-restart driver.
  Both items are re-exported at the crate root and in `prelude`. The existing 4-arg
  `soft_dtw_barycenter(data, gamma, max_iter, tol)` signature is preserved as a thin
  behavior-compatible wrapper — no call site, R/WASM binding, or example required any
  change.

- **Quality-evidence tests (`SDTW-04`).** Two deterministic `#[cfg(test)]` tests in
  `barycenter_escape_evidence_tests`: `escape_from_local_minimum` proves multi-restart
  reaches a strictly lower total soft-DTW cost than the MM baseline (367% improvement on a
  bimodal Gaussian-pulse MM-stall dataset); `legacy_bounded_convex_hull` proves the legacy
  path stays f64-bounded within the per-column convex hull of the input data.

## [0.44.0] - 2026-09-11

**Differentiable Core — Reverse-Mode & Broadened Subset.** Strictly additive,
non-breaking. Adds an in-crate reverse-mode (vector-Jacobian-product) automatic
differentiation core alongside the v0.39.0 forward-mode `Dual`, and broadens the
differentiable operation set across four algorithm families — all generic over the
existing `Scalar` trait, with f64 numerics preserved and no new crate dependency.

### Added

- **Reverse-mode autodiff (`DIF-F1`).** A hand-written Wengert-list tape with a `Var`
  scalar type (implements `Scalar`) and a `vjp` entry point that seeds the output
  adjoint and accumulates all input gradients in a single backward sweep — efficient
  for many-input→scalar objectives. Validated against forward-mode `Dual` and central
  finite differences. `Var`, `vjp` re-exported at the crate root and in `prelude`.
- **Generic scalar hot-path signatures (`DIF-F3`).** `l2_distance`, `trapz`,
  `inner_product`, and `inner_product_l2` are now generic over `T: Scalar` (resolving
  to `f64` at existing call sites), so every f64 call site, all 28 examples, and the
  R/WASM binding surfaces compile unchanged.
- **Broadened differentiable subset (`DIF-F2`).** New generic-over-`Scalar`,
  differentiable functions with f64 parity: `bspline_basis_from_knots<T>` and a new
  `fourier_basis_eval<T>` core (basis evaluation); `predict_curve_generic<T>`
  (scalar-on-function / FPCR prediction) and `penalty_value_generic<T>` (roughness
  penalty `λ·cᵀRc`); `modal_depth_generic<T>` (differentiable functional depth). All
  finite-difference-checked at both `Dual` and `Var`.
- **End-to-end gradient API (`API-01`).** Unified `grad` / `jacobian` / `vjp` entry
  points, a worked composition demo (`tests/differentiable_composition.rs`) chaining
  the broadened subset into one FD-checked scalar objective, and a running module
  doctest demonstrating forward/reverse-mode agreement.

## [0.43.0] - 2026-09-09

**Test Determinism & Release Hardening.** A quality + release-hardening release —
docs, tests, and CI only; **no numeric or behavioral change** to `fdars-core`, and
**MSRV is unchanged** (1.81 crate / 1.84 for the `linalg` feature). It clears the
**Quality** blocker on the 1.0 gap checklist (`documentation/ROADMAP-TO-1.0.md`).

- **Golden-flake root cause (finally diagnosed).** The long-standing
  `golden_co_cluster_parallel`, `golden_co_cluster_below_threshold`
  (`equivalence_phase48`) and `svd_sign_fpca_two_matrix_bit_identical`
  (`equivalence_phase49`) flakes are **not** an environmental / disk-pressure /
  thread-scheduling flake (that historical hypothesis is overturned). They are a
  *deterministic* faer-vs-nalgebra SVD backend divergence in `fdata_to_pc`: the golden
  reference values were captured under the `faer` backend (`--features linalg`), and a
  build without `linalg` routes through the `nalgebra` backend, which diverges the FPCA
  rotation categorically (co_cluster lands in a different CEM basin; svd_sign flips a
  near-zero column sign). The "intermittent under full parallel `cargo test`" symptom
  was simply whether `--features linalg` was present in a given invocation.
- **Fix — least invasive.** A test-side `#[cfg_attr(not(feature = "linalg"), ignore)]`
  guard on exactly the three affected golden tests — **no new dependency, no tolerance
  relaxation, no `src/` change**. The tests run and pass under both documented `linalg`
  configs and are *ignored* (not failed) without `linalg`. The PACE sibling is
  unaffected and left unguarded.
- **Suite-wide robustness audit.** A sweep across the whole suite (integration + lib +
  doctests) found **zero further fragile tests**: the backend-fragility class is fully
  covered (this fix plus pre-existing guards), RNG/thread-order is robust (bit-identical
  passed counts under a `RAYON_NUM_THREADS` sweep), and no env/disk dependence remains.
- **CI determinism guardrail.** A new `determinism-guardrail` job in `rust-ci.yml`
  exercises the full parallel `cargo test` path (repeated runs + a `RAYON_NUM_THREADS=1`
  run) and *positively asserts the golden tests actually ran* under `linalg`
  (anti-silent-skip) — so a future regression fails CI loudly. It also gates the
  crates.io `publish` job.

**Publishing note.** 0.43.0 **supersedes the unpublished 0.41.0 and 0.42.0** — the
crates.io registry is still at 0.40.0, so a single publish catches all three up (0.43.0
is a strict superset). Per project convention the operator-driven `git tag v0.43.0` push
→ crates.io publish (auto via `release.yml`) is the **sole remaining step**; it is not
performed as part of any development phase.

## [0.42.0] - 2026-09-08

**BREAKING — 1.0 API Finalization Pass.** The second breaking release: it clears the
entire **API section** of the 1.0 gap checklist (`documentation/ROADMAP-TO-1.0.md`) —
sealing the `wire` interchange surface, future-proofing every public config struct with
`#[non_exhaustive]`, and finishing the `_1d`/`_2d` → `Dim`-dispatch naming unification.
**Breaking is API *shape* only — there is no numeric or behavioral change** to any
retained computation (every consolidated signature routes to a byte-identical private
body; a code-review gate confirms no drift). **MSRV is unchanged** (1.81 for the crate,
1.84 for the `linalg` feature). This is the last breaking pass before a deliberate future
1.0 cut.

### Changed

#### Surface sealing — `wire` (SEAL-01 / AUD-09 + AUD-13)

- The `wire` module is now `pub(crate)` (`pub mod wire` → `pub(crate) mod wire`). Its ~24
  interchange types (`FdaData`, `LayerKey`, `Layer`, the `*Layer` structs, …) are no
  longer reachable by external callers. There were no crate-root/prelude re-exports of
  wire types, so no re-export removals were needed. `wire` was an unwired, unsupported
  interchange surface with no JS/R consumer; it is sealed until bindings need it, at which
  point it will be re-exposed deliberately.

#### Config structs `#[non_exhaustive]` + construction idiom (SEAL-02 / SEAL-03 / AUD-12)

- `#[non_exhaustive]` added to the **38** public `*Config` structs that did not already
  carry it (of 66 total; 28 were already sealed). Fields can now be added post-1.0 without
  a breaking bump.
- **Downstream impact (construction idiom change):** external code can no longer construct
  these configs with a struct literal — **not even functional-update `..Default::default()`
  form** (Rust rejects any struct expression on a `#[non_exhaustive]` type from another
  crate). All 38 already implement `Default`; construct them via
  `let mut c = SomeConfig::default(); c.field = …;` and pass `c`. Each struct's docs show
  the supported path.

#### Naming unification — small families (NAME-01/02/03/05 / AUD-19/20/21/23)

- `geometric_median_1d`/`_2d` → **`geometric_median(…, dim: Dim)`**.
- `hausdorff_self_1d`/`_2d`, `hausdorff_cross_1d`/`_2d` → **`hausdorff_self(…, dim: Dim)`** /
  **`hausdorff_cross(…, dim: Dim)`** (`hausdorff_3d` is unchanged — it takes point clouds).
- `functional_spatial_1d`/`_2d` → **`functional_spatial(…, dim: Dim)`**;
  `kernel_functional_spatial_1d`/`_2d` → **`kernel_functional_spatial(…, dim: Dim)`**.
- `LpeerResult` → **`LocalPeerResult`** (matches the `PeerResult` sibling), incl. `lib.rs`/
  `prelude.rs` re-exports.

#### Naming unification — large suffix batch (NAME-04 / AUD-22)

- **41** remaining lone `_1d` functions across `depth`, `metric`, and `fdata` are now
  reached through a single `Dim`-dispatched public signature per family — e.g. `band`,
  `hypograph_index`, `total_variation_depth`, `center`, `norm_lp`, and the metric
  self/cross families `dtw_self`/`dtw_cross`, `fourier_self`/`fourier_cross`,
  `pca_self`/`pca_cross`, `soft_dtw_self`/`soft_dtw_cross`, etc. Call the old `_1d` sites as
  `name(…, Dim::One)`.
- **5** genuinely-1D-only functions are plain-renamed to drop the now-meaningless `_1d`
  suffix (no `Dim` parameter): **`fdata_to_pc`**, **`fdata_to_pls`**, **`fourier_fit`**,
  **`pspline_fit`**, **`select_basis_auto`**.
- Unchanged: the `*_2d` functions with no `_1d` sibling (`fosr_2d`, `predict_fosr_2d`,
  `simpsons_weights_2d`), the `*_1d_seeded` RNG variants, and the R-interop shims
  (`fdata_to_basis_1d`, `basis_to_fdata_1d`).

### Migration

- Replace struct-literal config construction with `Config::default()` + field assignment.
- Replace a removed `name_1d(args)` call with `name(args, Dim::One)` (for consolidated
  families) or the bare `name(args)` (for the 5 plain-renamed functions).
- Rename `LpeerResult` → `LocalPeerResult`.
- All 28 bundled examples and all doctests were migrated to the new surface and compile
  against it — they are worked references for the new call forms.

## [0.41.0] - 2026-09-07

**BREAKING — 1.0 API Stabilization Pass.** This is the first breaking release after a
long additive run. It settles the public API ahead of a future 1.0 by removing
long-deprecated dimensional forms, sealing internals that were unintentionally public,
marking forward-compatible enums/structs `#[non_exhaustive]`, and unifying naming.
**Breaking is API *shape* only — there is no numeric or behavioral change** to any
retained computation. **MSRV is unchanged** (1.81 for the crate, 1.84 for the `linalg`
feature). See `documentation/STABILITY.md` for the semver + MSRV policy and
`documentation/ROADMAP-TO-1.0.md` for the remaining 1.0 gap checklist.

### Removed

- **API-01 — deprecated dimensional forms removed.** The 6 long-deprecated `*_2d`/
  legacy forms and their crate-root/prelude re-exports are gone:
  - `mean_2d` → use `mean(…, Dim::Two)`
  - `fanova` → use `fanova_seeded(…, 42)` (pass an explicit seed for determinism)
  - `random_tukey_2d` → use `random_tukey(…, Dim::Two)`
  - `random_projection_2d` → use `random_projection(…, Dim::Two)`
  - `fraiman_muniz_2d` → use `fraiman_muniz(…, Dim::Two)`
  - `modal_2d` → use `modal(…, Dim::Two)`

### Changed

#### Surface sealing (API-02)

- `sort_nan_safe` and `solve_gaussian_pub` are now `pub(crate)` (they were
  unintentionally `pub`). These were internal numerical helpers with no intended
  external contract; there is no public replacement.

#### `#[non_exhaustive]` (API-03)

- `#[non_exhaustive]` added to 10 public enums — `PeerPenalty`, `LambdaChoice`,
  `LambdaMethod`, `DesignCriterion`, `OptimalityKind`, `ExtrapolationPolicy`,
  `ImputationMethod`, `SelectionCriterion`, `BasisType`, `BasisCriterion` — and to 2
  result structs — `OptimBandwidthResult`, `KnnCvResult`.
  - **Downstream impact:** exhaustive `match` on these enums now requires a wildcard
    (`_ => …`) arm, and struct-literal construction of the two result structs from
    outside the crate is no longer possible; use the constructing API and read fields.
    This allows future variants/fields to be added without a breaking bump.

#### Naming unification (API-04)

- Renamed for casing/consistency:
  - `funhddC_cluster` → `fun_hddc_cluster`
  - `FosrResult2d` → `Fosr2dResult`
  - `GmmResult` → `GmmFitResult`
  - **Migration:** update call sites / type references to the new names (signatures
    and semantics are unchanged).
- Collapsed dimensional pairs into single dispatchers over a domain enum:
  - `deriv_1d` / `deriv_2d` → `deriv(…, DerivDomain)` returning `DerivResult`.
    **Migration:** call `deriv(&data, DerivDomain::OneD { argvals, nderiv })` and match
    `DerivResult::OneD(m)` (or `DerivDomain::TwoD { … }` / `DerivResult::TwoD(…)`).
  - `lp_self_1d` / `lp_cross_1d` / `lp_self_2d` / `lp_cross_2d` → `lp_self` / `lp_cross`
    taking an `LpDomain`. **Migration:** call `lp_self(…, LpDomain::OneD { … })` (or
    `LpDomain::TwoD { … }`) instead of the dimension-suffixed variants.

### Added

- **Stability documentation (API-04 / STAB deliverables):**
  - `documentation/STABILITY.md` — the semver + MSRV policy (0.x breaking-in-minor
    rule, feature-gate MSRV, deprecation approach).
  - `documentation/ROADMAP-TO-1.0.md` — the remaining 1.0 gap checklist.
- **New public types backing the collapsed APIs:** the `DerivDomain` and `LpDomain`
  domain-selector enums and the `DerivResult` return type.

## [0.40.0] - 2026-09-07

Release Hardening & Ship v0.40.0 (Phase 80). Folds in the unpublished v0.39.0
forward-mode AD core and applies correctness / build fixes from Phases 78 and 79.

### Fixed

- **CORR-01 — soft-DTW backward endpoint-seed gradient bug**: `soft_dtw_backward`
  previously overwrote the required `E[n][m] = 1` endpoint seed in its reverse loop,
  producing an all-zero gradient for every call. All gradient-dependent code paths
  (`soft_dtw_barycenter`, the generic autodiff path) now receive the correct gradient.
  (Phase 78)
- **BUILD-01 — serde feature build**: `cargo build --features serde` now compiles
  cleanly. The build had been broken since Phase 60 due to `ShapeletTransformClassifier`
  embedding a non-serde `ClassifFit`; serde derives have been added to `ClassifFit` and
  all affected types. (Phase 79)

### Changed

- **soft-DTW barycenter convergence (intentional behavior change)**: `soft_dtw_barycenter`
  now genuinely converges to the barycenter via an inverse-curvature optimizer step,
  replacing the previous behavior of silently returning the pointwise mean on an all-zero
  gradient. Output values will differ from prior versions wherever `soft_dtw_barycenter`
  was called. (Phase 78, SDTW-O1 for a full L-BFGS optimizer is backlogged.)
- **CORR-02 — gradient-pass audit**: All hand-written gradient implementations were
  audited; no additional bugs were found beyond CORR-01. (Phase 78)

## [0.39.0] - 2026-09-07

Forward-Mode Automatic Differentiation Core (Phases 75–77). Code-complete but never
published to crates.io; folded into this release.

### Added

- **Scalar trait + Dual number substrate (DIF-01)**: New `src/autodiff.rs` module
  exports a generic `Scalar` trait and a `Dual<T>` value/tangent number type, enabling
  forward-mode AD over arbitrary scalar types. Includes known-answer tests, central
  finite-difference cross-checks, and f64-parity tests. (Phase 75)
- **Differentiable elastic distance + FPCA scores (DIF-02, DIF-03)**: `soft_dtw_distance_generic`
  is a fully generic soft-DTW distance instantiable with `Dual<f64>` for exact gradient
  computation; `amplitude_distance_at_warp` likewise. FPCA score projection is now
  differentiable via the generic `Scalar` interface — `∂score_k/∂curve[j]` equals the
  analytic closed form `rotation[j,k] · weights[j]` to ≤1e-12. All three validation tiers
  (oracle, finite-difference, f64-parity) are covered by inline tests. (Phase 76)
- **Gradient API + composition demo (DIF-04)**: `grad(f, x)` returns `(value, gradient)`
  for any scalar objective over `Vec<f64>`; `jacobian` provides the full Jacobian matrix.
  A composition demo wires two Phase-76 differentiable ops into a single objective and
  verifies the composed gradient against central finite differences (≤1e-6). Crate-root
  and `prelude::*` re-exports for `Scalar`, `Dual`, `diff`, `grad`, and the generic
  distance functions. Module doctest passes under `cargo test --doc`. (Phase 77)

## [0.38.0] - 2026-09-05

VEESA — Elastic Shape Explainability & Conformal Anomaly Detection (Phases 72–74).
Closes the gaps against the VEESA paper (Goode, Tucker & Ries), the `sandialabs/veesa`
R package, and arXiv 2504.01172 (elastic conformal anomaly detection). Reuse-first over
the shipped jfPCA / elastic-distance / conformal machinery; no new dependency.

### Added

- **jfPCA fit/transform seam** — `jfpca_fit` returns a reusable `JfpcaModel` (stores the
  trained Karcher-mean template, `mean_q`/`mean_psi`/`mean_srsf`, joint eigenvector
  components, `balance_c`, `argvals`, eigenvalues, and the embedded `JointFpcaResult`);
  training scores reproduce `joint_fpca` within 1e-8. `JfpcaModel::transform` projects new
  out-of-sample curves onto the trained basis (aligns to the trained template via
  `align_to_target`); `JfpcaModel::score_training` gives the exact training round-trip;
  `JfpcaTransform` result. (VEE-01, VEE-02)
- **Model-agnostic permutation feature importance** — `elastic_pfi` over jfPCA PC scores,
  generic over any predictor closure `Fn(&FdMatrix) -> Vec<f64>`; `PfiMetric`
  (Mse/Mae/Accuracy/Custom); deterministic under seed; `ElasticPfiResult`. (VEE-03)
- **Principal-direction reconstruction** — `JfpcaModel::principal_directions` reconstructs
  μ ± c·σⱼ split into amplitude and phase parts (`PrincipalDirections`); `c = 0` reproduces
  the jfPCA mean. (VEE-04)
- **VEESA pipeline** — `veesa_pipeline` ties fit → transform → PFI end-to-end
  (`VeesaPipelineResult`). (VEE-05)
- **Elastic conformal anomaly detection** — `NonConformityScore` extended with
  `AmplitudeElastic` / `PhaseElastic` / `CombinedElastic`; `elastic_nonconformity` (scores a
  curve against a reference template, non-negative, zero for an identical curve);
  `elastic_conformal_anomaly` inductive detector (per-curve conformal p-values, anomaly flags
  at level α, calibrated threshold — catches magnitude AND shape outliers);
  `ConformalAnomalyConfig` / `ConformalAnomalyResult`. The existing `conformal_prediction_band`
  path is unchanged. (ECA-01, ECA-02, ECA-03)

## [0.37.0] - 2026-09-04

WAV — Wavelet-Domain Functional Regression (Phases 69–71). Promotes GAP-07.

### Added

- **Discrete wavelet transform** — new `wavelet/` module: single-level orthonormal DWT
  (Haar + Daubechies db2–db10) with exact-adjoint analysis/synthesis under periodic and
  symmetric boundaries; multi-level Mallat pyramid (`WaveletCoeffs`, `decompose`,
  `reconstruct`) plus the `FdMatrix` batch path; perfect reconstruction ≤1e-10 across
  families, levels, boundary modes, and non-power-of-2 lengths.
- **`wcr`** — wavelet-domain scalar-on-function regression (PCR or PLS on per-curve
  concatenated wavelet-coefficient designs; β(t) recovered by inverse DWT); `WcrResult` with
  out-of-sample `predict` + coefficient/fitted accessors.
- **`wnet`** — wavelet-domain elastic-net scalar-on-function regression (per-coefficient
  L1+L2 coordinate descent, deterministic cross-validated λ); `WnetResult` with `predict`.
  Full crate-root + prelude re-exports and a running end-to-end module doctest.

## [0.36.0] - 2026-09-04

PEER — Structured-Penalty & Longitudinal Scalar-on-Function Regression (Phases 66–68).
Promotes GAP-06. (Crate versions 0.36.0 and 0.37.0 were code-complete but not published
separately — their code ships in the 0.38.0 crate release.)

### Added

- **`peer`** — structured-penalty scalar-on-function regression via the null-space/range-space
  decomposition of the penalty operator; `PeerPenalty` families (Ridge, 2nd-difference,
  caller-supplied Decree Q); `PeerResult` (β(t), intercept, fitted values, effective df,
  diagnostics).
- **Automatic λ selection** — `LambdaChoice` (Fixed / Gcv / Reml): deterministic GCV grid +
  self-contained REML EM.
- **`lpeer`** — longitudinal PEER with subject random effects (REML EM via `famm`);
  `LpeerResult` with variance components. Out-of-sample `predict` on both; full re-exports +
  doctest.

## [0.35.0] - 2026-09-03

Optimal Experimental Design for Sparse FDA (FOptDes, Phases 64–65). Promotes GAP-05.
Crate bumped 0.34.0 → 0.35.0.

### Added

- **`design_criterion`** — evaluates a candidate design under `DesignCriterion::Trajectory`
  (integrated BLUP-MSE) or `DesignCriterion::Score(A|D)` (FPC-score posterior covariance A-/
  D-optimality); `OptimalityKind` enum. New `optimal_design.rs`.
- **`optimal_design`** — deterministic greedy sequential forward selection over an estimated
  `PaceFpcaResult` (read-only, no re-estimation); `OptDesConfig` / `OptDesResult`. Full
  re-exports + doctest + criterion benchmark.

## [0.34.0] - 2026-09-02

k-Shape Clustering & Shape-Based Distance (Phases 61–63). Promotes GAP-03. Crate bumped
0.33.0 → 0.34.0.

### Added

- **Shape-based distance** — `src/metric/sbd.rs`: `sbd` (FFT normalized cross-correlation →
  distance + optimal shift, scale/offset-invariant) and `sbd_distance_matrix`.
- **k-Shape clustering** — `src/kshape.rs`: `kshape_fd` (iterative SBD assignment +
  shape-extraction centroids, `n_init` restarts, deterministic seeding) and
  `KShapeResult::predict`.
- **SBD k-medoids** — `sbd_kmedoids`. Full re-exports + criterion bench.

## [0.33.0] - 2026-09-02

Shapelet Transform & Classification (Phases 57–60). Promotes GAP-02 (discovery-based;
learning-shapelets deferred). Crate bumped 0.32.0 → 0.33.0.

### Added

- **Shapelet distance core** — `src/shapelet/distance.rs`: per-window z-normalized
  min-Euclidean with early-abandon; the `Shapelet` type.
- **Discovery & ranking** — `src/shapelet/discovery.rs`: candidate generation + `QualityMeasure`
  (information gain / F-statistic) + top-K selection with self-similarity pruning
  (`ShapeletSet`); byte-deterministic.
- **Shapelet transform** — `src/shapelet/transform.rs`: `shapelet_transform`,
  `shapelet_transform_fit`, and out-of-sample `ShapeletTransformFit::transform`.
- **Bundled classifier** — `src/shapelet/classifier.rs`: `shapelet_classifier_fit`
  (discover → transform → classify; kNN default, LDA optional) + `ShapeletClassifierFit::predict`.
  Criterion bench.

## [0.32.0] - 2026-09-02

Global Alignment Kernel & Kernel Clustering (Phases 54–56). Promotes GAP-01. First
implementation milestone after three audit/consolidation cycles; crate bumped 0.30.0 → 0.32.0.

### Added

- **GAK kernel** — `metric/gak.rs`: Cuturi Triangular Global Alignment Kernel (log-domain
  forward DP atop the soft-DTW lattice, PSD in [0,1] with unit diagonal); `gak_gram_matrix`;
  `sigma_gak` median-distance bandwidth heuristic.
- **Gram-matrix export** — `gak_gram_train` / `gak_gram_predict` (cross-normalized against the
  stored training diagonals) for external `SVC(kernel='precomputed')` handoff.
- **Kernel-k-means** — `kernel_kmeans.rs`: kernel-trick clustering on the GAK Gram,
  `n_init` restarts, deterministic seeding, out-of-sample `predict`.

## [0.31.0] - 2026-09-02

Multi-Ecosystem Gap Audit (Phases 52–53). **No `fdars-core` code changes** — an audit-only
milestone producing `.planning/research/GAP-AUDIT-REPORT.md` (MATLAB, Julia, tidyfun/refund,
and Python-beyond-scikit-fda capability surveys) and a value-ranked `GAP-BACKLOG.md` (7
net-new gaps, GAP-01…GAP-08). No crate version change and no git tag.

## [0.30.0] - 2026-09-01

Covers the v0.30.0 development milestone (Performance & Consolidation Pass) — the
first internally-driven, measure-first depth pass rather than an external gap-audit.
**Behavior-preserving** (numeric outputs unchanged or provably-equivalent within
documented tolerance, proven by existing tests + before/after criterion benchmarks)
and **additive/non-breaking** — no existing public signature was removed and no new
crate dependency was added. Folds in the v0.29.0 development work as well (the
0.29.0 tag was published without a corresponding root changelog entry).

### Performance (behavior-preserving)

- **`face_covariance` −80.7% wall-time** (983.8 → 189.8 ms): the sparse FACE
  covariance estimator now precomputes per-observation Gaussian kernel-weight tables
  once instead of recomputing them per `(s, t)` grid cell (~98% fewer `exp()` calls);
  byte-equivalent output.
- **`fts::dpca` −54% allocations** (17,739 → 8,139 blocks): dynamic-PCA eigenvector
  materialization now uses an index-sort instead of staging into intermediate `Vec`s;
  golden-equivalent within 1e-12.
- **`fsvd` / `ssvd` / `functional_acf`**: eigen matrices are now built via
  `DMatrix::from_fn` (no `Vec` staging, no `m×m` copy); `functional_acf` also
  precomputes `sqrt(w)`. Byte-equivalent (golden 1e-12).
- **`fem_smooth`**: `phi_t_phi` and the assembly matrix are built in a single pass,
  dropping an `N×N` clone; byte-equivalent. (The `O(N³)` Cholesky/GCV cost is
  documented and deferred.)
- **Thread-scaling**: `frechet_anova` and co-clustering initialization gained
  feature-gated rayon parallelism via the `parallel.rs` macros, equivalence-tested
  vs. sequential (ON/OFF) with payback-threshold guards.

### Changed — internal consolidation (no API change)

- Duplicated numerical/statistical machinery factored into shared `pub(crate)`
  helpers: χ²/F survival distributions (`distributions.rs`), per-thread RNG seeding
  (`seed_for_thread`), permutation-test p-value scaffolding (`permutation_pvalue`),
  and the SVD sign-fix core — with all prior call sites migrated. Behavior unchanged.

### Added — additive API consolidation

- `fanova_seeded` — a seedable variant of the permutation `fanova` (the original
  non-seedable `fanova` is retained).
- A `Dim` dimensionality parameter plus 5 unified dispatchers over previously
  separate `_1d` / `_2d` entry points.
- New criterion `[[bench]]` coverage for the previously-unbenchmarked modules
  (`fts`, `frechet`, `boosting_regression`, `coclustering`, `fem_smoothing`,
  `density_fda`, `inference`, `fpca_variants`, `face`), plus a `BENCH-RESULTS.md`
  regression-guard ledger.

### Deprecated

- 6 redundant public forms are now marked `#[deprecated]` in favor of the unified
  alternatives above. **All deprecated signatures still compile and work** — the
  breaking removal is deferred to a future 1.0-readiness release.

## [0.28.0] - 2026-08-23

Covers the v0.28.0 development milestone (Spectral Functional Time Series &
Object-Data Fréchet Regression). Additive and non-breaking — no existing public
signature changed, no new crate dependency.

### Added — v0.28.0 (Spectral Functional Time Series, FTS-03)

- **Spectral functional time series** (`fdars-core::fts`, new `fts/spectral.rs`):
  - `spectral_density` — the spectral density operator: a Bartlett-weighted DFT (via
    `rustfft`) across the lag index of the reused `fts/acf.rs` autocovariance
    operators, evaluated at the Fourier frequencies `θ_k = 2πk/N`; per-frequency
    Hermitian m×m operator (`SpectralDensityResult`).
  - `dpca` — dynamic functional PCA: per-frequency dynamic eigen-filters (inverse-FFT
    of Simpson-metric-scaled eigenvectors) + dynamic scores over the valid interior
    (`DpcaResult`).
  - `dpca_reconstruct` — inverse dynamic filtering with a monotone-non-increasing
    integrated-L2 reconstruction error (`DpcaReconstruction`).
  - R baseline: `freqdom` / `ftsa`. Documented divergences: real-part
    (`SymmetricEigen`) eigendecomposition, `1/2π` omission, score trimming.
- **Functional VAR/VMA + FARMA simulators** (`fdars-core::simulation`):
  - `sim_fvarma` — VAR/VMA from user-supplied m×m operator kernels with Gaussian
    innovations, burn-in, deterministic `seed` (`FvarmaResult`).
  - `sim_farma` — combined AR+MA (FARMA) simulator (`FarmaResult`).

### Added — v0.28.0 (Object-Data Fréchet Regression, FRE-02)

- **Non-density `MetricSpace` backends** (`fdars-core::frechet`, new `frechet/spaces/`):
  - `SpdMatrixSpace` + `SpdMetric { Frobenius, Power(f64), LogCholesky }` — SPD
    covariance-matrix responses.
  - `CorrelationMatrixSpace`, `SphericalSpace` (geodesic exp/log + intrinsic Karcher
    mean), `NetworkSpace` (graph Laplacian), `PointProcessSpace` (intensity L2).
- **Generic Fréchet regression + ANOVA** over any `MetricSpace` backend:
  - `frechet_global_reg_space` / `frechet_local_reg_space` — return a predicted object
    per query row.
  - `frechet_anova_space` — Dubey–Müller Tₙ group-difference test (seeded permutation)
    over object responses.
  - The existing density `frechet_global_reg` / `frechet_local_reg` / `frechet_anova`
    now delegate to shared `pub(crate)` weight/Tₙ helpers — output bit-identical
    (non-breaking). R baseline: `frechet` 0.3.0.

## [0.27.0] - 2026-08-22

Published release covering the v0.25.0, v0.26.0, and v0.27.0 development
milestones (the 0.25.0 and 0.26.0 milestones shipped code but were never
published as separate crate versions; their additions are included here).

### Added — v0.27.0 (Functional Time Series & Fréchet Regression)

- **Functional time-series forecasting** (`fdars-core::fts`, new `fts/forecast.rs`):
  - `ftsm` — FPCA-based functional time-series model (decompose a time-ordered
    curve series via `fdata_to_pc_1d`, fit an independent Yule-Walker AR(p) with
    AIC order selection to each FPC-score sequence).
  - `ftsm_forecast` / `ftsm_forecast_multistep` — h-step-ahead FPC-score AR
    forecasts reconstructed into curves (iterative plug-in; `h = 1` is bit-identical
    across the two entry points).
  - `ftsm_update` — dynamic forecast update projecting new observation(s) onto the
    frozen FPC loadings and re-fitting the score AR models without refitting FPCA.
  - `fplsr` — functional PLS forecasting variant (lag-1 per-evaluation-point PLS).
  - Result types: `FtsmResult`, `FtsmForecastResult`, `ArModelResult`, `FplsrResult`.
  - R baseline: `ftsa`. Deterministic; reuses `fdata_to_pc_1d` + `scoring.rs` +
    `fts/acf.rs`.
- **Fréchet / object-data regression + statistics** (`fdars-core::frechet`, new
  `frechet/` module):
  - `MetricSpace` trait (distance + weighted-Fréchet-mean solver) with a
    `WassersteinDensitySpace` (1D-Wasserstein density) backend.
  - `wasserstein2_distance` — 1D 2-Wasserstein distance (quantile-L²).
  - `frechet_mean` / `frechet_variance` — sample Fréchet mean and variance.
  - `frechet_global_reg` (Petersen–Müller global linear weights) and
    `frechet_local_reg` (local-linear Gaussian-kernel weights) — conditional
    density-response regression over Euclidean predictors.
  - `frechet_anova` — Dubey–Müller group-difference test (seeded permutation
    p-value + asymptotic χ²(k−1)).
  - Result types: `FrechetGlobalRegResult`, `FrechetLocalRegResult`,
    `FrechetAnovaResult`.
  - R baseline: `frechet`. Reuses DENS-01's `density_fda.rs` quantile/Wasserstein
    machinery. Divergences (documented in rustdoc): signed-weight regression uses a
    sort-based isotonic projection instead of R's `osqp` QP; the Fréchet-ANOVA
    σ̂ₗ² variance estimator is `[ASSUMED]` (the permutation p-value is the primary,
    robust inference).

### Added — v0.26.0 (FPCA Breadth & Sparse Covariance)

- **Specialized FPCA variants** (`fpca_variants.rs`): `fpca_der`, `fsvd`
  (functional SVD / cross-FPCA via Gram-matrix eigendecomposition),
  `cross_covariance`, `dynamical_correlation`, `ssvd` (sandwich-smoother FPCA).
- **Sparse fast covariance & trajectory bands** (`irreg_fdata/face.rs`):
  `face_covariance` (FACE), `mface_covariance` (+ `MfaceCovResult`),
  `face_trajectory`.

### Added — v0.25.0 (Serial Dependence, Representation & Density Breadth)

- **Functional serial-dependence tooling** (`fts/acf.rs`): functional ACF/PACF with
  white-noise bands, a stationarity test, long-run covariance, and functional
  differencing.
- **Basis-system completions**: `monomial_basis` / `exponential_basis` /
  `power_basis` / `polygonal_basis` factories, a `MultiFunData` container, an
  `Lfd` linear-differential-operator object, and `principal_differential_analysis`.
- **Density object-data FDA** (`density_fda.rs`): log-quantile-density (LQD)
  transform + inverse, LQD-FPCA, 1D Wasserstein barycenter, density normalization.

### Notes

- Verified: whole-crate suite (2460 lib + 172 doc tests) green;
  `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean;
  `cargo fmt --check` clean.
- Per-milestone detail is archived under `.planning/milestones/`.

## [0.24.0] and earlier

See the git history and `.planning/milestones/` archives. Published crate versions
through 0.24.0 correspond to their matching `v0.X.0` git tags.
