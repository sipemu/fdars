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
