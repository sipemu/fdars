# Codebase Concerns

**Analysis Date:** 2026-08-07

## Tech Debt

**Matrix Dimension Safety (Medium Priority):**
- Issue: `FdMatrix::column()`, `FdMatrix::column_mut()`, `FdMatrix::row_to_buf()` use unchecked indexing with `debug_assert!` rather than runtime bounds checking. Out-of-bounds column access panics in release mode.
- Files: `src/matrix.rs` (lines 127-140, 157-168)
- Impact: Library users can trigger panics with invalid column indices. Panics are not part of the public API contract and reduce robustness.
- Fix approach: Replace `debug_assert!` with explicit range checks that return `Result<T, FdarError>` for the panic-prone accessors. Alternatively, document clearly that bounds must be validated by caller.

**Unwrap/Expect in Tests and Hot Paths:**
- Issue: 287 instances of `.clone()`, `.to_vec()`, and `.clone_from_slice()` throughout codebase. 50+ instances of `.unwrap()` and `.expect()` in test code and utility functions. `partial_cmp().unwrap()` in parallel code (line 174 of `src/parallel.rs`) assumes f64 comparisons always succeed.
- Files: Multiple files including `src/parallel.rs:174`, `src/function_on_scalar.rs` (test cases), `src/elastic_fpca.rs` (test cases)
- Impact: Panics on NaN comparisons; test failures cascade. Production code relies on tests not crashing.
- Fix approach: Use `.unwrap_or()` with explicit fallback ordering. For NaN handling, check `.is_finite()` before comparison or use `total_cmp()` (Rust 1.81+, already compatible with MSRV 1.81).

**Floating-Point Equality Checks:**
- Issue: Direct `==` comparisons with f64 values (e.g., `ss_tot == 0.0`, `ls == 0.0 && lt == 0.0`, `lambda_s_final == 0.0`) without epsilon tolerance. Lines 107/511 in `src/explain/importance.rs`, line 363 in `src/function_on_scalar_2d.rs`.
- Files: `src/function_on_scalar_2d.rs`, `src/explain/importance.rs`, `src/seasonal/change.rs`, others
- Impact: Numerical instability; thresholds can fail due to rounding errors. GCV, AIC, BIC calculations may miss zero conditions.
- Fix approach: Define epsilon constant (`const EPS: f64 = 1e-15`) and use `(value).abs() < EPS` or `.is_zero_approx()` helper. Already done in some paths (e.g., ALE computation line 97).

**Dead Code:**
- Issue: `cwt_morlet` function marked `#[allow(dead_code)]` in `src/seasonal/mod.rs:491`. Morlet wavelet implementation exists but is never called by the public API.
- Files: `src/seasonal/mod.rs` (lines 491-520)
- Impact: Code is tested but not used; may indicate incomplete seasonal analysis feature. Confuses maintainers about coverage.
- Fix approach: Either expose `cwt_morlet` as public API with documentation, or remove it. If future feature, document intention.

## Known Bugs (Recent Fixes)

**B-Spline Round-Trip Transposition (FIXED in v0.14.0):**
- Issue: `fdata_to_basis()` and `basis_to_fdata()` scrambled multi-curve results (GH #33). For n > 1 curves, the row-major flat buffer was incorrectly passed as column-major to `FdMatrix::from_column_major()`, producing coefficients/reconstruction outside data range and non-monotone in `n_basis`. Single-curve case coincided so it worked.
- Files: `src/basis/projection.rs` (fixed lines 35+), `src/smooth_basis.rs` (fixed lines 121+)
- Status: **FIXED** in commit 2fb6d3c9. Regression tests added to catch recurrence.

**B-Spline CV Criterion Always Selected Max n_basis (FIXED in v0.14.0):**
- Issue: `basis_nbasis_cv()` evaluated test curves against themselves (in-sample residual with no hold-out), resulting in monotone decreasing GCV score favoring maximum `n_basis` (GH #33).
- Files: `src/smooth_basis.rs` (fixed lines 121+)
- Status: **FIXED** in commit 2fb6d3c9. Changed to time-point cross-validation (fit on retained points, predict held-out points) — now shows interior minimum on noisy data.

**Elastic Alignment Gaussian Model Midpoint Anchor (FIXED in v0.14.0):**
- Issue: `gauss_model()` and `joint_gauss_model()` reconstructed curves using level coordinate at domain MIDPOINT (m/2) but passed it to `srsf_inverse()` as the value at `argvals[0]`, causing constant shift ~+1.0. Sample mean did not match data mean (GH #34).
- Files: `src/alignment/generative.rs` (fixed lines 63+)
- Status: **FIXED** in commit 6ed62398. Reconstruct from zero start, then shift curve so its midpoint equals recovered level. Regression test added.

**GMM Over-Splitting (FIXED in v0.13.2):**
- Issue: Gaussian Mixture Model clustering split clusters excessively due to missing data-scaled covariance floor. Small floating-point differences inflated component counts.
- Files: `src/gmm/em.rs`
- Status: **FIXED** in commit ec17d138 (v0.13.2). Added covariance floor scaling to data variance.

## Security Considerations

**Dependency Age (Non-Critical):**
- Risk: `faer` (0.23) requires Rust 1.84.0; gates `linalg` feature behind version check. R package (fdars-r) uses `default-features = false` to avoid this constraint.
- Files: `Cargo.toml` (lines 39-40), `src/lib.rs:47`
- Current mitigation: Feature flag separation; `linalg` is opt-in. Crate can build on Rust 1.81 without it.
- Recommendations: Document MSRV (1.81) clearly in README; clarify `linalg` requires 1.84+ in feature docs.

**Rand + StdRng Seeding (Adequate):**
- Risk: Per-thread RNG seeding (`StdRng::seed_from_u64(seed + k as u64)`) uses simple addition, not cryptographically secure. However, context is statistical/simulation, not cryptography.
- Files: `src/parallel.rs` (documented), `src/outliers.rs:154`, `src/elastic_explain.rs:314`
- Current mitigation: Seeding is deterministic by design (reproducibility). Not security-sensitive.
- Recommendations: None — design is appropriate for FDA algorithms.

**No Input Sanitization (Implicit Contracts):**
- Risk: Functions assume caller passes valid dimensions (`argvals.len() == data.ncols`, `data.nrows == y.len()`). No per-function guards on dimension mismatch.
- Files: All public functions with matrix inputs
- Current mitigation: Many functions return `Result<T, FdarError>` with validation. Examples: `src/regression.rs` lines 74-83, `src/matrix.rs` lines 54-62.
- Recommendations: Audit critical paths (regression, classification, alignment) for complete validation coverage. Consider adding a `validate_dimensions!` macro.

## Performance Bottlenecks

**Dense Matrix Reconstruction (Moderate Impact):**
- Problem: Row gathering in `FdMatrix::row()` is O(ncols) due to column-major layout; naïve transpose or full-matrix reconstruction is O(n*m). Used in distance calculations and cross-validation loops.
- Files: `src/matrix.rs:146-150`, callers in `src/distance.rs`, `src/cv.rs`
- Cause: Column-major storage is optimal for FPCA/SVD workflows but pessimal for row access.
- Improvement path: Provide `row_to_buf()` (zero-allocation) for hot paths. Consider caching transposed copy in expensive algorithms, or use block-wise access patterns. Already offers `row_to_buf()` (line 157+); ensure callers use it.

**Parallel Overhead in Small Problems:**
- Problem: Rayon parallelization adds overhead for n < ~100. Many examples and tests use small matrices.
- Files: All code using `iter_maybe_parallel!` macro (`src/parallel.rs`)
- Cause: Rayon's thread pool spinup dominates for tiny workloads.
- Improvement path: Benchmark threshold; disable parallelism below threshold (e.g., n < 50). Could add `.sequential()` hint in macros for callers. Currently relies on user opting out via `features = []`.

**Numerical Quadrature in Penalty Matrices (Minor Impact):**
- Problem: `bspline_penalty_matrix()` and `fourier_penalty_matrix()` use Simpson's rule with 10 sub-points per interval. For fine grids (m >> 100), this is expensive; coarser quadrature might suffice.
- Files: `src/smooth_basis.rs` (lines 87-110, 150-180)
- Cause: Default accuracy; no user control.
- Improvement path: Add optional parameter for quadrature fineness. Document convergence empirically for typical use cases.

## Fragile Areas

**Elastic Alignment Complex Pipeline:**
- Files: `src/alignment/nd.rs` (973 lines), `src/elastic_fpca.rs` (1252 lines), `src/alignment/generative.rs` (500+ lines)
- Why fragile: Recent bugs (GH #33, #34) show coordinate system conventions and level encoding are error-prone. SRSF (square-root slope function) involves SOS curves and phase/amplitude decomposition. Multi-dimensional alignment uses DP with complex state indexing.
- Safe modification: When changing level encoding, anchor reconstruction (midpoint shift), or coordinate transforms, add regression tests asserting: (1) sample mean matches data mean, (2) round-trip residuals stay in-range, (3) single-curve and multi-curve behaviors match.
- Test coverage: Gaps exist in generative model test scenarios; only peak-1 bump tested. Add tests for step functions, linear trends, periodic signals.

**Basis Representation and CV (Post-Fix Fragility):**
- Files: `src/smooth_basis.rs` (2806 lines), `src/basis/projection.rs` (200+ lines)
- Why fragile: Matrix layout transpose bug (GH #33) and CV criterion selection bug (GH #33) were not caught by existing tests. Both were silent correctness issues. Current regression tests added but coverage still narrow (smooth + noise data only).
- Safe modification: Changes to `fdata_to_basis()`, `basis_to_fdata()`, or `*_cv()` functions require adding regression tests for: (1) round-trip identity on n=1 and n>1 cases, (2) monotone improvement with n_basis on smooth data, (3) non-monotone (interior min) on noisy data. Use assertions on GCV/AIC/BIC values.
- Test coverage: Gaps in edge cases (single point, identical curves, near-singular covariance).

**Seasonal Strength and Period Estimation (Moderate Fragility):**
- Files: `src/seasonal/strength.rs`, `src/seasonal/mod.rs` (1192 lines), `src/seasonal/lomb_scargle.rs`
- Why fragile: Period estimation uses Lomb-Scargle FFT-based power spectrum; NaN/Inf handling occurs post-hoc via filtering (line 595: `filter(|x| x.is_finite())`). If upstream code returns unexpected NaN, it silently drops values rather than erroring.
- Safe modification: Validate output of seasonal algorithms before returning. Example: assert mean/strength is finite, component decompositions sum to input. Document edge cases (period == 0, noise >> signal).
- Test coverage: Good coverage of typical cases; gaps in extreme noise/constant signal scenarios.

**Explain Module with Generic Trait:**
- Files: `src/explain_generic.rs` (1000+ lines), `src/explain/advanced.rs` (1130 lines), `src/explain/helpers/` (10 files)
- Why fragile: `FpcPredictor` trait is generic over regression/classification tasks. Implementation must handle 3 task types (Regression, BinaryClassification, MulticlassClassification(k)) correctly. Span-wise predictions must match per-curve predictions. Importances must sum correctly.
- Safe modification: Changes to prediction, scoring, or importance aggregation require round-trip tests: (1) PDP values match direct span-wise computation, (2) importance sums < total variance, (3) SHAP values sum to gap between prediction and baseline.
- Test coverage: Comprehensive (1797 test cases) but mocked data; benefits from real-world dataset testing.

## Scaling Limits

**Memory Usage in Large n (Observations):**
- Current capacity: FdMatrix stores full n×m matrix in memory. For n=10,000, m=1,000, this is ~80 MB (feasible). For n=1,000,000, m=1,000, this is ~8 GB (marginal). Regression/classification operations compute n×k or n×n covariance (e.g., elastic alignment all-pairs DP is O(n² * m²)).
- Limit: O(n²) algorithms (k-means, GMM, alignment) break at n > 10,000 on typical hardware (16 GB RAM).
- Scaling path: (1) Implement mini-batch / streaming variants. Examples: `streaming_depth` already exists. Extend to `streaming_kmeans`, `streaming_elastic_align`. (2) Add sparse/block-wise covariance for large-n regression. (3) Support external storage (memory-mapped matrices).

**Computation Time in Large m (Evaluation Points):**
- Current capacity: SVD on m×m covariance matrix is O(m³); for m=10,000 this takes seconds. Basis expansions (k=50 basis funcs) multiply cost by k. Quadrature integration is O(m * n_quad).
- Limit: FPCA/regression on m > 5,000 evaluation points becomes slow (1-10 sec) even for n < 100.
- Scaling path: (1) Provide low-rank approximations (truncated SVD). Already done via `ncomp` parameter. (2) Use Nystrom approximation for covariance. (3) Implement fast DCT-based smoothing for Fourier bases (avoid full matrix solve).

**Elastic Alignment All-Pairs DP:**
- Current capacity: `elastic_align_many()` computes pairwise alignments, O(n² * m²) DP. For n=100, m=500, this is manageable (< 1 sec). For n=1,000, m=500, this becomes 250 million comparisons (~60 sec).
- Limit: n > 1,000 with alignment is impractical.
- Scaling path: (1) Use barycenter averaging instead of all-pairs (more efficient). (2) Implement Sakoe-Chiba band to reduce DP state space (already added v0.14.0). (3) Use approximate nearest-neighbor search for initialization.

## Dependencies at Risk

**faer (0.23, optional):**
- Risk: Depends on Rust 1.84.0+; if new Rust versions break MSRV, `linalg` feature becomes unavailable. Currently gated behind feature flag, so core functionality not at risk.
- Impact: Ridge regression (`ridge_regression_fit`) unavailable on older Rust. R package (fdars-r) uses default-features = false to avoid dependency.
- Migration plan: Keep faer optional; add alternative ridge via anofox-regression (already used). If faer API breaks, switch to `ndarray-linalg` or `la-rs` (both support older MSRV).

**rayon (1.10, optional):**
- Risk: Parallelism feature is optional; if rayon is deprecated, fall back to sequential via `#[cfg(not(feature = "parallel"))]`. No breaking risk.
- Impact: Low — core algorithms work without parallelism (tested via feature gate).
- Migration plan: Maintain sequential codepaths in macros. If rayon breaks, switch to `crossbeam` or `std::thread` (lower-level but stable).

**nalgebra (0.33):**
- Risk: nalgebra is used for SVD (via `nalgebra::SVD`). Major version changes may alter API. Currently pinned to 0.33 (semi-stable).
- Impact: Core FPCA, PCA, depth calculations depend on nalgebra. Breaking change would require significant refactoring.
- Migration plan: Monitor nalgebra releases. If API breaks in 0.34+, pin version and plan migration over 1-2 releases. Alternatively, implement custom compact SVD for the 1-2 leading components (most use cases).

**getrandom (0.2, optional):**
- Risk: WASM compatibility via `getrandom/js` feature. Older Node versions may not provide `crypto.getRandomValues()`.
- Impact: WASM builds fail on very old browsers/environments.
- Migration plan: Already handled via feature gating. Update docs to specify minimum WASM runtime version.

## Missing Critical Features

**Feature: Streaming Elastic Alignment:**
- Problem: All-pairs alignment is O(n²). No streaming variant for incremental curve arrival.
- Blocks: Large-scale time-series registration, online shape clustering.
- Priority: Medium (specialized use case).

**Feature: Cross-Domain Functional Regression:**
- Problem: Scalar-on-function and function-on-scalar are implemented, but not full Functional-on-Functional Regression with proper domain alignment and smoothness penalties beyond basic approaches.
- Blocks: Some advanced FDA workflows. Partial support exists in `src/fof_regression.rs`.
- Priority: Low (existing methods cover most cases).

**Feature: Conformal Inference for Confidence Bands:**
- Problem: Tolerance bands and conformal methods exist (`src/tolerance/`, `src/conformal/`), but coverage of multivariate quantile prediction is incomplete.
- Blocks: Rigorous uncertainty quantification in production models.
- Priority: Medium (important for decision-making systems).

## Test Coverage Gaps

**Elastic Alignment Edge Cases (Low Coverage):**
- What's not tested: Alignment of curves with zero derivatives, near-constant curves, curves with discontinuities (spikes). Phase/amplitude decomposition under extreme warping.
- Files: `src/alignment/tests.rs`, `src/alignment/generative.rs`
- Risk: Edge cases may produce NaN/Inf or incorrect phase/amplitude split. Silent failures possible.
- Priority: High (alignment is core feature).

**Basis Representation on Non-Uniform Grids (Minimal Coverage):**
- What's not tested: B-spline and Fourier basis on highly non-uniform `argvals` (e.g., logarithmic spacing, clustered points). Penalty matrix computation under domain compression.
- Files: `src/smooth_basis.rs`, `src/basis/pspline.rs`
- Risk: Quadrature accuracy may degrade; GCV/AIC selection may fail.
- Priority: Medium (less common use case).

**Classification on High-Dimensional Functional Data (Incomplete Coverage):**
- What's not tested: LDA/QDA on n > 1,000 with m > 500 (near-singular covariance). Multiclass classification with k > 10 classes (rare but possible). Kernel classifiers with bandwidth selection on edge cases.
- Files: `src/classification/`, `src/classification/lda.rs`, `src/classification/kernel.rs`
- Risk: Numerical instability, overfitting on high-dimensional problems.
- Priority: Low (specialist use).

**Seasonal Strength on Irregular Data (No Coverage):**
- What's not tested: Seasonal strength estimation when periods are fractional or data has large gaps. Lomb-Scargle on highly irregular grids.
- Files: `src/seasonal/strength.rs`, `src/seasonal/lomb_scargle.rs`
- Risk: Power spectrum estimates may be meaningless; period detection can fail silently.
- Priority: Medium (irregular data is increasingly common).

---

*Concerns audit: 2026-08-07*
