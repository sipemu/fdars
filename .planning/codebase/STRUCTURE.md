# Codebase Structure

**Analysis Date:** 2026-08-07

## Directory Layout

```
fdars/
├── Cargo.toml                      # Workspace root (single member: fdars-core)
├── Cargo.lock                      # Locked dependency versions
├── README.md                       # Project overview & examples
├── codecov.yml                     # Code coverage configuration
│
├── fdars-core/                     # Main crate (Rust 1.81+, edition 2021, v0.14.0)
│   ├── Cargo.toml                  # Core crate manifest (features: parallel, linalg, serde, js)
│   ├── src/                        # Source code
│   │   ├── lib.rs                  # Module declarations & public re-exports (~350 lines)
│   │   ├── prelude.rs              # Convenience re-exports for common types
│   │   ├── error.rs                # FdarError enum (4 variants)
│   │   ├── matrix.rs               # FdMatrix type (column-major, n×m)
│   │   ├── parallel.rs             # 5 conditional parallelism macros
│   │   ├── linalg.rs               # Shared: Cholesky, OLS, Mahalanobis (always available)
│   │   ├── helpers.rs              # Numerical integration & utility functions
│   │   ├── validation.rs           # Input dimension/parameter validation
│   │   ├── utility.rs              # General utilities
│   │   ├── test_helpers.rs         # Shared test utilities (internal)
│   │   │
│   │   ├── alignment/              # Elastic curve registration & shape analysis (10 files)
│   │   │   ├── mod.rs              # Module re-exports
│   │   │   ├── elastic_depth.rs    # Shape depth measures
│   │   │   ├── karcher.rs          # Karcher mean/median for curves
│   │   │   ├── phase_boxplot.rs    # Phase variation visualization
│   │   │   ├── pairwise.rs         # Pairwise alignment logic
│   │   │   ├── geodesic.rs         # Geodesic paths on shape space
│   │   │   ├── bayesian.rs         # Bayesian alignment
│   │   │   ├── constrained.rs      # Constrained alignment with landmarks
│   │   │   ├── outlier.rs          # Outlier detection for shapes
│   │   │   └── ... (6 more submodules)
│   │   │
│   │   ├── basis/                  # Basis representations for smoothing
│   │   │   ├── mod.rs              # B-spline, Fourier, polynomial bases
│   │   │   └── ... (submodules)
│   │   │
│   │   ├── classification/         # LDA, QDA, kNN, kernel classifiers (9 files)
│   │   │   ├── mod.rs              # Main interface
│   │   │   ├── lda.rs              # Linear Discriminant Analysis
│   │   │   ├── qda.rs              # Quadratic Discriminant Analysis
│   │   │   ├── knn.rs              # k-Nearest Neighbors
│   │   │   ├── kernel.rs           # Kernel-based classification
│   │   │   ├── dd.rs               # Depth-based classifier
│   │   │   ├── fit.rs              # Training interface
│   │   │   ├── cv.rs               # Cross-validation for model selection
│   │   │   └── tests.rs
│   │   │
│   │   ├── explain/                # Explainability for regression/logistic (10 files + helpers/)
│   │   │   ├── mod.rs              # 44+ public functions
│   │   │   ├── pdp.rs              # Partial dependence plots (1D & functional)
│   │   │   ├── shap.rs             # SHAP values (functional coefficients)
│   │   │   ├── ale_lime.rs         # ALE & LIME explanations
│   │   │   ├── sensitivity.rs      # Friedman H-statistic, Sobol indices
│   │   │   ├── importance.rs       # Permutation importance
│   │   │   ├── advanced.rs         # Counterfactual, prototype criticism
│   │   │   ├── diagnostics.rs      # Influence diagnostics (DFbetas, DFFits)
│   │   │   ├── helpers/            # 10 helper modules
│   │   │   │   ├── ale_lime.rs     # Core ALE/LIME computation
│   │   │   │   ├── shap_helpers.rs # SHAP computation details
│   │   │   │   ├── permutation.rs  # Permutation sampling
│   │   │   │   ├── saliency.rs     # Saliency computation
│   │   │   │   ├── projection.rs   # FPC score projection
│   │   │   │   └── ... (5 more)
│   │   │   └── tests.rs
│   │   │
│   │   ├── explain_generic/        # Generic explainability via FpcPredictor trait (12 files)
│   │   │   ├── mod.rs              # FpcPredictor trait + TaskType enum
│   │   │   ├── pdp.rs              # generic_pdp (works with any FpcPredictor)
│   │   │   ├── shap.rs             # generic_shap_values
│   │   │   ├── lime.rs             # generic_lime
│   │   │   ├── ale.rs              # generic_ale
│   │   │   ├── importance.rs       # generic_permutation_importance, etc.
│   │   │   ├── saliency.rs         # generic_saliency, domain_selection
│   │   │   ├── sensitivity.rs      # Friedman H, Sobol indices, VIF
│   │   │   ├── counterfactual.rs   # Counterfactual search
│   │   │   ├── anchor.rs           # Anchor explanations
│   │   │   └── ... (more)
│   │   │
│   │   ├── scalar_on_function/     # Scalar outcome, functional predictors (10 files)
│   │   │   ├── mod.rs              # Main interface
│   │   │   ├── fregre_lm.rs        # Linear model (FPC + scalar covariates)
│   │   │   ├── logistic.rs         # Functional logistic regression
│   │   │   ├── pls.rs              # Partial least squares
│   │   │   ├── robust.rs           # L1/Huber robust regression
│   │   │   ├── nonparametric.rs    # Kernel-based nonparametric regression
│   │   │   ├── cv.rs               # Cross-validation & model selection
│   │   │   ├── bootstrap.rs        # Confidence intervals via bootstrap
│   │   │   ├── multi.rs            # Multi-response regression
│   │   │   └── tests.rs
│   │   │
│   │   ├── function_on_scalar.rs   # Functional outcome, scalar predictors (1D FOSR)
│   │   ├── function_on_scalar_2d.rs # 2D FOSR with tensor-product penalty
│   │   │
│   │   ├── seasonal/               # Period detection, peak finding, decomposition (12 files)
│   │   │   ├── mod.rs              # Main interface
│   │   │   ├── autoperiod.rs       # Automatic period detection
│   │   │   ├── period.rs           # Period estimation methods
│   │   │   ├── peak.rs             # Peak detection & classification
│   │   │   ├── sazed.rs            # SAZED seasonal decomposition
│   │   │   ├── strength.rs         # Seasonal strength measures
│   │   │   ├── change.rs           # Change point detection
│   │   │   ├── hilbert.rs          # Hilbert transform
│   │   │   ├── matrix_profile.rs   # Subsequence matching
│   │   │   ├── ssa.rs              # Singular Spectrum Analysis
│   │   │   ├── lomb_scargle.rs     # Lomb-Scargle periodogram
│   │   │   └── tests.rs
│   │   │
│   │   ├── depth/                  # Depth measures (8 files)
│   │   │   ├── mod.rs              # Band, Fraiman-Muniz, modal, etc.
│   │   │   └── ... (submodules)
│   │   │
│   │   ├── streaming_depth/        # Online depth computation (7 files)
│   │   │   ├── mod.rs
│   │   │   └── ... (submodules)
│   │   │
│   │   ├── elastic_regression/     # Shape-based regression (varies)
│   │   ├── elastic_fpca/           # Elastic FPCA (vertical, horizontal, joint)
│   │   ├── elastic_changepoint.rs  # Changepoint detection on curves
│   │   ├── elastic_explain.rs      # Attribution for elastic models
│   │   │
│   │   ├── spm/                    # Statistical Process Monitoring (20 files)
│   │   │   ├── mod.rs              # Main interface
│   │   │   ├── phase.rs            # Phase 1 (reference set control limits)
│   │   │   ├── monitor.rs          # Phase 2 (monitor incoming data)
│   │   │   ├── ewma.rs             # EWMA charts
│   │   │   ├── cusum.rs            # CUSUM charts
│   │   │   ├── mewma.rs            # Multivariate EWMA
│   │   │   ├── amewma.rs           # Adaptive MEWMA
│   │   │   ├── control.rs          # Control limit computation
│   │   │   ├── stats.rs            # Hotelling T², SPE
│   │   │   ├── rules.rs            # Nelson rules, Western Electric rules
│   │   │   ├── contrib.rs          # Contribution analysis
│   │   │   ├── arl.rs              # Average Run Length
│   │   │   ├── elastic_spm.rs      # Elastic shape monitoring
│   │   │   ├── mfpca.rs            # Multivariate functional PCA for SPM
│   │   │   └── ... (more)
│   │   │
│   │   ├── gmm/                    # Gaussian mixture models (varies)
│   │   │   ├── mod.rs
│   │   │   └── ... (submodules)
│   │   │
│   │   ├── tolerance/              # Tolerance/confidence bands (varies)
│   │   ├── conformal/              # Conformal prediction for regression/classification
│   │   ├── detrend/                # STL detrending, decomposition
│   │   ├── outliers.rs             # Outlier detection methods
│   │   ├── irregfdata/             # Irregular/missing functional data
│   │   ├── fdata.rs                # Functional data ops (mean, center, derivatives, median)
│   │   ├── regression.rs           # FPCA, PLS, ridge regression (optional linalg)
│   │   ├── clustering.rs           # k-means, fuzzy c-means clustering
│   │   ├── smoothing.rs            # Smoothing: kernel, bandwidth selection
│   │   ├── distance.rs             # Distances: Lp, Hausdorff, DTW, Fourier
│   │   ├── metric.rs               # Metric functions
│   │   ├── covariance.rs           # Covariance kernels, Gaussian processes
│   │   ├── simulation.rs           # Data simulation for examples
│   │   ├── wire.rs                 # WIRE (Weighted Iterative Regression Estimation)
│   │   ├── fof_regression.rs       # Function-on-function regression
│   │   ├── warping.rs              # Curve warping utilities
│   │   ├── landmark.rs             # Landmark detection & registration
│   │   ├── andrews.rs              # Andrews curves for visualization
│   │   ├── famm.rs                 # Functional ANOVA mixed models
│   │   ├── elastic.rs              # Convenience re-export for all elastic_* modules
│   │   └── smooth_basis.rs         # Smooth basis (B-spline with smoothing)
│   │
│   ├── tests/                      # Integration tests & validation (9 test files)
│   │   ├── integration_explain_pdp.rs
│   │   ├── integration_explain_shap.rs
│   │   ├── integration_explain_sensitivity.rs
│   │   ├── integration_explain_diagnostics.rs
│   │   ├── integration_explain_advanced.rs
│   │   ├── validate_against_r.rs   # Compare fdars output to R's fda package
│   │   ├── validate_spm_math.rs    # SPM correctness tests
│   │   ├── validate_phase_bands.rs # Phase variation & tolerance bands
│   │   └── validate_new_modules.rs # Smoke tests for new modules
│   │
│   ├── benches/                    # Criterion benchmarks (8 files)
│   │   ├── seasonal_benchmarks.rs
│   │   ├── depth_benchmarks.rs
│   │   ├── classification_benchmarks.rs
│   │   ├── explain_benchmarks.rs
│   │   ├── smoothing_benchmarks.rs
│   │   ├── basis_benchmarks.rs
│   │   ├── matrix_benchmarks.rs
│   │   ├── regression_benchmarks.rs
│   │   └── alignment_benchmarks.rs
│   │
│   └── examples/                   # 28 runnable examples
│       ├── 01_simulation/          # Generate synthetic FDA data
│       ├── 02_functional_operations/
│       ├── 03_smoothing/
│       ├── 04_basis_representation/
│       ├── 05_depth_measures/
│       ├── 06_distances_and_metrics/
│       ├── 07_clustering/
│       ├── 08_regression/
│       ├── 09_outlier_detection/
│       ├── 10_seasonal_analysis/
│       ├── 11_detrending/
│       ├── 12_streaming_depth/
│       ├── 13_irregular_data/
│       ├── 14_complete_pipeline/
│       ├── 15_tolerance_bands/
│       ├── 16_elastic_alignment/
│       ├── 17_equivalence_test/
│       ├── 18_landmark_registration/
│       ├── 19_tsrvf/
│       ├── 20_scalar_on_function/
│       ├── 21_function_on_scalar/
│       ├── 22_gmm_clustering/
│       ├── 23_classification/
│       ├── 24_mixed_effects/
│       ├── 25_explainability/
│       ├── 26_elastic_analysis/
│       ├── 27_spm/
│       └── 28_berkeley_growth/
│
├── documentation/                  # User & API documentation
├── scripts/                        # Utility scripts (build, release, etc)
├── validation/                     # Validation data & expected outputs
│
├── .github/                        # GitHub Actions CI/CD
├── .githooks/                      # Git hooks (pre-commit, etc)
├── .claude/                        # Claude Code configuration
├── .beads/                         # Bead analysis artifacts
├── .planning/                      # GSD planning documents
│   └── codebase/                   # Generated by gsd-map-codebase
│       ├── ARCHITECTURE.md         # (this document)
│       └── STRUCTURE.md            # (this document)
│
└── target/                         # Build artifacts (git-ignored)
```

## Directory Purposes

**`fdars-core/src/`** — All source code organized by domain:
- Core infrastructure: `error.rs`, `matrix.rs`, `parallel.rs`, `linalg.rs`
- Functional data operations: `fdata.rs`, `regression.rs`, `smoothing.rs`
- Analysis: `alignment/`, `classification/`, `seasonal/`, `depth/`, etc.
- Interpretability: `explain/`, `explain_generic/`
- Monitoring: `spm/`

**`fdars-core/tests/`** — Integration & validation tests (not unit tests in src):
- Tests import from the compiled crate
- Validate against R's fda package
- SPM mathematical correctness
- Cross-module integration

**`fdars-core/benches/`** — Criterion benchmarks:
- Per-domain performance tracking
- HTML reports in `target/criterion/`

**`fdars-core/examples/`** — 28 runnable demonstrations:
- Entry point: `cargo run --example {name}`
- Cover all major domain modules
- Reproducible use cases for documentation

## Key File Locations

**Entry Points:**
- `src/lib.rs` — Crate root, module declarations, ~350 lines of re-exports
- `src/prelude.rs` — Convenience re-export (`use fdars_core::prelude::*;`)

**Configuration:**
- `Cargo.toml` — Features: `default=["parallel"]`, `linalg` (requires Rust 1.84+), `serde`, `js`
- No `.env` or `.config.json` (settings via Cargo features)

**Core Logic — Functional Data:**
- `src/matrix.rs` — `FdMatrix` type (column-major, index arithmetic)
- `src/fdata.rs` — Mean, center, derivatives, geometric median
- `src/regression.rs` — FPCA, PLS, ridge (optional linalg)
- `src/helpers.rs` — Integration weights (Simpson's rule), L2 distance, gradient

**Core Logic — Main Domains:**
- `src/scalar_on_function/fregre_lm.rs` — Scalar-on-function linear regression
- `src/classification/lda.rs`, `qda.rs`, `knn.rs` — Classifiers
- `src/alignment/mod.rs` — Elastic curve registration
- `src/seasonal/autoperiod.rs` — Period detection
- `src/depth/mod.rs` — Depth measures
- `src/spm/phase.rs`, `monitor.rs` — Statistical process monitoring
- `src/explain/pdp.rs`, `shap.rs` — Model interpretation

**Testing:**
- `tests/` — Integration tests (not unit tests)
- In-module unit tests: `mod tests { #[test] ... }` in each file

## Naming Conventions

**Files:**
- Snake case: `fregre_lm.rs`, `elastic_align_pair.rs`, `fraiman_muniz_1d.rs`
- Domain-specific prefixes: `elastic_*.rs`, `fregre_*.rs`, `classify_*.rs`
- Verb-noun pairs: `compute_*`, `predict_*`, `estimate_*`

**Directories:**
- Singular for modules: `alignment/`, `classification/`, `seasonal/` (not alignments)
- Submodule grouping: Related functions grouped in same directory

**Functions:**
- Snake case: `fdata_to_pc_1d()`, `elastic_align_pair()`, `bootstrap_ci_fregre_lm()`
- Public: lowercase snake_case
- Private (crate-internal): prefixed `_` or module-level visibility
- Verb-noun: `predict_fregre_lm()`, `compute_fpc_scores()`, `apply_elastic_alignment()`

**Types:**
- Pascal case: `FdMatrix`, `FpcaResult`, `FregreLmResult`, `ClassifFit`
- Result types: `*Result` suffix
- Configuration: `*Config` suffix
- Trait: No suffix convention (e.g., `FpcPredictor`)

**Constants & Macros:**
- Constants: UPPER_CASE with domain prefix: `NUMERICAL_EPS`, `DEFAULT_CONVERGENCE_TOL`
- Macros: `iter_maybe_parallel!`, `slice_maybe_parallel!` (snake_case!)

## Where to Add New Code

**New Regression Method:**
- Primary: `src/scalar_on_function/{method}.rs` (e.g., `ridge.rs`)
- Test: Inline `#[cfg(test)] mod tests` in same file or `tests/integration_regression.rs`
- Implement: Function `pub fn regress_*(...) -> Result<RegResult, FdarError>`
- Result type: `pub struct RegResult { ... }` in same file
- Re-export: Add to `src/scalar_on_function/mod.rs`

**New Classification Algorithm:**
- Primary: `src/classification/{method}.rs` (e.g., `gradient_boosting.rs`)
- Implementation: Create function `pub fn fclassif_{method}(...) -> Result<ClassifFit, FdarError>`
- Implement `FpcPredictor` trait if generic explainability needed
- Test: `src/classification/tests.rs` or new `integration_classification_*.rs`
- Re-export: `src/classification/mod.rs`

**New Depth Measure:**
- Primary: `src/depth/{measure}.rs` (e.g., `quantile_depth.rs`)
- Function signature: `pub fn {measure}_1d(...) -> Vec<f64>` (depth values per observation)
- Re-export: `src/depth/mod.rs` and `src/lib.rs`

**New Explainability Method:**
- Generic: Create `src/explain_generic/{method}.rs` (e.g., `tree_shap.rs`)
- Regression-specific: Create `src/explain/{method}.rs`
- Implement trait `FpcPredictor` for automatic generic availability
- Helper: If complex math, add to `src/explain/helpers/`

**New Domain Module (e.g., Functional ANOVA):**
- Create directory: `src/{domain}/`
- Create files: `src/{domain}/mod.rs`, `src/{domain}/algorithm1.rs`, `src/{domain}/tests.rs`
- Add to `src/lib.rs`: `pub mod {domain}; pub use {domain}::{types};`
- Example: `cargo run --example 24_mixed_effects` (shows FAMM use)

**Utilities & Helpers:**
- Shared helpers: `src/helpers.rs` (integration, gradients, utility functions)
- Numerical: `src/linalg.rs` (Cholesky, OLS, Mahalanobis)
- Validation: `src/validation.rs` (dimension/parameter checks)
- Domain-specific: Create submodule `src/{domain}/helpers.rs`

## Special Directories

**`src/explain/helpers/`:**
- Purpose: Shared computation for explainability (10 files)
- Generated: No
- Committed: Yes
- Contains: ALE/LIME core, SHAP helpers, permutation, projection, saliency, stability, etc.

**`tests/`:**
- Purpose: Integration tests (not unit tests)
- Generated: No
- Committed: Yes
- Runs via: `cargo test`

**`benches/`:**
- Purpose: Criterion benchmarks with HTML reports
- Generated: Reports in `target/criterion/`
- Committed: Source code only (.gitignored: reports)
- Runs via: `cargo bench`

**`examples/`:**
- Purpose: Runnable demonstrations (28 examples)
- Generated: No
- Committed: Yes
- Runs via: `cargo run --example {name}`

**`target/`:**
- Generated: All build artifacts
- .gitignored: Yes

---

*Structure analysis: 2026-08-07*
