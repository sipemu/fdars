<!-- refreshed: 2026-08-07 -->
# Architecture

**Analysis Date:** 2026-08-07

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                         Public API Layer                                  │
│         (Re-exports from lib.rs & prelude.rs)                            │
├──────────────────────────────────────────────────────────────────────────┤
│   Regression  │ Classification │ Explainability │ Alignment │ Seasonal   │
│  Clustering   │  Depth Measures│  Time Series   │ SPM       │ Detrending │
└────────┬──────┴────────┬────────┴───────┬────────┴─────┬─────┴──────┬────┘
         │               │                │              │             │
         ▼               ▼                ▼              ▼             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Domain Module Layer                                 │
│   (scalar_on_function/, classification/, explain/, alignment/, etc)     │
│              Each module handles core algorithm logic                   │
└────┬────────────────┬──────────────────┬──────────────┬────────────────┘
     │                │                  │              │
     ▼                ▼                  ▼              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               Shared Infrastructure & Data Layer                         │
│   matrix.rs    │  fdata.rs  │  error.rs  │  linalg.rs  │ parallel.rs   │
│   helpers.rs   │  validation.rs  │  utility.rs  │ warping.rs            │
│            (Foundation types & utilities)                               │
└────┬──────────────────────────────────────────────────────┬─────────────┘
     │                                                      │
     ▼                                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      External Dependencies                                │
│   nalgebra (SVD, DMatrix) │ rayon (parallelism) │ faer (ridge, Cholesky) │
│   anofox-regression       │  rand/rand_distr    │   rustfft              │
└──────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| FdMatrix | Column-major matrix storage & indexing for functional data | `src/matrix.rs` |
| FpcaResult | Functional PCA results (scores, loadings, mean, weights) | `src/regression.rs` |
| FregreLmResult | Scalar-on-function linear regression results | `src/scalar_on_function/fregre_lm.rs` |
| FpcPredictor trait | Generic interface for explainability across models | `src/explain_generic/mod.rs` |
| Classification | LDA, QDA, kNN, kernel, DD classifiers | `src/classification/` |
| Alignment | Elastic curve registration & shape analysis | `src/alignment/` |
| Seasonal | Period detection, peak finding, seasonal decomposition | `src/seasonal/` |
| Depth | Fraiman-Muniz, modal, band, random projection depth | `src/depth/` |
| SPM | Statistical process monitoring (control charts, outlier detection) | `src/spm/` |
| Explainability | Model interpretation (PDP, SHAP, LIME, ALE, importance) | `src/explain/`, `src/explain_generic/` |
| Elastic Analysis | Shape-based regression, FPCA, PDO curves | `src/elastic_regression/`, `src/elastic_fpca/` |

## Pattern Overview

**Overall:** Modular monolith with layered architecture

**Key Characteristics:**
- **Column-major matrix storage**: All functional data uses `FdMatrix` (column-major Vec<f64>)
- **Error-based flow control**: All public functions return `Result<T, FdarError>` (no panics)
- **Trait-based abstraction**: Generic explainability via `FpcPredictor` trait (Send+Sync)
- **Parallel-first, sequential-compatible**: 5 macros in `parallel.rs` gate rayon usage by feature flag
- **Per-thread RNG seeding**: `StdRng::seed_from_u64(seed + k as u64)` for thread safety

## Layers

**Public API Layer:**
- Purpose: Convenient re-exports and documentation
- Location: `src/lib.rs`, `src/prelude.rs`
- Contains: Function signatures and result type re-exports
- Depends on: All domain modules below
- Used by: External crates consuming the library

**Domain Module Layer:**
- Purpose: Algorithm implementations organized by problem domain
- Location: `src/{module_name}/` directories
- Contains: Core algorithms, fitting functions, result types
- Depends on: Shared infrastructure (matrix, linalg, helpers, fdata)
- Used by: Public API, other domain modules

**Shared Infrastructure Layer:**
- Purpose: Cross-cutting utilities and data structures
- Location: `src/{matrix.rs, fdata.rs, error.rs, linalg.rs, parallel.rs, helpers.rs, etc}`
- Contains: Matrix type, functional data operations, numerical helpers, validation
- Depends on: External crates (nalgebra, rayon conditional, faer optional)
- Used by: All domain modules

## Data Flow

### Primary Regression Path

1. **Data input** (`src/scalar_on_function/fregre_lm.rs:fregre_lm()`)
   - Takes functional data matrix (n × m) and response vector y (length n)
   - Validates dimensions via `FdarError::InvalidDimension` checks

2. **FPCA projection** (`src/regression.rs:fdata_to_pc_1d()`)
   - Computes SVD via nalgebra on centered functional data
   - Returns `FpcaResult` with scores, loadings, mean, weights

3. **Linear regression** (`src/scalar_on_function/fregre_lm.rs`)
   - Regresses y on FPC scores via Cholesky decomposition (`src/linalg.rs`)
   - Applies optional scalar covariates to augmented design matrix

4. **Result assembly** (`src/scalar_on_function/fregre_lm.rs:FregreLmResult`)
   - Computes fitted values, residuals, R², AIC, BIC
   - Returns `FregreLmResult` for prediction & explainability

### Classification Flow

1. **Training** (`src/classification/mod.rs:fclassif_lda()` etc)
   - Computes FPCA on training data
   - Fits LDA/QDA/kNN in FPC score space
   - Returns `ClassifFit` implementing `FpcPredictor` trait

2. **Prediction** (`src/classification/mod.rs:predict()`)
   - Projects new data to FPC scores
   - Applies fitted classifier logic
   - Returns class label or probabilities

### Explainability Flow (Generic)

1. **Model setup** - User provides any `FpcPredictor` implementor (`FregreLmResult`, `FunctionalLogisticResult`, `ClassifFit`, etc)

2. **Feature projection** (`src/explain_generic/*)
   - Project functional data to FPC scores
   - Optional: perturb scores for importance/sensitivity analysis

3. **Interpretation** (`src/explain_generic/*`)
   - Compute PDP, SHAP, LIME, ALE, etc via generic functions
   - Delegate to internal helpers in `src/explain/`
   - Return structured result (e.g., `FunctionalPdpResult`, `LimeResult`)

**State Management:**
- Immutable after construction (no interior mutability)
- Result types clone/serialize (when serde feature enabled)
- FPCA results embedded in regression results for projection during prediction

## Key Abstractions

**FdMatrix:**
- Purpose: Safe column-major matrix for functional data
- Examples: `src/matrix.rs`, used in all modules
- Pattern: Carries dimensions, prevents manual indexing errors

**FpcaResult:**
- Purpose: Functional PCA output for use in downstream models
- Examples: `src/regression.rs`
- Pattern: Embeds mean, rotation, scores, weights for reproducible projection

**FpcPredictor Trait:**
- Purpose: Unified interface for any FPC-based model (regression, classification, logistic)
- Examples: Implemented in `src/explain_generic/mod.rs` for `FregreLmResult`, `FunctionalLogisticResult`, `ClassifFit`
- Pattern: Defines `fpca_mean()`, `fpca_rotation()`, `ncomp()`, `predict_from_scores()`, `task_type()` for generic explainability

**Configuration Builders:**
- Purpose: Flexible algorithm tuning without long parameter lists
- Examples: `GmmClusterConfig`, `StlConfig`, `ElasticConfig`, `ConformalConfig` in respective modules
- Pattern: Builder pattern with serde support (when feature enabled)

**Result Types:**
- Purpose: Structured, immutable output from fitting functions
- Examples: `FregreLmResult`, `FunctionalLogisticResult`, `ClassifFit`, `FpcaResult`
- Pattern: All derive `Debug, Clone, PartialEq`, carry input metadata for reproducibility

## Entry Points

**Module Functions:**
- Location: Domain module `mod.rs` files (e.g., `src/scalar_on_function/mod.rs`)
- Triggers: Direct user calls from external code
- Responsibilities: Validate inputs, delegate to internal implementations, assemble result

**Examples:**
- `src/scalar_on_function/fregre_lm.rs:fregre_lm()` — scalar-on-function linear model
- `src/classification/fit.rs:fclassif_lda()` — LDA classifier training
- `src/alignment/mod.rs:elastic_align_pair()` — elastic curve registration
- `src/spm/phase.rs:spm_phase1()` — SPM control chart initialization

**FPCA Pipeline:**
- Entry: `src/regression.rs:fdata_to_pc_1d()` for 1D, similar for 2D
- SVD via nalgebra, center data via `src/fdata.rs`, compute weights via `src/helpers.rs:simpsons_weights()`

## Architectural Constraints

- **Column-major layout**: All matrices stored as flat Vec<f64> in column-major order. Index arithmetic: `data[i + j*nrows]`
- **Rust 1.81+ minimum**: Lower version to support CRAN Windows. `linalg` feature requires Rust 1.84+ (faer 0.23+)
- **Single-threaded by default in tests**: Tests use sequential iterators; `parallel` feature gates rayon usage
- **No interior mutability**: Result types are immutable after construction; no RefCell/Mutex in public API
- **Deterministic seeding**: Per-thread RNG in parallel loops seeded as `StdRng::seed_from_u64(seed + k as u64)` for reproducibility
- **SVD via nalgebra**: All SVD operations route through `nalgebra::SVD` after conversion via `to_dmatrix()`
- **Integration weights required**: Functional inner products and FPCA always use Simpson's or custom weights from `helpers.rs`

## Anti-Patterns

### Dense Matrix Copy in SVD

**What happens:** Functions convert `FdMatrix` to nalgebra `DMatrix`, compute SVD, convert back. This is done repeatedly in FPCA workflows.

**Why it's wrong:** Unnecessary allocations and copies. For large m (evaluation points), this overhead is significant.

**Do this instead:** Cache the SVD result in `FpcaResult` and reuse via `.project()` method (already done correctly in `src/regression.rs:FpcaResult`).

### Unvalidated Slice Access in Loops

**What happens:** Some internal helpers iterate over slices without bounds checking, relying on prior validation.

**Why it's wrong:** Fragile to refactoring; easy to introduce panics if validation is skipped.

**Do this instead:** Validate dimensions explicitly at function entry (already done throughout public functions via `FdarError::InvalidDimension`). Internal helpers should assert or document preconditions.

### NaN Handling Inconsistency

**What happens:** Some modules sort/filter NaN via `sort_nan_safe()`, others silently propagate NaN.

**Why it's wrong:** Inconsistent behavior across modules; hard to debug unexpected NaN results.

**Do this instead:** Document NaN behavior per function (already done in docstrings). Use `NUMERICAL_EPS` and explicit NaN checks in sensitive calculations (e.g., Mahalanobis distance in QDA).

## Error Handling

**Strategy:** Result-based error propagation. No panics in public functions; use `.ok_or_else()` to convert `Option` (from internal helpers) to `Result`.

**Patterns:**
- Dimension mismatch → `FdarError::InvalidDimension { parameter, expected, actual }`
- Out-of-range parameter → `FdarError::InvalidParameter { parameter, message }`
- Numerical failure (non-positive-definite Cholesky, SVD fail-to-converge) → `FdarError::ComputationFailed { operation, detail }`
- Enum conversion → `FdarError::InvalidEnumValue { enum_name, value }`

**Example:** `src/linalg.rs:cholesky_d()` returns `Err` if diagonal ≤ 0 with suggestion to add regularization.

## Cross-Cutting Concerns

**Logging:** None built-in (consumer is responsible via `tracing` or `log` crates if desired).

**Validation:** 
- Dimension checks at function entry (matrix shape, vector length match)
- Parameter range checks (ncomp ≤ min(n, m), bandwidth > 0)
- Integration weight sum validation (should be ≈ argvals range)

**Authentication:** Not applicable (library, no network/user management).

---

*Architecture analysis: 2026-08-07*
