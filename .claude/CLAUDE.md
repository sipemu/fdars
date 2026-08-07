<!-- GSD:project-start source:PROJECT.md -->

## Project

**fdars**

fdars is a mature Rust functional-data-analysis (FDA) library (crate `fdars-core`, v0.14.0) with broad algorithm coverage — regression, classification, clustering, depth measures, elastic shape analysis, seasonal decomposition, statistical process monitoring, and model explainability — plus WASM/JS and R bindings. This milestone is an **audit**: proactively review execution performance and map functionality gaps against Python's scikit-fda, producing a report and a prioritized, GSD-ready backlog for future work.

**Core Value:** Produce an evidence-backed picture of where fdars is slow and what it is missing (relative to scikit-fda), turned into a prioritized backlog — so future milestones target the highest-leverage performance and functionality work first.

### Constraints

- **Scope**: Audit-only milestone — deliverables are a report + backlog, not code changes to `fdars-core`.
- **Baseline**: scikit-fda is the sole functionality-gap yardstick for this milestone.
- **Tech stack**: Benchmarks use the existing criterion 0.5 harness; performance reasoning must respect the column-major `FdMatrix` layout and feature-gated parallelism model.
- **Output**: Backlog items must be GSD-ready (phrased as candidate requirements/phases) so they can be promoted into future milestones.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Rust 2021 edition - Main implementation language for `fdars-core`
- Minimum Rust version (MSRV): 1.81.0 - Set for CRAN Windows compatibility
- Runtime version in development: 1.97.0
- R - R bindings via separate `fdars-r` package (external CRAN package)

## Runtime

- Rust toolchain (stable/beta/nightly) - Cross-platform support via Cargo
- WASM target support: `wasm32-unknown-unknown` - Enables JavaScript interoperability
- Cargo - Rust dependency and build manager
- Lockfile: `Cargo.lock` (present at `/home/simonm/projects/rust/fdars/Cargo.lock`)

## Frameworks

- nalgebra 0.33 - Linear algebra operations (matrix/vector computations)
- rustfft 6.2 - Fast Fourier Transform for seasonal/frequency analysis
- rayon 1.10 - Parallel iteration (optional, enabled by default via `parallel` feature)
- faer 0.23 - Advanced linear algebra (Cholesky, ridge regression) — requires Rust 1.84+ (behind `linalg` feature)
- anofox-regression 0.4 - Ridge regression optimization via argmin solver
- argmin 0.11 - Gradient-free optimization framework (used by anofox-regression)
- statrs - Statistical distributions and functions
- rand 0.8, rand_distr 0.4 - Random number generation and distributions
- num-complex 0.4 - Complex number arithmetic
- criterion 0.5 - Benchmarking framework with HTML report generation
- Uses built-in Rust test harness (via `#[test]` and `#[cfg(test)]`)
- wasm-bindgen - JavaScript/WebAssembly bindings
- serde 1.0 - Serialization framework (optional, behind `serde` feature)
- serde_json 1.0 - JSON serialization for `serde` feature

## Key Dependencies

- nalgebra 0.33 - Matrix/vector operations underpin all functional data analysis
- rayon 1.10 - Enables multi-threaded parallelism for data-intensive algorithms (e.g., elastic alignment, FPCA)
- rustfft 6.2 - Powers seasonal decomposition and frequency-domain analysis
- faer 0.23 - Provides Cholesky factorization and ridge regression (required for `linalg` feature)
- getrandom 0.2 - Secure random number seeding; WASM-aware via `js` feature
- serde + serde_json - Optional persistence layer for pipeline workflows and `FdaData` containers
- rayon-core - Thread pool management for `rayon`
- crossbeam - Atomic utilities and synchronization (rayon dependency)
- bytemuck - Zero-copy memory casting (faer dependency)
- simba - SIMD abstraction (nalgebra dependency)

## Configuration

- Feature flags control compilation mode:
- No `.env` file usage — all configuration is compile-time via Cargo features or function parameters
- GitHub Actions CI reads from Codecov token stored in secrets: `CODECOV_TOKEN`, `CARGO_REGISTRY_TOKEN`
- `Cargo.toml` workspace root at `/home/simonm/projects/rust/fdars/Cargo.toml`
- Package manifest at `/home/simonm/projects/rust/fdars/fdars-core/Cargo.toml`
- 28 runnable examples with separate `[[example]]` entries in Cargo.toml
- 8 benchmarks using Criterion framework with HTML reports
- Code coverage configuration: `codecov.yml` (70% project target, 50% patch minimum)

## Platform Requirements

- Rust toolchain 1.81.0 or higher
- For `linalg` feature: Rust 1.84.0 or higher
- For WASM builds: `wasm32-unknown-unknown` target installed
- For documentation: `rustfmt` and `clippy` components
- Tested on: Linux (primary CI platform)
- Deployment target: crates.io (published Rust library)
- Cross-platform via Cargo compilation (Linux, macOS, Windows, WASM)
- No external runtime dependencies — pure Rust library with vendored dependencies
- GitHub Actions workflows at `.github/workflows/`
- Multi-version testing: stable, beta, nightly Rust
- Build targets: `x86_64-unknown-linux-gnu`, `wasm32-unknown-unknown`
- Coverage reporting via codecov.io

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- Module files use `snake_case`: `matrix.rs`, `scalar_on_function.rs`, `elastic_changepoint.rs`
- Submodule directories follow the same convention: `classification/`, `depth/`, `alignment/`
- Test modules: `#[cfg(test)] mod tests;` pattern (inline, not separate files)
- Public functions use `snake_case`: `fdata_to_pc_1d`, `elastic_align_pair`, `fraiman_muniz_1d`
- Function names frequently include type hints: `_1d`, `_2d`, `_nd` suffixes for dimensionality
- Regression functions prefix with domain: `fregre_lm`, `fregre_pls`, `fregre_huber`, `fregre_l1`
- Classification functions prefix with domain: `fclassif_lda`, `fclassif_knn`, `fclassif_kernel`
- Bootstrap/CV variants: `fregre_cv`, `bootstrap_ci_fregre_lm`, `fregre_basis_cv`
- Matrix dimensions: `nrows`, `ncols`, `n`, `m` (standard mathematical notation)
- Iteration indices: `i` (rows/observations), `j` (columns/evaluation points), `k` (components)
- Functional data abbreviations: `argvals` (evaluation points), `t` (time/parameter), `y` (response)
- Results: `fpca` (FPCA result), `scores`, `loadings` (functional components)
- Result types: `FpcaResult`, `FregreLmResult`, `FunctionalLogisticResult`, `ClassifResult`
- Matrix types: `FdMatrix` (column-major functional data matrix)
- Functional data: `FdCurveSet`, `IrregFdata` (irregular functional data)
- Configuration: `GmmClusterConfig`, `StlConfig`, `ConformalConfig`, `ClassifCvConfig`, `ElasticConfig`
- Enums: `CovType`, `ProjectionBasisType`, `TaskType`, `DepthMethod`

## Code Style

- No explicit formatting configuration files (rustfmt.toml or .editorconfig found)
- Rust edition 2021 with MSRV 1.81 (Cargo.toml)
- Default rustfmt behavior: 4-space indentation, line width 100
- Three explicit clippy allows at crate level in `src/lib.rs`:
- Justified for numerical algorithms with complex types and performance-critical tight loops
- No `.clippy.toml` or explicit deny rules beyond standard
- Standard derives: `#[derive(Debug, Clone, PartialEq)]` on all public types (consistency across 97+ types)
- Conditional serde: `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`
- Non-exhaustive enums: `#[non_exhaustive]` on public result structs for forward compatibility

## Import Organization

- No path aliases configured
- Prefer explicit submodule paths for clarity: `use fdars_core::alignment::karcher_mean` over barrel imports
- Prelude module `fdars_core::prelude::*` available for convenience: re-exports commonly used types
- Barrel files: `alignment/`, `classification/`, `depth/`, etc. use explicit `pub use` statements
- Root-level re-exports in `src/lib.rs` for common types: `FdMatrix`, `FdCurveSet`, `FdarError`, `AlignmentOutput`
- Avoids wildcard re-exports to maintain clarity

## Error Handling

- All public functions return `Result<T, FdarError>` (not `Option<T>`)
- `FdarError` enum variants: `InvalidDimension`, `InvalidParameter`, `ComputationFailed`, `InvalidEnumValue`
- Each variant includes descriptive context: parameter name, expected vs. actual values
- Example (from `matrix.rs`):
- Internal helpers may use `Option<T>` (e.g., explain module's `compute_ale`, `compute_lime`)
- Bridge Option to Result via `.ok_or_else()` when exposing to public API
- Dimension checks at function entry point (no silent truncation)
- Parameter range validation with contextual error messages
- Matrix operations check `nrows`/`ncols` consistency before computation

## Logging

- Codebase uses no logging (no `log`, `tracing`, `slog` dependencies)
- All computation is deterministic; no runtime diagnostics needed
- Debug assertions for internal invariants only (debug_assert!, debug_assert_eq!)
- Example (from `matrix.rs`):

## Comments

- Module-level documentation: always include module doc comment with examples
- Public item documentation: required for all public types, functions, and fields
- Complex algorithms: explain mathematical approach (e.g., "Karcher mean via gradient descent")
- Invariants: document column-major layout, integration weight conventions, etc.
- Deviations from standard: note when implementation differs from R/Python baselines
- Use Rust doc comment syntax (`///` for items, `//!` for modules)
- Include type-level documentation on result struct fields
- Example docs from `matrix.rs`:

## Function Design

- Typical public functions: 20–150 lines
- Large functions (300+ lines) reserved for complex algorithms with internal helper structs
- Examples: `elastic_align_pair_nd` (~300 lines), `gmm_em` (~400 lines)
- Factored into submodules when exceeding ~500 lines
- Consistent order: `(data, y/labels, [argvals,] [scalar_covariates,] config/options)`
- Functional data first, responses second
- Optional evaluation points `argvals` (if not provided, compute uniform grid)
- Scalar covariates included in same functions (no separate overloads)
- Configuration structs for complex methods: `ElasticConfig`, `ClassifCvConfig`
- Public functions: `Result<T, FdarError>` (never panic on input validation)
- Result types: struct with all relevant outputs (scores, residuals, diagnostics)
- Convenience projections: `FpcaResult.project(&new_data)` for reuse pattern
- `#[must_use]` on expensive computations (74+ functions marked):
- `#[inline]` on hot-path methods: `row_to_buf`, `row_dot`, `row_l2_sq` (matrix access)
- Feature gates for heavy dependencies: `linalg` feature requires Rust 1.84+

## Module Design

- Each submodule (depth, classification, etc.) has explicit `pub use` for all public items
- No wildcard re-exports within modules; all functions listed explicitly
- Top-level `src/lib.rs` re-exports critical types: `FdMatrix`, `FdarError`, `FdCurveSet`
- Submodules with 5+ files use barrel-like `mod.rs` pattern
- Examples: `alignment/mod.rs`, `classification/mod.rs`, `depth/mod.rs`
- Each re-exports all submodule functions at module level for flat API
- Inline tests via `#[cfg(test)] mod tests { ... }` within each module file
- Shared helpers in `src/test_helpers.rs`: `uniform_grid(n)` function
- Integration tests in `tests/` directory for cross-module validation

## Data Layout

- All matrix data stored column-major (Fortran layout) in `FdMatrix`
- Documentation explicit: "element (row, col) is at index `row + col * nrows`"
- Access pattern: `data[(i, j)]` for observation i at evaluation point j
- Zero-copy column access: `data.column(j)` returns contiguous slice
- Row operations: `row_to_buf()`, `row_dot()`, `row_l2_sq()` without materializing
- Rows = observations/curves
- Columns = evaluation points
- 2D surfaces: flattened to (n, m1*m2) matrices with documentation of grid structure

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

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

- **Column-major matrix storage**: All functional data uses `FdMatrix` (column-major Vec<f64>)
- **Error-based flow control**: All public functions return `Result<T, FdarError>` (no panics)
- **Trait-based abstraction**: Generic explainability via `FpcPredictor` trait (Send+Sync)
- **Parallel-first, sequential-compatible**: 5 macros in `parallel.rs` gate rayon usage by feature flag
- **Per-thread RNG seeding**: `StdRng::seed_from_u64(seed + k as u64)` for thread safety

## Layers

- Purpose: Convenient re-exports and documentation
- Location: `src/lib.rs`, `src/prelude.rs`
- Contains: Function signatures and result type re-exports
- Depends on: All domain modules below
- Used by: External crates consuming the library
- Purpose: Algorithm implementations organized by problem domain
- Location: `src/{module_name}/` directories
- Contains: Core algorithms, fitting functions, result types
- Depends on: Shared infrastructure (matrix, linalg, helpers, fdata)
- Used by: Public API, other domain modules
- Purpose: Cross-cutting utilities and data structures
- Location: `src/{matrix.rs, fdata.rs, error.rs, linalg.rs, parallel.rs, helpers.rs, etc}`
- Contains: Matrix type, functional data operations, numerical helpers, validation
- Depends on: External crates (nalgebra, rayon conditional, faer optional)
- Used by: All domain modules

## Data Flow

### Primary Regression Path

### Classification Flow

### Explainability Flow (Generic)

- Immutable after construction (no interior mutability)
- Result types clone/serialize (when serde feature enabled)
- FPCA results embedded in regression results for projection during prediction

## Key Abstractions

- Purpose: Safe column-major matrix for functional data
- Examples: `src/matrix.rs`, used in all modules
- Pattern: Carries dimensions, prevents manual indexing errors
- Purpose: Functional PCA output for use in downstream models
- Examples: `src/regression.rs`
- Pattern: Embeds mean, rotation, scores, weights for reproducible projection
- Purpose: Unified interface for any FPC-based model (regression, classification, logistic)
- Examples: Implemented in `src/explain_generic/mod.rs` for `FregreLmResult`, `FunctionalLogisticResult`, `ClassifFit`
- Pattern: Defines `fpca_mean()`, `fpca_rotation()`, `ncomp()`, `predict_from_scores()`, `task_type()` for generic explainability
- Purpose: Flexible algorithm tuning without long parameter lists
- Examples: `GmmClusterConfig`, `StlConfig`, `ElasticConfig`, `ConformalConfig` in respective modules
- Pattern: Builder pattern with serde support (when feature enabled)
- Purpose: Structured, immutable output from fitting functions
- Examples: `FregreLmResult`, `FunctionalLogisticResult`, `ClassifFit`, `FpcaResult`
- Pattern: All derive `Debug, Clone, PartialEq`, carry input metadata for reproducibility

## Entry Points

- Location: Domain module `mod.rs` files (e.g., `src/scalar_on_function/mod.rs`)
- Triggers: Direct user calls from external code
- Responsibilities: Validate inputs, delegate to internal implementations, assemble result
- `src/scalar_on_function/fregre_lm.rs:fregre_lm()` — scalar-on-function linear model
- `src/classification/fit.rs:fclassif_lda()` — LDA classifier training
- `src/alignment/mod.rs:elastic_align_pair()` — elastic curve registration
- `src/spm/phase.rs:spm_phase1()` — SPM control chart initialization
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

### Unvalidated Slice Access in Loops

### NaN Handling Inconsistency

## Error Handling

- Dimension mismatch → `FdarError::InvalidDimension { parameter, expected, actual }`
- Out-of-range parameter → `FdarError::InvalidParameter { parameter, message }`
- Numerical failure (non-positive-definite Cholesky, SVD fail-to-converge) → `FdarError::ComputationFailed { operation, detail }`
- Enum conversion → `FdarError::InvalidEnumValue { enum_name, value }`

## Cross-Cutting Concerns

- Dimension checks at function entry (matrix shape, vector length match)
- Parameter range checks (ncomp ≤ min(n, m), bandwidth > 0)
- Integration weight sum validation (should be ≈ argvals range)

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
