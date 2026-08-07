# Coding Conventions

**Analysis Date:** 2026-08-07

## Naming Patterns

**Files:**
- Module files use `snake_case`: `matrix.rs`, `scalar_on_function.rs`, `elastic_changepoint.rs`
- Submodule directories follow the same convention: `classification/`, `depth/`, `alignment/`
- Test modules: `#[cfg(test)] mod tests;` pattern (inline, not separate files)

**Functions:**
- Public functions use `snake_case`: `fdata_to_pc_1d`, `elastic_align_pair`, `fraiman_muniz_1d`
- Function names frequently include type hints: `_1d`, `_2d`, `_nd` suffixes for dimensionality
- Regression functions prefix with domain: `fregre_lm`, `fregre_pls`, `fregre_huber`, `fregre_l1`
- Classification functions prefix with domain: `fclassif_lda`, `fclassif_knn`, `fclassif_kernel`
- Bootstrap/CV variants: `fregre_cv`, `bootstrap_ci_fregre_lm`, `fregre_basis_cv`

**Variables:**
- Matrix dimensions: `nrows`, `ncols`, `n`, `m` (standard mathematical notation)
- Iteration indices: `i` (rows/observations), `j` (columns/evaluation points), `k` (components)
- Functional data abbreviations: `argvals` (evaluation points), `t` (time/parameter), `y` (response)
- Results: `fpca` (FPCA result), `scores`, `loadings` (functional components)

**Types:**
- Result types: `FpcaResult`, `FregreLmResult`, `FunctionalLogisticResult`, `ClassifResult`
- Matrix types: `FdMatrix` (column-major functional data matrix)
- Functional data: `FdCurveSet`, `IrregFdata` (irregular functional data)
- Configuration: `GmmClusterConfig`, `StlConfig`, `ConformalConfig`, `ClassifCvConfig`, `ElasticConfig`
- Enums: `CovType`, `ProjectionBasisType`, `TaskType`, `DepthMethod`

## Code Style

**Formatting:**
- No explicit formatting configuration files (rustfmt.toml or .editorconfig found)
- Rust edition 2021 with MSRV 1.81 (Cargo.toml)
- Default rustfmt behavior: 4-space indentation, line width 100

**Linting:**
- Three explicit clippy allows at crate level in `src/lib.rs`:
  ```rust
  #![allow(clippy::needless_range_loop)]
  #![allow(clippy::too_many_arguments)]
  #![allow(clippy::type_complexity)]
  ```
- Justified for numerical algorithms with complex types and performance-critical tight loops
- No `.clippy.toml` or explicit deny rules beyond standard

**Derive Macros:**
- Standard derives: `#[derive(Debug, Clone, PartialEq)]` on all public types (consistency across 97+ types)
- Conditional serde: `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`
- Non-exhaustive enums: `#[non_exhaustive]` on public result structs for forward compatibility

## Import Organization

**Order:**
1. Standard library crates (`use std::...`)
2. External crate imports (`use nalgebra::...`, `use rand::...`, etc.)
3. Internal crate imports (`use crate::matrix::FdMatrix`, `use crate::error::FdarError`)
4. Conditional imports (`#[cfg(...)] use ...`)

**Path Aliases:**
- No path aliases configured
- Prefer explicit submodule paths for clarity: `use fdars_core::alignment::karcher_mean` over barrel imports
- Prelude module `fdars_core::prelude::*` available for convenience: re-exports commonly used types

**Re-exports:**
- Barrel files: `alignment/`, `classification/`, `depth/`, etc. use explicit `pub use` statements
- Root-level re-exports in `src/lib.rs` for common types: `FdMatrix`, `FdCurveSet`, `FdarError`, `AlignmentOutput`
- Avoids wildcard re-exports to maintain clarity

## Error Handling

**Patterns:**
- All public functions return `Result<T, FdarError>` (not `Option<T>`)
- `FdarError` enum variants: `InvalidDimension`, `InvalidParameter`, `ComputationFailed`, `InvalidEnumValue`
- Each variant includes descriptive context: parameter name, expected vs. actual values
- Example (from `matrix.rs`):
  ```rust
  pub fn from_column_major(data: Vec<f64>, nrows: usize, ncols: usize) -> Result<Self, FdarError> {
      if data.len() != nrows * ncols {
          return Err(FdarError::InvalidDimension {
              parameter: "data",
              expected: format!("{}", nrows * ncols),
              actual: format!("{}", data.len()),
          });
      }
      Ok(Self { data, nrows, ncols })
  }
  ```
- Internal helpers may use `Option<T>` (e.g., explain module's `compute_ale`, `compute_lime`)
- Bridge Option to Result via `.ok_or_else()` when exposing to public API

**Validation:**
- Dimension checks at function entry point (no silent truncation)
- Parameter range validation with contextual error messages
- Matrix operations check `nrows`/`ncols` consistency before computation

## Logging

**Framework:** No structured logging framework used

**Patterns:**
- Codebase uses no logging (no `log`, `tracing`, `slog` dependencies)
- All computation is deterministic; no runtime diagnostics needed
- Debug assertions for internal invariants only (debug_assert!, debug_assert_eq!)
- Example (from `matrix.rs`):
  ```rust
  #[inline]
  pub fn row_to_buf(&self, row: usize, buf: &mut [f64]) {
      debug_assert!(row < self.nrows, "row {row} out of bounds");
      debug_assert!(buf.len() >= self.ncols, "buffer len {} < ncols {}", buf.len(), self.ncols);
      // ... implementation
  }
  ```

## Comments

**When to Comment:**
- Module-level documentation: always include module doc comment with examples
- Public item documentation: required for all public types, functions, and fields
- Complex algorithms: explain mathematical approach (e.g., "Karcher mean via gradient descent")
- Invariants: document column-major layout, integration weight conventions, etc.
- Deviations from standard: note when implementation differs from R/Python baselines

**JSDoc/TSDoc:**
- Use Rust doc comment syntax (`///` for items, `//!` for modules)
- Include type-level documentation on result struct fields
- Example docs from `matrix.rs`:
  ```rust
  /// Compute the dot product of two rows without materializing either one.
  ///
  /// The rows may come from different matrices (which must have the same `ncols`).
  ///
  /// # Panics
  /// Panics (in debug) if `row_a >= self.nrows`, `row_b >= other.nrows`,
  /// or `self.ncols != other.ncols`.
  #[inline]
  pub fn row_dot(&self, row_a: usize, other: &FdMatrix, row_b: usize) -> f64
  ```

## Function Design

**Size:**
- Typical public functions: 20–150 lines
- Large functions (300+ lines) reserved for complex algorithms with internal helper structs
- Examples: `elastic_align_pair_nd` (~300 lines), `gmm_em` (~400 lines)
- Factored into submodules when exceeding ~500 lines

**Parameters:**
- Consistent order: `(data, y/labels, [argvals,] [scalar_covariates,] config/options)`
- Functional data first, responses second
- Optional evaluation points `argvals` (if not provided, compute uniform grid)
- Scalar covariates included in same functions (no separate overloads)
- Configuration structs for complex methods: `ElasticConfig`, `ClassifCvConfig`

**Return Values:**
- Public functions: `Result<T, FdarError>` (never panic on input validation)
- Result types: struct with all relevant outputs (scores, residuals, diagnostics)
- Convenience projections: `FpcaResult.project(&new_data)` for reuse pattern

**Performance Markers:**
- `#[must_use]` on expensive computations (74+ functions marked):
  ```rust
  #[must_use = "expensive computation whose result should not be discarded"]
  pub fn karcher_mean(...) -> Result<Vec<f64>, FdarError>
  ```
- `#[inline]` on hot-path methods: `row_to_buf`, `row_dot`, `row_l2_sq` (matrix access)
- Feature gates for heavy dependencies: `linalg` feature requires Rust 1.84+

## Module Design

**Exports:**
- Each submodule (depth, classification, etc.) has explicit `pub use` for all public items
- No wildcard re-exports within modules; all functions listed explicitly
- Top-level `src/lib.rs` re-exports critical types: `FdMatrix`, `FdarError`, `FdCurveSet`

**Barrel Files:**
- Submodules with 5+ files use barrel-like `mod.rs` pattern
- Examples: `alignment/mod.rs`, `classification/mod.rs`, `depth/mod.rs`
- Each re-exports all submodule functions at module level for flat API

**Test Organization:**
- Inline tests via `#[cfg(test)] mod tests { ... }` within each module file
- Shared helpers in `src/test_helpers.rs`: `uniform_grid(n)` function
- Integration tests in `tests/` directory for cross-module validation

## Data Layout

**Column-Major Convention:**
- All matrix data stored column-major (Fortran layout) in `FdMatrix`
- Documentation explicit: "element (row, col) is at index `row + col * nrows`"
- Access pattern: `data[(i, j)]` for observation i at evaluation point j
- Zero-copy column access: `data.column(j)` returns contiguous slice
- Row operations: `row_to_buf()`, `row_dot()`, `row_l2_sq()` without materializing

**Functional Data Convention:**
- Rows = observations/curves
- Columns = evaluation points
- 2D surfaces: flattened to (n, m1*m2) matrices with documentation of grid structure

---

*Convention analysis: 2026-08-07*
