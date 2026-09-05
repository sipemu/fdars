<!-- generated-by: gsd-doc-writer -->
# Testing

This document describes the test organization, run commands, benchmark harness, coverage configuration, and numerical-testing conventions for `fdars-core`.

## Test Organization

fdars-core uses three complementary test layers, all driven by the standard Rust test harness (`cargo test`).

### Inline unit tests

Every source module contains a `#[cfg(test)] mod tests { ... }` block immediately below the implementation. These tests have full access to private helpers and are compiled only in test builds.

```rust
// Example from src/matrix.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_column_major_valid() {
        let mat = sample_3x4();
        assert_eq!(mat.nrows(), 3);
        assert_eq!(mat.ncols(), 4);
    }
}
```

Inline tests cover construction, indexing, dimension validation, and algorithm-level known-answer checks.

### Integration tests (`fdars-core/tests/`)

The `fdars-core/tests/` directory contains separate test binaries, each linked against the public crate API. These test files are:

| File | Purpose |
|---|---|
| `alloc_audit_dpca.rs` | Allocation profiling for DPCA paths (requires `dhat-heap` feature) |
| `alloc_audit_fpca.rs` | Allocation profiling for FPCA/SVD paths (requires `dhat-heap` feature) |
| `equivalence_phase47.rs` | Golden-value regression guard for Phase 47 hot-path optimizations |
| `equivalence_phase48.rs` | Golden-value regression guard for Phase 48 optimizations |
| `equivalence_phase49.rs` | Golden-value regression guard for Phase 49 optimizations |
| `equivalence_phase50.rs` | Golden-value regression guard for Phase 50 optimizations |
| `integration_explain_advanced.rs` | Bootstrap CI, elastic attribution, cross-checks |
| `integration_explain_diagnostics.rs` | Explainability diagnostics integration |
| `integration_explain_pdp.rs` | PDP/ICE curves for linear and logistic models |
| `integration_explain_sensitivity.rs` | Sensitivity analysis integration |
| `integration_explain_shap.rs` | SHAP value integration |
| `svd_equivalence.rs` | Cross-validates nalgebra SVD paths |
| `validate_against_r.rs` | Compares Rust outputs against R reference fixtures (JSON) |
| `validate_new_modules.rs` | Cross-module validation for newer algorithms |
| `validate_phase_bands.rs` | Tolerance-band validation |
| `validate_spm_math.rs` | SPM control chart mathematical validation |

The `validate_against_r.rs` suite loads pre-generated JSON fixtures from `validation/data/` and `validation/expected/` and compares against R packages `fda`, `fda.usc`, `roahd`, `cluster`, `fpc`, `dtw`, `pls`, and `glmnet`.

**Note on allocation tests:** `alloc_audit_fpca.rs` and `alloc_audit_dpca.rs` set a `#[global_allocator]` and require the `dhat-heap` feature. They must be run in isolation — never in the same `cargo test` invocation as other tests. See [Allocation profiling](#allocation-profiling) below.

### Doctests

All public items include `///` documentation with runnable examples. Doctests are compiled and executed separately from the test harness.

## Running Tests

All commands run from the `fdars-core/` directory or use `-p fdars-core` from the workspace root.

### Full test suite (recommended)

```bash
cargo test -p fdars-core --features linalg,parallel,serde
```

This matches the CI configuration and enables the `linalg` feature (faer/ridge regression), parallel iteration, and serde serialization. It runs inline unit tests, integration tests, and doctests.

### Sequential mode (no rayon)

```bash
cargo test -p fdars-core --no-default-features --features linalg
```

Disables the `parallel` feature to catch any code path that assumes rayon is present. CI also runs this configuration.

### Doctests only

```bash
cargo test -p fdars-core --features linalg,parallel --doc
```

### Scoped / filtered tests

Run a single test by name or module filter:

```bash
# All tests whose name contains "fpca"
cargo test -p fdars-core --lib --features linalg,parallel fpca

# A specific integration test file
cargo test --test validate_against_r --features linalg

# A specific test function by name
cargo test -p fdars-core --features linalg,parallel test_from_column_major_valid
```

### Allocation profiling

The `dhat-heap` feature activates a `#[global_allocator]` in the integration tests. Run allocation tests in isolation:

```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo test -p fdars-core --features dhat-heap,linalg \
  -- count_fpca_allocations_n500_m200 --nocapture
```

Do not combine `--features dhat-heap` with a general `cargo test` invocation — parallel test binaries sharing the global allocator will panic.

### CI test commands (from `rust-ci.yml`)

| Job | Command |
|---|---|
| `test` (stable/beta/nightly) | `cargo test --features linalg,parallel,serde` |
| `test` sequential mode | `cargo test --no-default-features --features linalg` |
| `clippy` | `cargo clippy --all-targets --features linalg,parallel,serde -- -D warnings ...` |
| `fmt` | `cargo fmt --all -- --check` |
| `coverage` | `cargo llvm-cov --features linalg,parallel,serde --lcov --output-path lcov.info` |

## Benchmarks

fdars-core uses [Criterion 0.5](https://crates.io/crates/criterion) with HTML report generation. Benchmarks are in `fdars-core/benches/` and declared as `[[bench]]` entries in `Cargo.toml`.

### Registered benchmarks

| Benchmark name | File | Focus area |
|---|---|---|
| `seasonal_benchmarks` | `benches/seasonal_benchmarks.rs` | Seasonal decomposition |
| `depth_benchmarks` | `benches/depth_benchmarks.rs` | Functional depth measures |
| `classification_benchmarks` | `benches/classification_benchmarks.rs` | LDA, kNN, kernel classifiers |
| `alignment_benchmarks` | `benches/alignment_benchmarks.rs` | Elastic curve alignment |
| `regression_benchmarks` | `benches/regression_benchmarks.rs` | FPCA, `fregre_lm`, logistic |
| `explain_benchmarks` | `benches/explain_benchmarks.rs` | PDP, SHAP, ALE, LIME |
| `smoothing_benchmarks` | `benches/smoothing_benchmarks.rs` | Curve smoothing |
| `basis_benchmarks` | `benches/basis_benchmarks.rs` | Basis representations |
| `matrix_benchmarks` | `benches/matrix_benchmarks.rs` | `FdMatrix` operations |
| `audit_hotpaths` | `benches/audit_hotpaths.rs` | Hot-path audit |
| `perf_hotpaths` | `benches/perf_hotpaths.rs` | Phase 47/51 performance regression guard |
| `perf_parallelism` | `benches/perf_parallelism.rs` | Phase 48/51 thread-scaling guard |
| `inference_benchmarks` | `benches/inference_benchmarks.rs` | `inference::t_perm_test` |
| `fts_benchmarks` | `benches/fts_benchmarks.rs` | `fts::ftsm` |
| `frechet_benchmarks` | `benches/frechet_benchmarks.rs` | `frechet::frechet_global_reg` |
| `boosting_regression_benchmarks` | `benches/boosting_regression_benchmarks.rs` | `boosting_regression::boost_fosr` |
| `coclustering_benchmarks` | `benches/coclustering_benchmarks.rs` | `coclustering::co_cluster_select` |
| `fem_smoothing_benchmarks` | `benches/fem_smoothing_benchmarks.rs` | `fem_smoothing::fem_smooth_gcv` |
| `density_fda_benchmarks` | `benches/density_fda_benchmarks.rs` | `density_fda::lqd_fpca` |
| `fpca_variants_benchmarks` | `benches/fpca_variants_benchmarks.rs` | `fpca_variants::fpca_der` |
| `face_benchmarks` | `benches/face_benchmarks.rs` | `irreg_fdata::mface_covariance` |
| `shapelet` | `benches/shapelet.rs` | Shapelet classifier pipeline |
| `kshape` | `benches/kshape.rs` | `sbd_distance_matrix` + `kshape_fd` |
| `optimal_design` | `benches/optimal_design.rs` | `design_criterion` + `optimal_design` |

### Running benchmarks

Run a single benchmark group:

```bash
cargo bench -p fdars-core --bench regression_benchmarks
```

Run all benchmarks (generates HTML report in `fdars-core/target/criterion/`):

```bash
cargo bench -p fdars-core
```

Run with a name filter:

```bash
cargo bench -p fdars-core --bench perf_hotpaths -- fpca
```

Criterion generates HTML comparison reports in `fdars-core/target/criterion/`. Open `fdars-core/target/criterion/report/index.html` in a browser to view timing charts and regression summaries.

**Note:** `perf_hotpaths` and `perf_parallelism` are permanent regression guards. A significant slowdown in either bench indicates a regression introduced by recent changes and should be investigated before merging.

## Shared Test Helpers (`src/test_helpers.rs`)

The `src/test_helpers.rs` module provides utilities compiled only in test builds:

| Helper | Signature | Purpose |
|---|---|---|
| `uniform_grid` | `fn uniform_grid(n: usize) -> Vec<f64>` | Generates `n` evenly-spaced points on `[0, 1]`: `[0/(n-1), 1/(n-1), ..., 1.0]` |
| `adjusted_rand_index` | `fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64` | Hubert & Arabie ARI for clustering quality checks (1.0 = perfect, ~0 = chance) |

Import in inline tests:

```rust
#[cfg(test)]
mod tests {
    use crate::test_helpers::{adjusted_rand_index, uniform_grid};

    #[test]
    fn my_test() {
        let t = uniform_grid(50);
        assert_eq!(t.len(), 50);
        assert!((t[0] - 0.0).abs() < 1e-15);
        assert!((t[49] - 1.0).abs() < 1e-15);
    }
}
```

## Coverage

Coverage is measured via `cargo-llvm-cov` and reported to [Codecov](https://codecov.io).

**Thresholds (from `codecov.yml`):**

| Scope | Threshold |
|---|---|
| Project (overall) | 70% |
| Patch (new code in PRs) | 50% |

Coverage paths:
- `fdars-core/` — tagged with the `rust` flag
- `fdars-r/` — tagged with the `r` flag (carries forward when not updated)

The `carryforward: true` setting means coverage data from flags not present in a given CI run is carried forward from the most recent run that included them.

Coverage is generated in the `coverage` job of `rust-ci.yml` using:

```bash
cargo llvm-cov --features linalg,parallel,serde --lcov --output-path lcov.info
```

## Numerical Testing Conventions

### Known-answer tests with explicit tolerances

Tests for numerical algorithms check against expected values with documented tolerances rather than structural properties alone. Assertion helpers follow the pattern:

```rust
// Absolute tolerance (from validate_against_r.rs)
fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < tol, "...");
    }
}

// Relative tolerance (from equivalence_phase47.rs)
fn assert_rel_close(a: f64, b: f64, tol: f64) {
    let scale = a.abs().max(1e-12);
    assert!((a - b).abs() <= tol * scale, "...");
}
```

Common tolerance levels:

| Context | Tolerance |
|---|---|
| Pure counting / integer paths | Exact (`assert_eq!`) |
| Equivalent float paths (no R/Rust convention difference) | `1e-10` relative |
| R-comparison (integration weight difference: Simpson vs trapezoidal) | `1e-4` to `1e-6` absolute |
| Range checks (depths must be in `[0, 1]`) | `1e-12` guard |

### Full-rank fixture requirement

Test data generators must produce predictor matrices that span a high-dimensional function space. Use `n >> m` curves with pseudo-random phases and amplitudes — never low-rank fixtures like phase-shifted single-frequency sinusoids, which span only a 2-dimensional subspace and cause recovery tests to silently fail or pass with loose tolerances.

**Correct pattern:**

```rust
fn generate_regression_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
    // Multi-frequency: spans a rich function space
    for i in 0..n {
        let phase = deterministic_phase(seed, i);
        let amplitude = deterministic_amplitude(seed, i);
        for j in 0..m {
            data[(i, j)] = (2.0 * PI * t[j] + phase).sin()
                + amplitude * (4.0 * PI * t[j]).cos();
        }
    }
}
```

**Avoid:**

```rust
// Low-rank: all curves are phase shifts of a single frequency → 2D span only
data[(i, j)] = (2.0 * PI * t[j] + i as f64 * 0.1).sin();
```

### R comparison convention differences

The `validate_against_r.rs` suite documents known implementation differences that affect numerical comparison:

| Area | Rust convention | R convention |
|---|---|---|
| Integration weights | Composite trapezoidal rule (`simpsons_weights`) | Simpson's 1/3 rule (`fda.usc`) |
| Fourier basis normalization | No `√2` factor | Includes `√2` for orthonormality |
| FPCA scores | `U * Σ` (scaled) | `svd()$u` (unscaled `U`) |
| B-spline boundary knots | Extended beyond data range | At endpoints with multiplicity = order |
| Eigenvalue formula | `exp(-k)` for `k = 1..m` | `exp(-(k-1))` for `k = 1..m` |

Tolerances in R-comparison tests account for these differences and are set explicitly in each test.

### Parallelism and reproducibility

Tests that involve random number generation use per-thread seeding:

```rust
let rng = StdRng::seed_from_u64(seed + thread_index as u64);
```

This ensures deterministic output regardless of the number of threads, so tests pass in both `parallel` and `--no-default-features` builds.
