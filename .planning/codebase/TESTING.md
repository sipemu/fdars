# Testing Patterns

**Analysis Date:** 2026-08-07

## Test Framework

**Runner:**
- Built-in Rust `cargo test` (no custom test harness)
- Config: `Cargo.toml` specifies `[dev-dependencies]` with test support
- Test count: 1,935+ unit tests across codebase (as of analysis date)

**Assertion Library:**
- Standard Rust `assert!`, `assert_eq!`, `assert_ne!` macros
- Custom assertion helpers in `tests/` directory for numerical tolerance:
  ```rust
  fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64, label: &str)
  fn assert_vec_close_abs(actual: &[f64], expected: &[f64], tol: f64, label: &str)
  ```

**Run Commands:**
```bash
cargo test -p fdars-core                    # Run all tests (default features)
cargo test -p fdars-core --features linalg  # Run tests with ridge regression
cargo test --doc -p fdars-core              # Run doc tests only
cargo test -p fdars-core -- --test-threads=1 # Deterministic run (for CI/debugging)
```

## Test File Organization

**Location:**
- **Unit tests**: Inline in source files via `#[cfg(test)] mod tests { ... }`
- **Integration tests**: Separate `tests/*.rs` files in `fdars-core/tests/` directory
- **Doc tests**: Embedded in module/function documentation (``` rust code blocks)

**Naming:**
- Unit test files: `#[cfg(test)] mod tests` within each module
- Integration test files: `validate_against_r.rs`, `integration_explain_pdp.rs`, `validate_new_modules.rs`
- Test function naming: `test_<function>_<scenario>` pattern

**Structure:**
```
fdars-core/
├── src/
│   ├── matrix.rs           # Contains #[cfg(test)] mod tests { ... }
│   ├── depth/
│   │   ├── mod.rs          # Re-exports, public API
│   │   ├── fraiman_muniz.rs # Has tests for that submodule
│   │   └── tests.rs        # Some depth tests in separate module
│   └── ...
├── tests/                  # Integration tests
│   ├── validate_against_r.rs
│   ├── integration_explain_pdp.rs
│   ├── integration_explain_shap.rs
│   └── ...
└── benches/               # Criterion benchmarks
    ├── seasonal_benchmarks.rs
    └── ...
```

## Test Structure

**Suite Organization:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_column_major_valid() {
        let mat = sample_3x4();
        assert_eq!(mat.nrows(), 3);
        assert_eq!(mat.ncols(), 4);
        assert_eq!(mat.shape(), (3, 4));
    }

    #[test]
    fn test_from_column_major_invalid() {
        assert!(FdMatrix::from_column_major(vec![1.0, 2.0], 3, 4).is_err());
    }
}
```

**Patterns:**

1. **Setup Pattern**: Test data generators at module level
   ```rust
   fn sample_3x4() -> FdMatrix {
       let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
       FdMatrix::from_column_major(data, 3, 4).unwrap()
   }
   
   #[test]
   fn test_from_slice() {
       let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
       let mat = FdMatrix::from_slice(&data, 2, 3).unwrap();
       // ...
   }
   ```

2. **Teardown Pattern**: None explicit (Rust tests are isolated)

3. **Assertion Pattern**: Direct assertions with descriptive messages
   ```rust
   assert!(
       (a - e).abs() < tol,
       "{} [{}]: Rust={:.12}, R={:.12}, diff={:.2e} > tol={:.2e}",
       label, i, a, e, (a - e).abs(), tol
   );
   ```

## Mocking

**Framework:** No mocking framework used (no `mockall`, `mocktopus`)

**Patterns:**
- Test data generators using deterministic seeded RNG
- Example (from `integration_explain_pdp.rs`):
  ```rust
  fn generate_regression_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
      let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
      let mut data = FdMatrix::zeros(n, m);
      let mut y = vec![0.0; n];
      for i in 0..n {
          let phase = (seed.wrapping_mul(17).wrapping_add(i as u64 * 31) % 1000) as f64 / 1000.0 * PI;
          // Deterministic synthetic data
      }
      (data, y)
  }
  ```

**What to Mock:**
- External RNG: use `StdRng::seed_from_u64(seed)` for reproducibility
- Test fixtures: pre-generated JSON data in `validation/data/` and `validation/expected/`
- R comparisons: load JSON fixtures via `load_json::<T>(dir, name)` helper

**What NOT to Mock:**
- Core algorithms: tests validate actual computation, not mocked behavior
- Matrix operations: test against real data
- Integration points: validate full pipeline (FPCA → regression → projection)

## Fixtures and Factories

**Test Data:**
```rust
// Deterministic generators with seed parameter
fn generate_regression_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>)

// Reusable fixtures in test modules
fn sample_3x4() -> FdMatrix
fn uniform_grid(n: usize) -> Vec<f64>  // From src/test_helpers.rs
```

**Location:**
- Inline generator functions within test modules
- Shared helpers in `src/test_helpers.rs` (compiled only during testing)
- JSON fixtures in `validation/data/` (loaded by integration tests)

**Example Fixture Loading (integration tests):**
```rust
fn load_json<T: serde::de::DeserializeOwned>(dir: &str, name: &str) -> T {
    let path = validation_dir().join(dir).join(format!("{}.json", name));
    let data = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", path.display(), e))
}
```

## Coverage

**Requirements:** No explicit coverage threshold enforced

**View Coverage:**
```bash
# Install tarpaulin
cargo tarpaulin -p fdars-core --out Html --features linalg

# Or with llvm-cov
cargo llvm-cov -p fdars-core --features linalg
```

**Coverage notes:**
- 1,935+ tests provide broad coverage of public API
- Each test file typically covers single module/feature
- Integration tests validate cross-module interactions
- Doc tests (embedded in comments) ensure example code runs correctly

## Test Types

**Unit Tests:**
- Scope: Single function or small component
- Location: Inline `#[cfg(test)] mod tests` within source files
- Example: `matrix.rs` tests index access, dimension validation, row operations
- Count: ~1,122 unit + doc tests (per memory)

**Integration Tests:**
- Scope: Multi-component pipelines, cross-module validation
- Location: `tests/validate_*.rs`, `tests/integration_*.rs` files
- Example: `validate_against_r.rs` compares FPCA, classification, regression against R implementations
- Setup: Load JSON fixtures from `validation/` directory, compare Rust output vs. R reference
- Count: ~532 integration tests

**E2E Tests:**
- Not formal E2E test suite
- Examples in `examples/` directory serve dual purpose: documentation + validation
- Example: `examples/01_simulation/main.rs` creates simulated data, applies FDA pipeline, validates output

## Common Patterns

**Async Testing:**
- Not applicable (no async code in fdars-core)
- All algorithms are synchronous

**Error Testing:**
```rust
#[test]
fn test_from_column_major_invalid() {
    assert!(FdMatrix::from_column_major(vec![1.0, 2.0], 3, 4).is_err());
}

#[test]
fn test_empty_data() {
    let result = fdata_to_pc_1d(&FdMatrix::zeros(0, 10), 3, &vec![]);
    assert!(result.is_err());
}
```

**Numerical Tolerance Testing:**
```rust
fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", label);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() || a.is_nan() {
            continue; // Skip NaN comparisons
        }
        assert!(
            (a - e).abs() < tol,
            "{} [{}]: Rust={:.12}, R={:.12}, diff={:.2e} > tol={:.2e}",
            label, i, a, e, (a - e).abs(), tol
        );
    }
}

// Usage in tests
#[test]
fn fpca_against_r() {
    let fpca = fdata_to_pc_1d(&data, 3, &argvals).unwrap();
    assert_vec_close(&fpca.singular_values, &expected_values, 1e-6, "singular values");
}
```

**Sign-Ambiguous Results (SVD, Eigenvectors):**
```rust
fn assert_vec_close_abs(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    // Compare absolute values for results that may flip sign between implementations
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a.abs() - e.abs()).abs() < tol, "{} [{}] mismatch", label, i);
    }
}
```

**Cross-Implementation Validation (Rust vs. R):**
```rust
// From validate_against_r.rs: Known convention differences documented
// - Integration weights: Rust trapezoidal, R Simpson's 1/3 → different tolerances
// - FPCA scores: Rust returns U*Σ, R returns U → expected scaling difference
// - B-spline knots: Rust extends beyond data, R at endpoints → different basis representations

#[test]
fn fpca_scores_vs_r_reference() {
    let (fpca, expected_scores) = load_from_json::<(Vec<f64>, Vec<f64>)>("fpca", "r_reference");
    assert_vec_close(&fpca.scores, &expected_scores, 1e-5, "FPCA scores vs R");
}
```

---

*Testing analysis: 2026-08-07*
