<!-- generated-by: gsd-doc-writer -->
# Development Guide

This guide covers the local development setup, project layout, coding conventions, and CI
pipeline for `fdars-core` — the Rust functional-data-analysis library at the heart of fdars.

---

## Workspace and Project Layout

The repository is a minimal Cargo workspace with a single published member:

```
fdars/
├── Cargo.toml              # Workspace root (resolver = "2", members = ["fdars-core"])
├── Cargo.lock              # Pinned dependency versions
├── fdars-core/             # Published crate (fdars-core v0.40.0)
│   ├── Cargo.toml          # Package manifest — features, bench/example entries
│   ├── src/
│   │   ├── lib.rs          # Crate root: module declarations, clippy allows, root re-exports
│   │   ├── error.rs        # FdarError enum (all four variants)
│   │   ├── matrix.rs       # FdMatrix — column-major functional data matrix
│   │   ├── parallel.rs     # 5 macros gating rayon vs sequential iteration
│   │   ├── prelude.rs      # fdars_core::prelude::* convenience re-exports
│   │   ├── helpers.rs      # Numerical helpers (simpsons_weights, uniform_grid, ...)
│   │   ├── linalg.rs       # Basic linear algebra (Cholesky, OLS); pub(crate)
│   │   ├── fdata.rs        # Functional data operations (mean, derivative, norm)
│   │   ├── test_helpers.rs # Shared test utilities (uniform_grid); cfg(test) only
│   │   ├── alignment/      # Elastic registration and shape analysis
│   │   ├── classification/ # LDA, QDA, kNN, kernel, DD classifiers
│   │   ├── depth/          # Fraiman-Muniz, modal, band, projection depth
│   │   ├── elastic_*/      # Elastic regression, FPCA, changepoint, explain, PFI
│   │   ├── explain/        # Model interpretation: PDP, SHAP, LIME, ALE, importance
│   │   ├── explain_generic/# FpcPredictor trait + generic explainability functions
│   │   ├── regression.rs   # FpcaResult, fdata_to_pc_1d/2d, PLS, ridge (linalg)
│   │   ├── scalar_on_function/ # FregreLmResult, functional_logistic, PLS, ...
│   │   ├── seasonal/       # STL, period detection, seasonal strength
│   │   ├── spm/            # Statistical process monitoring (control charts)
│   │   └── ...             # 40+ additional domain modules
│   ├── benches/            # Criterion benchmark suites (harness = false)
│   ├── examples/           # 28 runnable examples (examples/NN_name/main.rs)
│   └── tests/              # Integration/equivalence tests
└── docs/                   # Project documentation (this file)
```

All `cargo` commands should be run from the **`fdars-core/`** directory (or pass `-p fdars-core`
from the workspace root). The CI workflow sets `working-directory: fdars-core` for every job.

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|-----------------|-------|
| Rust toolchain | 1.81.0 | MSRV — set for CRAN Windows compatibility |
| Rust (linalg feature) | 1.84.0 | Required by `faer 0.23+` |
| Rust (development) | 1.97.0 | Current development toolchain |
| rustfmt | stable | `rustup component add rustfmt` |
| clippy | stable | `rustup component add clippy` |

No external runtime dependencies — `fdars-core` is a pure Rust library.

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/sipemu/fdars.git
cd fdars

# Install components needed for CI gates
rustup component add rustfmt clippy

# Verify the build compiles with all CI features
cd fdars-core
cargo build --features linalg,parallel,serde
```

---

## Feature Flags

| Feature | Default | Requires | Description |
|---------|---------|----------|-------------|
| `parallel` | **yes** | — | Enables rayon-backed parallelism via `iter_maybe_parallel!` macros |
| `linalg` | no | Rust 1.84+ | Enables `faer 0.23` and `anofox-regression` (Cholesky, ridge). Not WASM-compatible. |
| `serde` | no | — | Adds `Serialize`/`Deserialize` to core types; enables `serde_json::Value` in `ExplainLayer.extra` |
| `js` | no | WASM target | Enables `getrandom/js` for `wasm32-unknown-unknown` builds |
| `dhat-heap` | no | dev only | Activates `#[global_allocator]` in the test harness for allocation profiling. **Never enable in regular CI** — parallel test execution triggers a `"A profiler already exists"` panic. Run manually: `cargo test --features dhat-heap` |

The R package (`fdars-r`) builds with `default-features = false` to omit rayon (CRAN cross-compilation).

---

## Local Dev Gate Commands

Run these gates before pushing. They mirror the CI jobs exactly.

### 1. Tests — parallel mode (primary)

```bash
cd fdars-core
cargo test --features linalg,parallel,serde
```

### 2. Tests — sequential mode (no-default-features)

```bash
cd fdars-core
cargo test --no-default-features --features linalg
```

### 3. Clippy — all targets, all CI features

```bash
cd fdars-core
cargo clippy --all-targets --features linalg,parallel,serde -- -D warnings \
  -A clippy::too_many_arguments \
  -A clippy::useless_vec \
  -A clippy::type_complexity \
  -A clippy::manual_memcpy \
  -A clippy::wildcard_in_or_patterns
```

> **Important:** Always pass `--all-targets` — clippy must also lint test and bench code.
> A plain `-p fdars-core -D warnings` misses test-code warnings and produces false-green results.

### 4. Format check

```bash
cd fdars-core
cargo fmt --all -- --check
```

To auto-fix formatting:

```bash
cargo fmt --all
```

### 5. Documentation (doc build, no external links)

```bash
cd fdars-core
RUSTDOCFLAGS="-Dwarnings" cargo doc --no-deps --features linalg,parallel,serde
```

### 6. WASM builds

```bash
cd fdars-core
cargo build --target wasm32-unknown-unknown --no-default-features
cargo build --target wasm32-unknown-unknown --no-default-features --features js
```

Requires: `rustup target add wasm32-unknown-unknown`

### 7. Benchmarks (optional — local performance investigation)

```bash
cd fdars-core
cargo bench --features linalg,parallel --bench depth_benchmarks
# HTML reports written to target/criterion/
```

---

## CI Matrix

CI runs on every push/PR to `main` that touches `fdars-core/**` or the workflow file.

| Job | Trigger | Rust versions | Command |
|-----|---------|---------------|---------|
| `test` | push/PR | stable, beta, nightly | `cargo test --features linalg,parallel,serde` + sequential mode |
| `clippy` | push/PR | stable | `cargo clippy --all-targets --features linalg,parallel,serde -- -D warnings [allows]` |
| `fmt` | push/PR | stable | `cargo fmt --all -- --check` |
| `docs` | push/PR | stable | `cargo doc --no-deps --features linalg,parallel,serde` (`RUSTDOCFLAGS=-Dwarnings`) |
| `wasm` | push/PR | stable | `cargo build --target wasm32-unknown-unknown` (no-default + js variants) |
| `coverage` | push/PR | stable | `cargo llvm-cov --features linalg,parallel,serde` → Codecov (70% project target, 50% patch minimum) |
| `publish` | release published | stable | `cargo publish` — runs only after test/clippy/fmt/docs/wasm all pass |

The `dhat-heap` and `js` features are excluded from all CI jobs (dhat causes parallel-profiler
panic; js is covered by the dedicated wasm job).

---

## Coding Conventions

### Naming

- **Module files:** `snake_case` — `matrix.rs`, `elastic_changepoint.rs`
- **Submodule directories:** same — `classification/`, `depth/`, `alignment/`
- **Public functions:** `snake_case` with dimensionality and domain prefixes:
  - `_1d` / `_2d` / `_nd` suffixes indicate the data dimensionality
  - `fregre_*` prefix for scalar-on-function regression (`fregre_lm`, `fregre_pls`, `fregre_huber`)
  - `fclassif_*` prefix for classification (`fclassif_lda`, `fclassif_knn`)
  - CV/bootstrap variants: `fregre_cv`, `bootstrap_ci_fregre_lm`, `fregre_basis_cv`
- **Result types:** `PascalCase` with `Result` suffix — `FpcaResult`, `FregreLmResult`, `ClassifFit`
- **Config structs:** `PascalCase` with `Config` suffix — `GmmClusterConfig`, `StlConfig`, `ElasticConfig`
- **Enums:** `PascalCase` — `CovType`, `ProjectionBasisType`, `TaskType`, `DepthMethod`
- **Matrix dimensions:** `nrows`, `ncols`, `n`, `m`; loop indices `i` (rows/obs), `j` (eval points), `k` (components)
- **Functional data variables:** `argvals` (evaluation points), `t` (time/parameter), `y` (response)

### Mandatory Derives on Public Types

All public types derive `Debug`, `Clone`, and `PartialEq`:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct MyResult {
    pub scores: Vec<f64>,
    pub residuals: Vec<f64>,
}
```

Add `#[non_exhaustive]` on public result structs and enums to preserve forward compatibility
(new fields/variants can be added without a breaking change):

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MyAlgoResult {
    pub scores: Vec<f64>,
}
```

### Conditional Serde

Opt in to serde support with a `cfg_attr` — never add `serde` as a hard dependency:

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MyResult {
    pub scores: Vec<f64>,
}
```

Config structs that users persist across sessions should also carry this attribute.

### Error Handling

Every public function returns `Result<T, FdarError>` — never `Option<T>`, never a panic on
invalid input. The four `FdarError` variants cover all failure modes:

```rust
use crate::error::FdarError;

pub fn my_algo(data: &FdMatrix, ncomp: usize) -> Result<MyResult, FdarError> {
    // Dimension check at entry point — always before any computation
    if data.nrows() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{}x{}", data.nrows(), data.ncols()),
        });
    }
    // Parameter range check
    if ncomp == 0 || ncomp > data.ncols() {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("must be in 1..={}", data.ncols()),
        });
    }
    // Numerical failure
    let result = heavy_computation(data)
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "SVD",
            detail: "did not converge".to_string(),
        })?;
    Ok(result)
}
```

Internal helpers (e.g., inside `explain/`) may return `Option<T>` — bridge to `Result` via
`.ok_or_else()` before surfacing to the public API.

### `#[must_use]` on Expensive Computations

Mark expensive pure computations so callers can't silently discard the result:

```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fraiman_muniz_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64> {
    // ...
}
```

74+ functions in the library carry this attribute. Apply it to any fitting function, depth
measure, or matrix decomposition.

### Column-Major `FdMatrix` Layout

`FdMatrix` stores functional data column-major (Fortran order) in a flat `Vec<f64>`:

- **Element `(i, j)`** (observation `i`, evaluation point `j`): index `i + j * nrows`
- **Rows = observations/curves**; **columns = evaluation points**
- Zero-copy column access: `data.column(j)` — returns a contiguous slice
- Row gather (non-contiguous): `data.row_to_buf(i, &mut buf)`, `data.row_dot(i, &v)`, `data.row_l2_sq(i)`
- nalgebra interop: `data.to_dmatrix()` / `FdMatrix::from_dmatrix(&dm)` for SVD

Never store or return a transposed matrix without documenting it explicitly. Always validate
`nrows` and `ncols` at function entry before any index arithmetic.

### Parallelism via `parallel.rs` Macros

Use the five macros from `parallel.rs` — never call `rayon` directly in domain code:

```rust
use crate::{iter_maybe_parallel, slice_maybe_parallel, maybe_par_chunks_mut};

// Parallel range iteration (falls back to sequential without `parallel` feature)
let depths: Vec<f64> = iter_maybe_parallel!((0..n))
    .map(|i| compute_depth(i))
    .collect();
```

Available macros: `iter_maybe_parallel!`, `slice_maybe_parallel!`, `slice_maybe_parallel_mut!`,
`maybe_par_chunks_mut!`, `maybe_par_chunks_mut_enumerate!`.

For algorithms using per-thread RNG, seed deterministically to guarantee reproducibility:

```rust
use rand::SeedableRng;
use rand::rngs::StdRng;

let results: Vec<_> = iter_maybe_parallel!((0..n_threads))
    .map(|k| {
        let mut rng = StdRng::seed_from_u64(seed + k as u64);
        // ...
    })
    .collect();
```

### Documentation Requirements

- **Module-level:** `//!` doc comment with a description and at least one `# Example` code block
- **Public items:** `///` doc comment required on every public type, function, and field
- **Complex algorithms:** explain the mathematical approach (e.g., "Karcher mean via gradient descent")
- **Layout invariants:** document column-major conventions and integration weight expectations
- **Deviations:** note when the implementation differs from the R or Python reference

### Function Parameter Order

```
(data: &FdMatrix, y: &[f64], [argvals: Option<&[f64]>,] [scalar_covariates: Option<&FdMatrix>,] config: MyConfig)
```

Functional data first, response second, optional evaluation points third, optional scalar
covariates fourth, configuration last. Provide configuration structs for functions with more
than ~4 domain parameters.

### Clippy Allows (Crate-Level)

Three `#![allow(...)]` directives are set in `src/lib.rs` and reflected in the CI clippy
invocation because they fire legitimately in numerical code:

- `clippy::needless_range_loop` — explicit index loops over matrix rows/columns are intentional
- `clippy::too_many_arguments` — large algorithm signatures are unavoidable; use config structs where possible
- `clippy::type_complexity` — complex iterator/closure types appear in generic numerical code

Do not add new crate-level allows without a documented justification.

---

## Adding a New Module or Algorithm

Follow these five steps to add a new domain module. Use `depth/fraiman_muniz` and
`scalar_on_function/fregre_lm` as reference implementations.

### Step 1 — Create the source file(s)

For a single-file module (under ~500 lines):

```
fdars-core/src/my_algo.rs
```

For a multi-file module (5+ files or >500 lines):

```
fdars-core/src/my_algo/
    mod.rs          # barrel: pub mod subfile; pub use subfile::*;
    fit.rs          # fitting function + result type
    predict.rs      # prediction helpers
    tests.rs        # #[cfg(test)] integration tests
```

### Step 2 — Declare the module in `src/lib.rs`

Add `pub mod my_algo;` in alphabetical order with the other domain modules:

```rust
// src/lib.rs
pub mod my_algo;
```

### Step 3 — Write the barrel re-exports in `mod.rs` (multi-file modules)

List every public item explicitly — no wildcard `pub use submod::*`:

```rust
// src/my_algo/mod.rs
pub mod fit;
pub mod predict;

pub use fit::{my_algo_fit, MyAlgoResult, MyAlgoConfig};
pub use predict::my_algo_predict;

#[cfg(test)]
mod tests;
```

### Step 4 — Re-export from `src/lib.rs` (critical types only)

The crate root re-exports the most commonly used types. Add result types and config structs
that external callers will reference directly:

```rust
// src/lib.rs (near the bottom, with similar re-exports)
pub use my_algo::{MyAlgoResult, MyAlgoConfig};
```

### Step 5 — Add to `src/prelude.rs` (if frequently used)

For types that belong in `use fdars_core::prelude::*`:

```rust
// src/prelude.rs
pub use crate::my_algo::{my_algo_fit, MyAlgoResult, MyAlgoConfig};
```

### Step 6 — Write tests

Inline unit tests go in the same file as the implementation:

```rust
// At the bottom of src/my_algo/fit.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    #[test]
    fn my_algo_basic_smoke() {
        let grid = uniform_grid(50);
        let data = FdMatrix::from_column_major(/* ... */, 10, 50).unwrap();
        let result = my_algo_fit(&data, &grid, MyAlgoConfig::default()).unwrap();
        assert_eq!(result.scores.len(), 10);
    }
}
```

Cross-module integration tests go in `fdars-core/tests/`. Use the shared helper
`crate::test_helpers::uniform_grid(n)` for evaluation grids.

---

## Known Build Hazards

### `target/` disk exhaustion

`target/` grows to 100+ GB over time and can fill the home partition, causing example
link failures with `"linking with cc failed"` that look like code bugs. Free space with:

```bash
rm -rf fdars-core/target/debug/{incremental,examples}
```

### `/tmp` exhaustion blocks pre-commit

Doctests link against small binaries written to `/tmp`. If `/tmp` is a small tmpfs (e.g.,
inside a container), full disk causes all commits to fail with `"No space left on device"`.
Free `/tmp` or mount a larger tmpfs before running doctests.

### `--no-verify` commits leave format drift

Skipping pre-commit hooks with `--no-verify` also skips `cargo fmt`. Run
`cargo fmt --all` before each commit and do a whole-crate sweep at milestone end to
prevent CI format-check failures.

### `dhat-heap` feature — never in regular CI

The `dhat-heap` feature activates `#[global_allocator]` in `tests/alloc_audit_fpca.rs`.
Running it alongside other tests in the same process causes a panic
(`"A profiler already exists"`). Run it in isolation only:

```bash
cargo test --features dhat-heap -- --test-threads=1
```

### Synthetic test data must be full-rank

Tests for coefficient recovery (e.g., `fregre_lm` beta(t) recovery) silently fail when the
predictor matrix is low-rank. A set of phase-shifted single-frequency sinusoids spans only a
2-dimensional subspace — use pseudo-random curves with `n >> m` instead.

---

## Code Coverage

Coverage is measured by `cargo-llvm-cov` and uploaded to Codecov on every push/PR.

Thresholds (from `codecov.yml`):

| Scope | Minimum |
|-------|---------|
| Project overall | 70% |
| PR patch | 50% |

Run coverage locally (requires `cargo install cargo-llvm-cov`):

```bash
cd fdars-core
cargo llvm-cov --features linalg,parallel,serde --lcov --output-path lcov.info
cargo llvm-cov report --features linalg,parallel,serde
```
