<!-- generated-by: gsd-doc-writer -->
# Getting Started with fdars-core

`fdars-core` is a high-performance Functional Data Analysis (FDA) library for Rust. This guide takes you from zero to a working analysis in a few minutes.

## Prerequisites

- **Rust toolchain**: `>= 1.81.0` (MSRV for the default feature set)
- **Rust toolchain for `linalg` feature**: `>= 1.84.0` (required by `faer 0.23`)
- **Cargo**: included with the standard Rust toolchain installation

Install or update Rust via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

No external system libraries are required. `fdars-core` is a pure-Rust library with all dependencies vendored through Cargo.

## Installation

### Add to an existing project

```bash
cargo add fdars-core
```

Or add to `Cargo.toml` manually:

```toml
[dependencies]
fdars-core = "0.41"
```

### Feature flags

| Flag | Default | Requires | Description |
|------|---------|----------|-------------|
| `parallel` | yes | — | Rayon-based multi-core processing |
| `linalg` | no | Rust >= 1.84 | Faer Cholesky, ridge regression (`fregre_huber`, `fregre_l1`, `scalar_on_function` advanced models) |
| `serde` | no | — | `Serialize`/`Deserialize` on all public types |
| `js` | no | `wasm32` target | WASM/JavaScript random number seeding |

**Enabling `linalg`:**

```toml
[dependencies]
fdars-core = { version = "0.41", features = ["linalg"] }
```

**WASM build (disables `parallel` and `linalg`):**

```toml
[dependencies]
fdars-core = { version = "0.41", default-features = false, features = ["js"] }
```

## First Program

The example below builds an `FdMatrix`, computes the Fraiman-Muniz depth for each curve, and identifies the deepest (most central) curve. All symbols are from the verified public API.

```rust
use fdars_core::matrix::FdMatrix;
use fdars_core::depth::fraiman_muniz_1d;

fn main() {
    // 20 curves, each evaluated at 50 points on [0, 1]
    let n = 20;
    let m = 50;
    let t: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Build column-major matrix from flat data
    // Layout: element (row i, col j) is at index i + j * n
    let data: Vec<f64> = (0..(n * m))
        .map(|k| {
            let i = k % n;
            let j = k / n;
            (t[j] * std::f64::consts::PI * (i as f64 + 1.0) / n as f64).sin()
        })
        .collect();
    let mat = FdMatrix::from_column_major(data, n, m).unwrap();

    // Compute Fraiman-Muniz depth (reference set == sample set, normalise = true)
    let depths = fraiman_muniz_1d(&mat, &mat, true);

    // Find the deepest (most central) curve
    let deepest = depths
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, d)| (i, d))
        .unwrap();

    println!("Deepest curve: index {} (depth = {:.4})", deepest.0, deepest.1);
}
```

Build and run:

```bash
cargo run
```

### Using the prelude

For convenience, `fdars_core::prelude::*` re-exports the most commonly used types:

```rust
use fdars_core::prelude::*;
// Gives you: FdMatrix, FdCurveSet, FdarError, FpcaResult, PlsResult,
//            FregreLmResult, FunctionalLogisticResult, ClassifFit,
//            fraiman_muniz_1d, band_1d, AlignmentOutput, and many more.
```

## Running the Bundled Examples

The repository includes 28 runnable examples in `fdars-core/examples/`. Clone the repository to run them:

```bash
git clone https://github.com/sipemu/fdars.git
cd fdars
```

Run any example by its Cargo name (see table below):

```bash
# Examples that work with the default feature set
cargo run -p fdars-core --example depth_measures
cargo run -p fdars-core --example regression
cargo run -p fdars-core --example elastic_alignment

# Examples that require --features linalg (Rust >= 1.84 required)
cargo run -p fdars-core --example scalar_on_function --features linalg
cargo run -p fdars-core --example classification --features linalg
cargo run -p fdars-core --example elastic_analysis --features linalg
```

### Example index

| Cargo name | Source path | Topics | Needs `linalg` |
|---|---|---|---|
| `simulation` | `examples/01_simulation/main.rs` | KL expansion, GP generation | no |
| `functional_operations` | `examples/02_functional_operations/main.rs` | Mean, derivatives, norms | no |
| `smoothing` | `examples/03_smoothing/main.rs` | NW, local polynomial, k-NN | no |
| `basis_representation` | `examples/04_basis_representation/main.rs` | B-splines, Fourier, P-splines | no |
| `depth_measures` | `examples/05_depth_measures/main.rs` | 8 depth measures, outlier ranking | no |
| `distances_and_metrics` | `examples/06_distances_and_metrics/main.rs` | Lp, DTW, elastic, semimetrics | no |
| `clustering` | `examples/07_clustering/main.rs` | k-means, fuzzy c-means | no |
| `regression` | `examples/08_regression/main.rs` | FPCA, PLS | no |
| `outlier_detection` | `examples/09_outlier_detection/main.rs` | LRT bootstrap | no |
| `seasonal_analysis` | `examples/10_seasonal_analysis/main.rs` | FFT, Autoperiod, SAZED | no |
| `detrending` | `examples/11_detrending/main.rs` | Polynomial, LOESS, STL | no |
| `streaming_depth` | `examples/12_streaming_depth/main.rs` | Online depth | no |
| `irregular_data` | `examples/13_irregular_data/main.rs` | CSR storage, kernel estimation | no |
| `complete_pipeline` | `examples/14_complete_pipeline/main.rs` | End-to-end workflow | no |
| `tolerance_bands` | `examples/15_tolerance_bands/main.rs` | FPCA, conformal, Degras SCB | no |
| `elastic_alignment` | `examples/16_elastic_alignment/main.rs` | SRSF, DP alignment, Karcher mean | no |
| `equivalence_test` | `examples/17_equivalence_test/main.rs` | Functional TOST | no |
| `landmark_registration` | `examples/18_landmark_registration/main.rs` | Constrained alignment | no |
| `tsrvf` | `examples/19_tsrvf/main.rs` | Transported SRVF | no |
| `scalar_on_function` | `examples/20_scalar_on_function/main.rs` | FPC linear, logistic, kernel | yes |
| `function_on_scalar` | `examples/21_function_on_scalar/main.rs` | FOSR, FANOVA | yes |
| `gmm_clustering` | `examples/22_gmm_clustering/main.rs` | GMM-EM, BIC/ICL | yes |
| `classification` | `examples/23_classification/main.rs` | LDA, QDA, k-NN, DD | yes |
| `mixed_effects` | `examples/24_mixed_effects/main.rs` | FAMM, REML | yes |
| `explainability` | `examples/25_explainability/main.rs` | SHAP, ALE, PDP, anchors | yes |
| `elastic_analysis` | `examples/26_elastic_analysis/main.rs` | Elastic FPCA, regression, PCR | yes |
| `spm` | `examples/27_spm/main.rs` | Phase I/II, EWMA, CUSUM, rules | no |
| `berkeley_growth` | `examples/28_berkeley_growth/main.rs` | Growth-curve case study | no |

## Common Setup Issues

**Wrong Rust version for `linalg` feature**

The `linalg` feature requires `faer 0.23`, which in turn requires Rust `>= 1.84.0`. If you see a compile error about `faer` or `anofox-regression` when using `--features linalg`, update your toolchain:

```bash
rustup update stable
rustc --version  # should be >= 1.84.0
```

**Parallel feature and WASM**

Rayon (`parallel` feature) is incompatible with WASM targets. When targeting `wasm32-unknown-unknown`, always disable default features and add the `js` feature instead:

```toml
fdars-core = { version = "0.41", default-features = false, features = ["js"] }
```

**`serde` feature not enabled at build time**

The `Serialize`/`Deserialize` derives are gated behind the `serde` feature. If you need JSON serialisation of result types (e.g., `FpcaResult`, `FregreLmResult`), add:

```toml
fdars-core = { version = "0.41", features = ["serde"] }
```

**Large `target/` directory**

Full builds with examples and benchmarks can produce a large `target/` directory. To free space without losing incremental build artifacts for the library:

```bash
rm -rf target/debug/incremental target/debug/examples
```

## Next Steps

- **Architecture**: See `documentation/ARCHITECTURE.md` for a component diagram, data-flow description, and module responsibilities.
- **Configuration**: See `documentation/CONFIGURATION.md` for feature-flag details and Cargo configuration options.
- **API reference**: Full rustdoc at <https://docs.rs/fdars-core>
- **Python bindings**: [sipemu/pyfda](https://github.com/sipemu/pyfda) exposes the same algorithms via Python.
