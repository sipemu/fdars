# Technology Stack

**Analysis Date:** 2026-08-07

## Languages

**Primary:**
- Rust 2021 edition - Main implementation language for `fdars-core`
- Minimum Rust version (MSRV): 1.81.0 - Set for CRAN Windows compatibility
- Runtime version in development: 1.97.0

**Secondary:**
- R - R bindings via separate `fdars-r` package (external CRAN package)

## Runtime

**Environment:**
- Rust toolchain (stable/beta/nightly) - Cross-platform support via Cargo
- WASM target support: `wasm32-unknown-unknown` - Enables JavaScript interoperability

**Package Manager:**
- Cargo - Rust dependency and build manager
- Lockfile: `Cargo.lock` (present at `/home/simonm/projects/rust/fdars/Cargo.lock`)

## Frameworks

**Core:**
- nalgebra 0.33 - Linear algebra operations (matrix/vector computations)
- rustfft 6.2 - Fast Fourier Transform for seasonal/frequency analysis
- rayon 1.10 - Parallel iteration (optional, enabled by default via `parallel` feature)

**Scientific Computing:**
- faer 0.23 - Advanced linear algebra (Cholesky, ridge regression) — requires Rust 1.84+ (behind `linalg` feature)
- anofox-regression 0.4 - Ridge regression optimization via argmin solver
- argmin 0.11 - Gradient-free optimization framework (used by anofox-regression)
- statrs - Statistical distributions and functions
- rand 0.8, rand_distr 0.4 - Random number generation and distributions
- num-complex 0.4 - Complex number arithmetic

**Testing:**
- criterion 0.5 - Benchmarking framework with HTML report generation
- Uses built-in Rust test harness (via `#[test]` and `#[cfg(test)]`)

**Build/Dev:**
- wasm-bindgen - JavaScript/WebAssembly bindings
- serde 1.0 - Serialization framework (optional, behind `serde` feature)
- serde_json 1.0 - JSON serialization for `serde` feature

## Key Dependencies

**Critical:**
- nalgebra 0.33 - Matrix/vector operations underpin all functional data analysis
- rayon 1.10 - Enables multi-threaded parallelism for data-intensive algorithms (e.g., elastic alignment, FPCA)
- rustfft 6.2 - Powers seasonal decomposition and frequency-domain analysis

**Infrastructure:**
- faer 0.23 - Provides Cholesky factorization and ridge regression (required for `linalg` feature)
- getrandom 0.2 - Secure random number seeding; WASM-aware via `js` feature
- serde + serde_json - Optional persistence layer for pipeline workflows and `FdaData` containers

**Transitive (High Impact):**
- rayon-core - Thread pool management for `rayon`
- crossbeam - Atomic utilities and synchronization (rayon dependency)
- bytemuck - Zero-copy memory casting (faer dependency)
- simba - SIMD abstraction (nalgebra dependency)

## Configuration

**Environment:**
- Feature flags control compilation mode:
  - `parallel` (default) - Enables rayon-based parallelism
  - `linalg` - Enables faer/anofox-regression (requires Rust 1.84+, not default)
  - `serde` - Enables serialization support
  - `js` - Enables WASM JavaScript random number generation
- No `.env` file usage — all configuration is compile-time via Cargo features or function parameters
- GitHub Actions CI reads from Codecov token stored in secrets: `CODECOV_TOKEN`, `CARGO_REGISTRY_TOKEN`

**Build:**
- `Cargo.toml` workspace root at `/home/simonm/projects/rust/fdars/Cargo.toml`
- Package manifest at `/home/simonm/projects/rust/fdars/fdars-core/Cargo.toml`
- 28 runnable examples with separate `[[example]]` entries in Cargo.toml
- 8 benchmarks using Criterion framework with HTML reports
- Code coverage configuration: `codecov.yml` (70% project target, 50% patch minimum)

## Platform Requirements

**Development:**
- Rust toolchain 1.81.0 or higher
- For `linalg` feature: Rust 1.84.0 or higher
- For WASM builds: `wasm32-unknown-unknown` target installed
- For documentation: `rustfmt` and `clippy` components
- Tested on: Linux (primary CI platform)

**Production:**
- Deployment target: crates.io (published Rust library)
- Cross-platform via Cargo compilation (Linux, macOS, Windows, WASM)
- No external runtime dependencies — pure Rust library with vendored dependencies

**CI/CD Pipeline:**
- GitHub Actions workflows at `.github/workflows/`
- Multi-version testing: stable, beta, nightly Rust
- Build targets: `x86_64-unknown-linux-gnu`, `wasm32-unknown-unknown`
- Coverage reporting via codecov.io

---

*Stack analysis: 2026-08-07*
