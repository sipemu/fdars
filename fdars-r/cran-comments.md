# CRAN Submission Comments

## R CMD check results

0 errors | 2 warnings | 3 notes

### WARNINGs

1. **Compiled code contains exit/abort and non-API calls**

   ```
   Found '_exit', 'abort', 'exit' in rust/target/release/libfdars.a
   Found non-API calls to R: 'BODY', 'CLOENV', 'DATAPTR', 'ENCLOS', 'FORMALS'
   ```

   **exit/abort/_exit**: These symbols come from the Rust standard library's
   panic handling infrastructure. They are unreachable in normal operation:
   - All Rust code uses proper error handling via `Result` types
   - Panics are caught at the R-Rust boundary by extendr
   - No user-facing code path leads to these functions

   **Non-API R calls**: These come from libR-sys (part of the extendr v0.7
   framework) which generates FFI bindings to R's C API. The fdars code itself
   does NOT directly call these functions - it uses safe extendr abstractions.
   The extendr team is actively working on C API compliance
   (see https://github.com/extendr/extendr).

2. **Rust compilation** (on some platforms)

   Dependencies are now vendored using `cargo vendor`. The package builds
   offline with `--offline` flag. No network access is required during
   installation.

### NOTEs

1. **New submission**
   - This is a new submission to CRAN.
   - Package size (~46MB) is larger due to vendored Rust crate sources.

2. **Hidden files in vendor directory**
   - Files like `.cargo-checksum.json` are required by Cargo for vendored
     builds. These are standard for Rust packages following CRAN's
     vendoring recommendations.

3. **Installation time**
   - Rust compilation is CPU-intensive. Installation time varies by platform.
   - Build parallelism is limited to 2 jobs (`-j 2`) per CRAN policy.

4. **Example timing ratios**
   - High CPU/elapsed ratios in examples (e.g., `outliers.depth.pond`)
     are due to Rayon-based parallelization, which is expected behavior.

5. **Non-portable compilation flags**
   - These flags come from the system R configuration, not from the package.

## Package Description

fdars provides functional data analysis tools with a high-performance Rust backend.
The package offers methods for:
- Functional depth computation (10 methods)
- Distance metrics and semimetrics (10+ methods)
- Functional regression (PC, basis, nonparametric)
- Basis representation (B-spline, Fourier, P-splines)
- Clustering (k-means, fuzzy c-means)
- Outlier detection
- Seasonal analysis
- Statistical testing

## Rust Dependency

This package uses Rust for performance-critical algorithms. The Rust code is
compiled during installation using the cargo build system.

### Vendored Dependencies
All Rust crate dependencies are bundled in the package using `cargo vendor`.
The build uses `--offline` mode - no network access is required during
installation. This follows the recommendations in "Using Rust in CRAN packages"
(https://cran.r-project.org/web/packages/using_rust.html).

### Build Requirements
- Rust toolchain (rustc >= 1.81, cargo)
- Users can install Rust from https://rustup.rs/ or system package manager

### configure Script
The package includes a configure script that:
1. Checks for Rust toolchain availability
2. Validates Rust version (>= 1.81)
3. Provides clear error messages if Rust is missing

## Test Coverage

- All examples run without errors
- All tests pass
- 15 vignettes build successfully

## Test Environments

* Local: Manjaro Linux, R 4.5.2, Rust 1.84
* GitHub Actions: Ubuntu, macOS, Windows (R release and devel)

## Downstream Dependencies

None (new package).
