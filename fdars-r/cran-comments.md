# CRAN Submission Comments

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Package Description

fdars provides functional data analysis tools with a high-performance Rust backend.
The package offers methods for:
- Functional depth computation (7 methods)
- Distance metrics and semimetrics (10+ methods)
- Functional regression (PC, basis, nonparametric)
- Clustering (k-means, fuzzy c-means)
- Outlier detection
- Statistical testing

## Rust Dependency

This package uses Rust for performance-critical algorithms. The Rust code is
compiled during installation using the cargo build system.

### Build Requirements
- Rust toolchain (rustc >= 1.70, cargo)
- Users can install Rust from https://rustup.rs/

### configure Script
The package includes a configure script that:
1. Checks for Rust toolchain availability
2. Provides clear error messages if Rust is missing
3. Handles cross-compilation on supported platforms

## Test Coverage

- Rust core: 84%+ test coverage
- R package: 80%+ test coverage

## Test Environments

* Local: Manjaro Linux, R 4.5.2, Rust 1.91
* GitHub Actions: Ubuntu, macOS, Windows (R release and devel)

## Downstream Dependencies

None (new package).
