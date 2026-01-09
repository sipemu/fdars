# CRAN Submission Comments

## R CMD check results

0 errors | 1 warning | 4 notes

### WARNING

* checking top-level files ... WARNING
  A complete check needs the 'checkbashisms' script.

This is a local environment issue (checkbashisms not installed), not a package problem.
The configure script uses standard POSIX shell syntax.

### NOTEs

1. **New submission**
   - This is a new submission to CRAN.
   - Package size is ~30MB due to Rust source code and pre-built vignettes.

2. **Non-portable compilation flags**
   - These flags come from the system R configuration, not from the package.

3. **Compiled code contains exit/abort**
   - These come from the Rust standard library panic handling.
   - They are unreachable in normal operation as all Rust code uses
     proper error handling via Result types.

4. **Non-API R calls (BODY, CLOENV, etc.)**
   - These come from the extendr framework (v0.7) used for Rust-R bindings.
   - extendr is actively maintained and working on API compliance.

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

### Build Requirements
- Rust toolchain (rustc >= 1.81, cargo)
- Users can install Rust from https://rustup.rs/

### configure Script
The package includes a configure script that:
1. Checks for Rust toolchain availability
2. Provides clear error messages if Rust is missing
3. Handles cross-compilation on supported platforms

## Test Coverage

- All examples run without errors
- All tests pass
- 15 vignettes build successfully

## Test Environments

* Local: Manjaro Linux, R 4.5.2, Rust 1.84
* GitHub Actions: Ubuntu, macOS, Windows (R release and devel)

## Downstream Dependencies

None (new package).
