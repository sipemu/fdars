# CRAN Submission Comments

## R CMD check results

0 errors | 1 warning | 5 notes

### WARNING

1. **checkbashisms script not available**
   - System tool availability issue, not a package issue.
   - The configure script uses portable POSIX shell syntax.

### NOTEs

1. **New submission**
   - This is a new submission to CRAN.
   - Package size (~40MB) is due to vendored Rust crate sources.

2. **Hidden files in vendor directory**
   - Only `.cargo-checksum.json` files remain. These are **required** by Cargo
     for vendored builds - without them, the offline build fails with checksum
     verification errors.
   - This follows CRAN's Rust vendoring recommendations at
     https://cran.r-project.org/web/packages/using_rust.html
   - Other accepted CRAN packages with Rust (gifski, string2path) have the
     same `.cargo-checksum.json` files.

3. **Non-portable compilation flags**
   - These flags come from the system R configuration, not from the package.

4. **Compiled code contains exit/abort**
   - These symbols come from the Rust standard library's panic handling
     infrastructure and are present in all Rust-based packages on CRAN.
   - They are unreachable in normal operation:
     - All Rust code uses proper error handling via `Result` types
     - Panics are caught at the R-Rust boundary by extendr
     - The extendr framework converts Rust panics to R errors safely
   - This is standard for Rust packages and does not affect package safety.

5. **HTML tidy not available**
   - System tool availability issue, not a package issue.

## Changes Since Last Submission

- Updated extendr-api from 0.7 to 0.8.1
  - **Fixes non-API R calls** (BODY, CLOENV, DATAPTR, ENCLOS, FORMALS)
  - Uses new extendr-ffi crate instead of libR-sys
- **Removed all test/bench directories from vendored crates**
  - Fixes non-portable file path issues (paths >100 bytes)
  - Reduces package size from 46MB to 40MB
- Removed all non-essential hidden files (.github, .vim, etc.)
- Updated vendored dependencies to use versioned directory names

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
