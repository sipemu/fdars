# CRAN Submission Comments

## R CMD check results

0 errors | 1 warning | 6 notes

### WARNING

1. **checkbashisms script not available**

   ```
   A complete check needs the 'checkbashisms' script.
   ```

   This is a system tool availability issue, not a package issue.
   The configure script uses portable POSIX shell syntax.

### NOTEs

1. **New submission**
   - This is a new submission to CRAN.
   - Package size (~46MB) is larger due to vendored Rust crate sources.

2. **Hidden files in vendor directory**
   - Only `.cargo-checksum.json` files remain (required by Cargo for vendored
     builds). All other hidden files have been removed.
   - These are standard for Rust packages following CRAN's vendoring
     recommendations.

3. **Non-portable file paths**
   - A few files in `windows-sys` vendor crate have paths >100 bytes.
   - These are Windows API bindings required for Windows builds.

4. **Compiled code contains exit/abort**

   ```
   Found '_exit', 'abort', 'exit' in rust/target/release/libfdars.a
   ```

   These symbols come from the Rust standard library's panic handling
   infrastructure. They are unreachable in normal operation:
   - All Rust code uses proper error handling via `Result` types
   - Panics are caught at the R-Rust boundary by extendr
   - No user-facing code path leads to these functions

5. **Non-portable compilation flags**
   - These flags come from the system R configuration, not from the package.

6. **HTML tidy not available**
   - System tool availability issue, not a package issue.

## Changes Since Last Submission

- Updated extendr-api from 0.7 to 0.8.1
  - **Fixes non-API R calls** (BODY, CLOENV, DATAPTR, ENCLOS, FORMALS)
  - Uses new extendr-ffi crate instead of libR-sys
- Removed all non-essential hidden files from vendor directory
- Removed GNU Makefile from r-efi vendor crate
- Removed test directories with long file paths (regex-automata, zerocopy)

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
