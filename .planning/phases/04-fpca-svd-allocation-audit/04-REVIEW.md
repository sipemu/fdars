---
phase: 04-fpca-svd-allocation-audit
reviewed: 2026-08-08T08:32:24Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/Cargo.toml
  - fdars-core/tests/alloc_audit_fpca.rs
  - fdars-core/benches/audit_hotpaths.rs
findings:
  critical: 2
  warning: 1
  info: 3
  total: 6
status: issues_found
---

# Phase 04: Code Review Report

**Reviewed:** 2026-08-08T08:32:24Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Three files reviewed: `Cargo.toml` (dhat dev-dependency + feature gate), `tests/alloc_audit_fpca.rs` (dhat integration test harness with 3 measurement cells), and `benches/audit_hotpaths.rs` (criterion bench cells `bench_p4_fpca` and `bench_p4_elastic_fpca`).

The feature gate design is structurally sound — `dhat-heap` is not in `default` and all dhat symbols are correctly gated under `#[cfg(feature = "dhat-heap")]`. However, there are two critical defects: (1) the CI workflow's `--all-features` invocation activates `dhat-heap`, which causes the three dhat integration tests to run in parallel and panic since dhat prohibits concurrent `Profiler` instances; (2) the `count_fpca_allocations_n500_m200` test cell starts the profiler before calling `generate_test_curves`, so the reported baseline includes several hundred kilobytes of test-data setup allocations that are not part of the measured function's allocation profile. The criterion bench cells in `bench_p4_fpca` are otherwise correct: inputs are built outside `b.iter()` and criterion 0.5 automatically black-boxes the closure return value.

## Critical Issues

### CR-01: CI `--all-features` activates `dhat-heap`, causing parallel-Profiler panics in CI

**File:** `fdars-core/Cargo.toml:32` / `.github/workflows/rust-ci.yml:51,71,145`
**Issue:** The `dhat-heap` feature is an empty feature flag that the CI `cargo test --all-features` (line 51), `cargo clippy --all-targets --all-features` (line 71), and `cargo llvm-cov --all-features` (line 145) will all activate. When `dhat-heap` is active the three test functions in `alloc_audit_fpca.rs` are compiled and run by default. `cargo test` runs test functions in parallel within a single binary by default. `dhat 0.3` panics with "Error: A profiler already exists" when a second `dhat::Profiler` is constructed while one is live — the three tests each call `dhat::Profiler::builder().testing().build()` and two or more of them will collide, causing the CI run to panic/fail.

Additionally, with `dhat-heap` active, `dhat::Alloc` is installed as the global allocator for the `alloc_audit_fpca` test binary. While this does not affect other test binaries (each integration test is a separate binary), the custom allocator adds overhead to every allocation in that binary under CI.

**Fix:** Exclude `dhat-heap` from `--all-features` expansions. The cleanest approach is to mark it in Cargo.toml so it cannot be activated via `--all-features`. The conventional mechanism is to tag the feature as non-default and add an explicit exclusion comment; more robustly, rename to a path that is already excluded, or add an explicit `--exclude-features dhat-heap` to the CI commands. The recommended fix for the CI file:

```yaml
# rust-ci.yml — replace the three --all-features invocations:

# Test job (line 51):
run: cargo test --features linalg,parallel,serde,js

# Clippy job (line 71):
run: cargo clippy --all-targets --features linalg,parallel,serde -- -D warnings ...

# Coverage job (line 145):
run: cargo llvm-cov --features linalg,parallel,serde --lcov --output-path lcov.info
```

If `--all-features` must be retained, add `#[serial_test::serial]` (or equivalent) to the three dhat test functions, or add `RUST_TEST_THREADS=1` to the CI env for the test step. The simpler fix is the explicit feature list.

---

### CR-02: `generate_test_curves` called inside profiler scope in `count_fpca_allocations_n500_m200` — setup allocations contaminate the baseline

**File:** `fdars-core/tests/alloc_audit_fpca.rs:75-77`
**Issue:** The profiler is started on line 75, then `generate_test_curves(500, 200)` is called on line 76. `generate_test_curves` allocates a `Vec<f64>` of 100,000 elements (~800 KB), builds an `argvals` `Vec<f64>` of 200 elements (~1.6 KB), and calls `FdMatrix::from_column_major` which moves the data into the `FdMatrix` wrapper. These setup allocations are counted in the dhat `total_blocks` (23) and `total_bytes` (4,376,024) reported as the baseline for `fdata_to_pc_1d`. The doc comment on this test claims "Three O(n·m) allocations → ~2.4 MB total" but the actual measurement includes the data-generation allocations, inflating the count.

By contrast, the `vert_fpca` and `joint_fpca` cells correctly build `data`, `argvals`, and `karcher` outside the profiler scope (lines 96-100 and 119-123). The primary FPCA cell is inconsistent with those cells and with the stated intent ("setup vs measurement target separation" documented in 04-01-SUMMARY.md).

The contamination is approximately +2 blocks and +~802 KB in `total_bytes` (data vec + argvals vec), which also affects the `max_bytes` figure. The 04-02-SUMMARY.md records the baseline as-is ("not a regression gate"), but future comparisons using this number as the `fdata_to_pc_1d` reference will include setup cost that the elastic cells do not.

**Fix:** Move `generate_test_curves` before `Profiler::builder()`:

```rust
#[test]
#[cfg(feature = "dhat-heap")]
fn count_fpca_allocations_n500_m200() {
    // Build test data OUTSIDE the profiler — setup, not the measurement target.
    let (data, argvals) = generate_test_curves(500, 200);
    let _profiler = dhat::Profiler::builder().testing().build();
    let _ = fdata_to_pc_1d(&data, 5, &argvals);
    let stats = dhat::HeapStats::get();
    println!("Total heap blocks: {}", stats.total_blocks);
    println!("Total heap bytes: {}", stats.total_bytes);
    println!("Peak heap bytes: {}", stats.max_bytes);
}
```

After this fix, the recorded baseline artifacts in `.planning/research/bench/p4_dhat_fpca_n500_m200.txt` should be re-collected and the AUDIT-REPORT updated.

---

## Warnings

### WR-01: Redundant `use dhat;` import — `dhat::*` paths are already fully qualified

**File:** `fdars-core/tests/alloc_audit_fpca.rs:29`
**Issue:** Line 29 is `use dhat;` under `#[cfg(feature = "dhat-heap")]`. All usages in the file access the crate via fully-qualified paths: `dhat::Alloc`, `dhat::Profiler`, `dhat::HeapStats`. The bare `use dhat;` statement does not enable any shorter path form that is actually used and contributes nothing. Under Rust edition 2021, a bare `use crate_name;` only creates an alias for the crate root — since all call sites already use `dhat::` path qualifiers, this import is dead. Clippy with `--all-targets --all-features` will emit `unused_imports` for it.

**Fix:** Remove line 29:

```rust
// Remove:
#[cfg(feature = "dhat-heap")]
use dhat;
```

---

## Info

### IN-01: Module doc comment is stale — describes Wave 0 single-cell harness but file contains all 3 cells

**File:** `fdars-core/tests/alloc_audit_fpca.rs:3-5`
**Issue:** The module doc comment says "it wires `dhat` as the global allocator and provides one test cell for `fdata_to_pc_1d` at N=500, M=200. Wave 1 (Plans 02/03) will add the `vert_fpca` and `joint_fpca` cells and run them." All three cells were added by Plan 01 (per 04-01-SUMMARY.md Decision 4). The doc comment was not updated after the scope expansion. The inline comment at line 83 ("Plans 02/03 will add `vert_fpca` and `joint_fpca` cells") is also stale.

**Fix:** Update the module doc comment and the stale inline comment to reflect the final state:

```rust
//! dhat allocation-profiling integration tests for Phase 4 FPCA/SVD audit.
//!
//! Three measurement cells: `fdata_to_pc_1d` (N=500,M=200), `vert_fpca` (N=100,M=50),
//! and `joint_fpca` (N=100,M=50 with balance_c=Some(1.0) to bypass the golden-section
//! optimizer).  All cells are gated under `#[cfg(feature = "dhat-heap")]`.
```

---

### IN-02: `generate_test_curves` / `generate_curves` are duplicated across integration test and bench with no shared location

**File:** `fdars-core/tests/alloc_audit_fpca.rs:47-61` / `fdars-core/benches/audit_hotpaths.rs:39-53`
**Issue:** The 04-01-SUMMARY.md explicitly notes this as "verbatim replication of `audit_hotpaths.rs:38-52` logic." The two implementations are currently identical, which means any future change to the generation pattern (e.g., a different amplitude formula for a different audit phase) must be applied in two places. Divergence will produce incommensurable allocation baselines. This is a quality risk for the audit's reproducibility.

**Fix:** For test code that cannot share a common module between a bench and an integration test, the standard pattern is a `dev-dependency` helper crate or a shared `tests/common/mod.rs` (auto-imported by Cargo for integration tests). Since the bench also needs the function, a local helper crate under `fdars-core/benches/helpers/` could be used. However, given this is audit scaffolding (not long-lived production code), annotating the duplication with a `// NOTE: kept in sync with benches/audit_hotpaths.rs` comment is a minimum acceptable mitigation.

---

### IN-03: Timing tier comment for `n1000_m200` cell does not match observed measurement

**File:** `fdars-core/benches/audit_hotpaths.rs:752,822`
**Issue:** The `bench_p4_fpca` doc comment (line 752) and the inline cell comment (line 822) state "~64-256 ms/iter" for the `n1000_m200` cell. The actual measured mean from Plan 03 run1 is 38.307 ms and run2 is 38.311 ms — less than 60% of the documented lower bound. The comment was copied from the PATTERNS.md timing-tier estimate before the measurement was taken and was never updated after the actual run. This is a quality issue: a reader relying on the comment to predict runtime will over-provision time budgets.

**Fix:** Update both comment occurrences to reflect the actual measured range:

```rust
//   - N=1000, M=200: ~38 ms/iter  → sample_size(10)  (measured; tier: reduced samples)
```

and

```rust
// --- n1000_m200: ~38 ms/iter — reduced samples per timing-tier map ---
```

---

_Reviewed: 2026-08-08T08:32:24Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
