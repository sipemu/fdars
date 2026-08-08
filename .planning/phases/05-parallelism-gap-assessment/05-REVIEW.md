---
phase: 05-parallelism-gap-assessment
reviewed: 2026-08-08T00:00:00Z
depth: standard
files_reviewed: 1
files_reviewed_list:
  - fdars-core/benches/audit_hotpaths.rs
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-08-08
**Depth:** standard
**Files Reviewed:** 1
**Status:** issues_found

## Summary

Reviewed the phase-05 additions to `fdars-core/benches/audit_hotpaths.rs`: four new criterion bench functions (`bench_p5_karcher_threads`, `bench_p5_streaming_threads`, `bench_p5_karcher_paybackN`, `bench_p5_streaming_paybackN`) plus their `criterion_group!` registration. This is audit-only benchmark code; the sweep design (env-driven `RAYON_NUM_THREADS`, no in-code thread setting, no recompile between thread counts) is sound and correctly delegates parallelism control to rayon's global pool.

The benches compile under `cargo build --release -p fdars-core --features linalg --benches` and run. However, the build is **not warning-clean**: the two payback-N functions trigger `non_snake_case` compiler warnings, violating the project's own naming convention. Input construction, `black_box` usage, sentinel workloads, cited source anchors (`karcher.rs:185`, `fraiman_muniz.rs:82`), and the N=1 streaming edge case were all verified correct. No correctness bugs, panics, or unwraps that could invalidate a run were found.

## Warnings

### WR-01: Two bench function names violate snake_case and emit compiler warnings

**File:** `fdars-core/benches/audit_hotpaths.rs:978` and `fdars-core/benches/audit_hotpaths.rs:1010`
**Issue:** `bench_p5_karcher_paybackN` and `bench_p5_streaming_paybackN` use a trailing capital `N`, tripping Rust's `non_snake_case` lint. The build I ran emitted exactly:
```
warning: function `bench_p5_karcher_paybackN` should have a snake case name
warning: function `bench_p5_streaming_paybackN` should have a snake case name
warning: `fdars-core` (bench "audit_hotpaths") generated 2 warnings
```
This directly contradicts the project convention in `CLAUDE.md` ("Public functions use `snake_case`") and adds noise that can mask future genuine warnings. The three phase-1..4 sibling functions and the two other phase-5 functions (`bench_p5_karcher_threads`, `bench_p5_streaming_threads`) are all correctly snake_cased, so this is an inconsistency introduced by phase 05.
**Fix:** Rename to `bench_p5_karcher_payback_n` / `bench_p5_streaming_payback_n` (and update the two `criterion_group!` entries). The runtime group names (`audit_p5_karcher_paybackN`) are string literals unaffected by the rename, so artifact/report naming stays stable.
```rust
fn bench_p5_karcher_payback_n(c: &mut Criterion) { /* ... */ }
fn bench_p5_streaming_payback_n(c: &mut Criterion) { /* ... */ }
// criterion_group!(benches, ..., bench_p5_karcher_payback_n, bench_p5_streaming_payback_n);
```

### WR-02: Karcher 5th argument mislabeled as `band_frac` (it is `lambda`)

**File:** `fdars-core/benches/audit_hotpaths.rs:925` and `fdars-core/benches/audit_hotpaths.rs:993`
**Issue:** Both new karcher cells pass the 5th positional argument with the inline comment `// band_frac = 0.0 (unbanded full DP)`. But `karcher_mean`'s signature (`karcher.rs:293`) is `(data, argvals, max_iter, tol, lambda)` — the 5th parameter is `lambda`, not `band_frac`. `band_frac` is not a parameter of `karcher_mean` at all; it is hardcoded to `0.0` *inside* `karcher_mean_impl(.., 0.0)`. The pre-existing phase-3 sibling at line 299 labels the identical argument correctly as `// D-06 lambda = 0.0`. The passed value `0.0` is coincidentally correct for `lambda`, so there is no runtime bug — but the comment is misleading and could lead a future maintainer to believe they can enable banding by editing this literal (they cannot; changing it would alter the elastic-metric penalty `lambda` instead).
**Fix:** Correct the comment to match the actual parameter and the phase-3 convention:
```rust
black_box(0.0),     // lambda = 0.0 (band_frac is hardcoded 0.0 inside karcher_mean)
```

## Info

### IN-01: Intermediate `state`/`fm` bindings not black_boxed in streaming cells

**File:** `fdars-core/benches/audit_hotpaths.rs:958-960` and `fdars-core/benches/audit_hotpaths.rs:1020-1022`
**Issue:** In the two streaming cells, `state` and `fm` are constructed inside `b.iter()` but only `&data` (input) and the final `depth_batch(...)` result are wrapped in `black_box`. The intermediate construction is not itself black_boxed. In practice DCE is prevented because the black_boxed final result depends on `fm`, which depends on `state`, so the whole chain is kept — this matches the original `bench_streaming_sentinel` pattern and is not a defect. Noting only for completeness; no change required.
**Fix:** None required. Optionally `let fm = black_box(...)` if you want to defend against future refactors that decouple the result from construction.

### IN-02: Cell name reused across groups (acceptable, noted for clarity)

**File:** `fdars-core/benches/audit_hotpaths.rs:936`, `:958`, `:1018-1020`
**Issue:** The cell name `n500_m200` appears in `bench_streaming_sentinel`, `bench_p5_streaming_threads`, and `bench_p5_streaming_paybackN`; `n100_m50` appears in multiple karcher groups. Criterion namespaces cells under their `benchmark_group` string, so these do not collide and the report/artifact naming relies on the (distinct) group names plus the `run<N>` artifact suffix. This is intentional per the doc comments and correct.
**Fix:** None required.

---

_Reviewed: 2026-08-08_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
