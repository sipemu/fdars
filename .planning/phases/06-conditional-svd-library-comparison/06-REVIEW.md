---
phase: 06-conditional-svd-library-comparison
reviewed: 2026-08-09T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - fdars-core/benches/audit_hotpaths.rs
  - fdars-core/tests/svd_equivalence.rs
findings:
  critical: 1
  warning: 2
  info: 1
  total: 4
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-08-09
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Two new artifacts were added in this phase: `generate_weighted_input` + `bench_p6_svd_comparison` in `audit_hotpaths.rs`, and the `svd_equivalence` integration test in `tests/svd_equivalence.rs`. Both are audit/throwaway scaffolding — no shipped library code was changed.

The weighted-input replication of the FPCA path is **correct**: column-major centering and Simpson-weight scaling match `regression.rs:167-296` exactly, and both `generate_weighted_input` copies (bench + test) produce byte-for-byte identical inputs for the same `(n, m)`. The thin-SVD shape assertions in the test are correct for the faer-0.23 API. The `Par::Seq` placement outside `b.iter()` correctly enforces sequential execution across all 7 faer cells.

One correctness issue was found: the faer sub-group wraps `thin_svd()` in `black_box` without unwrapping, silently suppressing any SVD failure and producing a misleading timing measurement if SVD ever fails. Two methodology warnings and one documentation info item round out the findings.

## Critical Issues

### CR-01: faer bench silently discards SVD failure — benchmark runs without a result

**File:** `fdars-core/benches/audit_hotpaths.rs:1146`

**Issue:** `black_box(mat_ref.thin_svd())` wraps the `Result<Svd<f64>, SvdError>` directly into `black_box` without calling `.unwrap()` or `.expect()`. Criterion's `b.iter()` closure ignores the return value. If `thin_svd()` returns `Err`, the iteration finishes in nanoseconds (no SVD work done), and criterion records those near-zero timings as valid measurements. The resulting numbers would be completely wrong — a factor of 10,000× faster than reality — and the audit conclusion drawn from them would be invalid. Since the inputs are well-conditioned deterministic sine waves this is unlikely to fire in practice, but the failure mode is silent and undetectable from the output.

The `svd_equivalence.rs` test uses `.expect(...)` correctly and would catch a broken SVD, but it runs at only `n=500, m=200`. If a future cell size triggers `SvdError`, the bench would produce phantom numbers while the test passes.

**Fix:**
```rust
// Sub-group B — require SVD to succeed so a failure aborts the bench run
group.bench_function(format!("n{n}_m{m}"), |b| {
    b.iter(|| {
        let mat_ref = faer::MatRef::<f64>::from_column_major_slice(
            black_box(weighted.as_slice()),
            n,
            m,
        );
        black_box(mat_ref.thin_svd().expect("faer thin_svd failed — bench result invalid"))
    })
});
```

## Warnings

### WR-01: Nalgebra sub-group measures clone overhead not present in fdata_to_pc_1d

**File:** `fdars-core/benches/audit_hotpaths.rs:1113-1119`

**Issue:** The nalgebra sub-group pre-builds a `DMatrix` outside `b.iter()` and then clones it inside:
```rust
let dmatrix = weighted.to_dmatrix();
b.iter(|| {
    let dm = black_box(dmatrix.clone());
    black_box(SVD::new(dm, true, true))
})
```
`dmatrix.clone()` copies `n*m` doubles on the heap (e.g., 800 KB at N=500, M=200). The real `fdata_to_pc_1d` path (regression.rs:291–298) builds a fresh `FdMatrix` in place and calls `weighted.to_dmatrix()` — also a heap copy of the same size. The doc comment says "clone inside to match real fdata_to_pc_1d", but `DMatrix::clone` and `FdMatrix::to_dmatrix` allocate the same number of bytes. The methodology claim is defensible, but `DMatrix::clone` has additional bookkeeping vs the `from_column_slice` path used by `to_dmatrix`. For the audit's purpose (comparing nalgebra-vs-faer SVD cost), both groups should measure the same non-SVD overhead or neither should. As written, the nalgebra group measures clone+SVD while the faer group measures MatRef-construction (zero-copy) + SVD, making the raw number difference partially attributable to allocation asymmetry rather than SVD speed.

The sub-group C (conversion-only) attribution measurement partially compensates for this, but the report consumer must remember to subtract sub-group C from sub-group A to get the pure nalgebra SVD time — this is a non-obvious post-processing step that is not documented in the bench itself.

**Fix:** Either add a `to_dmatrix()` call inside the faer group's `b.iter()` to equalize allocation overhead, or add a doc note in the bench function explicitly stating that the A/B difference includes `clone` (n*m doubles) on the nalgebra side and that sub-group C must be subtracted from A for a pure SVD comparison:
```rust
// Sub-group A: cost = DMatrix clone (n*m alloc) + nalgebra SVD
// Sub-group B: cost = MatRef view (zero-copy) + faer SVD
// Net SVD difference = (A - C) vs B, where C is sub-group C (conversion only)
// NOTE: C is ~nanoseconds so the approximation A - C ≈ A holds, but document this.
```

### WR-02: `set_global_parallelism(Par::Seq)` persists beyond the bench binary's lifetime into any subsequent rayon-using code

**File:** `fdars-core/benches/audit_hotpaths.rs:1126`

**Issue:** `set_global_parallelism(Par::Seq)` writes to a global atomic inside the faer crate. While `bench_p6_svd_comparison` is the last group in `criterion_group!` so no subsequent criterion groups are affected within this binary, any external tool that loads or links against the same process after the bench finishes (unlikely for criterion, but possible in integration harnesses) would inherit the `Par::Seq` setting. More practically: if the test runner ever runs the bench binary in-process or the global pool leaks across cargo-test boundaries (e.g., with `cargo nextest` shared-process modes), downstream faer operations would run sequentially without warning.

The correct pattern for audit benchmarks is to restore the previous parallelism after the group:

**Fix:**
```rust
{
    let prev_par = faer::get_global_parallelism();
    set_global_parallelism(Par::Seq);
    let mut group = c.benchmark_group("audit_p6_svd_faer_seq");
    // ... bench cells ...
    group.finish();
    set_global_parallelism(prev_par); // restore
}
```

## Info

### IN-01: Stale doc comment in bench_p6_svd_comparison describes a single-cell tracer but implements 7 cells

**File:** `fdars-core/benches/audit_hotpaths.rs:1069-1070`

**Issue:** The function doc reads: "This is a TRACER function measuring the single cell N=500, M=200. Task 2 expands to the full 7-cell grid." But the implementation already includes all 7 cells — the tracer was expanded in place without updating the doc. The phrase "Task 2 expands to the full 7-cell grid" is now a false statement and could mislead a reader into thinking further expansion is pending.

**Fix:** Update the opening lines of the doc comment:
```rust
/// Phase-6 SVD library comparison — nalgebra vs faer thin_svd at fdars' real FPCA sizes.
///
/// Full 7-cell grid: N ∈ {100, 500, 1000} × M ∈ {50, 200} plus the square (500, 500)
/// crossover probe. Three sub-groups: ...
```

---

_Reviewed: 2026-08-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
