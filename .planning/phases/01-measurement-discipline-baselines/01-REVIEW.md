---
phase: 01-measurement-discipline-baselines
reviewed: 2026-08-07T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - fdars-core/benches/audit_hotpaths.rs
  - fdars-core/Cargo.toml
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-08-07
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found (2 Warnings, 2 Info — no Criticals)

## Summary

The benchmark harness is structurally sound. Column-major layout is correct (`data[i + j*n]`), all 7 sentinels build inputs outside `b.iter()`, and the `criterion_group!` registration matches the 7 defined functions exactly. Importantly: criterion 0.5's `Bencher::iter()` calls `black_box(routine())` internally (confirmed from source at line 88 of `criterion-0.5.1/src/bencher.rs`), so sentinels that do not manually wrap their return value in `black_box` are still correct. No linalg-gated API is called unconditionally; the bench compiles clean under `--no-default-features` and `--features linalg,parallel`.

Two warnings surface: (1) the FPCA and CV sentinels silently absorb `Err` results — if the timed function ever returns `Err`, the benchmark reports a spurious time without any signal; (2) the smooth sentinel calls `.unwrap()` inside `b.iter()` — a panic there kills the benchmark process rather than surfacing a measured time, creating an inconsistent failure mode relative to the other sentinels. Two info items cover latent divide-by-zero panics in the generator helpers at edge sizes that are never actually passed.

---

## Structural Findings (fallow)

No structural pre-pass provided.

---

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: FPCA and CV sentinels silently absorb `Err` results — a function failure produces a spurious timing

**File:** `fdars-core/benches/audit_hotpaths.rs:88` (FPCA), `fdars-core/benches/audit_hotpaths.rs:191-201` (CV)

**Issue:** `fdata_to_pc_1d` and `fclassif_cv` both return `Result<T, FdarError>`. The sentinels return the `Result` directly from the `b.iter()` closure without unwrapping it. criterion wraps the closure's return value in `black_box` and discards it — it does not inspect whether the value is `Ok` or `Err`. If either function returns `Err` (e.g., due to numerical failure, argvals length mismatch, or a regression in the library), criterion still records a "successful" iteration time that measures the error path rather than the hot path. The benchmark produces a misleadingly small time with no diagnostic output.

This is not triggered by the current inputs (n=500, m=200, ncomp=5 for FPCA; n=100, m=50, ncomp=5, nfold=5 for CV — all valid), but it is a latent measurement-correctness gap. If any future change causes these calls to return `Err`, the regression will be invisible.

The smooth sentinel (line 253) uses `.unwrap()` inside `b.iter()`, which panics visibly — an inconsistent failure mode across the harness.

**Fix:** Add `.expect(...)` (not `.unwrap()`) on both sentinels so a failure aborts the benchmark run loudly rather than silently succeeding:

```rust
// bench_fpca_sentinel
b.iter(|| {
    fdata_to_pc_1d(black_box(&data), black_box(5usize), black_box(&argvals))
        .expect("fdata_to_pc_1d failed in benchmark — inputs may be invalid")
})

// bench_cv_sentinel
b.iter(|| {
    fclassif_cv(
        black_box(&data),
        black_box(argvals.as_slice()),
        black_box(y.as_slice()),
        black_box(None),
        black_box("lda"),
        black_box(5usize),
        black_box(5usize),
        black_box(42u64),
    )
    .expect("fclassif_cv failed in benchmark — inputs may be invalid")
})
```

---

### WR-02: Smooth sentinel calls `.unwrap()` inside `b.iter()` — panic aborts the benchmark process

**File:** `fdars-core/benches/audit_hotpaths.rs:253`

**Issue:** The `nadaraya_watson` call inside `b.iter()` is unwrapped with `.unwrap()`. If `nadaraya_watson` returns `Err` (e.g., unknown kernel name, bandwidth <= 0, or length mismatch), the benchmark process panics and aborts, killing all remaining sentinels in the same process and producing no timing data. The current inputs are safe, but the failure mode is more disruptive than the silent-`Err` pattern in WR-01 and inconsistent with how the other `Result`-returning sentinels are handled.

**Fix:** Replace `.unwrap()` with `.expect("nadaraya_watson failed — check kernel name and input lengths")`. This preserves the panic-on-failure behaviour while providing a diagnostic message, and unifies the style across all `Result`-returning sentinels.

```rust
nadaraya_watson(
    black_box(&x),
    black_box(&y),
    black_box(&x_new),
    black_box(bandwidth),
    black_box("gaussian"),
)
.expect("nadaraya_watson failed in benchmark — check kernel name and input lengths")
```

---

## Info

### IN-01: `generate_curves` panics (subtraction overflow) on `m=0` and produces `inf`/`NaN` on `m=1` — latent edge-case in a shared helper

**File:** `fdars-core/benches/audit_hotpaths.rs:35`

**Issue:** The argvals computation is `(m - 1) as f64` used as the divisor. In Rust debug builds, `0usize - 1` panics with a subtraction overflow. In release builds it wraps to `usize::MAX`, and `j as f64 / usize::MAX as f64 ≈ 0` for all `j`, which silently produces degenerate (constant zero) argvals. When `m=1`, the divisor is 0 and `j=0` produces `0.0 / 0.0 = NaN`. No current caller passes `m < 2` (all use `m=50` or `m=200`), so this is not a live defect. If a future sentinel adds a small-m cell, the generator will silently return bad data.

Similarly, `generate_smoothing_data` has the same issue for `n < 2` (line 59: `(n - 1) as f64`).

**Fix:** Guard the divisor:

```rust
// generate_curves — line 35
let argvals: Vec<f64> = if m <= 1 {
    vec![0.0; m]
} else {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
};
```

---

### IN-02: `generate_curves` and `generate_smoothing_data` are not marked `#[inline]` — minor

**File:** `fdars-core/benches/audit_hotpaths.rs:34`, `58`

**Issue:** These helpers are called once per sentinel during the setup phase (outside `b.iter()`), so inlining is not a measurement concern. However, they are module-level functions in a bench binary where they can only be called from the same file; `#[inline]` or `#[allow(dead_code)]` would clarify intent to the reader. The compiler will likely inline them regardless since they are small and called once. This is a minor style note, not a correctness issue.

**Fix:** Optionally annotate with `#[inline]` or leave as-is — no functional impact.

---

_Reviewed: 2026-08-07_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
