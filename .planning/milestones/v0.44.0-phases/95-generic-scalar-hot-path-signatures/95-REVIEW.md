---
phase: 95-generic-scalar-hot-path-signatures
reviewed: 2026-09-11T08:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/helpers.rs
  - fdars-core/src/utility.rs
  - fdars-core/src/warping.rs
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: clean
---

# Phase 95: Code Review Report

**Reviewed:** 2026-09-11T08:00:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 95 generalizes four shared numeric kernels (`l2_distance`, `trapz`, `inner_product`, `inner_product_l2`) to `<T: Scalar>` in-place across `helpers.rs`, `utility.rs`, and `warping.rs`. The f64/generic boundary is applied correctly throughout — curve values become `T`, integration weights and grids stay `f64`, mixing uses `T::from_f64`. The fold-before-lift invariant is correctly applied in `trapz` (multiplying `0.5 * dx` in f64 before lifting). All four kernels are numerically correct at T=f64 and at T=Dual/Var.

There are no critical correctness bugs or security issues. Two warnings concern test reliability: the parity test for `inner_product` is circular (its "reference" mirrors the new body exactly rather than the pre-change iterator chain), and the same test contradicts its own doc comment by using a `1e-12` tolerance instead of `assert_eq!`. These are the only non-trivial issues; the implementation is sound.

## Warnings

### WR-01: `test_inner_product_parity` reference is circular — cannot detect a body regression

**File:** `fdars-core/src/utility.rs:342-356`
**Issue:** The parity test computes `acc_ref` using an explicit `for` loop that is structurally identical to the new generic body. The "reference" is therefore a copy of the implementation under test, not the pre-change iterator chain. If the body contained a subtle error (e.g., wrong multiplicand order), the test would still pass because both the function and its reference would produce the same wrong result.

The original body used:
```rust
curve1.iter().zip(curve2.iter()).zip(weights.iter())
    .map(|((&c1, &c2), &w)| c1 * c2 * w)
    .sum()
```

The parity test reference uses:
```rust
let mut acc_ref = 0.0_f64;
for i in 0..c1.len() {
    acc_ref += c1[i] * c2[i] * weights[i];
}
```

This loop is bit-identical to the new body, not a distinct oracle for the old behavior.

**Note on actual correctness:** Rust's `Iterator::sum::<f64>()` is defined as `fold(0.0_f64, Add::add)` — a left fold starting from `0.0`, which IS identical to the explicit loop. The values ARE bit-identical in practice. But the test cannot prove this because it uses the same logic.

**Fix:** Replace the `acc_ref` loop in the parity test with the original iterator-chain form to make it a genuine oracle:
```rust
fn test_inner_product_parity() {
    let argvals = vec![0.0_f64, 0.5, 1.0];
    let c1 = vec![1.0_f64, 2.0, 3.0];
    let c2 = vec![4.0_f64, 5.0, 6.0];
    let weights = simpsons_weights(&argvals);
    // Reference: the EXACT pre-change iterator chain
    let expected: f64 = c1.iter()
        .zip(c2.iter())
        .zip(weights.iter())
        .map(|((&a, &b), &w)| a * b * w)
        .sum();
    let got: f64 = inner_product(&c1, &c2, &argvals);
    assert_eq!(got, expected, "inner_product<f64> must be bit-identical to pre-change body");
}
```

---

### WR-02: `test_inner_product_parity` doc comment claims "Bit-identical parity" but assertion uses `1e-12` tolerance

**File:** `fdars-core/src/utility.rs:341`
**Issue:** The test function's doc comment reads `"Bit-identical parity: inner_product<f64> reproduces the inlined f64 loop."` but the assertion on line 352 uses:
```rust
assert!(
    (got - acc_ref).abs() < 1e-12,
    "inner_product<f64> parity: got {got}, expected {acc_ref}"
);
```

This is inconsistent with the analogous tests in the same phase — `test_l2_distance_parity` (helpers.rs:1216) and `test_trapz_parity` (helpers.rs:1342) both use `assert_eq!` to enforce bit-identical behavior. The loose `1e-12` tolerance silently accepts a non-bit-identical result, and will not catch a regression that introduces FP reordering on a future body edit. The RESEARCH.md and SUMMARY.md both state parity should be bit-identical for these kernels.

The SUMMARY.md for plan 02 notes "Parity is within 1e-12 (the sum order changes slightly)" — this reasoning is incorrect (Rust `Iterator::sum()` for f64 is a left fold identical to the explicit loop) and led to the intentionally loose assertion. This creates a false maintenance signal.

**Fix:** Change the assertion to `assert_eq!` to match the other parity tests in the phase:
```rust
assert_eq!(
    got, expected,
    "inner_product<f64> must be bit-identical to pre-change body"
);
```
Also update the doc comment to drop the qualifier "inlined f64 loop" and name the reference as "pre-change iterator chain" once WR-01 is applied.

---

## Info

### IN-01: `test_inner_product_l2_parity` is partially circular — reference calls the same generic `trapz`

**File:** `fdars-core/src/warping.rs:199-211`
**Issue:** The parity reference for `inner_product_l2` calls the now-generic `trapz` directly:
```rust
let prod_ref: Vec<f64> = psi1.iter().zip(psi2.iter()).map(|(&a, &b)| a * b).collect();
let expected: f64 = trapz(&prod_ref, &time);
```
Since `inner_product_l2` also calls `trapz` internally with the same `prod` vector, the test reduces to `trapz(prod) == trapz(prod)` — it can only fail if there is a non-determinism bug, not a regression in the `trapz` body itself. A more robust reference would inline the pre-change `trapz` loop body rather than relying on the function under evaluation.

In practice this is not a blocking concern — the correct parity for `trapz` itself is separately verified by `test_trapz_parity` with `assert_eq!`, and `inner_product_l2`'s body is so thin (two lines) that a structural error would also manifest in the Dual/Var flow tests. No change required before shipping; note for future hardening.

**Fix (optional):** Replace the `trapz` call in the reference with an inlined loop:
```rust
let mut expected = 0.0_f64;
for k in 1..prod_ref.len() {
    expected += 0.5 * (prod_ref[k] + prod_ref[k - 1]) * (time[k] - time[k - 1]);
}
```

---

_Reviewed: 2026-09-11T08:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
