---
phase: 69-discrete-wavelet-transform-primitive
reviewed: 2026-09-04T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/wavelet/mod.rs
  - fdars-core/src/wavelet/filters.rs
  - fdars-core/src/lib.rs
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: resolved
---

> **Resolution (2026-09-04):** All 5 findings addressed in commit
> `fix(69): address code-review findings (WR-01..03, IN-01/02)`.
> - **WR-01** — RESOLVED (doc): kept the internally-consistent, perfectly-reconstructing
>   Mallat sign convention (coefficients NOT flipped; round-trip and Haar known-answer
>   preserved). Corrected the `filters.rs` module doc to state the actual Mallat (1989)
>   convention `dec_hi[k] = (-1)^k * dec_lo[L-1-k]` with a note that it differs in highpass
>   sign from PyWavelets but is internally consistent.
> - **WR-02** — RESOLVED: added the odd-length `ceil(n/2)` ambiguity note to
>   `single_level_synthesis` doc and a `debug_assert!(output_len <= m, ...)` internal guard.
>   Public/return behavior unchanged; existing `FdarError` validation retained.
> - **WR-03** — RESOLVED: removed module-wide `#![allow(dead_code)]` from `mod.rs`; scoped
>   `#[allow(dead_code)]` onto only the unused `FilterBank::rec_lo`/`rec_hi` fields.
>   `clippy --all-targets -D warnings` clean.
> - **IN-01** — RESOLVED (doc): qualified the "Never emits NaN/inf" promise with
>   "when inputs are finite" (both `single_level_synthesis` and `reconstruct`).
> - **IN-02** — RESOLVED: added explicit end-to-end round-trip tests for db3/db5/db7/db9
>   (`odd_order_daubechies_round_trip_both_modes` single-level n=37 + multi-level n=201,
>   both boundary modes, ≤1e-10 relative). All pass — no odd-order round-trip bug.

# Phase 69: Code Review Report

**Reviewed:** 2026-09-04
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 69 adds the discrete wavelet transform primitive: a `wavelet` module with
hardcoded Daubechies filter tables (db1–db10), a single-level orthogonal filter
bank engine, and a multi-level Mallat pyramid (`decompose`/`reconstruct`) plus a
batch `decompose_matrix` path. The implementation is structurally sound, boundary
modes are mathematically self-consistent, and the multi-level level-length
bookkeeping is correct. No critical correctness bugs or panic paths were found in
any code reachable from public input.

Three warnings were identified: (1) the `FilterBank::dec_hi` and `rec_hi` fields
use a sign convention opposite to PyWavelets while the module doc claims PyWavelets
compatibility — this silently misleads future downstream consumers of `rec_hi`; (2)
the Periodic-mode length validation in `single_level_synthesis` has an inherent
ambiguity that allows a wrong `output_len` to pass silently; (3) the module-wide
`#![allow(dead_code)]` is broader than necessary. Two info-level gaps remain:
false `NaN/inf` promise in synthesis doc, and missing round-trip tests for
odd-order Daubechies families.

`lib.rs` change is exactly `pub mod wavelet;` — no re-exports added, no prelude
changes, no new Cargo.toml dependencies.

---

## Warnings

### WR-01: `dec_hi` / `rec_hi` sign convention contradicts the PyWavelets claim

**File:** `fdars-core/src/wavelet/filters.rs:6-13, 52-68`

**Issue:** The `FilterBank::from_dec_lo` function derives the analysis high-pass
filter as `dec_hi[k] = (-1)^k * dec_lo[L-1-k]` (positive sign at `k=0`, also
known as Mallat's convention). PyWavelets uses `(-1)^(k+1) * dec_lo[L-1-k]`
(negative sign at `k=0`). As a result, `dec_hi` (and `rec_hi`, which is its
time-reverse) in this codebase are the **negation** of the corresponding
PyWavelets fields.

The module doc comment (lines 6–13) explicitly states this derivation is "used by
PyWavelets (`wavedec`/`waverec`)" — a false claim. The engine's internal round-trip
is still self-consistent (analysis and synthesis use the same filter, forming a
correct adjoint pair), so the DWT round-trip produces correct results. The damage
is to **downstream consumers of `rec_hi`**: code in future phases that reads
`FilterBank::rec_hi` to build a convolutional synthesis step will get wrong-sign
impulse responses relative to any PyWavelets reference, with no error, no warning,
and wrong reconstructed output.

The `dec_lo` tables are copied faithfully from PyWavelets (sum and L2 norm
invariants pass at 1e-12), so the problem is confined to the QMF derivation sign.

**Fix:** Either (a) change the sign in `from_dec_lo` to match PyWavelets:
```rust
let dec_hi: Vec<f64> = (0..l)
    .map(|k| {
        let sign = if k % 2 == 0 { -1.0 } else { 1.0 };  // (-1)^(k+1)
        sign * dec_lo[l - 1 - k]
    })
    .collect();
```
and update the `haar_known_answer_coefficients` test assertion to
`detail[0] == -(a - b) / s2`, **or** (b) keep the current convention and correct
the module doc to say "Mallat (1989) convention" rather than "PyWavelets". Add
an explicit note that `rec_hi` has opposite sign relative to PyWavelets
`FilterBank.rec_hi`.

---

### WR-02: `single_level_synthesis` Periodic-mode length validation has a silent ambiguity

**File:** `fdars-core/src/wavelet/mod.rs:241-273`

**Issue:** For Periodic boundary mode, an odd-length-`n` analysis produces
`ceil(n/2)` coefficients — the same count as analysis of `n+1` (an even signal).
Example: both `n=7` and `n=8` produce 4 coefficients. The validation in
`single_level_synthesis` checks `approx.len() == coeff_len(output_len, mode)`. If
a future internal caller accidentally passes `output_len = n+1` when `n` was odd,
the validation passes silently and the function returns a signal of the wrong length
with wrong values — no error is raised.

The current callers (`reconstruct` reading from `level_lens`, single-level tests
passing `signal.len()`) are all correct. But because the functions are
`pub(crate)`, any future phase adding a synthesis call path must be careful to pass
the exact original signal length. The validation provides no protection against an
off-by-one in `output_len` for odd signals.

**Fix:** Document the ambiguity explicitly in the function-level doc comment and
add a `debug_assert!` that guards future callers:
```rust
// After computing m and before core_synthesis:
// For Periodic mode with odd original length, m=n+1 and truncation to output_len
// removes the padding sample. Callers must pass the EXACT original n, not n+1.
debug_assert!(
    output_len <= m,
    "output_len {output_len} > extended length {m}; likely off-by-one in odd-signal path"
);
```
A stronger fix encodes parity in `WaveletCoeffs::level_lens` (already done for
the multi-level path) or stores the original `signal_len` per level explicitly,
which the current `level_lens` field already provides correctly for `reconstruct`.

---

### WR-03: `#![allow(dead_code)]` scope is module-wide, suppressing all dead-code warnings in `wavelet::` and `wavelet::filters`

**File:** `fdars-core/src/wavelet/mod.rs:41`

**Issue:** The attribute is an inner attribute (`#![...]`) placed at the top of
`mod.rs`. In Rust, this applies to the entire `wavelet` module and its children
(including `filters`). The comment correctly identifies that `rec_lo`/`rec_hi` on
`FilterBank` are the intended targets, but the blanket suppression silences
dead-code warnings for **any** function, type, or field in either file — including
future accidental dead code.

For example, `FilterBank`'s four `pub` fields and the `filter_len()` helper are
currently used, but if any were accidentally removed from the call graph in a later
phase, the compiler would not warn.

**Fix:** Replace the module-level allow with a targeted attribute on the specific
fields, or use a `#[cfg_attr(not(test), allow(dead_code))]` approach limited to
the two synthesis filter fields:
```rust
pub struct FilterBank {
    pub dec_lo: Vec<f64>,
    pub dec_hi: Vec<f64>,
    #[allow(dead_code)]
    pub rec_lo: Vec<f64>,
    #[allow(dead_code)]
    pub rec_hi: Vec<f64>,
}
```
Remove the `#![allow(dead_code)]` from `mod.rs`.

---

## Info

### IN-01: `single_level_synthesis` doc promises "Never emits NaN/inf" unconditionally

**File:** `fdars-core/src/wavelet/mod.rs:231-233`

**Issue:** The doc comment states "Never emits NaN/inf." This is only true when the
input `approx` and `detail` slices contain finite values. If the signal fed to
`single_level_analysis` contained NaN or inf (from user data or a pathological
upstream computation), the coefficients propagate NaN through the arithmetic and
the synthesis output contains NaN. The claim is false for non-finite inputs.

**Fix:** Qualify the claim:
```
// "Never emits NaN/inf when inputs are finite."
```
or add an early guard in `single_level_analysis` to check all values are finite
(expensive but correct). The doc fix is lower cost and appropriate given the
`pub(crate)` scope.

---

### IN-02: Round-trip tests skip odd-order Daubechies families (db3, db5, db7, db9)

**File:** `fdars-core/src/wavelet/mod.rs:571-580`

**Issue:** `round_trip_families()` returns only `[Haar, db2, db4, db6, db8, db10]`.
Odd-order families (db3, db5, db7, db9) are exercised only by the filter-invariant
tests in `filters.rs` (sum, L2 norm, QMF structure). The filter invariants provide
strong indirect evidence — a correct `dec_lo` combined with a correct QMF derivation
guarantees correct round-trip — but they do not replace an explicit end-to-end
reconstruction check for each family.

**Fix:** Add odd-order families to `round_trip_families()`:
```rust
fn round_trip_families() -> Vec<WaveletFamily> {
    (1..=10).map(|n| WaveletFamily::from_db_order(n).unwrap()).collect()
}
```

---

_Reviewed: 2026-09-04_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
