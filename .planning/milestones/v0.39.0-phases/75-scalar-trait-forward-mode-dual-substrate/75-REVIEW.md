---
phase: 75-scalar-trait-forward-mode-dual-substrate
reviewed: 2026-09-06T00:00:00Z
depth: deep
files_reviewed: 2
files_reviewed_list:
  - fdars-core/src/autodiff.rs
  - fdars-core/src/lib.rs
findings:
  critical: 0
  high: 1
  medium: 1
  low: 2
  total: 4
status: resolved
resolution: All 4 findings fixed in fdars-core/src/autodiff.rs (HI-01 abs/signum
  zero-tangent contract now matches docs via explicit signum(0)=0 special-case;
  ME-01 value-only PartialEq hand-written to match value-only PartialOrd; LO-02
  sqrt/ln at-zero non-finite tangent guard tests added; LO-03 powf singular/linear/
  NaN edge guard tests added). Gates green: 24 autodiff + 2845 lib tests pass,
  clippy --all-targets clean, fmt clean, no new dependency.
---

# Phase 75: Code Review Report — Forward-Mode AD Substrate

**Reviewed:** 2026-09-06
**Depth:** deep (adversarial, numeric-AD-focused)
**Files Reviewed:** 2 (`autodiff.rs`, `lib.rs`)
**Status:** findings

## Summary

The AD substrate is, at the formula level, correct. I traced the chain-rule tangent
term for every operation (add, sub, mul, div, neg, sqrt, exp, ln, sin, cos, powf,
abs, signum) and each is analytically exact. The `f64` impl is a true zero-cost
passthrough to the std inherent methods with no reordering. `PartialOrd` is
hand-written and value-only as required, `infinity()` returns `+inf` with tangent
`0`, and there are no `unwrap`/panic paths on normal input. All 18 inline tests pass.

Two real defects were found, both centered on the **exact-zero subgradient of `abs`
/ `signum`**, plus two lower-severity items. None are formula errors on the interior
of any function's domain — they are edge/contract issues that the plan's own tests
deliberately do not probe. Because this is a numeric core whose behavior propagates
into Phases 76–77 (DTW branch semantics), the doc-vs-behavior mismatch is worth
fixing before downstream code relies on the stated contract.

`lib.rs`: the sole change is `pub mod autodiff;` in correct alphabetical position
(line 76, between `andrews` and `basis`). Clean, additive, non-breaking. No issue.

## High

### HI-01: `abs`/`signum` doc contract is factually wrong at exactly 0.0 — actual derivative is 1.0, not 0.0

**File:** `fdars-core/src/autodiff.rs:95-99, 372-386` (and module doc `:96`)

**Issue:** The trait docs and inline comments repeatedly assert that the
subgradient convention yields tangent `0.0` at `value == 0.0`:

- Line 96: *"the derivative at exactly `0.0` is `0.0`."*
- Line 373: *"Subdifferential convention: d/dx |v| = signum(v), with signum(0) = 0."*
- Line 98–99: *"Sign … tangent `0`."*

This is incorrect. Rust's `f64::signum` returns **`1.0`** for `+0.0` and **`-1.0`**
for `-0.0` (never `0.0`; only `NaN` yields `NaN`). Verified empirically:

```
signum(0.0)  = 1
signum(-0.0) = -1
abs tangent at value 0.0, tangent 1.0 = 1
```

Therefore:
- `Dual::abs(Dual{value: 0.0, tangent: t})` produces tangent `t * 1.0 = t`, **not
  `0.0`** as documented (`t * -1.0` for `-0.0`).
- `Dual::signum(Dual{value: 0.0})` produces value `1.0`, **not `0.0`** as documented.

The produced tangent `±1` is still a *valid* member of the subdifferential of `|x|`
at 0 (which is `[-1, 1]`), so results are not NaN/garbage — but the **stated
contract is false**, and Phase 76's DTW/soft-min branch reasoning is explicitly
predicated on these documented semantics. A downstream author who trusts "derivative
at 0 is 0" will reason incorrectly about a curve that touches 0. The defect is
untested: the test at line 484 explicitly notes *"(Not probed at 0.)"*, so CI is
green despite the mismatch.

**Fix:** Either (a) make the behavior match the docs by special-casing zero, or
(b) correct the docs to state the real convention. Option (a), if the intended
convention truly is `signum(0)=0`:

```rust
#[inline]
fn abs(self) -> Self {
    // Subdifferential convention: derivative at exactly 0 is 0.
    let sub = if self.value == 0.0 { 0.0 } else { self.value.signum() };
    Dual {
        value: self.value.abs(),
        tangent: self.tangent * sub,
    }
}

#[inline]
fn signum(self) -> Self {
    // signum is piecewise-constant; tangent is 0 a.e.
    let v = if self.value == 0.0 { 0.0 } else { self.value.signum() };
    Dual { value: v, tangent: 0.0 }
}
```

If instead the `f64::signum` behavior (`±1` at zero) is acceptable, keep the code
and rewrite the docs at lines 96, 98–99, and 373 to say *"uses `f64::signum`, which
returns `+1.0` at `+0.0` and `-1.0` at `-0.0` — a valid but arbitrary subgradient
selection; the derivative of `|x|` is undefined at 0."* Add a test that asserts the
chosen behavior at 0 so it cannot silently drift. Note the `f64` `Scalar::signum`
passthrough (line 149) inherits the same `±1`-at-zero behavior and its doc (line 98)
is likewise wrong; fix both impls' contract consistently.

## Medium

### ME-01: Derived `PartialEq` (both fields) is inconsistent with hand-written value-only `PartialOrd`

**File:** `fdars-core/src/autodiff.rs:163` (derive) vs `:284-289` (impl)

**Issue:** `Dual` derives `PartialEq` (compares **both** `value` and `tangent`) but
hand-writes `PartialOrd` to compare `value` **only**. For two Duals with equal value
and different tangents:

- `a == b` → `false` (derived; tangents differ)
- `a.partial_cmp(&b)` → `Some(Ordering::Equal)`

This violates the standard-library contract that `a.partial_cmp(&b) == Some(Equal)`
must hold **iff** `a == b`. The test at lines 543–554 actually demonstrates the
discrepancy (it asserts `partial_cmp == Some(Equal)` for equal-value/different-tangent
Duals, which are `!=` under the derived `PartialEq`). Generic code that mixes `==`
and `<`/`<=`/`cmp`-style logic (e.g. a `min`/`clamp` helper, a sort, or a de-dup
that assumes trichotomy) can behave surprisingly — `!(a < b) && !(a > b)` implies
"equal" for ordering but `a != b` for equality. This is a latent correctness trap
for the Phase 76 DP code that branches on comparisons.

**Fix:** Make the two relations consistent. Simplest and safest: hand-write a
value-only `PartialEq` to match the value-only `PartialOrd`, so both agree on the
"primal decides" semantics:

```rust
#[derive(Debug, Clone, Copy)]
pub struct Dual { pub value: f64, pub tangent: f64 }

impl PartialEq for Dual {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
```

If full-struct `PartialEq` is genuinely needed elsewhere (e.g. exact result
comparison in tests), keep the derive but add a `debug_assert`-free module note
explicitly documenting that `PartialEq` and `PartialOrd` use *different* keys and
that generic code must not assume trichotomy — and audit Phase 76 call sites. The
value-only `PartialEq` is the cleaner choice given the module's stated "primal
decides control flow" invariant.

## Low

### LO-02: `sqrt`/`ln` zero-value tangent divergence is documented but untested — no guard test locks the behavior

**File:** `fdars-core/src/autodiff.rs:321-346`

**Issue:** `sqrt(0)` → tangent `t/(2·0) = ±Inf` (or `NaN` when `t==0`, since
`0.0/0.0`), and `ln(v)` at `v==0` → tangent `t/0`. This is genuine forward-mode
non-differentiability and is correctly documented (lines 54–58, 82, 86) with the
"callers own range checking" disposition — **acceptable**. However, no test pins the
edge behavior, so a future refactor (e.g. someone "helpfully" clamping the
denominator) could silently change the singular-point result without a failing test.

**Fix:** Add a small guard test asserting the documented edge outcome, e.g.:

```rust
#[test]
fn dual_sqrt_at_zero_tangent_is_nonfinite() {
    let r = Scalar::sqrt(Dual::seed(0.0));   // tangent = 1/(2*0)
    assert_eq!(r.value, 0.0);
    assert!(r.tangent.is_infinite());        // documented divergence
}
```

This converts an implicit contract into an enforced one without changing behavior.

### LO-03: `powf` singular/NaN edges (`v==0, p<1`; negative base, non-integer `p`) documented but untested

**File:** `fdars-core/src/autodiff.rs:363-370`

**Issue:** `powf` correctly implements `t·p·v^(p-1)`. At `v==0, p<1` the tangent is
`Inf` (confirmed: `0.0.powf(-0.5) == inf`); for a negative base with non-integer `p`
both value and tangent are `NaN`. Documented at lines 56–58, 92–93 — acceptable. As
with LO-02, none of these edges are tested, so the "NaN/Inf propagates like plain
`f64`" contract is unenforced. Note also `p == 1.0` at `v == 0.0` relies on
`0.0.powf(0.0) == 1.0` (Rust returns `1.0`), giving tangent `t·1·1 = t` — this is the
*correct* `d/dx x = 1` and is a nice property, but it too is untested.

**Fix:** Add one edge test asserting `Scalar::powf(Dual::seed(0.0), 0.5).tangent`
is `is_infinite()` and that `Scalar::powf(Dual::seed(0.0), 1.0)` gives tangent `1.0`,
locking the documented and the pleasant-accident behaviors respectively.

---

## Non-issues explicitly checked (confirmed correct)

- **All interior-domain derivative formulas** (mul product rule, div quotient rule,
  sqrt/exp/ln/sin/cos/powf chain terms, neg) — exact.
- **`f64` parity** — every `Scalar` method is a direct passthrough to the inherent
  `f64::method`; no expression reordering that could perturb IEEE-754 bits. Tier-3
  `assert_eq!` tests confirm bit-for-bit equality.
- **`infinity()`** — `value = +inf`, `tangent = 0.0` for `Dual`; sensible DP sentinel.
- **`PartialOrd` is hand-written value-only**, not derived — correct forward-mode
  branch semantics (the primary Phase 76 requirement). Confirmed no `#[derive(PartialOrd)]`.
- **Copy/Clone soundness** — `Dual` is two `f64`s; `Copy` is trivially sound.
- **No new dependency / no `num_traits`** — confirmed absent from `autodiff.rs`.
- **`lib.rs`** — single additive `pub mod autodiff;` line in correct alphabetical slot.

---

_Reviewed: 2026-09-06_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
