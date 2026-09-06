# Phase 75: Scalar Trait & Forward-Mode Dual Substrate — Research

**Researched:** 2026-09-06
**Domain:** In-crate forward-mode automatic differentiation (dual numbers, Rust)
**Confidence:** HIGH for codebase facts (verified by file reads); LOW for external textbook patterns (training knowledge, tagged `[ASSUMED]`)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **In-crate forward-mode dual numbers** — NOT hand-written per-op gradients, NOT an external AD crate. Gradients must compose through arbitrary op chains.
- **Forward-mode only** (JVP / dual numbers), matching the ForwardDiff reference. Reverse-mode/VJP is deferred (DIF-F1).
- **No new crate dependency** — the `Dual` type and `Scalar` trait are entirely in-crate.
- **Additive/non-breaking** — existing f64 public signatures are untouched; the `Scalar` trait must be implemented for `f64` so f64-instantiated generic code compiles and runs identically to the current numerics.

### Claude's Discretion
- **Op set:** `Dual` must support ±, ×, ÷, `sqrt`, `exp`, `ln`, `sin`, `cos`, `powf`, `abs`, and partial comparisons, plus seed and extract helpers.
- **`Scalar` trait shape:** define the minimal trait the differentiable subset (Phase 76) will be written against — arithmetic/transcendental ops, `From<f64>`/constants, and comparison as needed. Implement for both `f64` and `Dual<f64>`.
- **Module placement:** a new self-contained module `src/autodiff.rs` or `src/autodiff/` is the natural home; follow existing module/naming conventions.
- **num-traits reuse:** only if already a transitive/available dependency with no Cargo.toml change.

### Deferred Ideas (OUT OF SCOPE)
- Nested/higher-order duals (second derivatives) — not required this milestone.
- Reverse-mode/VJP — deferred (DIF-F1).
- Making existing f64 public signatures themselves generic — deferred (DIF-F3).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DIF-01 | In-crate `Scalar` trait + forward-mode `Dual<T>` number — dual carries value + tangent and implements ±, ×, ÷, `sqrt`, `exp`, `ln`, `sin`, `cos`, `powf`, `abs`, partial comparisons, plus gradient seed/extract helpers. Known-answer tests reproduce analytical derivatives of composed elementary functions to ≤1e-10. No new crate dependency. | Codebase analysis confirms no num-traits direct dep; trait shape derived from FPCA projection + soft-DTW op set; dual rules are textbook forward-mode; validation architecture designed around the ≤1e-10 gate. |
</phase_requirements>

---

## Summary

Phase 75 delivers the numeric substrate for forward-mode automatic differentiation (AD) in `fdars-core`. The work is self-contained: a new `src/autodiff.rs` module (or `src/autodiff/` directory) exposing a `Scalar` trait and a `Dual<T>` struct with full arithmetic and transcendental ops.

The codebase analysis yielded four decisive findings. First, `num-traits` is confirmed as a **transitive-only** dependency — pulled in via `nalgebra`, `num-complex`, and `rand_distr`, but **not listed in `fdars-core/Cargo.toml`**. In Rust 2021 edition with resolver v2, transitive deps are not directly usable without a declaration. Therefore the `Scalar` trait **must** be defined entirely in-crate; adding `num-traits` to `Cargo.toml` would violate the "no new crate dependency" constraint.

Second, reading `src/metric/soft_dtw.rs` (the validation oracle for Phase 76) reveals that its cost computation uses `f64` subtraction, multiplication, `min`, `exp`, `ln`, and `is_finite` checks. The DP recurrence compares three values with `min` (a `PartialOrd` comparison) — for `Dual`, these comparisons must operate on the **value field only** (the primal part), so that control flow is determined by the primal while tangents propagate as-is. This is the canonical forward-mode branching semantics.

Third, reading `src/regression.rs` FPCA `project()` shows that the score projection path (lines 118–125) uses only subtraction, multiplication, and addition — all elementwise scalar ops. This is the simplest possible op set for Phase 76's DIF-03 work; the `Scalar` trait's arithmetic suffices.

Fourth, the existing module structure (modules declared as `pub mod x;` in `lib.rs`, minimal re-exports for now, expanded in Phase 77) is clear from reading `src/lib.rs`. The autodiff module should follow the same pattern.

**Primary recommendation:** Implement `src/autodiff.rs` with an in-crate `Scalar` trait and `Dual<f64>` struct (not generic `Dual<T>` for this phase unless it composes cleanly) with full op coverage, registered as `pub mod autodiff;` in `lib.rs` with no prelude re-exports yet.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `Scalar` trait definition | `src/autodiff.rs` (new) | — | Self-contained in-crate; consumed by Phase 76's generic code |
| `Dual<f64>` struct + ops | `src/autodiff.rs` (new) | — | All arithmetic + transcendental in one module |
| `Scalar` impl for `f64` | `src/autodiff.rs` (new) | — | Required for the additive/non-breaking guarantee |
| Gradient seed/extract helpers | `src/autodiff.rs` (new) | — | Thin wrappers over `Dual` constructors |
| Module registration | `src/lib.rs` | — | Follow existing `pub mod x;` pattern |
| Prelude re-exports | Deferred to Phase 77 | — | DIF-04's job |
| Known-answer tests | Inline `#[cfg(test)] mod tests` in `src/autodiff.rs` | — | Crate convention (inline tests per module) |

---

## Standard Stack

### Core (no new dependencies — all in-crate)

| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| `std::ops::{Add,Sub,Mul,Div,Neg,AddAssign,MulAssign,DivAssign,SubAssign}` | std | Arithmetic operator overloading | Required for `a + b`, `a * b` syntax in generic code |
| `std::cmp::PartialOrd` | std | Comparison for control flow | Needed for `if a < b` in DTW recurrences |
| `std::fmt::Debug` | std | Debug output on public type | Project convention: `#[derive(Debug, Clone, PartialEq)]` on all public types |
| `f64` intrinsic methods | std | `sqrt`, `exp`, `ln`, `sin`, `cos`, `powf`, `abs`, `signum` | Available on `f64` directly; Dual delegates to inner `f64` for primal, applies chain rule for tangent |

### No External Dependencies Required

`num-traits` is **NOT** a direct dependency of `fdars-core` [VERIFIED: fdars-core/Cargo.toml — no `num-traits` entry in `[dependencies]`] and may not be used without a `Cargo.toml` change. The entire `Scalar` trait is defined in-crate.

**Installation:** No `cargo add` required. This phase adds zero new dependencies.

---

## Package Legitimacy Audit

> Not applicable — this phase installs no new packages.

| Package | Registry | Verdict | Disposition |
|---------|----------|---------|-------------|
| (none) | — | — | N/A |

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious SUS:** none

---

## Architecture Patterns

### System Architecture Diagram

```
External generic code (Phase 76+)
       │ uses
       ▼
┌──────────────────────────────────────┐
│  Scalar trait (src/autodiff.rs)      │
│  + arithmetic + transcendental ops   │
│  + zero() / one() / from_f64()       │
│  + PartialOrd (value-only for Dual)  │
└──────────┬─────────────┬────────────┘
           │ impl        │ impl
           ▼             ▼
      ┌─────────┐   ┌─────────────────────────┐
      │   f64   │   │  Dual<f64>               │
      │ (passthrough│  { value: f64,           │
      │  to std) │   │    tangent: f64 }        │
      └─────────┘   │  (chain rule in each op) │
                    └──────────────┬──────────┘
                                   │ used by
                           ┌───────▼────────┐
                           │  seed(x, v)    │  set tangent = 1.0
                           │  extract(d)    │  return (d.value, d.tangent)
                           └────────────────┘
```

### Recommended Project Structure

```
fdars-core/src/
├── autodiff.rs      # NEW: Scalar trait + Dual<f64> + impls + tests (single file preferred)
│                    # Alternative: autodiff/ dir if the file grows >300 lines
├── lib.rs           # Add: pub mod autodiff;
└── ...              # Everything else unchanged
```

### Pattern 1: Scalar Trait Definition

**What:** Minimal in-crate trait bounding the op set Phase 76 needs.
**When to use:** Every generic function in Phases 76–77 writes `<S: Scalar>`.

```rust
// Source: in-crate (no external reference — textbook forward-mode AD pattern) [ASSUMED]
use std::fmt::Debug;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign};

pub trait Scalar:
    Copy
    + Clone
    + Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(v: f64) -> Self;

    // Transcendental ops (chain rule in Dual; direct f64 method for f64)
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn powf(self, p: f64) -> Self;   // fixed-f64 exponent (not generic S) [ASSUMED]
    fn abs(self) -> Self;
    fn signum(self) -> Self;         // needed for abs derivative
}
```

**Note on `powf` exponent type:** `powf(self, p: f64)` uses a concrete `f64` exponent rather than `Self` because the derivative rule `d/dx x^p = p * x^(p-1)` multiplies by the scalar `p` which is always a plain float in practice. A `powf(self, p: Self)` variant would require the exponent's tangent to be tracked as well (general case `d/dx f^g = f^g * (g' ln f + g f'/f)`), which is not needed for Phase 76. [ASSUMED]

### Pattern 2: Dual<f64> Struct

**What:** Value + tangent pair implementing `Scalar` via chain rules.
**When to use:** Instantiate generic functions at `Dual<f64>` to extract derivatives.

```rust
// Source: textbook forward-mode dual number, in-crate implementation [ASSUMED]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    pub value: f64,
    pub tangent: f64,
}

impl Dual {
    /// Seed: set this input's tangent to 1, all others to 0.
    #[inline]
    pub fn seed(value: f64) -> Self {
        Dual { value, tangent: 1.0 }
    }

    /// Constant: no gradient flows through a constant.
    #[inline]
    pub fn constant(value: f64) -> Self {
        Dual { value, tangent: 0.0 }
    }

    /// Extract the (value, derivative) pair after a computation.
    #[inline]
    pub fn extract(self) -> (f64, f64) {
        (self.value, self.tangent)
    }
}
```

**Note on naming:** The type is `Dual` (not `Dual<T>`) for this phase because nested/higher-order duals are deferred. If Phase 76 needs `Dual<T: Scalar>` for composability, the planner can decide; but the constraints say higher-order AD is out of scope. A simple `Dual` (concrete `f64` inner) satisfies all Phase 75 requirements. [ASSUMED]

### Pattern 3: Operator Implementations — Arithmetic

```rust
// Chain rules for Dual arithmetic [ASSUMED]
use std::ops::{Add, Mul, Sub, Div, Neg, AddAssign, SubAssign, MulAssign};

impl Add for Dual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Dual { value: self.value + rhs.value, tangent: self.tangent + rhs.tangent }
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Dual { value: self.value - rhs.value, tangent: self.tangent - rhs.tangent }
    }
}

impl Mul for Dual {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // Product rule: d(u*v) = u*v' + u'*v
        Dual {
            value: self.value * rhs.value,
            tangent: self.tangent * rhs.value + self.value * rhs.tangent,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        // Quotient rule: d(u/v) = (u'*v - u*v') / v^2
        let v2 = rhs.value * rhs.value;
        Dual {
            value: self.value / rhs.value,
            tangent: (self.tangent * rhs.value - self.value * rhs.tangent) / v2,
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Dual { value: -self.value, tangent: -self.tangent }
    }
}
```

### Pattern 4: Transcendental Ops via Scalar impl

```rust
// Scalar impl for Dual — chain rule for each transcendental [ASSUMED]
impl Scalar for Dual {
    fn zero() -> Self { Dual { value: 0.0, tangent: 0.0 } }
    fn one()  -> Self { Dual { value: 1.0, tangent: 0.0 } }
    fn from_f64(v: f64) -> Self { Dual { value: v, tangent: 0.0 } }

    fn sqrt(self) -> Self {
        let s = self.value.sqrt();
        Dual { value: s, tangent: self.tangent / (2.0 * s) }
    }
    fn exp(self) -> Self {
        let e = self.value.exp();
        Dual { value: e, tangent: self.tangent * e }
    }
    fn ln(self) -> Self {
        Dual { value: self.value.ln(), tangent: self.tangent / self.value }
    }
    fn sin(self) -> Self {
        Dual { value: self.value.sin(), tangent: self.tangent * self.value.cos() }
    }
    fn cos(self) -> Self {
        Dual { value: self.value.cos(), tangent: -self.tangent * self.value.sin() }
    }
    fn powf(self, p: f64) -> Self {
        let vp = self.value.powf(p);
        Dual { value: vp, tangent: self.tangent * p * self.value.powf(p - 1.0) }
    }
    fn abs(self) -> Self {
        Dual { value: self.value.abs(), tangent: self.tangent * self.value.signum() }
    }
    fn signum(self) -> Self {
        // signum is a step function — its derivative is 0 almost everywhere
        Dual { value: self.value.signum(), tangent: 0.0 }
    }
}
```

### Pattern 5: PartialOrd for Dual (value-only comparison)

**Critical for control flow in DTW recurrences.**

```rust
// PartialOrd compares only the primal (value) field [ASSUMED]
impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}
```

This is correct for forward-mode AD: branching decisions (which path DTW takes) are determined by the primal value; the tangent propagates through the selected branch automatically. This matches how ForwardDiff.jl handles `min`/`max` in Julia.

### Pattern 6: Scalar impl for f64

```rust
// f64 impl delegates to inherent methods — zero overhead, identical numerics [ASSUMED]
impl Scalar for f64 {
    fn zero() -> Self { 0.0 }
    fn one()  -> Self { 1.0 }
    fn from_f64(v: f64) -> Self { v }

    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn exp(self)  -> Self { f64::exp(self) }
    fn ln(self)   -> Self { f64::ln(self) }
    fn sin(self)  -> Self { f64::sin(self) }
    fn cos(self)  -> Self { f64::cos(self) }
    fn powf(self, p: f64) -> Self { f64::powf(self, p) }
    fn abs(self)  -> Self { f64::abs(self) }
    fn signum(self) -> Self { f64::signum(self) }
}
```

The `f64` impl is a zero-cost passthrough: every call inlines to the exact same `f64` intrinsic the original code called, satisfying the additive/non-breaking guarantee.

### Pattern 7: Gradient Helper Functions

```rust
// Public API helpers for computing directional derivatives [ASSUMED]
/// Compute the derivative of f at x in direction 1.0 (standard partial derivative w.r.t. x).
#[must_use]
pub fn diff<F: Fn(Dual) -> Dual>(f: F, x: f64) -> (f64, f64) {
    f(Dual::seed(x)).extract()
}
```

### Anti-Patterns to Avoid

- **Impl `PartialOrd` comparing tangents:** Dual tangents do not represent magnitude — comparing them as a tiebreaker makes branching semantics undefined. Compare the value field only.
- **Deriving `PartialOrd` on `Dual`:** `#[derive(PartialOrd)]` compares fields lexicographically (value first, then tangent) — this would add tangent comparison as a tiebreaker. Implement `PartialOrd` manually with value-only comparison.
- **Storing `Dual` in `FdMatrix`:** `FdMatrix` is `Vec<f64>` column-major — it does not hold `Dual` values. Generic functions in Phase 76 that operate over `Scalar` will work with `Vec<S>` or inline loops, not `FdMatrix`.
- **Using `f64::MIN` as infinity sentinel in generic code:** `f64::INFINITY` has no `Scalar` equivalent in a minimal trait. Generic soft-DTW (Phase 76) will need an `infinity()` method in `Scalar`. Include it in the trait design now even if Phase 75 tests don't exercise it directly, so Phase 76 does not need a trait-breaking change. [ASSUMED — check at plan time whether to include `infinity()` in the trait or add it for Phase 76]
- **Placing autodiff types in `prelude`:** Phase 77 (DIF-04) owns the full crate-root + prelude surface. Phase 75 only registers `pub mod autodiff;` in `lib.rs`.

---

## num-traits Dependency Finding (Critical)

**Finding:** `num-traits v0.2.19` is a transitive dependency of `fdars-core`, pulled in by `nalgebra`, `num-complex`, `rand_distr`, and others. [VERIFIED: Cargo.lock / `cargo tree` output examined this session]

**Finding:** `num-traits` is **not** listed as a direct dependency in `fdars-core/Cargo.toml`. [VERIFIED: fdars-core/Cargo.toml — no `num-traits` entry in `[dependencies]` or `[dev-dependencies]`]

**Implication:** In Rust 2021 edition with workspace resolver v2, a transitive-only dependency cannot be `use`-d in `fdars-core/src/` without adding it to `Cargo.toml`. Adding it would violate the "no new crate dependency" constraint. Therefore:

- The `Scalar` trait is **defined entirely in-crate** — no `use num_traits::...` anywhere in `autodiff.rs`.
- The trait can *mirror* the structure of `num_traits::Float` but must not depend on it.

---

## Soft-DTW Op Analysis (Phase 76 oracle)

Reading `src/metric/soft_dtw.rs` (lines 29–39, 154–168) reveals the ops the generic soft-DTW path will need from `Scalar`: [VERIFIED: fdars-core/src/metric/soft_dtw.rs:29-39, 155-168]

```
// softmin3 uses: subtraction, negation, division, exp, ln, addition, multiplication
// Verbatim from soft_dtw.rs lines 29-39:
let neg_inv_gamma = -1.0 / gamma;      // sub, div, neg → Scalar: div, neg
let ea = ((a - min_val) * neg_inv_gamma).exp();  // sub, mul, exp → Scalar: sub, mul, exp
// (ea + eb + ec).ln()                           // add, ln → Scalar: add, ln
min_val - gamma * (ea + eb + ec).ln()           // sub, mul, ln

// DP recurrence (lines 60-70):
let d = x[i - 1] - y[j - 1];          // sub → Scalar: sub
let cost = d * d;                       // mul → Scalar: mul
curr[j] = cost + softmin3(...)         // add → Scalar: add
```

The `min` call (`a.min(b).min(c)` for the `min_val` sentinel) is called on `f64` directly in the existing code and requires `PartialOrd`. For the generic path in Phase 76, the generic soft-DTW will need a `Scalar::min(self, other) -> Self` method or the function will use `if a < b { a } else { b }` style branching (which flows correctly through forward-mode). Include either approach; branching on value is correct.

Also notable: `r[i][j]` is initialized to `f64::INFINITY`. The generic path needs `Scalar::infinity() -> Self` or a convention for the initial sentinel. This should be in the `Scalar` trait to avoid Phase 76 needing a trait change.

**Ops required by soft-DTW:**
`sub`, `mul`, `div`, `neg`, `add`, `exp`, `ln`, `PartialOrd` (for min/branching), `infinity` (for DP initialization)

---

## FPCA Score Projection Op Analysis

Reading `src/regression.rs` `FpcaResult::project()` (lines 105–126) shows the score projection loop: [VERIFIED: fdars-core/src/regression.rs:118-125]

```rust
// Verbatim from regression.rs lines 118-125:
for i in 0..n {
    for k in 0..ncomp {
        let mut sum = 0.0;
        for j in 0..m {
            sum += (data[(i, j)] - self.mean[j]) * self.rotation[(j, k)] * self.weights[j];
        }
        scores[(i, k)] = sum;
    }
}
```

The generic version of this for Phase 76 (DIF-03) needs only: `sub`, `mul`, `add` (via `+=`), plus the ability to initialize `sum` as `Scalar::zero()`. This is the simplest possible generic path. The eigenbasis (`rotation`) and mean stay as `f64` (precomputed); only the input curve values become generic `S: Scalar`. The generic score is `S`, and the multiplication by `f64` weights/rotation entries requires either `Scalar::from_f64()` or a separate `Scalar * f64` multiply. This detail should be resolved at plan time.

**Ops required by FPCA score projection:**
`sub`, `mul`, `add` (addassign), `zero()`, and scalar-times-f64 multiply.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Chain-rule bookkeeping per op | Per-function gradient with hand-coded Jacobians (like current `soft_dtw_backward`) | `Dual<f64>` with operator overloading | Operators compose automatically; hand-coded gradients don't |
| Derivative validation | Manual comparison only | Central finite differences cross-check | FD catches sign errors and missing chain-rule terms cheaply |
| `f64` wrapper | A newtype that delegates everything | Just implement `Scalar for f64` directly | No allocation, inlines to same instructions |

**Key insight:** The whole point of the dual number approach is that composing operators composes derivative rules automatically. The existing hand-written `soft_dtw_backward` gradient (in `soft_dtw_accumulate_gradient`) does *not* compose — it is bespoke to soft-DTW and will be the validation oracle *for Phase 76*, not replaced in Phase 75.

---

## Common Pitfalls

### Pitfall 1: `abs` Derivative at Zero

**What goes wrong:** `abs'(0.0)` is undefined mathematically; `signum(0.0) = 0.0` in Rust `f64`. Using `signum` for the tangent of `abs` gives `tangent = 0.0` at zero — this is the standard subdifferential convention used by all major AD frameworks.
**Why it happens:** Strict mathematical undefined-at-zero vs. practical convention.
**How to avoid:** Use `self.value.signum()` in the `abs` impl; document the subdifferential convention. Tests should not probe `abs` exactly at zero for the ≤1e-10 gate.
**Warning signs:** Test value is exactly `0.0` — move the test point slightly off zero.

### Pitfall 2: Deriving `PartialOrd` on Dual

**What goes wrong:** `#[derive(PartialOrd)]` compares fields lexicographically — the tangent becomes a tiebreaker, creating semantically wrong comparisons (e.g., `Dual{value:1.0, tangent:0.5} > Dual{value:1.0, tangent:0.3}` would be `true` even though they represent the same primal point).
**Why it happens:** Rust's derive macro does lexicographic field order.
**How to avoid:** Manually implement `PartialOrd` with `self.value.partial_cmp(&other.value)` only.
**Warning signs:** Branching tests give wrong derivatives in DTW-like recurrences.

### Pitfall 3: Division by Zero in `sqrt` and `ln`

**What goes wrong:** `Dual::sqrt` computes `tangent / (2.0 * sqrt(value))` — if `value == 0.0`, the tangent diverges. `Dual::ln` divides by `value` — if `value == 0.0` or `value < 0`, the tangent is `±inf` or `NaN`.
**Why it happens:** Domain constraints of transcendental functions.
**How to avoid:** Known-answer tests must use values well inside the domain (e.g., `x = 2.0` for `sqrt`, not `x = 0.0`). Document domain restrictions in the `Scalar` trait's method docs.
**Warning signs:** NaN or Inf in computed tangents.

### Pitfall 4: `powf` Edge Cases

**What goes wrong:** `powf(self, 0.0)` gives `value = 1.0, tangent = 0 * v^(-1)` — when `value = 0.0`, `tangent = 0/0 = NaN` even though the mathematical derivative is 0. Similarly `powf(self, 1.0)` is always safe.
**Why it happens:** `0 * (0.0).powf(-1.0) = 0 * inf = NaN` in IEEE 754.
**How to avoid:** Special-case `p == 0.0` or `p == 1.0`; or document that `p > 1` is required when `value` may be zero.
**Warning signs:** NaN tangent for non-negative inputs.

### Pitfall 5: `AddAssign` / Compound Assignment Ops

**What goes wrong:** The FPCA projection loop uses `sum += ...`. If `Scalar` does not bound `AddAssign<Output = ()>`, the generic loop must rewrite as `sum = sum + ...` (allocation-free, works), but the natural idiom won't compile.
**Why it happens:** `AddAssign` is a separate trait from `Add`.
**How to avoid:** Include `AddAssign + SubAssign + MulAssign` in the `Scalar` trait bounds. The `Dual` type must implement them too.

### Pitfall 6: Clippy `--all-targets` catches test-code warnings

**What goes wrong:** CI runs `cargo clippy --all-targets --features linalg,parallel -- -D warnings`. If inline test code in `src/autodiff.rs` uses patterns like `#[allow(...)]` without the feature guard, it surfaces warnings.
**Why it happens:** `--all-targets` also lints `#[cfg(test)]` blocks.
**How to avoid:** Keep test code clean. Any `#[allow]` in tests is acceptable if justified, but avoid introducing new clippy suppressions.

---

## Validation Architecture

This is the make-or-break gate for DIF-01. Three tiers of tests are needed:

### Tier 1: Known-Answer Tests (≤1e-10 gate)

Verify that dual arithmetic exactly reproduces the **analytical derivative** of composed elementary functions. These are the DIF-01 acceptance tests.

**Test design principle:** Choose functions with closed-form derivatives that exercise multiple ops in composition. Evaluate at a specific `x` value, compute both primal and tangent, compare against closed-form.

**Required known-answer tests:**

| Test Function | Seed at | Closed-form f(x) | Closed-form f'(x) | Ops Exercised |
|--------------|---------|-------------------|-------------------|---------------|
| `f(x) = x^2` | `x=3.0` | `9.0` | `6.0` | `mul` (self*self) |
| `f(x) = sqrt(x)` | `x=4.0` | `2.0` | `0.25` | `sqrt` |
| `f(x) = exp(x)` | `x=1.0` | `e ≈ 2.71828` | `e ≈ 2.71828` | `exp` |
| `f(x) = ln(x)` | `x=2.0` | `ln(2) ≈ 0.6931` | `0.5` | `ln` |
| `f(x) = sin(x)` | `x=π/4` | `√2/2 ≈ 0.7071` | `cos(π/4) ≈ 0.7071` | `sin` |
| `f(x) = cos(x)` | `x=π/4` | `√2/2 ≈ 0.7071` | `-sin(π/4) ≈ -0.7071` | `cos` |
| `f(x) = x^1.5` | `x=4.0` | `8.0` | `3.0` | `powf` |
| `f(x) = abs(x)` | `x=2.0` | `2.0` | `1.0` (signum) | `abs` |
| `f(x) = exp(x) * sin(x)` | `x=1.0` | `e*sin(1)` | `e*(sin(1)+cos(1))` | `exp`, `sin`, `mul` |
| `f(x) = sqrt(exp(x)*sin(x) + x^2)` (CONTEXT.md example) | `x=1.0` | closed-form | chain rule | all ops composed |
| `f(x) = (x - c) * w` where c,w are constants | varies | linear | `w` | `sub`, `mul`, constants |

The composed test `sqrt(exp(x)*sin(x) + x^2)` is the non-trivial chain demanded by CONTEXT.md (specifics section). For `x=1.0`:
- `f(1) = sqrt(e * sin(1) + 1) ≈ sqrt(2.71828 * 0.84147 + 1) ≈ sqrt(3.28705) ≈ 1.81304`
- `f'(1) = (1/(2*f(1))) * (e*(sin(1)+cos(1)) + 2)` — compute analytically and compare.

**Test pattern (inline in `src/autodiff.rs`):**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    const TOL: f64 = 1e-10;

    #[test]
    fn dual_mul_known_answer() {
        let d = Dual::seed(3.0);
        let r = d * d;  // f(x) = x^2, f'(x) = 2x
        assert!((r.value - 9.0).abs() < TOL, "primal: {} != 9.0", r.value);
        assert!((r.tangent - 6.0).abs() < TOL, "tangent: {} != 6.0", r.tangent);
    }

    #[test]
    fn dual_sqrt_known_answer() {
        let d = Dual::seed(4.0);
        let r = Scalar::sqrt(d);  // f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
        assert!((r.value - 2.0).abs() < TOL);
        assert!((r.tangent - 0.25).abs() < TOL);
    }

    // ... one test per op and the composed chain test
}
```

### Tier 2: Finite-Difference Cross-Check (≤1e-6 tolerance)

Cross-validate `Dual` derivatives against central finite differences for several multi-op compositions. FD cannot reach ≤1e-10 (truncation error at h ≈ 1e-8 gives FD error ≈ 1e-6), so this tier uses a relaxed 1e-6 tolerance. [ASSUMED]

**Formula:** `f'(x) ≈ (f(x+h) - f(x-h)) / (2h)` with `h = 1e-8`

**Rationale:** FD catches sign errors, wrong chain-rule applications, and missing terms that known-answer tests might coincidentally pass (e.g., if the test point is a coincidental zero). Use 3–5 random-ish inputs.

```rust
#[test]
fn finite_diff_cross_check_composed() {
    let h = 1e-8_f64;
    let f = |x: f64| (x.exp() * x.sin() + x * x).sqrt();
    let x0 = 1.0_f64;
    let fd_deriv = (f(x0 + h) - f(x0 - h)) / (2.0 * h);
    let (_, ad_deriv) = diff(|d| {
        let e = Scalar::exp(d);
        let s = Scalar::sin(d);
        let x2 = d * d;
        let inner = e * s + x2;
        Scalar::sqrt(inner)
    }, x0);
    assert!((ad_deriv - fd_deriv).abs() < 1e-6,
            "AD={ad_deriv} FD={fd_deriv}");
}
```

### Tier 3: f64 Parity Check (≤1e-14 gate)

Verify that `Scalar for f64` returns identical (or within floating-point rounding, i.e., 0 ULP difference) results to calling the same `f64` methods directly.

**Rationale:** Guarantees the additive/non-breaking invariant — instantiating generic code at `f64` is numerically identical to the original non-generic code.

```rust
#[test]
fn f64_parity_transcendentals() {
    let x = 2.5_f64;
    assert_eq!(<f64 as Scalar>::sqrt(x), x.sqrt());
    assert_eq!(<f64 as Scalar>::exp(x), x.exp());
    assert_eq!(<f64 as Scalar>::ln(x), x.ln());
    assert_eq!(<f64 as Scalar>::sin(x), x.sin());
    assert_eq!(<f64 as Scalar>::cos(x), x.cos());
    assert_eq!(<f64 as Scalar>::powf(x, 1.5), x.powf(1.5));
    assert_eq!(<f64 as Scalar>::abs(-x), (-x).abs());
}
```

### Test Run Commands

| Gate | Command | Expected |
|------|---------|----------|
| Quick (per commit) | `cargo test -p fdars-core autodiff` | All autodiff tests green |
| Full suite | `cargo test -p fdars-core` | No regressions |
| Clippy gate | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Zero warnings |
| Fmt gate | `cargo fmt --check` | No drift |
| Doc test | `cargo test --doc -p fdars-core` | Module doctest green |

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|-----------------|-------|
| Hand-written gradients per function (current `soft_dtw_backward`) | Forward-mode dual numbers composing through ops | Dual approach enables gradients through any composed op chain |
| External crates (num-dual, autodiff, enzyme) | In-crate `Dual` + `Scalar` | Avoids dependency; sufficient for forward-mode |
| `num-traits::Float` bound | Custom in-crate `Scalar` trait | Avoids adding num-traits to direct deps |

**Relevant Rust ecosystem projects (informational, not used here):** [ASSUMED]
- `num-dual` — clean dual-number crate (MIT); not used because it would be a new dependency.
- `autodiff` — Enzyme-backed AD; requires nightly and LLVM; incompatible with MSRV 1.81.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `PartialOrd` comparing value-only is the correct forward-mode branching semantics for DTW | Architecture Patterns §5 | If wrong, DTW gradient would be incorrect in Phase 76 |
| A2 | `powf(self, p: f64)` with a concrete `f64` exponent (not `Self`) is the right signature | Pattern §1 Scalar trait | Phase 76 may need `powf(self, p: Self)` — check ops actually used |
| A3 | Central FD with h=1e-8 gives absolute error ≈ 1e-6 for smooth functions | Validation §Tier2 | If test functions have large 3rd derivatives, error may exceed 1e-6; use tighter FD tolerance in that case |
| A4 | `signum(0.0) = 0.0` in Rust `f64` — the subdifferential convention for `abs'(0)` | Pitfalls §1 | No risk — Rust `f64::signum(0.0) = 0.0` is documented |
| A5 | `Scalar::infinity()` should be added to the trait for DP sentinel initialization | Don't Hand-Roll / Pitfalls | If omitted, Phase 76 either needs a trait-breaking change or a workaround |
| A6 | `Dual` as a concrete `Dual` (not generic `Dual<T>`) is sufficient for Phase 75 | Architecture §Pattern §2 | If Phase 76 needs `Dual<T: Scalar>` for composability, trait signature changes slightly |

**A4 is not actually an assumption** — `f64::signum(0.0)` returning `0.0` is verified behavior in Rust's standard library. All other A-items are [ASSUMED].

---

## Open Questions

1. **Should `Scalar` include `infinity()` now or in Phase 76?**
   - What we know: soft-DTW uses `f64::INFINITY` as a DP sentinel; the generic version needs an infinity constant.
   - What's unclear: does Phase 76 require it in the `Scalar` trait, or will a `where S: Scalar, f64: Into<S>` workaround suffice?
   - Recommendation: Add `fn infinity() -> Self` to `Scalar` now; remove at plan time if not needed. Breaking a trait in Phase 76 is more expensive than adding one method in Phase 75.

2. **`Dual` struct vs. `Dual<T: Scalar>` generic?**
   - What we know: higher-order AD is deferred; CONTEXT.md says "Dual<T: Scalar> if it composes cleanly."
   - What's unclear: whether Phase 76's generic elastic distance code needs `Dual<Dual<f64>>` (nested).
   - Recommendation: Start with concrete `Dual` (aliases `Dual<f64>` effectively); the planner can revisit if Phase 76 requires nesting.

3. **`Scalar * f64` multiply for FPCA projection?**
   - What we know: the projection loop multiplies a `Scalar` input difference by `f64` constants (rotation matrix entries, integration weights).
   - What's unclear: should `Scalar` bound `Mul<f64, Output = Self>` or should we use `S::from_f64(w) * s`?
   - Recommendation: `S::from_f64(w) * s` avoids extra trait bounds in Phase 75; a dedicated `Mul<f64>` impl can be added in Phase 76 if awkward.

---

## Environment Availability

> Step 2.6: This phase has no external dependencies beyond the Rust toolchain.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | cargo build/test | ✓ | 1.97.0 | — |
| cargo clippy | CI gate | ✓ | (bundled with 1.97.0) | — |
| cargo fmt | fmt gate | ✓ | (bundled with 1.97.0) | — |

No missing dependencies.

---

## Security Domain

> `security_enforcement: true` per config.json. Applicable ASVS categories for a pure numeric substrate:

| ASVS Category | Applies | Control |
|---------------|---------|---------|
| V5 Input Validation | Minimal — domain checks (positive for sqrt/ln) documented in trait | Document in method docs |
| V2–V4 Auth/Session/Access | No — library type, no auth path | N/A |
| V6 Cryptography | No — not a crypto primitive | N/A |

No significant security surface in a numeric substrate. Domain violations (sqrt of negative, ln of zero) produce `NaN`/`Inf` which propagate through computation — the calling code is responsible for range checks, same as with `f64`.

---

## Sources

### Primary (HIGH confidence — verified by file reads this session)
- `fdars-core/Cargo.toml` — verified absence of `num-traits` direct dependency
- `fdars-core/src/metric/soft_dtw.rs:29-39, 155-168` — op set analysis for soft-DTW generic path
- `fdars-core/src/regression.rs:118-125` — op set analysis for FPCA score projection
- `fdars-core/src/lib.rs` — module registration pattern
- `fdars-core/src/error.rs` — `FdarError` enum variants (verbatim)
- `fdars-core/src/parallel.rs` — macro pattern, feature gate convention

### Secondary (LOW confidence — training knowledge, marked [ASSUMED])
- Forward-mode dual number arithmetic rules (textbook: Add, Sub, Mul, Div, chain rules for transcendentals)
- `PartialOrd` value-only comparison for branching semantics in forward-mode AD
- Central finite difference formula and tolerance analysis
- Rust `std::ops` trait implementation patterns for custom numeric types

---

## Metadata

**Confidence breakdown:**
- Codebase facts (module structure, deps, FPCA/DTW ops): HIGH — read directly from source files
- Dual number arithmetic rules: LOW — textbook knowledge, tagged [ASSUMED]
- Validation architecture: MEDIUM — FD/known-answer approach is standard; specific tolerances are [ASSUMED]

**Research date:** 2026-09-06
**Valid until:** 2026-10-06 (codebase is stable; dual-number math is timeless)
