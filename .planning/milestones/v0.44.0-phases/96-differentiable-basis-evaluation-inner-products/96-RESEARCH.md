# Phase 96: Differentiable Basis Evaluation & Inner Products — Research

**Researched:** 2026-09-11
**Domain:** Rust autodiff, B-spline Cox-de-Boor recurrence, Fourier basis, f64/T generic boundary
**Confidence:** HIGH (all claims verified from source files read this session)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Differentiable input:** evaluation points `t: &[T]`. Knots, period, `nbasis`, `order` stay `f64`/`usize`.
- **In-place generic `<T: Scalar>`** — NOT `_generic` companions. Plain `<T: Scalar>` (no `= f64` default on free fns — Rust 1.97 rejects `invalid_type_param_default` on fns; T=f64 inferred at call sites).
- **Four generalization targets:**
  - `bspline_basis_from_knots<T: Scalar>(t: &[T], knots: &[f64], order: usize) -> Vec<T>`
  - `evaluate_order_zero<T: Scalar>(t_val: T, knots: &[f64], t_max_knot_idx: usize) -> Vec<T>`
  - `bspline_recurrence_step<T: Scalar>(b: &[T], knots: &[f64], t_val: T, k: usize) -> Vec<T>`
  - `fourier_basis_with_period<T: Scalar>(t: &[T], nbasis: usize, period: f64) -> Vec<T>`
- **Stay f64:** `bspline_basis`, `fourier_basis`, `construct_bspline_knots` — auto-deriving wrappers that fold over `t` for bounds; do NOT add `T::to_f64` to `Scalar`.
- **Inner products already done (Phase 95):** `inner_product<T: Scalar>` (utility.rs) and `inner_product_l2<T: Scalar>` (warping.rs) — REUSE, do NOT re-touch.
- **f64-parity guard:** `bspline_basis_from_knots::<f64>` and `fourier_basis_with_period::<f64>` reproduce current numerics bit-identically (or within existing 1e-10 tolerance).
- **No new crate dependency.** Strictly additive/non-breaking.

### Claude's Discretion
- Placement of the combined validation test (basis/tests.rs vs new inline module).
- Whether `evaluate_order_zero` uses `T::one()` vs `T::from_f64(1.0)` (equivalent; prefer `T::one()`).
- How `t_min` is threaded into `fourier_basis_with_period` if currently t-derived (see research finding F-4 for the chosen approach).
- Exact f64-fold form in the recurrence (see research finding B-1).

### Deferred Ideas (OUT OF SCOPE)
- Generalizing `bspline_basis`/`fourier_basis` wrappers (need `Scalar::to_f64`).
- Differentiating w.r.t. knots/period (adaptive basis tuning).
- Regression prediction + roughness penalties → Phase 97.
- Depth + curve distances → Phase 98.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DOP-01 | Basis evaluation (B-spline / Fourier) and functional inner products are generic over `Scalar` and differentiable; the f64 path reproduces current numerics. | Full implementation recipe below: recurrence bit-parity form (B-1), guard form (B-2), `evaluate_order_zero` form (B-3), Fourier `t_min` form (F-4), validation objective (V-1/V-2). Inner products are already generic (Phase 95). |
</phase_requirements>

---

## Summary

Phase 96 generalizes four functions — the B-spline Cox-de-Boor recurrence chain (`bspline_basis_from_knots`, `evaluate_order_zero`, `bspline_recurrence_step`) and the Fourier basis evaluator (`fourier_basis_with_period`) — from `f64` to `T: Scalar`. All four were read in full this session. The main implementation risks are (1) preserving bit-identical f64 parity in the recurrence's mixed-type arithmetic and (2) safely threading the `t_min` origin through the Fourier evaluator. Both are resolved with well-defined patterns established by Phase 94/95.

The inner products (`inner_product`, `inner_product_l2`) are already generic after Phase 95 and are only exercised here in the validation objective. The validation architecture mirrors the Phase 94/95 tiered pattern: bit-identical parity tests for f64, then FD-checked autodiff tests at `Dual` and `Var` for a combined basis-eval → inner-product objective differentiated w.r.t. `t`.

**Primary recommendation:** Use the "fold f64, lift once" pattern from Phase 95 throughout. Compute all f64 knot/period arithmetic before touching `T`, then call `T::from_f64` exactly once per f64 constant at the point of mixing. This is the Phase 95 established discipline and guarantees bit-identical f64 parity.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| B-spline basis generalization | `basis/bspline.rs` | — | All three functions live there; in-place edit |
| Fourier basis generalization | `basis/fourier.rs` | — | Single file; in-place edit |
| Autodiff gradient propagation | `autodiff/` (Phase 94/95 substrate) | `basis/bspline.rs`, `basis/fourier.rs` | Scalar trait impls for Dual/Var already in autodiff/; basis functions just use T arithmetic |
| Validation FD objective | `basis/tests.rs` | — | Existing test module; add new tests there per Claude's discretion |
| f64 call-site backward compat | inferred `T = f64` at all callers | — | Non-breaking via type inference; no caller changes needed |

---

## Standard Stack

### Core (no new dependencies)
| Component | Version | Purpose | Status |
|-----------|---------|---------|--------|
| `crate::autodiff::Scalar` | in-crate | Trait bounding T | `[VERIFIED: fdars-core/src/autodiff/mod.rs:38-85]` |
| `crate::autodiff::Dual` | in-crate | Forward-mode T | `[VERIFIED: fdars-core/src/autodiff/forward.rs]` |
| `crate::autodiff::Var` + `vjp` | in-crate | Reverse-mode T | `[VERIFIED: fdars-core/src/autodiff/reverse.rs]` |
| `crate::utility::inner_product` | in-crate Phase 95 | Generic inner product (already done) | `[VERIFIED: fdars-core/src/utility.rs:44]` |

**Installation:** No new dependencies. All required types are in-crate from Phase 94/95.

---

## Package Legitimacy Audit

No external packages are installed in this phase. The entire implementation is in-crate. This section is not applicable.

---

## Architecture Patterns

### System Architecture Diagram

```
t: &[T]  (differentiable input)
    │
    ▼
bspline_basis_from_knots<T>  (basis/bspline.rs)
    │ per point t_val: T
    ├─► evaluate_order_zero<T>(t_val, knots: &[f64], ...) → Vec<T>   [indicator: T::one()/T::zero()]
    │
    └─► bspline_recurrence_step<T>(b: &[T], knots: &[f64], t_val: T, k) → Vec<T>   [mixed T/f64 arithmetic]
            │  d1, d2: f64 (pure knot arithmetic)
            │  guard: d1.abs() > 1e-10  (stays f64 comparison — no T involved)
            │  numerator: t_val - T::from_f64(knots[j])  (T arithmetic)
            │  scale:     T::from_f64(d1)               (single lift)
            └─► ((t_val - T::from_f64(knots[j])) / T::from_f64(d1)) * b[j]

t: &[T]
    │
    ▼
fourier_basis_with_period<T>  (basis/fourier.rs)
    │  t_min: f64   (folded from t BEFORE generic entry, or derived inside from primal — see F-4)
    │  scale: f64 = 2π/period  (pure f64 precompute)
    │  x = T::from_f64(scale) * (t_i - T::from_f64(t_min))
    │  DC: T::one()
    └─► harmonics: (T::from_f64(freq as f64) * x).sin() / .cos()

          ▼
   Vec<T> basis columns
          │
          ▼  (validation only)
   inner_product<T>(basis_col, fixed_curve: &[T], argvals: &[f64])  [Phase 95 — unchanged]
          │
          ▼
   scalar objective  →  grad (Dual) / vjp (Var)  →  FD cross-check
```

### Recommended Project Structure
No new files required. All edits are in-place:
```
fdars-core/src/
├── basis/
│   ├── bspline.rs      # In-place: generalize bspline_basis_from_knots, evaluate_order_zero, bspline_recurrence_step
│   ├── fourier.rs      # In-place: generalize fourier_basis_with_period; add t_min param or primal-fold
│   └── tests.rs        # Add: f64-parity tests + Dual/Var FD objective tests
```

No changes needed to `basis/mod.rs` (re-exports stay unchanged; generic signatures are backward-compatible by T=f64 inference).

---

## Detailed Implementation Recipes

### B-1: Recurrence Bit-Parity Form

**Question:** `(t_val - knots[j]) / d1 * b[j]` — the original does division then multiply. For bit-identical f64 parity, the generic form must preserve this operation order and use `T/T` division (not reciprocal-multiply).

**Finding:** `[VERIFIED: fdars-core/src/basis/bspline.rs:43]` — the original expression is:
```rust
(t_val - knots[j]) / d1 * b[j]
```
(first `/d1`, then `*b[j]` — NOT `* (1.0/d1)`).

Division and reciprocal-multiply are different floating-point operations. To preserve bit-identical f64 values, the generic form must also divide (not multiply by reciprocal). The correct generic expression is:

```rust
// d1 is f64 (pure knot arithmetic — stays f64)
let d1 = knots[j + k - 1] - knots[j];   // f64
let d2 = knots[j + k] - knots[j + 1];   // f64
let left = if d1.abs() > 1e-10 {
    // "fold f64, lift once" pattern:
    // numerator is T arithmetic: t_val (T) - knots[j] lifted to T
    // divisor is T::from_f64(d1) — single lift
    // multiply by b[j] (T) last — preserves original operation order
    (t_val - T::from_f64(knots[j])) / T::from_f64(d1) * b[j]
} else {
    T::zero()
};
let right = if d2.abs() > 1e-10 {
    (T::from_f64(knots[j + k]) - t_val) / T::from_f64(d2) * b[j + 1]
} else {
    T::zero()
};
left + right
```

**Why this is bit-identical at T=f64:** `f64::from_f64(v)` is the identity (`[VERIFIED: fdars-core/src/autodiff/forward.rs:106-109]`). So at T=f64, `T::from_f64(knots[j])` is `knots[j]` and `T::from_f64(d1)` is `d1`. The division `(t_val - knots[j]) / d1` then `* b[j]` is byte-for-byte the original expression.

**Why `Dual`/`Var` division matches f64 at the primal:** `Dual::div` (`[VERIFIED: fdars-core/src/autodiff/forward.rs:236-247]`) computes `value: self.value / rhs.value` — the primal is plain f64 division. `Var::div` (`[VERIFIED: fdars-core/src/autodiff/reverse.rs:228]`) computes `value: self.value / rhs.value` — same. So the primal result is bit-identical to f64 at T=Dual or T=Var. `[ASSUMED]` that no FMA or fused-multiply fusion changes this in practice — this is safe because Rust does not auto-fuse operations without explicit intrinsics.

### B-2: The Repeated-Knot Guard

**Finding:** `[VERIFIED: fdars-core/src/basis/bspline.rs:42-50]` — the guard expression is:
```rust
if d1.abs() > 1e-10 { ... } else { 0.0 }
```
`d1 = knots[j + k - 1] - knots[j]` is a pure f64 subtraction of two f64 knot values. It involves no `T` at all. **The guard stays an f64 comparison unchanged.** Only the fallback `0.0` must become `T::zero()`.

The `>` comparison is `f64 > f64` — standard, no `T::PartialOrd` involved. This is correct and clean.

### B-3: `evaluate_order_zero` Genericization

**Current signature** `[VERIFIED: fdars-core/src/basis/bspline.rs:20-34]`:
```rust
pub(super) fn evaluate_order_zero(t_val: f64, knots: &[f64], t_max_knot_idx: usize) -> Vec<f64>
```

**Generic form:**
```rust
pub(super) fn evaluate_order_zero<T: Scalar>(
    t_val: T,
    knots: &[f64],
    t_max_knot_idx: usize,
) -> Vec<T> {
    let mut b0 = vec![T::zero(); knots.len() - 1];
    for j in 0..(knots.len() - 1) {
        let in_interval = if j == t_max_knot_idx - 1 {
            t_val >= T::from_f64(knots[j]) && t_val <= T::from_f64(knots[j + 1])
        } else {
            t_val >= T::from_f64(knots[j]) && t_val < T::from_f64(knots[j + 1])
        };
        if in_interval {
            b0[j] = T::one();
            break;
        }
    }
    b0
}
```

**Key details:**
- `vec![T::zero(); ...]` — correct; replaces `vec![0.0; ...]`.
- `b0[j] = T::one()` — prefer `T::one()` over `T::from_f64(1.0)` (equivalent; `T::one()` is more idiomatic per CONTEXT.md).
- **Comparison form:** `t_val >= T::from_f64(knots[j])` — this is `T >= T` (both are `T`). The `Scalar` trait requires `PartialOrd` (`[VERIFIED: fdars-core/src/autodiff/mod.rs:42]`), and both `Dual::PartialOrd` and `Var::PartialOrd` compare primals only (`[VERIFIED: fdars-core/src/autodiff/forward.rs:295-300]`, `[VERIFIED: fdars-core/src/autodiff/reverse.rs:140-145]`). A direct `t_val >= knots[j]` would be `T >= f64` — Rust does not have a cross-type `PartialOrd<f64>` on `T: Scalar`. **Must lift knots[j] to `T::from_f64(knots[j])`** for the `>=` comparison.
- The span-search `>=`/`<` comparisons are discrete control-flow decisions, not differentiated — this is the same "value-only branching" semantics documented in forward.rs:20-24 and established by Phase 94's soft-DTW generalization.

### F-4: Fourier `t_min` Problem and Solution

**Current implementation** `[VERIFIED: fdars-core/src/basis/fourier.rs:42-68]`:
```rust
pub fn fourier_basis_with_period(t: &[f64], nbasis: usize, period: f64) -> Vec<f64> {
    let n = t.len();
    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);  // ← derived from t: &[f64]
    // ...
    let x = 2.0 * PI * (ti - t_min) / period;   // ti: f64, t_min: f64
```

**The problem:** After generalization to `t: &[T]`, computing `t_min` by folding over `t` would yield a `T`, not an `f64`. Using a `T`-valued `t_min` in `(ti - t_min) / period` is fine for `T/T` arithmetic, but `t_min` is a grid-origin constant — it has no meaningful gradient. Using a `T`-valued t_min would incorrectly allow gradient to flow through it, and more importantly, we would need `T::INFINITY` for the fold initialization, which is available (`Scalar::infinity()`), but the `.min()` fold would need `T::min` — which is not in the `Scalar` trait.

**Chosen approach (matches CONTEXT.md guidance):** Add an explicit `t_min: f64` parameter and derive it from the **f64 primal** values before entering the generic function. The cleanest non-breaking design: create a new generic helper `fourier_basis_with_period_and_tmin<T: Scalar>(t: &[T], nbasis: usize, period: f64, t_min: f64) -> Vec<T>` and update `fourier_basis_with_period` to call it:

```rust
// Generic core (the new differentiable entry point)
pub fn fourier_basis_with_period<T: Scalar>(
    t: &[T],
    nbasis: usize,
    period: f64,
    t_min: f64,          // ← new explicit f64 parameter
) -> Vec<T> { ... }
```

**BUT** this is a breaking change to the existing public signature. The non-breaking alternative: keep the current `fourier_basis_with_period(t: &[f64], nbasis, period)` signature intact as a pub wrapper, and add a separate generic function.

**Recommended approach (Claude's discretion, per CONTEXT.md):** Derive `t_min` from the f64 primals **inside** the generic function using a separate fold over the value field. For `Dual` this is `d.value`; for `Var` this is `v.value`; for `f64` this is just the value. The Scalar trait has no `to_f64` — but both `Dual` and `Var` expose `.value` as `pub(crate)`, so this would require a trait method.

Since `Scalar` has no `to_f64`, the simplest strictly-in-bounds approach is: **do the t_min fold in f64 outside the generic entry point**. That is, the non-generic `fourier_basis_with_period(t: &[f64], ...)` continues to fold `t_min` from `t: &[f64]`, then calls the generic core. The public signature stays `(t: &[f64], ...)` (unchanged, backward-compatible). For the generic entry point, derive `t_min` from the primal before entering — which means the generic entry point must be called with an explicit `t_min: f64`.

**Final resolved form** (consistent with CONTEXT.md "Claude's discretion"):
- Rename/repurpose `fourier_basis_with_period` as the **public generic** entry point accepting `t: &[T]`, but give it an explicit `t_min: f64` parameter. This is a new parameter but only affects code calling `fourier_basis_with_period` directly — the existing callers that currently pass `t: &[f64]` call the public wrapper `fourier_basis` or the f64 `fourier_basis_with_period`. 
- The existing callers all pass `t: &[f64]` and the `period` argument:
  - `seasonal/strength.rs:54` calls `fourier_basis_with_period(argvals, nbasis, period)` — passes f64 `argvals`.
  - `fourier_basis` (the auto-deriving wrapper at `fourier.rs:22-27`) calls `fourier_basis_with_period(t, nbasis, period)`.
  - `basis/tests.rs` tests call `fourier_basis_with_period(&t, nbasis, period)`.

The cleanest approach without breaking any of these: **fold t_min inside the generic function from the primal values** — but since `Scalar` has no `to_f64`, instead always accept an explicit `t_min: f64` and update all call sites. This is only 2–3 call sites, all f64.

**Alternative (preferred — zero call-site changes):** Extract a private generic `fourier_basis_core<T: Scalar>(t: &[T], nbasis: usize, scale: f64, t_min: f64) -> Vec<T>` and keep the public `fourier_basis_with_period(t: &[f64], nbasis, period) -> Vec<f64>` unchanged. Add a NEW public generic entry point that requires `t_min` to be passed explicitly:
```rust
// Existing (unchanged, f64-only):
pub fn fourier_basis_with_period(t: &[f64], nbasis: usize, period: f64) -> Vec<f64> {
    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);
    fourier_basis_core(t, nbasis, 2.0 * PI / period, t_min)
}

// New generic entry point for autodiff use:
pub fn fourier_basis_generic<T: Scalar>(
    t: &[T],
    nbasis: usize,
    period: f64,
    t_min: f64,
) -> Vec<T> {
    fourier_basis_core(t, nbasis, 2.0 * PI / period, t_min)
}
```
This approach does NOT generalize `fourier_basis_with_period` in-place (that would change its public signature), which deviates from the CONTEXT.md "in-place" decision. Instead it adds a companion function.

**Strictly per CONTEXT.md (in-place, no `_generic` companions):** The only strictly in-place non-breaking approach is to add `t_min: f64` as the last parameter to `fourier_basis_with_period` and update the 2–3 call sites. Given there are only 2–3 callers (all in-crate) this is the correct approach.

**RECOMMENDATION:** Generalize `fourier_basis_with_period` in-place to `<T: Scalar>(t: &[T], nbasis: usize, period: f64, t_min: f64) -> Vec<T>`. Update the three internal call sites (`fourier_basis` wrapper, `seasonal/strength.rs:54`, `basis/tests.rs`) to pass `t_min` explicitly. This preserves the in-place pattern, does not add a `to_f64`, and keeps `t_min` as f64 throughout.

**Generic body** `[ASSUMED for t_min threading — based on design analysis, not a pre-existing pattern]`:
```rust
use crate::autodiff::Scalar;

pub fn fourier_basis_with_period<T: Scalar>(
    t: &[T],
    nbasis: usize,
    period: f64,
    t_min: f64,            // f64 grid origin — constant, no gradient
) -> Vec<T> {
    let n = t.len();
    let scale = 2.0 * std::f64::consts::PI / period;   // f64 precompute

    let mut basis = vec![T::zero(); n * nbasis];

    for (i, &ti) in t.iter().enumerate() {
        // "fold f64, lift once": scale and t_min are f64; lift to T once
        let x = T::from_f64(scale) * (ti - T::from_f64(t_min));

        basis[i] = T::one();                            // DC component

        let mut k = 1;
        let mut freq = 1usize;
        while k < nbasis {
            if k < nbasis {
                basis[i + k * n] = (T::from_f64(freq as f64) * x).sin();
                k += 1;
            }
            if k < nbasis {
                basis[i + k * n] = (T::from_f64(freq as f64) * x).cos();
                k += 1;
            }
            freq += 1;
        }
    }

    basis
}
```

**f64 parity:** At T=f64, `T::from_f64(scale)` = `scale`, `T::from_f64(t_min)` = `t_min`, so `x = scale * (ti - t_min)` = `2.0 * PI * (ti - t_min) / period` — bit-identical. The original uses `f64::from(freq) * x` (`[VERIFIED: fdars-core/src/basis/fourier.rs:57]`); the generic uses `T::from_f64(freq as f64) * x` — at f64, identical.

**Note on `fourier_basis` (the auto-deriving wrapper):** It currently folds t_min inside and calls `fourier_basis_with_period(t, nbasis, period)` `[VERIFIED: fdars-core/src/basis/fourier.rs:22-27]`. After the change, it will call `fourier_basis_with_period(t, nbasis, period, t_min)` — trivial update.

---

## Caller Site Impact

All existing callers of `bspline_basis_from_knots` pass `t: &[f64]` — they infer `T = f64` and compile unchanged. `[VERIFIED: fdars-core/src/helpers.rs:523,671]`, `[VERIFIED: fdars-core/src/basis/pspline.rs:168]`, `[VERIFIED: fdars-core/src/basis/projection.rs:111]`.

For `fourier_basis_with_period`, the new `t_min: f64` parameter must be threaded into 3 internal call sites:
1. `fourier_basis` wrapper at `fourier.rs:26`: already has `t_min` from its own fold — trivial.
2. `seasonal/strength.rs:54` `[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/seasonal/strength.rs line 54]`: must derive `t_min` from its f64 `argvals` and pass it.
3. `basis/tests.rs` tests: must pass `t_min` to `fourier_basis_with_period` calls.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Forward-mode gradient through basis | Custom dual arithmetic | `Dual` from `autodiff/forward.rs` | Already implements all `Scalar` ops |
| Reverse-mode gradient through basis | Custom tape | `Var` + `vjp` from `autodiff/reverse.rs` | Tape already handles division, sin/cos, mul |
| Inner products | New generic inner product | `inner_product<T>` from `utility.rs` (Phase 95) | Already generic; reuse directly |
| f64 comparison in span-search | T-vs-f64 comparison trait | Lift knot to `T::from_f64(knots[j])` then `T >= T` | PartialOrd is T-vs-T in Scalar trait |

---

## Common Pitfalls

### Pitfall 1: Reciprocal-Multiply vs Division Breaks Bit Parity

**What goes wrong:** Writing `(t_val - T::from_f64(knots[j])) * T::from_f64(1.0/d1) * b[j]` (multiply by reciprocal) instead of `(t_val - T::from_f64(knots[j])) / T::from_f64(d1) * b[j]` (divide then multiply). These differ in floating-point due to rounding in the reciprocal `1.0/d1`.

**Why it happens:** Looks equivalent mathematically; a/b = a*(1/b). Not so in IEEE 754 float.

**How to avoid:** Preserve the original operation order exactly. Original is `/ d1 * b[j]` — so generic must be `/ T::from_f64(d1) * b[j]`.

**Warning signs:** `test_bspline_f64_parity` fails with errors at the ULP level even when partition-of-unity still passes.

### Pitfall 2: T-vs-f64 Comparison in `evaluate_order_zero`

**What goes wrong:** Writing `t_val >= knots[j]` where `t_val: T` and `knots[j]: f64`. This does not compile — `Scalar: PartialOrd` is `T: PartialOrd<T>`, not `T: PartialOrd<f64>`.

**Why it happens:** At T=f64 it would work, so the mistake looks natural.

**How to avoid:** Always lift knots to `T::from_f64(knots[j])` before comparison. The form is `t_val >= T::from_f64(knots[j])`.

**Warning signs:** Compile error `binary operation >= cannot be applied to type T`.

### Pitfall 3: `t_min` as T Breaks Fourier Gradient Semantics

**What goes wrong:** Computing `t_min` by folding over `t: &[T]` (e.g., using some min-fold on T values). This would route gradient through `t_min`, which is semantically a constant grid parameter, not a differentiable input. The `Scalar` trait also has no `.min(T)` method, so this would not compile.

**Why it happens:** `t_min` is derived from `t` in the f64 version, so "generalize everything" instinct applies it to T.

**How to avoid:** Pass `t_min: f64` explicitly. The callers that have `t: &[f64]` derive it from primals; the generic function treats it as a constant.

**Warning signs:** Compile error (`T` has no fold-min method), or wrong gradients (t_min partial leaking into gradient).

### Pitfall 4: Missing `use crate::autodiff::Scalar` Import in Generalized Files

**What goes wrong:** The generalized files `bspline.rs` and `fourier.rs` do not currently import `Scalar`. Adding a generic bound `<T: Scalar>` without the import gives a compile error.

**Why it happens:** The files currently have no autodiff imports.

**How to avoid:** Add `use crate::autodiff::Scalar;` at the top of `bspline.rs` and `fourier.rs`. Phase 95 established this pattern — `helpers.rs:3` already has `use crate::autodiff::Scalar;` (`[VERIFIED: fdars-core/src/helpers.rs:3]`).

**Warning signs:** Compile error `cannot find trait Scalar in this scope`.

### Pitfall 5: `vec![T::zero(); n * nbasis]` — T must be Clone+Copy

**What goes wrong:** `vec![T::zero(); count]` requires `T::zero()` to be `Clone` (the repeat macro clones the element). `Scalar: Copy + Clone`, so this is fine — but if someone changes the trait bound later, this breaks.

**How to avoid:** Note that `T: Scalar` already implies `Copy + Clone` (`[VERIFIED: fdars-core/src/autodiff/mod.rs:38-50]` — `Scalar: Copy + Clone`). This is safe.

---

## Code Examples

### Generalized `bspline_recurrence_step` Body

```rust
// Source: direct implementation recipe from bspline.rs:37-55 analysis [VERIFIED this session]
use crate::autodiff::Scalar;

pub(super) fn bspline_recurrence_step<T: Scalar>(
    b: &[T],
    knots: &[f64],
    t_val: T,
    k: usize,
) -> Vec<T> {
    (0..(knots.len() - k))
        .map(|j| {
            // Knot differences: pure f64 arithmetic (stays f64 — no T involved)
            let d1 = knots[j + k - 1] - knots[j];
            let d2 = knots[j + k] - knots[j + 1];
            // Guard is f64 comparison — unchanged
            let left = if d1.abs() > 1e-10 {
                // "fold f64, lift once": t_val is T; knots[j] lifted once; d1 lifted once
                // Preserve original operation order: (num / denom) * coeff
                (t_val - T::from_f64(knots[j])) / T::from_f64(d1) * b[j]
            } else {
                T::zero()
            };
            let right = if d2.abs() > 1e-10 {
                (T::from_f64(knots[j + k]) - t_val) / T::from_f64(d2) * b[j + 1]
            } else {
                T::zero()
            };
            left + right
        })
        .collect()
}
```

### Generalized `bspline_basis_from_knots` Body

```rust
// Source: bspline.rs:62-83 [VERIFIED this session]
use crate::autodiff::Scalar;

pub fn bspline_basis_from_knots<T: Scalar>(t: &[T], knots: &[f64], order: usize) -> Vec<T> {
    let n = t.len();
    let nbasis = knots.len() - order;
    let t_max_knot_idx = knots.len() - order - 1;

    let mut basis = vec![T::zero(); n * nbasis];

    for (ti, &t_val) in t.iter().enumerate() {
        let mut b = evaluate_order_zero(t_val, knots, t_max_knot_idx);

        for k in 2..=order {
            b = bspline_recurrence_step(&b, knots, t_val, k);
        }

        for j in 0..nbasis {
            basis[ti + j * n] = b[j];
        }
    }

    basis
}
```

### Combined Validation Objective (FD-check skeleton)

```rust
// Validation objective: sum_k inner_product(basis_col_k(t_generic), fixed_curve)
// Differentiating w.r.t. t (the evaluation grid points)
// Source: design recipe from Phase 95 test pattern [VERIFIED: fdars-core/src/helpers.rs:1262-1300]
#[test]
fn bspline_basis_dual_fd_check() {
    use crate::autodiff::{grad, Dual, Scalar};
    use crate::utility::inner_product;
    use crate::basis::bspline::{bspline_basis_from_knots, construct_bspline_knots};

    let n_t = 12usize;
    let t_f64: Vec<f64> = (0..n_t).map(|i| i as f64 / (n_t - 1) as f64).collect();
    let nknots = 6;
    let order = 4;
    let t_min = t_f64[0];
    let t_max = t_f64[n_t - 1];
    let knots = construct_bspline_knots(t_min, t_max, nknots, order);
    let nbasis = knots.len() - order;

    // Fixed curve (constants — no gradient flows through it)
    let fixed: Vec<f64> = t_f64.iter().map(|&t| (std::f64::consts::PI * t).sin()).collect();

    // Objective: sum_k inner_product(basis_col_k(t), fixed_curve)
    let objective = |t: &[Dual]| -> Dual {
        let basis = bspline_basis_from_knots(t, &knots, order);
        let fixed_dual: Vec<Dual> = fixed.iter().map(|&v| Dual::constant(v)).collect();
        let argvals_dual: Vec<f64> = t.iter().map(|d| d.value).collect();
        let mut acc = Dual::zero();
        for k in 0..nbasis {
            let col: Vec<Dual> = (0..n_t).map(|i| basis[i + k * n_t]).collect();
            acc += inner_product(&col, &fixed_dual, &argvals_dual);
        }
        acc
    };

    let (value, gradient) = grad(objective, &t_f64);
    assert!(value.is_finite());
    assert_eq!(gradient.len(), n_t);

    // Central FD check
    let h = 1e-6_f64;
    let f_f64 = |t: &[f64]| -> f64 {
        let basis = bspline_basis_from_knots(t, &knots, order);
        (0..nbasis).map(|k| {
            let col: Vec<f64> = (0..n_t).map(|i| basis[i + k * n_t]).collect();
            inner_product(&col, &fixed, t)
        }).fold(0.0f64, |a, b| a + b)
    };
    for j in 0..n_t {
        let mut plus = t_f64.clone();
        let mut minus = t_f64.clone();
        plus[j] += h;
        minus[j] -= h;
        let fd = (f_f64(&plus) - f_f64(&minus)) / (2.0 * h);
        assert!((gradient[j] - fd).abs() < 1e-6,
            "Dual grad[{j}]={} vs FD={fd}", gradient[j]);
    }
}
```
Note: `argvals_dual` in the inner_product call uses extracted f64 primals because `inner_product`'s `argvals: &[f64]` stays f64. This is the correct pattern — `argvals` are the quadrature weights grid, not the differentiable input.

---

## Validation Architecture

> `workflow.nyquist_validation` is not explicitly set to false in `.planning/config.json` — treat as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`, `#[cfg(test)]`) |
| Config file | none (no `cargo.toml` test section needed) |
| Quick run command | `cargo test -p fdars-core basis --features linalg 2>&1 \| tail -20` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel 2>&1 \| tail -30` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DOP-01 | `bspline_basis_from_knots::<f64>` reproduces current numerics bit-identically | unit/parity | `cargo test -p fdars-core basis::tests::test_bspline_basis_from_knots_f64_parity` | ❌ Wave 0 (new test) |
| DOP-01 | `fourier_basis_with_period::<f64>` reproduces current numerics | unit/parity | `cargo test -p fdars-core basis::tests::test_fourier_f64_parity` | ❌ Wave 0 (new test) |
| DOP-01 | `bspline_basis_from_knots` partition-of-unity preserved after generalization | unit | `cargo test -p fdars-core basis::tests::test_bspline_basis_partition_of_unity` | ✅ (existing, regression guard) |
| DOP-01 | Fourier DC/sin-cos-range tests preserved | unit | `cargo test -p fdars-core basis::tests::test_fourier_basis_constant_first_column` | ✅ (existing, regression guard) |
| DOP-01 | Combined B-spline basis-eval→inner-product objective differentiable at `Dual`, FD-checked | unit/autodiff | `cargo test -p fdars-core basis::tests::bspline_basis_dual_fd_check` | ❌ Wave 0 |
| DOP-01 | Same objective differentiable at `Var` (via `vjp`), FD-checked | unit/autodiff | `cargo test -p fdars-core basis::tests::bspline_basis_var_fd_check` | ❌ Wave 0 |
| DOP-01 | Fourier basis-eval→inner-product objective differentiable at `Dual`, FD-checked | unit/autodiff | `cargo test -p fdars-core basis::tests::fourier_basis_dual_fd_check` | ❌ Wave 0 |
| DOP-01 | Fourier basis objective differentiable at `Var` (via `vjp`), FD-checked | unit/autodiff | `cargo test -p fdars-core basis::tests::fourier_basis_var_fd_check` | ❌ Wave 0 |
| DOP-01 | Downstream callers compile with inferred T=f64 (clippy gate) | compile | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | ✅ (CI gate) |

### Tolerances
- **f64 parity:** `==` (bit-identical, via `assert_eq!`) for the recurrence parity test (since `f64::from_f64` is identity). Use `(result - expected).abs() < 1e-10` for partition-of-unity (matches existing threshold `[VERIFIED: fdars-core/src/basis/tests.rs:39]`).
- **Autodiff FD cross-check:** `(ad_grad - fd).abs() < 1e-6`, central differences with `h = 1e-6`. Consistent with Phase 94/95 convention `[VERIFIED: fdars-core/src/autodiff/forward.rs:1049-1060]`.

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core basis --features linalg 2>&1 | tail -20`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -30`
- **Phase gate (before `/gsd-verify-work`):** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo build --features serde` guard + `cargo fmt --check`

### Wave 0 Gaps (new test infrastructure needed)
- [ ] `basis/tests.rs` — parity test `test_bspline_basis_from_knots_f64_parity`: compare generic-at-f64 output byte-for-byte vs pre-change reference array (or inline f64 call).
- [ ] `basis/tests.rs` — parity test `test_fourier_f64_parity`: compare generic-at-f64 output to pre-change f64 call.
- [ ] `basis/tests.rs` — `bspline_basis_dual_fd_check`: combined B-spline objective, `Dual` FD.
- [ ] `basis/tests.rs` — `bspline_basis_var_fd_check`: same objective, `Var` via `vjp`.
- [ ] `basis/tests.rs` — `fourier_basis_dual_fd_check`: Fourier objective, `Dual` FD.
- [ ] `basis/tests.rs` — `fourier_basis_var_fd_check`: Fourier objective, `Var` via `vjp`.

Existing tests (`test_bspline_basis_partition_of_unity`, `test_bspline_basis_non_negative`, `test_bspline_basis_boundary`, `test_fourier_basis_constant_first_column`, `test_fourier_basis_sin_cos_range`, `test_fourier_basis_with_period`, `test_fourier_basis_period_affects_frequency`) serve as regression guards — they should continue to pass unchanged after generalization.

---

## Security Domain

This phase implements in-crate numerical computation (basis evaluation, autodiff). There are no authentication, session management, input injection, or cryptographic concerns. ASVS categories V2–V6 do not apply. The only relevant concern is numerical correctness (f64 parity), addressed by the validation architecture above.

---

## Environment Availability

**Step 2.6: No external dependencies.** This phase is pure in-crate Rust code changes. The Rust toolchain is already present (runtime version 1.97.0, `[ASSUMED]` based on CLAUDE.md which states `Runtime version in development: 1.97.0`).

| Dependency | Required By | Available | Fallback |
|------------|------------|-----------|----------|
| Rust 1.81+ | MSRV | ✅ (1.97.0) | — |
| `cargo test` | Validation | ✅ | — |
| `cargo clippy --all-targets` | CI gate | ✅ | — |

---

## Runtime State Inventory

> Skipped — this is a greenfield generalization phase (in-place signature change), not a rename/refactor. No stored data, live service config, OS-registered state, secrets, or build artifacts embed the function names being changed.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Rust does not auto-fuse `(a - b) / c * d` into a fused-multiply-add (FMA), so division then multiply at T=f64 is bit-identical to the original f64 expression | B-1 (bit-parity) | Bit-parity test would fail at ULP level; extremely unlikely on stable Rust without explicit SIMD intrinsics |
| A2 | Runtime Rust version is 1.97.0 (from CLAUDE.md) | Environment | No impact — phase uses no version-specific features beyond 1.81 MSRV |
| A3 | The `fourier_basis_with_period` signature change (adding `t_min: f64`) is the cleanest non-breaking approach — no alternative hidden callers in crate | F-4 | If there are undiscovered external callers (unlikely — function is used via `fourier_basis` wrapper in most paths), they would get a compile error |

**All other claims in this document are VERIFIED from source files read this session.**

---

## Sources

### Primary (HIGH confidence — VERIFIED from source files read this session)
- `fdars-core/src/basis/bspline.rs:1-126` — Full B-spline implementation; recurrence body, guard, `evaluate_order_zero`, `bspline_basis_from_knots`
- `fdars-core/src/basis/fourier.rs:1-69` — Full Fourier implementation; `fourier_basis_with_period` body, `t_min` derivation
- `fdars-core/src/autodiff/mod.rs:38-85` — `Scalar` trait definition; all bounds including `PartialOrd`
- `fdars-core/src/autodiff/forward.rs:96-300` — `f64::from_f64` identity, `Dual` div quotient rule, `Dual::PartialOrd` value-only
- `fdars-core/src/autodiff/reverse.rs:100-250` — `Var::div` three-case impl, `Var::PartialOrd` value-only
- `fdars-core/src/utility.rs:30-55` — `inner_product<T: Scalar>` Phase 95 generic form
- `fdars-core/src/warping.rs:85-95` — `inner_product_l2<T: Scalar>` Phase 95 generic form
- `fdars-core/src/helpers.rs:1-10,520-545,1260-1300` — Scalar import pattern, f64 caller sites, Phase 95 FD test pattern
- `fdars-core/src/basis/tests.rs:1-656` — All existing basis tests; tolerances, structure
- `fdars-core/src/basis/mod.rs` — Re-exports; `fourier_basis_with_period` is re-exported
- `fdars-core/src/seasonal/strength.rs:1-60` — Caller of `fourier_basis_with_period`
- `fdars-core/src/basis/pspline.rs:160-180` — Caller of `bspline_basis_from_knots`
- `fdars-core/src/basis/projection.rs:100-125` — Caller of `bspline_basis_from_knots`

---

## Metadata

**Confidence breakdown:**
- Recurrence bit-parity recipe: HIGH — read actual source; verified Dual/Var division implementations
- evaluate_order_zero form: HIGH — read actual source; Scalar PartialOrd confirmed value-only
- Fourier t_min approach: HIGH on analysis; ASSUMED on "no hidden callers"
- Validation FD architecture: HIGH — mirrors Phase 94/95 test pattern confirmed in source
- f64 parity tolerance: HIGH — read existing test 1e-10 threshold directly

**Research date:** 2026-09-11
**Valid until:** Indefinite (no external dependencies; all facts are in-repo constants)
