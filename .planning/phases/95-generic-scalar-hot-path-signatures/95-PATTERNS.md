# Phase 95: Generic Scalar Hot-Path Signatures — Pattern Map

**Mapped:** 2026-09-11
**Files analyzed:** 4 modified functions across 3 source files + 1 read-only integration point
**Analogs found:** 4 / 4

---

## File Classification

| Modified Function / File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------------|------|-----------|----------------|---------------|
| `l2_distance` — `helpers.rs:56` | utility kernel | transform | `soft_dtw_distance_inner` (`metric/soft_dtw.rs:102`) + `project_scores_generic` (`regression.rs:232`) | exact — same `Σ T·T::from_f64(f64_weight)` + `Scalar::sqrt` accumulation |
| `trapz` — `helpers.rs:253` | utility kernel | transform | `soft_dtw_distance_inner` (`metric/soft_dtw.rs:102`) for accumulator loop; `project_scores_generic` for f64-fold-before-lift rule | exact — same fold-f64-first-then-`from_f64` pattern |
| `inner_product` — `utility.rs:34` | utility kernel | transform | `project_scores_generic` (`regression.rs:232`) | exact — same `Σ T * T * T::from_f64(weight)` accumulator; body also requires `.sum()→loop` rewrite (see note) |
| `inner_product_l2` — `warping.rs:83` | utility kernel | transform | `project_scores_generic` (`regression.rs:232`) + calls generic `trapz` once generalized | exact — mechanical: `Vec<T>` + delegated `trapz::<T>` |

---

## Pattern Assignments

### `l2_distance<T: Scalar>` — `fdars-core/src/helpers.rs:56`

**Analog 1:** `fdars-core/src/metric/soft_dtw.rs:102–126` — `soft_dtw_distance_inner<S: Scalar>`
Shows `S::zero()`, `S::infinity()`, and `cost = d * d` where `d: S` — the same `T * T → T` squared-diff pattern.

**Analog 2:** `fdars-core/src/regression.rs:232–251` — `project_scores_generic<S: Scalar>`
Shows `S::from_f64(rotation[(j,k)] * weights[j])` — the f64-sub-expression-first-then-one-lift rule.

**Current body** (`helpers.rs:56–63`, verbatim):
```rust
pub fn l2_distance(curve1: &[f64], curve2: &[f64], weights: &[f64]) -> f64 {
    let mut dist_sq = 0.0;
    for i in 0..curve1.len() {
        let diff = curve1[i] - curve2[i];
        dist_sq += diff * diff * weights[i];
    }
    dist_sq.sqrt()
}
```

**Target signature and body** (copy this shape):
```rust
pub fn l2_distance<T: Scalar = f64>(curve1: &[T], curve2: &[T], weights: &[f64]) -> T {
    let mut dist_sq = T::zero();
    for i in 0..curve1.len() {
        let diff = curve1[i] - curve2[i];
        dist_sq += diff * diff * T::from_f64(weights[i]);
    }
    dist_sq.sqrt()   // Scalar::sqrt — not f64::sqrt
}
```

Key mechanics:
- `diff * diff` — `T * T → T` via `Mul<Output=Self>` (in `Scalar` supertrait); no lift needed.
- `T::from_f64(weights[i])` — single lift of the f64 weight, then `T * T`.
- `dist_sq.sqrt()` — calls `Scalar::sqrt(self)` (`autodiff/mod.rs:63`), which is `f64::sqrt` at `T=f64`.
- No callers need changes: `l2_distance_matrix` (`distance.rs:70`) passes `data.row(i)` which returns `Vec<f64>`, so `T=f64` infers.

**Import to add** (at top of `helpers.rs`):
```rust
use crate::autodiff::Scalar;
```

---

### `trapz<T: Scalar>` — `fdars-core/src/helpers.rs:253`

**Analog:** `fdars-core/src/regression.rs:232–251` — `project_scores_generic<S: Scalar>` for the f64-fold-before-lift rule; `soft_dtw_distance_inner` for the explicit accumulator loop shape.

**Current body** (`helpers.rs:253–259`, verbatim):
```rust
pub fn trapz(y: &[f64], x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for k in 1..y.len() {
        sum += 0.5 * (y[k] + y[k - 1]) * (x[k] - x[k - 1]);
    }
    sum
}
```

**Target signature and body:**
```rust
pub fn trapz<T: Scalar = f64>(y: &[T], x: &[f64]) -> T {
    let mut sum = T::zero();
    for k in 1..y.len() {
        let half_dx = T::from_f64(0.5 * (x[k] - x[k - 1]));  // fold f64 first, lift once
        sum += half_dx * (y[k] + y[k - 1]);                    // T * T = T
    }
    sum
}
```

Ordering invariant (from RESEARCH.md §Pitfall 3 and §f64/Generic Boundary Mechanics):
- Compute `0.5 * (x[k] - x[k-1])` entirely in f64 first, then call `T::from_f64(...)` once.
- This preserves the exact same floating-point accumulation order when `T=f64`, giving bit-identical parity.
- Do NOT write `T::from_f64(0.5) * T::from_f64(x[k] - x[k-1])` — two lifts instead of one (cleaner to use one).

`x` stays `&[f64]` (grid spacings are quadrature constants, not differentiable curve data).

---

### `inner_product<T: Scalar>` — `fdars-core/src/utility.rs:34`

**Analog:** `fdars-core/src/regression.rs:232–251` — `project_scores_generic<S: Scalar>` (exact match: same `S::zero()` accumulator + `S::from_f64(weights[j])` lift pattern).

**Current body** (`utility.rs:34–46`, verbatim):
```rust
pub fn inner_product(curve1: &[f64], curve2: &[f64], argvals: &[f64]) -> f64 {
    if curve1.len() != curve2.len() || curve1.len() != argvals.len() || curve1.is_empty() {
        return 0.0;
    }
    let weights = simpsons_weights(argvals);
    curve1
        .iter()
        .zip(curve2.iter())
        .zip(weights.iter())
        .map(|((&c1, &c2), &w)| c1 * c2 * w)
        .sum()
}
```

**CRITICAL NOTE — `.sum()` incompatibility:** The body ends in `.sum()` which calls `std::iter::Sum`. `T: Scalar` does NOT include `Sum`. The iterator chain must be replaced with an explicit accumulator loop. This is a deliberate body rewrite, not a mechanical translation.

**Target signature and body:**
```rust
pub fn inner_product<T: Scalar = f64>(curve1: &[T], curve2: &[T], argvals: &[f64]) -> T {
    if curve1.len() != curve2.len() || curve1.len() != argvals.len() || curve1.is_empty() {
        return T::zero();
    }
    let weights = simpsons_weights(argvals);  // Vec<f64> — unchanged
    let mut acc = T::zero();
    for i in 0..curve1.len() {
        acc += curve1[i] * curve2[i] * T::from_f64(weights[i]);
    }
    acc
}
```

The `project_scores_generic` analog (`regression.rs:241–248`) shows this exact pattern:
```rust
// regression.rs:241-248 (analog)
let mut sum = S::zero();
for j in 0..m {
    let centered = curve[j] - S::from_f64(mean[j]);
    let w_rot = S::from_f64(rotation[(j, k)] * weights[j]);
    sum += centered * w_rot;
}
```

`simpsons_weights` stays `f64` — it is a quadrature-constant helper, not curve data.

**Import to add** (at top of `utility.rs`):
```rust
use crate::autodiff::Scalar;
```

---

### `inner_product_l2<T: Scalar>` — `fdars-core/src/warping.rs:83`

**Analog:** `fdars-core/src/regression.rs:232–251` for the generic boundary rule; delegates to `trapz<T>` once that is generalized.

**Current body** (`warping.rs:83–86`, verbatim):
```rust
pub fn inner_product_l2(psi1: &[f64], psi2: &[f64], time: &[f64]) -> f64 {
    let prod: Vec<f64> = psi1.iter().zip(psi2.iter()).map(|(&a, &b)| a * b).collect();
    trapz(&prod, time)
}
```

**Target signature and body:**
```rust
pub fn inner_product_l2<T: Scalar = f64>(psi1: &[T], psi2: &[T], time: &[f64]) -> T {
    let prod: Vec<T> = psi1.iter().zip(psi2.iter()).map(|(&a, &b)| a * b).collect();
    trapz(&prod, time)
}
```

Mechanics:
- `a * b` where `a: T, b: T` — `T * T → T` via `Mul<Output=Self>`. No lift needed.
- `collect::<Vec<T>>()` — works because `T: Copy` (in `Scalar` bound).
- `trapz(&prod, time)` — once `trapz` is generalized, `T` infers from `&[T]` argument. No turbofish needed.
- Callers (`warping.rs:90, 96, 177`; `alignment/bayesian.rs`, `fpns.rs`, `tests.rs`) all pass `&[f64]`, so `T=f64` infers and they continue calling `.max(0.0)`, `.clamp(-1.0, 1.0)`, etc. on the `f64` return — unchanged.

**Import to add** (at top of `warping.rs`):
```rust
use crate::autodiff::Scalar;
```
Also ensure `trapz` is imported from `helpers` (check existing imports in `warping.rs` — it likely already imports `trapz`).

---

## Shared Patterns

### `Scalar` Trait — the bound for all four kernels

**Source:** `fdars-core/src/autodiff/mod.rs:38–85`

```rust
pub trait Scalar:
    Copy + Clone + Debug + PartialOrd
    + Add<Output = Self> + Sub<Output = Self>
    + Mul<Output = Self> + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign + SubAssign + MulAssign
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(v: f64) -> Self;   // identity for f64; tangent-0 for Dual; sentinel for Var
    fn infinity() -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn powf(self, p: f64) -> Self;
    fn abs(self) -> Self;
    fn signum(self) -> Self;
}
```

**Apply to:** All four modified functions. Add `use crate::autodiff::Scalar;` at the top of each file being modified.

**`f64::from_f64` is the identity** (`autodiff/forward.rs:106–108`):
```rust
fn from_f64(v: f64) -> Self {
    v
}
```
This is why bit-identical parity holds at `T=f64` — no transformation, no rounding.

### f64 / Generic Boundary Invariant

**Source:** `fdars-core/src/regression.rs:228–231` (doc comment) + body lines 244–245

```rust
// From project_scores_generic doc comment:
// The `rotation[(j,k)] * weights[j]` product is folded into a single `f64`
// before lifting via `S::from_f64`, so the analytic gradient is exactly that f64 constant.

// From body:
let w_rot = S::from_f64(rotation[(j, k)] * weights[j]);  // f64 * f64 = f64, then lift once
sum += centered * w_rot;
```

**Rule:** Curve values (`c1[i]`, `y[k]`, `psi1[i]`) → `T`. Everything else (weights, argvals, dx, gamma) → stays `f64`. Where `T` must multiply by an `f64` coefficient, compute the `f64` sub-expression first, then lift once via `T::from_f64(...)`.

**Apply to:** All four modified functions.

### Accumulator Loop Pattern (replaces iterator `.sum()`)

**Source:** `fdars-core/src/regression.rs:240–248`

```rust
let mut sum = S::zero();          // S::zero() not 0.0
for j in 0..m {
    // ... compute S-typed term ...
    sum += term;                   // AddAssign is in Scalar supertrait
}
```

**Apply to:** `inner_product` (mandatory — current body uses `.sum()` which is incompatible with `T: Scalar`). Also used in `l2_distance` and `trapz`.

---

## Read-Only Integration Point: `l2_distance_matrix`

**Source:** `fdars-core/src/distance.rs:67–71`

```rust
pub fn l2_distance_matrix(data: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let weights = simpsons_weights(argvals);
    let n = data.nrows();
    pairwise_distance_matrix(n, |i, j| l2_distance(&data.row(i), &data.row(j), &weights))
}
```

`data.row(i)` returns `Vec<f64>` — so the closure passes `&[f64]` to `l2_distance`, and `T=f64` infers. This call site is **unchanged** after generalization. Do not touch `distance.rs`.

Similarly, `warping.rs:90` chain:
```rust
inner_product_l2(psi, psi, time).max(0.0).sqrt()  // returns f64; .max() is f64 inherent
```
`T=f64` infers from `psi: &[f64]` arg — the `.max(0.0).sqrt()` chain stays valid on the `f64` return.

---

## No Analog Found

None — all four targets have direct analogs in the existing Phase 94 generic templates.

---

## Caller Audit: Grep Patterns for Executor

The executor must run these greps **before any source edit** to confirm zero function-pointer coercions or explicit `: f64` return-type annotations that would need turbofish after generalization.

### Pre-Edit Safety Greps

```bash
# 1. Zero fn-pointer coercions on target kernels
grep -rn "fn.*l2_distance\|fn.*inner_product\|fn.*trapz\b\|fn.*inner_product_l2" \
  fdars-core/src fdars-core/examples fdars-core/benches fdars-core/tests \
  --include="*.rs" | grep -v "pub fn\|fn test_\|//\|#\["

# 2. Zero explicit `: f64 = <kernel>(...)` annotations
grep -rn ": f64 = l2_distance\|: f64 = inner_product\|: f64 = trapz\|: f64 = inner_product_l2" \
  fdars-core/src fdars-core/examples fdars-core/tests --include="*.rs"

# 3. Baseline diff (should be empty before edits)
git diff --stat HEAD
```

### Call-Site Inventory Greps (for post-edit compile-gate scope awareness)

```bash
# All l2_distance call sites (~8+ in src, 1+ in examples)
grep -rln "l2_distance(" fdars-core/src fdars-core/examples fdars-core/benches fdars-core/tests --include="*.rs"

# All trapz call sites (~12+ in src, 1 in tests — validate_against_r.rs:3401 is LOCAL fn, not crate::trapz)
grep -rln "trapz(" fdars-core/src fdars-core/examples fdars-core/benches fdars-core/tests --include="*.rs"

# All inner_product call sites (2 confirmed: examples/02 + tests/validate_against_r)
grep -rln "inner_product(" fdars-core/src fdars-core/examples fdars-core/benches fdars-core/tests --include="*.rs"

# All inner_product_l2 call sites (~6: warping.rs x3 + alignment/{bayesian,fpns,tests})
grep -rln "inner_product_l2(" fdars-core/src fdars-core/examples fdars-core/benches fdars-core/tests --include="*.rs"
```

### Key Verified Call-Site Facts (from RESEARCH.md audit)

| Kernel | Call Sites | Data Type Passed | Risk |
|--------|-----------|-----------------|------|
| `l2_distance` | `distance.rs:70`, `clustering.rs` (8x), `alignment/` (9 files), `classification/kernel.rs`, `scalar_on_function/nonparametric.rs`, `metric/hshift.rs:36`, `examples/16_elastic_alignment/main.rs:67,70` | All `Vec<f64>` / `&[f64]` → `T=f64` infers | None |
| `trapz` | `density_fda.rs` (8+), `frechet/` (4 files), `alignment/` (4 files), `fts/acf.rs`, `warping.rs:258` | All `Vec<f64>` / `&[f64]` | `validate_against_r.rs:3401` defines LOCAL `fn trapz` — does NOT call crate `trapz` |
| `inner_product` | `examples/02_functional_operations/main.rs:159–160`, `tests/validate_against_r.rs:721` | `f64` | None |
| `inner_product_l2` | `warping.rs:90,96,177`, `alignment/{bayesian,fpns,tests}.rs` | `&[f64]` → chains `.max(0.0)`, `.clamp(-1.0,1.0)` on `f64` return | None — `f64` inherent methods still valid |

**Method-chaining call sites that look risky but are safe:**
- `inner_product_l2(...).max(0.0).sqrt()` (`warping.rs:90`) — `.max` and `.sqrt` are `f64` inherent; `T=f64` infers; no change needed.
- `inner_product_l2(...).clamp(-1.0, 1.0)` (`warping.rs:96, 177`; `alignment/tests.rs:1521`) — same reasoning.
- `l2_distance(...).powi(2)` (`clustering.rs:162`) — `f64` inherent; `T=f64` infers.

---

## Non-Churn Verification (Post-Edit)

After all edits, only these files should differ:

```bash
git diff --name-only HEAD
# Expected set:
#   fdars-core/src/helpers.rs      (l2_distance + trapz generalized + parity/Dual tests added)
#   fdars-core/src/utility.rs      (inner_product generalized + parity/Dual tests added)
#   fdars-core/src/warping.rs      (inner_product_l2 generalized + parity/Dual tests added)
# NOT expected: distance.rs, clustering.rs, alignment/*, warping.rs callers, examples/*, etc.
```

---

## Metadata

**Analog search scope:** `fdars-core/src/metric/`, `fdars-core/src/regression.rs`, `fdars-core/src/autodiff/`
**Files read:** 9 source files (all git-tracked, verified)
**Pattern extraction date:** 2026-09-11
