# Phase 96: Differentiable Basis Evaluation & Inner Products — Pattern Map

**Mapped:** 2026-09-11
**Files analyzed:** 2 modified source files + 1 test file
**Analogs found:** 3 / 3

---

## File Classification

| Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---------------|------|-----------|----------------|---------------|
| `fdars-core/src/basis/bspline.rs` | utility (numeric transform) | transform | `fdars-core/src/helpers.rs` (`l2_distance<T>`, `trapz<T>`) + `fdars-core/src/regression.rs` (`project_scores_generic`) | exact — same `T::zero()` accumulator + `T::from_f64` f64-lift + f64-guard pattern |
| `fdars-core/src/basis/fourier.rs` | utility (numeric transform) | transform | `fdars-core/src/helpers.rs` (`l2_distance<T>`) + `fdars-core/src/utility.rs` (`inner_product<T>`) | exact — same "fold f64, lift once" per-element pattern |
| `fdars-core/src/basis/tests.rs` | test | request-response | `fdars-core/src/helpers.rs` (tests module: `test_l2_distance_dual`, `test_l2_distance_var`) | exact — same Dual seed/extract + central-FD + vjp pattern |

---

## Pattern Assignments

### `fdars-core/src/basis/bspline.rs` (utility, transform — IN-PLACE generalization)

**Edit targets (current f64 code to replace):**

**`evaluate_order_zero` — current body** (`bspline.rs` lines 20–34):
```rust
pub(super) fn evaluate_order_zero(t_val: f64, knots: &[f64], t_max_knot_idx: usize) -> Vec<f64> {
    let mut b0 = vec![0.0; knots.len() - 1];
    for j in 0..(knots.len() - 1) {
        let in_interval = if j == t_max_knot_idx - 1 {
            t_val >= knots[j] && t_val <= knots[j + 1]
        } else {
            t_val >= knots[j] && t_val < knots[j + 1]
        };
        if in_interval {
            b0[j] = 1.0;
            break;
        }
    }
    b0
}
```

**`bspline_recurrence_step` — current body** (`bspline.rs` lines 37–55):
```rust
pub(super) fn bspline_recurrence_step(b: &[f64], knots: &[f64], t_val: f64, k: usize) -> Vec<f64> {
    (0..(knots.len() - k))
        .map(|j| {
            let d1 = knots[j + k - 1] - knots[j];
            let d2 = knots[j + k] - knots[j + 1];
            let left = if d1.abs() > 1e-10 {
                (t_val - knots[j]) / d1 * b[j]
            } else {
                0.0
            };
            let right = if d2.abs() > 1e-10 {
                (knots[j + k] - t_val) / d2 * b[j + 1]
            } else {
                0.0
            };
            left + right
        })
        .collect()
}
```

**`bspline_basis_from_knots` — current body** (`bspline.rs` lines 62–83):
```rust
pub fn bspline_basis_from_knots(t: &[f64], knots: &[f64], order: usize) -> Vec<f64> {
    let n = t.len();
    let nbasis = knots.len() - order;
    let t_max_knot_idx = knots.len() - order - 1;
    let mut basis = vec![0.0; n * nbasis];
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

**Analog — import pattern** (`fdars-core/src/helpers.rs` line 3):
```rust
use crate::autodiff::Scalar;
```
Copy this import verbatim to the top of `bspline.rs` (currently has no autodiff import).

**Analog — `T::zero()` accumulator + `T::from_f64` lift** (`fdars-core/src/helpers.rs` lines 66–73):
```rust
pub fn l2_distance<T: Scalar>(curve1: &[T], curve2: &[T], weights: &[f64]) -> T {
    let mut dist_sq = T::zero();
    for i in 0..curve1.len() {
        let diff = curve1[i] - curve2[i];
        dist_sq += diff * diff * T::from_f64(weights[i]);
    }
    dist_sq.sqrt()
}
```
Key pattern: `vec![T::zero(); n]` replaces `vec![0.0; n]`; f64 constants lifted with `T::from_f64(v)` exactly once per use; no `T::to_f64` anywhere.

**Analog — "fold f64, lift once" with mixed T/f64 arithmetic** (`fdars-core/src/regression.rs` lines 232–251):
```rust
pub fn project_scores_generic<S: Scalar>(
    curve: &[S],
    mean: &[f64],
    rotation: &FdMatrix,
    weights: &[f64],
    ncomp: usize,
) -> Vec<S> {
    let m = curve.len();
    let mut scores = vec![S::zero(); ncomp];
    for (k, score) in scores.iter_mut().enumerate() {
        let mut sum = S::zero();
        for j in 0..m {
            let centered = curve[j] - S::from_f64(mean[j]);
            let w_rot = S::from_f64(rotation[(j, k)] * weights[j]);
            sum += centered * w_rot;
        }
        *score = sum;
    }
    scores
}
```
Key: f64 constants (`mean[j]`, product `rotation[(j,k)] * weights[j]`) are computed/kept in f64 and lifted with a single `S::from_f64`. This is the exact boundary discipline for the recurrence's knot differences.

**Generic target forms to write:**

`evaluate_order_zero<T: Scalar>` — change points:
- Signature: `t_val: f64` → `t_val: T`; return `Vec<f64>` → `Vec<T>`
- `vec![0.0; ...]` → `vec![T::zero(); ...]`
- Comparisons: `t_val >= knots[j]` (was T vs f64 — won't compile) → `t_val >= T::from_f64(knots[j])` and `t_val < T::from_f64(knots[j + 1])` and `t_val <= T::from_f64(knots[j + 1])`
- `b0[j] = 1.0` → `b0[j] = T::one()`

`bspline_recurrence_step<T: Scalar>` — change points:
- Signature: `b: &[f64], t_val: f64` → `b: &[T], t_val: T`; return `Vec<f64>` → `Vec<T>`
- `d1` and `d2` stay `f64` (pure knot subtraction — no T involved; guard `d1.abs() > 1e-10` stays f64)
- `0.0` fallbacks → `T::zero()`
- Left arm: `(t_val - knots[j]) / d1 * b[j]` → `(t_val - T::from_f64(knots[j])) / T::from_f64(d1) * b[j]`
  — division THEN multiply (preserves operation order for bit-identical f64 parity; NOT reciprocal-multiply)
- Right arm: `(knots[j + k] - t_val) / d2 * b[j + 1]` → `(T::from_f64(knots[j + k]) - t_val) / T::from_f64(d2) * b[j + 1]`

`bspline_basis_from_knots<T: Scalar>` — change points:
- Signature: `t: &[f64]` → `t: &[T]`; return `Vec<f64>` → `Vec<T>`
- `vec![0.0; n * nbasis]` → `vec![T::zero(); n * nbasis]`
- Iteration and loop body unchanged structurally; all type changes flow from callee signatures

---

### `fdars-core/src/basis/fourier.rs` (utility, transform — IN-PLACE generalization)

**Edit target (current f64 code to replace):**

**`fourier_basis_with_period` — current body** (`fourier.rs` lines 42–69):
```rust
pub fn fourier_basis_with_period(t: &[f64], nbasis: usize, period: f64) -> Vec<f64> {
    let n = t.len();
    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);  // ← derived from t: &[f64]

    let mut basis = vec![0.0; n * nbasis];

    for (i, &ti) in t.iter().enumerate() {
        let x = 2.0 * PI * (ti - t_min) / period;

        basis[i] = 1.0;

        let mut k = 1;
        let mut freq = 1;
        while k < nbasis {
            if k < nbasis {
                basis[i + k * n] = (f64::from(freq) * x).sin();
                k += 1;
            }
            if k < nbasis {
                basis[i + k * n] = (f64::from(freq) * x).cos();
                k += 1;
            }
            freq += 1;
        }
    }

    basis
}
```

**`fourier_basis` wrapper — current body** (`fourier.rs` lines 22–27, STAYS f64, UPDATED to pass `t_min`):
```rust
pub fn fourier_basis(t: &[f64], nbasis: usize) -> Vec<f64> {
    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let period = t_max - t_min;
    fourier_basis_with_period(t, nbasis, period)  // ← after change: add t_min arg
}
```

**Analog — "fold f64, lift once" per-element** (`fdars-core/src/utility.rs` lines 44–55):
```rust
pub fn inner_product<T: Scalar>(curve1: &[T], curve2: &[T], argvals: &[f64]) -> T {
    // ...
    let weights = simpsons_weights(argvals); // Vec<f64> — unchanged
    let mut acc = T::zero();
    for i in 0..curve1.len() {
        acc += curve1[i] * curve2[i] * T::from_f64(weights[i]);
    }
    acc
}
```
Key: f64 constants precomputed, then lifted with a single `T::from_f64` per use inside the loop. The Fourier body follows the same discipline.

**Generic target form for `fourier_basis_with_period`:**

Signature change: `t: &[f64], nbasis: usize, period: f64` → `t: &[T], nbasis: usize, period: f64, t_min: f64` (new `t_min: f64` explicit param; `t` becomes `&[T]`)

Body change points:
- Remove `let t_min = t.iter().copied()...` (now passed as f64 param)
- `vec![0.0; n * nbasis]` → `vec![T::zero(); n * nbasis]`
- `let x = 2.0 * PI * (ti - t_min) / period` →
  ```rust
  let scale = 2.0 * PI / period;   // f64 precompute (only once, can hoist outside loop)
  let x = T::from_f64(scale) * (ti - T::from_f64(t_min));
  ```
- `basis[i] = 1.0` → `basis[i] = T::one()`
- `(f64::from(freq) * x).sin()` → `(T::from_f64(freq as f64) * x).sin()`
- `(f64::from(freq) * x).cos()` → `(T::from_f64(freq as f64) * x).cos()`

`fourier_basis` wrapper update: add `t_min` fold (already has `t_min` from its own fold) and pass it:
```rust
fourier_basis_with_period(t, nbasis, period, t_min)
```

**Import to add at top of `fourier.rs`:**
```rust
use crate::autodiff::Scalar;
```
(Currently absent; see pitfall 4 in RESEARCH.md.)

---

## Caller Site Impact — Audit List

### `bspline_basis_from_knots` callers (compile unchanged at inferred T=f64)

All pass `t: &[f64]` → `T = f64` inferred, no source changes needed:

| File | Line | Call |
|------|------|------|
| `fdars-core/src/helpers.rs` | 523 | `crate::basis::bspline::bspline_basis_from_knots(query_points, &knots, order)` |
| `fdars-core/src/helpers.rs` | 671 | `crate::basis::bspline::bspline_basis_from_knots(&effective, &knots, order)` |
| `fdars-core/src/basis/pspline.rs` | 168 | `bspline_basis_from_knots(new_argvals, &result.knots, result.order)` |
| `fdars-core/src/basis/projection.rs` | 111 (approx) | (uses `bspline_basis`, not `_from_knots` — see grep) |

Grep pattern to confirm all callers:
```
grep -rn "bspline_basis_from_knots" fdars-core/src/
```

### `fourier_basis_with_period` callers (new `t_min: f64` param — ALL must be updated)

The RESEARCH.md listed 3 callers; the actual codebase has **4** (one additional: `smooth_basis.rs` and `seasonal/period.rs`):

| File | Line | Current call | Update needed |
|------|------|-------------|---------------|
| `fdars-core/src/basis/fourier.rs` | 26 | `fourier_basis_with_period(t, nbasis, period)` | Add `t_min` — already has it from own fold |
| `fdars-core/src/seasonal/strength.rs` | 54 | `fourier_basis_with_period(argvals, nbasis, period)` | Derive `t_min` from f64 `argvals` and pass |
| `fdars-core/src/smooth_basis.rs` | 1042 | `fourier_basis_with_period(argvals, nbasis, *period)` | Derive `t_min` from f64 `argvals` and pass |
| `fdars-core/src/seasonal/period.rs` | 185 | `fourier_basis_with_period(argvals, nbasis, period)` | Derive `t_min` from f64 `argvals` and pass |
| `fdars-core/src/basis/tests.rs` | 113, 128–129 | `fourier_basis_with_period(&t, nbasis, period)` | Add `t_min` from `t[0]` or explicit fold |

Pattern for deriving `t_min` at each f64 call site (all callers already have f64 `argvals`/`t`):
```rust
let t_min = argvals.iter().copied().fold(f64::INFINITY, f64::min);
let basis = fourier_basis_with_period(argvals, nbasis, period, t_min);
```

Grep pattern to find all callers:
```
grep -rn "fourier_basis_with_period" fdars-core/src/
```

### `fourier_basis` callers (signature UNCHANGED — no updates)

`fourier_basis(t: &[f64], nbasis)` stays f64-only. Its internal call to `fourier_basis_with_period` requires updating (see above), but its own public signature is untouched:

| File | Line | Note |
|------|------|------|
| `fdars-core/src/basis/projection.rs` | 68 | `fourier_basis(argvals, nbasis)` — unchanged |
| `fdars-core/src/basis/auto_select.rs` | 93 | `fourier_basis(argvals, nbasis)` — unchanged |
| `fdars-core/src/basis/fourier_fit.rs` | 66 | `fourier_basis(argvals, nbasis)` — unchanged |
| `fdars-core/src/elastic_regression/scalar_on_shape.rs` | 116 | `fourier_basis(argvals, nbasis)` — unchanged |

---

## Pattern Assignments (Tests)

### `fdars-core/src/basis/tests.rs` (test additions)

**Analog — Dual forward-mode FD test** (`fdars-core/src/helpers.rs` lines 1224–1260):
```rust
#[test]
fn test_l2_distance_dual() {
    use crate::autodiff::Dual;
    let c1_f64 = vec![1.0_f64, 2.0, 3.0];
    let c2_f64 = vec![0.5_f64, 1.5, 2.5];
    let w = vec![0.25_f64, 0.5, 0.25];
    let h = 1e-5_f64;
    let idx = 1;
    let c1_dual: Vec<Dual> = c1_f64
        .iter()
        .enumerate()
        .map(|(i, &v)| if i == idx { Dual::seed(v) } else { Dual::constant(v) })
        .collect();
    let c2_dual: Vec<Dual> = c2_f64.iter().map(|&v| Dual::constant(v)).collect();
    let (_, tangent) = l2_distance(&c1_dual, &c2_dual, &w).extract();
    let mut c1_plus = c1_f64.clone();
    let mut c1_minus = c1_f64.clone();
    c1_plus[idx] += h;
    c1_minus[idx] -= h;
    let fd = (l2_distance::<f64>(&c1_plus, &c2_f64, &w)
        - l2_distance::<f64>(&c1_minus, &c2_f64, &w))
        / (2.0 * h);
    let tol = 1e-5 * fd.abs().max(1e-10);
    assert!((tangent - fd).abs() <= tol, ...);
}
```
For basis tests: adapt by iterating over ALL `t` coordinates (not just one `idx`), using the combined objective (basis columns → inner_product). Use `h = 1e-6` per Phase 95 convention (RESEARCH.md: 1e-6 for basis FD checks).

**Analog — Var reverse-mode FD test** (`fdars-core/src/helpers.rs` lines 1264–1300):
```rust
#[test]
fn test_l2_distance_var() {
    use crate::autodiff::{vjp, Var};
    let (_, grad) = vjp(
        |x: &[Var]| {
            let c2_lifted: Vec<Var> = c2_var
                .iter()
                .map(|&v| <Var as crate::autodiff::Scalar>::from_f64(v))
                .collect();
            l2_distance(x, &c2_lifted, &w)
        },
        &c1_f64,
    );
    for idx in 0..c1_f64.len() {
        // central FD per coordinate
        let fd = (f(&plus) - f(&minus)) / (2.0 * h);
        assert!((grad[idx] - fd).abs() <= tol, ...);
    }
}
```
For basis `Var` tests: `vjp(|t: &[Var]| { ... combined objective ... }, &t_f64)`.

**f64 parity tests:**

Pattern — compare generic-at-f64 vs pre-change reference:
```rust
#[test]
fn test_bspline_basis_from_knots_f64_parity() {
    // Run both the old f64 path and the generic path at T=f64
    // They should produce bit-identical results because f64::from_f64 is the identity
    let t: Vec<f64> = ...;
    let knots = construct_bspline_knots(t_min, t_max, nknots, order);
    let reference: Vec<f64> = bspline_basis_from_knots(&t, &knots, order);   // pre-change (save as literal)
    // After generalization, the call site is unchanged (T=f64 inferred):
    let generic: Vec<f64> = bspline_basis_from_knots(&t, &knots, order);
    assert_eq!(reference, generic);   // bit-identical: assert_eq!, not abs-diff
}
```
Use `assert_eq!` (not `< 1e-10`) for bit-identical parity; fall back to `< 1e-10` for partition-of-unity (matching existing threshold at `tests.rs` line 39).

**Existing regression guards (DO NOT MODIFY, verify still pass):**
- `test_bspline_basis_partition_of_unity` (tests.rs:28–45)
- `test_bspline_basis_non_negative` (tests.rs:48–55)
- `test_bspline_basis_boundary` (tests.rs:58–67)
- `test_fourier_basis_constant_first_column` (tests.rs:81–94)
- `test_fourier_basis_sin_cos_range` (tests.rs:97–106)
- `test_fourier_basis_with_period` (tests.rs:109–121) — needs `t_min` arg after signature change
- `test_fourier_basis_period_affects_frequency` (tests.rs:123–...) — same

---

## Shared Patterns

### `use crate::autodiff::Scalar` Import
**Source:** `fdars-core/src/helpers.rs` line 3  
**Apply to:** `basis/bspline.rs` (top), `basis/fourier.rs` (top, after existing `use std::f64::consts::PI`)  
```rust
use crate::autodiff::Scalar;
```

### `T::zero()` / `T::one()` / `T::from_f64` idioms
**Source:** `fdars-core/src/helpers.rs` lines 66–73 (`l2_distance`), `fdars-core/src/utility.rs` lines 44–55 (`inner_product`)  
**Apply to:** all three generalized B-spline functions, `fourier_basis_with_period`  
- `0.0` → `T::zero()` (accumulator init, fallback arms, `vec!` initializer)
- `1.0` → `T::one()` (DC term, indicator)
- f64 constants → `T::from_f64(constant)` at the single point of mixing with T arithmetic

### f64 Guard Stays f64
**Source:** `fdars-core/src/basis/bspline.rs` lines 40–41, 47–48 (verified current)  
**Apply to:** `bspline_recurrence_step`  
```rust
let d1 = knots[j + k - 1] - knots[j];   // f64 — stays f64
let d2 = knots[j + k] - knots[j + 1];   // f64 — stays f64
if d1.abs() > 1e-10 { ... }              // f64 comparison — stays f64
```
Guard is `f64 > f64`, unchanged. Only the fallback `0.0` becomes `T::zero()`.

### Span-Search Comparison Form
**Source:** RESEARCH.md finding B-3 (verified from `autodiff/forward.rs:295-300`, `autodiff/reverse.rs:140-145`)  
**Apply to:** `evaluate_order_zero`  
`T: Scalar` has `PartialOrd` as `T: PartialOrd<T>` — cross-type `T >= f64` does not compile. Always lift knot to T before comparison:
```rust
t_val >= T::from_f64(knots[j]) && t_val < T::from_f64(knots[j + 1])
```

### Division Before Multiply (Bit-Parity)
**Source:** RESEARCH.md finding B-1 (operation order verified from `bspline.rs:43`)  
**Apply to:** both arms of `bspline_recurrence_step`  
Original: `(t_val - knots[j]) / d1 * b[j]` — division first, then multiply.  
Generic must preserve order: `(t_val - T::from_f64(knots[j])) / T::from_f64(d1) * b[j]`  
NOT: `(t_val - T::from_f64(knots[j])) * T::from_f64(1.0 / d1) * b[j]` (reciprocal-multiply breaks ULP parity).

### `t_min` as Explicit f64 Parameter (Fourier)
**Source:** RESEARCH.md finding F-4 / fourier.rs:44 (verified: current derivation is `t.iter().copied().fold(f64::INFINITY, f64::min)`)  
**Apply to:** `fourier_basis_with_period` signature  
Remove the internal `t_min` fold; add `t_min: f64` as last parameter. Each f64 call site folds it from its `&[f64]` argument before calling. Generic callers pass the f64 primal they derive externally.

### FD Validation Tolerance
**Source:** `fdars-core/src/helpers.rs` lines 1230, 1270  
**Apply to:** all new autodiff tests in `basis/tests.rs`  
- Forward (Dual): `h = 1e-6`; tolerance `(ad_grad - fd).abs() < 1e-6`
- Reverse (Var / vjp): `h = 1e-6`; same tolerance

---

## No Analog Found

None. All generalization patterns have direct, verified analogs in the Phase 94/95 substrate already present in the codebase.

---

## Metadata

**Analog search scope:** `fdars-core/src/` (all source files)
**Files scanned:** `basis/bspline.rs`, `basis/fourier.rs`, `basis/tests.rs`, `helpers.rs`, `utility.rs`, `regression.rs`, `seasonal/strength.rs`, `smooth_basis.rs`, `seasonal/period.rs`, `autodiff/mod.rs` (via RESEARCH.md)
**Caller audit grep patterns:**
```bash
grep -rn "bspline_basis_from_knots" fdars-core/src/
grep -rn "fourier_basis_with_period" fdars-core/src/
grep -rn "fourier_basis(" fdars-core/src/
```
**Pattern extraction date:** 2026-09-11
**All analog paths verified git-tracked:** `git ls-files -- <path>` non-empty for all 9 files
