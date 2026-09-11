# Phase 94: Reverse-Mode Autodiff Core (VJP Tape) - Pattern Map

**Mapped:** 2026-09-10
**Files analyzed:** 5 (3 new, 2 modified)
**Analogs found:** 5 / 5

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/src/autodiff/reverse.rs` | utility/numeric | transform | `fdars-core/src/autodiff.rs` (Dual + Scalar impls + tests, lines 225–1143) | exact |
| `fdars-core/src/autodiff/forward.rs` | utility/numeric | transform | `fdars-core/src/autodiff.rs` (verbatim move, full file) | exact |
| `fdars-core/src/autodiff/mod.rs` | module/barrel | N/A | `fdars-core/src/classification/mod.rs` + `fdars-core/src/depth/mod.rs` | role-match |
| `fdars-core/src/lib.rs` (line 76 only) | config | N/A | itself — line unchanged after dir refactor | N/A (no-op) |
| `fdars-core/src/prelude.rs` (line 21 only) | config | N/A | itself — extend existing `pub use crate::autodiff::{...}` | N/A (line edit) |

---

## Pattern Assignments

### `fdars-core/src/autodiff/reverse.rs` (utility, transform)

**Analog:** `fdars-core/src/autodiff.rs`

---

#### Struct declaration pattern (lines 225–231 of analog)

Copy `#[derive(Debug, Clone, Copy)]` and field visibility (`pub(crate)`) from `Dual`:

```rust
// fdars-core/src/autodiff.rs:225-231
#[derive(Debug, Clone, Copy)]
pub struct Dual {
    /// The primal value of the computation.
    pub value: f64,
    /// The tangent (directional derivative) accumulated by the chain rule.
    pub tangent: f64,
}
```

Mirror for `Var` (per CONTEXT.md decision):

```rust
#[derive(Debug, Clone, Copy)]
pub struct Var {
    /// The primal (forward) value.
    pub(crate) value: f64,
    /// Index into the thread-local tape. `usize::MAX` = constant (off-tape).
    pub(crate) node: usize,
}

const SENTINEL: usize = usize::MAX;
```

---

#### Thread-local tape pattern (analog: `fdars-core/src/alignment/mod.rs:465-474`)

This is the exact idiom already proven rayon-compatible in production:

```rust
// fdars-core/src/alignment/mod.rs:465-473
/// Per-thread reusable DP scratch: `(cost grid, parent pointers)`.
///
/// The grid fill in [`dp_grid_solve`] allocates two `nrows·ncols` buffers on
/// every alignment; in Karcher-mean / distance-matrix routines that is
/// `n·iters` (or `n²`) allocations of buffers that are cleared and rewritten
/// anyway. Keeping them in thread-local storage removes the allocator
/// round-trip while staying compatible with the `iter_maybe_parallel!` outer
/// loops (each rayon worker gets its own scratch).
static DP_SCRATCH: RefCell<(Vec<f64>, Vec<u32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
```

Access pattern (lines 506-508):

```rust
// fdars-core/src/alignment/mod.rs:506-508
DP_SCRATCH.with(|cell| {
    let mut scratch = cell.borrow_mut();
    // ...
})
```

Apply to tape:

```rust
thread_local! {
    static TAPE: RefCell<Vec<Node>> = const { RefCell::new(Vec::new()) };
}
```

---

#### Arithmetic op impl pattern — binary (lines 286–296 of analog)

```rust
// fdars-core/src/autodiff.rs:286-296
impl Mul for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: d(u*v) = u'*v + u*v'
        Dual {
            value: self.value * rhs.value,
            tangent: self.tangent * rhs.value + self.value * rhs.tangent,
        }
    }
}
```

Reverse-mode mirror — push a node instead of combining tangents:

```rust
impl Mul for Var {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let value = self.value * rhs.value;
        if self.node == SENTINEL && rhs.node == SENTINEL {
            return Var { value, node: SENTINEL };
        }
        // ∂(u*v)/∂u = v,  ∂(u*v)/∂v = u
        let node = push_binary(self.node, rhs.value, rhs.node, self.value);
        Var { value, node }
    }
}
```

---

#### Arithmetic op impl pattern — assign-ops (lines 322–341 of analog)

Delegate to binary ops; do NOT hand-write value-only mutation (would skip tape push):

```rust
// fdars-core/src/autodiff.rs:322-341
impl AddAssign for Dual {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for Dual {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl MulAssign for Dual {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
```

Mirror identically for `Var` — `*self = *self + rhs` ensures `push_binary` is called and `self.node` is updated.

---

#### Value-only `PartialEq` / `PartialOrd` (lines 347–362 of analog)

Do NOT derive — derive would compare `node` field and corrupt control flow in `softmin3_generic`:

```rust
// fdars-core/src/autodiff.rs:347-362
// Hand-written value-only equality, matching the value-only `PartialOrd` below.
impl PartialEq for Dual {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

// Hand-written value-only ordering. Do NOT `#[derive(PartialOrd)]`
impl PartialOrd for Dual {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}
```

---

#### `Scalar` impl pattern — constants (lines 364–392 of analog)

Dual uses `tangent: 0.0` for constants; `Var` uses `node: SENTINEL`:

```rust
// fdars-core/src/autodiff.rs:364-392
impl Scalar for Dual {
    #[inline]
    fn zero() -> Self { Dual { value: 0.0, tangent: 0.0 } }
    #[inline]
    fn one() -> Self { Dual { value: 1.0, tangent: 0.0 } }
    #[inline]
    fn from_f64(v: f64) -> Self { Dual { value: v, tangent: 0.0 } }
    #[inline]
    fn infinity() -> Self { Dual { value: f64::INFINITY, tangent: 0.0 } }
    // ...
}
```

Mirror for `Var` — sentinel, no tape push:

```rust
impl Scalar for Var {
    fn zero() -> Self     { Var { value: 0.0,          node: SENTINEL } }
    fn one() -> Self      { Var { value: 1.0,          node: SENTINEL } }
    fn from_f64(v: f64) -> Self { Var { value: v,      node: SENTINEL } }
    fn infinity() -> Self { Var { value: f64::INFINITY, node: SENTINEL } }
    // transcendentals: push_unary with correct local partial
}
```

---

#### `Scalar` impl pattern — transcendentals (lines 394–468 of analog)

Each transcendental computes the forward value and the local partial; `Dual` stores the partial multiplied by the incoming tangent; `Var` pushes a node instead:

```rust
// fdars-core/src/autodiff.rs:394-402  (sqrt)
#[inline]
fn sqrt(self) -> Self {
    // d/dx sqrt(v) = 1 / (2*sqrt(v))
    let s = self.value.sqrt();
    Dual { value: s, tangent: self.tangent / (2.0 * s) }
}
```

```rust
// fdars-core/src/autodiff.rs:444-458  (abs — subdifferential convention)
#[inline]
fn abs(self) -> Self {
    let sub = if self.value == 0.0 { 0.0 } else { self.value.signum() };
    Dual { value: self.value.abs(), tangent: self.tangent * sub }
}
```

```rust
// fdars-core/src/autodiff.rs:459-467  (signum — zero tangent everywhere)
#[inline]
fn signum(self) -> Self {
    Dual { value: self.value.signum(), tangent: 0.0 }
}
```

Mirror for `Var` (sqrt example — `signum` always pushes zero-weight node or returns sentinel):

```rust
fn sqrt(self) -> Self {
    let s = self.value.sqrt();
    if self.node == SENTINEL { return Var { value: s, node: SENTINEL }; }
    let node = push_unary(self.node, 1.0 / (2.0 * s));  // local partial = 1/(2√v)
    Var { value: s, node }
}
```

---

#### `#[must_use]` on entry point (line 510 of analog)

```rust
// fdars-core/src/autodiff.rs:510-511
#[must_use]
pub fn grad<F: Fn(&[Dual]) -> Dual>(f: F, x: &[f64]) -> (f64, Vec<f64>) {
```

Apply identically to `vjp`:

```rust
#[must_use]
pub fn vjp<F: Fn(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>) {
```

---

#### `grad` entry point body — template for `vjp` (lines 511–535 of analog)

```rust
// fdars-core/src/autodiff.rs:511-535
#[must_use]
pub fn grad<F: Fn(&[Dual]) -> Dual>(f: F, x: &[f64]) -> (f64, Vec<f64>) {
    let m = x.len();
    if m == 0 {
        return (f(&[]).value, Vec::new());
    }
    let mut gradient = vec![0.0; m];
    let mut value = 0.0;
    for k in 0..m {
        let duals: Vec<Dual> = (0..m)
            .map(|j| {
                if j == k { Dual::seed(x[j]) } else { Dual::constant(x[j]) }
            })
            .collect();
        let (v, t) = f(&duals).extract();
        if k == 0 { value = v; }
        gradient[k] = t;
    }
    (value, gradient)
}
```

`vjp` runs ONE forward pass + ONE reverse sweep (not a loop). The body structure from RESEARCH.md (already codebase-verified) is the impl target.

---

#### Test module structure (lines 629–634 of analog)

```rust
// fdars-core/src/autodiff.rs:629-634
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;
```

Mirror exactly — `use super::*` to pull in `vjp`, `Var`, `Scalar`; same `const TOL`.

---

#### Tier 1 known-answer test shape (lines 640–659 of analog)

```rust
// fdars-core/src/autodiff.rs:640-659
#[test]
fn dual_mul_known_answer() {
    // f(x) = x^2, f'(x) = 2x. At x = 3: f = 9, f' = 6.
    let d = Dual::seed(3.0);
    let r = d * d;
    assert!((r.value - 9.0).abs() < TOL, "primal {} != 9.0", r.value);
    assert!((r.tangent - 6.0).abs() < TOL, "tangent {} != 6.0", r.tangent);
}

#[test]
fn dual_sqrt_known_answer() {
    // f(x) = sqrt(x), f'(x) = 1/(2 sqrt(x)). At x = 4: f = 2, f' = 0.25.
    let r = Scalar::sqrt(Dual::seed(4.0));
    assert!((r.value - 2.0).abs() < TOL);
    assert!((r.tangent - 0.25).abs() < TOL);
}
```

Reverse-mode mirror uses `vjp` instead of `Dual::seed`:

```rust
#[test]
fn var_mul_known_answer() {
    let (value, grad) = vjp(|x| x[0] * x[0], &[3.0]);
    assert!((value - 9.0).abs() < TOL);
    assert!((grad[0] - 6.0).abs() < TOL);
}
```

---

#### Tier 2 singular-point guard tests (lines 847–889 of analog)

```rust
// fdars-core/src/autodiff.rs:847-857
#[test]
fn dual_sqrt_at_zero_tangent_is_nonfinite() {
    // LO-02: sqrt(0) tangent = 1/(2*0) diverges (non-finite).
    let r = Scalar::sqrt(Dual::seed(0.0));
    assert_eq!(r.value, 0.0);
    assert!(!r.tangent.is_finite(), "tangent {} should be non-finite", r.tangent);
}
```

Mirror for `Var` — check `grad[0].is_infinite()` / `.is_nan()` not `.is_finite()`, identical to guard intent.

---

#### `central_fd` helper and FD cross-check pattern (lines 895–913 of analog)

```rust
// fdars-core/src/autodiff.rs:895-897
fn central_fd(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-8_f64;
    (f(x + h) - f(x - h)) / (2.0 * h)
}
```

Keep the same helper in `reverse.rs` tests.

---

#### Composed-objective test — full structure (lines 1029–1119 of analog)

```rust
// fdars-core/src/autodiff.rs:1029-1119
#[test]
fn grad_composed_objective_matches_finite_diff() {
    use crate::matrix::FdMatrix;
    use crate::metric::soft_dtw_distance_generic;
    use crate::regression::{fdata_to_pc, project_scores_generic};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let m = 24usize;
    let n = 40usize;
    let ncomp = 3usize;
    let gamma = 0.1_f64;
    let lambda = 1.0_f64;

    let argvals: Vec<f64> = (0..m).map(|j| 0.1 + 0.8 * j as f64 / (m - 1) as f64).collect();
    let mut rng = StdRng::seed_from_u64(20260906);
    // ... [spanning full-rank training set] ...
    let fpca = fdata_to_pc(&data, ncomp, &argvals).unwrap();

    // Composed scalar objective:
    let objective = |c: &[Dual]| -> Dual {
        let sdtw = soft_dtw_distance_generic(c, &reference_duals, gamma);
        let scores = project_scores_generic(c, &mean, &rotation, &weights, ncomp);
        let mut acc = Dual::constant(0.0);
        for s in &scores { acc += *s * *s; }
        sdtw + Dual::constant(lambda) * acc
    };

    let (value, gradient) = grad(objective, &curve);
    // ...
    let h = 1e-6_f64;
    for j in 0..m {
        // central FD per component, tol 1e-6
    }
}
```

Mirror for `vjp` — replace `Dual::constant(r)` with `Scalar::from_f64(r)` on `Var`, `grad(objective, &curve)` with `vjp(objective, &curve)`. Use the same seed, same `m=24`, `n=40`, `ncomp=3`.

---

### `fdars-core/src/autodiff/forward.rs` (utility, transform — verbatim move)

**Analog:** `fdars-core/src/autodiff.rs` (full file, lines 1–628 excluding the `Scalar` trait definition which moves to `mod.rs`)

This file is a `git mv`-style move. Content: imports block (lines 1–25), `Scalar` trait (moves to `mod.rs` instead), `impl Scalar for f64` (lines 158–208), `Dual` struct + impls (lines 225–468), `diff` / `grad` / `jacobian` / `directional_derivative` entry points (lines 484–627).

**Imports pattern** (lines 1–25 of analog to replicate in `forward.rs`):

```rust
// fdars-core/src/autodiff.rs:1-25  (approximate — read full file for exact block)
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
```

`forward.rs` adds `use super::Scalar;` at top to reference the trait now defined in `mod.rs`.

---

### `fdars-core/src/autodiff/mod.rs` (module/barrel)

**Analog:** `fdars-core/src/classification/mod.rs` and `fdars-core/src/depth/mod.rs`

**Barrel pattern** (`classification/mod.rs:1-30`):

```rust
// fdars-core/src/classification/mod.rs:1-27
//! Functional classification with mixed scalar/functional predictors.
// ...
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc;

pub mod cv;
pub mod dd;
pub mod fit;
// ...

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------
```

**Depth barrel re-export style** (`depth/mod.rs:29-31`):

```rust
// fdars-core/src/depth/mod.rs:29-31
// Re-export all public functions
pub use band::{band, modified_band, modified_epigraph_index};
```

Apply to `autodiff/mod.rs`:

```rust
//! Automatic differentiation — forward-mode ([`Dual`]) and reverse-mode ([`Var`]).

pub mod forward;
pub mod reverse;

// Shared trait (used by both forward and reverse)
// (Scalar trait definition lives here, moved from autodiff.rs)

// Re-export everything so `autodiff::Dual`, `autodiff::Var`, `autodiff::vjp`,
// `autodiff::grad`, etc. all resolve — preserving all existing public paths.
pub use forward::{diff, directional_derivative, grad, jacobian, Dual};
pub use reverse::{vjp, Var, Tape};  // Tape only if it has public surface
```

---

### `fdars-core/src/lib.rs` (line 76 — no-op)

**Current line 76:**

```rust
// fdars-core/src/lib.rs:76
pub mod autodiff;
```

This line is **unchanged** after the directory refactor. Cargo resolves `pub mod autodiff;` to either `src/autodiff.rs` or `src/autodiff/mod.rs` automatically. No edit required — the planner should note this as a verification step only (confirm line 76 still reads exactly as above after the `git mv`).

---

### `fdars-core/src/prelude.rs` (line 21 — extend re-export)

**Current line 21:**

```rust
// fdars-core/src/prelude.rs:21
pub use crate::autodiff::{diff, directional_derivative, grad, jacobian, Dual, Scalar};
```

**Target line 21 after Phase 94:**

```rust
pub use crate::autodiff::{diff, directional_derivative, grad, jacobian, vjp, Dual, Scalar, Var};
```

`Tape` is left at `autodiff::Tape` only (not re-exported in prelude) — it is opaque and fully hidden behind `vjp` per CONTEXT.md discretion guidance.

---

## Shared Patterns

### `#[must_use]` on all entry points
**Source:** `fdars-core/src/autodiff.rs:483, 510, 558, 611`
**Apply to:** `vjp` in `autodiff/reverse.rs`

```rust
// fdars-core/src/autodiff.rs:483-484
#[must_use]
pub fn diff<F: Fn(Dual) -> Dual>(f: F, x: f64) -> (f64, f64) {
```

```rust
// fdars-core/src/autodiff.rs:510-511
#[must_use]
pub fn grad<F: Fn(&[Dual]) -> Dual>(f: F, x: &[f64]) -> (f64, Vec<f64>) {
```

### Error handling (no `Result` — AD is infallible)
**Source:** `fdars-core/src/autodiff.rs:484-486` — `diff` returns `(f64, f64)` directly, no `Result`.
**Apply to:** `vjp` returns `(f64, Vec<f64>)` directly. Domain errors (NaN/Inf adjoints) propagate as values, not panics. This matches the `Dual` contract: "callers own range checking."

### `const TOL` in tests
**Source:** `fdars-core/src/autodiff.rs:634`
```rust
const TOL: f64 = 1e-10;
```
**Apply to:** `#[cfg(test)] mod tests` in `autodiff/reverse.rs` — same constant, same name.

### Module doc comment style
**Source:** `fdars-core/src/classification/mod.rs:1-12`
```rust
//! Functional classification with mixed scalar/functional predictors.
//!
//! Implements supervised classification for functional data using:
//! - [`fclassif_lda`] ...
```
**Apply to:** `autodiff/mod.rs` and `autodiff/reverse.rs` module-level `//!` doc with one-sentence summary + list of public items.

---

## Read-Only Integration Points (NOT modified — called with `Var`)

| Function | File | Line | Role |
|----------|------|------|------|
| `soft_dtw_distance_generic<S: Scalar>` | `fdars-core/src/metric/soft_dtw.rs` | 158 | Validation target #1 — already generic, instantiate at `Var` in tests |
| `project_scores_generic<S: Scalar>` | `fdars-core/src/regression.rs` | 232 | Validation target #2 — already generic, instantiate at `Var` in tests |
| `soft_dtw_distance_generic::<f64>` | same | 158 | Used as FD oracle in cross-check tests |
| `project_scores_generic::<f64>` | same | 232 | Used as FD oracle in cross-check tests |

These functions require zero call-site changes. The planner should note them as compile-check targets only: `cargo check -p fdars-core --features linalg` must succeed with `soft_dtw_distance_generic::<Var>` and `project_scores_generic::<Var>` instantiated.

---

## No Analog Found

All files have strong analogs. No entries in this section.

---

## Metadata

**Analog search scope:** `fdars-core/src/autodiff.rs`, `fdars-core/src/alignment/mod.rs`, `fdars-core/src/classification/mod.rs`, `fdars-core/src/depth/mod.rs`, `fdars-core/src/prelude.rs`, `fdars-core/src/lib.rs`
**Files scanned:** 6
**Pattern extraction date:** 2026-09-10
