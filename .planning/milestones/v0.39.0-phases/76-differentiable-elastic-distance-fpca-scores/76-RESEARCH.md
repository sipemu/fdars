# Phase 76: Differentiable Elastic Distance & FPCA Scores — Research

**Researched:** 2026-09-06
**Domain:** Forward-mode AD generics over in-crate `Scalar` substrate; soft-DTW and SRSF-based amplitude distance; FPCA score projection
**Confidence:** HIGH (all claims verified by direct file reads this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Generic-over-`Scalar`, gradients compose.** Write the op body once over `S: Scalar`; instantiate at `f64` and `Dual`.
- **Forward-mode only.**
- **Additive/non-breaking.** Existing f64 functions (`soft_dtw_distance`, `elastic_distance`, `amplitude_distance`, `phase_distance_pair`, and the FPCA scoring path in `regression.rs`) MUST stay byte-for-byte unchanged in signature and behavior. Add the generic version ALONGSIDE (e.g. a `*_generic<S: Scalar>` companion, or a private generic core that the existing f64 fn delegates to — delegation is allowed ONLY if the f64 output is provably identical within 1e-12 and no public signature changes). Protects R + WASM bindings + 28 examples.
- **No new crate dependency.**

### Claude's Discretion
- **Container for generic curves:** generic versions take `&[S]` slices (NOT a generic `FdMatrix<S>`; `FdMatrix` stays `Vec<f64>`, unchanged). `argvals`, `gamma`, `lambda`, FPCA `rotation`/`mean`/integration `weights` stay `f64` — only the curve whose gradient we want goes generic. Lift f64 constants into the scalar via `S::from_f64(...)`.
- **Which distance(s) to genericize (DIF-02):** at minimum `soft_dtw_distance` (the pilot) AND the amplitude/elastic distance. Research must confirm the SRSF q-transform and DP/warping internals are differentiable under `Dual`. If closed-curve / nd / banded variants are hard, they are OUT OF SCOPE.
- **FPCA score entry (DIF-03):** the projection that maps a (centered) curve onto trained loadings → scores. Rotation/mean/weights stay f64; input curve `&[S]`; output scores `Vec<S>`.
- **Gradient direction:** w.r.t. the input curve's sample values (a `&[S]` of length m).

### Deferred Ideas (OUT OF SCOPE)
- Closed-curve / n-D / banded elastic-distance generic variants.
- Full length-m Jacobian / ergonomic public gradient API + composition example — Phase 77 (DIF-04).
- Reverse-mode — DIF-F1.
- Making the existing f64 public signatures themselves generic — DIF-F3.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DIF-02 | Differentiable elastic distance — generic-over-`Scalar` elastic (soft-DTW / amplitude+phase) distance path. At `Dual` yields exact forward-mode gradients vs central FD (≤1e-6) AND vs hand-written soft-DTW gradient oracle (tight tolerance); at `f64` reproduces current numerics within 1e-12. | Soft-DTW DP recurrence is fully differentiable under `Dual`; SRSF amplitude distance with fixed warping is differentiable; DP warp search is a discrete argmin (non-differentiable — see DIF-02 scope call below). |
| DIF-03 | Differentiable FPCA scores — generic-over-`Scalar` FPCA score projection so gradients of FPC scores w.r.t. input-curve values flow through when instantiated at `Dual`; validated vs central FD (≤1e-6); `f64` instantiation reproduces existing FPCA scores within tolerance. | `FpcaResult::project` inner loop uses only sub/mul/addassign/zero; rotation/mean/weights stay f64; input curve becomes `&[S]`. Linear in the input — gradient is a constant (inner-product of rotation column with weights), verifiable analytically. |
</phase_requirements>

---

## Summary

Phase 76 adds two generic-over-`Scalar` FDA operations alongside their existing f64 counterparts, so that instantiating at `Dual` yields exact forward-mode gradients and instantiating at `f64` reproduces the existing numerics exactly.

**DIF-02 — differentiable distance.** Soft-DTW is cleanly differentiable under `Dual`: its DP recurrence uses only the `Scalar` ops already in the trait (sub, mul, div, neg, add, exp, ln, infinity for initialization, PartialOrd for min/branching). The SRSF-based amplitude distance (`elastic_distance` / `amplitude_distance`) routes through a DP warp search (`dp_alignment_core_banded`) that is a discrete argmin over path cells — this is fundamentally non-differentiable. Therefore DIF-02's achievable elastic-distance deliverable is the **SRSF amplitude distance with a fixed (pre-computed) warping**: given a warping `gamma: &[f64]`, compute the L2 distance in SRSF space between q1 and the aligned q2, where the alignment is fixed and not differentiated. The SRSF q-transform itself uses `sign * sqrt(|f'|)` which is differentiable except at f'=0; see the non-smoothness section below.

**DIF-03 — differentiable FPCA scores.** `FpcaResult::project` uses a three-level loop: for each observation i and component k, accumulate `sum += (data[(i,j)] - mean[j]) * rotation[(j,k)] * weights[j]`. The generic version replaces the f64 accumulator and input data with `S: Scalar`, while `mean`, `rotation`, and `weights` remain f64. The gradient is a constant linear functional of the input — cleanest possible AD target.

**Primary recommendation:** Implement (1) `soft_dtw_distance_generic<S: Scalar>(x: &[S], y: &[S], gamma: f64) -> S` as a private core with `soft_dtw_distance` delegating to it; (2) `amplitude_distance_at_warp<S: Scalar>(q1: &[S], f2_aligned: &[S], argvals: &[f64]) -> S` (fixed-warping path); (3) a free function `project_scores_generic<S: Scalar>(curve: &[S], mean: &[f64], rotation: &FdMatrix, weights: &[f64]) -> Vec<S>` for DIF-03.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `soft_dtw_distance_generic` | `src/metric/soft_dtw.rs` (new private core) | — | Lives next to the existing f64 fn; delegation pattern keeps f64 parity bit-identical |
| `softmin3_generic` | `src/metric/soft_dtw.rs` (new private helper) | — | Mirrors `softmin3(f64)` but typed over `S: Scalar` |
| `amplitude_distance_at_warp_generic` | `src/alignment/pairwise.rs` OR new `src/alignment/differentiable.rs` | — | Fixed-warp amplitude distance; adjacent to existing `amplitude_distance` |
| `srsf_l2_distance_generic` | `src/alignment/differentiable.rs` OR inline | — | Generic L2 distance in SRSF space: needs sub/mul/addassign/sqrt |
| `project_scores_generic` | `src/regression.rs` (new pub fn) | — | Free function alongside `FpcaResult::project`; DIF-03 target |
| Module re-exports | `src/lib.rs` (minimal additions) | — | Phase 77 owns full prelude surface; this phase adds only what tests need |
| Tests | Inline `#[cfg(test)]` in each module | — | Crate convention |

---

## Source File Analysis

### `src/metric/soft_dtw.rs` — The Pilot + Gradient Oracle

**Exact `soft_dtw_distance` signature** [VERIFIED: fdars-core/src/metric/soft_dtw.rs:49-71]:
```rust
pub fn soft_dtw_distance(x: &[f64], y: &[f64], gamma: f64) -> f64
```

**DP recurrence — verbatim from lines 56-70:**
```rust
let mut prev = vec![f64::INFINITY; m + 1];
let mut curr = vec![f64::INFINITY; m + 1];
prev[0] = 0.0;

for i in 1..=n {
    curr.fill(f64::INFINITY);
    for j in 1..=m {
        let d = x[i - 1] - y[j - 1];
        let cost = d * d;
        curr[j] = cost + softmin3(prev[j], curr[j - 1], prev[j - 1], gamma);
    }
    std::mem::swap(&mut prev, &mut curr);
}
```

**`softmin3` function — verbatim from lines 29-39:**
```rust
pub(super) fn softmin3(a: f64, b: f64, c: f64, gamma: f64) -> f64 {
    let min_val = a.min(b).min(c);
    if !min_val.is_finite() {
        return min_val;
    }
    let neg_inv_gamma = -1.0 / gamma;
    let ea = ((a - min_val) * neg_inv_gamma).exp();
    let eb = ((b - min_val) * neg_inv_gamma).exp();
    let ec = ((c - min_val) * neg_inv_gamma).exp();
    min_val - gamma * (ea + eb + ec).ln()
}
```

**Gradient oracle — `soft_dtw_accumulate_gradient` (lines 238-249):**
```rust
fn soft_dtw_accumulate_gradient(bary: &[f64], xi: &[f64], gamma: f64, grad: &mut [f64]) {
    let m = bary.len();
    let r = soft_dtw_forward(bary, xi, gamma);
    let e = soft_dtw_backward(bary, xi, &r, gamma);
    for k in 1..=m {
        let mut g = 0.0;
        for j in 1..=xi.len() {
            g += e[k][j] * 2.0 * (bary[k - 1] - xi[j - 1]);
        }
        grad[k - 1] += g;
    }
}
```

This oracle computes d(soft_dtw_distance(bary, xi)) / d(bary[k]) for each k. It routes through a forward/backward pass — a classical dynamic-programming gradient. The SC #1 test compares the Dual gradient against this oracle.

**Scalar ops needed by soft-DTW generic path:**
- `sub` (d = x[i-1] - y[j-1])
- `mul` (cost = d * d)
- `add` (cost + softmin3_result)
- DP initialization: `S::infinity()` (sentinel for prev/curr cells)
- In `softmin3_generic`: `PartialOrd` (for `min_val = a.min(b).min(c)`), `sub`, `mul`, `div`, `neg`, `exp`, `ln`, `add`, `from_f64` (for `-1.0/gamma`)
- `is_finite()` check on `min_val`: **gap — see Scalar Trait Gap Analysis**

**Key implementation note — `fill` and `mem::swap`:** The 2-row rolling buffer uses `Vec::fill` and `mem::swap`. For a generic `Vec<S>`, `fill` requires `S: Clone` (already in `Scalar` bounds). `mem::swap` is type-agnostic. The generic version allocates `Vec<S>` the same way, initialized via `vec![S::infinity(); m+1]`.

### `src/alignment/pairwise.rs` — Elastic/Amplitude/Phase Distances

**`elastic_distance` (line 103):**
```rust
pub fn elastic_distance(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    elastic_align_pair(f1, f2, argvals, lambda).distance
}
```
This delegates to `elastic_align_pair` → `elastic_align_pair_from_srsf` [VERIFIED: fdars-core/src/alignment/pairwise.rs:103-105, 129-153]:
1. Compute SRSFs: `q1 = srsf_single(f1, argvals)`, `q2 = srsf_single(f2, argvals)` (f64 → f64)
2. Find optimal warping via DP: `gamma = dp_alignment_core_banded(q1, q2, argvals, lambda, band)` (discrete argmin — **non-differentiable**)
3. Apply warping: `f_aligned = reparameterize_curve(f2, argvals, &gamma)` (linear interpolation at gamma points — piecewise linear, not smooth)
4. Compute aligned SRSF: `q_aligned = srsf_single(&f_aligned, argvals)` (derivative-based)
5. L2 distance: `l2_distance(q1, &q_aligned, &weights)` (weighted Euclidean)

**`amplitude_distance` (line 384):** [VERIFIED: fdars-core/src/alignment/pairwise.rs:384-386]
```rust
pub fn amplitude_distance(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    elastic_distance(f1, f2, argvals, lambda)
}
```
Exact alias — same computation path.

**`phase_distance_pair` (line 389):** [VERIFIED: fdars-core/src/alignment/pairwise.rs:389-392]
```rust
pub fn phase_distance_pair(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    let alignment = elastic_align_pair(f1, f2, argvals, lambda);
    crate::warping::phase_distance(&alignment.gamma, argvals)
}
```
Computes geodesic distance of the optimal warp from identity. Depends on DP warp search — **non-differentiable**.

**`l2_distance` (helpers.rs:56-63):** [VERIFIED: fdars-core/src/helpers.rs:56-63]
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
This is `sqrt(sum_j (q1[j] - q_aligned[j])^2 * w[j])`. A generic version needs: `sub`, `mul`, `add` (addassign), `from_f64` (to convert weights), `sqrt`, `zero`. Straightforward.

### `src/alignment/srsf.rs` — SRSF Transform

**`srsf_transform` — verbatim (lines 36-52):** [VERIFIED: fdars-core/src/alignment/srsf.rs:36-52]
```rust
pub fn srsf_transform(data: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    // ...
    let deriv = deriv_1d(data, argvals, 1);
    let mut result = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let d = deriv[(i, j)];
            result[(i, j)] = d.signum() * d.abs().sqrt();
        }
    }
    result
}
```

The SRSF formula is `q(t) = sign(f'(t)) * sqrt(|f'(t)|)`. In the generic setting:
- `signum(f'(t))` — piecewise constant; derivative is 0 everywhere (correct for AD)
- `abs(f'(t))` — non-differentiable at f'=0 (subdifferential convention; Dual.abs handles it)
- `sqrt(|f'(t)|)` — diverges at f'=0 (Dual.sqrt gives inf tangent at 0)

The derivative chain is `d/dx q(t) = d/dx [sign(f') * |f'|^0.5]`. This is *not* needed for the fixed-warping amplitude distance deliverable, since `srsf_single` takes `&[f64]` and returns `&[f64]` (the SRSF of the ALREADY-WARPED curve stays f64). The generic input curve values affect the amplitude distance via a *different* path in the fixed-warping formulation — see the Recommended Deliverable Scope section.

**`srsf_single` signature (line 109):** [VERIFIED: fdars-core/src/alignment/srsf.rs:109-114]
```rust
pub(crate) fn srsf_single(f: &[f64], argvals: &[f64]) -> Vec<f64>
```
Takes f64 slices only. Cannot take `&[S]` without rewriting the derivative computation inside `srsf_transform` to be generic. This is a significant internal chain.

### `src/regression.rs` — FPCA Score Projection

**`FpcaResult::project` inner loop — verbatim (lines 117-125):** [VERIFIED: fdars-core/src/regression.rs:117-125]
```rust
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

The generic projection of a single curve (one row i) onto component k is:
```
score_k = sum_j (curve[j] - mean[j]) * rotation[j,k] * weights[j]
```
Where `mean[j]`, `rotation[j,k]`, `weights[j]` are f64 (from a trained FpcaResult), and `curve[j]` is `S`. The triple product mixes `S` and `f64`. The clean approach: convert f64 constants to `S` via `S::from_f64(c)` before multiplying.

**`FpcaResult` fields (all f64):** [VERIFIED: fdars-core/src/regression.rs:49-62]
```rust
pub struct FpcaResult {
    pub singular_values: Vec<f64>,
    pub rotation: FdMatrix,       // m x ncomp, column-major Vec<f64>
    pub scores: FdMatrix,         // n x ncomp
    pub mean: Vec<f64>,           // length m
    pub centered: FdMatrix,       // n x m
    pub weights: Vec<f64>,        // length m (Simpson's integration weights)
}
```
`FdMatrix` is `Vec<f64>` column-major; indexing is `matrix[(row, col)]`. Element `(j, k)` is at `j + k * nrows`. These stay f64 — the generic function receives a pre-trained `&FpcaResult` and a generic input curve.

---

## Recommended Deliverable Scope

### DIF-02 — Scope Call: Two Deliverables

**Deliverable 2a — `soft_dtw_distance_generic<S: Scalar>`:** Fully differentiable. The DP recurrence and softmin are smooth functions of the input series values. The existing `soft_dtw_distance(x, y, gamma)` can be made to delegate to this, preserving bit-identical f64 output.

**Deliverable 2b — `amplitude_distance_at_warp_generic<S: Scalar>`:** Fixed-warp SRSF amplitude distance. Given:
- `q1_ref: &[f64]` — pre-computed SRSF of the reference curve (f64, not differentiated)
- `curve2: &[S]` — the second curve (generic — gradient flows through these values)
- `warping: &[f64]` — pre-computed optimal warping gamma (f64, fixed, not differentiated)
- `argvals: &[f64]` — evaluation grid
- `weights: &[f64]` — Simpson integration weights

The computation is:
1. `f2_aligned[j] = linear_interp(argvals, curve2_as_f64, gamma[j])` — BUT this requires materializing `curve2` as f64, losing the gradient. Instead: implement a generic linear interpolation `generic_linear_interp(argvals, curve: &[S], t: f64) -> S` (linear in the input values — straightforward for `S: Scalar`).
2. Compute generic SRSF of the aligned curve: this requires a numeric derivative of `f2_aligned` — a finite-difference approximation on the `S` values. The standard SRSF uses `deriv_1d` internally (which is also `f64`). For the generic path, implement a generic central-difference derivative: `d_j = (f2_aligned[j+1] - f2_aligned[j-1]) / S::from_f64(2.0 * h)` — differentiable under Dual.
3. `q2_aligned[j] = S::signum(d_j) * S::sqrt(S::abs(d_j))` — this requires `signum` and `abs` which are in the `Scalar` trait. Non-smoothness at d_j=0 is handled by the subdifferential convention already in `Dual::abs` and `Dual::signum`.
4. Generic L2 distance: `sqrt(sum_j (q1_ref[j] - q2_aligned[j])^2 * weights[j])` — uses `sub`, `mul`, `add`, `from_f64`, `sqrt`, `zero` — all in `Scalar`.

**Why NOT genericize the full `elastic_distance`:** The DP warp search (`dp_alignment_core_banded`) is a discrete `argmin` over a grid of cells. The optimal path is selected by comparing accumulated costs with `<` (PartialOrd on values), but the path *index* selected is non-differentiable (it is an integer lookup, not a smooth function of the inputs). Backpropagating through this argmin would require either (a) a continuous relaxation (soft-DTW's approach!) or (b) treating the warping as fixed. The current code explicitly uses DP with parent-pointer traceback, a piecewise-constant function of the input — Dual's PartialOrd correctly handles the branching for the cost accumulation, but the path indices `(sr, sc, tr, tc)` used in `dp_relax_cell` are integers, so the final gamma is also piecewise-constant in the inputs. Forward-mode Dual would propagate gradients through the cost values but the gamma used in step 3 would have zero gradient everywhere — producing misleading (not wrong, but zero) gradients for `elastic_distance`.

**Therefore: DIF-02 delivers `soft_dtw_distance_generic` (full) + `amplitude_distance_at_warp_generic` (fixed-warping formulation). The full `elastic_distance` (warp-searched) is explicitly OUT OF SCOPE because the argmin is non-differentiable.**

### DIF-03 — `project_scores_generic<S: Scalar>`

Fully differentiable. Single free function (not a method on `FpcaResult` — methods on `FpcaResult` return `FdMatrix` which is `Vec<f64>`):

```rust
pub fn project_scores_generic<S: Scalar>(
    curve: &[S],
    mean: &[f64],
    rotation: &FdMatrix,
    weights: &[f64],
    ncomp: usize,
) -> Vec<S>
```

---

## Standard Stack

### Core (no new dependencies)

| Component | Purpose | Why Standard |
|-----------|---------|--------------|
| `crate::autodiff::{Scalar, Dual}` | The `S: Scalar` bound and the concrete `Dual` type for gradient extraction | Phase 75 deliverable, already committed |
| `crate::matrix::FdMatrix` | Stays `Vec<f64>` — rotation/mean/weights are read from it as f64 constants | Unchanged |
| `crate::helpers::simpsons_weights` | Integration weights for FPCA — already f64, unchanged | Unchanged |
| `std::mem::swap` | Rolling buffer swap in soft-DTW DP | Zero-cost, no allocation |

### No new dependencies

This phase adds zero new crate dependencies.

---

## Package Legitimacy Audit

Not applicable — this phase installs no new packages.

---

## Scalar Trait Gap Analysis

**Trait as shipped** [VERIFIED: fdars-core/src/autodiff.rs:59-105]: The `Scalar` trait has:
```
Copy, Clone, Debug, PartialOrd, Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign,
zero(), one(), from_f64(), infinity(),
sqrt(), exp(), ln(), sin(), cos(), powf(), abs(), signum()
```

**Gap found — `is_finite()` method:**

The `softmin3` function has a guard: [VERIFIED: fdars-core/src/metric/soft_dtw.rs:31-33]
```rust
let min_val = a.min(b).min(c);
if !min_val.is_finite() {
    return min_val;
}
```

The `Scalar` trait does NOT have an `is_finite()` method. For the generic `softmin3_generic<S: Scalar>`, the guard needs to be handled. Options:

1. **Preferred — no trait change:** In the generic core, replace the `is_finite` guard with a comparison: `if min_val >= S::infinity() { return min_val; }`. This works because `S::infinity()` is in the trait, and `PartialOrd` compares values. For `f64`, `f64::INFINITY >= f64::INFINITY` is true; for `Dual`, the value-only comparison works identically. However, `f64::NEG_INFINITY` would slip through (it is less than `infinity()`). A safer check: `if min_val == S::infinity() { return min_val; }` — but Dual's PartialEq is value-only and would catch this correctly. **This approach avoids any Phase 75 trait change.**

2. **Alternative — add `fn is_finite(self) -> bool` to `Scalar`:** Clean but requires reopening the Phase 75 trait. This is a **trait-breaking change** if any downstream code implements `Scalar`. Since only `f64` and `Dual` implement it (both in-crate), this is safe in practice. However it is technically a breaking trait change in the semver sense.

**Recommendation:** Use option 1 (no trait change). The `>= S::infinity()` guard (or a dedicated `fn is_pos_infinite(self) -> bool` on `Dual`/`f64` outside the trait) avoids touching Phase 75. Document this in the plan as the chosen approach. If the plan reveals a cleaner case for `is_finite`, the plan task can add it to `Scalar` explicitly — it is a backward-compatible addition (new method with a provided default) if a default `fn is_finite(self) -> bool { self.abs() != Self::infinity() }` is added. Given MSRV 1.81 stability, a provided default on the trait is safest.

**Gap found — `f64 * S` mixed multiply:**

The `softmin3_generic` uses `gamma * (...).ln()` where `gamma: f64` and `(...)` is `S`. The current `Scalar` trait bounds `Mul<Output = Self>` (i.e., `S * S`), not `Mul<f64, Output = S>`. To multiply an `S` by a plain `f64` coefficient, use `S::from_f64(gamma) * s` — this is the already-recommended pattern from Phase 75 research (open question 3). No trait change needed. The `neg_inv_gamma = -1.0 / gamma` is a plain `f64` computation that produces a `f64`; lift it with `S::from_f64(neg_inv_gamma)` before multiplying.

**Summary — no Scalar trait changes are REQUIRED for Phase 76.** All needed ops are available via `S::from_f64(c)` conversions and `S::infinity()`. An optional `is_finite` addition could be considered but is not blocking.

---

## Architecture Patterns

### System Architecture Diagram

```
                 test driver
                    │
        ┌───────────┴─────────────────┐
        │                             │
 DIF-02a: soft_dtw_distance_generic   DIF-02b: amplitude_distance_at_warp_generic
        │    S: Scalar                         │   S: Scalar
        │                                      │
  softmin3_generic<S>              generic_linear_interp<S>
        │                                      │
  [S::sub, S::mul, S::div,           generic_srsf_central_diff<S>
   S::neg, S::exp, S::ln,                      │
   S::add, S::infinity,             [S::sub, S::div, S::signum,
   S::PartialOrd (min),              S::abs, S::sqrt]
   S::from_f64]                                │
                                    generic_l2_distance<S>
                                               │
                                    [S::sub, S::mul, S::add,
                                     S::from_f64, S::sqrt, S::zero]

 DIF-03: project_scores_generic
        │   S: Scalar
        │
  [S::sub, S::mul, S::add (+=),
   S::from_f64, S::zero]
```

### Recommended Project Structure

```
fdars-core/src/
├── metric/
│   └── soft_dtw.rs          # ADD: softmin3_generic<S>, soft_dtw_distance_generic<S>
│                            # MODIFY: soft_dtw_distance delegates to generic core
├── alignment/
│   └── differentiable.rs   # NEW: amplitude_distance_at_warp_generic<S>,
│                            #      generic_linear_interp<S>,
│                            #      generic_srsf_central_diff<S>,
│                            #      generic_l2_distance<S>
│   └── mod.rs              # ADD: pub mod differentiable; pub use differentiable::...
├── regression.rs            # ADD: project_scores_generic<S>
└── lib.rs                   # ADD: pub use for the 3 new public fns
```

### Pattern 1: Soft-DTW Generic Core (DIF-02a)

**Recommended signature:**
```rust
// Private core — both f64 and Dual paths use this
fn softmin3_generic<S: Scalar>(a: S, b: S, c: S, gamma: f64) -> S {
    // min by value-only PartialOrd (correct for Dual — value-only comparison)
    let min_val = if a <= b { if a <= c { a } else { c } } else { if b <= c { b } else { c } };
    // Guard: if min_val is +infinity, return it (handles initial sentinel cells)
    // Uses >= S::infinity() since Scalar has no is_finite() method
    if min_val >= S::infinity() {
        return min_val;
    }
    let neg_inv_gamma = S::from_f64(-1.0 / gamma);
    let ea = S::exp((a - min_val) * neg_inv_gamma);
    let eb = S::exp((b - min_val) * neg_inv_gamma);
    let ec = S::exp((c - min_val) * neg_inv_gamma);
    min_val - S::from_f64(gamma) * S::ln(ea + eb + ec)
}

/// Generic soft-DTW distance — private inner kernel.
fn soft_dtw_distance_inner<S: Scalar>(x: &[S], y: &[S], gamma: f64) -> S {
    let n = x.len();
    let m = y.len();
    if n == 0 || m == 0 {
        return S::zero();
    }
    let mut prev = vec![S::infinity(); m + 1];
    let mut curr = vec![S::infinity(); m + 1];
    prev[0] = S::zero();
    for i in 1..=n {
        for v in curr.iter_mut() { *v = S::infinity(); }
        for j in 1..=m {
            let d = x[i - 1] - y[j - 1];
            let cost = d * d;
            curr[j] = cost + softmin3_generic(prev[j], curr[j - 1], prev[j - 1], gamma);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

/// Public generic entry point (DIF-02a).
pub fn soft_dtw_distance_generic<S: Scalar>(x: &[S], y: &[S], gamma: f64) -> S {
    soft_dtw_distance_inner(x, y, gamma)
}
```

**Delegation for f64 parity:** Make `soft_dtw_distance` delegate to the inner kernel at `S = f64`:
```rust
pub fn soft_dtw_distance(x: &[f64], y: &[f64], gamma: f64) -> f64 {
    soft_dtw_distance_inner(x, y, gamma)
}
```
This is **bit-identical** at f64 because `Scalar for f64` is a zero-cost passthrough to the same `f64` intrinsics (verified in Phase 75 research: `impl Scalar for f64` delegates to `f64::exp`, `f64::ln`, etc. directly). The parity test (SC #2) will confirm with ≤1e-12 tolerance — expected to be 0 ULP difference.

**f64-parity risk:** LOW — the only change to the f64 code path is that `a.min(b)` becomes an `if/else` branch. IEEE 754 `f64::min(a,b)` returns the non-NaN argument if one is NaN; the `if a <= b { a } else { b }` branch behaves identically for NaN-free inputs (both return `b` when `a` is NaN because `NaN <= b` is false). Verify with the f64-parity test. If any discrepancy appears, keep the original `soft_dtw_distance` untouched and add `soft_dtw_distance_generic` as a pure companion (no delegation).

### Pattern 2: Fixed-Warp Amplitude Distance (DIF-02b)

**What is being differentiated:** The gradient of the amplitude distance w.r.t. the VALUES of the second curve `f2` at the evaluation grid, assuming the warping `gamma` is held fixed (pre-computed from the f64 path). This is useful for gradient-based optimization of `f2` values toward a target amplitude distance.

**The key internal helpers needed (all new generic helpers):**

```rust
/// Generic piecewise-linear interpolation: f(t) for t in [argvals[0], argvals[m-1]].
/// argvals and t_query are f64; curve values are S (gradient flows through these).
fn generic_linear_interp<S: Scalar>(argvals: &[f64], curve: &[S], t: f64) -> S {
    // Binary search for bracket; linear combination of two S values
    // gradient is (1-alpha)*curve[lo_idx] + alpha*curve[hi_idx] for alpha in [0,1]
    // — purely linear in curve values, alpha is f64
    let m = argvals.len();
    if m == 0 { return S::zero(); }
    if t <= argvals[0] { return curve[0]; }
    if t >= argvals[m-1] { return curve[m-1]; }
    // find j: argvals[j] <= t < argvals[j+1]
    let j = argvals.partition_point(|&a| a <= t).saturating_sub(1).min(m - 2);
    let dt = argvals[j+1] - argvals[j];
    if dt <= 0.0 { return curve[j]; }
    let alpha = S::from_f64((t - argvals[j]) / dt);
    curve[j] * (S::one() - alpha) + curve[j+1] * alpha
}

/// Generic central-difference SRSF: q[j] = sign(f'[j]) * sqrt(|f'[j]|).
/// Uses central differences with h = (argvals[m-1] - argvals[0]) / (m-1).
/// Non-smooth at f'=0 — handled by Dual's abs/signum subdifferential convention.
fn generic_srsf_central_diff<S: Scalar>(curve: &[S], argvals: &[f64]) -> Vec<S> {
    let m = curve.len();
    let h = if m > 1 { (argvals[m-1] - argvals[0]) / (m-1) as f64 } else { 1.0 };
    let inv2h = S::from_f64(1.0 / (2.0 * h));
    let mut q = vec![S::zero(); m];
    for j in 0..m {
        let deriv = if j == 0 {
            // Forward difference at left boundary
            (curve[1] - curve[0]) * S::from_f64(1.0 / h)
        } else if j == m - 1 {
            // Backward difference at right boundary
            (curve[m-1] - curve[m-2]) * S::from_f64(1.0 / h)
        } else {
            (curve[j+1] - curve[j-1]) * inv2h
        };
        q[j] = S::signum(deriv) * S::sqrt(S::abs(deriv));
    }
    q
}

/// Generic weighted L2 distance: sqrt(sum_j (q1[j] - q2[j])^2 * w[j]).
/// q1 is f64 (reference); q2 is S (generic); weights is f64.
fn generic_l2_srsf_distance<S: Scalar>(q1: &[f64], q2: &[S], weights: &[f64]) -> S {
    let mut dist_sq = S::zero();
    for j in 0..q1.len() {
        let diff = S::from_f64(q1[j]) - q2[j];
        dist_sq += diff * diff * S::from_f64(weights[j]);
    }
    S::sqrt(dist_sq)
}

/// Amplitude distance with a FIXED pre-computed warping (DIF-02b).
/// gradient w.r.t. curve2 values flows through.
pub fn amplitude_distance_at_warp_generic<S: Scalar>(
    q1_ref: &[f64],        // SRSF of reference curve f1 (pre-computed, f64)
    curve2: &[S],          // second curve values (generic — gradient here)
    warping: &[f64],       // optimal warp gamma: [f64] (fixed, pre-computed)
    argvals: &[f64],       // evaluation grid
    weights: &[f64],       // Simpson integration weights
) -> S {
    // Step 1: Reparameterize curve2 by warping (generic linear interp)
    let m = argvals.len();
    let f2_aligned: Vec<S> = (0..m)
        .map(|j| generic_linear_interp(argvals, curve2, warping[j]))
        .collect();
    // Step 2: Compute SRSF of aligned curve2
    let q2_aligned = generic_srsf_central_diff(&f2_aligned, argvals);
    // Step 3: Weighted L2 distance
    generic_l2_srsf_distance(q1_ref, &q2_aligned, weights)
}
```

### Pattern 3: FPCA Score Projection Generic (DIF-03)

**Recommended signature:**
```rust
/// Project a single generic curve onto the FPCA loading space, returning ncomp scores.
///
/// `mean`, `rotation` (m×ncomp FdMatrix), and `weights` come from a trained
/// `FpcaResult` and stay as f64. Only `curve` is generic — the gradient of each
/// score w.r.t. `curve[j]` flows through when `S = Dual`.
///
/// The gradient of score_k w.r.t. curve[j] is exactly `rotation[(j,k)] * weights[j]`
/// (a constant — this is a linear function of the input).
pub fn project_scores_generic<S: Scalar>(
    curve: &[S],
    mean: &[f64],
    rotation: &FdMatrix,  // m x ncomp, column-major
    weights: &[f64],
    ncomp: usize,
) -> Vec<S> {
    let m = curve.len();
    let mut scores = vec![S::zero(); ncomp];
    for k in 0..ncomp {
        let mut sum = S::zero();
        for j in 0..m {
            let centered = curve[j] - S::from_f64(mean[j]);
            let w_rot = S::from_f64(rotation[(j, k)] * weights[j]);
            sum += centered * w_rot;
        }
        scores[k] = sum;
    }
    scores
}
```

**Note:** This is a free function, not a method on `FpcaResult` — because `FpcaResult::project` returns `FdMatrix` (Vec<f64>), which cannot hold `S` values. The free function takes the relevant fields from `FpcaResult` directly. A convenience wrapper is:

```rust
impl FpcaResult {
    /// Generic version: project a `&[S]` slice onto FPCA loadings.
    /// Accepts any type implementing `Scalar`, including `Dual` for AD.
    pub fn project_generic<S: crate::autodiff::Scalar>(
        &self,
        curve: &[S],
    ) -> Vec<S> {
        project_scores_generic(curve, &self.mean, &self.rotation, &self.weights, self.rotation.ncols())
    }
}
```

Either form is acceptable. The standalone free function is simpler for tests and the method form is more ergonomic for users.

### Anti-Patterns to Avoid

- **Genericizing `srsf_single` or `srsf_transform`:** These take `&FdMatrix` (Vec<f64>) and return `Vec<f64>`. Making them generic would require a generic `FdMatrix<S>`, which is out of scope. Use the central-difference approximation in `generic_srsf_central_diff` instead for the differentiable path.
- **Using `f64::min(a, b)` in generic code:** Use `if a <= b { a } else { b }` style — `PartialOrd` is the Scalar bound, and `f64::min` is not a trait method.
- **Filling `Vec<S>` with a constant via `.fill()`:** `Vec::fill` is available for `S: Clone` (which `Scalar: Clone` provides). But safer is `vec![S::infinity(); m+1]` initializer, which works for `S: Clone`.
- **Using `vec![0.0; m]` in generic code:** Use `vec![S::zero(); m]` instead.
- **`FdMatrix::zeros()` for generic accumulators:** `FdMatrix::zeros` is hardwired to `Vec<f64>`. Generic score accumulation lives in `Vec<S>` with explicit `S::zero()` initialization.
- **Differentiating through the DP warp search:** The path indices are integers. The cost accumulated at each cell can carry a tangent, but the gamma output of `dp_path_to_gamma` is piecewise-constant in the inputs — the tangent of gamma values would be zero almost everywhere. Do not attempt to genericize `dp_alignment_core_banded`.
- **Avoiding SRSF non-smoothness at f'=0:** In test data, ensure derivatives of test curves are strictly nonzero everywhere. Phase-shifted sinusoids are fine; avoid stationary-point curves (e.g., constant curves). The subdifferential convention in `Dual::abs`/`Dual::signum` handles this correctly at runtime but FD cross-checks may show jumps — document this in tests.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gradient of soft-DTW | Bespoke gradient via `soft_dtw_backward` pattern | `Dual` instantiation of `soft_dtw_distance_inner` | Operator overloading composes automatically; backward pass is only needed for the barycenter optimizer |
| Piecewise-linear interpolation for generic curve | Custom cubic spline in S | `generic_linear_interp` (linear in curve values) | Linear interp gradient is trivial; splines require additional ops not in `Scalar` |
| SRSF derivative | Port `deriv_1d` to generic S | `generic_srsf_central_diff` (central differences) | `deriv_1d` uses `FdMatrix` (f64); central differences are a clean 3-point formula in S |
| f64 constants in generic expressions | Bare float literals (`1.0`, `2.0`, etc.) | `S::from_f64(1.0)` | Bare floats cause type inference failures in generic `S: Scalar` contexts |

---

## Common Pitfalls

### Pitfall 1: `softmin3` infinity guard — `is_finite` not in Scalar

**What goes wrong:** `softmin3` guards early return on `!min_val.is_finite()`. `Scalar` has no `is_finite()` method.
**Why it happens:** `is_finite` is a method on `f64` but not in the trait.
**How to avoid:** Use `min_val >= S::infinity()` as the guard. Dual's value-only PartialOrd makes `Dual { value: f64::INFINITY, .. } >= Dual { value: f64::INFINITY, .. }` return true correctly. Document this in the generic function's comments.
**Warning signs:** Type error `no method named is_finite found for type S`.

### Pitfall 2: SRSF non-smoothness at derivative zeros

**What goes wrong:** `q(t) = sign(f'(t)) * sqrt(|f'(t)|)` has a square-root singularity at `f'(t) = 0`. `Dual::sqrt` at value=0 produces `tangent = 1/(2*0) = Inf`.
**Why it happens:** Mathematical domain constraint of sqrt.
**How to avoid:** Test curves must have strictly nonzero derivatives everywhere. For sinusoids `sin(2*pi*t)`, ensure the evaluation grid avoids t=0.25 and t=0.75 (zeros of the derivative). Use curves like `exp(-t) * sin(3*pi*t)` evaluated on `[0.1, 0.9]` to stay away from derivative zeros.
**Warning signs:** `Inf` or `NaN` in Dual tangents for SRSF-based tests.

### Pitfall 3: Low-rank test data for FPCA gradient

**What goes wrong:** FPCA gradient tests silently pass (or give trivially correct zero gradients) when test curves are all sine waves of the same frequency — these span only a 2-dimensional subspace, and the gradient of the k-th score w.r.t. curve values that are entirely in the null space of rotation column k is zero.
**Why it happens:** Project memory flag: "synthetic β(t)-recovery tests silently fail on a low-rank predictor".
**How to avoid:** Use spanning, well-conditioned test curves. Recommended: `curve[j] = a_1 * sin(pi*t[j]) + a_2 * sin(2*pi*t[j]) + a_3 * cos(pi*t[j]) + noise` with coefficients `a_i` varying across the curve set. For the gradient test of a single curve, seed the curve values independently: `curve[j] = Dual::seed(j-th value)` in turn (m separate forward passes). The gradient w.r.t. `curve[j]` is `rotation[(j,k)] * weights[j]` — this is non-zero as long as the rotation has a significant loading at point j.
**Warning signs:** All m gradient components are zero or identical — implies a degenerate rotation column.

### Pitfall 4: Delegation changes `soft_dtw_distance` f64 path

**What goes wrong:** If `soft_dtw_distance` delegates to the generic core, the `if/else` min logic replaces `f64::min`. For NaN inputs, `f64::min(NaN, b) = b` but `if NaN <= b { NaN } else { b }` returns `b` (because `NaN <= b` is false). Identical behavior for non-NaN inputs.
**Why it happens:** IEEE 754 NaN comparison semantics.
**How to avoid:** The soft-DTW inputs are time series values — NaN inputs are not expected. The f64-parity test (SC #2) will confirm bit-identical output on non-NaN inputs. Document NaN behavior as undefined (same as existing code, which does not guard against NaN either).
**Warning signs:** f64 parity test fails — if so, abandon delegation and keep both paths separate.

### Pitfall 5: `fill` vs explicit loop for `Vec<S>` initialization

**What goes wrong:** `curr.fill(S::infinity())` requires `S: Clone` — which `Scalar` does bound. However, the `fill` method signature on `Vec<T>` requires `T: Clone`. This works, but `curr.fill(f64::INFINITY)` (the original code) cannot be used in generic code.
**Why it happens:** `f64::INFINITY` is not `S`.
**How to avoid:** Replace `curr.fill(f64::INFINITY)` with `for v in curr.iter_mut() { *v = S::infinity(); }` or rebuild the vec: `curr = vec![S::infinity(); m+1]`. Either works. The `for` loop avoids the `Clone` bound concern entirely (assignment is `Copy`, and `Scalar: Copy`).

### Pitfall 6: Clippy `--all-targets` on generic test code

**What goes wrong:** CI runs `cargo clippy --all-targets --features linalg,parallel -- -D warnings`. Generic code can produce unused-variable warnings for type parameters if the test coverage is incomplete.
**Why it happens:** CI memory (see CLAUDE.md / project memory).
**How to avoid:** Run the full clippy gate locally before committing. Add `#[allow(unused_variables)]` only with justification.

---

## Validation Architecture

### Test Placement

All tests are inline `#[cfg(test)] mod tests { ... }` in the module where the function lives (crate convention: `matrix.rs`, `regression.rs`, `soft_dtw.rs` pattern).

### SC #1 — Dual gradient of soft-DTW vs hand-written oracle (DIF-02a)

**Target:** `soft_dtw_distance_generic::<Dual>` gradient w.r.t. `x[k]` must match `soft_dtw_accumulate_gradient(bary=x, xi=y, gamma, grad)` output for each k.

**Test construction:**
- Use x = bary = `[0.1, 0.4, 0.9, 1.2, 0.7]` (5 points, non-degenerate), y = `[0.2, 0.3, 1.0, 1.1, 0.6]`, gamma = 1.0.
- Compute oracle gradient: call `soft_dtw_accumulate_gradient(&x, &y, gamma, &mut grad_oracle)`.
- Compute Dual gradient for each k: seed `x[k]` at 1.0, all others at 0.0 (constant), run `soft_dtw_distance_inner(&x_dual, &y_const, gamma)`, extract tangent.
- Tolerance: ≤ 1e-8 (tighter than FD since the oracle is an exact backward pass, not a numerical approximation).
- Curve properties: well-conditioned 5-point sequences spanning multiple DP paths.

**Note on y:** The oracle computes grad w.r.t. bary (first argument). Set bary=x, xi=y. The Dual gradient seeds x values, treating y as constants. This matches the oracle exactly.

### SC #2 — f64 parity for soft-DTW (DIF-02a)

**Target:** `soft_dtw_distance_generic::<f64>(x, y, gamma) == soft_dtw_distance(x, y, gamma)` (bit-identical if delegating, or ≤ 1e-12 if companion).
- Use the same test vectors as SC #1.
- Additional cases: longer series (m=20, m=50), gamma=0.1 and gamma=10.0.
- Tolerance: 0 ULP (if delegating) or ≤ 1e-12.

### SC #3 — Dual gradient of soft-DTW vs central finite differences (DIF-02a)

**Target:** Dual gradient of `soft_dtw_distance_generic` vs central FD, tolerance ≤ 1e-6.
- h = 1e-8, central: `(f(x+h*e_k) - f(x-h*e_k)) / (2h)` where `e_k` is the k-th unit vector.
- Use 5-point series, 3 gamma values (0.5, 1.0, 2.0), gradient for all 5 x coordinates.

### SC #4 — Dual gradient of fixed-warp amplitude distance vs FD (DIF-02b)

**Target:** `amplitude_distance_at_warp_generic::<Dual>` gradient w.r.t. `curve2[j]` vs central FD.
- Construct: q1_ref = SRSF of sin(2*pi*t) for t in [0.1, 0.9] (20 points, derivative nonzero everywhere). curve2 = cos(2*pi*t) + 0.1. warping = identity (gamma(t) = t). weights = simpsons_weights(&argvals).
- Compute Dual gradient: seed curve2[j] for each j.
- Compute FD: perturb curve2[j] by ±h, call the generic function at f64.
- Tolerance: ≤ 1e-5 (slightly relaxed because the SRSF central-diff introduces a h^2 consistency error on top of FD's h^2 error — the compound numerical error is larger).
- Avoid SRSF derivative zeros: ensure test curves have derivatives > 0.01 everywhere on the grid.

### SC #5 — Dual gradient of FPCA scores vs FD (DIF-03)

**Target:** `project_scores_generic::<Dual>` gradient w.r.t. `curve[j]` vs central FD.
- Construct FPCA: generate n=15, m=40 spanning curves (sin + cos + linear combination, distinct coefficients). Run `fdata_to_pc_1d` for ncomp=3.
- Test curve: a curve NOT in the training set (e.g., `curve[j] = 0.7*sin(3*pi*t[j]) + 0.4*cos(pi*t[j])`).
- Compute Dual gradient: seed `curve[j]` for each j (m=40 forward passes), for each component k — giving a 40×3 Jacobian.
- Compute FD: same structure.
- Tolerance: ≤ 1e-6.
- **Analytical check:** The exact gradient of score_k w.r.t. curve[j] is `rotation[(j,k)] * weights[j]`. Include a separate test asserting Dual gradient matches this analytical formula to ≤ 1e-12. This is the strongest possible check for a linear function.

### SC #6 — f64 parity for FPCA scores (DIF-03)

**Target:** `project_scores_generic::<f64>(curve, ...)` reproduces `fpca.project(&data)` scores within ≤ 1e-12.
- Same FPCA, same test curve materialized as `Vec<f64>`.

### SC #7 — f64 parity for fixed-warp amplitude distance (DIF-02b)

**Target:** `amplitude_distance_at_warp_generic::<f64>` vs manually computed `l2_distance(q1_ref, q2_aligned_f64, weights)` where q2_aligned is computed with the f64 path.
- Tolerance: ≤ 1e-12 (noting that the generic path uses central differences for SRSF rather than `deriv_1d` — these may differ slightly; tighten or document the comparison if they differ by more than 1e-10).

### Spanning Test-Curve Construction

Per project memory: "use non-degenerate curves". Recommended pattern for all gradient tests:

```rust
fn spanning_test_curves(n: usize, m: usize, seed: u64) -> Vec<Vec<f64>> {
    // n curves spanning a space with dimension >= 3
    // Use sin(k*pi*t) + cos(k*pi*t) combinations with coefficients from a seeded RNG
    // NOT just phase-shifted versions of a single sinusoid
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m-1) as f64).collect();
    (0..n).map(|_| {
        let a: f64 = rng.gen_range(-1.0..1.0);
        let b: f64 = rng.gen_range(-1.0..1.0);
        let c: f64 = rng.gen_range(-0.5..0.5);
        t.iter().map(|&tj| {
            a * (std::f64::consts::PI * tj).sin()
            + b * (2.0 * std::f64::consts::PI * tj).cos()
            + c * (3.0 * std::f64::consts::PI * tj).sin()
        }).collect()
    }).collect()
}
```

### Test Run Commands

| Gate | Command | Expected |
|------|---------|----------|
| Quick (per commit) | `cargo test -p fdars-core autodiff soft_dtw project_scores` | All new tests green |
| Full suite | `cargo test -p fdars-core --features linalg,parallel` | No regressions |
| Clippy gate | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Zero warnings |
| Fmt gate | `cargo fmt --check` | No drift |
| Doc test | `cargo test --doc -p fdars-core` | Module doctest green |

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|-----------------|--------|
| Hand-written backward pass (`soft_dtw_backward`) for each function separately | Generic `S: Scalar` forward pass — Dual propagates tangents automatically | Gradients compose; new functions get AD "for free" |
| `elastic_distance` = opaque f64 black box | `amplitude_distance_at_warp_generic` = differentiable w.r.t. second curve values (fixed warp) | Enables gradient-based optimization of curve shape toward a target distance |
| `FpcaResult::project` = f64-only, no gradient | `project_scores_generic` = forward-mode Jacobian via m seeded passes | FPC scores can flow into any differentiable objective |

**Deprecated/outdated patterns for this phase:**
- Hand-written gradient oracle (`soft_dtw_accumulate_gradient`): not replaced — it remains the SC #1 validation oracle. Its continued existence is valuable as a check.
- `soft_dtw_forward` / `soft_dtw_backward`: not replaced — they power the barycenter optimizer. The generic Dual path is additive alongside.

---

## Environment Availability

This phase has no external dependencies beyond the Rust toolchain (same as Phase 75).

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | cargo build/test | Yes | 1.97.0 | — |
| cargo clippy | CI gate | Yes | bundled | — |
| cargo fmt | fmt gate | Yes | bundled | — |

---

## Validation Architecture (Nyquist)

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` harness |
| Config file | None (inline tests per module) |
| Quick run command | `cargo test -p fdars-core --lib autodiff soft_dtw project_scores` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DIF-02 | Dual gradient of soft-DTW == oracle (SC #1) | unit | `cargo test -p fdars-core soft_dtw::tests::dual_gradient_vs_oracle` | No — Wave 0 |
| DIF-02 | f64 parity for soft-DTW (SC #2) | unit | `cargo test -p fdars-core soft_dtw::tests::f64_parity` | No — Wave 0 |
| DIF-02 | Dual gradient of soft-DTW vs FD (SC #3) | unit | `cargo test -p fdars-core soft_dtw::tests::dual_gradient_vs_fd` | No — Wave 0 |
| DIF-02 | Dual gradient of fixed-warp amplitude distance vs FD (SC #4) | unit | `cargo test -p fdars-core alignment::differentiable::tests::amplitude_gradient_vs_fd` | No — Wave 0 |
| DIF-03 | Dual gradient of FPCA scores vs analytical + FD (SC #5) | unit | `cargo test -p fdars-core regression::tests::fpca_score_gradient_dual` | No — Wave 0 |
| DIF-03 | f64 parity for FPCA scores (SC #6) | unit | `cargo test -p fdars-core regression::tests::fpca_score_generic_f64_parity` | No — Wave 0 |
| DIF-02 | f64 parity for fixed-warp amplitude distance (SC #7) | unit | `cargo test -p fdars-core alignment::differentiable::tests::amplitude_f64_parity` | No — Wave 0 |
| DIF-02+03 | Full crate remains green after additions | regression | `cargo test -p fdars-core --features linalg,parallel` | Yes |

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core autodiff soft_dtw project_scores`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `fdars-core/src/metric/soft_dtw.rs` — add `softmin3_generic`, `soft_dtw_distance_inner`, `soft_dtw_distance_generic` (tests inline)
- [ ] `fdars-core/src/alignment/differentiable.rs` — new file: `generic_linear_interp`, `generic_srsf_central_diff`, `generic_l2_srsf_distance`, `amplitude_distance_at_warp_generic` (tests inline)
- [ ] `fdars-core/src/regression.rs` — add `project_scores_generic` (tests inline)
- [ ] `fdars-core/src/alignment/mod.rs` — add `mod differentiable; pub use differentiable::amplitude_distance_at_warp_generic;`
- [ ] `fdars-core/src/lib.rs` — add minimal re-exports for the 3 new public functions

---

## Security Domain

| ASVS Category | Applies | Control |
|---------------|---------|---------|
| V5 Input Validation | Minimal — domain checks (positive for sqrt/ln) documented | Document in function docs |
| V2–V4 Auth/Session/Access | No | N/A |
| V6 Cryptography | No | N/A |

No significant security surface. NaN/Inf propagation on out-of-domain inputs (SRSF at f'=0) is documented behavior — same as existing f64 code.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Central finite differences for the SRSF derivative (`generic_srsf_central_diff`) introduce a compound numerical error of ~h^2, making the FD cross-check tolerance for SC #4 need relaxing to ≤1e-5 rather than ≤1e-6 | Validation Architecture §SC #4 | If error is smaller, the test is overly permissive (not harmful); if larger, SC #4 fails — tighten or document |
| A2 | `soft_dtw_distance` delegating to `soft_dtw_distance_inner<f64>` produces bit-identical output to the existing implementation (the only difference is `if a <= b { a } else { b }` vs `a.min(b)` for non-NaN inputs) | Pattern 1, f64-parity risk | If FP rounding differs, keep both implementations separate (non-delegating companion) |
| A3 | The fixed-warp deliverable for DIF-02b (gradient w.r.t. curve2 values with gamma fixed) is the intended semantic for "differentiable amplitude distance" in practice | Recommended Deliverable Scope | If the user needs gradient w.r.t. the warping path itself, a different formulation (soft-DTW or a relaxed aligner) is needed |
| A4 | The existing `deriv_1d` function (used in `srsf_transform`) gives results within ~h^2 of the central-difference formula in `generic_srsf_central_diff`; this difference is ≤1e-4 for typical test curves | SC #7 parity tolerance | If the discrepancy exceeds 1e-10, the fixed-warp f64 parity test (SC #7) needs a looser tolerance and a note explaining the derivative-method difference |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed. Table is NOT empty; A1–A4 should be checked at execution time.

---

## Open Questions

1. **Should `soft_dtw_distance` delegate to the generic core?**
   - What we know: delegation gives bit-identical f64 output if `if/else` min matches `f64::min` for non-NaN inputs.
   - What's unclear: whether any call site passes NaN or -Infinity values that would expose the behavioral difference.
   - Recommendation: delegate, with an explicit SC #2 parity test; if SC #2 fails, revert to a pure companion pattern.

2. **Where to place `amplitude_distance_at_warp_generic` — new `differentiable.rs` file or inline in `pairwise.rs`?**
   - What we know: `pairwise.rs` is already 686 lines; the new generic helpers are ~80 lines total.
   - Recommendation: new `src/alignment/differentiable.rs` file to keep `pairwise.rs` clean and signal that this is a differentiable-path module. Register it in `alignment/mod.rs`.

3. **`FpcaResult::project_generic` as a method vs standalone `project_scores_generic`?**
   - What we know: methods on `FpcaResult` are idiomatic; standalone function is simpler for tests. The method would need `use crate::autodiff::Scalar` inside `regression.rs`.
   - Recommendation: both — implement the standalone free function and a thin method wrapper. The standalone function is what tests use; the method is the user-facing ergonomic API (even though Phase 77 owns full prelude surfacing, the method being available doesn't hurt).

---

## Sources

### Primary (HIGH confidence — verified by file reads this session)
- `fdars-core/src/autodiff.rs:59-105` — full `Scalar` trait definition and method set
- `fdars-core/src/autodiff.rs:175-417` — `Dual` struct, all operator impls, `PartialOrd`, `Scalar for Dual`
- `fdars-core/src/metric/soft_dtw.rs:29-249` — `softmin3`, `soft_dtw_distance` DP recurrence, `soft_dtw_accumulate_gradient` oracle
- `fdars-core/src/alignment/pairwise.rs:103-105, 129-175, 383-392` — `elastic_distance`, `elastic_align_pair_from_srsf`, `amplitude_distance`, `phase_distance_pair`
- `fdars-core/src/alignment/mod.rs:564-623` — `dp_alignment_core`, `dp_alignment_core_banded` (DP warp search structure)
- `fdars-core/src/alignment/srsf.rs:36-114` — `srsf_transform`, `srsf_single` implementations
- `fdars-core/src/regression.rs:49-62, 105-127` — `FpcaResult` fields, `FpcaResult::project` inner loop
- `fdars-core/src/helpers.rs:56-63` — `l2_distance` implementation
- `fdars-core/src/warping.rs:1-179` — SRSF-to-psi conversion, `phase_distance`
- `.planning/phases/76-differentiable-elastic-distance-fpca-scores/76-CONTEXT.md` — locked decisions
- `.planning/phases/75-scalar-trait-forward-mode-dual-substrate/75-RESEARCH.md` — Phase 75 substrate research
- `.planning/REQUIREMENTS.md` — DIF-02, DIF-03 requirements

### No external sources consulted
All findings are derived from direct file reads of the in-crate source. No web searches were needed — the phase is entirely about adapting existing in-crate code to a generic trait.

---

## Metadata

**Confidence breakdown:**
- Scalar trait completeness (gap analysis): HIGH — read autodiff.rs:59-105 verbatim
- Soft-DTW DP recurrence and gradient oracle: HIGH — read soft_dtw.rs:29-249 verbatim
- FPCA score projection arithmetic: HIGH — read regression.rs:117-125 verbatim
- DP warp search non-differentiability: HIGH — read mod.rs:564-623, confirmed discrete argmin structure
- SRSF transform formula and differentiability: HIGH — read srsf.rs:36-52 verbatim
- Generic implementation signatures (Pattern sections): MEDIUM — derived from verified source reads + AD chain rules (chain rules are textbook)
- SC tolerance values (A1–A4): LOW-MEDIUM — compound numerical error analysis is [ASSUMED]

**Research date:** 2026-09-06
**Valid until:** 2026-10-06 (codebase stable; AD math is timeless)
