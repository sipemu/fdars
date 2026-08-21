# Phase 35: Basis-System Completions — Research

**Researched:** 2026-08-21
**Domain:** Functional data basis systems, linear differential operators, multi-domain containers, principal differential analysis
**Confidence:** MEDIUM

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Basis factory API & return type**
- New factories return `Result<BasisSystem, FdarError>` (a new struct bundling column-major eval matrix + penalty matrix + metadata), per ROADMAP SC1.
- Signatures: `monomial_basis(argvals, nbasis)`, `exponential_basis(argvals, rates: &[f64])`, `power_basis(argvals, exponents: &[f64])`, `polygonal_basis(argvals, knots)`.
- Eval-matrix layout: column-major flat `Vec<f64>` of shape `(n × nbasis)`, matching `bspline_basis`'s `basis[ti + j*n]` convention.
- Existing `bspline`/`fourier`/`constant` factories LEFT UNTOUCHED.

**Penalty matrices**
- Analytic penalty where a closed form exists (monomial/power: Gram of the `lfd_order`-th derivative); otherwise numeric Gram mirroring `smooth_basis::fourier_penalty_matrix` / `differentiate_basis_columns`.
- Default penalty is 2nd-derivative (curvature) roughness; expose `lfd_order` parameter.
- Penalty carried INSIDE the `BasisSystem` result.
- Polygonal uses a 1st-order roughness penalty (2nd derivative is 0 a.e. for piecewise-linear), documented in rustdoc.

**MultiFunData container**
- New `fdars-core/src/multi_fdata.rs`: `MultiFunData { components: Vec<FdComponent> }` where each `FdComponent` holds an `FdMatrix` plus its own `argvals`.
- Invariant: equal observation count (rows) across all components; each component's argvals length must match its `FdMatrix` column count.
- Constructor `MultiFunData::new(components) -> Result<Self, FdarError>` validates both invariants.
- Accessors: `n_obs()`, `n_components()`, `component(k)`, `argvals(k)`.

**Lfd object & PDA**
- `Lfd { coefs: Vec<Vec<f64>> }` — grid-sampled weight functions β₀(t)…β_{m-1}(t). Method `apply(data, argvals) -> Result<FdMatrix, FdarError>`.
- `principal_differential_analysis(data, argvals, order) -> Result<PdaResult, FdarError>` — pointwise LS recovering βₖ(t).
- `PdaResult { coefficients: Vec<Vec<f64>>, ... }`.
- Test: PDA recovers harmonic oscillator x'' = −ω²x → β₀(t) ≈ ω², β₁(t) ≈ 0 within tolerance.

### Claude's Discretion
- Exact struct/field names, `BasisSystem`/`FdComponent`/`PdaResult` shapes, finite-difference derivative scheme, and numeric-Gram grid density are at planner/executor discretion, guided by `smooth_basis.rs`/`pspline.rs` conventions and the `fda` reference.

### Deferred Ideas (OUT OF SCOPE)
- Full `tidyfun`-style tidy multi-representation vector semantics beyond `MultiFunData` — REP-02, deferred.
- Retrofitting existing `bspline`/`fourier`/`constant` factories to Result-returning — out of scope (breaking).
- General non-linear differential operators / PDA with basis-expanded coefficients beyond pointwise-LS.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REP-01 | Complete basis-system family: monomial/exponential/power/polygonal factories with penalty matrices; MultiFunData container; Lfd operator; PDA estimator. Additive/non-breaking. | Closed-form evaluation formulas (§Basis Definitions), analytic penalty formulas (§Penalty Matrices), MultiFunData invariants (§MultiFunData), Lfd/PDA formulation (§Lfd & PDA), test reference values (§Test Strategy) |

</phase_requirements>

---

## Summary

Phase 35 adds four basis factories (`monomial_basis`, `exponential_basis`, `power_basis`, `polygonal_basis`), each returning a `BasisSystem` struct bundling a column-major evaluation matrix and a penalty matrix. It also introduces `MultiFunData` (a multi-domain container), `Lfd` (a linear differential operator object), and `principal_differential_analysis` (PDA, a linear-ODE estimator). All additions are purely additive — zero existing public signatures change.

The most implementation-complex pieces are (1) the analytic penalty matrix for monomial/power bases (a closed-form polynomial Gram matrix computable without quadrature), (2) the Lfd `apply` method (repeated finite-difference differentiation via the existing `gradient`/`gradient_uniform` from `helpers.rs`), and (3) the PDA pointwise normal-equation solver (independent least-squares at each time point, with a singular-design guard). The `MultiFunData` container is straightforward once the two invariants (matched row counts, matched argvals lengths) are enforced in the constructor.

Every component has hand-computable reference values for tests: monomial evaluates to powers at t=0,1,2; exponential is exp(rate·t); polygonal produces hat-function values at knot midpoints; PDA recovers ω² from harmonic-oscillator trajectories. No new crate dependency is required.

**Primary recommendation:** Organise into four plans: (1) basis factories + `BasisSystem` struct, (2) `MultiFunData` container, (3) `Lfd` object, (4) `PdaResult` + `principal_differential_analysis`. Plans 2–4 are independent of each other; plan 1 (`BasisSystem`) is a prerequisite for nothing else in this phase.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Basis evaluation matrices | `basis/` module | — | Direct extension of existing bspline/fourier/constant factory pattern |
| Penalty matrices (analytic) | `basis/` module | `helpers.rs` (quadrature fallback) | Penalty lives inside `BasisSystem`; analytic formula replaces quadrature for monomial/power |
| Multi-domain container | `src/multi_fdata.rs` (new) | `matrix.rs` (FdMatrix) | New top-level module following the `fdata.rs` / `fts/` convention |
| Linear differential operator | `src/basis/` or new `src/pda.rs` | `helpers.rs` (gradient) | Lfd consumes derivative functions already in helpers; placement at planner discretion |
| PDA estimator | Same module as Lfd | `matrix.rs`, `helpers.rs` | PDA is a direct consumer of Lfd + derivative infrastructure |
| Crate-root re-exports | `src/lib.rs` | `basis/mod.rs` | All new public items follow the established `pub use basis::{...}` pattern |

---

## Standard Stack

### Core — No New Crates

| Component | Source | Purpose |
|-----------|--------|---------|
| `nalgebra::DMatrix` | existing dep | Penalty matrix construction, normal-equation solve in PDA |
| `crate::helpers::{gradient_uniform, gradient, simpsons_weights, trapz}` | existing `helpers.rs` | Finite-difference derivatives, quadrature for numeric penalty fallback |
| `crate::matrix::FdMatrix` | existing `matrix.rs` | Evaluation matrices and component storage |
| `crate::smooth_basis::differentiate_basis_columns` | existing `smooth_basis.rs` (internal) | Numeric Gram penalty pattern to copy for exponential/polygonal |
| `crate::smooth_basis::integrate_symmetric_penalty` | existing `smooth_basis.rs` (internal) | Symmetric quadrature sum — copy pattern for numeric penalties |
| `crate::error::FdarError` | existing `error.rs` | All validation returns |

**Installation:** No `cargo add` required. Zero new crate dependencies. [VERIFIED: 35-CONTEXT.md — "no new crate dependency"]

---

## Package Legitimacy Audit

No external packages are introduced by this phase. All implementation reuses existing `fdars-core` infrastructure.

| Package | Verdict | Disposition |
|---------|---------|-------------|
| (none) | — | No new packages |

---

## Architecture Patterns

### System Architecture Diagram

```
User call: monomial_basis(argvals, nbasis)
  │
  ▼
[basis/monomial.rs]
  ├─ validate: argvals.len() >= 2, nbasis >= 1, exponents distinct non-negative integers
  ├─ eval_matrix: n × nbasis column-major Vec<f64>
  │    B[ti + j*n] = argvals[ti].powi(exponents[j])
  ├─ penalty_matrix: nbasis × nbasis analytic Gram (or numeric fallback)
  └─ return Ok(BasisSystem { eval_matrix, penalty_matrix, nbasis, domain })

User call: MultiFunData::new(components)
  │
  ▼
[multi_fdata.rs]
  ├─ check: all components have same nrows (n_obs)
  ├─ check: each component.argvals.len() == component.data.ncols()
  └─ return Ok(MultiFunData { components })

User call: Lfd::apply(data, argvals)
  │
  ▼
[pda.rs or basis/lfd.rs]
  ├─ for each curve i:
  │    compute x, Dx, D²x, ..., D^m x  ← helpers::gradient (iterated)
  │    Lx = D^m x + Σ_{k=0}^{m-1} coefs[k][·] * D^k x  (pointwise)
  └─ return FdMatrix of Lx values

User call: principal_differential_analysis(data, argvals, order)
  │
  ▼
[pda.rs]
  ├─ compute derivatives D⁰x, D¹x, ..., D^m x for all curves (n × m grid)
  ├─ at each t_j: solve normal equations
  │    X_j = [x_i(t_j), Dx_i(t_j), ..., D^{m-1}x_i(t_j)]  (n × m matrix, all curves)
  │    y_j = -D^m x(t_j)  (n-vector)
  │    β(t_j) = (X_j^T X_j)^{-1} X_j^T y_j  (SVD fallback for rank-deficient)
  └─ return PdaResult { coefficients: vec![β₀(t), β₁(t), ..., β_{m-1}(t)] }
```

### Recommended Project Structure

```
fdars-core/src/
├── basis/
│   ├── mod.rs             # add: pub use monomial/exponential/power/polygonal, BasisSystem
│   ├── monomial.rs        # NEW: monomial_basis + analytic penalty
│   ├── exponential.rs     # NEW: exponential_basis + numeric penalty
│   ├── power.rs           # NEW: power_basis + analytic penalty (fractional-exponent aware)
│   ├── polygonal.rs       # NEW: polygonal_basis + 1st-order numeric penalty
│   ├── bspline.rs         # UNTOUCHED
│   ├── fourier.rs         # UNTOUCHED
│   ├── constant.rs        # UNTOUCHED
│   └── ...                # other existing files UNTOUCHED
├── multi_fdata.rs          # NEW: MultiFunData, FdComponent
├── pda.rs                  # NEW: Lfd, principal_differential_analysis, PdaResult
└── lib.rs                  # add pub mod multi_fdata; pub mod pda; pub use ...
```

---

## Basis Function Definitions

### Pattern 1: Monomial Basis

**What:** `B_j(t) = t^{e_j}` where `e_j` are distinct non-negative integers (defaults to 0, 1, …, nbasis-1). [CITED: search.r-project.org/CRAN/refmans/fda/html/create.monomial.basis.html]

**Evaluation formula (exact):**
```rust
// Source: R fda create.monomial.basis convention
// eval_matrix[ti + j * n] = argvals[ti].powi(exponents[j] as i32)
for (ti, &t) in argvals.iter().enumerate() {
    for j in 0..nbasis {
        eval_matrix[ti + j * n] = t.powi(exponents[j] as i32);
    }
}
```

**Reference test values (hand-computable):**
- `t = [0.0, 1.0, 2.0]`, `nbasis = 3`, `exponents = [0, 1, 2]`
- Column 0 (B₀ = 1): `[1.0, 1.0, 1.0]`
- Column 1 (B₁ = t): `[0.0, 1.0, 2.0]`
- Column 2 (B₂ = t²): `[0.0, 1.0, 4.0]`

**Penalty matrix (analytic, preferred):**
The `d`-th derivative of `t^{e}` is `c(e,d) · t^{e-d}` where `c(e,d) = e! / (e-d)!` (falling factorial) for `e >= d`, else 0.

The penalty Gram entry R[i,j] with `lfd_order = d`, domain `[a, b]`: [ASSUMED — standard polynomial calculus, not verified against fda source]

```
c_i = falling_factorial(e_i, d)   // = e_i * (e_i-1) * ... * (e_i - d + 1)
c_j = falling_factorial(e_j, d)

if c_i == 0 or c_j == 0:
    R[i,j] = 0
else:
    power = e_i + e_j - 2*d + 1
    R[i,j] = c_i * c_j * (b.powi(power) - a.powi(power)) / power as f64
```

For the standard domain `[0, 1]`, `a = 0`, so `R[i,j] = c_i * c_j / (e_i + e_j - 2d + 1)`.

**Validation:** `d=2`, `e = [0,1,2]`, domain `[0,1]`:
- R[0,0] = 0 (c₀ = 0, e=0 < d=2)
- R[1,1] = 0 (e=1 < d=2)
- R[2,2] = 2·2 / (2+2-4+1) = 4/1 = 4.0
- R[2,3] = 2·6/(2+3-4+1) = 12/2 = 6.0 for nbasis=4, e=3

### Pattern 2: Exponential Basis

**What:** `B_j(t) = exp(rates[j] · t)`. When `rates[j] = 0`, `B_j(t) = 1` (constant). [CITED: search.r-project.org/CRAN/refmans/fda/html/create.exponential.basis.html]

**Evaluation formula (exact):**
```rust
// Source: R fda create.exponential.basis convention (exp(ratevec[i]*x))
for (ti, &t) in argvals.iter().enumerate() {
    for j in 0..nbasis {
        eval_matrix[ti + j * n] = (rates[j] * t).exp();
    }
}
```

**Reference test values:**
- `t = [0.0, 1.0]`, `rates = [0.0, -1.0, -5.0]`
- Column 0: `[1.0, 1.0]` (exp(0))
- Column 1: `[1.0, exp(-1.0)]` ≈ `[1.0, 0.3679]`
- Column 2: `[1.0, exp(-5.0)]` ≈ `[1.0, 0.00674]`

**Penalty matrix:** Numeric Gram (analytic possible but not closed-form simple for mixed rates). Use the `differentiate_basis_columns` + `integrate_symmetric_penalty` pattern from `smooth_basis.rs`. The `d`-th derivative of `exp(r·t)` is `r^d · exp(r·t)`, so an analytic formula for the Gram exists: [ASSUMED]

```
R_analytic[i,j] = (rates[i])^d * (rates[j])^d * ∫_a^b exp((rates[i]+rates[j])·t) dt

If rates[i]+rates[j] ≠ 0:
    = r_i^d * r_j^d * (exp((r_i+r_j)*b) - exp((r_i+r_j)*a)) / (r_i+r_j)
If rates[i]+rates[j] = 0:
    = r_i^d * r_j^d * (b - a)
```

**Recommendation:** Implement as numeric Gram (simpler, consistent with existing fourier penalty pattern) unless the planner opts for analytic. Document which approach is used in rustdoc.

### Pattern 3: Power Basis

**What:** `B_j(t) = t^{exponents[j]}` where exponents can be non-integer or negative. **Domain constraint: argvals must be strictly positive** (no zero or negative values when any exponent is non-integer or negative). [CITED: search.r-project.org/CRAN/refmans/fda/html/create.power.basis.html]

**Evaluation formula:**
```rust
// Source: R fda create.power.basis convention
for (ti, &t) in argvals.iter().enumerate() {
    for j in 0..nbasis {
        eval_matrix[ti + j * n] = t.powf(exponents[j]);
    }
}
```

**Validation:** Emit `FdarError::InvalidParameter` if any `argvals[i] <= 0.0` when any `exponents[j]` is non-integer or negative. When all exponents are non-negative integers, zero is tolerated (mirrors monomial).

**Penalty matrix:** Same analytic Gram formula as monomial when exponents are integers. For non-integer exponents, use numeric Gram. [ASSUMED — extend monomial formula to non-integer via `powf`]:

```rust
// For non-integer exponents, falling-factorial generalization:
// d/dt^1 (t^e) = e * t^(e-1); iterated => D^d(t^e) = e*(e-1)*...*(e-d+1) * t^(e-d)
// The coefficient is still the falling factorial, but now it's a product of floats.
// Gram integral: same closed form holds as long as e_i + e_j - 2d + 1 > -1 (i.e., integrable).
```

**Reference test values** (domain `[1, 2]`, exponents `[0.5, 1.5]`):
- B₀(1.5) = 1.5^0.5 ≈ 1.2247
- B₁(1.5) = 1.5^1.5 ≈ 1.8371

### Pattern 4: Polygonal Basis (Piecewise-Linear Hat Functions)

**What:** `B_j(t)` is the piecewise-linear "hat function" that equals 1 at knot `knots[j]`, 0 at all other knots, and varies linearly between adjacent knots. Equivalent to B-spline of order 2 on the knot sequence. [CITED: search.r-project.org/CRAN/refmans/fda/html/create.polygonal.basis.html]

**Evaluation formula:**
```rust
// knots must be strictly increasing, length >= 2
// B_j(t) = hat function centered at knots[j]
// For each t, at most two basis functions are nonzero (the left and right neighbors of t's interval)

// For knot j (0-indexed), B_j(t):
// if knots[j-1] <= t <= knots[j]:   (t - knots[j-1]) / (knots[j] - knots[j-1])
// if knots[j]   <= t <= knots[j+1]: (knots[j+1] - t) / (knots[j+1] - knots[j])
// else: 0.0
// Boundary knots (j=0, j=nbasis-1) only have one ramp.
```

**Reference test values:**
- `knots = [0.0, 0.5, 1.0]`, 3 basis functions
- At `t = 0.25` (midpoint of [0, 0.5]):
  - B₀(0.25) = (0.5 - 0.25)/0.5 = 0.5
  - B₁(0.25) = (0.25 - 0.0)/0.5 = 0.5
  - B₂(0.25) = 0.0
- At `t = 0.5`: B₀=0.0, B₁=1.0, B₂=0.0
- Partition of unity: B₀(t) + B₁(t) + B₂(t) = 1.0 for all t in [0,1]

**Penalty matrix (1st-order, analytic):**
The first derivative of each hat function is piecewise constant: +1/h on the left ramp, -1/h on the right ramp, where h is the interval width. The second derivative is 0 a.e. (delta at knots, measure zero). [ASSUMED — standard hat-function calculus]

For `lfd_order = 1` and uniform knot spacing `h`:
```
// Interior knots: D B_j = +1/h_left on [k_{j-1}, k_j], -1/h_right on [k_j, k_{j+1}]
// R[j,j] = integral of (D B_j)^2 = (1/h_left)^2 * h_left + (1/h_right)^2 * h_right
//         = 1/h_left + 1/h_right
// R[j,j±1] = integral of D B_j * D B_{j±1} = -1/h  (shared interval contributes)
```

**Recommendation:** Use numeric Gram via the `differentiate_basis_columns` + `integrate_symmetric_penalty` pattern on a fine sub-grid. This avoids the edge-case arithmetic for non-uniform knot spacing and is consistent with the B-spline penalty implementation. Document that `lfd_order >= 2` yields an all-zero penalty (2nd derivative is 0 a.e. for piecewise-linear) and that `lfd_order = 1` is the natural choice. [ASSUMED — numeric approach is the safe fallback]

**Validation:** Knots must be strictly increasing (`knots[i+1] > knots[i]` for all i) — emit `FdarError::InvalidParameter` otherwise.

---

## Penalty Matrices — Implementation Guide

### Analytic Gram (Monomial & Power with Integer Exponents)

Use this pattern — it avoids quadrature entirely: [ASSUMED — derived from standard calculus]

```rust
fn monomial_gram_entry(ei: f64, ej: f64, d: usize, a: f64, b: f64) -> f64 {
    // falling factorial c(e, d) = e * (e-1) * ... * (e-d+1)
    fn falling(e: f64, d: usize) -> f64 {
        (0..d).fold(1.0, |acc, k| acc * (e - k as f64))
    }
    let ci = falling(ei, d);
    let cj = falling(ej, d);
    if ci.abs() < 1e-15 || cj.abs() < 1e-15 { return 0.0; }
    let power = ei + ej - 2.0 * d as f64 + 1.0;
    if power.abs() < 1e-15 {
        // logarithmic case: ∫ t^(-1) dt = ln(t)
        ci * cj * (b.ln() - a.ln())
    } else {
        ci * cj * (b.powf(power) - a.powf(power)) / power
    }
}
```

Build the `nbasis × nbasis` symmetric penalty matrix by calling `monomial_gram_entry` for each pair.

### Numeric Gram (Exponential, Polygonal, Power with non-integer exponents)

Mirror `smooth_basis::bspline_penalty_matrix`: [VERIFIED: fdars-core/src/smooth_basis.rs:82-116]

```rust
// 1. Build fine quad grid (10 sub-points per original interval, or 201 uniform points)
// 2. Evaluate basis on fine grid
// 3. Call differentiate_basis_columns (exists in smooth_basis.rs as private fn)
// 4. Call integrate_symmetric_penalty (exists in smooth_basis.rs as private fn)
```

Since these helpers are private (`fn`, not `pub fn`) in `smooth_basis.rs`, the executor has two options:
- **Option A (recommended):** Copy the two helper functions into the new basis files (they are small: ~15 lines each).
- **Option B:** Make them `pub(crate)` in `smooth_basis.rs` and import. This requires a one-line change to `smooth_basis.rs` but is cleaner for reuse.

The planner should pick Option B (mark `differentiate_basis_columns` and `integrate_symmetric_penalty` as `pub(crate)`) to avoid code duplication. This is an internal change that does not affect the public API.

---

## BasisSystem Struct Design

```rust
// Source: CONTEXT.md locked decision — new struct bundling eval + penalty + metadata
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BasisSystem {
    /// Column-major evaluation matrix, shape (n_eval × nbasis).
    /// Element (t_i, j) is at index: i + j * n_eval.
    pub eval_matrix: Vec<f64>,
    /// Square penalty matrix, shape (nbasis × nbasis), column-major.
    /// R[j,k] = ∫ D^lfd_order B_j(t) · D^lfd_order B_k(t) dt
    pub penalty_matrix: Vec<f64>,
    /// Number of basis functions.
    pub nbasis: usize,
    /// Number of evaluation points (rows of eval_matrix).
    pub n_eval: usize,
    /// Roughness penalty order used to compute penalty_matrix.
    pub lfd_order: usize,
}
```

**Why `#[non_exhaustive]`:** Allows adding `domain: [f64; 2]` or `basis_type: BasisKind` in future versions without breaking downstream. Consistent with `PsplineFitResult`, `FpcaResult`, etc. [VERIFIED: fdars-core/src/basis/pspline.rs:41-63 — PsplineFitResult uses #[non_exhaustive]]

---

## MultiFunData Container

### Struct Design

```rust
// Source: CONTEXT.md locked decision
#[derive(Debug, Clone, PartialEq)]
pub struct FdComponent {
    /// Functional data matrix for this component (n_obs × m_j columns).
    pub data: FdMatrix,
    /// Evaluation points for this component (length m_j).
    pub argvals: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MultiFunData {
    components: Vec<FdComponent>,
}
```

### Invariants (from funData R package semantics)

[CITED: search.r-project.org/CRAN/refmans/funData/html/multiFunData-class.html]

1. **Shared observation count:** All components must have the same `data.nrows()`. Checked in `MultiFunData::new`.
2. **Argvals-column match:** For each component, `argvals.len() == data.ncols()`. Checked in `MultiFunData::new`.
3. **Non-empty components:** `components.len() >= 1`. Otherwise return `FdarError::InvalidParameter`.
4. **Components may have different domains:** Intentional — this is the multi-domain feature.

### Constructor Error Cases

```rust
pub fn new(components: Vec<FdComponent>) -> Result<Self, FdarError> {
    if components.is_empty() { return Err(FdarError::InvalidParameter { ... }); }
    let n_obs = components[0].data.nrows();
    for (k, c) in components.iter().enumerate() {
        if c.data.nrows() != n_obs { return Err(FdarError::InvalidDimension { ... }); }
        if c.argvals.len() != c.data.ncols() { return Err(FdarError::InvalidDimension { ... }); }
    }
    Ok(Self { components })
}
```

### Accessors

```rust
pub fn n_obs(&self) -> usize { self.components[0].data.nrows() }
pub fn n_components(&self) -> usize { self.components.len() }
pub fn component(&self, k: usize) -> Result<&FdComponent, FdarError> { ... }
pub fn argvals(&self, k: usize) -> Result<&[f64], FdarError> { ... }
```

Out-of-range index returns `FdarError::InvalidParameter`.

---

## Lfd (Linear Differential Operator)

### Definition

`Lx(t) = D^m x(t) + β_{m-1}(t) · D^{m-1}x(t) + … + β₀(t) · x(t)` [CITED: rdrr.io/cran/fda/man/Lfd.html]

The operator of order `m` holds `m` weight functions `β₀, …, β_{m-1}`, each sampled on the same grid as the data.

### Struct Design

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct Lfd {
    /// Weight functions beta_0(t), ..., beta_{m-1}(t).
    /// coefs[k] has length equal to the number of grid points.
    /// coefs.len() == order (m).
    pub coefs: Vec<Vec<f64>>,
}
```

**Constant-coefficient Lfd (special case):** A length-1 `coefs[k]` is broadcast to all grid points. The `apply` method should handle this: if `coefs[k].len() == 1`, use `coefs[k][0]` for all t.

### `apply` Method

```rust
pub fn apply(&self, data: &FdMatrix, argvals: &[f64]) -> Result<FdMatrix, FdarError> {
    let m = self.coefs.len();  // operator order
    let (n, n_pts) = data.shape();
    // Validate argvals length matches data columns
    // For each curve i:
    //   1. Extract curve x_i as Vec<f64>
    //   2. Compute D^0 x = x, D^1 x, ..., D^m x using helpers::gradient (iterated)
    //   3. Lx_i(t_j) = D^m x_i(t_j) + sum_{k=0}^{m-1} coef_k(t_j) * D^k x_i(t_j)
    // Return FdMatrix of Lx values
}
```

**Finite-difference scheme:** Use `crate::helpers::gradient` (auto-detects uniform/non-uniform grid). [VERIFIED: fdars-core/src/helpers.rs:839-855 — `gradient` dispatches to `gradient_uniform` (5-point stencil, O(h⁴)) for uniform grids or `gradient_nonuniform` (3-point Lagrange) for non-uniform]

**Iterated differentiation:** Apply `gradient` `m` times. The `differentiate_basis_columns` function in `smooth_basis.rs` does exactly this for columns of a matrix — copy the same pattern for curve rows. [VERIFIED: fdars-core/src/smooth_basis.rs:550-570]

---

## PDA (Principal Differential Analysis)

### Mathematical Formulation

Given `n` observed curves `x_1(t), …, x_n(t)` on a grid `t_1, …, t_p`, PDA estimates the order-`m` linear ODE: [CITED: arxiv.org/abs/2406.18484]

```
D^m x(t) = -β₀(t) · x(t) - β₁(t) · D x(t) - … - β_{m-1}(t) · D^{m-1} x(t)
```

**Pointwise normal equations:** At each grid point `t_j`, form:

```
X_j = [x_i(t_j), Dx_i(t_j), ..., D^{m-1}x_i(t_j)]   n × m design matrix
y_j = -[D^m x_i(t_j)]                                  n-vector

β(t_j) = (X_j^T X_j)^{-1} X_j^T y_j
```

This is independent least-squares at each time point.

**Implementation pattern:**

```rust
pub fn principal_differential_analysis(
    data: &FdMatrix,
    argvals: &[f64],
    order: usize,
) -> Result<PdaResult, FdarError> {
    let (n, n_pts) = data.shape();
    // 1. Compute all derivatives: derivs[k] is FdMatrix of D^k x for k=0..=order
    // 2. For each t_j (0..n_pts):
    //    a. Build X_j (n × order): col k = derivs[k].column(j) for k=0..order-1
    //    b. Build y_j (n-vec): derivs[order].column(j), negated
    //    c. Solve X_j^T X_j β = X_j^T y_j via SVD (nalgebra::SVD)
    //    d. Guard: if all singular values < threshold (1e-10 * max), set β = 0
    // 3. Collect coefficients: result.coefficients[k][j] = β_k(t_j)
    Ok(PdaResult { coefficients, ... })
}
```

**Singular design guard:** When `n < order` (fewer curves than ODE order), `X_j` is rank-deficient. Use `nalgebra::SVD` pseudoinverse with threshold `1e-10 * max_singular_value`. Return `FdarError::InvalidDimension` if `n < 2` (degenerate, not enough curves).

### PdaResult Struct

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PdaResult {
    /// Recovered coefficient functions. coefficients[k] = β_k(t) sampled at argvals.
    /// Length: order (m). Each inner Vec has length n_pts.
    pub coefficients: Vec<Vec<f64>>,
    /// ODE order.
    pub order: usize,
    /// Residuals at each time point (optional diagnostic).
    pub residuals: Option<FdMatrix>,
}
```

### Harmonic Oscillator Test

The canonical recovery test: `x''(t) = -ω² x(t)` with `ω = 2π` (period 1). [ASSUMED — standard PDA validation]

Generate `n = 20` solution curves with varied initial conditions:
```rust
// x_i(t) = A_i * cos(ω*t) + B_i * sin(ω*t)
// with A_i, B_i varied (e.g., A_i = i as f64, B_i = (i+1) as f64)
// Derivatives: x_i'(t) = -A_i*ω*sin(ω*t) + B_i*ω*cos(ω*t)
//              x_i''(t) = -ω²*(A_i*cos(ω*t) + B_i*sin(ω*t)) = -ω²*x_i(t)
```

After PDA with `order = 2`:
- `coefficients[0]` (β₀(t)) ≈ ω² = (2π)² ≈ 39.478 uniformly
- `coefficients[1]` (β₁(t)) ≈ 0.0 uniformly

Test assertion: `|β₀(t_j) - ω²| < 0.5` for all j, `|β₁(t_j)| < 0.5` for all j. (Tolerance accounts for finite-difference error in derivative estimation.)

**Key constraint:** With noiseless solution curves and a fine grid (m ≥ 51 points), the recovery should be much tighter (< 0.01). With `m = 51` uniform points on `[0, 1]` using `gradient_uniform` (5-point stencil, O(h⁴)), `h = 1/50 = 0.02`, so derivative error is O(h⁴) ≈ 1.6×10⁻⁷.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Finite-difference derivatives | Custom FD stencil | `crate::helpers::gradient` | Already implements 5-point stencil for uniform + 3-point Lagrange for non-uniform; tested [VERIFIED: helpers.rs:728-830] |
| Quadrature integration weights | Custom weights | `crate::helpers::simpsons_weights` | Already handles uniform/non-uniform, uniform composite Simpson's [VERIFIED: helpers.rs:57-86] |
| Numeric Gram penalty integration | Custom loop | Copy `smooth_basis::integrate_symmetric_penalty` pattern | Symmetric loop already exploits R[j,k]=R[k,j] [VERIFIED: smooth_basis.rs:572-591] |
| Matrix inversion for PDA normal equations | LU/Cholesky inline | `nalgebra::SVD` pseudoinverse | SVD handles rank-deficient design matrices; same pattern as `pspline_fit_1d` / `smooth_basis.rs` invert_penalized_system |
| Basis column differentiation | New iteration | Copy `smooth_basis::differentiate_basis_columns` | Pattern already exists; iterated `gradient_uniform` per column [VERIFIED: smooth_basis.rs:550-570] |

---

## Common Pitfalls

### Pitfall 1: Power basis — zero/negative argvals with non-integer exponents

**What goes wrong:** `t.powf(-0.5)` for `t = 0.0` returns `f64::INFINITY`; for `t < 0` returns `NaN`. Silent NaN propagation poisons penalty matrices and downstream computations.

**Why it happens:** Power basis is only defined on strictly positive domains when exponents are non-integer or negative.

**How to avoid:** Validate at function entry: if any exponent is non-integer or negative, check `argvals.iter().all(|&t| t > 0.0)`. Return `FdarError::InvalidParameter` otherwise.

**Warning signs:** Penalty matrix diagonal contains `f64::INFINITY` or `NaN`; `clippy` does not catch this.

### Pitfall 2: Analytic penalty — division by zero when `e_i + e_j - 2d + 1 = 0`

**What goes wrong:** The analytic Gram entry formula divides by `(e_i + e_j - 2d + 1)`. If this sum is zero, the integral is `∫ t^{-1} dt = ln(t)`, not a simple power. The division-by-zero produces `NaN` or `Inf`.

**Why it happens:** For `d=1`, `e_i=0`, `e_j=1` on domain `[a,b]` with `a=0`, `0^0 = 1` or similar edge cases can arise. More commonly: `d=2`, `e_i=e_j=1` → sum = 1+1-4+1 = -1 (negative, not zero, but integrand `t^{-1}` at `a=0` diverges).

**How to avoid:** Guard the formula: if `power < -1e-15`, the integral is improper (domain must not include 0). Check that `a > 0` for such cases, or fall back to numeric Gram. If `|power| < 1e-15`, use the `ln(b/a)` formula. [ASSUMED]

### Pitfall 3: Lfd `apply` — broadcasting constant-coefficient `coefs[k].len() == 1`

**What goes wrong:** If `coefs[k]` has length 1 (constant coefficient), indexing `coefs[k][j]` for `j > 0` panics with index out of bounds.

**How to avoid:** In the `apply` inner loop, use `let beta_k = if self.coefs[k].len() == 1 { self.coefs[k][0] } else { self.coefs[k][j] };`.

### Pitfall 4: PDA — not enough curves (`n < order`)

**What goes wrong:** With fewer curves than ODE order, `X_j` has fewer rows than columns → underdetermined system; `(X_j^T X_j)` is singular.

**How to avoid:** Validate `n >= order + 1` at function entry (need at least `order+1` curves for a well-posed pointwise regression). Return `FdarError::InvalidDimension` with a descriptive message.

### Pitfall 5: Polygonal basis — duplicate or non-monotone knots

**What goes wrong:** Hat functions are undefined when `knots[i] >= knots[i+1]` (zero-width intervals produce division by zero in the ramp formula).

**How to avoid:** Validate `knots.windows(2).all(|w| w[1] > w[0])` at entry. Return `FdarError::InvalidParameter`.

### Pitfall 6: Column-major index confusion in eval_matrix

**What goes wrong:** The `bspline_basis` convention is `basis[ti + j * n]` (row = time index, column = basis function). Transposing accidentally produces `basis[j + ti * nbasis]`, which is valid memory but semantically wrong.

**How to avoid:** Always write `eval_matrix[ti + j * n_eval] = value` where `n_eval = argvals.len()`. Add a test that checks `eval_matrix.len() == argvals.len() * nbasis`. [VERIFIED: fdars-core/src/basis/bspline.rs:77-79 — bspline uses `basis[ti + j * n]`]

### Pitfall 7: Clippy `--all-targets` fails on benchmark/example code

**What goes wrong:** The clippy gate runs `--all-targets --features linalg,parallel` which lints test and bench code. Unused imports in `#[cfg(test)]` blocks or new public items not used in any example trigger warnings-as-errors.

**How to avoid:** Run `cargo clippy --all-targets --features linalg,parallel -- -D warnings` locally before committing. [VERIFIED: fdars-project MEMORY.md — "CI clippy uses --all-targets"]

---

## Code Examples

### BasisSystem struct construction (pattern from existing result structs)

```rust
// Source: pattern from PsplineFitResult (fdars-core/src/basis/pspline.rs:39-63)
pub fn monomial_basis(argvals: &[f64], nbasis: usize) -> Result<BasisSystem, FdarError> {
    let n = argvals.len();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: ">= 2".to_string(),
            actual: n.to_string(),
        });
    }
    if nbasis < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "nbasis",
            message: "must be >= 1".to_string(),
        });
    }
    let exponents: Vec<usize> = (0..nbasis).collect();
    let mut eval_matrix = vec![0.0_f64; n * nbasis];
    for (ti, &t) in argvals.iter().enumerate() {
        for j in 0..nbasis {
            eval_matrix[ti + j * n] = t.powi(exponents[j] as i32);
        }
    }
    let lfd_order = 2;
    let a = argvals[0];
    let b = argvals[n - 1];
    let penalty_matrix = monomial_penalty_analytic(&exponents, lfd_order, a, b);
    Ok(BasisSystem { eval_matrix, penalty_matrix, nbasis, n_eval: n, lfd_order })
}
```

### Registration in basis/mod.rs (following existing pattern)

```rust
// Source: pattern from fdars-core/src/basis/mod.rs
pub mod monomial;
pub mod exponential;
pub mod power;
pub mod polygonal;

pub use monomial::monomial_basis;
pub use exponential::exponential_basis;
pub use power::power_basis;
pub use polygonal::polygonal_basis;
pub use basis_system::BasisSystem;  // or define BasisSystem in a new basis/basis_system.rs
```

### Registration in lib.rs (crate-root re-export)

```rust
// Source: pattern from fdars-core/src/lib.rs:483-490
pub use basis::{
    // ... existing exports ...
    monomial_basis, exponential_basis, power_basis, polygonal_basis, BasisSystem,
};

pub mod multi_fdata;
pub use multi_fdata::{MultiFunData, FdComponent};

pub mod pda;
pub use pda::{Lfd, PdaResult, principal_differential_analysis};
```

### Polygonal hat function evaluation

```rust
// Source: derived from R fda create.polygonal.basis semantics [ASSUMED]
fn hat_function(t: f64, knots: &[f64], j: usize) -> f64 {
    let n = knots.len();
    let left_ok = j > 0 && t >= knots[j-1] && t <= knots[j];
    let right_ok = j < n-1 && t >= knots[j] && t <= knots[j+1];
    let left = if left_ok {
        (t - knots[j-1]) / (knots[j] - knots[j-1])
    } else { 0.0 };
    let right = if right_ok {
        (knots[j+1] - t) / (knots[j+1] - knots[j])
    } else { 0.0 };
    left + right
}
```

---

## Test Strategy

### Mandatory Test Coverage Per Component

| Component | Test | Assertion |
|-----------|------|-----------|
| `monomial_basis` | Closed-form at t=0,1,2 for nbasis=3 | B[ti + j*n] == [1,0,0; 1,1,1; 1,2,4] |
| `monomial_basis` | Penalty symmetry | P[i+j*k] == P[j+i*k] for all i,j |
| `monomial_basis` | Penalty PSD | all diagonal elements >= -1e-10 |
| `monomial_basis` | Invalid nbasis=0 | returns Err(InvalidParameter) |
| `exponential_basis` | Closed-form at t=0 | all values == 1.0 (exp(r*0)) |
| `exponential_basis` | rate=0 gives constant | all values == 1.0 |
| `power_basis` | integer exponents match monomial | same eval matrix |
| `power_basis` | negative argval with negative exponent | returns Err(InvalidParameter) |
| `polygonal_basis` | partition of unity | sum over j of B_j(t) == 1.0 for interior t |
| `polygonal_basis` | hat peaks at knot | B_j(knots[j]) == 1.0 |
| `polygonal_basis` | non-monotone knots | returns Err(InvalidParameter) |
| `BasisSystem` | eval_matrix.len() == n_eval * nbasis | shape invariant |
| `BasisSystem` | penalty_matrix.len() == nbasis^2 | shape invariant |
| `MultiFunData::new` | mismatched n_obs | returns Err(InvalidDimension) |
| `MultiFunData::new` | argvals length mismatch | returns Err(InvalidDimension) |
| `MultiFunData` | n_obs/n_components accessors | correct values |
| `MultiFunData` | out-of-range component(k) | returns Err |
| `Lfd::apply` | constant operator on constant function | D^m const = 0, Lf = β₀ * f |
| `Lfd::apply` | mismatched argvals length | returns Err |
| `principal_differential_analysis` | harmonic oscillator recovery | β₀ ≈ ω², β₁ ≈ 0 within 0.5 |
| `principal_differential_analysis` | n < order | returns Err(InvalidDimension) |

### Harmonic Oscillator Test (PDA)

```rust
#[test]
fn pda_recovers_harmonic_oscillator() {
    use std::f64::consts::PI;
    let omega = 2.0 * PI;  // period 1
    let n_pts = 101;
    let t: Vec<f64> = (0..n_pts).map(|i| i as f64 / (n_pts-1) as f64).collect();
    let n_curves = 20;
    // Generate solution curves: x_i(t) = A_i*cos(ω*t) + B_i*sin(ω*t)
    let mut data = FdMatrix::zeros(n_curves, n_pts);
    for i in 0..n_curves {
        let a = (i + 1) as f64;
        let b = (i + 2) as f64;
        for (j, &tj) in t.iter().enumerate() {
            data[(i, j)] = a * (omega * tj).cos() + b * (omega * tj).sin();
        }
    }
    let result = principal_differential_analysis(&data, &t, 2).unwrap();
    // β₀(t) should be ≈ ω² = (2π)² ≈ 39.478
    for &beta0_j in &result.coefficients[0] {
        assert!((beta0_j - omega*omega).abs() < 1.0, "β₀={beta0_j}");
    }
    // β₁(t) should be ≈ 0
    for &beta1_j in &result.coefficients[1] {
        assert!(beta1_j.abs() < 1.0, "β₁={beta1_j}");
    }
}
```

---

## Runtime State Inventory

**Not applicable:** This is a greenfield additive phase. No existing runtime state, stored data, or OS-registered services are renamed or migrated. All additions are new files and new public items.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `#[cfg(test)]` (no criterion for unit tests) |
| Config file | none (uses cargo test) |
| Quick run command | `cargo test -p fdars-core --features linalg,parallel basis 2>&1 \| tail -20` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel 2>&1 \| tail -30` |
| Clippy gate | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REP-01 | monomial_basis closed-form eval | unit | `cargo test basis::monomial::tests` | ❌ Wave 0 |
| REP-01 | monomial_basis analytic penalty symmetric+PSD | unit | `cargo test basis::monomial::tests` | ❌ Wave 0 |
| REP-01 | exponential_basis eval formula | unit | `cargo test basis::exponential::tests` | ❌ Wave 0 |
| REP-01 | power_basis domain validation | unit | `cargo test basis::power::tests` | ❌ Wave 0 |
| REP-01 | polygonal_basis partition of unity | unit | `cargo test basis::polygonal::tests` | ❌ Wave 0 |
| REP-01 | MultiFunData invariant enforcement | unit | `cargo test multi_fdata::tests` | ❌ Wave 0 |
| REP-01 | Lfd::apply on known function | unit | `cargo test pda::tests` | ❌ Wave 0 |
| REP-01 | PDA harmonic oscillator recovery | unit | `cargo test pda::tests::pda_recovers_harmonic_oscillator` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core --features linalg,parallel <module_filter> 2>&1 | tail -20`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -30` + clippy gate
- **Phase gate:** Full suite green + clippy clean before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `fdars-core/src/basis/monomial.rs` — unit tests covering REP-01 monomial eval + penalty
- [ ] `fdars-core/src/basis/exponential.rs` — unit tests covering REP-01 exponential eval
- [ ] `fdars-core/src/basis/power.rs` — unit tests covering REP-01 power eval + domain validation
- [ ] `fdars-core/src/basis/polygonal.rs` — unit tests covering REP-01 polygonal partition-of-unity + knot validation
- [ ] `fdars-core/src/multi_fdata.rs` — unit tests covering MultiFunData invariant enforcement
- [ ] `fdars-core/src/pda.rs` — unit tests covering Lfd::apply + PDA harmonic oscillator recovery

---

## Security Domain

This phase adds pure numerical computation (no I/O, no network, no user-supplied code execution). No ASVS categories apply. The only relevant validation is input bounds checking (already covered by `FdarError::InvalidParameter` returns).

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes (narrowly) | `FdarError::InvalidParameter` for all out-of-range inputs |
| V2–V4, V6 | no | Pure numerical library |

---

## Environment Availability

All implementation uses the existing Rust toolchain and Cargo workspace. No external dependencies to probe.

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Rust stable 1.81+ | MSRV | ✓ (1.97.0 in dev) | 1.97.0 | — |
| nalgebra 0.33 | PDA normal equations | ✓ (existing dep) | 0.33 | — |
| `linalg` feature | Not needed for this phase | ✓ (optional) | faer 0.23 | Not required |

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| bare `Vec<f64>` factory (bspline/fourier/constant) | `BasisSystem` struct bundling eval+penalty | New factories carry their penalty — no separate penalty call needed |
| separate `fourier_penalty_matrix` fn | penalty inside `BasisSystem.penalty_matrix` | Planner/executor: use `BasisSystem.penalty_matrix` directly, don't call separate penalty fn |
| Numeric-only Gram (bspline penalty) | Analytic Gram for monomial/power | Faster and exact for polynomial bases; numeric fallback for exponential/polygonal |

**Deprecated/outdated (within this phase):**
- Manual computation of `∫ D^m B_j D^m B_k dt` by hand — use the analytic formula documented above for monomial/power, or the numeric pattern from `smooth_basis.rs` for others.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Analytic Gram formula for monomial: `R[i,j] = c_i * c_j * (b^p - a^p) / p` | Penalty Matrices | Formula would produce wrong penalty matrix → smoothing misbehaves, but tests would catch it |
| A2 | Falling-factorial generalization works for non-integer exponents in power basis penalty | Penalty Matrices | Non-integer penalty may need numeric fallback; easy to switch |
| A3 | Polygonal 2nd derivative = 0 a.e. → only 1st-order penalty makes sense | Basis Definitions | If fda uses a different convention, document in rustdoc |
| A4 | PDA pointwise LS recovery tolerance of 0.5 is achievable for harmonic oscillator with 101 points | Test Strategy | If finite-difference error is larger, relax tolerance to 2.0 or use more grid points |
| A5 | Numeric Gram grid density = 10 sub-points per original interval (same as `bspline_penalty_matrix`) | Penalty Matrices | Lower density may give inaccurate penalty; this is the existing bspline default |
| A6 | Constant-coefficient Lfd is represented as `coefs[k].len() == 1` broadcast convention | Lfd Design | Alternative: always store `coefs[k].len() == n_pts`; pick at planner discretion |
| A7 | `differentiate_basis_columns` and `integrate_symmetric_penalty` should be promoted to `pub(crate)` in `smooth_basis.rs` to avoid duplication | Architecture | If planner prefers copy-paste, that also works with no API change |

---

## Open Questions

1. **BasisSystem struct placement:** Should `BasisSystem` live in `basis/basis_system.rs` (its own file) or inline in `basis/mod.rs`? Given the struct is shared across four factories, a dedicated file is cleaner. The planner can decide.
   - Recommendation: `basis/basis_system.rs`, re-exported from `basis/mod.rs`.

2. **Lfd/PDA module placement:** `basis/lfd.rs` vs. a new top-level `pda.rs`. Since Lfd is not a basis function (it's a differential operator that consumes basis output), a top-level `src/pda.rs` is a cleaner separation.
   - Recommendation: `src/pda.rs` containing both `Lfd` and `principal_differential_analysis`.

3. **Exponential basis penalty:** Analytic formula exists (product of `r_i^d * r_j^d` times an exponential integral) but requires special-casing `r_i + r_j = 0`. Numeric Gram is simpler. Should the executor implement analytic or numeric?
   - Recommendation: Numeric Gram for exponential (consistent with polygonal; avoids the zero-sum special case). Document in rustdoc that numeric quadrature is used.

4. **PDA `residuals` field:** Should `PdaResult.residuals` be `Some(FdMatrix)` always or `Option`? Computing residuals adds one FdMatrix allocation. Keeping it `Option` with `None` default preserves the `non_exhaustive` extension path.
   - Recommendation: `residuals: Option<FdMatrix>` defaulting to `None`; can be computed by caller from `Lfd::apply` if needed.

---

## Sources

### Primary (MEDIUM confidence — official R package docs)
- [R fda create.monomial.basis](https://search.r-project.org/CRAN/refmans/fda/html/create.monomial.basis.html) — evaluation formula, exponent semantics
- [R fda create.exponential.basis](https://search.r-project.org/CRAN/refmans/fda/html/create.exponential.basis.html) — rate parameter, domain, R source form
- [R fda create.power.basis](https://search.r-project.org/CRAN/refmans/fda/html/create.power.basis.html) — non-integer/negative exponents, domain constraint
- [R fda create.polygonal.basis](https://search.r-project.org/CRAN/refmans/fda/html/create.polygonal.basis.html) — hat function semantics, order-2 B-spline equivalence
- [R fda Lfd object](https://rdrr.io/cran/fda/man/Lfd.html) — exact Lf(t) definition, weight function list
- [R funData multiFunData-class](https://search.r-project.org/CRAN/refmans/funData/html/multiFunData-class.html) — invariants, component access, multi-domain semantics

### Secondary (MEDIUM confidence — scikit-fda docs and arxiv paper)
- [scikit-fda Monomial basis](https://fda.readthedocs.io/en/latest/modules/autosummary/skfda.representation.basis.Monomial.html) — confirmed B_k(t) = t^k, derivative reference values
- [arxiv 2406.18484 — Understanding PDA](https://arxiv.org/abs/2406.18484) — mth-order ODE formulation, function-on-function regression connection

### Tertiary (LOW confidence — web searches confirming general mathematical facts)
- R pda.fd documentation — pointwise LS estimation approach, bwtlist output
- General roughness penalty matrix literature — analytic vs. numeric Gram approaches

---

## Metadata

**Confidence breakdown:**
- Basis evaluation formulas: HIGH — directly confirmed from R fda documentation and scikit-fda
- Analytic penalty matrix formulas: MEDIUM — derived from standard polynomial calculus (cross-checked against scikit-fda description)
- Lfd definition: HIGH — directly from rdrr.io/cran/fda/man/Lfd.html
- PDA formulation: MEDIUM — from arxiv paper and R docs (exact normal-equation construction is ASSUMED)
- MultiFunData invariants: MEDIUM — from funData R package documentation
- Test tolerance values: LOW — based on expected finite-difference error order (not empirically verified)

**Research date:** 2026-08-21
**Valid until:** 2026-09-20 (R fda package is stable; these are decade-old definitions)
