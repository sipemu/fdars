# Phase 14: Shift Registration - Research

**Researched:** 2026-08-12
**Domain:** Functional data shift registration and registration quality validation (Rust / fdars-core `alignment/`)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Area 1 — Shift Registration API (FEAT-06)
- **Return type:** dedicated `ShiftRegistrationResult { registered_data: FdMatrix, shifts: Vec<f64> }`, mirroring the existing `AlignmentSetResult { gammas, aligned_data, distances }` pattern. Derive `Debug, Clone, PartialEq`; `#[cfg_attr(feature = "serde", ...)]` per convention.
- **Reference target:** single-pass alignment to the cross-sectional sample mean via `fdata::mean_1d`. Deterministic; matches the backlog phrasing "align each curve to the sample mean." (Iterative template update was considered and deferred — not needed for the parity gap.)
- **Out-of-domain evaluation** when computing `fᵢ(t − δ)`: reuse the v0.16.0 `ExtrapolationPolicy` enum, defaulting to `Boundary` (clamp to endpoints). This composes with the existing interpolation infrastructure rather than inventing new boundary logic.
- **Optimizer:** golden-section search over `δ ∈ [−max_shift, +max_shift]`, minimizing `‖fᵢ(t − δ) − mean(t)‖²` (Simpson-weighted L2), with the shifted curve resampled via `helpers::linear_interp`. `max_shift` default = 0.25 × domain range; expose as a parameter.

#### Area 2 — Registration-Quality Scores (FEAT-07)
- **Return type:** `Result<f64, FdarError>` for all three scores.
- **Signatures (match backlog):** `least_squares_score(registered: &FdMatrix, argvals: &[f64])`, `pairwise_correlation_score(registered: &FdMatrix, argvals: &[f64])`, `sobolev_least_squares_score(registered: &FdMatrix, argvals: &[f64], lambda: f64)`.
- **`least_squares_score`** = (1/n) Σᵢ ∫‖registeredᵢ − mean‖² dt, mean = cross-sectional `mean_1d`, Simpson-weighted integral.
- **`pairwise_correlation_score`** = mean Pearson correlation over all n(n−1)/2 curve pairs via the Simpson inner product; O(n²·m), documented (no sampling cap).
- **`sobolev_least_squares_score`** = first-derivative Sobolev (W¹,²): LS term + λ·(1/n) Σᵢ ∫(fᵢ′ − mean′)² dt, derivative via the existing uniform-gradient helper used by `warp_smoothness`.

#### Area 3 — Integration & Conventions
- **Crate-root re-export:** re-export `least_squares_shift_registration`, `ShiftRegistrationResult`, and the three score functions at the crate root (milestone SC), alongside the existing `alignment/` re-exports.
- **Tests:** inline `#[cfg(test)]` in the respective module files.
- **Non-breaking:** purely additive; no existing `alignment/` signature is modified; no new dependencies.

### Claude's Discretion
- Exact module placement of `least_squares_shift_registration` within `alignment/` (new file e.g. `shift.rs` vs an existing module), tolerance constants in tests, golden-section iteration count / convergence tolerance, and whether `max_shift` / `ExtrapolationPolicy` are positional params or bundled in a small config — all at Claude's discretion, guided by codebase conventions.

### Deferred Ideas (OUT OF SCOPE)
- Iterative multi-pass shift registration with template re-estimation (scikit-fda's iterative mode).
- PREP-06 (LDO-regularized FPCA) and ACC-VALIDATE (numerical validation vs scikit-fda) — explicitly v2 per REQUIREMENTS.md.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FEAT-06 | `least_squares_shift_registration(data, argvals, ...)` aligns each curve to the sample mean by minimizing the L2 distance under a per-curve constant shift δᵢ via golden-section search over the objective evaluated via linear interpolation, returning registered curves + per-curve shift values. | Golden-section algorithm, `linear_interp` + `simpsons_weights` reuse, `mean_1d` reuse, `ShiftRegistrationResult` struct pattern from `AlignmentSetResult`. |
| FEAT-07 | Three registration-quality scores added to `alignment/quality.rs`: `least_squares_score`, `pairwise_correlation_score`, `sobolev_least_squares_score`. | Formula verification vs scikit-fda semantics (see Divergence Note below), `gradient_uniform` reuse, `simpsons_weights` + `mean_1d` reuse, `Result<f64, FdarError>` return type. |
</phase_requirements>

---

## Summary

Phase 14 delivers two additive, non-breaking items within `fdars-core/src/alignment/`. Both items share the `alignment/` subtree and their re-export sites (`alignment/mod.rs`, `lib.rs`), which is the only coordination point.

**FEAT-06** (`least_squares_shift_registration`) is a new pure-Rust function implementing rigid horizontal shift registration: for each of the n curves, minimize the Simpson-weighted L2 distance between the shifted curve fᵢ(t − δᵢ) and the cross-sectional sample mean, using golden-section search on the scalar δᵢ over `[−max_shift, +max_shift]`. The shifted curve is evaluated by `helpers::linear_interp(argvals, row, t - delta)` which already implements `Boundary`-clamping for out-of-range t. The result struct `ShiftRegistrationResult { registered_data: FdMatrix, shifts: Vec<f64> }` follows the `AlignmentSetResult` pattern exactly, including `#[non_exhaustive]`, the standard derives, and optional serde. Parallelism follows `align_to_target`'s pattern: `iter_maybe_parallel!(0..n)` to compute per-curve shifts, then a sequential row-assembly loop.

**FEAT-07** (three quality scores) are added to `alignment/quality.rs` beside the existing raw-return neighbors. All three return `Result<f64, FdarError>` (not raw f64), which is intentional and documented. The formulas implemented in fdars differ from scikit-fda's ratio-based scores: fdars implements **standalone energy** metrics (absolute L2 spread) rather than scikit-fda's quotient (registered vs original). This divergence is locked by CONTEXT.md. The Sobolev score's derivative uses `gradient_uniform` (already imported in `quality.rs` via `crate::helpers`); the pairwise correlation uses the L2-weighted Simpson inner product and requires n ≥ 2.

**Primary recommendation:** New file `alignment/shift.rs` for FEAT-06; FEAT-07 appended to `alignment/quality.rs`. Re-export both at `alignment/mod.rs` and `lib.rs`. Tests inline in each file. The re-export edits to `mod.rs` and `lib.rs` must be serialized (one plan step) to prevent merge collision.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Shift optimization (per-curve golden-section) | Library / Algorithm | — | Pure numerical computation; no I/O, no service boundary |
| Shifted-curve evaluation | Helpers layer (`helpers::linear_interp`) | — | Existing point-evaluation infrastructure handles boundary clamping |
| Cross-sectional mean | `fdata::mean_1d` | — | Already parallelized, column-major correct |
| Quality scores (LS, pairwise, Sobolev) | `alignment/quality.rs` | — | Convention: quality functions live beside `warp_complexity` etc. |
| Integration weights | `helpers::simpsons_weights` | `helpers::trapz` | Same infrastructure used throughout `alignment/` |
| Derivative approximation (Sobolev term) | `helpers::gradient_uniform` | — | Already used by `warp_smoothness` in `quality.rs` |
| Public API surface | `alignment/mod.rs` → `lib.rs` | — | Standard fdars flat re-export chain |

---

## Standard Stack

### Core (no new dependencies)

All capabilities are implemented from existing crate infrastructure:

| Asset | Location (verified) | Purpose in Phase 14 |
|-------|---------------------|---------------------|
| `mean_1d` | `src/fdata.rs:167` | Cross-sectional sample mean for shift target and score mean |
| `linear_interp` | `src/helpers.rs:172` | Pointwise evaluation of fᵢ(t − δ) with built-in Boundary clamping |
| `simpsons_weights` | `src/helpers.rs:57` | Simpson integration weights for L2 objective and score integrals |
| `gradient_uniform` | `src/helpers.rs:728` | First-derivative approximation for Sobolev score |
| `l2_distance` | `src/helpers.rs:37` | `√(Σ (f−g)² wᵢ)` — used by `alignment_quality`; usable in tests |
| `iter_maybe_parallel!` | `src/parallel.rs:42` | Feature-gated parallelism for per-curve shift loop |
| `AlignmentSetResult` struct | `src/alignment/mod.rs:134` | Pattern reference for `ShiftRegistrationResult` |
| `FdarError` | `src/error.rs` | Return type for score functions |
| `ExtrapolationPolicy` | `src/helpers.rs:865` | Enum already in scope (in-scope note: `linear_interp` already clamps; `ExtrapolationPolicy::Boundary` is conceptually equivalent but `linear_interp` does not require it as a parameter) |

**No new Cargo.toml dependencies.** [VERIFIED: src/alignment/set.rs:1-8, src/alignment/quality.rs:1-8, src/fdata.rs:167-179, src/helpers.rs:37-44, src/helpers.rs:57-86, src/helpers.rs:172-191, src/helpers.rs:728-761, src/parallel.rs:42-55]

---

## Package Legitimacy Audit

> Not applicable — Phase 14 introduces zero new crate dependencies.

---

## Architecture Patterns

### System Architecture Diagram

```
 User calls least_squares_shift_registration(data, argvals, max_shift)
         │
         ▼
 mean_target = fdata::mean_1d(data)          ← cross-sectional mean (m floats)
         │
         ├─ iter_maybe_parallel!(0..n)
         │       │
         │       ▼  (per curve i)
         │   golden_section_search(
         │       objective = |δ| L2_shift_objective(data.row(i), argvals, mean_target, δ),
         │       lo = -max_shift, hi = +max_shift,
         │       tol, max_iter
         │   )  →  δᵢ
         │       │
         │       ▼
         │   best_shift_delta[i] = δᵢ
         │
         ▼
 collect shifts: Vec<f64>    (parallel → sequential assembly)
         │
         ▼
 build registered_data: FdMatrix  (n × m)
     for each curve i:
         for each j in 0..m:
             registered_data[(i,j)] = linear_interp(argvals, row_i, argvals[j] - δᵢ)
         │
         ▼
 return Ok(ShiftRegistrationResult { registered_data, shifts })

─────────────────────────────────────────────────────────

 User calls least_squares_score(registered, argvals)
         │
         ▼
 mean = mean_1d(registered)
 weights = simpsons_weights(argvals)
 score = (1/n) Σᵢ Σⱼ (registeredᵢⱼ − meanⱼ)² × weightsⱼ
 return Ok(score)

─────────────────────────────────────────────────────────

 User calls pairwise_correlation_score(registered, argvals)
         │
         ▼
 weights = simpsons_weights(argvals)
 For all n(n-1)/2 pairs (i,k):
     corr(i,k) = Σⱼ (row_i[j] × row_k[j] × weights[j])
               / sqrt(Σⱼ row_i[j]² × weights[j] × Σⱼ row_k[j]² × weights[j])
 return Ok(mean over all pairs)

─────────────────────────────────────────────────────────

 User calls sobolev_least_squares_score(registered, argvals, lambda)
         │
         ▼
 h = (argvals[m-1] - argvals[0]) / (m-1)  ← uniform spacing assumed
 mean = mean_1d(registered)
 mean_prime = gradient_uniform(mean, h)
 weights = simpsons_weights(argvals)
 ls_term    = (1/n) Σᵢ Σⱼ (registeredᵢⱼ − meanⱼ)² × weightsⱼ
 sobol_term = (1/n) Σᵢ Σⱼ (fi_prime[j] − mean_prime[j])² × weightsⱼ
 return Ok(ls_term + lambda × sobol_term)
```

### Recommended Project Structure

```
fdars-core/src/alignment/
├── shift.rs          ← NEW: least_squares_shift_registration + ShiftRegistrationResult
├── quality.rs        ← EXTEND: three new score functions appended
├── mod.rs            ← EXTEND: pub use shift::{...}; add score fn re-exports to quality block
fdars-core/src/
└── lib.rs            ← EXTEND: add shift fn + result type + score fns to alignment use block
```

### Pattern 1: Golden-Section Search (1-D Minimization)

**What:** Bracket-based optimizer for unimodal objectives over a closed interval. Converges to tolerance `ε` in `~log((hi-lo)/ε) / log(1/φ)` evaluations (φ = golden ratio ≈ 1.618). No derivatives needed.

**When to use:** Scalar-shift objective is smooth and assumed unimodal (L2 distance to the mean is convex in δ when the mean is fixed and evaluation is via linear interpolation).

**Example (Rust implementation sketch):**
```rust
// Source: standard textbook golden-section search
fn golden_section_search<F>(f: F, mut lo: f64, mut hi: f64, tol: f64, max_iter: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    const PHI: f64 = 1.6180339887498949;
    let mut x1 = hi - (hi - lo) / PHI;
    let mut x2 = lo + (hi - lo) / PHI;
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    for _ in 0..max_iter {
        if (hi - lo) < tol {
            break;
        }
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - (hi - lo) / PHI;
            f1 = f(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + (hi - lo) / PHI;
            f2 = f(x2);
        }
    }
    (lo + hi) / 2.0
}
```

**Recommended parameters:**
- `tol = 1e-6` (sub-microsecond shift precision is sufficient for all practical FDA grids)
- `max_iter = 100` (never approached in practice; golden-section converges in ~50 steps for `tol=1e-6` over `[-0.25, +0.25]`)

### Pattern 2: L2-Shift Objective Evaluation

**What:** For a given shift δ, compute the Simpson-weighted L2 distance between the shifted curve fᵢ(t − δ) and the mean.

**Implementation note:** `linear_interp(argvals, row_i, argvals[j] - delta)` already handles out-of-domain points with boundary clamping (returns `row_i[0]` or `row_i[last]`). The `ExtrapolationPolicy` enum does not need to be threaded into this path — `linear_interp` is itself `Boundary`-policy by implementation. [VERIFIED: src/helpers.rs:172-191]

```rust
fn l2_shift_objective(
    row: &[f64],
    argvals: &[f64],
    mean: &[f64],
    weights: &[f64],
    delta: f64,
) -> f64 {
    // Source: CONTEXT.md formula; linear_interp at src/helpers.rs:172
    argvals.iter().zip(mean.iter()).zip(weights.iter()).enumerate()
        .map(|(j, ((&t, &m_j), &w))| {
            let fi_shifted = crate::helpers::linear_interp(argvals, row, t - delta);
            let diff = fi_shifted - m_j;
            diff * diff * w
        })
        .sum::<f64>()
}
```

### Pattern 3: Pairwise Pearson Correlation via Simpson Inner Product

**What:** Functional Pearson correlation between two curves fᵢ and fₖ using Simpson weights.

```rust
// Source: CONTEXT.md definition; simpsons_weights at src/helpers.rs:57
fn functional_pearson_corr(ri: &[f64], rk: &[f64], weights: &[f64]) -> f64 {
    let norm_sq = |v: &[f64]| v.iter().zip(weights).map(|(&x, &w)| x * x * w).sum::<f64>();
    let inner   = ri.iter().zip(rk).zip(weights).map(|((&a, &b), &w)| a * b * w).sum::<f64>();
    let denom = norm_sq(ri).sqrt() * norm_sq(rk).sqrt();
    if denom < 1e-15 { 0.0 } else { inner / denom }
}
```

**Note:** This computes the normalized L2 inner product (cosine similarity in the L2 space), which equals Pearson correlation for zero-mean curves and approximates it for non-zero-mean curves. The CONTEXT.md decision to use the Simpson inner product is consistent with the rest of `alignment/quality.rs`. [ASSUMED: exact definition of "Pearson correlation" for functions is not uniquely specified; the Simpson inner product form is the natural functional extension used throughout fdars]

### Pattern 4: Sobolev Score via `gradient_uniform`

**What:** Derivative via `gradient_uniform(row, h)` where `h = (argvals[m-1] - argvals[0]) / (m-1)` (assumes uniform grid).

```rust
// Source: quality.rs:warp_smoothness uses gradient_uniform identically at src/alignment/quality.rs:43-55
let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
let fi_prime = gradient_uniform(&fi_row, h);
let mean_prime = gradient_uniform(&mean, h);
```

**Note:** `gradient_uniform` requires a uniform grid spacing `h`. For non-uniform grids, the function still computes a numerical gradient but with h = average spacing. This is acceptable under the same assumption already made by `warp_smoothness` in the same file. If argvals is non-uniform, document the limitation in rustdoc. [VERIFIED: src/alignment/quality.rs:49, src/helpers.rs:728]

### Pattern 5: `ShiftRegistrationResult` Struct

```rust
// Source: AlignmentSetResult pattern at src/alignment/mod.rs:134-141
/// Result of least-squares shift registration.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShiftRegistrationResult {
    /// Registered (shifted) functional data (n × m).
    pub registered_data: FdMatrix,
    /// Per-curve horizontal shifts δᵢ applied to each curve.
    pub shifts: Vec<f64>,
}
```

[VERIFIED: src/alignment/mod.rs:134-141 for `#[non_exhaustive]` + `AlignmentSetResult` precedent]

### Anti-Patterns to Avoid

- **`#[must_use]` on `Result`-returning functions:** clippy::double_must_use fires under `-D warnings`. Do NOT add `#[must_use]` to `least_squares_shift_registration` or the three score functions. (Raw-f64-returning functions like `warp_complexity` and `warp_smoothness` carry `#[must_use]` — this convention does not extend to `Result`-returning functions.) [VERIFIED: src/alignment/quality.rs:38, 43 — `warp_complexity` and `warp_smoothness` carry no `#[must_use]` on their public signatures either, consistent with codebase convention that `#[must_use]` appears on expensive struct-returning fns like `align_to_target`]
- **New Cargo.toml dependencies:** Phase 14 is zero-dependency. Do not add crates.
- **Modifying existing `alignment/` signatures:** All changes are purely additive.
- **Panicking on invalid input:** All public functions must return `Result<T, FdarError>` and validate dimensions at entry. Use `InvalidDimension` for n=0/m=0/argvals length mismatch.
- **Re-exporting from `alignment/mod.rs` and `lib.rs` in parallel plans:** Merge collision risk. Serialize re-export edits to a single plan step.

---

## Critical Semantic Divergence: fdars vs scikit-fda Scores

**This is the most important research finding for the planner.**

The CONTEXT.md specifies **standalone energy metrics** for the three scores. Scikit-fda implements **ratio-based improvement metrics** (registered vs original). These are fundamentally different:

| Score | fdars (CONTEXT.md / locked) | scikit-fda (actual) |
|-------|----------------------------|---------------------|
| `least_squares_score` | `(1/n) Σᵢ ∫(regᵢ − mean)² dt` — absolute L2 spread | `1 − [(1/N) Σᵢ ∫(regᵢ − LOO_mean)²] / [∫(origᵢ − LOO_mean)²]` — relative reduction |
| `pairwise_correlation_score` | `mean Pearson corr over n(n-1)/2 pairs` | `Σᵢ≠ⱼ cc(regᵢ, regⱼ) / Σᵢ≠ⱼ cc(origᵢ, origⱼ)` — ratio to original |
| `sobolev_least_squares_score` | `LS_term + λ × derivative_term` — combined absolute | `1 − Σᵢ ∫(regᵢ′ − mean′)² / Σᵢ ∫(origᵢ′ − mean′)²` — derivative relative |

**Action for planner:** Implement the fdars standalone formulas as locked in CONTEXT.md. Document in rustdoc that fdars' scores are standalone diagnostic metrics (unlike scikit-fda's ratio-based scorers). Scikit-fda's pairwise correlation requires the original data as input (which fdars does not take); fdars' version takes only the registered data. [CITED: https://fda.readthedocs.io/en/latest/modules/preprocessing/autosummary/skfda.preprocessing.registration.validation.LeastSquares.html; https://fda.readthedocs.io/en/latest/modules/preprocessing/autosummary/skfda.preprocessing.registration.validation.SobolevLeastSquares.html; https://fda.readthedocs.io/en/latest/modules/preprocessing/autosummary/skfda.preprocessing.registration.validation.PairwiseCorrelation.html]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cross-sectional mean | Custom column-mean loop | `fdata::mean_1d` | Already parallel, tested, column-major correct |
| Simpson integration weights | Custom weight formula | `helpers::simpsons_weights` | Handles uniform/non-uniform, even/odd n correctly |
| Numerical derivative | FD stencil from scratch | `helpers::gradient_uniform` | 5-point O(h⁴) stencil, boundary-aware, already in scope |
| Point interpolation with clamping | Custom lerp + clamp | `helpers::linear_interp` | Already clamps to boundary values (Boundary policy built in) |
| 1-D minimizer | Bisection / Brent's / gradient | Golden-section search (self-contained) | Unimodal assumption valid; ~50 lines, no dependencies |

**Key insight:** The entire Phase 14 implementation reuses 5 existing helpers; the only novel code is the golden-section search (~50 lines) and the score formulas (~60 lines).

---

## Common Pitfalls

### Pitfall 1: `#[must_use]` on `Result`-returning functions
**What goes wrong:** Clippy `double_must_use` lint fires under `-D warnings`, breaking CI.
**Why it happens:** The project marks expensive struct-returning computation functions with `#[must_use]`, but `Result` is already `#[must_use]` by the standard library; adding it again triggers the lint.
**How to avoid:** Do not add `#[must_use]` to `least_squares_shift_registration` or the three score functions. Only mark `ShiftRegistrationResult` fields with `#[must_use]` if applicable (it is not needed on the struct itself given `#[non_exhaustive]`).
**Warning signs:** CI clippy pass fails with `double_must_use`.

### Pitfall 2: Re-export merge collision in `mod.rs` and `lib.rs`
**What goes wrong:** If two plans both edit `alignment/mod.rs` or `src/lib.rs`, git creates a conflict on the `pub use` lines.
**Why it happens:** FEAT-06 adds `ShiftRegistrationResult` + `least_squares_shift_registration`; FEAT-07 adds three score function names. Both touch the same `pub use quality::{...}` and the new `pub use shift::{...}` blocks.
**How to avoid:** Serialize all `mod.rs` and `lib.rs` re-export edits to a single plan that runs after both implementation plans complete.
**Warning signs:** Git merge conflict on `alignment/mod.rs` line with `pub use`.

### Pitfall 3: Golden-section search on a non-unimodal objective
**What goes wrong:** For pathological input (e.g., multi-modal functional data), the L2-to-mean objective may not be unimodal in δ, and golden-section returns a local minimum.
**Why it happens:** The convexity of `‖fᵢ(t−δ)−mean(t)‖²` in δ holds only when `linear_interp` is monotone in δ (which it is for small shifts, but not guaranteed for large shifts across multiple bumps).
**How to avoid:** Document the unimodal assumption in rustdoc. The `max_shift` parameter naturally limits the search bracket; the default 0.25×domain_range avoids the worst-case multi-modal regimes.
**Warning signs:** Test with injected-offset bumps recovers wrong sign of δ.

### Pitfall 4: Non-uniform grid passed to `sobolev_least_squares_score`
**What goes wrong:** `gradient_uniform` computes a numerical derivative assuming uniform grid spacing. On a non-uniform grid, `h` computed as `(argvals[m-1] - argvals[0])/(m-1)` is approximate.
**Why it happens:** The existing `warp_smoothness` function in the same file uses the same approximation. The behavior is silent — no error is raised.
**How to avoid:** Document the uniform-grid assumption in rustdoc (consistent with `warp_smoothness`). For the test suite, only use uniform grids.
**Warning signs:** Sobolev score gives unexpected values on non-uniform grids.

### Pitfall 5: Pairwise correlation with n=1
**What goes wrong:** If n=1, there are no pairs: `n*(n-1)/2 = 0`. Division by zero or empty sum.
**Why it happens:** The O(n²·m) loop iterates over `(0..n).flat_map(|i| (i+1..n)...)` — for n=1 this is empty.
**How to avoid:** Validate n ≥ 2 at function entry and return `Err(FdarError::InvalidParameter)` for n < 2.
**Warning signs:** Returns NaN or panics on single-curve input.

### Pitfall 6: `iter_maybe_parallel!` pattern for row assembly
**What goes wrong:** Parallel iterator returns results in arbitrary order; using `push` into a shared `Vec` from a parallel map produces incorrect row assignment.
**Why it happens:** `align_to_target` (the pattern reference at `set.rs:59-82`) does a parallel `.collect::<Vec<AlignmentResult>>()` then a sequential row-assignment loop. The same pattern must be used here.
**How to avoid:** Collect per-curve `(shift, shifted_row)` results into a `Vec<(f64, Vec<f64>)>` from the parallel map, then sequentially assemble into `FdMatrix`. [VERIFIED: src/alignment/set.rs:59-82]
**Warning signs:** Test with n > 1 produces jumbled row assignments.

---

## Code Examples

Verified patterns from the live codebase:

### `align_to_target` parallel-collect-then-assemble pattern (template for FEAT-06)
```rust
// Source: src/alignment/set.rs:59-82 [VERIFIED]
let results: Vec<AlignmentResult> = iter_maybe_parallel!(0..n)
    .map(|i| {
        let fi = data.row(i);
        elastic_align_pair(target, &fi, argvals, lambda)
    })
    .collect();

let mut gammas = FdMatrix::zeros(n, m);
let mut aligned_data = FdMatrix::zeros(n, m);
let mut distances = Vec::with_capacity(n);

for (i, r) in results.into_iter().enumerate() {
    for j in 0..m {
        gammas[(i, j)] = r.gamma[j];
        aligned_data[(i, j)] = r.f_aligned[j];
    }
    distances.push(r.distance);
}
```

### `alignment_quality` — how `mean_1d` and `simpsons_weights` are already used in `quality.rs`
```rust
// Source: src/alignment/quality.rs:67-99 [VERIFIED]
let (n, m) = data.shape();
let weights = simpsons_weights(argvals);
let orig_mean = crate::fdata::mean_1d(data);
let total_var: f64 = (0..n)
    .map(|i| {
        let fi = data.row(i);
        let d = l2_distance(&fi, &orig_mean, &weights);
        d * d
    })
    .sum::<f64>()
    / n as f64;
```

### `warp_smoothness` — `gradient_uniform` usage pattern for Sobolev term
```rust
// Source: src/alignment/quality.rs:43-55 [VERIFIED]
pub fn warp_smoothness(gamma: &[f64], argvals: &[f64]) -> f64 {
    let m = gamma.len();
    if m < 3 { return 0.0; }
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let gam_prime = gradient_uniform(gamma, h);
    let gam_pprime = gradient_uniform(&gam_prime, h);
    let integrand: Vec<f64> = gam_pprime.iter().map(|&g| g * g).collect();
    crate::helpers::trapz(&integrand, argvals)
}
```

### `AlignmentSetResult` / `ShiftRegistrationResult` struct pattern
```rust
// Source: src/alignment/mod.rs:134-141 [VERIFIED]
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AlignmentSetResult {
    pub gammas: FdMatrix,
    pub aligned_data: FdMatrix,
    pub distances: Vec<f64>,
}
```

### `mean_1d` signature
```rust
// Source: src/fdata.rs:167-179 [VERIFIED]
pub fn mean_1d(data: &FdMatrix) -> Vec<f64> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 { return Vec::new(); }
    iter_maybe_parallel!(0..m)
        .map(|j| {
            let col = data.column(j);
            col.iter().sum::<f64>() / n as f64
        })
        .collect()
}
```

### `linear_interp` — built-in Boundary clamping
```rust
// Source: src/helpers.rs:172-191 [VERIFIED]
pub fn linear_interp(x: &[f64], y: &[f64], t: f64) -> f64 {
    if t <= x[0] { return y[0]; }          // ← Boundary clamp: lower
    let last = x.len() - 1;
    if t >= x[last] { return y[last]; }    // ← Boundary clamp: upper
    // ... binary search + interpolate
}
```

### Crate-root re-export block (existing alignment section in `lib.rs`)
```rust
// Source: src/lib.rs:139-170 [VERIFIED — showing key lines for new additions]
pub use alignment::{
    // ... existing items ...
    warp_complexity, warp_smoothness, warp_statistics,
    // NEW items to add:
    // least_squares_shift_registration, ShiftRegistrationResult,
    // least_squares_score, pairwise_correlation_score, sobolev_least_squares_score,
};
```

---

## Exact Public API Surface

### FEAT-06: `alignment/shift.rs`

```rust
// New file: fdars-core/src/alignment/shift.rs
pub struct ShiftRegistrationResult {
    pub registered_data: FdMatrix,
    pub shifts: Vec<f64>,
}

pub fn least_squares_shift_registration(
    data: &FdMatrix,
    argvals: &[f64],
    max_shift: f64,
) -> Result<ShiftRegistrationResult, FdarError>
```

**Parameter guidance (Claude's Discretion):** Keep `max_shift` as a positional parameter (not config struct) — the function has only 3 parameters total, below the 5-parameter threshold where config structs become beneficial. Default `max_shift` is not a Rust concept; document `0.25 * (argvals.last() - argvals.first())` as the recommended caller value.

### FEAT-07: Additions to `alignment/quality.rs`

```rust
pub fn least_squares_score(
    registered: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError>

pub fn pairwise_correlation_score(
    registered: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError>

pub fn sobolev_least_squares_score(
    registered: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
) -> Result<f64, FdarError>
```

### Re-exports

**`alignment/mod.rs`** — add to `pub use quality::{...}` block:
```
least_squares_score, pairwise_correlation_score, sobolev_least_squares_score,
```
Add new block:
```rust
pub use shift::{least_squares_shift_registration, ShiftRegistrationResult};
```
Add `mod shift;` to the module list.

**`lib.rs`** alignment `pub use` block — add:
```
least_squares_score, pairwise_correlation_score, sobolev_least_squares_score,
least_squares_shift_registration, ShiftRegistrationResult,
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Landmark-only shift (internal, not returned) | `least_squares_shift_registration` (new, standalone, returns δᵢ) | Phase 14 (this work) | Fills the "simplest registration method" gap |
| No quality scores beyond `alignment_quality` struct | Three standalone score functions returning `Result<f64>` | Phase 14 (this work) | Enables post-hoc registration diagnostics |

**Deprecated/outdated:**
- The backlog PREP-04 entry's proposed return type name `RegistrationResult` is superseded by `ShiftRegistrationResult` (locked in CONTEXT.md to avoid confusion with `AlignmentResult`).
- The backlog PREP-05 entry's proposed signatures used raw `f64` returns; CONTEXT.md locks these as `Result<f64, FdarError>` to enable dimension validation.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Pairwise correlation implemented as normalized Simpson inner product (L2 inner product form), not unweighted Pearson over raw grid values | Architecture Patterns §Pattern 3 | Scores will not match any reference implementation; mismatches may surface in integration tests |
| A2 | Golden-section search is sufficient (objective is unimodal in δ for typical functional data within the default `max_shift` bracket) | Architecture Patterns §Pattern 1 | Local-minimum trapping on multi-modal data; `max_shift` parameter partially mitigates |
| A3 | `gradient_uniform` on a non-uniform grid uses h = average spacing (approximation accepted) | Architecture Patterns §Pattern 4 | Sobolev score inaccurate on non-uniform grids; acceptable given same assumption in `warp_smoothness` |
| A4 | fdars standalone score formulas (not scikit-fda ratio forms) are correctly specified in CONTEXT.md | Critical Semantic Divergence section | Implementation differs from scikit-fda; documented in rustdoc, but comparison tests against scikit-fda will fail if expected otherwise |

---

## Open Questions

1. **`max_shift` parameter naming and default documentation**
   - What we know: CONTEXT.md says expose as a parameter; default = 0.25 × domain range.
   - What's unclear: Whether to hard-code the 0.25 default as a constant or just document it in rustdoc.
   - Recommendation: Define a private `const DEFAULT_MAX_SHIFT_FRACTION: f64 = 0.25;` in `shift.rs` and document in rustdoc. Let the caller pass `0.25 * (argvals.last() - argvals.first())`. Do not make it optional/defaultable at the type level (Rust has no default arguments).

2. **Pairwise correlation score normalization**
   - What we know: "mean Pearson correlation over all n(n-1)/2 pairs" (CONTEXT.md). Pearson correlation for functions is not uniquely defined.
   - What's unclear: Whether to use (a) L2 inner product normalized by L2 norms (functional analogue), or (b) compute over the discrete evaluation grid points without weights.
   - Recommendation: Use (a) Simpson-weighted inner product, normalized by Simpson-weighted norms. This is consistent with all other integration operations in `alignment/quality.rs` and avoids treating the evaluation grid as uniformly weighted.

---

## Environment Availability

Step 2.6: SKIPPED — Phase 14 is purely code/config changes within the Rust crate with no external service dependencies. The existing Rust toolchain and Cargo are sufficient.

**Pre-commit note from MEMORY.md:** `/tmp` tmpfs exhaustion can block doctest linking with bogus "No space left on device". Use `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` for doctest-heavy commits, or `--no-verify` for doc-only commits. [CITED: .planning/STATE.md §Blockers/Concerns]

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` harness + Criterion 0.5 (bench only) |
| Config file | `fdars-core/Cargo.toml` (no separate test config) |
| Quick run command | `cargo test -p fdars-core --features linalg -- alignment::shift::tests alignment::quality::tests 2>&1` |
| Full suite command | `cargo test -p fdars-core --features linalg 2>&1` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FEAT-06-A | Already-aligned set (constant curves) → δᵢ ≈ 0.0 for all i | unit | `cargo test -p fdars-core --features linalg -- test_shift_already_aligned` | ❌ Wave 0 |
| FEAT-06-B | Injected constant offset δ_known recovered within tolerance (e.g., δ=0.1 → recovered δᵢ ≈ 0.1 ± 1e-4) | unit | `cargo test -p fdars-core --features linalg -- test_shift_recovers_injected_offset` | ❌ Wave 0 |
| FEAT-06-C | Registered curves are re-evaluated at the correct shifted argvals (spot-check registered_data values) | unit | `cargo test -p fdars-core --features linalg -- test_shift_registration_curve_values` | ❌ Wave 0 |
| FEAT-06-D | Empty data returns `Err(FdarError::InvalidDimension)` | unit | `cargo test -p fdars-core --features linalg -- test_shift_registration_empty_data` | ❌ Wave 0 |
| FEAT-06-E | Argvals length mismatch returns `Err(FdarError::InvalidDimension)` | unit | `cargo test -p fdars-core --features linalg -- test_shift_registration_argvals_mismatch` | ❌ Wave 0 |
| FEAT-07-A | `least_squares_score` on identical constant curves = 0.0 | unit | `cargo test -p fdars-core --features linalg -- test_ls_score_identical_curves` | ❌ Wave 0 |
| FEAT-07-B | `least_squares_score` drops after `least_squares_shift_registration` on synthetic shifted bumps | unit | `cargo test -p fdars-core --features linalg -- test_ls_score_drops_after_registration` | ❌ Wave 0 |
| FEAT-07-C | `pairwise_correlation_score` rises after registration on shifted bumps | unit | `cargo test -p fdars-core --features linalg -- test_pairwise_corr_rises_after_registration` | ❌ Wave 0 |
| FEAT-07-D | `pairwise_correlation_score` with n=1 returns `Err(FdarError::InvalidParameter)` | unit | `cargo test -p fdars-core --features linalg -- test_pairwise_corr_n1_error` | ❌ Wave 0 |
| FEAT-07-E | `sobolev_least_squares_score` with lambda=0 equals `least_squares_score` | unit | `cargo test -p fdars-core --features linalg -- test_sobolev_score_lambda_zero` | ❌ Wave 0 |
| FEAT-07-F | `sobolev_least_squares_score` with lambda > 0 > lambda=0 score (derivative penalty adds positive term) | unit | `cargo test -p fdars-core --features linalg -- test_sobolev_score_lambda_positive` | ❌ Wave 0 |

### Synthetic Bump Generator (test fixture)

All direction tests (FEAT-06-B, FEAT-07-B, FEAT-07-C) use the same synthetic bumps pattern:

```rust
// Gaussian bump at position mu, used as test fixture
fn gaussian_bump(argvals: &[f64], mu: f64, sigma: f64) -> Vec<f64> {
    argvals.iter().map(|&t| (-(t - mu).powi(2) / (2.0 * sigma * sigma)).exp()).collect()
}

// n curves with injected shifts [0.0, delta, 2*delta, ...]
fn make_shifted_bumps(n: usize, m: usize, delta: f64) -> (FdMatrix, Vec<f64>, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    let mut true_shifts = Vec::with_capacity(n);
    for i in 0..n {
        let shift = i as f64 * delta;
        let row = gaussian_bump(&argvals, 0.5 + shift, 0.05);
        for j in 0..m { data[(i, j)] = row[j]; }
        true_shifts.push(shift);
    }
    (data, argvals, true_shifts)
}
```

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core --features linalg -- alignment 2>&1`
- **Per wave merge:** `cargo test -p fdars-core --features linalg 2>&1`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `fdars-core/src/alignment/shift.rs` — covers FEAT-06-A through FEAT-06-E (new file, inline `#[cfg(test)] mod tests`)
- [ ] `fdars-core/src/alignment/quality.rs` additions — covers FEAT-07-A through FEAT-07-F (inline `#[cfg(test)] mod tests` block extension or added tests block)

*(No missing framework — built-in `#[test]` harness already configured. No new test fixtures file needed; synthetic helpers defined inline in module tests.)*

---

## Security Domain

Security enforcement is enabled (`security_enforcement: true`, ASVS Level 1). Phase 14 is a pure numerical computation library addition with no I/O, no authentication, no network, no file access, and no user-controlled string parsing. ASVS categories analysis:

| ASVS Category | Applies | Rationale |
|---------------|---------|-----------|
| V2 Authentication | No | No auth path in this phase |
| V3 Session Management | No | Stateless pure functions |
| V4 Access Control | No | Library function, no access decisions |
| V5 Input Validation | **Yes** | Dimension checks on `&FdMatrix` and `&[f64]` at function entry |
| V6 Cryptography | No | No crypto operations |
| V7 Error Handling | **Yes** | All errors returned via `Result<T, FdarError>`; no panics on invalid input |

**V5 / V7 controls for Phase 14:**
- `least_squares_shift_registration`: validate `data.ncols() == argvals.len()`, `data.nrows() > 0`, `argvals.len() >= 2`, `max_shift > 0.0`. Return `Err(FdarError::InvalidDimension)` / `Err(FdarError::InvalidParameter)` on failure.
- `least_squares_score`, `pairwise_correlation_score`, `sobolev_least_squares_score`: validate `registered.ncols() == argvals.len()`, `registered.nrows() >= 1` (≥2 for pairwise). Return `Err(FdarError::InvalidParameter)` with a descriptive message for `lambda < 0.0`.
- No panic paths in any public function (dimension checks at entry prevent index-out-of-bounds in the computation loops).

**Known threat patterns:**

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow in pairwise index computation `n*(n-1)/2` | Tampering | Use `usize` arithmetic; n is bounded by memory capacity; document O(n²) complexity |
| NaN propagation from degenerate data (all-zero curves) | Information disclosure | `l2_norm = sqrt(0)` → denom=0 in correlation; handle with `if denom < 1e-15 { 0.0 }` guard |
| Infinite shift objective (NaN from linear_interp) | Tampering | `linear_interp` returns boundary value for OOB, never NaN; no issue |

---

## Sources

### Primary (HIGH confidence)
- `fdars-core/src/alignment/quality.rs:1-229` — existing quality functions, imports, struct patterns [VERIFIED: read this session]
- `fdars-core/src/alignment/set.rs:1-124` — `align_to_target` pattern for parallel collect + row assembly [VERIFIED: read this session]
- `fdars-core/src/alignment/mod.rs:1-170` — re-export chain, `AlignmentSetResult` struct [VERIFIED: read this session]
- `fdars-core/src/fdata.rs:167-179` — `mean_1d` signature and implementation [VERIFIED: read this session]
- `fdars-core/src/helpers.rs:37-44` — `l2_distance` signature [VERIFIED: read this session]
- `fdars-core/src/helpers.rs:57-86` — `simpsons_weights` [VERIFIED: read this session]
- `fdars-core/src/helpers.rs:172-191` — `linear_interp` with Boundary clamping [VERIFIED: read this session]
- `fdars-core/src/helpers.rs:728-761` — `gradient_uniform` 5-point stencil [VERIFIED: read this session]
- `fdars-core/src/helpers.rs:865-884` — `ExtrapolationPolicy` enum variants [VERIFIED: read this session]
- `fdars-core/src/parallel.rs:42-55` — `iter_maybe_parallel!` macro [VERIFIED: read this session]
- `fdars-core/src/lib.rs:139-180` — crate-root re-export site for alignment + helpers [VERIFIED: read this session]

### Secondary (MEDIUM confidence)
- scikit-fda `LeastSquares` documentation — formula confirmed via `fda.readthedocs.io` [CITED: https://fda.readthedocs.io/en/latest/modules/preprocessing/autosummary/skfda.preprocessing.registration.validation.LeastSquares.html]
- scikit-fda `SobolevLeastSquares` documentation — derivative-variance ratio formula [CITED: https://fda.readthedocs.io/en/latest/modules/preprocessing/autosummary/skfda.preprocessing.registration.validation.SobolevLeastSquares.html]
- scikit-fda `PairwiseCorrelation` documentation — pairwise sum-of-correlations ratio formula [CITED: https://fda.readthedocs.io/en/latest/modules/preprocessing/autosummary/skfda.preprocessing.registration.validation.PairwiseCorrelation.html]
- BACKLOG.md PREP-04 item — backlog formulas and API shape [VERIFIED: .planning/research/BACKLOG.md:339-375 — read this session]
- BACKLOG.md PREP-05 item — backlog formulas and return types [VERIFIED: .planning/research/BACKLOG.md:359-376 — read this session]

### Tertiary (LOW confidence — not used for primary design decisions)
- Training knowledge: golden-section search algorithm parameters [ASSUMED]

---

## Metadata

**Confidence breakdown:**
- Standard stack (reusable assets): HIGH — all signatures read from live source files this session
- Architecture (struct patterns, re-export chain): HIGH — verified against live codebase
- Formulas (fdars standalone metrics): HIGH — locked in CONTEXT.md; scikit-fda ratio-forms confirmed as divergent via official docs
- Pitfalls: HIGH — `#[must_use]` pitfall confirmed in STATE.md; re-export collision from STATE.md wave/serialization note; others from code reading
- Test design: HIGH — all test assertions derive from verified code paths

**Research date:** 2026-08-12
**Valid until:** 2026-09-12 (stable library, MSRV 1.81, no external service dependencies)
