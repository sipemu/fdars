# Architecture Research

**Domain:** Rust functional-data-analysis library — Optimal Experimental Design for Sparse FDA (FOptDes, v0.35.0)
**Researched:** 2026-09-02
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Public API layer                                │
│   lib.rs (pub use) · prelude.rs                                      │
│   optimal_design · OptDesConfig · OptDesResult · DesignCriterion     │
│   design_criterion · OptimalityKind · CriterionKind                  │
├──────────────────────────────────────────────────────────────────────┤
│                      Domain module (NEW)                             │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │  optimal_design.rs (NEW, top-level peer of kshape.rs)         │   │
│  │                                                               │   │
│  │  OptDesConfig — candidate grid, budget p, criterion selector  │   │
│  │  OptDesResult — selected indices, selected argvals, criterion │   │
│  │  DesignCriterion — enum: Trajectory | Score                   │   │
│  │  OptimalityKind  — enum: A | D                                │   │
│  │                                                               │   │
│  │  design_criterion(model, indices) -> Result<f64, FdarError>   │   │
│  │    (public, reusable criterion evaluator — greedy inner obj.) │   │
│  │                                                               │   │
│  │  optimal_design(model, config) -> Result<OptDesResult, Fdar>  │   │
│  │    (entry point — greedy sequential selection loop)           │   │
│  │                                                               │   │
│  │  Private helpers:                                             │   │
│  │    trajectory_mse(model, indices, argvals, w) -> f64          │   │
│  │    score_posterior_var(model, indices, argvals) -> f64        │   │
│  │    build_sigma_design(model, indices, argvals) -> Vec<f64>    │   │
│  │    achieved_criterion(model, indices, config) -> f64          │   │
│  └───────────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────┤
│                      Input / Prior model (REUSED)                    │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  pace_fpca.rs (EXISTING, unchanged)                            │   │
│  │  PaceFpcaResult {                                              │   │
│  │    eigenvalues: Vec<f64>,   // λ_k — component variances       │   │
│  │    eigenfunctions: FdMatrix,// φ_k on work grid, m × ncomp    │   │
│  │    sigma2: f64,             // σ² measurement-error variance   │   │
│  │    argvals: Vec<f64>,       // work grid length m              │   │
│  │    ncomp: usize,            // K                               │   │
│  │    ...                                                         │   │
│  │  }                                                             │   │
│  └────────────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────┤
│                      Shared infrastructure (REUSED)                  │
│  ┌──────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │ matrix.rs    │  │ helpers.rs           │  │ linalg.rs           │  │
│  │ FdMatrix     │  │ simpsons_weights()   │  │ cholesky_solve()    │  │
│  │ (col-major)  │  │ linear_interp()      │  │  (Σ_d^{-1} solves)  │  │
│  └──────────────┘  └─────────────────────┘  └─────────────────────┘  │
│  ┌──────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │ parallel.rs  │  │ error.rs            │  │ (no new dep)        │  │
│  │ iter_maybe_  │  │ FdarError           │  │                     │  │
│  │ parallel!    │  │                     │  │                     │  │
│  └──────────────┘  └─────────────────────┘  └─────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `OptDesConfig` | Caller-supplied design problem: candidate grid, budget `p`, criterion (`Trajectory`/`Score`), optimality (`A`/`D`) | `optimal_design.rs` (NEW) |
| `OptDesResult` | Output: selected indices into candidate grid, selected argvals, achieved criterion value | `optimal_design.rs` (NEW) |
| `DesignCriterion` | Enum selecting which objective to minimize: trajectory BLUP-MSE vs FPC-score posterior variance | `optimal_design.rs` (NEW) |
| `OptimalityKind` | Enum selecting matrix summary: A-optimality (trace) vs D-optimality (log-det) | `optimal_design.rs` (NEW) |
| `design_criterion()` | Public reusable function: evaluate the scalar criterion for a caller-supplied index set — the greedy inner objective, independently useful | `optimal_design.rs` (NEW) |
| `optimal_design()` | Entry point: greedy forward-sequential selection, calls `design_criterion` for each candidate at each step | `optimal_design.rs` (NEW) |
| `PaceFpcaResult` | Prior model: provides `eigenvalues`, `eigenfunctions`, `sigma2`, `argvals`, `ncomp` — not modified | `pace_fpca.rs` (EXISTING, UNCHANGED) |

## Recommended Project Structure

```
fdars-core/src/
├── optimal_design.rs       # NEW — OptDesConfig, OptDesResult, optimal_design(),
│                           #        design_criterion(), private helpers
├── pace_fpca.rs            # EXISTING — PaceFpcaResult consumed here (UNCHANGED)
├── kshape.rs               # EXISTING — structural precedent (unchanged)
├── kernel_kmeans.rs        # EXISTING — structural precedent (unchanged)
├── linalg.rs               # EXISTING — cholesky_solve() reused (UNCHANGED)
├── helpers.rs              # EXISTING — simpsons_weights(), linear_interp() reused
│                           #             (UNCHANGED)
└── lib.rs                  # MODIFIED (additive only) — pub mod optimal_design;
                            #   + pub use re-exports
```

Supporting changes (additive only, no signature edits):

```
fdars-core/src/prelude.rs   # MODIFIED (additive) — add OptDesConfig, OptDesResult,
                            #   DesignCriterion, OptimalityKind
fdars-core/src/lib.rs       # MODIFIED (additive) — pub mod + pub use block
```

### Structure Rationale

**`optimal_design.rs` (single top-level file, not a submodule):** FOptDes is self-contained: one config struct, one result struct, two public functions, and ~3 private helpers that share the same mathematical context. There is no second-file distance primitive to separate out (unlike `metric/sbd.rs` + `kshape.rs`). The `kshape.rs` and `kernel_kmeans.rs` precedents demonstrate that a complete algorithm with config/result/fit/predict fits cleanly in one flat file. A dedicated `optimal_design/` directory would add module boilerplate for no practical gain.

**Peer of `kshape.rs` / `kernel_kmeans.rs`:** All three are self-contained algorithms living at the crate root (`src/*.rs`), consuming domain infrastructure from `pace_fpca.rs` / `metric/` without being placed inside it. This keeps module depth flat (no nested path required to import) and matches the existing naming convention for algorithm files.

**No `metric/` submodule entry:** FOptDes is not a distance metric; it is an experimental design algorithm. Placing it under `metric/` would be a category error.

## Architectural Patterns

### Pattern 1: Config Struct + Result Struct + Entry Point Function (mirrors `PaceFpcaConfig` / `PaceFpcaResult` / `pace_fpca`)

**What:** Every algorithm in fdars exposes a `FooConfig` (plain struct, `Default` impl, no `#[non_exhaustive]`), a `FooResult` (`#[non_exhaustive]`, all fields `pub`, serde-gated derives), and a standalone `fn foo(...)` entry point returning `Result<FooResult, FdarError>`. Configuration structs allow struct-literal construction in tests (no `#[non_exhaustive]`), matching `PaceFpcaConfig` and `ElasticPcrConfig` conventions.

**When to use:** Every new public algorithm. No exceptions in the codebase.

**Trade-offs:** Minor verbosity vs. ergonomics gain (callers name fields explicitly, forward-compatible result struct).

**Example:**
```rust
/// Configuration for optimal experimental design ([`optimal_design`]).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptDesConfig {
    /// Candidate measurement grid — the pool of time points from which `p` are selected.
    /// Must be a subset of (or equal to) the model's work grid, length >= `budget`.
    pub candidate_grid: Vec<f64>,
    /// Design budget: number of measurement points to select (1 <= budget <= candidate_grid.len()).
    pub budget: usize,
    /// Which prediction objective to minimize.
    pub criterion: DesignCriterion,
    /// A- or D-optimality for the score-prediction criterion (ignored for Trajectory criterion).
    pub optimality: OptimalityKind,
}

/// Result of optimal experimental design for sparse FDA.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptDesResult {
    /// Indices into `config.candidate_grid` of the selected measurement points (length `budget`).
    /// Ordered by greedy selection sequence (first = most informative marginal gain).
    pub selected_indices: Vec<usize>,
    /// The actual time values of the selected design points (length `budget`).
    pub selected_argvals: Vec<f64>,
    /// The achieved criterion value (integrated BLUP-MSE or score-posterior-variance trace/log-det).
    pub achieved_criterion: f64,
}
```

### Pattern 2: Criterion-Evaluator → Greedy Loop Dependency (build inner objective first, wrap in loop second)

**What:** The greedy selection loop in `optimal_design` is just a wrapper that calls `design_criterion` repeatedly with growing `current_set + candidate` index sets. Because the criterion evaluator is independent and reusable, it is implemented first as a public function with its own validation and tests. The greedy loop then delegates entirely to it. This is the same dependency ordering as `metric/sbd.rs` (Phase 61) preceding `kshape.rs` (Phase 62): build the composable primitive first, then the algorithm that wraps it.

**When to use:** Whenever a design algorithm has an inner objective that is independently useful to callers (e.g. evaluating a researcher-supplied design set against the PACE prior).

**Trade-offs:** One extra public function vs. keeping the evaluator private. Given that the MATLAB PACE `FOptDes` exposes criterion evaluation separately and fdars convention emphasizes reuse, public is correct here.

**Example:**
```rust
/// Evaluate the optimal-design criterion for a caller-supplied set of design indices.
///
/// `indices` are positions into `model.argvals` (the work grid). Returns the scalar
/// criterion value (integrated BLUP-MSE for Trajectory; A/D-optimal posterior-variance
/// summary for Score). Lower is better.
#[must_use = "criterion evaluation result should not be discarded"]
pub fn design_criterion(
    model: &PaceFpcaResult,
    indices: &[usize],
    criterion: DesignCriterion,
    optimality: OptimalityKind,
) -> Result<f64, FdarError> { ... }

/// Greedy forward-sequential selection of `config.budget` measurement points.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn optimal_design(
    model: &PaceFpcaResult,
    config: &OptDesConfig,
) -> Result<OptDesResult, FdarError> {
    // validate ...
    let mut current = Vec::with_capacity(config.budget);
    for _ in 0..config.budget {
        let next = candidate_grid_indices.iter()
            .filter(|i| !current.contains(i))
            .min_by(|&a, &b| {
                let mut sa = current.clone(); sa.push(*a);
                let mut sb = current.clone(); sb.push(*b);
                let va = design_criterion(model, &sa, config.criterion, config.optimality)?;
                let vb = design_criterion(model, &sb, config.criterion, config.optimality)?;
                va.partial_cmp(&vb).unwrap_or(Equal)
            });
        current.push(next);
    }
    // assemble OptDesResult ...
}
```

### Pattern 3: Two Criteria Share a Common Conditional-Covariance Core

**What:** Both the trajectory-reconstruction criterion and the score-prediction criterion require the same intermediate quantity: the conditional-covariance matrix of either x̂(t) or ξ under the design-point set `d`. Concretely, both need `Σ_d = Φ_d diag(λ) Φ_d^T + σ²I_{|d|}` and its inverse (via `linalg::cholesky_solve`). The trajectory criterion integrates the pointwise conditional variance of x̂(t) over the work grid (weighted by `simpsons_weights`); the score criterion forms the `K × K` posterior covariance of the score vector and computes its trace (A-optimality) or negative log-det (D-optimality).

**When to use:** Always. Both criterion branches inside `design_criterion` call the same `build_sigma_design(model, indices)` private helper, then branch only on how they use the result.

**Trade-offs:** Shared code path means the Cholesky solve is not duplicated. The branching happens after the solve, not before — so adding a third criterion variant in the future requires only a new post-solve computation, not a new solve path.

**Example:**
```rust
/// (private) Build the p×p design-covariance matrix Σ_d = Φ_d diag(λ) Φ_d^T + σ²I_p
/// for the given index set, as a row-major flat Vec<f64> (length p*p).
fn build_sigma_design(model: &PaceFpcaResult, indices: &[usize]) -> Vec<f64> {
    let p = indices.len();
    let K = model.ncomp;
    // phi_d[j, k] = model.eigenfunctions[(indices[j], k)]
    let mut sigma_d = vec![0.0_f64; p * p];
    for row in 0..p {
        for col in 0..p {
            let mut s = 0.0_f64;
            for k in 0..K {
                s += model.eigenfunctions[(indices[row], k)]
                    * model.eigenvalues[k]
                    * model.eigenfunctions[(indices[col], k)];
            }
            sigma_d[row * p + col] = s;
        }
        sigma_d[row * p + row] += model.sigma2;
    }
    sigma_d
}

fn trajectory_mse(model: &PaceFpcaResult, indices: &[usize]) -> Result<f64, FdarError> {
    let p = indices.len();
    let sigma_d = build_sigma_design(model, indices);
    // For each work-grid point t_j, conditional variance:
    //   Var(x̂(t_j) | d) = Σ_k λ_k φ_k(t_j)^2 - φ_d(t_j)^T Σ_d^{-1} φ_d(t_j) (projected)
    // Integrate over j using Simpson's weights.
    // ... cholesky_solve per work-grid point or batch solve ...
}

fn score_posterior_var(
    model: &PaceFpcaResult,
    indices: &[usize],
    optimality: OptimalityKind,
) -> Result<f64, FdarError> {
    let p = indices.len();
    let K = model.ncomp;
    let sigma_d = build_sigma_design(model, indices);
    // K×K posterior covariance: Ω_score = diag(λ) - Λ Φ_d^T Σ_d^{-1} Φ_d Λ
    // A-opt: trace(Ω_score)
    // D-opt: -log_det(Ω_score) (or log_det of the information matrix)
    // ... cholesky_solve(sigma_d, phi_col_k) for each k ...
}
```

### Pattern 4: Feature-Gated Parallelism in the Greedy Candidate Loop

**What:** For each of the `budget` greedy selection steps, the algorithm evaluates `|remaining candidates|` criterion values. This is the natural parallelism point: each candidate evaluation is independent and can use `iter_maybe_parallel!`. The inner `design_criterion` calls are stateless (no RNG), so there is no per-thread seeding needed.

**When to use:** The candidate evaluation loop is O(budget × |candidate_grid|) and each evaluation requires Cholesky solves of size p×p (growing from 1 to budget). Parallelism is beneficial for large candidate grids (>50 points) but negligible overhead for small budgets.

**Trade-offs:** Unlike `pace_fpca.rs` where parallel paths require per-thread RNG seeding, FOptDes has no randomness — the only gotcha is that `iter_maybe_parallel!` returns unordered results; use `min_by` with an explicit argmin accumulator rather than relying on iteration order.

```rust
// Greedy step: evaluate all remaining candidates in parallel.
let best_next: usize = iter_maybe_parallel!(remaining.iter())
    .map(|&cand_idx| {
        let mut trial = current.clone();
        trial.push(cand_idx);
        let val = design_criterion(model, &trial, criterion, optimality)
            .unwrap_or(f64::INFINITY);
        (cand_idx, val)
    })
    .reduce(|| (usize::MAX, f64::INFINITY), |(ai, av), (bi, bv)| {
        if bv < av { (bi, bv) } else { (ai, av) }
    })
    .0;
```

## Data Flow

### Trajectory-Reconstruction Criterion (PACE FOptDes canonical path)

```
PaceFpcaResult { eigenvalues λ, eigenfunctions Φ (m×K), sigma2 σ², argvals (m) }
+ indices d ⊆ {0..m-1}  (design point positions in work grid)
    |
    v
build_sigma_design(model, d)
  -> Σ_d = Φ_d diag(λ) Φ_d^T + σ²I_p   (p×p, row-major)
    |
    v
For each work-grid point t_j  (j = 0..m-1):
  phi_at_j(k) = eigenfunctions[(j, k)]
  phi_d_at_j  = [eigenfunctions[(d[0],k)], ..., eigenfunctions[(d[p-1],k)]] for k=0..K-1
  v_j = cholesky_solve(Σ_d, phi_d_at_j)  -> p-vector
  prior_var_j  = Σ_k λ_k * phi_at_j(k)^2
  reduction_j  = dot(phi_d_at_j_projected, v_j)    [= Φ_d(t_j)^T Σ_d^{-1} Φ_d(t_j) projected]
  cond_var_j   = max(prior_var_j - reduction_j, 0.0)
    |
    v
w = simpsons_weights(&model.argvals)
mse = dot(w, cond_var)         [integrated pointwise conditional variance]
    |
    v
f64 (criterion value; lower = better design)
```

### Score-Prediction Criterion

```
PaceFpcaResult + indices d + OptimalityKind
    |
    v
build_sigma_design(model, d)  -> Σ_d (p×p)
    |
    v
For each component k = 0..K-1:
  phi_col_k = [eigenfunctions[(d[0],k)], ..., eigenfunctions[(d[p-1],k)]]
  sol_k = cholesky_solve(Σ_d, phi_col_k)       -> p-vector  [= Σ_d^{-1} Φ_d[:,k]]
    |
    v
Build K×K posterior covariance:
  A_mat[k,l] = λ_k * dot(phi_col_k, sol_l) * λ_l
  Ω_score[k,l] = (k==l ? λ_k : 0) - A_mat[k,l]   [posterior cov of scores]
    |
    v
A-optimality: trace(Ω_score) = Σ_k Ω_score[k,k]
D-optimality: -log_det(Ω_score)  [use Cholesky to compute stable log-det]
    |
    v
f64 (criterion value; lower = better design)
```

### Greedy Forward-Sequential Design

```
PaceFpcaResult + OptDesConfig { candidate_grid, budget p, criterion, optimality }
    |
    v
[Validate: budget >= 1, budget <= |candidate_grid|, all candidates in argvals,
           model has at least 1 component, sigma2 > 0]
    |
    v
map candidate_grid values -> indices into model.argvals
  (nearest-grid lookup; error if any candidate is not in model.argvals)
    |
    v
current = []
for step in 0..budget:
    remaining = candidate_indices \ current
    iter_maybe_parallel!(remaining)
        .map(|cand| {
            trial = current + [cand]
            val = design_criterion(model, &trial, criterion, optimality)
            (cand, val)
        })
        .argmin() -> best_cand
    current.push(best_cand)
    |
    v
achieved = design_criterion(model, &current, criterion, optimality)
    |
    v
OptDesResult {
    selected_indices: current,               // indices into candidate_grid
    selected_argvals: candidate_grid[current],
    achieved_criterion: achieved,
}
```

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `optimal_design.rs` -> `pace_fpca.rs` | `use crate::pace_fpca::PaceFpcaResult` | Read-only borrow of `eigenvalues`, `eigenfunctions`, `sigma2`, `argvals`, `ncomp`. Zero changes to `pace_fpca.rs`. |
| `optimal_design.rs` -> `linalg` | `use crate::linalg::cholesky_solve` | Solves p×p systems (p = current design size, grows 1..budget). Already `pub(crate)`. Same usage pattern as `pace_fpca.rs`. |
| `optimal_design.rs` -> `helpers` | `use crate::helpers::{simpsons_weights, linear_interp}` | `simpsons_weights` integrates pointwise conditional variance; `linear_interp` not needed if candidates are constrained to model's work grid (preferred), but available if interpolation is needed. |
| `optimal_design.rs` -> `parallel` | `use crate::iter_maybe_parallel!` | Parallelise the candidate evaluation at each greedy step. No RNG required (criterion is deterministic). |
| `optimal_design.rs` -> `matrix` | `use crate::matrix::FdMatrix` | Only to access `model.eigenfunctions[(j, k)]` indexing. No new FdMatrix construction needed in most paths. |
| `optimal_design.rs` -> `error` | `use crate::error::FdarError` | `InvalidDimension`, `InvalidParameter`, `ComputationFailed` variants (Cholesky fail). |
| `lib.rs` -> `optimal_design.rs` | `pub mod optimal_design; pub use optimal_design::{...}` | Additive re-export block; no existing symbols touched. |

### External Constraints (WASM / R binding safety)

All new symbols are additive — no existing public signatures change. `OptDesConfig` and `OptDesResult` must carry the same derives as every other config/result: `Debug + Clone + PartialEq`, serde-gated behind `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`. `#[non_exhaustive]` on `OptDesResult` (result struct) but not on `OptDesConfig` (config struct) — matches `PaceFpcaConfig` / `PaceFpcaResult` precedent exactly. No new crate dependency introduced: `linalg::cholesky_solve` uses the existing nalgebra stack; no `rand` needed (FOptDes is deterministic).

## New vs Modified Files

### New files

| File | What it contains |
|------|-----------------|
| `fdars-core/src/optimal_design.rs` | `OptDesConfig`, `OptDesResult`, `DesignCriterion`, `OptimalityKind`, `optimal_design()`, `design_criterion()`, private helpers `build_sigma_design`, `trajectory_mse`, `score_posterior_var`; all tests |

### Modified files (additive only — zero existing signature changes)

| File | Change |
|------|--------|
| `fdars-core/src/lib.rs` | Add `pub mod optimal_design;` + `pub use optimal_design::{optimal_design, design_criterion, OptDesConfig, OptDesResult, DesignCriterion, OptimalityKind};` |
| `fdars-core/src/prelude.rs` | Add `pub use crate::optimal_design::{OptDesConfig, OptDesResult, DesignCriterion, OptimalityKind};` |

### Unchanged files (confirmed reused, not modified)

- `fdars-core/src/pace_fpca.rs` — `PaceFpcaResult` consumed as immutable borrow, no edits
- `fdars-core/src/linalg.rs` — `cholesky_solve` used as-is (already `pub(crate)`)
- `fdars-core/src/helpers.rs` — `simpsons_weights` used as-is
- `fdars-core/src/parallel.rs` — `iter_maybe_parallel!` macro used as-is
- `fdars-core/src/matrix.rs` — `FdMatrix` indexing used as-is
- All other existing modules — untouched

## Dependency-Ordered Build Sequence (Phase Decomposition)

The dependency graph is linear (no parallel phases needed):

### Phase 64 — Criterion Evaluator (`design_criterion` + supporting math)

**Goal:** Implement and test the reusable criterion-evaluation function plus the shared `build_sigma_design` private helper. No greedy loop yet.

**Inputs required:** `PaceFpcaResult` (existing), `cholesky_solve` (existing), `simpsons_weights` (existing).

**Deliverables:**
- `DesignCriterion` enum (`Trajectory`, `Score`)
- `OptimalityKind` enum (`A`, `D`)
- Private `build_sigma_design(model: &PaceFpcaResult, indices: &[usize]) -> Vec<f64>` — builds p×p `Σ_d` row-major
- Private `trajectory_mse(model, indices) -> Result<f64, FdarError>` — integrated conditional BLUP-MSE
- Private `score_posterior_var(model, indices, optimality) -> Result<f64, FdarError>` — trace or log-det of K×K score posterior covariance
- Public `fn design_criterion(model: &PaceFpcaResult, indices: &[usize], criterion: DesignCriterion, optimality: OptimalityKind) -> Result<f64, FdarError>` — dispatches to above two
- `lib.rs` additive: `pub mod optimal_design; pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};`
- Tests: empty-set criterion (returns prior variance, no Cholesky solve), single-point trajectory criterion vs. hand-computed analytic value, score criterion trace agrees with hand-computed posterior for 1-component model, Cholesky failure returns `ComputationFailed` (not panic)

**Why first:** The greedy loop (Phase 65) and the score criterion (which reuses `build_sigma_design` from the trajectory path) both depend on this phase. Validating criterion correctness in isolation is essential before wrapping it in an O(budget × |grid|) selection loop.

### Phase 65 — Greedy Selection Loop (`optimal_design`)

**Goal:** Implement and test the forward-sequential greedy wrapper, `OptDesConfig`, `OptDesResult`, and complete re-exports.

**Inputs required:** Phase 64 complete (`design_criterion` public + validated).

**Deliverables:**
- `OptDesConfig` struct (`candidate_grid: Vec<f64>`, `budget: usize`, `criterion: DesignCriterion`, `optimality: OptimalityKind`) with `Default` impl (sensible defaults: uniform 10-point grid from model argvals, `budget=5`, `Trajectory`, `A`)
- `OptDesResult` struct (`selected_indices: Vec<usize>`, `selected_argvals: Vec<f64>`, `achieved_criterion: f64`, `#[non_exhaustive]`)
- Public `fn optimal_design(model: &PaceFpcaResult, config: &OptDesConfig) -> Result<OptDesResult, FdarError>` — greedy loop with `iter_maybe_parallel!` over candidates
- Input validation: `budget == 0`, `budget > candidate_grid.len()`, any candidate not found in `model.argvals`, `model.ncomp == 0`, `model.sigma2 <= 0`
- `lib.rs` additive: extend `pub use optimal_design::{..., optimal_design, OptDesConfig, OptDesResult};`
- `prelude.rs` additive: add `OptDesConfig`, `OptDesResult`, `DesignCriterion`, `OptimalityKind`
- Tests: greedy monotone property (criterion non-increasing as budget grows), trajectory vs. score criterion selects different points on a synthetic sinusoidal example, determinism (same config = same result), out-of-budget validation error, crate-root re-export smoke test
- Module-level doctest demonstrating the full workflow (fit PACE, call `optimal_design`, read `selected_argvals`)

**Why second:** The greedy loop is a thin O(budget × |grid|) wrapper around Phase 64. It adds no new math — only the selection strategy and the `OptDesConfig`/`OptDesResult` types. This ordering matches the `sbd.rs` → `kshape.rs` precedent (Phase 61 → 62).

**Phase count rationale:** Two phases suffice because there is no third independent primitive analogous to `sbd_kmedoids` (the convenience adapter in Phase 63). Both criteria share `build_sigma_design` already in Phase 64; the score criterion is implemented within `design_criterion` alongside the trajectory criterion, not in a separate phase. A single-phase implementation (64+65 combined) is possible but splitting criterion-evaluator from greedy-loop is the correct decomposition — criterion evaluation has its own tests and is independently callable.

## Anti-Patterns

### Anti-Pattern 1: Constraining candidates to the full work grid and calling `linear_interp`

**What people do:** Accept arbitrary `candidate_grid` values and interpolate eigenfunctions at those positions during criterion evaluation.

**Why it's wrong:** Interpolation introduces approximation error into the criterion and makes the mathematical guarantee (integrated conditional variance for the exact work-grid points) unclear. The PACE FOptDes reference uses the model's own work grid as the candidate pool — the design selects *which* of the already-evaluated grid points to use, not arbitrary sub-T points.

**Do this instead:** Require that every value in `config.candidate_grid` appear (within floating-point tolerance) in `model.argvals`. Return `InvalidParameter` if any candidate is not found. This keeps criterion evaluation exact (index arithmetic only, no interpolation) and matches the PACE reference formulation.

### Anti-Pattern 2: Embedding the greedy loop inside `design_criterion`

**What people do:** Implement one monolithic function that both selects design points and evaluates the criterion, returning the selected set at the end.

**Why it's wrong:** The criterion evaluator is independently useful — callers may supply their own design (expert knowledge, existing measurement schedule) and want to evaluate it against the PACE prior without running the greedy optimizer. Conflating selection with evaluation removes this use case.

**Do this instead:** Keep `design_criterion` as a pure, public function that takes an already-chosen index set and returns a scalar. The greedy loop in `optimal_design` calls it in a forward-selection wrapper. Two separate functions, clearly documented.

### Anti-Pattern 3: Building a new Cholesky solver or using nalgebra's built-in in the hot path

**What people do:** Call `nalgebra::DMatrix::cholesky(&sigma_d)` directly instead of routing through `linalg::cholesky_solve`.

**Why it's wrong:** `linalg::cholesky_solve` already handles the `pub(crate)` interface, converts from flat row-major `Vec<f64>` to nalgebra and back, and centralises the one-place that needs to be updated if the nalgebra version changes. More critically, `pace_fpca.rs` uses it for identical p×p solves — routing through the same helper keeps behaviour and error messages consistent with the existing BLUP infrastructure.

**Do this instead:** `use crate::linalg::cholesky_solve;` — same as `pace_fpca.rs`. If `Σ_d` becomes degenerate (budget > natural dimensionality of the eigenfunction system), add the same `1e-8` ridge retry that `pace_fpca.rs` uses.

### Anti-Pattern 4: Re-implementing the score posterior covariance from scratch in a separate function unaware of the trajectory path's `build_sigma_design`

**What people do:** Implement `trajectory_mse` and `score_posterior_var` as fully independent functions that each build `Σ_d` internally.

**Why it's wrong:** Both criteria need the same `Σ_d = Φ_d diag(λ) Φ_d^T + σ²I_p`. Duplicating this computation doubles the Cholesky allocation and invites divergence if, e.g., a ridge-stabilisation fix is applied to one branch but not the other.

**Do this instead:** Extract `build_sigma_design` as a shared private helper called by both criterion branches. Both `trajectory_mse` and `score_posterior_var` call `build_sigma_design` first, then branch on what they do with the result. This is the same centralisation principle that motivated the shared `build_sigma_design` in `pace_fpca.rs`'s BLUP + band-solve split.

### Anti-Pattern 5: Creating a new top-level Benchmark file for Phase 64

**What people do:** Add a `benches/optimal_design.rs` criterion benchmark in Phase 64 (the criterion-evaluator phase).

**Why it's wrong:** The benchmark is most useful when the full `optimal_design` entry point exists and can be measured end-to-end (greedy selection at representative grid/budget sizes). Benchmarking `design_criterion` in isolation is useful only to diagnose performance bottlenecks, not as primary evidence. The k-Shape precedent put the benchmark in the final phase (Phase 63), after all public symbols were in place.

**Do this instead:** Add `benches/optimal_design.rs` and the `[[bench]]` Cargo.toml entry in Phase 65 alongside the full integration. The benchmark should cover: `design_criterion` (standalone, budget=1 with p=10 grid), `optimal_design` with `Trajectory` criterion (budget=5, 10-point grid), and `optimal_design` with `Score` criterion.

## Reuse Map

| Reused item | Location | How reused in v0.35.0 |
|-------------|----------|-----------------------|
| `PaceFpcaResult` (eigenfunctions, eigenvalues, sigma2, argvals, ncomp) | `pace_fpca.rs` | The prior model input — all criterion math reads from its fields; zero changes to `pace_fpca.rs` |
| `cholesky_solve(sigma, rhs, n)` | `linalg.rs` | Solves p×p `Σ_d` systems; same call site pattern as in `pace_fpca.rs` BLUP + band solve |
| `simpsons_weights(argvals)` | `helpers.rs` | Quadrature weights for integrating pointwise conditional variance over the work grid (trajectory criterion) |
| `iter_maybe_parallel!` macro | `parallel.rs` | Parallelise candidate evaluations at each greedy step; no RNG needed (criterion is deterministic) |
| `FdMatrix[(i,j)]` indexing | `matrix.rs` | Access `model.eigenfunctions[(j, k)]` in column-major layout |
| `FdarError::InvalidDimension` / `InvalidParameter` / `ComputationFailed` | `error.rs` | Input validation + Cholesky failure path |
| Config/Result struct pattern (no `#[non_exhaustive]` on Config, yes on Result) | `pace_fpca.rs`, `kernel_kmeans.rs`, `kshape.rs` | `OptDesConfig` / `OptDesResult` follow the same convention |
| `#[must_use]` on entry-point functions | 74+ functions crate-wide | Applied to both `optimal_design()` and `design_criterion()` |
| Serde-gated derives | Convention throughout codebase | `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` on both structs and enums |
| `pub use` flat re-export in `lib.rs` | Convention throughout codebase | `pub use optimal_design::{optimal_design, design_criterion, OptDesConfig, OptDesResult, DesignCriterion, OptimalityKind}` |

## Sources

- Yao, Müller & Wang (2005), "Functional Data Analysis for Sparse Longitudinal Data", JASA 100(470), 577–590 — PACE BLUP score prediction variance formula (eq. 3.2); the mathematical basis for both design criteria
- PACE@2.17 (MATLAB) `FOptDes` — reference baseline for trajectory-reconstruction optimal design; measurement-point selection minimizing integrated BLUP-MSE
- `fdars-core/src/pace_fpca.rs` — primary reuse target: `PaceFpcaResult` type + `Σ_yi` Cholesky solve pattern + confidence-band conditional-variance computation (direct mathematical ancestor of `build_sigma_design`)
- `fdars-core/src/kshape.rs` + `fdars-core/src/kernel_kmeans.rs` — structural precedent for top-level algorithm module (config/result/entry-point/`#[must_use]`/`#[non_exhaustive]` pattern)
- `fdars-core/src/linalg.rs` — `cholesky_solve` public(crate) interface
- `fdars-core/src/helpers.rs` — `simpsons_weights` quadrature
- `fdars-core/src/lib.rs` + `fdars-core/src/prelude.rs` — re-export patterns (additive `pub use` blocks, prelude categories)
- `.planning/research/GAP-BACKLOG.md` GAP-05 item block — confirmed FOptDes scope and reuse targets
- `.planning/research/survey-matlab.md` MAT-01 — PACE FOptDes capability description and fdars absent-status verification

---
*Architecture research for: Optimal Experimental Design for Sparse FDA (FOptDes) in fdars-core v0.35.0*
*Researched: 2026-09-02*
