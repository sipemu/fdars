# Stack Research: Optimal Experimental Design for Sparse FDA (v0.35.0)

**Domain:** Rust functional-data-analysis library — adding FOptDes (optimal sparse-measurement design) to fdars-core, built on the shipped `pace_fpca` estimator
**Researched:** 2026-09-02
**Confidence:** HIGH (codebase read = HIGH; fdapace R source read = HIGH; Ji–Müller 2017 formulas extracted from ar5iv HTML = MEDIUM cross-checked against BestDes_TR.R / BestDes_SR.R source = HIGH-verified)

---

## Dependency Verdict: NO NEW CRATE DEPENDENCIES REQUIRED

All v0.35.0 deliverables — trajectory-reconstruction design, score-prediction design, and design-criterion evaluation — can be built entirely on the existing dependency set. This is the primary finding.

Explicit confirmation per deliverable:

| Deliverable | Required new dep? | Rationale |
|-------------|------------------|-----------|
| Integrated BLUP prediction MSE criterion | NO | `cholesky_solve` + `simpsons_weights` cover the full computation; no matrix library beyond current use |
| Greedy sequential design-point selection (trajectory) | NO | Repeated `cholesky_solve` on the growing p×p submatrix `Γ*(S,S)`; no rank-1 update required (R reference re-inverts each step; p is small, ≤ 20 design points) |
| Score-prediction / A-optimality criterion | NO | Same `cholesky_solve` path; criterion is `C(S)ᵀ [Γ*(S,S)]⁻¹ C(S)` — one solve per candidate |
| Quadrature for integrated prediction variance | NO | `simpsons_weights` already computes Simpson weights on any grid |
| Eigenstructure input (`PaceFpcaResult`) | NO | `pace_fpca.rs` already produces `eigenvalues`, `eigenfunctions`, `sigma2`, `argvals` — passed by reference |
| Covariance surface reconstruction (kernel G(s,t)) | NO | Reconstruct as `Σ_k λ_k φ_k(s)φ_k(t)` from `PaceFpcaResult` fields — pure arithmetic on existing `FdMatrix` |
| Linear interpolation of eigenfunctions to candidate grid | NO | `helpers::linear_interp` already used in `pace_fpca.rs` for interpolating φ_k to observed times |
| Parallelism for outer exhaustive-search variant | NO | `iter_maybe_parallel!` gates rayon on candidate-point sweeps |
| `linalg` / faer | NOT NEEDED | All solves are on the p×p system (p ≤ 20); `linalg::cholesky_solve` (always available, no feature gate) is sufficient |
| MSRV change | NONE | All operations are f64 arithmetic on existing types; MSRV remains 1.81 |

---

## Core Technologies

### Primary Technologies (all existing deps — zero Cargo.toml changes)

| Technology | Version in Cargo.toml | Role in v0.35.0 | Existing Usage Anchor |
|------------|----------------------|-----------------|----------------------|
| Rust (MSRV 1.81) | 1.81 min / 1.97 dev | All implementation | Entire codebase |
| nalgebra | 0.33 | `DMatrix::symmetric_eigen()` used inside `pace_fpca.rs` to produce the eigenstructure consumed by FOptDes; no direct nalgebra call needed in FOptDes itself | `src/pace_fpca.rs:187`; `src/regression.rs` |
| rand | 0.8 | Not needed by FOptDes core (design is deterministic given `PaceFpcaResult`); used if a random-start exhaustive fallback is added | `src/kernel_kmeans.rs:263` |
| rayon | 1.10 (optional, `parallel` feature) | Parallelize the candidate-point sweep in the greedy inner loop via `iter_maybe_parallel!` | `src/pace_fpca.rs:434` |

### Reused Internal Primitives (HIGH confidence — direct codebase read)

| Primitive | Location | v0.35.0 Role | Reuse Type |
|-----------|----------|-------------|------------|
| `pace_fpca(data, config) -> PaceFpcaResult` | `src/pace_fpca.rs` | Prior input: caller runs `pace_fpca` once; FOptDes takes the result by reference | Caller-supplied, consumed by reference |
| `PaceFpcaResult.eigenvalues: Vec<f64>` | `src/pace_fpca.rs:102` | λ_k values for covariance kernel reconstruction and criterion weighting | Field access |
| `PaceFpcaResult.eigenfunctions: FdMatrix` | `src/pace_fpca.rs:105` | φ_k(t) values on the work grid; interpolated to candidate design points | Field access + `linear_interp` |
| `PaceFpcaResult.sigma2: f64` | `src/pace_fpca.rs:117` | Ridge term σ² in Γ*(S,S) = G(S,S) + σ²I_p | Field access |
| `PaceFpcaResult.argvals: Vec<f64>` | `src/pace_fpca.rs:115` | Work grid on which φ_k are defined; candidate design points are drawn from or mapped to this grid | Field access |
| `helpers::simpsons_weights(grid: &[f64]) -> Vec<f64>` | `src/helpers.rs:76` | Quadrature weights for ∫ Var(x̂(t) \| U_S) dt — the integrated prediction MSE criterion | Direct call |
| `helpers::linear_interp(grid, vals, t) -> f64` | `src/helpers.rs` | Interpolate φ_k(t_j) at candidate design points not on the work grid | Direct call (identical usage to `pace_fpca.rs:457`) |
| `linalg::cholesky_solve(a, b, p) -> Result<Vec<f64>>` | `src/linalg.rs:131` | Solve Γ*(S,S) x = b for each criterion evaluation and for each candidate addition in the greedy loop | Direct call (same signature as in `pace_fpca.rs:493`) |
| `linalg::cholesky_d(mat, d) -> Result<Vec<f64>>` | `src/linalg.rs:16` | Alternative Cholesky factor when the determinant / log-det is needed (D-optimality variant, if added later) | Direct call |
| `linalg::cholesky_factor(a, p)` + `cholesky_forward_back(l, b, p)` | `src/linalg.rs:85,113` | Used internally by `cholesky_solve`; available if factoring once and solving multiple RHS is preferred (one factor, then k solves for k eigenfunctions) | Via `cholesky_solve` or direct |
| `FdMatrix` | `src/matrix.rs` | Input format for functional data (not directly consumed by FOptDes, but `PaceFpcaResult.eigenfunctions` is an `FdMatrix`) | Existing type |
| `FdarError` | `src/error.rs` | `Result<T, FdarError>` throughout | Existing type |
| `iter_maybe_parallel!(0..n_candidates).map(...)` | `src/parallel.rs` | Parallelize criterion evaluation across all candidate grid points in greedy inner loop | Existing macro |

---

## Algorithm: Linear Algebra Operations Required

This section specifies exactly what FOptDes must compute, derived from the Ji–Müller 2017 framework (verified against fdapace 0.5.9 `BestDes_TR.R` and `BestDes_SR.R` source).

### Shared Setup: Covariance Surface from Eigenstructure

FOptDes does **not** re-run covariance smoothing. It reconstructs the truncated-FPCA approximation from `PaceFpcaResult`:

```
G(s, t) ≈ Σ_{k=1}^{K} λ_k · φ_k(s) · φ_k(t)
```

For any set of candidate design points `t_1, ..., t_p` (drawn from `PaceFpcaResult.argvals` or interpolated), the p×p submatrix is:

```
Γ(S, S)[i,j] = G(t_i, t_j) = Σ_k λ_k · φ_k(t_i) · φ_k(t_j)
```

The ridge-regularized version adds σ²I:

```
Γ*(S, S) = Γ(S, S) + σ² · I_p
```

This is a p×p symmetric positive-definite matrix (p is the number of selected design points, at most 20 in practice).

**Implementation:** Build `Γ*(S,S)` as a flat row-major `Vec<f64>` of length p²; compute by looping over k, accumulating `λ_k · φ_k(t_i) · φ_k(t_j)` with `φ_k` values from `linear_interp`, then add `σ²` to diagonal. No nalgebra DMatrix conversion needed.

### Criterion 1: Trajectory-Recovery (Integrated BLUP Prediction MSE)

Reference: `BestDes_TR.R::TRCri`, Ji–Müller 2017 §2.2.

**Criterion value** (maximize R²_X, equivalently maximize the integrated quadratic form):

```
TRCri(S) = ∫_T γ(t, S)ᵀ · [Γ*(S,S)]⁻¹ · γ(t, S) dt
```

where `γ(t, S)` is the p-vector of covariances between X(t) and the design observations U_S:

```
γ(t, S)[i] = G(t, t_i) = Σ_k λ_k · φ_k(t) · φ_k(t_i)
```

**Integral approximation** using the m-point work grid T with Simpson weights w:

```
TRCri(S) ≈ Σ_{j=1}^{m} w_j · γ(t_j, S)ᵀ · [Γ*(S,S)]⁻¹ · γ(t_j, S)
         = Σ_j w_j · v_j · v_j      (where v_j = [Γ*(S,S)]⁻¹/² γ(t_j, S))
```

**Efficient computation:** Factor `Γ*(S,S) = L Lᵀ` (Cholesky) once. For each work-grid point j, forward-solve `L v_j = γ(t_j, S)` (cost O(p²) per grid point). Integrate `Σ_j w_j ||v_j||²`. Total cost per candidate set evaluation: O(p³) for factorization + O(m·p²) for the forward solves.

```
// In Rust pseudocode:
let l = cholesky_factor(&gamma_star_ss, p)?;                  // p×p lower factor
let mut criterion = 0.0_f64;
for j in 0..m {
    let gamma_tj_s: Vec<f64> = (0..p).map(|i| g(t_j, t_i)).collect();
    let v = cholesky_forward_back(&l, &gamma_tj_s, p);        // forward-back solve
    criterion += w[j] * v.iter().map(|x| x * x).sum::<f64>();
}
```

`linalg::cholesky_factor` + `linalg::cholesky_forward_back` are already `pub(crate)` in `src/linalg.rs` and cover this exactly.

### Criterion 2: Score-Prediction (Posterior Score Variance / A-Optimality)

Reference: `BestDes_SR.R::SRCri`, Ji–Müller 2017 §3.

**Criterion value** (maximize scalar quadratic form using cross-covariance of scores to response):

```
SRCri(S) = C(S)ᵀ · [Γ*(S,S)]⁻¹ · C(S)
```

where `C(S)` is the p-vector of covariances between the response Y and design observations U_S. For the PACE score-prediction variant without an external response, this generalizes to minimizing the posterior variance of FPC score ξ_k given U_S:

```
Var(ξ_k | U_S) = λ_k - λ_k · φ_k(S)ᵀ · [Γ*(S,S)]⁻¹ · (λ_k · φ_k(S))
               = λ_k · (1 - λ_k · φ_k(S)ᵀ · [Γ*(S,S)]⁻¹ · φ_k(S))
```

A-optimality minimizes the trace:

```
AOptCri(S) = Σ_k Var(ξ_k | U_S) = Σ_k λ_k · (1 - λ_k · φ_k(S)ᵀ · [Γ*(S,S)]⁻¹ · φ_k(S))
```

**Implementation:** For each component k, compute `[Γ*(S,S)]⁻¹ · (λ_k · φ_k(S))` via one `cholesky_solve` call. The criterion is a sum of p scalar dot products. Total cost: O(p³) factorization + O(K·p²) solves, where K is the number of FPC components.

### Greedy Sequential Selection (both criteria)

Reference: `BestDes_TR.R` sequential branch; Ji–Müller 2017 §2.3.

Algorithm:
1. Start with empty selected set `S = {}`.
2. For iteration 1..=desired_p:
   a. For each candidate point `t_c` not in S, form `S' = S ∪ {t_c}`.
   b. Build `Γ*(S', S')` (size (|S|+1)×(|S|+1)) — rebuild from scratch (cheap for small p).
   c. Evaluate criterion on S'.
   d. Select `t_c` with highest criterion value; add to S.
3. Return final S.

**No rank-1 / Sherman-Morrison update required.** The fdapace reference re-inverts from scratch at each step (`solve(ridgeCov[design,design])`). For p ≤ 20 and p candidate additions per step, total cost is O(p³ · m · p) = O(p⁴ · m) — negligible for m ≤ 200 and p ≤ 20.

**Optional optimization:** A Schur complement / rank-1 update can be applied if performance profiling reveals it as a hot path (unlikely for this parameter regime). The Sherman-Morrison-Woodbury formula for adding one row/column to an SPD inverse is:

```
[Γ*(S∪{c}, S∪{c})]⁻¹  (block matrix inverse)
= block inversion of [[Γ*(S,S), g_c], [g_cᵀ, g_cc + σ²]]
```

where `g_c = γ(S, t_c)` and `g_cc = G(t_c, t_c)`. The Schur complement of the new point is `s_c = G(t_c, t_c) + σ² - g_cᵀ [Γ*(S,S)]⁻¹ g_c`. This is a standard block-matrix inversion — implementable via `cholesky_solve` on the existing p×p factor — but the reference does not use it and it is not needed for correctness or performance at this scale.

### Exhaustive Search (non-sequential, small p)

For small `p` (≤ 4), exhaustive search over all C(m, p) combinations is feasible and finds the global optimum. Implement as an optional mode: `is_sequential: bool` in config (default `true`). Exhaustive mode parallelized over all combinations via `iter_maybe_parallel!`.

---

## Existing Primitives: Complete Reuse Map

### Already Present in fdars-core (HIGH confidence — direct codebase read)

| Primitive | Location | v0.35.0 Reuse |
|-----------|----------|--------------|
| `pace_fpca(data, config) -> Result<PaceFpcaResult>` | `src/pace_fpca.rs:266` | Caller produces the prior; FOptDes consumes `&PaceFpcaResult` |
| `PaceFpcaResult` (eigenvalues, eigenfunctions, sigma2, argvals, ncomp) | `src/pace_fpca.rs:99` | All fields consumed directly by FOptDes design criterion |
| `linalg::cholesky_solve(a, b, p)` | `src/linalg.rs:131` | Solve `Γ*(S,S) x = b` for each criterion evaluation — identical call signature to `pace_fpca.rs:493` |
| `linalg::cholesky_factor(a, p)` | `src/linalg.rs:85` | Factor once, solve multiple RHS (one per work-grid point in TRCri) |
| `linalg::cholesky_forward_back(l, b, p)` | `src/linalg.rs:113` | Forward-back solve after one factorization — avoids re-factoring per grid point |
| `helpers::simpsons_weights(grid)` | `src/helpers.rs:76` | Quadrature for ∫ criterion(t) dt |
| `helpers::linear_interp(grid, vals, t)` | `src/helpers.rs` | Evaluate φ_k at candidate design points; same call as `pace_fpca.rs:457` |
| `iter_maybe_parallel!(range).map(...)` | `src/parallel.rs` | Parallelize inner greedy candidate sweep |
| `FdMatrix` | `src/matrix.rs` | `eigenfunctions` field type; `(j, k)` column-major indexing used to access φ_k(t_j) |
| `FdarError` | `src/error.rs` | Error type throughout |
| `#[cfg_attr(feature = "serde", derive(...))]` | project-wide convention | Apply to `OptDesConfig` and `OptDesResult` |
| `#[must_use]` | project-wide convention | Apply to `foptdes()` (expensive computation) |

### Not Required (confirmed absent from algorithm)

| Primitive | Why Not Needed |
|-----------|---------------|
| `rustfft::FftPlanner` | No FFT in FOptDes; criterion is a quadratic form, not a convolution |
| `nalgebra::DMatrix::symmetric_eigen()` | Eigendecomposition already done by `pace_fpca`; FOptDes receives λ_k and φ_k as inputs |
| `nalgebra::SVD` | No SVD path in criterion evaluation |
| `faer` (linalg feature) | All solves are on p×p matrices (p ≤ 20); `linalg::cholesky_solve` is always available without the `linalg` feature gate |
| `rand` / `StdRng` | Design is deterministic given the FPCA prior; random starts only if an optional random-exhaustive mode is added |
| `statrs` / `rand_distr` | No distribution sampling needed |
| `cov_irreg` / `mean_irreg` | FOptDes reuses `PaceFpcaResult` — no re-smoothing |

---

## Module Placement

```
src/optimal_design.rs           — single top-level module (matches PROJECT.md description)
    OptDesConfig                — n_design_pts, is_sequential, criterion, ridge, work_grid (optional override)
    OptDesResult                — selected_pts: Vec<f64>, criterion_value: f64, r2: f64, r2adj: f64
    DesignCriterion enum        — TrajectoryRecovery, ScorePrediction
    foptdes(prior, config)      — top-level entry point
    fn build_gamma_star_ss(...)  — assemble Γ*(S,S) from eigenstructure
    fn tr_criterion(...)         — integrate BLUP prediction variance for TRCri
    fn score_criterion(...)      — A-optimal score posterior variance for SRCri
    fn greedy_sequential(...)    — iterative point addition
    fn exhaustive_search(...)    — optional, gated on config.is_sequential=false + small p
```

Crate-root re-export in `src/lib.rs`:
```rust
pub mod optimal_design;
pub use optimal_design::{foptdes, OptDesConfig, OptDesResult, DesignCriterion};
```

Follows the single-file pattern of `pace_fpca.rs` (one public module, no sub-directory needed given the algorithm is self-contained in ~300 lines).

---

## Reference Baseline — Pinned Version & Exact API

### fdapace@0.5.9 — `FOptDes` (R CRAN)

**Function signature (from CRAN refman, fdapace 0.5.9):**

```r
FOptDes(
  Ly,              # list of observed values per subject
  Lt,              # list of observation times per subject
  Resp = NULL,     # scalar response vector (NULL → trajectory recovery mode)
  p = 3,           # number of design points requested
  optns = list(),  # FPCA options (bandwidth, ncomp, etc.)
  isRegression = is.null(Resp),
  isSequential = FALSE,   # FALSE = exhaustive search; TRUE = greedy sequential
  RidgeCand = NULL        # ridge penalty candidates (NULL → CV selection)
)
```

**Return value:** list with `OptDes` (vector of p optimal grid points), `R2` (coefficient of determination at optimum), `R2adj`, `OptRidge`.

**Algorithm (from BestDes_TR.R source, verified):**
1. Run `FPCA(Ly, Lt, optns)` internally to obtain covariance surface `Cov` and FPCA decomposition.
2. Select ridge λ from `RidgeCand` by cross-validation (ensures `ridgeCov[design,design]` is PD for all candidate sets).
3. `TRCri(design, Cov, ridge)`: compute `trapzRcpp(workGrid, diag(t(Cov[design,]) %*% solve(Cov[design,design] + λI) %*% Cov[design,]))`.
4. Sequential or exhaustive search over candidate grid points maximizing `TRCri`.

**Key difference from fdars v0.35.0 plan:** fdapace runs `FPCA` internally. In fdars, `pace_fpca` is already shipped and the caller passes `PaceFpcaResult`. This is the two-stage design — cleaner, avoids re-estimation, and matches the PROJECT.md spec exactly.

**fdapace R² formula** (trajectory recovery):
```
R2 = TRCri(OptDes) / VarX
where VarX = integral of diag(Cov) over workGrid (total functional variance)
```

**fdapace R²adj:**
```
R2adj = 1 - (1 - R2) * (m - 1) / (m - p - 1)
where m = length(workGrid), p = number of design points
```

### Reference Paper

Ji, H. & Müller, H.-G. (2017). Optimal Designs for Longitudinal and Functional Data. *Journal of the Royal Statistical Society: Series B*, 79(3), 859–876. arXiv:1604.05375.

---

## Feature-Flag Considerations

| Aspect | Recommendation |
|--------|---------------|
| Parallel inner greedy loop | YES — gate via `parallel` feature + `iter_maybe_parallel!` on the candidate sweep. Each candidate criterion evaluation is independent (no shared mutable state). |
| `linalg` feature required? | NO — all solves use `linalg::cholesky_solve` from `src/linalg.rs`, which is always available (not behind the `linalg` Cargo feature gate). The `linalg` Cargo feature gates `faer` + `anofox-regression`, which FOptDes does not need. |
| WASM compatibility | YES — pure f64 arithmetic; no FFT; rayon off on WASM. Same as `pace_fpca.rs`. |
| `serde` feature | Apply `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` to `OptDesConfig` and `OptDesResult` following the project convention. |
| MSRV 1.81 | PRESERVED — no post-1.81 stabilizations are used. All called functions (`cholesky_solve`, `simpsons_weights`, `linear_interp`, `iter_maybe_parallel!`) are stable on 1.81. |

---

## MSRV and Feature Compatibility Matrix

| Scenario | Works? | Notes |
|----------|--------|-------|
| Default features (`parallel`) | YES | Greedy inner loop parallelized; MSRV 1.81 |
| No features (sequential only) | YES | All loops sequential-compatible; `cholesky_solve` always available |
| `linalg` feature | YES | FOptDes adds nothing to `linalg`; features are orthogonal |
| `serde` feature | YES | Add derive attributes to OptDesConfig / OptDesResult |
| WASM (`js` feature) | YES | No FFT, no blocking dependencies; rayon off on WASM |
| Rust 1.81 (MSRV) | YES | No stabilizations post-1.81 required |

---

## Alternatives Considered

| Decision | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Sherman-Morrison rank-1 update in greedy loop | Rebuild Γ*(S,S) from scratch each step | Apply Schur complement / block-matrix inversion to avoid re-factoring | p ≤ 20: re-factoring costs O(p³) ≈ 8000 ops — negligible. Rank-1 update code is complex and the fdapace reference does not use it. Add only if profiling flags it. |
| faer for SPD solve | `linalg::cholesky_solve` (always available) | `faer` Cholesky (via `linalg` feature) | `faer` requires MSRV 1.84 and the `linalg` feature gate. There is no performance benefit for 20×20 systems. `linalg::cholesky_solve` is correct and already used in `pace_fpca.rs`. |
| nalgebra DMatrix for Γ*(S,S) | Flat `Vec<f64>` row-major | `nalgebra::DMatrix` | `linalg::cholesky_solve` already accepts a flat row-major slice. Converting to DMatrix adds a copy and a type boundary for no benefit at p ≤ 20. |
| Re-run `pace_fpca` internally (like fdapace) | Accept `&PaceFpcaResult` as prior | Run FPCA internally | Two-stage design is cleaner, avoids re-estimation overhead, and matches the PROJECT.md spec. The caller can choose any FPCA variant; FOptDes is a pure design step. |
| Trapezoidal rule for criterion integration | `simpsons_weights` (existing, Simpson's rule) | Hand-coded trapezoid | `simpsons_weights` is already the project-wide quadrature standard (used in `pace_fpca.rs`, `regression.rs`, `fdata.rs`). Higher-order accuracy for the same grid size. |
| Enumerate combinations via `itertools::combinations` | Hand-rolled combination iterator | `itertools` crate | New dependency not justified; a simple recursive/index-based combination enumerator for the exhaustive mode is ~20 lines of Rust, well within the pattern of other crate utilities. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Re-estimating the covariance surface in FOptDes | Doubles computation time; caller already has `PaceFpcaResult` | Accept `&PaceFpcaResult` — the two-stage architecture specified in PROJECT.md |
| `nalgebra::symmetric_eigen()` in FOptDes | Eigendecomposition is already done; FOptDes consumes λ_k and φ_k from the prior | Direct field access on `PaceFpcaResult.eigenvalues` and `PaceFpcaResult.eigenfunctions` |
| The `linalg` Cargo feature for SPD solve | Requires MSRV 1.84, breaks CRAN compatibility, not needed for p ≤ 20 | `linalg::cholesky_solve` from `src/linalg.rs` (always available, MSRV 1.81) |
| `itertools` or `combinations` crate | New dependency for a simple loop | A hand-rolled index-based combination enumerator (20 lines) |
| Storing Γ*(S,S) as a nalgebra `DMatrix` | Unnecessary type boundary; `cholesky_solve` takes `&[f64]` | Flat `Vec<f64>` row-major — identical to how `pace_fpca.rs` builds `sigma_yi` |
| D-optimality (log-det criterion) as primary criterion | Harder to interpret; not what Ji–Müller 2017 target; fdapace implements A/R² only | Integrated BLUP prediction MSE (TRCri) and score A-optimality (SRCri) |

---

## Version Compatibility

| Package | Version | Compatibility Notes |
|---------|---------|---------------------|
| nalgebra | 0.33 | Provides eigenstructure via `pace_fpca.rs`; not directly called by FOptDes |
| `linalg::cholesky_solve` | always available (no feature gate) | Stable since the function was added; `cholesky_factor` + `cholesky_forward_back` are `pub(crate)` in `src/linalg.rs` |
| `helpers::simpsons_weights` | always available | Stable since `src/helpers.rs` was first added; used project-wide |
| `helpers::linear_interp` | always available | Stable; used in `pace_fpca.rs:457` |
| Rust MSRV | 1.81 | No change — FOptDes uses only features stable on 1.81 |

---

## Sources

- fdapace 0.5.9 `R/FOptDes.R` — function signature, return values, ridge CV, criterion dispatch — fetched via rdrr.io — HIGH confidence (authoritative source, direct code read)
- fdapace 0.5.9 `R/BestDes_TR.R` — `TRCri` matrix formula, greedy sequential algorithm, criterion computation — fetched via rdrr.io — HIGH confidence (authoritative source, direct code read)
- fdapace 0.5.9 `R/BestDes_SR.R` — `SRCri` formula, scalar-response / A-optimality criterion, greedy algorithm — fetched via rdrr.io — HIGH confidence (authoritative source, direct code read)
- Ji & Müller (2017) arXiv:1604.05375 (ar5iv HTML) — mathematical formulation of B(X(t)|U), prediction error variance, R²_X criterion, sequential selection rationale — extracted via WebFetch — MEDIUM confidence (HTML render of arxiv paper, formulas partially prose-described; cross-checked against fdapace source = HIGH-verified)
- fdapace CRAN refman — `FOptDes` function signature and return values — https://cran.r-project.org/web/packages/fdapace/refman/fdapace.html — HIGH confidence (official CRAN documentation)
- fdars-core/src/pace_fpca.rs — confirmed `PaceFpcaResult` fields (eigenvalues, eigenfunctions, sigma2, argvals, ncomp), `cholesky_solve` call signature, `linear_interp` usage pattern, `iter_maybe_parallel!` usage, ridge-retry logic — HIGH confidence (direct codebase read)
- fdars-core/src/linalg.rs — confirmed `cholesky_solve(a, b, p)`, `cholesky_factor(a, p)`, `cholesky_forward_back(l, b, p)` are `pub(crate)`, always available (no feature gate), correct for row-major SPD inputs — HIGH confidence (direct codebase read)
- fdars-core/src/helpers.rs — confirmed `simpsons_weights(argvals) -> Vec<f64>` and `linear_interp` signatures — HIGH confidence (direct codebase read)
- fdars-core/Cargo.toml — confirmed MSRV = "1.81", nalgebra 0.33, `linalg` feature gates faer 0.23 (requires Rust 1.84), `parallel` feature gates rayon 1.10 — HIGH confidence (direct file read)

---

*Stack research for: v0.35.0 Optimal Experimental Design for Sparse FDA (FOptDes) in fdars-core*
*Researched: 2026-09-02*
