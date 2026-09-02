# Feature Research: FOptDes — Optimal Experimental Design for Sparse FDA (v0.35.0)

**Domain:** Optimal design of sparse measurement locations for trajectory recovery and FPC score prediction — `fdars-core` Rust crate
**Milestone:** v0.35.0 (promotes GAP-05 from `GAP-BACKLOG.md`)
**Researched:** 2026-09-02
**Confidence:** HIGH — Mathematical formulas derived from the authoritative source (Ji & Müller 2017, JRSSB 79:859-876), cross-checked against the full `fdapace`/tPACE R package source code (`BestDes_TR.R`, `BestDes_SR.R`, `FOptDes.R`) and a 2025 review paper (Huang et al. 2025, WIREs). The exact criterion algebra and greedy algorithm structure are confirmed. Complexity analysis is inferred from algorithm structure (HIGH confidence).

**In-scope:** Trajectory-reconstruction design criterion; FPC-score-prediction design criterion; design-criterion evaluation function (independent scorer for any candidate set); two-stage workflow on a `PaceFpcaResult`. **Out-of-scope:** σ² estimation (already deferred in `pace_fpca`), bandwidth selection, FPCA re-estimation, arbitrary-loss design criteria, Bayesian D-optimal designs (Huang et al. 2025 extension).

---

## Precise Mathematical Specification

All notation follows `pace_fpca.rs` exactly. This section is the normative formula source for requirements definition.

### Prerequisite: fitted PACE model

Given a `PaceFpcaResult` (already estimated), the design step uses only these fields:

- `eigenvalues`: λ₁ ≥ … ≥ λ_K > 0 (length K = `ncomp`)
- `eigenfunctions`: m × K matrix, φ_k(t_j) for j = 0..m-1, k = 0..K-1 (on `argvals` work grid)
- `sigma2`: σ² > 0 (measurement-error variance)
- `argvals`: work grid t_0 < t_1 < … < t_{m-1}

The candidate design grid is any finite subset of the work grid (typically all m points). The budget is p ≥ 1 points to select.

---

### A. Background: process covariance kernel

Under the Karhunen-Loève expansion:

```
C(s, t) = Σ_{k=1}^K  λ_k · φ_k(s) · φ_k(t)
```

Evaluated at m grid points this is an m × m matrix C with C[j1, j2] = Σ_k λ_k φ_k(t_{j1}) φ_k(t_{j2}).

In `fdapace` `FOptDes`, this is called `Cov` and is constructed from the FPCA fitted covariance (`fittedCov`). In `fdars` the implementor assembles it as Φ diag(λ) Φᵀ where Φ is the m × K eigenfunction matrix from `PaceFpcaResult`. This is an O(m² K) operation done once before the greedy loop.

---

### B. Criterion 1: Trajectory-Reconstruction Design (TR)

#### B.1 Conditional prediction variance at a single time point

For a candidate design set S = {t_{s_1}, …, t_{s_p}} (p indices into the work grid), the BLUP conditional variance of x̂(t) given noisy observations at S is:

```
Var(x̂(t) | Y_S) = C(t,t) − C(t,S) · [C(S,S) + σ²I_p]⁻¹ · C(S,t)
```

where:
- C(t,t) = Σ_k λ_k φ_k(t)² (scalar, total process variance at t)
- C(t,S) is a length-p row vector: C(t,S)[j] = Σ_k λ_k φ_k(t) φ_k(t_{s_j})
- C(S,S) is the p×p sub-matrix of C restricted to rows and columns indexed by S
- [C(S,S) + σ²I_p]⁻¹ is the ridge-regularized inverse (adds σ² to the diagonal)

This is the standard BLUP prediction formula for a Gaussian process with kernel C(·,·) observed with noise variance σ².

#### B.2 Integrated prediction variance (IPV) — the design objective

The TR design criterion is the **integral of the conditional prediction variance** over the work grid:

```
IPV(S) = ∫ Var(x̂(t) | Y_S) dt
       = ∫ C(t,t) dt − ∫ C(t,S) · [C(S,S) + σ²I_p]⁻¹ · C(S,t) dt
```

Numerically, on the work grid of m points (trapezoidal or Simpson weights w_j):

```
IPV(S) ≈ Σ_j w_j · [C(t_j,t_j) − c(t_j,S)ᵀ · (C(S,S) + σ²I_p)⁻¹ · c(t_j,S)]
```

The criterion to **maximize** (explained variance integral) is:

```
TRCri(S) = trapz_workGrid( diag( C_{S,:}ᵀ · (C_{SS} + σ²I)⁻¹ · C_{S,:} ) )
```

where C_{S,:} is the p × m sub-matrix of C restricted to the rows indexed by S, and diag(·) extracts the diagonal of the resulting m × m matrix — equivalently the pointwise reduction in prediction variance due to the design. This is exactly what `BestDes_TR.R`'s `TRCri` function computes.

The `fdars` implementation uses `simpsons_weights` (already in `helpers.rs`) rather than trapezoidal integration, for consistency with the rest of the codebase.

#### B.3 R² form (Ji & Müller 2017)

The criterion is expressed as a coefficient of determination:

```
R²_TR(S) = TRCri(S) / ∫ C(t,t) dt = TRCri(S) / VarX
```

where VarX = Σ_k λ_k (total process variance, the integral of C(t,t) under L²-orthonormal eigenfunctions). A design with R²_TR → 1.0 fully recovers the trajectory on average; R²_TR = 0 gives no information. The implementor returns both the absolute TRCri value and R²_TR at each greedy step.

---

### C. Criterion 2: FPC-Score-Prediction Design (SP)

#### C.1 Score posterior covariance

Under PACE, the posterior covariance of FPC scores ξ = (ξ_1, …, ξ_K)ᵀ given noisy observations Y_S at design points S is:

```
Cov(ξ | Y_S) = Λ − Λ Φ_Sᵀ [C(S,S) + σ²I_p]⁻¹ Φ_S Λ    (K × K matrix)
```

where:
- Λ = diag(λ_1, …, λ_K) — the K×K eigenvalue diagonal matrix
- Φ_S is the p × K matrix of eigenfunction values at the design points: Φ_S[j, k] = φ_k(t_{s_j})
- C(S,S) = Φ_S Λ Φ_Sᵀ + 0·I is the process-only covariance at S; the full noisy covariance is C(S,S) + σ²I_p

This is the posterior covariance of a multivariate Gaussian with prior Cov(ξ) = Λ, likelihood noise σ², and design matrix Φ_S.

**Critical note:** this matrix is exactly the **Ω_i** matrix computed in `pace_fpca.rs` (lines 549–558), except Ω_i is computed for each curve's *actual* observed points, while here it is computed for a *prospective* design set S. The formula is:

```
Ω[k,l] = λ_k δ_{kl} − A[k,l]
A[k,l] = λ_k · Φ_S[:,k]ᵀ · (C(S,S) + σ²I)⁻¹ · Φ_S[:,l] · λ_l
```

The implementor can directly adapt the A_mat computation from lines 547–558 of `pace_fpca.rs`.

#### C.2 A-optimality criterion (trace)

A-optimal design minimizes the trace of the posterior score covariance — minimizes total posterior variance across all K components:

```
A-crit(S) = trace( Cov(ξ | Y_S) ) = trace(Λ) − trace(Λ Φ_Sᵀ (C(S,S) + σ²I)⁻¹ Φ_S Λ)
```

Equivalently, **maximize the trace reduction** (the "information gained"):

```
ΔA(S) = trace(Λ) − A-crit(S) = Σ_{k,l} A[k,l] · δ_{kl}
       = Σ_k λ_k² · φ_k(S)ᵀ · (C(S,S) + σ²I_p)⁻¹ · φ_k(S)
```

A-optimal R²: ΔA(S) / trace(Λ) = ΔA(S) / Σ_k λ_k.

#### C.3 D-optimality criterion (log-det)

D-optimal design minimizes the log-determinant of the posterior score covariance:

```
D-crit(S) = log det( Cov(ξ | Y_S) )
```

Equivalently (using the matrix determinant lemma for the Schur complement):

```
ΔD(S) = log det(Λ) − D-crit(S) = log det(I_p + (1/σ²) Φ_S Λ Φ_Sᵀ)
```

Implementation note: compute log det via the Cholesky factor of Cov(ξ|Y_S) as 2·Σ log(diagonal of L). For numerical stability, compute the Cholesky of the p×p matrix (C(S,S) + σ²I_p) and use the matrix determinant lemma rather than factorizing the K×K posterior directly when K > p.

D-optimal R²: (log det(Λ) − D-crit(S)) / log det(Λ).

#### C.4 fdapace SR-criterion distinction

`FOptDes` with `isRegression = TRUE` uses `SRCri(S) = CCov[S]ᵀ · (C(S,S) + σ²I)⁻¹ · CCov[S]` where `CCov` is the cross-covariance between the functional predictor and a scalar response. This is distinct from score A/D-optimality — it optimizes for a specific scalar outcome. For v0.35.0, only score A/D-optimality is implemented (response-agnostic); SR-design is deferred.

---

### D. Greedy Sequential Algorithm

The greedy algorithm is identical for all criteria; only the inner criterion function varies.

#### D.1 Algorithm (greedy forward selection)

```
Input:  PaceFpcaResult (φ_k, λ_k, σ², argvals of length m),
        G = candidate grid (subset of 0..m indices, typically all m),
        p = budget (number of points to select, 1 ≤ p ≤ |G|),
        criterion ∈ {TR, A-optimal, D-optimal},
        ridge = regularization scalar (default: sigma2)

Output: S ⊆ G with |S| = p, selected indices
        criterion_curve[0..p] — criterion value after adding i-th point

Algorithm:
  C ← Φ diag(λ) Φᵀ           // m×m process covariance (build once, O(m² K))
  S ← ∅
  for iter = 1 to p:
    best_gain ← −∞
    best_idx  ← None
    for each candidate index g ∈ G \ S:
      S' ← S ∪ {g}
      gain ← criterion_fn(S', C, ridge)   // O(p² + p·m) for TR, O(K·p) for A
      if gain > best_gain OR (gain == best_gain AND g < best_idx):
        best_gain ← gain
        best_idx  ← g
    S ← S ∪ {best_idx}
    criterion_curve[iter-1] ← best_gain
  return S, criterion_curve
```

Tie-breaking rule: when two candidates produce identical `gain` values (exact floating-point equality), prefer the candidate with the smaller index (earlier in the work grid). This is deterministic and consistent with the `iter_maybe_parallel!` pattern which does a post-loop argmax over the result vector.

#### D.2 Per-candidate cost and total complexity

For **TR criterion** at iteration iter (|S| = iter − 1, adding one point to form S' of size iter):
- Form C_{S',:} (iter × m sub-matrix): O(iter · m) (row extraction)
- Solve (C_{S'S'} + σ²I_{iter})⁻¹: O(iter³) naive; O(iter²) with rank-1 Cholesky update from the previous iteration
- Compute diag(C_{S',:}ᵀ M C_{S',:}) and integrate: O(iter · m)
- **Per candidate at iteration iter**: O(iter² + iter · m)
- **Total over all p iterations with |G| = m candidates**: O(p² · m + p · m²) ≈ O(p · m²) when m >> p
- Practical: m=100, p=5, K=3 → ~50,000 FLOPs per greedy step. Sub-millisecond without parallelism.

For **A-optimal score criterion** at iteration iter:
- Build Φ_{S'} (iter × K): O(iter · K) (row extraction from Φ)
- Solve (C(S'S') + σ²I_{iter})⁻¹: O(iter³) or O(iter²) with update
- Compute A_mat (K×K): O(K² · iter)
- Compute trace: O(K)
- **Per candidate**: O(iter³ + K² · iter) = O(K² · iter) for K > iter
- **Total**: O(p · m · (K² · p)) = O(K² · p² · m)
- Practical: K=3, p=5, m=100 → small.

For **D-optimal score criterion**:
- Same structure as A-optimal but adds log-det of K×K posterior covariance
- log-det from Cholesky: O(K³) per candidate
- **Total**: O(K³ · p · m)

#### D.3 Implementation note on the rank-1 Cholesky update

For the TR criterion, the Cholesky factor of (C_{S'S'} + σ²I_{iter}) can be maintained across iterations using a rank-1 update (Cholesky append): each new row/column added extends the factorization without re-factorizing from scratch. This reduces per-iteration cost from O(iter³) to O(iter²). However, for the expected regime (p ≤ 10, m ≤ 200), the O(iter³) naive approach is also acceptable. The v0.35.0 implementation may start naive and optimize only if benchmarks show a bottleneck.

---

### E. Inputs and Outputs

#### E.1 Public function signatures (proposed Rust API)

```rust
/// Trajectory-reconstruction design: greedy sequential selection of p design points
/// minimizing the integrated BLUP prediction MSE of x̂(t).
///
/// Uses the TR criterion from Ji & Müller (2017): maximizes
///   TRCri(S) = trapz( diag( C_{S,:}^T (C_{SS} + ridge·I)^{-1} C_{S,:} ) )
/// where C = Φ diag(λ) Φ^T is the process covariance assembled from the PACE model.
#[must_use]
pub fn foptdes_trajectory(
    model: &PaceFpcaResult,
    candidate_grid: Option<&[usize]>,  // indices into model.argvals; None = all m
    budget: usize,                      // p ≥ 1, ≤ |candidate_grid|
    ridge: Option<f64>,                 // diagonal regularizer; None → model.sigma2
) -> Result<FOptDesResult, FdarError>

/// FPC-score-prediction design: greedy sequential selection minimizing the
/// posterior score covariance under A-optimality (trace) or D-optimality (log-det).
#[must_use]
pub fn foptdes_scores(
    model: &PaceFpcaResult,
    candidate_grid: Option<&[usize]>,
    budget: usize,
    criterion: ScoreCriterion,  // AOptimal or DOptimal
    ridge: Option<f64>,
) -> Result<FOptDesResult, FdarError>

/// Score any candidate design set against both TR and score criteria.
/// Independent of the greedy loop — useful for evaluating an existing design.
#[must_use]
pub fn foptdes_eval(
    model: &PaceFpcaResult,
    design: &[usize],   // indices into model.argvals (sorted, non-empty, in-bounds)
    ridge: Option<f64>,
) -> Result<FOptDesEval, FdarError>

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScoreCriterion {
    /// Minimize trace of posterior score covariance (sum of posterior variances).
    AOptimal,
    /// Minimize log-det of posterior score covariance (generalized posterior variance).
    DOptimal,
}
```

#### E.2 Result struct

```rust
/// Result of a greedy optimal design selection (TR or score criterion).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FOptDesResult {
    /// Selected design point indices into model.argvals (sorted ascending, length = budget).
    pub design_indices: Vec<usize>,
    /// Selected design point time locations: model.argvals[design_indices[i]] (length = budget).
    pub design_times: Vec<f64>,
    /// Achieved criterion value at each greedy step (length = budget, non-decreasing).
    /// For TR: TRCri(S_iter) = integral of explained prediction variance.
    /// For A-optimal: ΔA(S_iter) = trace reduction = trace(Λ) − trace(Cov(ξ|Y_{S_iter})).
    /// For D-optimal: ΔD(S_iter) = log det information gain.
    pub criterion_curve: Vec<f64>,
    /// R² at each greedy step (length = budget, non-decreasing, in [0,1]).
    /// For TR: criterion_curve[i] / Σ_k λ_k.
    /// For A-optimal: criterion_curve[i] / Σ_k λ_k.
    /// For D-optimal: criterion_curve[i] / log det(Λ).
    pub r2_curve: Vec<f64>,
    /// Final achieved R² (r2_curve[budget-1]).
    pub r2: f64,
    /// Total process variance Σ_k λ_k (denominator of R²_TR and R²_A).
    pub total_variance: f64,
}

/// Multi-criterion evaluation of a given design set.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FOptDesEval {
    /// IPV — integrated prediction variance: ∫ Var(x̂(t)|Y_S) dt (TR absolute criterion).
    pub integrated_pred_variance: f64,
    /// TRCri — explained variance integral = total_variance − IPV.
    pub tr_criterion: f64,
    /// R²_TR: tr_criterion / total_variance.
    pub r2_trajectory: f64,
    /// Trace of posterior score covariance Cov(ξ|Y_S) (A-criterion absolute value).
    pub trace_score_cov: f64,
    /// Trace reduction ΔA = trace(Λ) − trace_score_cov (A-criterion gain).
    pub a_criterion: f64,
    /// R²_A: a_criterion / trace(Λ) = a_criterion / total_variance.
    pub r2_score_a: f64,
    /// Log-det of posterior score covariance (D-criterion absolute value).
    pub logdet_score_cov: f64,
    /// Log-det reduction ΔD = log det(Λ) − logdet_score_cov (D-criterion gain).
    pub d_criterion: f64,
    /// Total process variance Σ_k λ_k.
    pub total_variance: f64,
}
```

---

### F. Edge Cases and Robustness

#### F.1 σ² → 0 (noiseless or very small noise)

When σ² is small relative to the eigenvalues, C(S,S) + σ²I_p can be nearly singular if the design points are clustered in a low-variance region. Mitigation: the `ridge` parameter (defaulting to `model.sigma2`) always adds a positive diagonal. The implementor should enforce `ridge ≥ 1e-10` internally regardless of what the caller passes, mirroring the 1e-8 ridge retry in `pace_fpca.rs` (lines 480–489).

#### F.2 Near-collinear design points

Two candidate grid points adjacent in the work grid may have nearly identical eigenfunction values, making C(S,S) ill-conditioned. The greedy `G \ S` exclusion prevents exact duplicates; the ridge handles near-duplicates. Document that design points on a fine grid (m > 100) may produce near-collinear C(S,S) and that ridge should be ≥ 1e-6 in that regime.

#### F.3 Budget p ≥ |G| (over-budget)

Return `Err(FdarError::InvalidParameter)` if `budget > candidate_grid.len()`. If `budget == candidate_grid.len()`, the design trivially selects all candidates and R² is maximized (not an error, but the greedy loop runs and returns the full set).

#### F.4 budget = 0

Return `Err(FdarError::InvalidParameter { parameter: "budget", message: "must be at least 1" })`.

#### F.5 budget = 1

The greedy loop runs once. For TR: the best single point maximizes TRCri({g}) = (C[g,:] · w)² / (C[g,g] + σ²) · trapz(C[:,g]² / (C[g,g]+σ²)). The result is typically near the largest eigenfunction extremum.

#### F.6 K = 1 (single component)

Degenerates cleanly: C(s,t) = λ₁ φ₁(s) φ₁(t). The best single TR point is the t_g maximizing φ₁(t_g)² (normalized by the denominator). A-optimal and D-optimal score criteria coincide at K=1 (both are the scalar posterior variance). This is a useful known-answer test case.

#### F.7 Monotone criterion invariant

The greedy forward selection guarantees `criterion_curve[i] ≥ criterion_curve[i-1]` for all i. This is because adding a point to S can only increase TRCri (the projection onto the span of {C(·, t_{s_j})} can only grow). The implementation should `debug_assert!` this invariant in test builds.

#### F.8 Parallel inner loop determinism

When the parallel feature flag is enabled, the inner loop over candidates runs in parallel. The greedy argmax must be computed from the full result vector after all parallel evaluations complete — never from a racing first-found maximum. Use `iter_maybe_parallel!` over the candidates, collect results into a `Vec<f64>`, then `argmax` the vector sequentially (with the tie-breaking rule: smallest index wins). This matches the existing pattern in `pace_fpca.rs`.

---

## Feature Landscape

### Table Stakes (Users Expect These)

Features the PACE optimal design module must provide for users to consider it complete relative to the MATLAB `FOptDes` and R `fdapace` reference.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Trajectory-reconstruction greedy design** (`foptdes_trajectory`) | The canonical FOptDes criterion — minimize integrated BLUP prediction MSE of x̂(t). Every published reference implements this first. | MEDIUM | Builds C = Φ diag(λ) Φᵀ then greedy loop with TRCri. Uses `cholesky_solve` from `linalg.rs` and `simpsons_weights` from `helpers.rs`. |
| **FPC-score-prediction greedy design A-optimal** (`foptdes_scores(AOptimal)`) | Minimizing trace(Cov(ξ|Y_S)) is the natural complement to TR when FPC scores (not trajectories) are the downstream use. | MEDIUM | Reuses the A_mat computation pattern from `pace_fpca.rs` lines 547–558, generalized to a prospective design S. |
| **Design-criterion evaluator for arbitrary candidate sets** (`foptdes_eval`) | Users must evaluate a hand-crafted or historically collected design against the model — independent of the greedy loop. | LOW | Wraps TRCri, A-crit, D-crit computations; no loop. All three criteria computed in one call sharing one C(S,S) inverse. |
| **Two-stage workflow on `PaceFpcaResult`** | The standard usage pattern: fit PACE once, then run design as a pure post-processing step. Matches the MATLAB/R `FOptDes` pipeline. | LOW | Takes `&PaceFpcaResult` as input; no re-estimation. The design module has no FPCA code. |
| **`criterion_curve` / per-step R² output** | Researchers need to plot R² vs budget p to choose the minimum sufficient budget. MATLAB FOptDes and R fdapace both return this. | LOW | Accumulated during the greedy loop at zero extra cost. |
| **Ridge regularization** | Required when C(S,S) is near-singular. R fdapace uses CV over `RidgeCand`; minimal version uses σ² itself or a caller-supplied scalar. | LOW | Default ridge = `model.sigma2`. Accept `Option<f64>` for expert control. Enforce `ridge ≥ 1e-10` internally. |
| **Deterministic reproducibility** | All outputs must be identical bit-for-bit across runs and platforms (no randomness). | LOW | No RNG needed. Tie-breaking by smallest index ensures determinism in the presence of equal criterion values. |

### Differentiators (Competitive Advantage)

Features that go beyond the MATLAB/R reference and add specific value for the `fdars` Rust implementation.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **D-optimal score design** (`foptdes_scores(DOptimal)`) | D-optimality (log-det) has stronger theoretical properties than A-optimality for multivariate score prediction; fdapace does not expose this. | MEDIUM | log-det from Cholesky factor of K×K posterior covariance. Straightforward given the A_mat already computed. |
| **`FOptDesEval` multi-criterion struct** | All three criteria (TR, A-optimal, D-optimal) evaluated in a single call sharing one C(S,S) Cholesky factorization. No reference implementation does this. | LOW | Three computations, one solve: O(p³ + K² p) total. |
| **Parallel candidate evaluation** (via `parallel` feature flag) | Inner loop over G\S candidates is embarrassingly parallel. For dense grids (m=200) and large budget (p=10), speedup is proportional to thread count. | MEDIUM | Use `iter_maybe_parallel!` macro. Collect results into `Vec<f64>`, then sequential argmax with tie-break. Maintains determinism. |
| **`total_variance` field in result** | Exposes Σ_k λ_k so callers can compute their own R² variants without re-parsing the model. | LOW | One extra field in `FOptDesResult` and `FOptDesEval`. Zero cost. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Global exhaustive search over all C(m,p) combinations** | For small p, exhaustive search finds the true optimum (greedy is only locally optimal). | C(m,p) grows as m^p/p! — for m=100, p=5: ~75 million combinations. Combinatorially infeasible for production. Ji & Müller (2017) show greedy is near-optimal for smooth functional data. | Use greedy. Document the bounded suboptimality in function-level docs. Exhaustive search is available in fdapace only for tiny p via `isSequential=FALSE`. |
| **Bayesian D-optimal design via MCMC/Metropolis-Hastings** | Sounds more principled; Huang et al. 2025 describe this extension. | Massively more complex (MCMC sampling), non-deterministic, slow, and requires a prior distribution over the FPCA model parameters — far beyond a pure design step on a fitted model. | Use greedy D-optimal on the posterior covariance (implemented above). The MCMC extension is a separate research-grade feature. |
| **Scalar-response prediction design (SRCri, fdapace `isRegression=TRUE`)** | FOptDes with a scalar response optimizes measurement times for predicting a specific outcome. | Requires a scalar response Resp as input to the design step, coupling design to a specific outcome variable and making the function stateful. The score-prediction criterion (A/D-optimal) is response-agnostic and generalizes. | Implement SR design in a later phase (v0.36.0+) once the base two criteria ship and user demand is confirmed. |
| **Automatic σ² estimation inside the design step** | Users may want the design function to auto-estimate σ² rather than using the value from `PaceFpcaResult`. | σ² estimation belongs to `pace_fpca`; doing it again inside the design step duplicates machinery and introduces API coupling. The `PaceFpcaResult` already carries `sigma2`. | The two-stage workflow is correct: set σ² during `pace_fpca` estimation, then pass the result to the design function. |
| **Non-greedy optimization (gradient descent on criterion)** | The criterion is differentiable in t; gradient methods could find better local optima. | The criterion is defined over a discrete candidate grid — gradient methods require continuous relaxation and add implementation complexity with no practical benefit for the expected grid sizes. | Greedy on the discrete grid is the standard approach (Ji & Müller 2017, fdapace). |

---

## Feature Dependencies

```
PaceFpcaResult (already shipped: pace_fpca.rs)
    ├──required by──> foptdes_trajectory     (reads eigenvalues, eigenfunctions, sigma2, argvals)
    ├──required by──> foptdes_scores         (reads eigenvalues, eigenfunctions, sigma2)
    └──required by──> foptdes_eval           (reads eigenvalues, eigenfunctions, sigma2, argvals)

linalg::cholesky_solve (already shipped)
    └──required by──> all three public functions
                      (solves (C(S,S) + ridge·I)^{-1} at each greedy step)

helpers::simpsons_weights (already shipped)
    └──required by──> foptdes_trajectory, foptdes_eval
                      (numerical integration of prediction variance over work grid)

parallel::iter_maybe_parallel! (already shipped)
    └──enhances──> foptdes_trajectory, foptdes_scores
                   (parallel inner candidate evaluation loop, feature-gated)

ScoreCriterion enum
    └──required by──> foptdes_scores

FOptDesResult
    └──returned by──> foptdes_trajectory, foptdes_scores

FOptDesEval
    └──returned by──> foptdes_eval

process_covariance_matrix [private helper]
    └──shared by──> all three public functions
                    (builds C = Φ diag(λ) Φᵀ once; O(m² K))

tr_criterion_fn [private helper]
    └──shared by──> foptdes_trajectory (greedy loop inner) + foptdes_eval

a_criterion_fn [private helper]
    └──shared by──> foptdes_scores(AOptimal) (greedy loop inner) + foptdes_eval

d_criterion_fn [private helper]
    └──shared by──> foptdes_scores(DOptimal) (greedy loop inner) + foptdes_eval
```

### Dependency Notes

- **All three functions require `PaceFpcaResult`:** The design step is a pure consumer of the fitted model; no estimation code belongs in `optimal_design.rs`.
- **`foptdes_eval` reuses the same criterion helpers as the greedy loops:** Factor TRCri, A-crit, D-crit into private functions (`tr_criterion_fn`, etc.) so both the greedy loops and `foptdes_eval` call the same code path. This is the primary deduplication opportunity.
- **`cholesky_solve` is the critical primitive:** Every criterion evaluation at every step inverts a p×p matrix. Reuse `linalg::cholesky_solve` exactly as `pace_fpca.rs` does — no new linear algebra dependency.
- **`simpsons_weights` vs trapezoidal:** `helpers::simpsons_weights` is already used in `pace_fpca.rs`; use it for the TR integral to maintain numerical consistency. The fdapace R code uses `trapzRcpp`; the numerical difference is negligible for smooth integrands over 50+ points.

---

## MVP Definition

### Phase 64 — Core criterion machinery + TR greedy design (P1)

Minimum viable deliverable that validates the mathematical correctness and establishes the integration pattern.

- [ ] `optimal_design.rs` — new top-level module in `fdars-core/src/`
- [ ] `process_covariance_matrix` private helper: builds C = Φ diag(λ) Φᵀ (m×m FdMatrix), O(m² K)
- [ ] `TRCri(S, C, ridge, weights)` private helper: integrated explained-variance criterion
- [ ] `foptdes_trajectory` public function: greedy TR design → `FOptDesResult`
- [ ] `FOptDesResult` struct with `design_indices`, `design_times`, `criterion_curve`, `r2_curve`, `r2`, `total_variance`
- [ ] Full input validation: `budget ≥ 1`, `budget ≤ |G|`, `model.ncomp ≥ 1`, `ridge > 0`, all candidate indices in-bounds
- [ ] Crate-root re-export; doctest example
- [ ] Tests: monotone criterion curve (VT-1); K=1 known-answer best point near φ₁ extremum (VT-2); 2-component synthetic model R²_TR > 0.8 with p=3 (VT-3 partial); full-grid R²→1 (VT-5); error paths (VT-7)

### Phase 65 — Score-prediction criteria + evaluator (P1)

- [ ] `a_criterion_fn` and `d_criterion_fn` private helpers (reusing A_mat pattern from `pace_fpca.rs` lines 547–558)
- [ ] `foptdes_scores` public function: greedy A/D-optimal score design → `FOptDesResult`
- [ ] `ScoreCriterion` enum (`AOptimal`, `DOptimal`)
- [ ] `foptdes_eval` public function → `FOptDesEval` (computes TR, A, D criteria in one call)
- [ ] `FOptDesEval` struct with all fields
- [ ] Tests: A-optimal + D-optimal on 2-component synthetic model (VT-3 full); greedy monotone on A-crit (VT-1 variant); `foptdes_eval` consistency with greedy result (VT-4); D-optimal K=1 matches A-optimal (VT-6); error paths for `foptdes_eval` (VT-7)
- [ ] Doctest example showing the full two-stage workflow: `pace_fpca` → `foptdes_trajectory` → `foptdes_eval`

### Add After Validation (v0.35.x)

- [ ] Parallel inner candidate loop via `iter_maybe_parallel!` — add when m > 100 benchmarks show measurable speedup (likely Phase 65 or a follow-on tweak)
- [ ] Scalar-response prediction design (`SRCri`, fdapace `isRegression=TRUE` mode) — requires scalar response Resp as design input; defer until user demand is confirmed

### Future Consideration (v2+)

- [ ] Bayesian / MCMC-based D-optimal design (Huang et al. 2025 extension)
- [ ] Adaptive ridge selection via cross-validation (fdapace `RidgeCand` mode)
- [ ] Multi-subject design accounting for between-subject heterogeneity in the design matrix

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| TR greedy design (`foptdes_trajectory`) | HIGH — canonical FOptDes, directly from Ji & Müller 2017 | MEDIUM | P1 |
| `foptdes_eval` independent scorer | HIGH — evaluate any existing design against the PACE model | LOW | P1 |
| A-optimal score design | HIGH — score prediction is the primary downstream use of PACE scores | MEDIUM | P1 |
| `criterion_curve` / per-step R² output | HIGH — essential for budget-selection workflow | LOW | P1 |
| D-optimal score design | MEDIUM — stronger theoretical properties; not in fdapace | MEDIUM | P2 |
| `FOptDesEval` multi-criterion evaluator | MEDIUM — convenience; shares one Cholesky solve | LOW | P2 |
| Parallel inner candidate loop | LOW — fast enough sequentially for typical m ≤ 200, p ≤ 10 | MEDIUM | P3 |
| Scalar-response design (SRCri) | MEDIUM — needed for regression design tasks | MEDIUM | P2 |

---

## Verification / Known-Answer Tests

Concrete, self-contained test specifications — each verifiable without external data.

### VT-1: Monotone criterion curve (TR and A-optimal)

For any valid `PaceFpcaResult` with K ≥ 1 and budget p ≥ 2, the returned `criterion_curve` must be non-decreasing:

```rust
for i in 1..budget {
    assert!(criterion_curve[i] >= criterion_curve[i-1] - 1e-10,
        "criterion_curve not monotone at step {}", i);
}
```

Greedy forward selection guarantees this; a violation indicates a criterion evaluation bug.

### VT-2: K=1 known-answer — best single point near φ₁ extremum

Synthetic: K=1, φ₁(t) = √2 sin(πt) on [0,1], λ₁=1.0, σ²=0.01, m=51. Budget p=1.

The best single TR design point should have `design_times[0]` within 2/m of 0.5 (the maximum of φ₁²). This tests that the covariance matrix is correctly assembled and the criterion is correctly evaluated.

### VT-3: Two-component orthonormal model

Synthetic: K=2, φ₁(t) = √2 sin(πt), φ₂(t) = √2 cos(πt), λ₁=1.0, λ₂=0.5, σ²=0.01, m=51 on [0,1].

- `foptdes_trajectory` with p=3: `r2 > 0.85` (three points cover both eigenfunctions well)
- `foptdes_scores(AOptimal)` with p=3: `r2 > 0.80`
- Both: `criterion_curve[1] > criterion_curve[0]` (strictly improving)
- Both: `design_indices` are a sorted subset of 0..51

### VT-4: `foptdes_eval` consistency with greedy result

After `foptdes_trajectory` returns `result`, calling `foptdes_eval(model, &result.design_indices, ridge)` must return `tr_criterion` within 1e-10 (absolute) of `result.criterion_curve[budget-1]`. Tests that the greedy loop and the evaluator use the same formula.

### VT-5: Full-grid design saturates criterion

Budget = m (all grid points selected). `r2 > 1.0 − sigma2 / (sigma2 + lambda_1)`. For the VT-3 model: `r2 > 0.99`. Tests the upper bound of the criterion.

### VT-6: K=1 A-optimal and D-optimal coincide

For K=1 (scalar score), A-crit = D-crit = the scalar posterior variance λ₁ − λ₁² φ₁(S)ᵀ(C(S,S)+σ²I)⁻¹ φ₁(S). The greedy best single point from `foptdes_trajectory` and `foptdes_scores(AOptimal)` should select the same index (within floating-point equality). Tests criterion consistency.

### VT-7: Error paths

| Call | Expected error |
|------|----------------|
| `budget = 0` | `Err(InvalidParameter { parameter: "budget" })` |
| `budget > candidate_grid.len()` | `Err(InvalidParameter { parameter: "budget" })` |
| `design = []` in `foptdes_eval` | `Err(InvalidParameter { parameter: "design" })` |
| `design` index ≥ m | `Err(InvalidDimension { parameter: "design" })` |
| `ridge = Some(0.0)` | `Err(InvalidParameter { parameter: "ridge" })` |
| `ridge = Some(-0.1)` | `Err(InvalidParameter { parameter: "ridge" })` |
| `model.ncomp = 0` (impossible in practice; guard defensively) | `Err(InvalidParameter { parameter: "model" })` |

---

## Reference Implementations

| Reference | Version | Criterion | Algorithm | Notes |
|-----------|---------|-----------|-----------|-------|
| MATLAB PACE `FOptDes` | 2.17 | TR (MISE) + SR (scalar regression) | Greedy sequential (`isSequential=TRUE`) | Original MATLAB reference. No score A/D-optimality exposed. |
| R `fdapace` `FOptDes` | 0.6.0 (tPACE) | TR + SR | Greedy (`isSequential=TRUE`) or exhaustive (`=FALSE`) | Returns OptDes (time values), R2, R2adj. Exhaustive only for small p. Cross-validates ridge over `RidgeCand`. |
| Ji & Müller (2017) | JRSSB 79:859-876 | TR (R²_TR) + SR (R²_SR) | Greedy sequential (§5.3) | Primary theoretical reference. Proves near-optimality of greedy. |
| Huang et al. (2025) | WIREs review | TR + A/D/E-optimal + Bayesian | Various | Extends Ji & Müller; introduces explicit A/D-optimality for scores. Bayesian extensions are out of scope for v0.35.0. |

---

## Sources

- Ji, H. & Müller, H.G. (2017). Optimal Designs for Longitudinal and Functional Data. *JRSSB* 79(3), 859-876. [Oxford Academic](https://academic.oup.com/jrsssb/article/79/3/859/7040637) / [arXiv:1604.05375](https://arxiv.org/pdf/1604.05375)
- fdapace R package (tPACE), v0.6.0: [FOptDes documentation](https://rdrr.io/cran/fdapace/), [BestDes_TR.R source](https://raw.githubusercontent.com/functionaldata/tPACE/master/R/BestDes_TR.R), [BestDes_SR.R source](https://raw.githubusercontent.com/functionaldata/tPACE/master/R/BestDes_SR.R), [FOptDes.R source](https://github.com/functionaldata/tPACE/blob/master/R/FOptDes.R)
- Yao, F., Müller, H.G. & Wang, J.L. (2005). Functional Data Analysis for Sparse Longitudinal Data. *JASA* 100(470), 577-590. — PACE BLUP formula (ξ_ik, Σ_yi, Ω_i); already implemented in `pace_fpca.rs`. [utstat.utoronto.ca](http://utstat.utoronto.ca/fyao/2005-jasa.pdf)
- Huang, H. et al. (2025). Optimal Experimental Designs for Sparse Functional Data: A Review. *WIREs Computational Statistics*. [DOI:10.1002/wics.70039](https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.70039)
- arXiv:2508.00176 — New Pilot-Study Design in FDA (supplementary derivation of IPV formula and score posterior covariance). [arXiv:2508.00176](https://arxiv.org/pdf/2508.00176)

---
*Feature research for: FOptDes optimal experimental design for sparse FDA (v0.35.0)*
*Researched: 2026-09-02*
