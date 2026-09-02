# Pitfalls Research

**Domain:** Optimal Experimental Design on PACE/BLUP sparse-FDA model (FOptDes) — Rust numerical FDA library (fdars-core v0.35.0)
**Researched:** 2026-09-02
**Confidence:** HIGH

---

## Critical Pitfalls

### Pitfall 1: Wrong Σ_yi Assembly — Missing σ² Ridge or Misidentifying Who Carries Noise

**What goes wrong:**
The observation covariance matrix for curve i at selected points S is:

```
Σ_yi = Φ_S diag(λ) Φ_Sᵀ + σ²I_{|S|}
```

where Φ_S is |S|×K (eigenfunctions at selected times), λ is the K-vector of eigenvalues, and σ²
is the scalar measurement-error variance. Two common errors:

1. **Omitting σ²I entirely** — Σ_yi becomes rank-at-most-K (singular for |S| > K). Cholesky
   fails. Prediction intervals collapse to zero. The Woodbury/Schur update formula blows up.

2. **Adding σ² to the full covariance surface before calling this formula** — double-counting.
   `pace_fpca.rs` comments explicitly warn: "Do NOT subtract σ² from the surface before
   eigendecomposition — σ² enters only as the ridge term σ²I in Σ_yi." The same discipline
   must hold here: eigenvalues λ and eigenfunctions Φ come from the PACE eigendecomposition of
   the *smoothed* covariance (which already absorbed σ²); only the per-curve Σ_yi for the
   design problem gets the +σ²I ridge.

3. **Using the work-grid covariance matrix (m×m) instead of the observation covariance** — the
   full m×m matrix is never assembled for the design step; only the |S|×|S| block at candidate
   points is needed.

**Why it happens:**
Developers conflate the PACE covariance surface (m×m smoothed bivariate covariance from
`cov_irreg`) with the per-design-point covariance Σ_yi needed for BLUP/optimal-design math.
The PACE paper uses consistent notation but the two matrices serve different roles.

**How to avoid:**
- Copy the exact Σ_yi assembly from `pace_fpca.rs` lines 461–474: build Φ_S as |S|×K by
  interpolating eigenfunctions at candidate times, then `Σ_S[row,col] = Σ_k φ_k(t_row) λ_k
  φ_k(t_col)` plus δ_{row,col} σ²`.
- Assert `sigma2 > 0` at the start of every design-criterion function (same guard as
  `PaceFpcaConfig.sigma2` validation).
- Add a test: with 1 design point and 1 eigenfunction, Σ_yi = λ φ(t)² + σ² is a scalar;
  verify against hand-computed value.

**Warning signs:**
- Cholesky fails immediately when |S| > K (singular Σ_yi without σ²I).
- Predicted score variance is negative or the criterion is non-monotone as points are added.
- `ComputationFailed` Cholesky errors on valid, non-degenerate inputs.

**Phase to address:** Phase 64 (core criterion evaluation — design-criterion function). Gate:
verify Σ_yi values numerically against hand-computed formula for a 2-eigenfunction synthetic.

---

### Pitfall 2: Score Posterior Covariance — Inverting the Wrong Matrix

**What goes wrong:**
The posterior covariance of scores given selected observations Y_S is:

```
Cov(ξ | Y_S) = (diag(1/λ) + σ⁻² Φ_Sᵀ Φ_S)⁻¹
```

This p×p matrix (p = ncomp) is cheap to invert for p ≤ 20. Two common errors:

1. **Forgetting the prior term `diag(1/λ)`** — the matrix becomes σ⁻² Φ_Sᵀ Φ_S, which is rank
   ≤ |S|. When |S| < K, it is singular. Even when |S| ≥ K, omitting the prior produces the
   wrong formula (equivalent to uniform/improper prior on scores instead of the PACE Gaussian
   prior).

2. **Using the Woodbury identity in the wrong direction** — computing `Σ_yi^{-1}` via p×p
   inversion is correct (Woodbury on the large n_i×n_i matrix), but some implementations
   instead work with the big matrix and pay O(|S|³) per candidate. At |S| = m (work-grid
   size, up to 51+ points), this is up to 1000× more expensive than the p×p version.

3. **Recomputing from scratch vs. rank-1 update** — the greedy loop adds one point at a time.
   If you recompute the full p×p inverse at each step by Cholesky-solving the p×p system, it
   costs O(G · K³) per greedy step where G = number of candidates. If you instead track the
   *running p×p matrix* M_S = diag(1/λ) + σ⁻² Φ_Sᵀ Φ_S and do rank-1 updates
   `M_{S∪{t}} = M_S + σ⁻² φ(t) φ(t)ᵀ`, each update is O(K²) and the new inverse follows
   from the Sherman-Morrison formula in O(K²) as well. This is the efficient path but
   correctness must be proven first.

**Why it happens:**
The Woodbury/matrix-inversion lemma has two equivalent forms; picking the wrong one (or forgetting
the prior) silently produces a plausible-looking but incorrect matrix. No compile-time error flags
this.

**How to avoid:**
- Implement score-posterior covariance as the straightforward p×p Cholesky solve first:
  assemble `M = diag(1/λ) + σ⁻² Φ_Sᵀ Φ_S`, then invert via Cholesky. Verify against
  the brute-force `Σ_yi^{-1}` Woodbury expansion on a 2-eigenfunction test case.
- Known-answer check: with |S| = 0 (no observations), `Cov(ξ | ∅) = diag(λ)` (prior only);
  verify the formula degenerates correctly (M = diag(1/λ), M^{-1} = diag(λ)).
- Add rank-1 Sherman-Morrison update only after the brute-force version passes all tests,
  with a side-by-side comparison assertion in tests.

**Warning signs:**
- A-optimality (trace of posterior covariance) *increases* when a point is added — this means
  the formula is wrong (the trace must never increase; each new observation can only reduce or
  maintain the posterior variance).
- Posterior covariance is not positive definite (eigenvalues ≤ 0) even with reasonable inputs.
- Score posterior variance equals prior variance diag(λ) even after adding points — omitted
  Φ_Sᵀ Φ_S term.

**Phase to address:** Phase 64 (core criterion evaluation). Gate: monotone-decrease assertion
on A-optimality trace; known-answer tests for |S|=0 and |S|=1 with analytic ground truth.

---

### Pitfall 3: Integrated Prediction Variance — Missing Quadrature Weights or Mean Term

**What goes wrong:**
The trajectory-reconstruction criterion is the integrated BLUP prediction MSE:

```
MSE(S) = ∫ Var(x̂(t) - x(t)) dt ≈ Σ_j w_j · Var(x̂(t_j) - x(t_j))
```

where w_j are the Simpson weights from `helpers::simpsons_weights` over the work grid, and the
pointwise prediction variance is:

```
Var(x̂(t_j) - x(t_j)) = φ(t_j)ᵀ Ω_S φ(t_j)
```

with Ω_S = Cov(ξ | Y_S) the posterior score covariance (pitfall 2). Three common errors:

1. **Summing without weights** — `Σ_j Var(x̂(t_j))` instead of `Σ_j w_j · Var(x̂(t_j))`.
   Produces the wrong value for non-uniform grids and fails to approximate the integral
   for non-uniform work grids. Even for uniform grids, the result is off by a constant factor
   equal to the grid spacing.

2. **Omitting the mean-function term** — the BLUP prediction variance includes both score
   uncertainty (through Ω_S) and any residual measurement noise not captured in the model.
   In the PACE framework the correct integrated prediction variance is the weighted integral
   of φ(t_j)ᵀ Ω_S φ(t_j), which already accounts for the full covariance structure via Ω_S.
   If Ω_S is computed incorrectly (e.g., missing σ²I in Σ_yi), the integrated variance is
   also wrong.

3. **Not squaring the eigenfunction values** — computing `Σ_k Ω_S[k,k] φ_k(t_j)` (linear
   combination) instead of `Σ_{k,l} Ω_S[k,l] φ_k(t_j) φ_l(t_j)` (quadratic form). The
   off-diagonal terms of Ω_S are nonzero in general and must be included. This is the same
   pattern as the confidence-band variance in `pace_fpca.rs` lines 562–576 — mirror that code.

**Why it happens:**
Developers prototype with unweighted sums (simpler loop), then forget to add weights. The
missing-weights version gives results that look numerically plausible but are dimensionally
incorrect (units off by argvals-range / m).

**How to avoid:**
- Always call `simpsons_weights(&pace_result.argvals)` and store the result; never use a
  uniform weight of `1.0` or `1.0/m`.
- Mirror the confidence-band quadratic form in `pace_fpca.rs` exactly for the Ω_S quadratic
  form — the code structure is identical.
- Known-answer test: for a single-eigenfunction model with φ(t) = √2 sin(πt) on [0,1], the
  prior integrated variance = λ · ∫ φ²(t) dt = λ · 1.0 (by L² orthonormality). With
  Simpson weights on a 51-point grid, the integral should be ≈ 1.0 to within 1e-6. Adding
  one observation at t* reduces it by λ² φ(t*)² / (λ φ(t*)² + σ²). Verify both against
  analytic answers.

**Warning signs:**
- Integrated MSE for |S|=0 is not equal to `Σ_k λ_k` (total prior variance) when computed
  correctly. (With Simpson weights on [0,1] and L²-orthonormal eigenfunctions, this identity
  holds exactly in theory.)
- Criterion value is proportional to m (grid size) instead of being grid-size–invariant.
- Criterion does not converge as m increases.

**Phase to address:** Phase 64 (core criterion evaluation). Gate: identity check
`MSE(∅) ≈ Σ_k λ_k` on a synthetic 2-eigenfunction dataset with orthonormal eigenfunctions.

---

### Pitfall 4: Optimality-Sign Convention — Criterion Must Non-Increase as Points Are Added

**What goes wrong:**
Both A-optimality (trace of posterior covariance) and D-optimality (log-det of posterior
covariance) must **decrease or stay flat** as design points are added — more information always
reduces or maintains posterior uncertainty. In the greedy loop, the algorithm selects the point
that **minimizes** the criterion. Two sign errors:

1. **D-optimality sign flip** — `log det Cov(ξ|Y_S)` is always ≤ 0 (since all eigenvalues of
   the posterior covariance are ≤ the prior eigenvalues λ_k ≤ some λ_max). Implementing
   `+det` instead of `-log det`, or forgetting that minimizing log-det *increases* information
   (D-optimal is max-det *information*, min-det *covariance*). The correct objective to
   *minimize* is `log det Cov(ξ|Y_S)` (equivalently `-log det I(S)` where I is the
   information matrix). Greedy *minimization* of this quantity is correct.

2. **Monotonicity failure** — if the objective ever increases when a new point is selected, the
   implementation has a bug. The mathematical property is: adding a point to S can only reduce
   or maintain `trace Cov(ξ|Y_{S∪{t}}) ≤ trace Cov(ξ|Y_S)` and
   `log det Cov(ξ|Y_{S∪{t}}) ≤ log det Cov(ξ|Y_S)` because the posterior covariance is a
   decreasing-in-information semidefinite sequence.

3. **Treating "larger = better" criterion as minimization** — some formulations express
   D-optimality as maximizing the information (det of Fisher matrix). When translating from
   such a reference, forgetting to negate produces a maximizer in the greedy loop, which
   selects the *worst* point at each step.

**Why it happens:**
D-optimality is expressed inconsistently across references — some maximize log-det information,
some minimize log-det covariance. The sign convention is easy to flip when adapting formulae
from different papers.

**How to avoid:**
- Define a single `DesignCriterion` enum: `AOpt` (minimize trace Cov) and `DOpt` (minimize
  log-det Cov). Both are minimized by the greedy loop.
- Add a monotone-decrease assertion as a test: run greedy selection for K+1 steps on a
  synthetic dataset and assert `criterion[k+1] <= criterion[k] + 1e-12` for all k.
- For D-optimality, verify: with |S|=0, log-det = Σ_k log λ_k. With |S|=m (full grid),
  log-det approaches -∞ as σ²→0. Always log-det ≤ log-det at previous step.

**Warning signs:**
- Any greedy step produces `criterion[k+1] > criterion[k]`.
- With many design points, A-optimality criterion stays at the prior value (Σ_k λ_k).
- D-optimality criterion is positive (would require all posterior eigenvalues > 1).

**Phase to address:** Phase 64 (core criterion evaluation) and Phase 65 (greedy selection
loop). Gate: monotone-decrease test on a 2-eigenfunction synthetic with both AOpt and DOpt.

---

### Pitfall 5: Near-Duplicate Candidate Points and σ²→0 — Rank Deficiency in Σ_yi

**What goes wrong:**
When two candidate points t_a ≈ t_b are both in the selected set S, the rows of Φ_S become
nearly identical (φ_k(t_a) ≈ φ_k(t_b) by continuity of eigenfunctions). Σ_yi becomes
nearly singular — its smallest eigenvalue approaches σ², and the Cholesky factorization either
fails or is numerically unreliable. Two related failure modes:

1. **Collinear candidates from a fine candidate grid** — if the candidate set is the full work
   grid (51+ evenly spaced points), consecutive candidates at spacing h ≈ 0.02 may produce
   near-duplicate eigenfunction rows. The PACE ridge stabilization in `pace_fpca.rs` (1e-8
   retry) guards against this but the retry is a safety net, not a design choice.

2. **σ²→0 with exact duplicates** — if σ² is very small (e.g., 1e-6) and candidates include
   exact work-grid duplicates, the diagonal of Σ_yi is dominated by the low-rank part
   Φ_S diag(λ) Φ_Sᵀ and the ridge is invisible. The Cholesky encounters a near-zero diagonal
   and returns ComputationFailed.

**Why it happens:**
The candidate set is often taken as the work grid for convenience, making consecutive selections
inherently close. σ² is sometimes set small for "near-noiseless" scenarios, inadvertently
removing the only regularization.

**How to avoid:**
- **Do not allow duplicate candidate times in S.** Track selected indices; exclude already-
  selected candidates from the greedy search at each step.
- **Validate σ² > minimum floor** (e.g., 1e-8) at the design entry point — mirror the
  `pace_fpca.rs` validation which requires `sigma2 > 0` strictly.
- **Carry the same ridge-retry logic** from `pace_fpca.rs` (add 1e-8 and retry once) into
  the criterion evaluation function, not just the BLUP solve.
- **Thin candidate grids** when the work grid is fine: take every other point, or allow the
  caller to supply a coarser candidate set.

**Warning signs:**
- Cholesky fails for small σ² (< 1e-4) even for valid well-separated design points.
- Greedy loop selects the same or adjacent time point twice.
- Criterion does not improve after several steps despite valid starting conditions.

**Phase to address:** Phase 64 (criterion evaluation — ridge-retry) and Phase 65 (greedy
loop — duplicate exclusion). Gate: run greedy selection for m steps (full grid) and assert
no index appears twice; verify Cholesky succeeds for σ² = 1e-4 on a 2-eigenfunction synthetic.

---

### Pitfall 6: Eigenfunctions at Candidate Points — Interpolation vs. Work-Grid Access

**What goes wrong:**
`PaceFpcaResult.eigenfunctions` is an m×K `FdMatrix` storing eigenfunction values on the
**work grid** only (at indices 0..m-1 corresponding to `pace_result.argvals[0..m-1]`). A
candidate point t* from a user-supplied set may not lie exactly on the work grid.

Two errors:

1. **Direct index access instead of interpolation** — indexing `eigenfunctions[(j, k)]` where
   `j` is an integer index into the work grid, then treating the grid index as if it were a
   continuous time point. If the caller supplies candidate times that are off-grid, there is
   no corresponding index.

2. **Linear interpolation beyond the grid range** — `helpers::linear_interp` clamps or
   extrapolates; candidate times outside `[argvals[0], argvals[m-1]]` produce undefined
   extrapolated values that can be arbitrarily wrong.

**Why it happens:**
The prototype uses the work grid itself as the candidate set (integer indices 0..m-1), so
direct array access works. When the API is later generalized to accept arbitrary candidate
times, the same index-based access silently returns wrong values.

**How to avoid:**
- In the canonical design API, accept candidate times as `&[f64]` and interpolate
  eigenfunctions at each candidate via `helpers::linear_interp(&pace_result.argvals,
  &ef_col, t_candidate)` — exactly the pattern in `pace_fpca.rs` line 457.
- Validate that all candidate times lie within the work grid range; return
  `FdarError::InvalidParameter` for out-of-range candidates.
- When candidates *are* the work grid, take the fast path: extract the column directly
  from `eigenfunctions.column(k)` without interpolation. Document this optimization clearly.

**Warning signs:**
- Design criterion computed from integer indices produces wrong values when candidate set
  is a rescaled grid (e.g., [0.0, 0.1, 0.2] vs. work grid [0.0, 0.02, 0.04, ...]).
- Off-grid candidate times silently return extrapolated eigenfunction values.

**Phase to address:** Phase 64 (criterion evaluation). Gate: compute criterion for a
candidate set that is a strict *subset* of the work grid (off-grid relative to a coarser
index space) and verify eigenfunctions are correctly interpolated.

---

### Pitfall 7: Greedy Argmin Determinism — Ties Must Break Deterministically

**What goes wrong:**
When two or more candidate points yield the same (or numerically indistinguishable) criterion
value, the greedy selection is ambiguous. If the tie-breaking relies on iteration order of a
HashMap, parallel rayon iteration order, or float comparison instability, the selected design
is non-deterministic across runs or platforms.

Two failure modes:

1. **Parallel candidate sweep with rayon** — `iter_maybe_parallel!(candidates).map(...)
   .min_by(...)` with rayon does not guarantee which tied minimum is returned; the result
   depends on thread scheduling. This violates `pace_fpca.rs`'s determinism test pattern.

2. **Float equality without tolerance** — two candidates may produce criterion values that
   differ by 1e-16 due to floating-point ordering of operations. A strict `<` comparison
   picks whichever appears first in a sequential sweep; this *is* deterministic but brittle —
   a minor refactor of the inner loop can silently reorder candidates.

**Why it happens:**
Developers parallelize the candidate sweep for speed and forget that `min_by` over a parallel
iterator is not stable. The problem may not appear in tests (which use small grids where ties
are unlikely) but surfaces in production with many candidates.

**How to avoid:**
- **Sequential sweep first.** In Phase 65, implement the greedy candidate sweep sequentially
  (not via rayon). Use `Iterator::enumerate().min_by(...)` on a sorted candidate list; ties
  break by index (lower index wins). This is deterministic and fast for K ≤ 51 candidates.
- **Parallelize only the criterion evaluation**, not the argmin. Collect all criterion values
  into a `Vec`, then take the sequential `argmin` with index tie-breaking.
- **Add a determinism test** (mirror `test_determinism` from `pace_fpca.rs`): run
  `greedy_design(...)` twice on the same inputs; assert the selected point sequences are
  identical via `assert_eq!`.
- **Document** that parallel feature does not change the selected design, only the evaluation
  speed.

**Warning signs:**
- Selected design differs between two identical calls in the parallel build.
- Design test fails intermittently with different selected point orderings.
- `cargo test -- --test-threads=1` passes while `cargo test` fails (thread-scheduling flakiness).

**Phase to address:** Phase 65 (greedy selection). Gate: determinism assertion on identical
inputs; verify greedy output is identical with and without `--features parallel`.

---

### Pitfall 8: Greedy O(G·p·cost) Blowup — Correctness First, Then Efficiency

**What goes wrong:**
The greedy loop evaluates the design criterion at each of G candidate points for each of
n_select greedy steps. If the criterion re-assembles Σ_yi from scratch at each evaluation,
total cost is O(n_select · G · |S|²·K + K³). For G = 51, K = 5, n_select = 10, this is
fully manageable (seconds). But if the candidate grid is large (G = 1000, K = 20,
n_select = 50), naive re-evaluation becomes a bottleneck — and the temptation is to cache
the intermediate matrices incorrectly.

Two failure modes:

1. **Incorrectly caching the Cholesky factor of Σ_yi across candidates** — Σ_yi depends on
   which points S are selected, not the candidate being evaluated. The same Σ_yi is valid
   for all G candidates at a given greedy step (since S is fixed at that step). Caching
   across *greedy steps* is wrong; caching across *candidates within one step* is correct.

2. **Implementing the rank-1 Sherman-Morrison update before the brute-force version is
   verified** — the update formula is more complex and easy to get wrong. A subtle sign error
   in the rank-1 update produces plausible-looking but wrong posterior covariances.

**Why it happens:**
Performance anxiety drives optimization before correctness. The PACE `pace_fpca.rs` comment
structure is a guide: correctness first, then optimization, each with benchmark evidence.

**How to avoid:**
- **Milestone discipline:** implement correctness-first brute-force in Phase 64; benchmark
  in Phase 65 if needed; add rank-1 updates only if benchmark shows a hotspot with G > 200.
- **The correct caching structure:** at each greedy step, pre-compute `M_S =
  diag(1/λ) + σ⁻² Φ_Sᵀ Φ_S` once (K×K), then evaluate the criterion for each candidate t*
  as a rank-1 update `M_{S∪{t*}} = M_S + σ⁻² φ(t*)φ(t*)ᵀ` without mutating M_S. This is
  already O(K²) per candidate — fast enough for K ≤ 20, G ≤ 1000.
- **Add a `#[must_use]` annotation** on the criterion function so the compiler warns if the
  result is discarded in a loop (a subtle correctness check).

**Warning signs:**
- Criterion values at step k+1 are identical to step k (M_S was mutated instead of copied).
- Benchmarks show that criterion evaluation scales as O(G²) rather than O(G·K²).

**Phase to address:** Phase 64 (correctness) and Phase 65 (greedy — only optimize if
benchmark evidence warrants).

---

## Codebase Integration Pitfalls

### Pitfall 9: Additive/Non-Breaking Constraint — R and WASM Bindings + 28 Examples

**What goes wrong:**
FOptDes is a new top-level module (`optimal_design.rs` or `optimal_design/`), but any change
to re-exports in `src/lib.rs` or `src/prelude.rs` can break R bindings, WASM/JS bindings,
or the 28 `[[example]]` entries if existing public names are shadowed, removed, or
their signatures changed.

Specific risks:
- Adding a `use pace_fpca::PaceFpcaResult` re-export that conflicts with an existing
  `use crate::pace_fpca::PaceFpcaResult` in calling code.
- Changing `PaceFpcaResult` to add fields (breaks `#[non_exhaustive]`-based struct-update
  syntax in callers — though `#[non_exhaustive]` already prevents callers from constructing
  it, so *reading* new fields is safe).
- Adding a function name that collides with an existing name in `prelude.rs`.

**Why it happens:**
Milestone-v0.35.0 adds real code and a crates.io publish (unlike audit milestones). Every
lib.rs edit risks a public-API change.

**How to avoid:**
- Add `pub mod optimal_design;` and `pub use optimal_design::{...};` in `lib.rs`; never
  modify or remove existing re-exports.
- Run `cargo test -p fdars-core --features linalg` — all 28 examples and existing tests must
  still compile and pass.
- Check `cargo doc --features linalg` builds without errors (doc tests cover re-export
  consistency).
- Confirm no name collision: `grep -r "optimal_design\|foptdes\|greedy_design" fdars-core/src/`
  before choosing function names.

**Warning signs:**
- Any existing doc-test fails after adding the new module.
- `cargo test --examples` fails on an example unrelated to FOptDes.

**Phase to address:** Every phase (Phase 64, 65 both). Gate: run the full test suite including
`--examples` before committing each phase.

---

### Pitfall 10: Column-Major FdMatrix Indexing — Row vs. Column Off-by-One

**What goes wrong:**
`FdMatrix` uses column-major layout: `element (row, col)` is at flat index `row + col * nrows`.
The eigenfunctions matrix is `m × K` (m rows = work-grid points, K columns = components).
Accessing eigenfunctions as if they were row-major produces transposed data.

In `pace_fpca.rs`, the eigenfunctions are accessed as `eigenfunctions[(j, k)]` where `j` is
the work-grid index and `k` is the component. The FOptDes code must use the same convention.

Specific mistake: building Φ_S as a local K×|S| matrix in *row-major* order (Rust's natural
`Vec<Vec<f64>>` layout) and then multiplying it as if it were column-major. The matrix-vector
product `Φ_S diag(λ) Φ_Sᵀ` gives the wrong result if Φ_S is transposed.

**Why it happens:**
Rust's `Vec<Vec<f64>>` is naturally row-major. When assembling a local matrix for the design
problem, developers use `phi[k][j]` (component k, observation j) while the math uses
`Φ_S[j,k]` (observation j, component k). The transposition silently computes
`Φ_Sᵀ diag(λ) Φ_S` (K×K) instead of `Φ_S diag(λ) Φ_Sᵀ` (|S|×|S|).

**How to avoid:**
- Use the same local matrix layout as `pace_fpca.rs`: `phi_i[j * actual_ncomp + k]` (row-major,
  observation index j is the outer loop, component k is the inner). When porting this layout
  to FOptDes, `phi_s[j * K + k]` means "at design point j, component k".
- For the Σ_yi assembly, mirror lines 461–474 of `pace_fpca.rs` verbatim and add a
  `// FOptDes port of pace_fpca.rs:461-474` comment.
- Add a shape-check assertion: for `|S|` design points and K components, Σ_yi must be |S|×|S|
  (not K×K). If the assembled matrix has the wrong size, the Cholesky call fails with a
  dimension error.

**Warning signs:**
- Σ_yi is K×K instead of |S|×|S| (wrong matrix assembled).
- Criterion value is independent of σ² (transposed Φ_S drops the σ²I diagonal).
- Criterion for |S|=1 is K² times the expected value (K×K instead of 1×1 sigma_yi).

**Phase to address:** Phase 64 (criterion evaluation). Gate: shape-assertion test confirming
Σ_yi is |S|×|S| for |S| in {1, 2, 5}.

---

### Pitfall 11: `!Send` Concerns for Parallelizing the Candidate Sweep

**What goes wrong:**
`iter_maybe_parallel!` from `parallel.rs` uses rayon when the `parallel` feature is enabled.
If the greedy candidate sweep is parallelized via this macro, any state shared across the
closure must be `Send + Sync`.

`PaceFpcaResult` derives `Clone` and contains only `Vec<f64>` and `FdMatrix` — both are
`Send + Sync`. However, if the criterion closure captures a mutable reference (e.g., to a
pre-computed matrix `M_S`) that is mutated per-candidate, the closure cannot be `Send`.

The `FftPlanner` from `rustfft` (used in kshape.rs) is `!Send`. FOptDes does not use FFT,
but if future refactoring moves any SBD/FFT helper into the candidate sweep, this breaks.

The existing codebase's RNG concern (`StdRng::seed_from_u64(seed + k)`) is not relevant to
FOptDes (no randomness needed in the greedy loop — the criterion is deterministic).

**Why it happens:**
Developers parallelize the candidate sweep with rayon for speed, then capture `&mut M_S` in
the closure without noticing that mutable closure captures are not `Send`.

**How to avoid:**
- In the candidate sweep: compute `M_S` once (immutable, `Send`), then map over candidates
  with closures that capture only immutable references to `M_S`. Each closure creates its
  own `M_cand = M_S + σ⁻² φφᵀ` (a fresh local K×K copy) without mutating `M_S`.
- Do not share mutable state across rayon worker closures — all per-candidate state must be
  locally allocated.
- Verify `Send` at compile time: the parallel build (`--features parallel`) must compile
  without error. The `iter_maybe_parallel!` macro already requires the iterator items to be
  `Send`.

**Warning signs:**
- `cargo build --features parallel` fails with "`M_S` cannot be shared between threads
  safely" or "closure may outlive the current function".
- Adding rayon causes test failures not present in the sequential build.

**Phase to address:** Phase 65 (greedy loop). Gate: `cargo build --features parallel`
passes; `cargo test --features parallel` passes.

---

### Pitfall 12: CI Gate — `--all-targets --features linalg,parallel` and fmt Drift

**What goes wrong:**
CI runs `cargo clippy --all-targets --features linalg,parallel -- -D warnings`. "All targets"
includes test code, bench code, and examples. A public function added to `optimal_design.rs`
that is only used in tests may generate an unused-function warning in the main target but not
in a plain `cargo clippy -p fdars-core`. The CI gate catches this; a local check without
`--all-targets` misses it.

The second CI hazard is fmt drift on `--no-verify` commits. `pace_fpca.rs` was committed with
`--no-verify` in prior milestones (executor timeout); if the same happens here, CI's
`cargo fmt --check` will fail even though clippy passes.

**Why it happens:**
Long fdars builds (executor subagent stall, documented in MEMORY.md) push developers toward
`--no-verify` commits. `cargo fmt` is then skipped, leaving unformatted code.

**How to avoid:**
- Run `cargo fmt -p fdars-core` before every commit, even `--no-verify` ones.
- Run the full CI gate locally before final commit:
  ```bash
  cargo fmt --check -p fdars-core
  cargo clippy --all-targets --features linalg,parallel -- -D warnings
  cargo test -p fdars-core --features linalg
  ```
- Do not add `#[allow(dead_code)]` on public items — make them actually public and used in
  doc examples, which serves as both documentation and dead-code prevention.

**Warning signs:**
- CI fails on `fmt --check` but `clippy` passes locally.
- Clippy warning about unused functions appears only in `--all-targets` mode.

**Phase to address:** Both phases (64, 65). Gate: local CI gate before phase completion.

---

### Pitfall 13: /tmp and target/ Disk Pressure on Full Builds

**What goes wrong:**
MEMORY.md records two disk-pressure failure modes specific to fdars:

1. **`/tmp` exhaustion blocks pre-commit hooks** — doc-test linkage uses a small `/tmp` tmpfs.
   When `/tmp` is full, all commits fail with "No space left on device" even for
   `--no-verify` bypasses of the slow hooks. The pre-commit hook itself fails before the
   bypass takes effect.

2. **`target/` fills `/home` partition** — `target/` grows to 100+GB. When the home partition
   fills, `cargo test` dies at the link step with "linking with cc failed" — not a code error.
   The symptom is indistinguishable from a linker misconfiguration.

For FOptDes specifically: adding a new module with examples and benchmarks increases
incremental build artifacts. If a benchmark (`[[bench]]`) is added in Phase 65, it adds
another rustc artifact under `target/debug/`.

**How to avoid:**
- Before any full `cargo test` or `cargo bench` run: check `df -h /tmp /home` and free space
  if < 2GB.
- To free target/: `rm -rf target/debug/{incremental,examples}` recovers ~108GB per MEMORY.md.
- Use `cargo test -p fdars-core` (not workspace-wide) to minimize artifact accumulation.
- If pre-commit hooks fail with "No space left": free /tmp first, then retry.

**Warning signs:**
- "No space left on device" during `cargo test` or commit hooks.
- "linking with cc failed" on a green clippy check (disk full, not a code error).
- Benchmark add causes a significant increase in `target/debug/` size.

**Phase to address:** Both phases (64, 65), especially if benchmarks are added.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Brute-force Σ_yi inversion (no rank-1 update) | Simpler, correct first pass | O(G·K³) per greedy step — bottleneck at K>20, G>200 | Acceptable in Phase 64; profile before optimizing in Phase 65 |
| Sequential candidate sweep (no rayon) | Deterministic, no Send constraints | O(G·n_select·K²) total — fine for G≤51, n_select≤20 | Acceptable; parallelize only if benchmark shows >1s wall time |
| Work grid = candidate set (no user-supplied candidates) | Avoids interpolation complexity | Limits design flexibility; callers cannot supply off-grid candidates | Acceptable for v0.35.0 MVP; generalizable in a later milestone |
| Single σ² for all design criteria | Simpler API | Cannot model heteroskedastic measurement noise | Acceptable; PACE already enforces single σ² |
| Skip criterion monotonicity assertion in release builds | Avoids O(n_select) extra cost | Hides criterion-direction bugs in production | Never — the assertion is O(1) overhead after the greedy loop and catches make-or-break correctness issues |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `PaceFpcaResult` eigenvalues | Using raw eigenvalues from `eigenfunctions` decomposition (pre-sign-fix) instead of `result.eigenvalues` | Always read `result.eigenvalues` and `result.eigenfunctions` from the returned `PaceFpcaResult`; never re-decompose |
| `helpers::simpsons_weights` | Computing weights once from `&[0.0..1.0]` instead of from `pace_result.argvals` | Pass `&pace_result.argvals` directly; the weight sum equals the argvals range, not 1.0 |
| `linalg::cholesky_solve` | Passing a column-major matrix as if it were row-major | `cholesky_solve` expects row-major (it uses `mat[i*d+j]`); local assembly of Σ_S must match |
| `lib.rs` re-exports | Importing `pace_fpca::PaceFpcaResult` in `optimal_design.rs` without adding the path alias | Use `crate::pace_fpca::PaceFpcaResult` everywhere in the new module; add `pub use optimal_design::{...}` to `lib.rs` additively |
| `FdMatrix.column(k)` | Using it to get a mutable reference for in-place update | `column()` returns `&[f64]`; for mutation, extract to a `Vec` and write back |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Re-assembling full m×m matrix in criterion loop | Criterion evaluation takes O(m²·K) per candidate instead of O(K²) | Assemble only the |S|×|S| Σ_S, not the full m×m covariance | Noticeable at m=100, K=10, G=100 |
| Inverting K×K posterior covariance K times per greedy step | O(n_select · G · K³) total | Pre-compute M_S once per step; apply rank-1 update per candidate | Breaks at n_select=50, K=20 (>10⁶ flops per step) |
| Collecting all criterion values in parallel then serial argmin | Allocates G-vector unnecessarily | Use `par_iter().enumerate().min_by(...)` with care for determinism | Negligible — only matters at G > 10,000 |
| Computing Simpson weights inside the criterion loop | Redundant computation, O(m) per criterion call | Call `simpsons_weights` once at design entry; pass weights as a slice | No correctness issue; O(G · n_select · m) wasted ops at G=100, n_select=20, m=51 |

---

## "Looks Done But Isn't" Checklist

- [ ] **Criterion monotonicity:** verify `criterion[k+1] <= criterion[k]` for all greedy steps — the loop may select a point but the objective check was skipped.
- [ ] **|S|=0 prior recovery:** `MSE(∅) ≈ Σ_k λ_k` and `A-opt(∅) = Σ_k λ_k` — often passes by accident if σ² is wrong but compensating.
- [ ] **Determinism with `--features parallel`:** must produce identical selected designs in both sequential and parallel builds — easy to skip this cross-build test.
- [ ] **Eigenfunction interpolation off-grid:** if candidates are a strict subset of the work grid (not indices 0..m-1), interpolation is needed — a test with off-index candidates catches the missing interpolation.
- [ ] **D-optimality sign:** `log det Cov(ξ|Y_S)` is negative (all posterior eigenvalues < prior); if the implementation returns a positive value, the sign convention is wrong.
- [ ] **No duplicate candidates in S:** the greedy loop must exclude already-selected indices; easy to forget when using a flat candidate Vec without an exclusion mask.
- [ ] **Full CI gate before merge:** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel`, `cargo test --features linalg` — the last two are often run but fmt is skipped after `--no-verify` commits.
- [ ] **R + WASM bindings still compile:** run `cargo build --target wasm32-unknown-unknown` and verify R binding example code still compiles (smoke test).

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong Σ_yi (missing σ²I) | LOW | Add `+= sigma2` to diagonal in assembly; re-run tests |
| Wrong score posterior (missing prior term) | LOW | Add `diag(1/λ)` to M before inversion; verify known-answer test |
| Missing quadrature weights | LOW | Multiply inner sum by `simpsons_weights`; check `MSE(∅) ≈ Σ_k λ_k` |
| D-optimality sign flip | LOW | Negate objective or change greedy from argmax to argmin; verify monotone-decrease test |
| Greedy selects duplicate points | LOW | Add exclusion mask `selected_indices.contains(&j)` in candidate loop |
| Cholesky fails (near-singular Σ_yi) | MEDIUM | Add ridge-retry (mirror pace_fpca.rs); if still fails, validate σ² floor |
| Non-determinism from rayon argmin | MEDIUM | Collect criterion Vec sequentially, then serial argmin — remove rayon from argmin step |
| Column-major vs. row-major transposition | MEDIUM | Mirror pace_fpca.rs `phi_i[j * ncomp + k]` layout; add shape-check assertion |
| fmt drift / CI failure | LOW | `cargo fmt -p fdars-core && git add -u && git commit --amend` (if pre-merge) |
| disk pressure blocking build | MEDIUM | `rm -rf target/debug/{incremental,examples}` then retry |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Wrong Σ_yi assembly (P1) | Phase 64 — criterion evaluation | Hand-computed Σ_yi for 1-point, 2-eigenfunction synthetic |
| Wrong score posterior covariance (P2) | Phase 64 — criterion evaluation | Known-answer: `Cov(ξ|∅) = diag(λ)`; monotone-decrease on A-opt trace |
| Missing quadrature weights (P3) | Phase 64 — criterion evaluation | `MSE(∅) ≈ Σ_k λ_k` identity; grid-size invariance check |
| Optimality-sign convention (P4) | Phase 64 (definition) + Phase 65 (greedy) | Monotone-decrease assertion on both AOpt and DOpt over greedy steps |
| Near-duplicate / σ²→0 singularity (P5) | Phase 64 (ridge-retry) + Phase 65 (dedup) | Greedy output has no duplicate indices; Cholesky succeeds for σ²=1e-4 |
| Eigenfunction interpolation (P6) | Phase 64 — criterion evaluation | Off-grid candidate test; out-of-range candidate returns InvalidParameter |
| Greedy argmin determinism (P7) | Phase 65 — greedy loop | Identical output for two identical calls; matches sequential vs. parallel build |
| O(G·p·cost) blowup (P8) | Phase 65 — greedy loop | Benchmark on G=51, K=5, n_select=10; correctness before optimization |
| Additive/non-breaking (P9) | Both phases | Existing test suite fully passes; `--examples` compiles |
| Column-major indexing (P10) | Phase 64 — criterion evaluation | Shape-assertion: Σ_S is `|S|×|S|` not `K×K` |
| Send constraints for rayon (P11) | Phase 65 — greedy loop | `cargo build --features parallel` passes; `cargo test --features parallel` |
| CI gate / fmt drift (P12) | Both phases | Local CI gate before each phase commit |
| Disk pressure (P13) | Both phases | `df -h` check before full builds; `rm -rf target/debug/{incremental,examples}` as needed |

---

## Sources

- `fdars-core/src/pace_fpca.rs` — BLUP formulas (Σ_yi assembly lines 461-474, Ω_i band variance lines 547-576, ridge-retry lines 480-490, σ² validation line 316-324)
- `fdars-core/src/helpers.rs` — `simpsons_weights` API and quadrature convention
- `fdars-core/src/linalg.rs` — `cholesky_solve` layout convention (row-major)
- `fdars-core/src/matrix.rs` — `FdMatrix` column-major indexing contract
- `fdars-core/src/parallel.rs` — `iter_maybe_parallel!` Send requirement
- Project MEMORY.md — `/tmp` exhaustion, `target/` disk pressure, executor stall, `--no-verify` fmt drift, `--all-targets` CI gate
- Yao, Müller & Wang (2005) JASA 100(470) — PACE BLUP formulas (§2.2, §3.2)
- PACE@2.17 MATLAB `FOptDes` — reference implementation for criterion monotonicity and greedy selection convention

---
*Pitfalls research for: Optimal Experimental Design on PACE/BLUP model (FOptDes), fdars-core v0.35.0*
*Researched: 2026-09-02*
