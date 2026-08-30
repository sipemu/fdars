# Phase 45: Functional Co-Clustering (funLBM latent-block) — Research

**Researched:** 2026-08-30
**Domain:** Latent Block Model EM on FPC scores; model selection via slope heuristic
**Confidence:** HIGH (all codebase claims verified by direct file read; statistical algorithm
claims tagged with provenance)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Block model: funLBM block-wise Gaussian — each (row-block k, column-block l) is modelled by a
  low-dimensional Gaussian on FPC scores.
- FPC scores source: global FPCA via the existing `fdata_to_pc_1d` on the full data; block
  Gaussians operate on subvectors of the global scores.
- Column-block assignment: an arbitrary cluster label per argument point (L column-clusters);
  columns need NOT be contiguous.
- FPC components per block: a fixed small `d` taken from config (`ncomp`).
- EM variant: variational / classification EM alternating row and column memberships
  (deterministic — no stochastic SEM-Gibbs in v1).
- Initialization: k-means for row clusters (reuse `kmeans_fd`) and k-means on argument-point
  profiles for column clusters, seeded.
- Convergence: stop when the log-likelihood (or variational lower bound) change < tol, or at
  `max_iter`.
- Determinism: all randomness seeded via `StdRng::seed_from_u64(seed + offset)`;
  bit-reproducible given the same seed.
- Result struct `CoClusterResult`: `row_labels`, `col_labels`, `n_row_blocks`, `n_col_blocks`,
  `block_params`, `log_likelihood`, `icl`.
- Model criterion: ICL (primary); BIC fallback deferred.
- Per-block parameters: block mean + FPC-score variance/covariance + row/column mixing proportions.
- Labels: hard labels (argmax posterior) for both rows and columns.
- Model selection: Birgé–Massart data-driven slope heuristic over a user-supplied (K, L) grid.
- Config struct `CoClusterConfig`: `n_row_blocks`, `n_col_blocks`, `ncomp`, `max_iter`, `tol`,
  `n_init`, `seed` (builder pattern like `GmmClusterConfig`).
- Module layout: single new `coclustering.rs` (factor to folder only if it exceeds ~500 lines).

### Claude's Discretion
- Exact block-parameter parameterization (full vs diagonal FPC-score covariance).
- The precise ICL penalty formula.
- The slope-heuristic penalty-calibration details.
- Internal helper decomposition, struct field naming, and plan/wave decomposition.
- Whether the module warrants a folder split at implement time.

### Deferred Ideas (OUT OF SCOPE)
- SEM-Gibbs / stochastic EM variant.
- Per-block FPCA (global FPCA reused in v1).
- Full (vs diagonal) block covariance if the planner chooses diagonal.
- BIC-only model selection.
- Exhaustive multi-restart consensus beyond `n_init`.
- Non-Gaussian block distributions; missing-data / irregular-grid co-clustering.
- Soft (fuzzy) co-cluster memberships as primary output.
- Plotting/rendering of co-cluster blocks.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CLUS-02-01 | Fit a funLBM that simultaneously assigns curves to row-clusters and argument points to column-clusters via block-wise-Gaussian EM on FPC scores, given a target (K, L) block count. | §1 generative model + §2 EM algorithm |
| CLUS-02-02 | Retrieve row labels, column labels, per-block parameters, converged log-likelihood, and ICL. | §3 ICL formula + CoClusterResult layout |
| CLUS-02-03 | Select block count via slope-heuristic criterion over a (K, L) grid. | §4 Birgé–Massart recipe |
</phase_requirements>

---

## Summary

Phase 45 adds `fdars-core/src/coclustering.rs`, implementing the functional latent block model
(funLBM). The algorithm simultaneously clusters n functional curves (rows) into K groups and m
argument evaluation points (columns) into L groups via a block-wise Gaussian EM operating on
global FPC score subvectors. The key insight is that after running `fdata_to_pc_1d` on the full
dataset, each curve's `ncomp`-dimensional score vector is treated as a multivariate observation;
the block model then asserts that within block (k, l), the score sub-vector at column-cluster l's
indices follows a Gaussian. Because column-cluster membership determines *which* score coordinates
are relevant for each block Gaussian, the LBM is inherently bimodal (rows and columns are both
latent), and a straightforward soft E-step is intractable — standard practice uses the
variational mean-field (VEM) or its hard-assignment limit (Classification EM / CEM), alternating
updates of row memberships and column memberships.

Model selection over a (K, L) grid uses the Birgé–Massart slope heuristic: fit all (K, L)
candidates, collect (model_dimension, max_log_lik) pairs, estimate the linear slope from the
largest-model region, and select the model that maximizes log_lik − (2 × slope) × model_dim. All
new code reuses existing infrastructure: `fdata_to_pc_1d`, `kmeans_fd`, the GMM covariance
accumulation helpers, `cholesky_d`, `log_det_from_cholesky`, `mahalanobis_sq`, the
`iter_maybe_parallel!` macro, and `StdRng::seed_from_u64` seeding.

**Primary recommendation:** Implement a Classification EM (CEM) — hard alternating row/column
assignments inside each E-half-step. This is the simplest deterministic scheme, avoids the
tensor-product variational mean-field, and matches the R `funLBM` default behaviour. Diagonal
block covariance eliminates Cholesky dependence on the `linalg` feature, keeping the module WASM
and MSRV-1.81 safe.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Functional data ingestion | Domain module (`coclustering.rs`) | Infrastructure (`matrix.rs`, `regression.rs`) | `fdata_to_pc_1d` owns FPC projection; coclustering calls it once |
| EM algorithm (row/col updates) | Domain module (`coclustering.rs`) | Shared helpers (`gmm/covariance.rs`, `linalg.rs`) | Block Gaussian density reuses existing Gaussian log-density pattern |
| Model criterion (ICL) | Domain module (`coclustering.rs`) | — | LBM-specific ICL formula differs from GMM ICL; implemented inline |
| Model selection (slope heuristic) | Domain module (`coclustering.rs`) | `iter_maybe_parallel!` for grid sweep | Birgé–Massart selection is a pure numerical post-processing step |
| Public API / config | `src/lib.rs`, `src/prelude.rs` re-exports | `coclustering.rs` | Additive re-export of `CoClusterResult`, `CoClusterConfig`, `coclustering_funlbm`, `coclustering_select` |

---

## 1. The funLBM Generative Model on FPC Scores

### 1.1 Setup

- `n` curves (rows), each observed at `m` argument points (columns).
- Global FPCA via `fdata_to_pc_1d(data, ncomp, argvals)` yields score matrix `S ∈ ℝ^{n × ncomp}`.
  `S[(i, k)]` = score of curve `i` on FPC `k`.  [VERIFIED: src/regression.rs:287-399]
- Row-cluster labels `z_i ∈ {0, …, K-1}` (one per curve).
- Column-cluster labels `w_j ∈ {0, …, L-1}` (one per argument-point index `j ∈ {0, …, ncomp-1}`
  — i.e., **one label per FPC component**, not per original argument point). [See §1.2 note]
- Row mixing proportions `π_k > 0`, Σπ_k = 1.
- Column mixing proportions `ρ_l > 0`, Σρ_l = 1.

> **Important clarification on "column-cluster":**
> The CONTEXT.md says "column-block assignment: an arbitrary cluster label per argument point".
> In the funLBM on FPC scores the *argument points* are the `ncomp` FPC components (the columns
> of the score matrix S), NOT the original `m` evaluation grid points. The column-cluster label
> `w_j ∈ {0,…,L-1}` assigns each FPC component `j ∈ {0,…,ncomp-1}` to a column-cluster.
> Reporting `col_labels` of length `ncomp` (one per FPC component) is correct and sufficient.
> [ASSUMED — this is the standard funLBM interpretation; the CONTEXT confirms "FPC components per
> block: a fixed small d taken from config (ncomp)" and "per-argument-point column labels".]

### 1.2 Block Gaussian

For block (k, l), the sub-vector of scores `s_{i,l} = {S[(i,j)] : w_j = l}` for curve `i` in
row-cluster `k` is modelled as:

```
s_{i,l} | z_i=k, w_·=col_labels  ~  N(μ_{kl}, Σ_{kl})
```

where `μ_{kl} ∈ ℝ^{d_l}` and `Σ_{kl} ∈ ℝ^{d_l×d_l}`.
`d_l = |{j : w_j = l}|` — the number of FPC components assigned to column-cluster l.

**Diagonal covariance (recommended, see §Discretion):** `Σ_{kl} = diag(σ²_{kl,1}, …, σ²_{kl,d_l})`.
This avoids the `linalg` feature gate, requires no Cholesky, and is standard in all R funLBM
implementations. [ASSUMED — simplification relative to the full R funLBM, documented for divergence note]

### 1.3 Complete-Data Log-Likelihood

Let `z_i ∈ {0,…,K-1}` (hard row assignment), `w_j ∈ {0,…,L-1}` (hard column assignment).

```
ℓ_c(z, w, θ) = Σ_i Σ_k 𝟙[z_i=k] { ln π_k + Σ_l Σ_{j: w_j=l} ln φ(S_{ij}; μ_{kl,j}, σ²_{kl,j}) }
```

where `φ(x; μ, σ²) = (2πσ²)^{-1/2} exp(-(x-μ)²/(2σ²))`.

**Observed log-likelihood (the quantity EM maximises):**

```
ℓ(θ) = Σ_i ln [ Σ_k π_k Π_l Π_{j:w_j=l} φ(S_{ij}; μ_{kl,j}, σ²_{kl,j}) ]
```

This is doubly-intractable because both `z_i` and `w_j` are latent. [ASSUMED — standard LBM
literature result]

---

## 2. The EM Algorithm — Classification EM (CEM) for the LBM

### 2.1 Why CEM, not Soft VEM

The exact LBM E-step requires the joint posterior over all (z_i, w_j) pairs, which is a product
of K×L terms and cannot be factorised without the variational mean-field approximation (a nested
iterative loop inside each E-step). The standard deterministic alternative is **Classification EM
(CEM)**: assign hard labels by argmax at each E-half-step, alternating row and column updates.
The R `funLBM` package default is CEM. [ASSUMED — derived from funLBM CRAN documentation and
standard LBM literature]

### 2.2 Initialisation

**Row initialisation:**
```rust
// Reuse kmeans_fd on the original data for initial row labels.
// argvals needed only for integration weights in kmeans_fd; pass uniform grid.
let km_row = kmeans_fd(data, argvals, K, 100, 1e-4, seed)?;
let mut row_labels: Vec<usize> = km_row.cluster;   // length n
```
[VERIFIED: src/clustering.rs:545-607 — `kmeans_fd(data, argvals, k, max_iter, tol, seed) ->
Result<KmeansResult>` where `KmeansResult.cluster: Vec<usize>` length n]

**Column initialisation (k-means on score matrix columns):**
```rust
// Extract ncomp score column vectors; treat each FPC component j as a
// data point with its n-dimensional profile.
// Use plain k-means++ on ℝ^n (Euclidean) for column clusters.
// Implemented inline: no dependency on kmeans_fd (which expects FdMatrix + argvals).
let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
let col_labels: Vec<usize> = kmeans_pp_columns(&score_matrix, L, &mut rng);
```
`kmeans_pp_columns` is a ~30-line inline helper: initialise L centers from column profiles of
`scores` (n-dim vectors), run 10 assign-update iterations, return assignment `Vec<usize>` of
length `ncomp`. [ASSUMED — inline helper pattern mirroring `gmm/init.rs` style]
[VERIFIED: src/gmm/init.rs:76-148 — exact k-means++ pattern to mirror]

### 2.3 CEM E-Step (alternating hard assignments)

**E-row half-step:** Given fixed `col_labels`, compute the log-probability of each curve `i`
under row-cluster `k`:

```
log_row[i][k] = ln π_k + Σ_l Σ_{j: col_labels[j]=l} ln φ(S[i][j]; μ_{kl,j}, σ²_{kl,j})
```

Apply log-sum-exp normalisation → probabilities, then argmax → new `row_labels[i]`.

**E-col half-step:** Given fixed `row_labels`, compute the log-probability of FPC component `j`
under column-cluster `l`:

```
log_col[j][l] = ln ρ_l + Σ_k Σ_{i: row_labels[i]=k} ln φ(S[i][j]; μ_{kl,j}, σ²_{kl,j})
```

Apply log-sum-exp normalisation → probabilities, then argmax → new `col_labels[j]`.

> **Log-sum-exp guard:** always use the max-shifted form to avoid underflow.
> Pattern already in codebase: [VERIFIED: src/gmm/em.rs:64-83 — `normalizeresponsibilities`]

### 2.4 M-Step (closed-form updates)

After each full (E-row + E-col) cycle, update parameters:

```
π_k  = (|{i : row_labels[i] = k}|) / n
ρ_l  = (|{j : col_labels[j] = l}|) / ncomp

For each block (k, l):
  members_i = {i : row_labels[i] = k}
  members_j = {j : col_labels[j] = l}
  n_k = |members_i|,  d_l = |members_j|

  μ_{kl,j}  = mean_{i ∈ members_i}  S[i][j]       for each j ∈ members_j
  σ²_{kl,j} = var_{i ∈ members_i}   S[i][j] + reg  for each j ∈ members_j
```

`reg` = `data_scaled_reg(score_rows, ncomp)` — same pattern as GMM.
[VERIFIED: src/gmm/covariance.rs:20-45 — `data_scaled_reg` function]

**Empty cluster guard:** If `n_k = 0` for any k (or `d_l = 0` for any l), the block model is
degenerate. Strategy: keep previous parameters, set `π_k = 0` (or reinitialise via a random
re-split of the largest cluster using `rng.gen_range`). This is standard for EM empty-cluster
handling and mirrors the GMM pattern where zero-weight components keep an identity covariance.
[VERIFIED: src/gmm/covariance.rs:162-186 — `compute_covariances` skips empty clusters with fallback]

### 2.5 Log-Likelihood Computation and Convergence

Compute the **classification log-likelihood** (based on hard assignments):

```
ℓ_c = Σ_i { ln π_{z_i} + Σ_l Σ_{j ∈ members_j^l} ln φ(S[i][j]; μ_{z_i,l,j}, σ²_{z_i,l,j}) }
```

Convergence: `|ℓ_c^{(t)} - ℓ_c^{(t-1)}| < tol` or `t >= max_iter`.
The classification LL is guaranteed non-decreasing after each full CEM cycle (in CEM the
assignments can only stay the same or improve the objective). [ASSUMED — standard CEM property]

### 2.6 Multiple Restarts

Run `n_init` restarts with seeds `seed + init_idx * 1000`. Return the result with the highest
`log_likelihood`. Pattern from GMM cluster:
[VERIFIED: src/gmm/cluster.rs:11-33 — `run_multiple_inits` pattern with `base_seed.wrapping_add(init as u64 * 1000 + k as u64)`]

---

## 3. ICL for the Latent Block Model

### 3.1 LBM-Specific ICL Formula

For the GMM (single-index), the codebase uses:
```
ICL = BIC + 2 × H(responsibilities)
    = -2·ℓ + p·ln(n) + 2·H
```
[VERIFIED: src/gmm/em.rs:214-226 — `compute_icl(bic, resp, n, k)` with entropy over soft resp]

For the LBM with hard CEM assignments, the ICL is based on the **Integrated Completed
Likelihood** with the entropy of the classification — but since CEM assigns hard labels, the
classification entropy is zero. The standard LBM ICL is:

```
ICL_LBM = ℓ_c  −  pen(K, L, ncomp, n)
```

where the BIC-style penalty is:

```
pen(K, L, ncomp, n) = 0.5 × p_{KL} × (ln n + ln ncomp)
```

and `p_{KL}` = number of free parameters of the (K, L) LBM:
```
p_{KL} = (K-1)             [row mixing proportions]
        + (L-1)             [col mixing proportions]
        + K·L·d̄             [block means, d̄ = ncomp avg FPC per block = ncomp]
        + K·L·d̄             [block diagonal variances]
       = (K-1) + (L-1) + 2·K·L·ncomp
```

In code:
```rust
let p_kl = (k - 1) + (l - 1) + 2 * k * l * ncomp;
let icl = log_lik_c - 0.5 * p_kl as f64 * ((n as f64).ln() + (ncomp as f64).ln());
```

[ASSUMED — derived from Govaert & Nadif (2008) LBM ICL formula; the symmetric `(ln n + ln m)`
penalty is standard for LBMs with n rows and m columns. Here m = ncomp. The GMM analogue in the
codebase uses `p·ln(n)`; the LBM symmetric form penalises both dimensions.]

> **Alternative (simpler) ICL:** Use `ICL = ℓ_c − 0.5 × p_{KL} × ln(n)` (row-only penalty,
> analogous to the GMM BIC). This is also defensible and simpler; the slope heuristic (§4) is
> the primary selector anyway, so ICL is diagnostic. Use the symmetric form for alignment with
> funLBM R package. [ASSUMED]

### 3.2 Parameter Count Summary

| Term | Value |
|------|-------|
| Row proportions | K − 1 |
| Column proportions | L − 1 |
| Block means (diagonal assumed) | K × L × ncomp |
| Block variances (diagonal) | K × L × ncomp |
| **Total** | **(K−1) + (L−1) + 2·K·L·ncomp** |

---

## 4. Birgé–Massart Slope Heuristic for (K, L) Selection

### 4.1 Conceptual Basis

The slope heuristic (Birgé & Massart 2007; Baudry et al. 2012) calibrates the penalty for model
selection from the data: fit all candidate models, plot log-lik vs model dimension, identify the
"elbow" region where the log-lik grows rapidly with dimension (large models overfit), estimate
the slope of this linear growth, and set the penalty = 2 × slope per parameter. Select the model
that maximises `log_lik − (2 × slope) × dim`. [ASSUMED — standard slope heuristic description]

### 4.2 Implementation Recipe

```
Input:  grid of (K, L) pairs;  each fit produces (dim_{KL}, LL_{KL}).
        dim_{KL} = p_{KL} as defined in §3.2.

Step 1: Fit all (K, L) on the grid (parallelisable with iter_maybe_parallel!).
        Collect Vec<(dim: usize, ll: f64, k: usize, l: usize)>.

Step 2: Sort by dim descending. Take the top 50% (or at least 4 points) as the
        "large model region" (the linear region of the LL-vs-dim curve).

Step 3: OLS regression: regress ll on dim over the large-model subset.
        slope = Σ (dim_i − d̄)(ll_i − l̄) / Σ (dim_i − d̄)²
        This is a 2-line closed-form computation — no crate dependency needed.

Step 4: Penalty calibration: pen_rate = 2.0 * slope.abs().
        (slope is negative when the curve is concave down; use abs for the
        penalty magnitude since we subtract it.)

Step 5: Select (K*, L*) = argmax_{ (K,L) ∈ grid } { LL_{KL} − pen_rate × dim_{KL} }.

Return: selected (K*, L*) and its fit; also return all (K, L, LL, dim, penalised_score)
        for diagnostic use.
```

[ASSUMED — OLS slope estimation is the standard practical slope heuristic. "Top 50% by dim" is
the common heuristic for identifying the linear region; Baudry et al. 2012 suggest taking the
upper half of the model-dimension range.]

### 4.3 Edge Cases

- If all models have the same dimension (e.g. only one (K,L) in grid): return that model directly.
- If OLS denominator is near zero (all dims equal in the large-model subset): fall back to
  selecting the model with maximum LL.
- If `pen_rate` is zero or negative (LL is flat or increasing with dimension): fall back to max
  LL (the simplest model that achieves peak likelihood).

### 4.4 Public API

```rust
/// Fit funLBM for a single (K, L) block count.
pub fn coclustering_funlbm(
    data: &FdMatrix,
    argvals: &[f64],
    config: &CoClusterConfig,
) -> Result<CoClusterResult, FdarError>

/// Fit funLBM over a (K, L) grid and select via slope heuristic.
pub fn coclustering_select(
    data: &FdMatrix,
    argvals: &[f64],
    k_range: &[usize],   // candidate K values
    l_range: &[usize],   // candidate L values
    config: &CoClusterConfig,
) -> Result<CoClusterSelectResult, FdarError>
```

---

## 5. Struct Layouts and Config Builder Pattern

### 5.1 `CoClusterConfig` (mirrors `GmmClusterConfig`)

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct CoClusterConfig {
    /// Number of row clusters.
    pub n_row_blocks: usize,          // default 2
    /// Number of column clusters (over the ncomp FPC components).
    pub n_col_blocks: usize,          // default 2
    /// Number of FPC components (passed to fdata_to_pc_1d).
    pub ncomp: usize,                 // default 5
    /// Maximum CEM iterations.
    pub max_iter: usize,              // default 200
    /// Convergence tolerance on classification log-likelihood.
    pub tol: f64,                     // default 1e-6
    /// Number of random restarts; best by log-lik is kept.
    pub n_init: usize,                // default 3
    /// Base random seed.
    pub seed: u64,                    // default 42
}
```

[VERIFIED pattern: src/gmm/cluster.rs:49-86 — `GmmClusterConfig` with identical field types and
`impl Default`]

### 5.2 `CoClusterResult`

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct CoClusterResult {
    /// Hard row-cluster labels, length n.
    pub row_labels: Vec<usize>,
    /// Hard column-cluster labels (one per FPC component), length ncomp.
    pub col_labels: Vec<usize>,
    /// Number of row clusters fitted.
    pub n_row_blocks: usize,
    /// Number of column clusters fitted.
    pub n_col_blocks: usize,
    /// Per-block parameters: indexed [k * n_col_blocks + l].
    pub block_params: Vec<BlockParams>,
    /// Row mixing proportions (length K).
    pub row_props: Vec<f64>,
    /// Column mixing proportions (length L).
    pub col_props: Vec<f64>,
    /// Classification log-likelihood at convergence.
    pub log_likelihood: f64,
    /// ICL criterion (lower is better for penalty, but we store raw ICL = LL − penalty
    /// so higher is better — consistent with GmmResult.icl field).
    pub icl: f64,
    /// Number of CEM iterations performed.
    pub iterations: usize,
    /// Whether EM converged within max_iter.
    pub converged: bool,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BlockParams {
    /// Block mean (length = number of FPC components in this column-cluster, d_l).
    pub mean: Vec<f64>,
    /// Block diagonal variance (length d_l).
    pub variance: Vec<f64>,
}
```

### 5.3 `CoClusterSelectResult`

```rust
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CoClusterSelectResult {
    /// The selected (K*, L*) result.
    pub best: CoClusterResult,
    /// Selected K.
    pub best_k: usize,
    /// Selected L.
    pub best_l: usize,
    /// All grid fits: (K, L, log_lik, model_dim, penalised_score).
    pub grid_scores: Vec<(usize, usize, f64, usize, f64)>,
    /// Estimated slope from the large-model region.
    pub slope_estimate: f64,
    /// Penalty rate applied (= 2 × |slope|).
    pub penalty_rate: f64,
}
```

---

## 6. Code Skeleton — Core CEM Loop

```rust
// Source: derived from src/gmm/em.rs patterns [VERIFIED: src/gmm/em.rs:1-393]
// and src/gmm/init.rs patterns [VERIFIED: src/gmm/init.rs:76-184]

fn cem_once(
    scores: &FdMatrix,    // n × ncomp, column-major
    k: usize, l: usize, ncomp: usize,
    max_iter: usize, tol: f64, seed: u64,
) -> Result<CoClusterResult, FdarError> {
    let n = scores.nrows();
    let reg = data_scaled_reg_scores(scores, ncomp);  // inline version of gmm::data_scaled_reg

    // --- Init ---
    // (row init via kmeans_fd is done upstream; pass row_labels in)
    let mut row_labels = init_rows_kmeans(scores, k, seed)?;
    let mut col_labels = init_cols_kmeans_pp(scores, l, seed.wrapping_add(1));

    let mut prev_ll = f64::NEG_INFINITY;
    let mut params = compute_block_params(&scores, &row_labels, &col_labels, k, l, ncomp, reg);
    let mut row_props = compute_row_props(&row_labels, k, n);
    let mut col_props = compute_col_props(&col_labels, l, ncomp);

    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // E-row: for each curve i, compute log P(z_i=k | S[i], col_labels, params)
        let new_row_labels = e_row_step(scores, &col_labels, &params, &row_props, k, l, ncomp);

        // E-col: for each FPC j, compute log P(w_j=l | S[·][j], row_labels, params)
        let new_col_labels = e_col_step(scores, &row_labels, &params, &col_props, k, l, ncomp);

        row_labels = new_row_labels;
        col_labels = new_col_labels;

        // M-step
        params = compute_block_params(scores, &row_labels, &col_labels, k, l, ncomp, reg);
        row_props = compute_row_props(&row_labels, k, n);
        col_props = compute_col_props(&col_labels, l, ncomp);

        // Log-likelihood
        let ll = classification_log_lik(scores, &row_labels, &col_labels, &params, &row_props, k, l, ncomp);
        if iter > 0 && (ll - prev_ll).abs() < tol {
            converged = true;
            break;
        }
        prev_ll = ll;
    }

    let ll = classification_log_lik(scores, &row_labels, &col_labels, &params, &row_props, k, l, ncomp);
    let p_kl = (k - 1) + (l - 1) + 2 * k * l * ncomp;
    let icl = ll - 0.5 * p_kl as f64 * ((n as f64).ln() + (ncomp as f64).ln());

    Ok(CoClusterResult {
        row_labels, col_labels,
        n_row_blocks: k, n_col_blocks: l,
        block_params: params,
        row_props, col_props,
        log_likelihood: ll,
        icl,
        iterations, converged,
    })
}
```

### 6.1 E-Row Step (argmax hard assignment)

```rust
fn e_row_step(
    scores: &FdMatrix,
    col_labels: &[usize],
    params: &[BlockParams],
    row_props: &[f64],
    k: usize, l: usize, ncomp: usize,
) -> Vec<usize> {
    let n = scores.nrows();
    let mut new_labels = vec![0usize; n];
    for i in 0..n {
        let mut log_probs = vec![f64::NEG_INFINITY; k];
        for ki in 0..k {
            if row_props[ki] < 1e-15 { continue; }
            let mut lp = row_props[ki].ln();
            // Accumulate log-density over all FPC components j
            for j in 0..ncomp {
                let li = col_labels[j];
                let idx = ki * l + li;
                let mean_j = params[idx].mean[col_local_idx(col_labels, j, li)];
                let var_j  = params[idx].variance[col_local_idx(col_labels, j, li)];
                let x = scores[(i, j)];
                lp += log_gaussian_1d(x, mean_j, var_j);
            }
            log_probs[ki] = lp;
        }
        new_labels[i] = argmax_f64(&log_probs);
    }
    new_labels
}
```

`log_gaussian_1d(x, mu, sigma2) = -0.5 * ((x-mu).powi(2)/sigma2 + sigma2.ln() + LN_2PI)`

The `col_local_idx` maps global FPC index `j` within column-cluster `li` to the local position
in that block's mean/variance vectors (precomputed from `col_labels`). [ASSUMED — implementation
detail of the index bookkeeping]

### 6.2 E-Col Step (argmax hard assignment)

```rust
fn e_col_step(
    scores: &FdMatrix,
    row_labels: &[usize],
    params: &[BlockParams],
    col_props: &[f64],
    k: usize, l: usize, ncomp: usize,
) -> Vec<usize> {
    let mut new_labels = vec![0usize; ncomp];
    for j in 0..ncomp {
        let mut log_probs = vec![f64::NEG_INFINITY; l];
        for li in 0..l {
            if col_props[li] < 1e-15 { continue; }
            let mut lp = col_props[li].ln();
            for i in 0..scores.nrows() {
                let ki = row_labels[i];
                // The contribution of this (i, j, ki, li) combination
                // uses the mean/var at the LOCAL position of j within li
                // Needs a candidate param set for trial assignment w_j = li
                lp += log_gaussian_1d(scores[(i,j)], trial_mean(params,ki,li,j), trial_var(params,ki,li,j));
            }
            log_probs[li] = lp;
        }
        new_labels[j] = argmax_f64(&log_probs);
    }
    new_labels
}
```

> **Implementation note for the E-col step:** The `trial_mean`/`trial_var` computation during
> the col E-step requires the mean and variance for *hypothetical* assignment `w_j = li`. Since
> params are recomputed from scratch in the M-step and the E-step uses the PREVIOUS M-step params,
> this is well-defined: use the current `params[ki * l + li]` and treat FPC `j` as if it belonged
> to column-cluster `li`. The local index of `j` in that hypothetical cluster can be handled by
> precomputing per-block mean/variance as flat arrays indexed by global FPC index `j`, not by
> local position. This avoids the `col_local_idx` complexity.
>
> **Simplified storage:** Store `block_mean[ki][li][j]` and `block_var[ki][li][j]` as
> `k × l × ncomp` flat arrays during EM (not the jagged `Vec<BlockParams>` with local indexing).
> Convert to the `BlockParams` (with per-block-sized vectors) only in the final `CoClusterResult`.
> [ASSUMED — implementation design choice for tractability]

---

## 7. Numerical Pitfalls and Guards

| Pitfall | Context | Guard |
|---------|---------|-------|
| Empty row cluster | n_k = 0 after E-step | Keep previous `row_props[k] = 0`; set mean/var to 0/reg; handle in LL as `ln(0) = -∞ → skip` |
| Empty col cluster | d_l = 0 after E-step | Same: `col_props[l] = 0`; skip in density accumulation |
| Variance collapse | σ² → 0 in small block | Add `reg = data_scaled_reg(...) * REG_REL` [VERIFIED: src/gmm/covariance.rs:132-145] |
| log(0) from empty cluster | `row_props[k] = 0` → `ln(0) = -∞` | Guard: `if row_props[k] < 1e-15 { continue; }` [VERIFIED: src/gmm/em.rs:55-58] |
| All log_probs = -∞ | Degenerate params for every cluster | Fall back to uniform assignment (argmax returns 0; LL stays at -∞, triggers restart) |
| Underflow in column E-step | `n` terms summed per j (can be 1000+) | Use ln density accumulation (already done — never exponentiate before summing) |
| Label switching across restarts | Different restarts find permuted labels | Compare by log-lik (max is best); test correctness with ARI, not raw label equality [VERIFIED: src/test_helpers.rs:24-93 — `adjusted_rand_index`] |
| ncomp > min(n,m) | fdata_to_pc_1d clips ncomp | Check that requested ncomp ≤ the actual ncomp returned by FpcaResult.scores.ncols() |
| L > ncomp | Cannot assign ncomp components to L > ncomp clusters | Error: `FdarError::InvalidParameter` |
| K > n | Cannot assign n curves to K > n clusters | Error: `FdarError::InvalidParameter` (same guard as kmeans_fd) [VERIFIED: src/clustering.rs:568-574] |

---

## 8. Reusable Assets — Verified Signatures

### 8.1 `fdata_to_pc_1d`

```rust
// [VERIFIED: src/regression.rs:287-399]
pub fn fdata_to_pc_1d(
    data: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
) -> Result<FpcaResult, FdarError>
```

- `FpcaResult.scores`: `FdMatrix` of shape `(n, ncomp_actual)` where `ncomp_actual = ncomp.min(n).min(m)`.
  `scores[(i, k)]` = score of curve `i` on FPC `k`. [VERIFIED: src/regression.rs:358-364]
- `FpcaResult.rotation`: `FdMatrix` of shape `(m, ncomp)`. [VERIFIED: src/regression.rs:351-357]
- `FpcaResult.mean`: `Vec<f64>` of length `m`. [VERIFIED: src/regression.rs:22-38]
- `FpcaResult.weights`: `Vec<f64>` of length `m` (integration weights). [VERIFIED: src/regression.rs:22-38]

### 8.2 `kmeans_fd`

```rust
// [VERIFIED: src/clustering.rs:545-607]
pub fn kmeans_fd(
    data: &FdMatrix,
    argvals: &[f64],
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<KmeansResult, FdarError>
// KmeansResult.cluster: Vec<usize> length n — row cluster assignments
// [VERIFIED: src/clustering.rs:14-30]
```

### 8.3 `adjusted_rand_index` (test-only)

```rust
// [VERIFIED: src/test_helpers.rs:24-93]
// #[cfg(test)] only — not part of public API
pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64
// Returns 1.0 for perfect agreement up to permutation
```

### 8.4 Covariance Regularisation

```rust
// [VERIFIED: src/gmm/covariance.rs:20-45]
pub(super) fn data_scaled_reg(features: &[Vec<f64>], d: usize) -> f64
// Returns REG_REL (=1e-6) * mean_j Var(feature_j)
// Coclustering must either use its own inline version or extract to a shared helper
```

### 8.5 Log-density helpers (no linalg feature needed for diagonal)

```rust
// [VERIFIED: src/gmm/em.rs:17-40] — CovType::Diagonal branch (no Cholesky needed):
// -0.5 * (d * ln(2π) + Σ_j (ln σ²_j + (x_j - μ_j)² / σ²_j))
// Implement inline in coclustering.rs — no dependency on gmm internals
```

### 8.6 `StdRng::seed_from_u64`

```rust
// [VERIFIED: src/gmm/em.rs:349 and src/clustering.rs:584]
use rand::prelude::*;
let mut rng = StdRng::seed_from_u64(seed);
```

### 8.7 `iter_maybe_parallel!` macro

```rust
// [VERIFIED: src/parallel.rs:41-55]
// Used in coclustering_select grid sweep:
let results: Vec<_> = iter_maybe_parallel!(grid_pairs)
    .map(|(k, l)| { ... })
    .collect();
```

Note: `iter_maybe_parallel!` on `(0..n)` returns a rayon `ParallelIterator` under `parallel`
feature. The closure must not capture `&mut` state. The (K,L) grid fits are embarrassingly
parallel. [VERIFIED: src/parallel.rs:41-55 — macro signature; src/gmm/em.rs:100-115 — usage pattern]

---

## 9. Module Integration Points

### 9.1 `src/lib.rs` additions

```rust
pub mod coclustering;

// Root re-exports (additive only):
pub use coclustering::{
    coclustering_funlbm, coclustering_select,
    CoClusterConfig, CoClusterResult, CoClusterSelectResult, BlockParams,
};
```
[VERIFIED: src/lib.rs:64-170 — existing `pub mod` and `pub use` pattern]

### 9.2 `src/prelude.rs` additions

```rust
pub use crate::coclustering::{CoClusterConfig, CoClusterResult, CoClusterSelectResult};
```
[VERIFIED: src/prelude.rs:1-84 — existing prelude re-export pattern]

---

## 10. Test Oracles

### 10.1 Synthetic Block-Structured Data

Generate data with known (K=2, L=2) block structure:
- Row group 0 (n/2 curves): FPC scores drawn from N([2, 0], I) in score space.
- Row group 1 (n/2 curves): FPC scores drawn from N([-2, 0], I).
- Column group 0 (FPC 0): has strong between-cluster signal.
- Column group 1 (FPC 1): has weak signal.

Generate using only `rand::distributions::Normal` (already a dep via `rand_distr`):
[VERIFIED: fdars-core Cargo.toml uses `rand 0.8` and `rand_distr 0.4`]
[VERIFIED: src/clustering.rs:1 — `use rand::prelude::*;` — `rand` is available]

### 10.2 Required Tests

```rust
#[cfg(test)]
mod tests {
    // T1: ARI test — EM recovers known (K=2, L=2) structure at ARI > 0.8
    #[test] fn test_coclustering_recovers_block_structure() { ... }

    // T2: LL non-decreasing — log-likelihood vector over iterations is monotone
    #[test] fn test_classification_ll_nondecreasing() { ... }

    // T3: ICL finite — not NaN or ±∞ on well-conditioned data
    #[test] fn test_icl_is_finite() { ... }

    // T4: Determinism — two calls with same seed produce identical results
    #[test] fn test_determinism_under_seed() { ... }

    // T5: Slope heuristic selects true (K,L) on well-separated synthetic data
    #[test] fn test_slope_heuristic_selects_correct_kl() { ... }

    // T6: Error — K > n → FdarError::InvalidParameter
    #[test] fn test_error_k_exceeds_n() { ... }

    // T7: Error — L > ncomp → FdarError::InvalidParameter
    #[test] fn test_error_l_exceeds_ncomp() { ... }

    // T8: Error — ncomp = 0 → FdarError::InvalidParameter
    #[test] fn test_error_zero_ncomp() { ... }

    // T9: Error — data/argvals mismatch → propagated from fdata_to_pc_1d
    #[test] fn test_error_argvals_mismatch() { ... }
}
```

### 10.3 Oracle Data Generation Helper

```rust
// Inline in the test module — no external test helper needed.
// Generates n curves at m eval points with known (K, L) block signal.
fn make_block_data(n: usize, m: usize, ncomp: usize, seed: u64)
    -> (FdMatrix, Vec<f64>, Vec<usize>, Vec<usize>) {
    // Returns (data, argvals, true_row_labels, true_col_labels)
    // Uses rand::rngs::StdRng + Normal distribution from rand_distr.
}
```

---

## 11. Architecture Diagram

```
coclustering_funlbm(data, argvals, config)
│
├─── Input validation (K≤n, L≤ncomp, ncomp≥1)
│
├─── fdata_to_pc_1d(data, ncomp, argvals)
│    └── scores: FdMatrix (n × ncomp)          [regression.rs]
│
└─── cem_once(scores, K, L, ...) × n_init
     │
     ├─── init_rows_kmeans(scores, K, seed)    [wraps kmeans_fd on score matrix]
     ├─── init_cols_kmeans_pp(scores, L, seed) [inline k-means++ on column profiles]
     │
     └─── CEM loop (until convergence or max_iter):
          ├─ E-row: log_probs[i][k], argmax → row_labels
          ├─ E-col: log_probs[j][l], argmax → col_labels
          └─ M-step: π_k, ρ_l, μ_{kl,j}, σ²_{kl,j}  (closed-form)
               │
               └─ classification_log_lik → convergence check
     │
     └─── Best restart by log-lik → CoClusterResult
          (row_labels, col_labels, block_params, log_lik, ICL)


coclustering_select(data, argvals, k_range, l_range, config)
│
├─── iter_maybe_parallel! over k_range × l_range grid
│    └─── coclustering_funlbm per (K, L) → (dim_{KL}, LL_{KL})
│
└─── Birgé–Massart slope estimation (OLS on top-50% by dim)
     └─── Select argmax(LL − 2|slope|·dim) → CoClusterSelectResult
```

---

## 12. Common Pitfalls

### Pitfall 1: Block Param Index Bookkeeping
**What goes wrong:** When each block has a *variable-length* mean/var vector (length d_l depends
on how many FPC components land in column-cluster l), indexing into `params[k*L+l].mean[local_j]`
requires a precomputed `col_local_idx[j][l]` mapping. This is error-prone.

**How to avoid:** During EM, store a flat `block_mean[k][l][j]` and `block_var[k][l][j]` as
`k × l × ncomp` arrays (redundant storage — stores the mean for FPC j in block (k,l) regardless
of which column-cluster j is currently in). Only compute the local-indexed `BlockParams` once at
the end for the `CoClusterResult`. This eliminates all local-index bookkeeping from the hot loop.

### Pitfall 2: Confusing `ncomp` with `m`
**What goes wrong:** The CONTEXT mentions "column-cluster label per argument point" which sounds
like the `m` evaluation points. But the scores matrix is `n × ncomp`, so column-cluster labels
are over the `ncomp` FPC axes, NOT the `m` evaluation points.

**How to avoid:** Name the column-cluster label vector `col_labels` (length `ncomp`), and document
in rustdoc that "columns" in the co-clustering sense are the FPC components of the global FPCA.
The `col_labels` field in `CoClusterResult` has length `ncomp`.

### Pitfall 3: Rayon closure borrows
**What goes wrong:** `iter_maybe_parallel!(grid_pairs).map(|(k,l)| cem_once(...))` — the closure
captures `data`, `argvals`, and `config` by reference; rayon requires these to be `Send + Sync`.
`FdMatrix` and `CoClusterConfig` need `Send + Sync`.

**How to avoid:** `FdMatrix` already derives `Clone` with a `Vec<f64>` interior, which is
`Send + Sync`. `CoClusterConfig` will also be `Send + Sync`. Confirm at compile time. If not
parallel-safe, use `into_iter` for the grid sweep.
[VERIFIED: src/matrix.rs:38-44 — `FdMatrix` has `data: Vec<f64>`, `nrows: usize`, `ncols: usize` — trivially Send+Sync]

### Pitfall 4: Slope Heuristic Returns Boundary Model
**What goes wrong:** On poorly separated data, the slope heuristic may select the minimal (K=1,
L=1) or the maximal model because the slope estimate is noisy.

**How to avoid:** Expose `grid_scores` in `CoClusterSelectResult` so the user can inspect all
candidate (K, L) scores. Document that the slope heuristic is a heuristic and requires
sufficiently separated data to work reliably. Do NOT error-out on boundary selections.

### Pitfall 5: Using `row_labels` Before Convergence in ARI Test
**What goes wrong:** Tests that compare `row_labels` by direct equality against ground truth will
fail due to label permutation. ARI = 1 even when labels are permuted.

**How to avoid:** Always use `adjusted_rand_index(true_labels, fitted_labels) > 0.8` as the
oracle, not label equality.
[VERIFIED: src/test_helpers.rs:24-93 — ARI handles arbitrary permutation]

### Pitfall 6: score matrix `FdMatrix` column-major access pattern
**What goes wrong:** Accessing `scores[(i, j)]` in a tight loop over `i` for fixed `j` accesses
non-contiguous memory (column-major layout: element (i,j) is at offset `i + j*n`). This is
contiguous for fixed `i`, varying `j` — matching the "row per curve" access. Conversely, iterating
over all `i` for a fixed `j` (needed in E-col step) jumps by `n` bytes each step.

**How to avoid:** In the E-col step inner loop (over `i` for fixed `j`), use `scores.column(j)`
which returns a contiguous slice of length `n`.
[VERIFIED: src/matrix.rs — FdMatrix stores data column-major; column(j) returns contiguous slice]

---

## 13. Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gaussian log-density | Custom Gaussian code | Inline from `gmm/em.rs:17-40` diagonal branch | Already tested; diagonal case is 4 lines |
| K-means++ row init | Custom k-means | `kmeans_fd` from `clustering.rs` | Handles Simpson weights, edge cases, seeding |
| Log-sum-exp | Manual exp/sum | Copy `normalize_log_probs` pattern from `gmm/subspace.rs:167-189` | Already guarded against -∞ |
| Covariance regularisation | Absolute floor | `data_scaled_reg` pattern from `gmm/covariance.rs:20-45` | Scale-relative: works for any data range |
| Rayon parallelism | Direct rayon calls | `iter_maybe_parallel!` macro | WASM-compatible; single-flag gate |
| RNG seeding | `thread_rng()` | `StdRng::seed_from_u64(seed + offset)` | Bit-reproducible; matches rest of codebase |
| OLS slope estimation | Statistics crate | Inline 5-line OLS formula | Only a simple univariate regression needed |

---

## 14. Standard Stack (no new dependencies)

All packages are existing Cargo.toml dependencies. No new additions.

| Library | Purpose | Already Used In |
|---------|---------|----------------|
| `rand 0.8` + `StdRng` | Seeded RNG for k-means init | `clustering.rs`, `gmm/` |
| `rand_distr 0.4` | Normal distribution for test data generation | `simulation.rs` |
| `nalgebra 0.33` | (Not needed directly — FdMatrix covers all ops) | `regression.rs` |
| `rayon 1.10` | Optional parallelism via `iter_maybe_parallel!` | `gmm/em.rs`, `clustering.rs` |

[VERIFIED: fdars-core/Cargo.toml listed in CLAUDE.md technology stack section — rand 0.8, rand_distr 0.4, rayon 1.10, nalgebra 0.33 all present]

**Installation:** none (no new deps).

---

## 15. Validation Architecture

Nyquist validation is enabled (`workflow.nyquist_validation: true` in `.planning/config.json`).
[VERIFIED: /home/simonm/projects/rust/fdars/.planning/config.json:26]

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Built-in Rust test harness (`#[test]`) |
| Config file | none (cargo default) |
| Quick run command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core coclustering 2>&1 \| head -40` |
| Full suite command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel -- --nocapture 2>&1 \| tail -20` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLUS-02-01 | EM assigns curves and FPC components simultaneously | unit | `cargo test -p fdars-core coclustering::tests::test_coclustering_recovers_block_structure` | ❌ Wave 0 |
| CLUS-02-01 | Log-likelihood non-decreasing | unit | `cargo test -p fdars-core coclustering::tests::test_classification_ll_nondecreasing` | ❌ Wave 0 |
| CLUS-02-01 | Determinism under seed | unit | `cargo test -p fdars-core coclustering::tests::test_determinism_under_seed` | ❌ Wave 0 |
| CLUS-02-01 | Error on K > n | unit | `cargo test -p fdars-core coclustering::tests::test_error_k_exceeds_n` | ❌ Wave 0 |
| CLUS-02-01 | Error on L > ncomp | unit | `cargo test -p fdars-core coclustering::tests::test_error_l_exceeds_ncomp` | ❌ Wave 0 |
| CLUS-02-02 | ICL is finite on well-conditioned data | unit | `cargo test -p fdars-core coclustering::tests::test_icl_is_finite` | ❌ Wave 0 |
| CLUS-02-03 | Slope heuristic selects correct (K,L) | unit | `cargo test -p fdars-core coclustering::tests::test_slope_heuristic_selects_correct_kl` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core coclustering`
- **Per wave merge:** `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`

### Wave 0 Gaps
- [ ] `fdars-core/src/coclustering.rs` — entire new module
- [ ] Inline `#[cfg(test)] mod tests` with all 9 test cases listed in §10.2
- [ ] No separate test file needed (inline pattern throughout codebase)

---

## 16. Security Domain

`security_enforcement: true`, `security_asvs_level: 1`.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Pure numerical computation library |
| V3 Session Management | No | No session state |
| V4 Access Control | No | Library function, no access model |
| V5 Input Validation | Yes | All public functions validate dimensions via `FdarError::InvalidParameter` and `FdarError::InvalidDimension` |
| V6 Cryptography | No | `StdRng` used for reproducibility, not security |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| NaN / ±∞ inputs in FPC scores | Tampering | Validate via dimension checks; `log(0)` guarded by `< 1e-15` threshold |
| K=0 or L=0 integer overflow in `k*l*ncomp` | Tampering | Validate K ≥ 1 and L ≥ 1 before any arithmetic |
| seed = u64::MAX + 1 wrapping | — | `wrapping_add` used throughout codebase — safe |

---

## 17. Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Column-cluster labels in funLBM on FPC scores are over the `ncomp` FPC components (not the original `m` evaluation points) | §1.1 | If the intent is to cluster over `m` points, `col_labels` should have length `m` and the block Gaussian would be over row sub-curves evaluated at column-cluster points — a fundamentally different model requiring careful restatement |
| A2 | CEM (hard assignment) is the simplest valid deterministic scheme and matches R funLBM default | §2.1 | Soft VEM converges to a different (generally better) stationary point; CEM may get stuck in poor local optima |
| A3 | Diagonal block covariance is the recommended choice | §1.2 | Full covariance would require Cholesky (linalg feature gate) and be more numerically fragile in small blocks |
| A4 | ICL formula uses symmetric `0.5 × p_{KL} × (ln n + ln ncomp)` penalty | §3.1 | The row-only `0.5 × p_{KL} × ln n` penalty is also defensible; discrepancy affects model selection (not EM convergence) |
| A5 | Birgé–Massart slope estimation: OLS over top-50% by model dimension | §4.2 | Alternative: top-third, top-quartile. Noisy on small grids (< 6 models). Slope heuristic is a heuristic, not a theorem. |
| A6 | Flat `block_mean[k][l][j]` storage (k×l×ncomp) during EM, converted to `BlockParams` at end | §6.2 note | Redundant memory (`ncomp` entries per block regardless of column-cluster size); for ncomp≤20 and K,L≤10 this is ≤2000 f64 — negligible |

---

## 18. Open Questions

1. **Column-label semantics:** If the user intends L column-clusters over the original `m`
   argument points (not the `ncomp` FPC axes), the block model must operate on a different
   quantity — the curve values at the argument points in column-cluster l, marginalized over FPC
   components, which requires a different block parameterization. **Recommendation:** Confirm with
   CONTEXT.md decision "column-block assignment: arbitrary cluster label per argument point" —
   the `m` evaluation points or the `ncomp` FPC components? The CONTEXT.md says both "per
   argument point" AND "FPC components per block: a fixed d from ncomp", which is ambiguous.
   The research here assumes col_labels has length `ncomp` (over FPC axes). If it should be
   length `m`, the block Gaussian needs a different design (see A1).

2. **n_init restart strategy:** Multiple restarts of CEM add `O(n_init × K × L)` grid fits.
   For large grids (e.g. K∈{2..8} × L∈{2..4}), total fits = `n_init × |k_range| × |l_range|`.
   With n_init=3 and a 7×3 grid, this is 63 full EM runs. Document expected wall-clock time in
   rustdoc.

---

## 19. Project Constraints (from CLAUDE.md)

- No new crate dependency — all code must use existing Cargo.toml dependencies only.
- Additive/non-breaking only — zero changes to existing public signatures.
- All public functions `Result<T, FdarError>`-returning — no panics on input validation.
- Inline `#[cfg(test)] mod tests` — not separate test files.
- `#[derive(Debug, Clone, PartialEq)]` + `#[non_exhaustive]` on all public result structs.
- `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` on public types.
- `#[must_use]` on expensive computation functions.
- MSRV 1.81 — no stabilised APIs newer than 1.81 (diagonal covariance avoids `linalg` / faer 0.23 which requires 1.84).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must be clean.
- `cargo fmt` must be run before commit.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` must prefix any cargo command that builds.
- No examples in this phase (benchmark/example targets are separate).
- Rustdoc must document divergences from R funLBM: global vs per-block FPCA; CEM vs SEM-Gibbs; diagonal vs full covariance.

---

## Sources

### Primary (HIGH confidence — verified by direct file Read this session)
- `src/regression.rs:287-399` — `fdata_to_pc_1d` signature, `FpcaResult` fields, scores layout
- `src/clustering.rs:14-30, 545-607` — `KmeansResult` struct, `kmeans_fd` signature and validation
- `src/gmm/em.rs:1-393` — EM loop pattern, log-sum-exp, covariance accumulation, ICL/BIC helpers
- `src/gmm/init.rs:76-184` — k-means++ init, `kmeans_init_assignments`, `init_params_from_assignments`
- `src/gmm/cluster.rs:11-86` — multiple-restart pattern, `GmmClusterConfig` builder
- `src/gmm/covariance.rs:1-186` — `data_scaled_reg`, `regularize_cov`, `identity_cov`
- `src/gmm/subspace.rs:1-236` — `normalize_log_probs`, E-step subspace pattern
- `src/linalg.rs:1-70` — `cholesky_d`, `forward_solve`, `mahalanobis_sq`, `log_det_from_cholesky`
- `src/parallel.rs:41-55` — `iter_maybe_parallel!` macro exact expansion
- `src/test_helpers.rs:24-93` — `adjusted_rand_index` ARI formula and implementation
- `src/error.rs:1-52` — `FdarError` enum variants and field types
- `src/matrix.rs:1-80` — `FdMatrix` column-major layout, `from_column_major`
- `src/lib.rs:60-170` — module registration pattern
- `src/prelude.rs:1-84` — prelude re-export pattern
- `.planning/config.json:26` — `nyquist_validation: true`

### Secondary (ASSUMED — training knowledge, not verified against primary sources this session)
- funLBM generative model and CEM algorithm: Govaert & Nadif (2008) "Block clustering with
  Bernoulli mixture models", Bouveyron et al. (2015) funLBM paper
- Birgé–Massart slope heuristic: Baudry, Maugis & Michel (2012) "Slope heuristics: overview
  and implementation"
- ICL for LBM: symmetric penalty formula `0.5 × p × (ln n + ln m)`
