# Phase 34: Functional Serial-Dependence Tooling - Research

**Researched:** 2026-08-21
**Domain:** Functional Time Series (FTS) — serial-dependence diagnostics, autocovariance operators, stationarity testing, long-run covariance estimation
**Confidence:** MEDIUM (algorithms verified from R source code + authoritative documentation; limiting-distribution approximation flagged ASSUMED where Monte-Carlo replacement is needed)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- New `fts/` directory with `fts/acf.rs` implementation + `fts/mod.rs` barrel (mirrors the `inference/` module structure).
- One result struct per tool: `FacfResult` (lags, acf, pacf, confidence bands), a stationarity-test result, a long-run-covariance result — each deriving `Debug, Clone, PartialEq` with conditional serde, `#[non_exhaustive]` per convention.
- Named public entry points: `functional_acf`, `functional_pacf`, `stationarity_test`, `long_run_covariance`, `functional_difference` (final names at planner's discretion but this surface).
- Crate-root `pub use` re-export for all entry points and result types (project convention).
- L2-norm functional ACF following the `fdaACF` convention: autocorrelation at lag h as the L2 norm of the lag-h autocovariance operator, normalized by the lag-0 term over the domain (Simpson/quadrature-weighted).
- White-noise confidence bands derived from the strong-white-noise limiting distribution (the `fdaACF` quadratic-form / χ²-mixture band). Document in rustdoc if the limiting distribution is approximated.
- Default lag range `max_lag = min(20, N/4)` when unspecified.
- Partial ACF via Durbin-Levinson-style recursion over the functional ACF sequence.
- Monte-Carlo functional stationarity test (`ftsa` T_stationary style): a test statistic plus a seeded resampling p-value.
- Long-run covariance via a Bartlett kernel-sandwich (HAC) estimator by default, with a bandwidth argument.
- Default bandwidth `⌊N^{1/3}⌋` (standard HAC rule); bandwidth 0 reduces to the lag-0 sample covariance.
- Reproducible randomness via a single shared `StdRng::seed_from_u64(seed)` seed parameter (mirrors the permutation-test convention — NOT per-lag seed+k).
- Functional first-difference operator (order 1): output curve series has length N−1 and round-trips against a cumulative-sum reconstruction within a documented tolerance.
- Return `FdarError` (never panic) on: empty matrix, fewer curves than requested max lag, argvals/values length mismatch, degenerate/zero-variance columns, invalid (negative) bandwidth.
- Numeric output only — no plotting.
- Document any divergence from the R baseline (esp. white-noise band approximation) in rustdoc, per prior-milestone convention.

### Claude's Discretion

- Exact final function/struct names, internal helper factoring, and the precise χ²-mixture band quantile approximation are at the planner/executor's discretion, guided by the fdaACF/ftsa references and codebase conventions.

### Deferred Ideas (OUT OF SCOPE)

- Full functional time series forecasting (ftsm, FPC-regression, fplsr, updating) — FTS-01, deferred to v2.
- Spectral / dynamic FTS methods — FTS-03, deferred.
- Flat-top / Parzen long-run-covariance kernels and configurable higher-order differencing (order d) — kept as future extensions; this phase ships Bartlett + first-difference.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FTS-02 | Add functional serial-dependence tooling in a new `fts/acf.rs` module — L2-norm functional autocorrelation (fACF) and partial ACF (fPACF) with the strong-white-noise limiting distribution for confidence bands, a functional stationarity test, a long-run covariance kernel-sandwich estimator, and a functional differencing operator. Reuses `helpers` quadrature + `covariance.rs`. Additive/non-breaking; independent of REP-01/DENS-01. | §Algorithm Formulations pins each of the 5 entry points; §Reuse Map documents exact helper calls; §Test Strategy documents the numeric assertions from the phase success criteria. |
</phase_requirements>

---

## Summary

Phase 34 adds a self-contained `fts/` module to `fdars-core` implementing five functional serial-dependence tools: the L2-norm functional ACF (fACF), partial ACF (fPACF), strong-white-noise confidence bands, a Monte-Carlo stationarity test, a Bartlett kernel-sandwich long-run covariance estimator, and a functional first-difference operator. The R references are the `fdaACF` package (Mestre et al. 2021, *Computational Statistics & Data Analysis*) and the `ftsa` package (Shang / Hyndman).

The algorithms are mathematically well-understood and the key formulas have been verified from R source code (fdaACF GitHub) and authoritative documentation (ftsa CRAN). The main divergence from R is the white-noise band construction: fdaACF uses Imhof's method (via the `CompQuadForm` R package) as its non-MC path; that external dependency is not available in pure Rust. The executor must implement a Monte-Carlo approximation as the primary path and document this divergence in rustdoc — which is consistent with the CONTEXT.md decision to "document in rustdoc if the limiting distribution is approximated."

The fPACF implementation in fdaACF is **not** a Durbin-Levinson scalar recursion on rho_h values. It is a residual-cross-covariance approach (fit ARH(p-1) forward and backward, compute L2 norm of the cross-covariance of residuals). The CONTEXT.md says "Durbin-Levinson-style recursion over the functional ACF sequence" — this is the implementable scalar Durbin-Levinson applied to the sequence rho_1, rho_2, … (i.e., exactly as in the scalar PACF). This is a known simplification vs the full fdaACF residual method; document divergence in rustdoc.

**Primary recommendation:** Implement the five entry points in `src/fts/acf.rs` with an internal `fts/mod.rs` barrel, reusing `helpers::simpsons_weights`, `helpers::trapz`, `helpers::cumulative_trapz`, and drawing the RNG pattern from `inference/permutation.rs`. No new dependency required.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| fACF / fPACF estimation | Library Core (`fts/acf.rs`) | — | Pure computation over FdMatrix rows |
| White-noise confidence bands | Library Core (`fts/acf.rs`) | `inference/dist.rs` reuse | χ²-mixture quantile via MC; chi_square_sf_df already exists in `inference/dist.rs` if needed for approximation |
| Stationarity test (Monte-Carlo p-value) | Library Core (`fts/acf.rs`) | — | Single-seed StdRng resampling, no external service |
| Long-run covariance (HAC sandwich) | Library Core (`fts/acf.rs`) | — | Bartlett kernel sum over lagged sample covariance operators |
| Functional differencing | Library Core (`fts/acf.rs`) | `helpers::cumulative_trapz` | First-difference of curve series rows; round-trip via cumulative sum |

---

## Standard Stack

### Core (no new dependencies — everything is already in `fdars-core`)

| Item | Version / Location | Purpose | Why Use |
|------|-------------------|---------|---------|
| `FdMatrix` | `src/matrix.rs` | Column-major curve storage; rows = time-ordered curves | Project-wide type; row access via `row_to_buf` / `row_dot` |
| `helpers::simpsons_weights` | `src/helpers.rs:57-86` | Quadrature weights for L2 inner products | Already used throughout; project convention |
| `helpers::trapz` | `src/helpers.rs:234-240` | Scalar integration for normalization | Used for ρ_h normalization (integrate diagonal of C_0) |
| `helpers::cumulative_trapz` | `src/helpers.rs:197-231` | Cumulative integration for differencing round-trip | Used to verify first-difference round-trip |
| `helpers::l2_distance` | `src/helpers.rs:37-44` | L2 distance with weights | Pattern reference; fACF uses a related integral |
| `rand::rngs::StdRng` / `rand::SeedableRng` | `Cargo.toml` (existing) | Seeded RNG for MC bands and stationarity test | Project pattern: `StdRng::seed_from_u64(seed)` |
| `rand_distr::ChiSquared` / `rand::Rng` | `Cargo.toml` (existing) | MC sampling for chi-squared mixture | Available from existing `rand_distr` dep |
| `inference/dist.rs::chi_square_sf_df` | `src/inference/dist.rs:118-132` | Real-df χ² survival function | Reuse for any parametric fallback |

**Installation:** No new dependencies. All required crates (`rand`, `rand_distr`, `nalgebra`) are already in `Cargo.toml`. `rand_distr` is already a dependency (used in `simulation.rs`).

### Supporting (project-internal patterns to mirror)

| Item | Location | Purpose |
|------|----------|---------|
| `inference/permutation.rs` | Pattern: seeded Fisher-Yates + p-value `(n_ge + 1)/(n_perm + 1)` | Exact RNG seeding and MC p-value pattern to copy |
| `inference/mod.rs` | Pattern: barrel with `pub use` + shared `TestResult` | Module structure to mirror for `fts/mod.rs` |
| `inference/anova.rs` | Pattern: Box/Satterthwaite χ² approximation in rustdoc | How to document the approximation and its limitations |

---

## Algorithm Formulations

This section pins the exact, implementable algorithm for each entry point. All formulas verified from R source code (fdaACF GitHub, ftsa CRAN).

### 1. Lag-h Sample Autocovariance Operator (internal helper)

Given N time-ordered curves in FdMatrix `data` (N×m, column-major), and evaluation points `argvals` (length m):

1. Compute the sample mean curve: `xbar[j] = (1/N) Σ_{i=0}^{N-1} data[(i,j)]` for each j.
2. Center: `xi_centered[i,j] = data[(i,j)] - xbar[j]`.
3. The **lag-h sample autocovariance kernel** C_h(s,t) is estimated by the m×m matrix:
   ```
   C_hat_h[j1,j2] = (1/N) Σ_{i=0}^{N-h-1}  xi_centered[i,j1] * xi_centered[i+h,j2]
   ```
   for h ≥ 0. This is an m×m matrix (one entry per grid-point pair).
   
   Storage: allocate an `m×m` flat `Vec<f64>` in column-major order. For the autocovariance at lag 0, use h=0 (this is the sample covariance operator C_0).

**Divergence note:** [ASSUMED] The ftsa and fdaACF packages use `(1/N)` normalization (not `(1/(N-h))`), consistent with the formula above. Executor should match this convention and document it.

### 2. fACF at lag h — `functional_acf`

**Source: verified from fdaACF/R/functional_autocorrelation.R (GitHub)**

The scalar functional autocorrelation at lag h is [CITED: github.com/GMestreM/fdaACF/blob/master/R/functional_autocorrelation.R]:

```
ρ_h = sqrt( ‖Ĉ_h‖²_L2 ) / ∫_T Ĉ_0(t,t) dt
```

where:
- `‖Ĉ_h‖²_L2 = ∫∫_T Ĉ_h(s,t)² ds dt` — the squared Hilbert-Schmidt norm of the m×m autocovariance matrix, computed as a double Simpson-weighted sum over the grid.
- `∫_T Ĉ_0(t,t) dt` — the Simpson-weighted integral of the diagonal of the lag-0 autocovariance matrix (the trace of C_0), used as the normalization denominator.

**Implementable formula:**
```
‖Ĉ_h‖²_L2 ≈ Σ_{j1} Σ_{j2} C_hat_h[j1,j2]² * w[j1] * w[j2]

normalization = Σ_j C_hat_0[j,j] * w[j]   (trapezoidal or Simpson over diagonal)

ρ_h = sqrt(‖Ĉ_h‖²_L2) / normalization
```

where `w[j]` are the `simpsons_weights(argvals)`.

**Return:** `FacfResult { lags: Vec<u32>, acf: Vec<f64>, pacf: Vec<f64>, upper_band: Vec<f64> }`

**Default lags:** h = 1, 2, …, max_lag where `max_lag = min(20, N/4)`.

### 3. White-Noise Confidence Band

**Source: verified from fdaACF/R/estimate_distribution.R (GitHub, raw)**

Under the hypothesis of strong functional white noise (curves i.i.d.), the distribution of `N * ‖Ĉ_h‖²_L2` converges to:

```
Q ~ Σ_{j=1}^K Σ_{k=1}^K λ_j * λ_k * χ²_1(j,k)
```

where λ_1, …, λ_K are the eigenvalues of the lag-0 sample autocovariance operator C_0 (obtained via `nalgebra::SymmetricEigen` on the m×m C_0 matrix), and χ²_1(j,k) are independent chi-squared(1) variables. [CITED: github.com/GMestreM/fdaACF/blob/master/R/estimate_distribution.R]

The **exact Monte-Carlo approach** (which the executor MUST implement as the primary path — the Imhof exact method requires `CompQuadForm` which is not in pure Rust):

```
for _ in 0..n_sim {
    let q: f64 = eigenvalues.iter().flat_map(|&lj| {
        eigenvalues.iter().map(move |&lk| {
            lj * lk * chi2_1_sample   // one independent chi2(1) per (j,k) pair
        })
    }).sum();
    realizations.push(q / n as f64);  // divide by N
}
```

The upper-band threshold at confidence level α (e.g. 0.95) is the (α)-quantile of the realizations. Then:

```
upper_band[h] = sqrt( q_alpha ) / normalization
```

where `normalization` is the same denominator used for `ρ_h`.

**Truncation:** Only use eigenvalues with `λ_j / λ_1 > epsilon` (default epsilon = 1e-4, matching fdaACF's default).

**Divergence to document in rustdoc:** fdaACF offers both MC and Imhof exact methods; this implementation provides MC only (Imhof requires the `CompQuadForm` R package which has no pure-Rust equivalent without new crate dependency). The MC approximation converges with sufficient `n_sim` (default 10_000 matching fdaACF default).

### 4. fPACF — `functional_pacf`

**Source: CONTEXT.md decision; fdaACF uses residual-cross-covariance method, but executor uses scalar Durbin-Levinson as per locked decision.**

The CONTEXT.md locked decision specifies "Partial ACF via Durbin-Levinson-style recursion over the functional ACF sequence." This is the scalar Durbin-Levinson algorithm applied to the sequence ρ_1, ρ_2, … ρ_{max_lag}:

**Scalar Durbin-Levinson on rho sequence:**

Given rho[0] = 1, rho[1], …, rho[max_lag]:

```
pacf[1] = rho[1]
phi[1,1] = rho[1]
for k = 2..=max_lag:
    phi[k,k] = (rho[k] - Σ_{j=1}^{k-1} phi[k-1,j] * rho[k-j]) 
               / (1 - Σ_{j=1}^{k-1} phi[k-1,j] * rho[j])
    for j = 1..=k-1:
        phi[k,j] = phi[k-1,j] - phi[k,k] * phi[k-1,k-j]
    pacf[k] = phi[k,k]
```

**Divergence to document in rustdoc:** The `fdaACF` package computes fPACF via a residual-cross-covariance approach (fit ARH(p-1) models forward and backward using FPCA, correlate residuals). The scalar Durbin-Levinson over ρ_h values is a simpler, valid approximation — it gives the PACF of the scalar-valued sequence {ρ_h} rather than the operator-valued sequence. For diagnosing AR(p) vs MA(q) structure this is practically useful.

**The white-noise band for fPACF** uses the same MC distribution as for fACF (same null: each lag's PACF is approximately 0 under white noise).

### 5. Stationarity Test — `stationarity_test`

**Source: ftsa::T_stationary documentation (CRAN / rdrr.io)**

The `ftsa::T_stationary` test (Horváth, Kokoszka, Rice 2014, *Journal of Econometrics* 179:66–82) tests H0: the functional time series is stationary.

**Algorithm (matched by capability, not R signature):**

The test statistic is a KPSS-style functional test based on partial sums of the centered curves. The implementable version:

1. Center the curves: `x_i_c[j] = data[(i,j)] - xbar[j]`.
2. Compute partial sums: `S_k[j] = Σ_{i=0}^{k-1} x_i_c[j]` for k = 1, …, N.
3. Compute the test statistic T:
   ```
   T = (1/N²) * Σ_{k=1}^{N} ‖S_k‖²_L2
     = (1/N²) * Σ_{k=1}^{N} Σ_j S_k[j]² * w[j]
   ```
4. Compute the long-run variance scaling: multiply T by 1 / ‖Ĉ_LRC‖ (or by a scalar estimate of the long-run variance) to standardize. [ASSUMED — the exact T_stationary normalization involves the long-run covariance operator; pin by reading the HKR 2014 paper at execution time if needed, or use the unnormalized T statistic with pure MC resampling for the p-value.]
5. **Monte-Carlo p-value:** generate `n_perm` (default 999) resampled versions by randomly reordering the centered curves (`StdRng::seed_from_u64(seed)` + Fisher-Yates shuffle on row indices), recomputing T for each permutation, and computing the p-value as `(#{perm_T >= observed_T} + 1) / (n_perm + 1)`.

**Return:** `StationarityResult { statistic: f64, p_value: f64, n_perm: usize }` (mirrors `TestResult` from `inference/`).

**[ASSUMED] regarding exact T_stationary normalization:** The ftsa documentation confirms MC_rep (default 1000) and shows the statistic is KPSS-flavored, but the exact normalization constant (involving a long-run covariance factor) is not pinned from the documentation alone. The permutation p-value is valid regardless of normalization, so the executor may defer the exact normalization to future work and document this as a known divergence.

### 6. Long-Run Covariance — `long_run_covariance`

**Source: ftsa documentation + Berkes/Horváth/Rice 2016 asymptotic normality paper.**

The Bartlett kernel-sandwich (HAC) estimator [CITED: rdrr.io/cran/ftsa/man/long_run_covariance_estimation.html]:

```
Ĉ_LRC(s,t) = Σ_{h=-(N-1)}^{N-1}  K_Bartlett(h / b)  *  Ĉ_h(s,t)
```

where:
- `K_Bartlett(x) = (1 - |x|) * 1_{|x| <= 1}` (Bartlett kernel) [CITED: websearch, standard HAC literature]
- `b` is the bandwidth (integer), default `⌊N^{1/3}⌋` [CITED: websearch — "window size of the kernel is the cube root of the sample size"]
- `Ĉ_h` for h ≥ 0 is the lag-h sample autocovariance operator computed as in §1 above
- `Ĉ_h` for h < 0 = `Ĉ_{|h|}^T` (transpose, since the operator is symmetric for stationary series)
- bandwidth = 0 ⇒ Ĉ_LRC = Ĉ_0 (lag-0 sample covariance), as per the CONTEXT.md locked decision

**Implementation:** The estimator is an m×m matrix. Build by:
1. Compute Ĉ_0 (the m×m sample covariance matrix, already needed for fACF).
2. For h = 1, 2, …, min(b, N-1):
   - `w_h = 1.0 - (h as f64) / (b as f64)`  (Bartlett weight; = 0 when h ≥ b)
   - Compute Ĉ_h (m×m), add `w_h * Ĉ_h` and `w_h * Ĉ_h^T` to the accumulator.
3. Add Ĉ_0 once.

**Return:** `LongRunCovResult { bandwidth: usize, n_curves: usize, cov_matrix: Vec<f64>, m: usize }`
where `cov_matrix` is an m×m matrix stored as flat column-major Vec<f64>.

**Validation at bandwidth=0:** `long_run_covariance(data, argvals, 0, ...)` must return the lag-0 sample covariance matrix.

### 7. Functional Differencing — `functional_difference`

**Source: ftsa::diff.fts documentation (rdrr.io)**

The first-difference operator applied to a time-ordered curve series [CITED: rdrr.io/cran/ftsa/man/diff.fts.html]:

```
D[i,j] = data[(i+1,j)] - data[(i,j)]    for i = 0..=N-2, j = 0..=m-1
```

Output is an `(N-1)×m` FdMatrix.

**Round-trip via cumulative sum:**
```
reconstructed[0,j] = data[(0,j)]
for i = 1..N:
    reconstructed[i,j] = reconstructed[i-1,j] + D[i-1,j]
```

The executor must verify `|reconstructed[i,j] - data[(i,j)]| < tol` (where `tol = 1e-10` or similar) in the unit tests.

**Error:** return `FdarError::InvalidDimension` if N < 2 (cannot difference fewer than 2 curves).

---

## Reuse Map

Precise mapping of which existing codebase functions are reused in each step.

| Need | Reuse | Location |
|------|-------|---------|
| Integration weights | `simpsons_weights(argvals)` | `src/helpers.rs:57` [VERIFIED: src/helpers.rs:57-86] |
| Scalar integration of diagonal | `trapz(diag_values, argvals)` | `src/helpers.rs:234` [VERIFIED: src/helpers.rs:234-240] |
| Cumulative integration (round-trip check) | `cumulative_trapz(y, x)` | `src/helpers.rs:197` [VERIFIED: src/helpers.rs:197-231] |
| Seeded RNG | `StdRng::seed_from_u64(seed)` | pattern from `src/inference/permutation.rs:173` [VERIFIED: src/inference/permutation.rs:173] |
| Fisher-Yates shuffle (stationarity test resampling) | `shuffle_labels` pattern | `src/inference/permutation.rs:120-127` [VERIFIED: src/inference/permutation.rs:120-127] |
| MC p-value formula | `(n_ge + 1) / (n_perm + 1)` | `src/inference/permutation.rs:183` [VERIFIED: src/inference/permutation.rs:183] |
| Module barrel structure | `src/inference/mod.rs` | [VERIFIED: src/inference/mod.rs:1-59] |
| Result struct pattern | `TestResult` in `inference/mod.rs:44-59` | [VERIFIED: src/inference/mod.rs:44-59] |
| GP white-noise test data | `generate_gaussian_process(...)` | `src/covariance.rs` [VERIFIED: src/covariance.rs:1-60] |
| Column-major matrix ops | `FdMatrix::zeros`, `FdMatrix[(i,j)]`, `.column(j)`, `.row_to_buf(i, buf)` | `src/matrix.rs` [VERIFIED: src/matrix.rs:84-213] |
| Eigendecomposition of C_0 | `nalgebra::SymmetricEigen::new(dm)` (via `to_dmatrix()`) | existing nalgebra dep |

**`covariance.rs` for test data generation:**
- `CovKernel::WhiteNoise { variance }` + `generate_gaussian_process(argvals, kernel, n, seed)` → i.i.d. white-noise curves [VERIFIED: src/covariance.rs:55-59]
- `CovKernel::Matern { ... }` → smooth GP curves for non-white-noise tests

**How to compute the lag-h autocovariance matrix from FdMatrix:**

```rust
// w[j] = simpsons_weights(argvals)[j]
// xbar[j] = mean over i of data[(i,j)]
// c_h_mat[j1 + j2*m] = (1.0/n as f64) * Σ_{i=0}^{n-h-1} (data[(i,j1)] - xbar[j1]) * (data[(i+h,j2)] - xbar[j2])
let mut c_h = vec![0.0_f64; m * m];
for i in 0..(n - h) {
    for j1 in 0..m {
        let xi1 = data[(i, j1)] - xbar[j1];
        for j2 in 0..m {
            let xi2 = data[(i + h, j2)] - xbar[j2];
            c_h[j1 + j2 * m] += xi1 * xi2;
        }
    }
}
for x in &mut c_h { *x /= n as f64; }
```

Note: this is O(N·m²) per lag. For m = 50–100 and N = 100–500 this is acceptable. For large m, the executor may add a note that the asymptotic O(N·m²·max_lag) cost is dominated by the double loop and suggest capping m at reasonable values.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Integration weights | Custom quadrature rule | `simpsons_weights(argvals)` from `helpers.rs` | Already handles uniform and non-uniform grids, tested |
| Eigendecomposition | Custom SVD/power-iteration | `nalgebra::SymmetricEigen` (symmetric, real) on `to_dmatrix()` result | nalgebra is already a dep; SymmetricEigen is more stable than general SVD for covariance matrices |
| Chi-squared samples | Hand-rolled chi-squared | `rand_distr::ChiSquared::new(1.0)` + `rng.sample(dist)` | Already in `rand_distr` dep (used in `simulation.rs`) |
| Seeded MC loop | Custom PRNG | `StdRng::seed_from_u64(seed)` + `rand::Rng::gen_range` | Project convention (see `inference/permutation.rs`) |
| Scalar p-value | Hard-coded critical values | `(n_ge + 1) / (n_perm + 1)` MC formula | Exact formula used throughout `inference/`; critical values would need n/alpha/normalization assumptions |

**Key insight:** The chi-squared mixture distribution for the white-noise band is the one place where a new computation is needed — but it is fully implementable in MC form using `rand_distr::ChiSquared` which is already a transitive dependency via `rand_distr`. No new crate dependency is required.

---

## Architecture Patterns

### System Architecture Diagram

```
User code
    │
    ▼
fdars_core::functional_acf(data, argvals, max_lag, n_sim, seed)
    │
    ├─→ validate_fts_input()          [dim checks at entry]
    ├─→ compute_mean_curve()           [xbar[j] = mean over i]
    ├─→ center_curves()               [xi_c[i,j] = data[i,j] - xbar[j]]
    ├─→ helpers::simpsons_weights()   [w[j]]
    │
    ├─→ For h = 1..=max_lag:
    │       compute_autocovariance_matrix(h)   [m×m C_hat_h, O(N·m²)]
    │       l2_hs_norm(C_hat_h, w)             [Σ C_h[j1,j2]² w[j1] w[j2]]
    │       ρ_h = sqrt(norm) / normalization
    │
    ├─→ eigendecompose(C_hat_0)        [nalgebra::SymmetricEigen]
    ├─→ mc_band_quantile(eigenvalues, n, n_sim, seed, ci)
    │       [StdRng::seed_from_u64(seed) → 10_000 chi2-mixture draws → α-quantile]
    │       → upper_band[h] = sqrt(q_alpha) / normalization
    │
    ▼
FacfResult { lags, acf, upper_band, pacf }
    acf[h] = ρ_h
    pacf[h] = Durbin-Levinson scalar recursion on ρ_1..ρ_{max_lag}
    upper_band[h] = sqrt(mc_quantile) / normalization

fdars_core::stationarity_test(data, argvals, n_perm, seed)
    │
    ├─→ center_curves() → partial_sums S_k
    ├─→ T = (1/N²) Σ_k ‖S_k‖²_L2  [Simpson-weighted]
    ├─→ permutation loop (Fisher-Yates shuffle of row indices, recompute T)
    ▼
StationarityResult { statistic, p_value, n_perm }

fdars_core::long_run_covariance(data, argvals, bandwidth)
    │
    ├─→ C_hat_0 (lag-0 covariance, m×m)
    ├─→ for h = 1..bandwidth:
    │       w_h = 1 - h/bandwidth  [Bartlett weight]
    │       C_hat_h (m×m)
    │       accumulate w_h * (C_h + C_h^T)
    ▼
LongRunCovResult { cov_matrix: Vec<f64>, m, bandwidth, n_curves }

fdars_core::functional_difference(data)
    │
    ├─→ D[i,j] = data[i+1,j] - data[i,j]  for i=0..N-2
    ▼
FdMatrix (N-1) × m
```

### Recommended Project Structure

```
fdars-core/src/fts/
├── mod.rs        # barrel: pub use acf::{...}
└── acf.rs        # all 5 entry points + internal helpers
```

Register in `src/lib.rs`:
```rust
pub mod fts;
pub use fts::{
    functional_acf, functional_pacf, stationarity_test, long_run_covariance, functional_difference,
    FacfResult, StationarityResult, LongRunCovResult,
};
```

### Pattern: Result Struct Design

Following the convention from `inference/mod.rs` and `TestResult`:

```rust
/// Result of functional ACF/PACF estimation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FacfResult {
    /// Lag values (1..=max_lag).
    pub lags: Vec<u32>,
    /// Functional autocorrelation ρ_h at each lag.
    pub acf: Vec<f64>,
    /// Functional partial autocorrelation at each lag (Durbin-Levinson scalar approximation).
    pub pacf: Vec<f64>,
    /// Upper confidence band under the strong-white-noise null (Monte-Carlo quantile).
    pub upper_band: Vec<f64>,
}

/// Result of the functional stationarity test.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct StationarityResult {
    /// Test statistic T.
    pub statistic: f64,
    /// Monte-Carlo p-value.
    pub p_value: f64,
    /// Number of permutations used.
    pub n_perm: usize,
}

/// Result of the Bartlett kernel-sandwich long-run covariance estimator.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LongRunCovResult {
    /// Estimated m×m long-run covariance matrix (column-major).
    pub cov_matrix: Vec<f64>,
    /// Grid dimension m (cov_matrix is m×m).
    pub m: usize,
    /// Bandwidth used.
    pub bandwidth: usize,
    /// Number of curves N.
    pub n_curves: usize,
}
```

### Anti-Patterns to Avoid

- **O(N·m²·max_lag) with pre-allocated m×m matrices per lag:** Instead, compute each lag-h autocovariance in a single pass and immediately accumulate into ρ_h; don't store all lag matrices simultaneously unless needed for long-run covariance (which needs them accumulated).
- **Using `data.row(i)` in the hot m² inner loop:** `row()` allocates a Vec per call. Use `row_to_buf(i, buf)` into a pre-allocated `[f64; m]` buffer or access `data[(i,j)]` directly via index.
- **Eigendecomposing C_0 repeatedly:** Compute once, reuse for both the band calculation and any downstream use.
- **Using `rng.gen_range` for chi-squared samples:** Use `rand_distr::ChiSquared::new(1.0)` sampler — correct distribution, already available.

---

## Common Pitfalls

### Pitfall 1: Column-major vs row-major confusion in the autocovariance matrix

**What goes wrong:** Storing C_h as an `m×m` matrix in row-major order then accessing it via column-major indexing (or vice versa) corrupts the Hilbert-Schmidt norm computation.
**Why it happens:** FdMatrix is column-major; the natural autocovariance loop iterates j1 (outer) and j2 (inner), which differs from the column-major storage of C_h if not written carefully.
**How to avoid:** Use `c_h[j1 + j2 * m]` consistently (column-major: j1 is the row index, j2 is the column index, stride is m). Write a unit test asserting `C_h[j1,j2] == C_h[j2,j1]` when h=0 (lag-0 covariance is symmetric).
**Warning signs:** fACF at lag 0 is not 1.0; or fACF values are > 1.0.

### Pitfall 2: Normalization denominator is zero or near-zero

**What goes wrong:** If the diagonal of C_0 integrates to near zero (constant curves), `ρ_h` overflows or returns NaN.
**Why it happens:** All curves are identical or the data is nearly degenerate.
**How to avoid:** Check `normalization < NUMERICAL_EPS` at entry; return `FdarError::ComputationFailed { operation: "functional_acf", detail: "lag-0 covariance diagonal integrates to near zero" }`.
**Warning signs:** NaN in `acf` output.

### Pitfall 3: MC chi-squared mixture uses wrong chi-squared weights

**What goes wrong:** The weights in `Σ_{j,k} λ_j * λ_k * χ²_1` are products of pairs of eigenvalues, not individual eigenvalues. Using `λ_j * χ²_1` (single eigenvalue weight) gives a different distribution.
**Why it happens:** Misreading the fdaACF source. The MC loop is `for jj in 1..K: for kk in 1..K: sum += l[jj]*l[kk]*rchisq(1)`.
**How to avoid:** The double loop is correct. Cross-check: with one eigenvalue λ, the sum is λ²·χ²_1, and `E[sum/N] ≈ λ²/N` which should converge to the correct variance.
**Warning signs:** White-noise ACF values significantly exceed or fall below the confidence band on simulated i.i.d. data.

### Pitfall 4: Durbin-Levinson numerical instability for large lags

**What goes wrong:** The Durbin-Levinson denominator `1 - Σ phi[k-1,j] * rho[j]` can approach zero if the ρ sequence is highly correlated.
**Why it happens:** The recursion's denominator represents 1 minus the model variance explained — it can vanish for near-unit-root series.
**How to avoid:** Check the denominator before division; if `|denominator| < NUMERICAL_EPS`, set `pacf[k] = 0.0` and stop the recursion (or document truncation in the result).

### Pitfall 5: Wrong long-run covariance boundary handling

**What goes wrong:** Including h = bandwidth in the Bartlett sum gives `K_Bartlett(b/b) = 0`, which is correct (contributes nothing) but wastes an autocovariance computation.
**How to avoid:** Loop `h = 1..bandwidth` (exclusive), computing `w_h = 1.0 - h as f64 / b as f64`. At h = b, `w_h = 0.0`, so the loop body can break early or use `h < bandwidth` as the termination.

### Pitfall 6: /tmp exhaustion blocking doctests

**What goes wrong:** `cargo test` fails with "No space left on device" because the doctest linker uses /tmp tmpfs which fills up.
**How to avoid:** Set `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` before running cargo commands. See MEMORY.md pointer `tmp-exhaustion-blocks-precommit.md`. This phase adds real code + doctests, so this pointer is directly relevant.

---

## Code Examples

### Example: Computing the lag-h autocovariance matrix (internal helper)

```rust
// Source: fdaACF R package algorithm, adapted for column-major FdMatrix
fn autocovariance_matrix(data: &FdMatrix, xbar: &[f64], h: usize) -> Vec<f64> {
    let (n, m) = data.shape();
    let mut c_h = vec![0.0f64; m * m];
    for i in 0..(n - h) {
        for j1 in 0..m {
            let xi1 = data[(i, j1)] - xbar[j1];
            for j2 in 0..m {
                let xi2 = data[(i + h, j2)] - xbar[j2];
                c_h[j1 + j2 * m] += xi1 * xi2;
            }
        }
    }
    let inv_n = 1.0 / n as f64;
    for x in &mut c_h { *x *= inv_n; }
    c_h
}
```

### Example: Hilbert-Schmidt L2 norm of autocovariance matrix

```rust
// Source: fdaACF obtain_suface_L2_norm algorithm, adapted
fn hs_norm_sq(c_h: &[f64], m: usize, weights: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for j1 in 0..m {
        let w1 = weights[j1];
        for j2 in 0..m {
            let val = c_h[j1 + j2 * m];
            sum += val * val * w1 * weights[j2];
        }
    }
    sum
}
```

### Example: MC white-noise band (chi-squared mixture)

```rust
// Source: fdaACF estimate_iid_distr_MC algorithm, adapted
// eigenvalues: truncated to those > epsilon * eigenvalues[0]
fn mc_band_threshold(
    eigenvalues: &[f64], n: usize, n_sim: usize, ci: f64, seed: u64,
) -> f64 {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{ChiSquared, Distribution};
    let mut rng = StdRng::seed_from_u64(seed);
    let chi2 = ChiSquared::new(1.0).expect("df=1 is valid");
    let k = eigenvalues.len();
    let mut realizations = Vec::with_capacity(n_sim);
    for _ in 0..n_sim {
        let mut q = 0.0f64;
        for &lj in eigenvalues {
            for &lk in eigenvalues {
                q += lj * lk * chi2.sample(&mut rng);
            }
        }
        realizations.push(q / n as f64);
    }
    realizations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((ci * n_sim as f64) as usize).min(n_sim - 1);
    realizations[idx]
}
```

### Example: Scalar Durbin-Levinson for fPACF

```rust
// Source: classical Durbin-Levinson algorithm applied to scalar rho sequence
fn durbin_levinson_pacf(rho: &[f64]) -> Vec<f64> {
    let p = rho.len(); // rho[0] = rho_1, rho[1] = rho_2, ...
    if p == 0 { return vec![]; }
    let mut phi = vec![vec![0.0f64; p + 1]; p + 1]; // phi[k][j]: 1-indexed
    let mut pacf = vec![0.0f64; p];
    phi[1][1] = rho[0];
    pacf[0] = rho[0];
    for k in 2..=p {
        let num = rho[k - 1] - (1..k).map(|j| phi[k-1][j] * rho[k-1-j]).sum::<f64>();
        let den = 1.0 - (1..k).map(|j| phi[k-1][j] * rho[j-1]).sum::<f64>();
        if den.abs() < 1e-12 { pacf[k-1] = 0.0; break; }
        phi[k][k] = num / den;
        for j in 1..k {
            phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k-j];
        }
        pacf[k-1] = phi[k][k];
    }
    pacf
}
```

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|-----------------|-------|
| Scalar ACF on mean curve (period detection in `seasonal/`) | L2-norm functional ACF on operator-valued process | Distinct; do not reuse the scalar path |
| `fdaACF` Imhof exact method for bands | MC approximation (this impl) | Imhof requires CompQuadForm R package; no pure Rust equivalent without new dep |
| `fdaACF` ARH(p-1) residual fPACF | Scalar Durbin-Levinson on rho_h | Simpler; document divergence |

**Deprecated/outdated:**
- Using the `seasonal::autocorrelation` helper for any of these: that function operates on scalar time series of mean-curve values, not on the functional ACF operator. Do not use or reuse it.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The T_stationary test statistic uses `T = (1/N²) Σ_k ‖S_k‖²_L2` (KPSS-style partial-sum norm) normalized by the long-run variance | §Algorithm Formulations §5 | Stationarity test has wrong null rejection rate; needs to be pinned from HKR 2014 paper during execution |
| A2 | The normalization `(1/N)` for lag-h autocovariance (not `1/(N-h)`) matches ftsa/fdaACF conventions | §Algorithm Formulations §1 | Minor bias difference at small N; practical impact small |
| A3 | `rand_distr::ChiSquared` is already a transitive dependency via the existing `rand_distr` dep | §Standard Stack | Would require new dep if wrong; executor should verify with `cargo tree -p fdars-core | grep rand_distr` |
| A4 | The Bartlett bandwidth default `⌊N^{1/3}⌋` is correct for the `ftsa::long_run_covariance_estimation` function | §Algorithm Formulations §6 | Different default bandwidth would produce different LRC estimate; pin by reading ftsa source at execution time if needed |

---

## Open Questions

1. **Exact T_stationary normalization constant**
   - What we know: T_stationary uses MC_rep replications and is KPSS-flavored (partial-sum functional test)
   - What's unclear: the exact scaling by long-run variance — does it use an estimated long-run variance as denominator, or is the p-value purely Monte-Carlo from row-permutations?
   - Recommendation: implement with pure permutation p-value first (always valid), document the exact HKR 2014 normalization as a future precision improvement.

2. **n_sim default for MC band**
   - fdaACF uses 10,000 by default. For a library function this may be expensive.
   - Recommendation: default to 1000 (matching `DEFAULT_N_PERM = 999` convention in `inference/`) with a note that 10,000 gives better band precision. Expose as a parameter.

3. **Memory cost of lag-h autocovariance matrices**
   - For large m (200+ grid points) the m×m autocovariance matrix is 320KB+ per lag, and max_lag = 20 lags = 6.4MB total.
   - Recommendation: compute and discard each lag-h matrix after extracting ρ_h; only keep the C_0 matrix and the LRC accumulator.

---

## Environment Availability

All required tools are already part of the project:

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `rust` stable | Build | ✓ | 1.97.0 (dev) | — |
| `rand_distr` | MC chi-squared sampling | ✓ (transitive) | 0.4 | — |
| `nalgebra` | Eigendecomposition of C_0 | ✓ | 0.33 | — |
| `cargo clippy --all-targets --features linalg,parallel` | CI gate | ✓ | see MSRV | — |

No external services, no new packages required.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `#[cfg(test)]` (inline) |
| Config file | none — standard Cargo test |
| Quick run command | `cargo test -p fdars-core fts 2>&1` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel 2>&1` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| FTS-02 | fACF on i.i.d. white-noise curves: all lags inside the band | unit | `cargo test -p fdars-core fts::acf::tests::facf_whitenoise_inside_band` |
| FTS-02 | fACF on lag-1 AR curves: lag-1 fACF exceeds band | unit | `cargo test -p fdars-core fts::acf::tests::facf_ar1_exceeds_band` |
| FTS-02 | fPACF cuts off after AR order | unit | `cargo test -p fdars-core fts::acf::tests::fpacf_ar1_cutoff` |
| FTS-02 | functional_difference round-trips via cumulative sum | unit | `cargo test -p fdars-core fts::acf::tests::diff_roundtrip` |
| FTS-02 | stationarity_test: does not reject stationary series | unit | `cargo test -p fdars-core fts::acf::tests::stat_test_stationary` |
| FTS-02 | stationarity_test: rejects trended series | unit | `cargo test -p fdars-core fts::acf::tests::stat_test_nonstationary` |
| FTS-02 | long_run_covariance: bandwidth=0 returns sample covariance C_0 | unit | `cargo test -p fdars-core fts::acf::tests::lrc_bandwidth_zero` |
| FTS-02 | All entry points return FdarError on invalid input | unit | `cargo test -p fdars-core fts::acf::tests::error_handling` |
| FTS-02 | Results are deterministic across identical seeds | unit | `cargo test -p fdars-core fts::acf::tests::deterministic_seed` |

### Synthetic Data Generators (for tests)

All test data uses the existing `covariance::generate_gaussian_process`:

1. **White-noise curves:** `CovKernel::WhiteNoise { variance: 1.0 }` — i.i.d. curves, each independent of the others.
2. **Lag-1 AR curves:** Generate curves `X_i = 0.8 * X_{i-1} + ε_i` where each ε_i is a GP sample with `CovKernel::Gaussian { ... }`. This creates functional AR(1) dependence where lag-1 fACF should exceed the white-noise band.
3. **Trended series (non-stationary):** Add a deterministic linear trend to each curve: `X_i(t) = i * t + GP_sample`.
4. **Stationary series:** `CovKernel::Matern { ... }` samples with no trend.

### Sampling Rate

- **Per-task commit:** `cargo test -p fdars-core fts`
- **Per-wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `src/fts/mod.rs` — module barrel (Wave 0 creates this)
- [ ] `src/fts/acf.rs` — all 5 entry points + inline tests (Wave 0 creates this)
- [ ] Register `pub mod fts;` in `src/lib.rs` + `pub use fts::{...}` re-exports (Wave 0 creates this)

---

## Security Domain

`security_enforcement: true` per config.json; `security_asvs_level: 1`.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | no user auth in a library crate |
| V3 Session Management | no | stateless computation |
| V4 Access Control | no | no access control surface |
| V5 Input Validation | yes | validate dimensions at entry (`FdarError::InvalidDimension`), parameter ranges (`FdarError::InvalidParameter` for negative bandwidth / zero n_perm) |
| V6 Cryptography | no | StdRng is for reproducibility, not security — not applicable |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow in autocovariance loop index `i + h` | Tampering | Use saturating/checked arithmetic or validate `h < n` at entry (already in error spec) |
| NaN propagation from degenerate (zero-variance) input | Tampering | Check normalization denominator > NUMERICAL_EPS before division |
| Stack overflow from large `phi` matrix in Durbin-Levinson | Denial of service | Allocate on heap (`Vec<Vec<f64>>`), not stack; current algorithm already does this |

No network, filesystem, or external service calls. Security surface is limited to input validation.

---

## Package Legitimacy Audit

No new external packages are introduced in this phase. All computation uses existing `fdars-core` dependencies.

| Package | Status | Notes |
|---------|--------|-------|
| `rand_distr` | Existing dep | Used for chi-squared sampling; confirmed present in `Cargo.toml` |
| `nalgebra` | Existing dep | Used for eigendecomposition; confirmed present in `Cargo.toml` |

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

---

## Project Constraints (from CLAUDE.md)

- Column-major `FdMatrix` throughout: `data[(i,j)]` = `data[i + j*nrows]`, `FdMatrix::zeros(nrows, ncols)`.
- All public functions return `Result<T, FdarError>` — never panic on input validation.
- `FdarError` variants: `InvalidDimension { parameter, expected, actual }`, `InvalidParameter { parameter, message }`, `ComputationFailed { operation, detail }`.
- All public types derive `Debug, Clone, PartialEq`; `#[non_exhaustive]` on result structs.
- Conditional serde: `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`.
- Inline `#[cfg(test)] mod tests { ... }` — no separate test files.
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- Crate-root `pub use fts::{...}` re-exports for all entry points and result types.
- MSRV 1.81.0 — no features requiring 1.84+ (the `linalg` feature gating faer uses `#[cfg(feature = "linalg")]`; this phase does NOT require linalg since nalgebra SVD is already available).
- No new crate dependencies.
- Additive/non-breaking: zero changes to existing public signatures.

---

## Sources

### Primary (algorithm-level verification from source code)

- [fdaACF R/functional_autocorrelation.R — fACF and fPACF algorithms](https://raw.githubusercontent.com/GMestreM/fdaACF/master/R/functional_autocorrelation.R) — fACF normalization formula and fPACF residual method
- [fdaACF R/estimate_distribution.R — MC and Imhof band algorithms](https://raw.githubusercontent.com/GMestreM/fdaACF/master/R/estimate_distribution.R) — chi-squared mixture weights λ_j·λ_k
- [rdrr.io T_stationary docs — stationarity test arguments](https://rdrr.io/cran/ftsa/man/T_stationary.html) — MC_rep=1000, KPSS-style statistic
- [rdrr.io facf docs — fACF formula ρ_h](https://search.r-project.org/CRAN/refmans/ftsa/html/facf.html) — formula `ρ̂ᵢ = ‖γ̂ᵢ‖ / ∫ γ̂₀(t,t)dt`
- [rdrr.io diff.fts docs — differencing](https://search.r-project.org/CRAN/refmans/ftsa/html/diff.fts.html) — first-difference at lag=1
- fdars-core source files read this session — `helpers.rs`, `inference/permutation.rs`, `inference/mod.rs`, `inference/dist.rs`, `covariance.rs`, `matrix.rs`, `lib.rs`

### Secondary (MEDIUM confidence)

- [fdaACF CRAN page](https://cran.r-project.org/web/packages/fdaACF/index.html) — Mestre et al. 2021 reference
- [ftsa CRAN long_run_covariance docs](https://rdrr.io/cran/ftsa/man/long_run_covariance_estimation.html) — Bartlett kernel confirmed, adaptive bandwidth algorithm

### Tertiary (LOW confidence — training knowledge)

- Bartlett kernel formula `K(x) = (1-|x|)·1_{|x|≤1}` — standard HAC literature; bandwidth `⌊N^{1/3}⌋` confirmed by websearch citing "cube root of sample size"
- Durbin-Levinson scalar recursion for PACF — classical time series; code example in §Code Examples verified internally consistent

---

## Metadata

**Confidence breakdown:**

- fACF algorithm: HIGH — verified from fdaACF R source code on GitHub
- White-noise band construction: HIGH — verified from fdaACF estimate_distribution.R source code
- fPACF (scalar Durbin-Levinson): MEDIUM — algorithm is classical and correct; divergence from fdaACF's ARH residual method is intentional per CONTEXT.md and must be documented
- Stationarity test (T_stationary): MEDIUM — statistic type confirmed (KPSS-partial-sum, MC p-value); exact normalization constant flagged ASSUMED
- Long-run covariance (Bartlett kernel): MEDIUM — kernel formula and bandwidth rule confirmed from documentation and websearch; ftsa source code not read directly
- Functional differencing: HIGH — diff.fts documentation is explicit

**Research date:** 2026-08-21
**Valid until:** 2027-02-21 (stable R packages; algorithms unlikely to change)
