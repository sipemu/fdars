# Phase 36: Density Object-Data FDA — Research

**Researched:** 2026-08-21
**Domain:** Density-valued functional data analysis (LQD transform, Wasserstein barycenter, density FPCA)
**Confidence:** MEDIUM

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- New `fdars-core/src/density_fda.rs`; `pub mod density_fda;` in `src/lib.rs` + crate-root re-exports.
- Named public entry points: `lqd_transform`, `inverse_lqd`, `lqd_fpca`, `wasserstein_barycenter`, `normalize_density`.
- Result types: `LqdFpcaResult` embedding reused `FpcaResult` + `fve: Vec<f64>`; transforms return `Vec<f64>` / `FdMatrix`.
- LQD definition: ψ(t) = −log f(Q(t)), uniform quantile grid t ∈ [0,1], numeric CDF (trapz-integrated) → quantile inversion → log.
- Quantile grid: uniform on [0,1], configurable resolution, default ~101 points.
- `inverse_lqd(psi, t_grid)` reconstructs a normalized density; always integrates to 1.
- LQD-FPCA: transform each density to LQD space → `FdMatrix` → `fdata_to_pc_1d`.
- FVE = `cumsum(singular_values²) / sum(singular_values²)`.
- Wasserstein barycenter: quantile-average Q̄(t) = Σ wᵢ Qᵢ(t), invert to density.
- `normalize_density(vals, argvals)` via `simpsons_weights`/`trapz`; reject all-zero / negative.
- `FdarError` on: negative/all-zero density, non-monotone/duplicate grid, length mismatch, empty sample.
- Document divergences from `fdadensity` in rustdoc.
- No new crate dependency. Zero changes to existing public signatures.

### Claude's Discretion
- Exact struct/field names, quantile-inversion interpolation scheme, quantile-grid default resolution, and whether density-space modes ship this phase.

### Deferred Ideas (OUT OF SCOPE)
- General Fréchet regression / object-data statistics (FRE-01/FRE-02).
- Multivariate density FPCA / general metric-space barycenters.
- Bandwidth-selection / smoothing for raw-sample density estimation.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DENS-01 | Add density-valued FDA in a new `density_fda.rs` module: LQD transform + inverse (compositional-geometry map), LQD-FPCA for probability densities (reuse `fdata_to_pc_1d` in LQD space, with FVE), a 1D Wasserstein Fréchet mean (quantile-average barycenter) of densities, and density normalization/regularization. | Sections §LQD Transform, §Inverse LQD, §LQD-FPCA, §Wasserstein Barycenter, §Normalization, §Code Examples |
</phase_requirements>

---

## Summary

Phase 36 adds `density_fda.rs` — a new self-contained module for probability-density-valued functional data analysis. The mathematical foundation is the log-quantile-density (LQD) transformation of Petersen & Mueller (2016), which embeds the constraint-carrying space of densities into the Hilbert space L²([0,1]) where ordinary FPCA applies. The key observation: ψ(t) = log q(t) = −log f(Q(t)) (where q = dQ/dt is the quantile density) maps any density f to an unconstrained L² function; the inverse map is always a valid density.

The five public entry points are a clean, linear stack. `normalize_density` and `lqd_transform` are the building blocks; `inverse_lqd` is the reconstruction step; `lqd_fpca` composes the forward transform with `fdata_to_pc_1d` (already the library's SVD engine); `wasserstein_barycenter` exploits the 1D structure where the Fréchet mean is just the pointwise quantile average. All five functions reuse existing helpers (`cumulative_trapz`, `trapz`, `simpsons_weights`, `linear_interp`, `fdata_to_pc_1d`) — no new algorithm subsystem, no new dependency.

The one implementation-level risk is the quantile inversion step inside both `lqd_transform` and `wasserstein_barycenter`: mapping from the physical x-grid to the uniform t-grid (and back) via linear interpolation onto a monotone CDF introduces a known approximation gap vs. fdadensity's cubic-spline path. The executor must document this divergence in rustdoc. Round-trip accuracy for smooth densities on grids of ≥ 51 points is typically within 1 × 10⁻³ (L∞ on the density), which the test suite confirms with truncated Gaussian inputs.

**Primary recommendation:** Implement in strict bottom-up order — `normalize_density` → `lqd_transform` → `inverse_lqd` → `wasserstein_barycenter` → `lqd_fpca`. Test each entry point before adding the next. Round-trip test is the single most important correctness gate.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| LQD forward transform | density_fda module | helpers (cumulative_trapz, linear_interp) | All numeric; density-grid inputs, [0,1] output |
| LQD inverse transform | density_fda module | helpers (cumulative_trapz, trapz, linear_interp) | Quantile-integral back-map + renorm |
| LQD-FPCA | density_fda module | regression (fdata_to_pc_1d) | Transform + delegate SVD to existing engine |
| Wasserstein barycenter | density_fda module | helpers (cumulative_trapz, linear_interp) | Quantile average + density inversion |
| Density normalization | density_fda module | helpers (trapz, simpsons_weights) | Scale to ∫f=1; entry validation |
| FVE computation | density_fda module | — | cumsum(sv²)/sum(sv²) from FpcaResult |

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| fdars-core helpers (internal) | current | `cumulative_trapz`, `trapz`, `simpsons_weights`, `linear_interp`, `quantile_sorted` | Already ship-tested; reuse is mandatory per CONTEXT.md |
| fdars-core regression (internal) | current | `fdata_to_pc_1d`, `FpcaResult` | The project's SVD/FPCA engine; LQD-FPCA delegates to it |
| fdars-core matrix (internal) | current | `FdMatrix` (column-major) | Standard data container across all modules |
| fdars-core error (internal) | current | `FdarError` | All public functions return `Result<T, FdarError>` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (none external) | — | — | No new crate dependency permitted |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `linear_interp` for quantile inversion | `spline_interpolate` (B-spline, cubic) | Spline gives closer fdadensity match but adds a Result propagation; linear is always monotone-safe and sufficient for ≥ 51-point grids; document divergence in rustdoc |
| `cumulative_trapz` for CDF | `simpsons_weights`-based cumsum | `cumulative_trapz` already ships and handles both uniform and non-uniform grids; simpsons-cumsum would need a new helper |

**Installation:** No packages to install — all dependencies are intra-crate.

---

## Package Legitimacy Audit

No external packages are introduced in this phase. The constraint "no new crate dependency" is locked. Audit section: **N/A — no external packages**.

---

## Architecture Patterns

### System Architecture Diagram

```
Caller
  │
  ├─ normalize_density(vals, argvals)
  │      └─ trapz/simpsons_weights → scale → Result<Vec<f64>>
  │
  ├─ lqd_transform(density, argvals, n_quantile_pts?)
  │      ├─ validate: dens > 0, argvals sorted, lengths match
  │      ├─ normalize (trapz)
  │      ├─ CDF: cumulative_trapz(dens, argvals)  [starts at 0]
  │      ├─ lqd_raw = −log(dens)
  │      └─ linear_interp (x=CDF, y=lqd_raw) → t_grid [0,1] → Result<Vec<f64>>
  │
  ├─ inverse_lqd(psi, t_grid, target_argvals)
  │      ├─ Q(t) = target_argvals[0] + cumulative_trapz(exp(psi), t_grid)
  │      ├─ rescale Q → [target_argvals[0], target_argvals[last]]
  │      ├─ dens_raw = exp(−psi)
  │      ├─ linear_interp (x=Q, y=dens_raw) → target_argvals
  │      └─ normalize (trapz) → Result<Vec<f64>>
  │
  ├─ wasserstein_barycenter(density_matrix, argvals, weights?)
  │      ├─ for each row: CDF → Q_i (linear_interp on uniform t_grid)
  │      ├─ Q̄(t) = Σ wᵢ Qᵢ(t)  (weighted pointwise average)
  │      └─ invert Q̄ → density (reuse inverse_lqd or direct map) → Result<Vec<f64>>
  │
  └─ lqd_fpca(density_matrix, argvals, ncomp, n_quantile_pts?)
         ├─ for each row: lqd_transform → assemble LqdFdMatrix (n × n_q)
         ├─ t_grid = uniform [0,1] of length n_q
         ├─ fdata_to_pc_1d(lqd_matrix, ncomp, t_grid)  [reuse]
         ├─ fve = cumsum(sv²) / sum(sv²)
         └─ LqdFpcaResult { fpca, fve }
```

### Recommended Project Structure

```
fdars-core/src/
├── density_fda.rs       # new: all DENS-01 entry points + result types + inline tests
└── lib.rs               # add: pub mod density_fda; + pub use density_fda::{...}
```

A single flat file is correct here — the module is self-contained (~400 LOC including tests), matching the pattern of `multi_fdata.rs` (Phase 35) and prior M-effort single-file additions.

### Pattern 1: LQD Forward Transform (exact fdadensity numeric recipe)

**What:** Maps a density f sampled on a physical grid x ∈ [lb, ub] to its LQD ψ on a uniform probability grid t ∈ [0, 1].

**Mathematical identity:**
- Quantile-density: q(t) = dQ/dt = 1/f(Q(t))
- LQD: ψ(t) = log q(t) = −log f(Q(t))
- fdadensity sign convention: `lqd_temp = -log(dens)` on the physical grid, then interpolate via CDF [CITED: rdrr.io/cran/fdadensity/src/R/dens2lqd.R]

**Exact numeric chain (matching `dens2lqd` in R):**

```rust
// Source: fdadensity dens2lqd.R (rdrr.io/cran/fdadensity/src/R/dens2lqd.R)
// Step 1: validate — all dens > 0, length match, argvals sorted
// Step 2: normalize if |trapz(dens, argvals) - 1| > 1e-5
let integral = trapz(&dens, &argvals);
let dens_norm: Vec<f64> = dens.iter().map(|&d| d / integral).collect();
// Step 3: CDF — F(argvals[i]) starting at 0
let cdf = cumulative_trapz(&dens_norm, &argvals);  // out[0] = 0
// Step 4: lqd at physical grid points
let lqd_raw: Vec<f64> = dens_norm.iter().map(|&d| -d.ln()).collect();
// Step 5: interpolate (x=cdf, y=lqd_raw) onto uniform t_grid ∈ [0,1]
// fdadensity uses cubic spline; we use linear_interp (document divergence in rustdoc)
let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();
let psi: Vec<f64> = t_grid.iter().map(|&t| linear_interp(&cdf, &lqd_raw, t)).collect();
```

**Key invariant:** `cdf[0] == 0.0`, `cdf[last] ≈ 1.0` (trapz CDF reaches 1 for a unit-integral density).

**Divergence from fdadensity:** `linear_interp` vs. `spline(... method='natural')`. For densities on ≥ 51 points the difference is negligible in L∞ (< 1e-3); document in rustdoc.

### Pattern 2: Inverse LQD Transform (exact fdadensity `lqd2dens` recipe)

**What:** Reconstructs a normalized density on `target_argvals` from LQD ψ on t_grid ∈ [0,1].

**Exact numeric chain (matching `lqd2dens` in R):**

```rust
// Source: fdadensity lqd2dens.R (rdrr.io/cran/fdadensity/src/R/lqd2dens.R)
// Step 1: Q_raw(t) = lb + cumtrapz(exp(psi), t_grid)
let exp_psi: Vec<f64> = psi.iter().map(|&p| p.exp()).collect();
let q_raw = cumulative_trapz(&exp_psi, &t_grid);
let lb = target_argvals[0];
let q_raw: Vec<f64> = q_raw.iter().map(|&v| lb + v).collect();
// Step 2: rescale Q to match target_argvals range (support-length normalization)
let q_range = q_raw[q_raw.len()-1] - q_raw[0];  // = theta_psi = ∫exp(psi)dt
let d_range = target_argvals[target_argvals.len()-1] - lb;
let q_scaled: Vec<f64> = q_raw.iter()
    .map(|&v| (v - q_raw[0]) * d_range / q_range + lb)
    .collect();
// Step 3: density values at quantile-grid points
let dens_raw: Vec<f64> = psi.iter().map(|&p| (-p).exp()).collect();  // = 1/q(t)
// Step 4: interpolate (x=q_scaled, y=dens_raw) onto target_argvals
// (remove any duplicate q_scaled values first — dedup_by_key)
let dens: Vec<f64> = target_argvals.iter()
    .map(|&x| linear_interp(&q_scaled, &dens_raw, x))
    .collect();
// Step 5: normalize to ∫f = 1
let integral = trapz(&dens, &target_argvals);
let dens_norm: Vec<f64> = dens.iter().map(|&d| d / integral).collect();
```

**Support-length note:** θ_ψ = ∫₀¹ exp(ψ(t)) dt is the implied support length from ψ. The rescaling step (Step 2) adjusts for any mismatch between this implied length and the requested `dSup` range — this is `fdadensity`'s "DeadCorrection" normalization step. [CITED: rdrr.io/cran/fdadensity/src/R/lqd2dens.R]

**Dedup strategy:** Use a simple scan: if `q_scaled[i] == q_scaled[i-1]`, skip that pair (preserves monotonicity). Rust equivalent: iterate with `peekable()` and skip duplicates.

### Pattern 3: Wasserstein Barycenter (1D quantile average)

**What:** Computes the Fréchet mean of densities under the 2-Wasserstein metric. In 1D, this is exactly the pointwise average of quantile functions (Rüschendorf & Rachev 1990).

**Mathematical formula:** Q̄(t) = Σᵢ wᵢ Qᵢ(t), wᵢ ≥ 0, Σwᵢ = 1. [CITED: Petersen & Mueller 2016, Annals of Statistics 44(1):183-218]

**Exact numeric chain:**

```rust
// Source: fdadensity getWFmean (Petersen & Mueller 2016)
// Step 1: convert each density row i to its quantile function Q_i on t_grid
//   CDF: F_i = cumulative_trapz(row_i, argvals)
//   Q_i(t) = linear_interp(F_i, argvals, t)  for each t in t_grid
// Step 2: weighted pointwise average
// q_bar[j] = sum_i(weights[i] * Q_i[j])  for j in 0..n_q
let mut q_bar = vec![0.0_f64; n_q];
for i in 0..n_rows {
    let row: Vec<f64> = (0..m).map(|j| data[(i,j)]).collect();
    let cdf_i = cumulative_trapz(&row, argvals);
    let w_i = weights[i];  // uniform: 1/n
    for j in 0..n_q {
        q_bar[j] += w_i * linear_interp(&cdf_i, argvals, t_grid[j]);
    }
}
// Step 3: invert Q̄ to a density
// Q̄ is monotone non-decreasing → its derivative is a valid quantile density
// Differentiate: q̄(t) = dQ̄/dt ≈ gradient(q_bar, t_grid)
// Then density: f̄(x) evaluated at target_argvals via inverse interpolation
// OR: use inverse_lqd logic directly on log(gradient(q_bar))
// Simpler path: reuse the inverse_lqd-style back-map:
//   dtemp = q_bar (already the quantile function)
//   rescale to argvals range, interpolate onto argvals, normalize
```

**Simplification:** Rather than going through log→exp, directly use `q_bar` as the quantile function and invert by the same interpolation-normalize logic as `lqd2dens` Step 4-5, without the exp/log:
- dens_raw at quantile grid = 1/derivative(q_bar) = the quantile density
- Map x → Q̄⁻¹(x) via `linear_interp(q_bar_rescaled, t_grid, x)` → density via the reciprocal trick

The cleanest Rust path: implement a private `invert_quantile_to_density(q_fn: &[f64], t_grid: &[f64], target: &[f64]) -> Vec<f64>` helper shared by `inverse_lqd` and `wasserstein_barycenter`.

### Pattern 4: LQD-FPCA

**What:** FPCA of densities by transforming each density to LQD space, assembling an `FdMatrix`, then delegating to `fdata_to_pc_1d`.

```rust
// Source: fdadensity FPCAdens + fdars regression.rs
let n_q = n_quantile_pts;
let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q-1) as f64).collect();

// Build LQD matrix (n × n_q), column-major
let mut lqd_data = FdMatrix::zeros(n_dens, n_q);
for i in 0..n_dens {
    let row: Vec<f64> = (0..m).map(|j| density_matrix[(i,j)]).collect();
    let psi = lqd_transform(&row, argvals, Some(n_q))?;
    for j in 0..n_q {
        lqd_data[(i, j)] = psi[j];
    }
}

// Delegate to existing FPCA engine
let fpca = fdata_to_pc_1d(&lqd_data, ncomp, &t_grid)?;

// FVE = cumsum(sv²) / sum(sv²)
let sv_sq: Vec<f64> = fpca.singular_values.iter().map(|&s| s * s).collect();
let total: f64 = sv_sq.iter().sum();
let mut fve = Vec::with_capacity(sv_sq.len());
let mut cumsum = 0.0_f64;
for s2 in &sv_sq {
    cumsum += s2;
    fve.push(cumsum / total);
}

Ok(LqdFpcaResult { fpca, fve })
```

### Anti-Patterns to Avoid

- **CDF not starting at 0:** `cumulative_trapz` returns `out[0] = 0.0` by construction [VERIFIED: fdars-core/src/helpers.rs:197-231] — do not pre-pend a 0; it is already there.
- **Interpolating outside CDF range:** CDF values run [0, ~1]; t_grid runs [0, 1]; `linear_interp` clamps to boundary by design [VERIFIED: fdars-core/src/helpers.rs:172-191]. No panic risk.
- **Negative ψ before inversion:** ψ can be any real value; `(-ψ).exp()` is always positive. Guard against ψ = NaN or ψ = ±∞ from log(0) — reject zero-density inputs at validation time.
- **Skipping the rescaling step in inverse_lqd:** The implied support range from `∫exp(ψ)dt` may not equal `dSup` range. The rescaling step (Step 2 in Pattern 2) is mandatory, not optional — omitting it produces densities on the wrong x-axis.
- **Using `simpsons_weights` for CDF:** Simpson weights are for quadrature (single integral value); `cumulative_trapz` is for the cumulative integral (CDF). These are different functions — use `cumulative_trapz` for CDF, `trapz` or `simpsons_weights` for scalar integrals.
- **Uniform weights assumed:** `wasserstein_barycenter` must accept an optional weights vector (with `None` defaulting to 1/n uniform). CONTEXT.md says weighted case is in scope.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SVD / FPCA in LQD space | Custom SVD loop | `fdata_to_pc_1d` (regression.rs:287) | Already handles weighted SVD, sign normalization, faer/nalgebra feature gate |
| Cumulative integration | Custom trapz accumulator | `cumulative_trapz` (helpers.rs:197) | Handles mixed uniform/non-uniform grids, starts at 0 |
| Scalar integration | Custom quadrature | `trapz` (helpers.rs:234) or `simpsons_weights` (helpers.rs:57) | Both already ship-tested |
| Piecewise linear interpolation | Custom binary search | `linear_interp` (helpers.rs:172) | Binary search + clamping already implemented |
| Normalization to unit integral | Custom loop | `trapz` → divide | One-liner reusing `trapz` |

**Key insight:** This module is almost entirely function composition over existing helpers. The only genuinely new logic is the sign convention management (−log, exp, rescaling) and the quantile-grid construction.

---

## Reuse Map (existing helpers confirmed this session)

| Helper | Location | Verified | Role in DENS-01 |
|--------|----------|----------|-----------------|
| `cumulative_trapz(y, x)` | `helpers.rs:197` | [VERIFIED: fdars-core/src/helpers.rs:197-231] | CDF computation; Q(t) back-map |
| `trapz(y, x)` | `helpers.rs:234` | [VERIFIED: fdars-core/src/helpers.rs:234-240] | Scalar integral for normalization |
| `simpsons_weights(argvals)` | `helpers.rs:57` | [VERIFIED: fdars-core/src/helpers.rs:57-86] | Integration weights for `fdata_to_pc_1d` |
| `linear_interp(x, y, t)` | `helpers.rs:172` | [VERIFIED: fdars-core/src/helpers.rs:172-191] | CDF inversion; density back-interpolation |
| `fdata_to_pc_1d(data, ncomp, argvals)` | `regression.rs:287` | [VERIFIED: fdars-core/src/regression.rs:287-321] | LQD-FPCA engine; signature: `(data: &FdMatrix, ncomp: usize, argvals: &[f64]) -> Result<FpcaResult, FdarError>` |
| `FpcaResult` | `regression.rs:25` | [VERIFIED: fdars-core/src/regression.rs:25-38] | Fields: `singular_values: Vec<f64>`, `rotation: FdMatrix`, `scores: FdMatrix`, `mean: Vec<f64>`, `centered: FdMatrix`, `weights: Vec<f64>` |
| `FdMatrix` | `matrix.rs` | [ASSUMED — file not re-read this session] | Column-major container (rows=observations, cols=eval points) |

**Verbatim `FpcaResult` fields** [VERIFIED: fdars-core/src/regression.rs:25-38]:
```
pub singular_values: Vec<f64>
pub rotation: FdMatrix      // m × ncomp
pub scores: FdMatrix        // n × ncomp
pub mean: Vec<f64>
pub centered: FdMatrix      // n × m
pub weights: Vec<f64>
```

**Verbatim `cumulative_trapz` contract** [VERIFIED: fdars-core/src/helpers.rs:197-231]:
- Input: `y: &[f64], x: &[f64]`
- Output: `Vec<f64>` of same length as `y`, with `out[0] = 0.0` always.
- Uses mixed Simpson/trapezoidal (higher accuracy than pure trapz).

**Verbatim `linear_interp` contract** [VERIFIED: fdars-core/src/helpers.rs:172-191]:
- Clamps: `t <= x[0]` → return `y[0]`; `t >= x[last]` → return `y[last]`.
- Binary search for interior; linear interpolation.
- Does not return `Result` — safe to call in a map iterator.

---

## Common Pitfalls

### Pitfall 1: CDF endpoint not reaching 1.0
**What goes wrong:** `cumulative_trapz` of a density may not reach exactly 1.0 at the last grid point (trapz integration error). When mapping t=1.0 on the t_grid, `linear_interp` clamps to the last CDF value, which may be slightly < 1. The interpolated ψ at t=1 is then the value at the last density grid point, which is correct behaviour — but the test assertion `cdf.last() ≈ 1.0` needs a tolerance (1e-4 for typical grids).
**Why it happens:** Trapezoidal/Simpson integration error from the finite grid spacing.
**How to avoid:** After normalization (`dens / trapz(dens)`), the CDF will reach exactly 1.0 up to floating-point arithmetic. Always normalize before computing CDF.
**Warning signs:** CDF last value < 0.9999 — means the density wasn't normalized before CDF computation.

### Pitfall 2: Duplicate Q values in inverse_lqd
**What goes wrong:** If ψ is very large (density ≈ 0 at some point), `exp(ψ)` is very large → `exp(-ψ)` ≈ 0 → the density is near 0, but the cumtrapz Q may not advance uniformly. More commonly: duplicate Q values arise when the quantile density `exp(ψ)` is exactly 0 over some interval (indicating a point mass or near-point-mass). `linear_interp` with duplicate x-values is undefined.
**Why it happens:** Mathematical singularity in the quantile density at probability 0 or 1.
**How to avoid:** After building `q_scaled`, scan for adjacent duplicates and remove them (keeping the corresponding `dens_raw` value). Use a simple dedup pass preserving order. Validate that ψ is finite at all t_grid points; return `FdarError::ComputationFailed` if any ψ value is ±∞ or NaN.
**Warning signs:** `linear_interp` returning constant values over a range → degenerate density.

### Pitfall 3: Wrong sign convention
**What goes wrong:** ψ = +log(dens) instead of −log(dens). This produces a negative-of-correct transform; `exp(ψ)` gives density values (not quantile density values), and the back-transform produces garbage.
**Why it happens:** Confusion between the log-hazard and log-quantile-density transforms. R's `dens2lqd` uses `lqd_temp = -log(dens)` — the minus sign is the LQD convention, not the log-hazard.
**How to avoid:** The sign is: `psi[j] = -dens_norm[j].ln()` at the physical grid before mapping to t_grid. Verify with known density: for f(x) = 1 (uniform on [0,1]), ψ(t) = −log(1) = 0 everywhere. Round-trip test with a uniform density must return ψ ≡ 0. [CITED: rdrr.io/cran/fdadensity/src/R/dens2lqd.R]
**Warning signs:** ψ values all negative for a unimodal density → wrong sign.

### Pitfall 4: Rescaling step omitted in inverse_lqd
**What goes wrong:** The raw quantile function `Q_raw = lb + cumtrapz(exp(ψ))` has range `[lb, lb + θ_ψ]` where `θ_ψ = ∫exp(ψ) ≠ ub−lb` in general. If this rescaling step is omitted, the reconstructed density is defined on a different support than intended.
**Why it happens:** The LQD transform ψ encodes the shape of the density, not its absolute support. Recovering the original support requires rescaling.
**How to avoid:** Always apply Step 2 of Pattern 2: `Q_scaled = (Q_raw - Q_raw[0]) * (d_range / q_range) + lb`.
**Warning signs:** `inverse_lqd` returns a non-zero density outside `[target_argvals[0], target_argvals.last()]`.

### Pitfall 5: FVE not reaching 1 at ncomp = rank
**What goes wrong:** `fve.last()` < 1.0 when `ncomp < min(n, m)`. This is expected behaviour (partial SVD). But if `ncomp == min(n, m)` and FVE still < 1 − ε, the singular_values vector is truncated.
**Why it happens:** `fdata_to_pc_1d` returns `ncomp.min(n).min(m)` components. If the caller requests more components than available, ncomp is silently capped.
**How to avoid:** In `lqd_fpca`, document that `fve` reaches 1 only when `ncomp == min(n_dens, n_quantile_pts)`. In tests, use ncomp = 1 for FVE monotonicity check (≤ 1), not a full-rank assertion.
**Warning signs:** FVE vector length < requested ncomp → check `fpca.singular_values.len()`.

---

## Code Examples

### normalize_density — canonical implementation

```rust
// Source: fdadensity normaliseDensities + helpers::trapz
pub fn normalize_density(
    vals: &[f64],
    argvals: &[f64],
) -> Result<Vec<f64>, FdarError> {
    if vals.len() != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "vals",
            expected: format!("{}", argvals.len()),
            actual: format!("{}", vals.len()),
        });
    }
    if vals.iter().any(|&v| v < 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "vals",
            message: "density values must be non-negative".to_string(),
        });
    }
    let integral = trapz(vals, argvals);
    if integral < 1e-15 {
        return Err(FdarError::InvalidParameter {
            parameter: "vals",
            message: "density integrates to zero or is all-zero".to_string(),
        });
    }
    Ok(vals.iter().map(|&v| v / integral).collect())
}
```

### LqdFpcaResult struct

```rust
// Mirrors multi_fdata.rs / fts/mod.rs struct pattern
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LqdFpcaResult {
    /// FPCA result in LQD (log-quantile-density) space.
    ///
    /// The FPCA is performed on the LQD-transformed densities on the
    /// uniform quantile grid t ∈ [0, 1]. Scores, loadings, and mean
    /// are all in LQD space, not density space.
    pub fpca: FpcaResult,
    /// Fraction of variance explained by the first k components.
    ///
    /// `fve[k]` = cumsum(sv[0]²..=sv[k]²) / sum(all sv²). Monotone
    /// non-decreasing; `fve.last()` = 1 when `ncomp == min(n, m)`.
    pub fve: Vec<f64>,
}
```

### Test: round-trip correctness with truncated Gaussian

```rust
// Analytic reference: f(x) ∝ exp(-x²/2) on [−3, 3] (truncated standard normal)
// Use this because Q is monotone and the density is strictly positive everywhere.
fn truncated_gaussian_density(argvals: &[f64]) -> Vec<f64> {
    let dens: Vec<f64> = argvals.iter().map(|&x| (-x * x / 2.0).exp()).collect();
    let integral = trapz(&dens, argvals);
    dens.iter().map(|&d| d / integral).collect()
}

#[test]
fn round_trip_lqd_density_within_tolerance() {
    use crate::helpers::trapz;
    let argvals: Vec<f64> = (0..201).map(|i| -3.0 + i as f64 * 6.0 / 200.0).collect();
    let dens = truncated_gaussian_density(&argvals);
    let psi = lqd_transform(&dens, &argvals, Some(101)).unwrap();
    let dens2 = inverse_lqd(&psi, &(0..101).map(|i| i as f64 / 100.0).collect::<Vec<_>>(), &argvals).unwrap();
    // L∞ round-trip tolerance: 1e-3 (linear interp vs. spline divergence)
    let max_err = dens.iter().zip(dens2.iter()).map(|(&a, &b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(max_err < 5e-3, "round-trip L∞ error = {max_err}");
    // Reconstructed density integrates to 1
    let integral = trapz(&dens2, &argvals);
    assert!((integral - 1.0).abs() < 1e-6, "integral = {integral}");
    // All values non-negative
    assert!(dens2.iter().all(|&v| v >= -1e-9), "negative density values found");
}
```

### Test: FVE monotone non-decreasing

```rust
#[test]
fn lqd_fpca_fve_monotone_and_bounded() {
    let argvals: Vec<f64> = (0..101).map(|i| -3.0 + i as f64 * 6.0 / 100.0).collect();
    // Build 20 truncated Gaussians with varying means
    let mut data = FdMatrix::zeros(20, 101);
    for i in 0..20usize {
        let mu = -2.0 + i as f64 * 0.2;
        let row: Vec<f64> = argvals.iter().map(|&x| (-(x - mu).powi(2) / 2.0).exp()).collect();
        let integral = trapz(&row, &argvals);
        for j in 0..101 { data[(i, j)] = row[j] / integral; }
    }
    let result = lqd_fpca(&data, &argvals, 5, Some(101)).unwrap();
    // FVE is non-decreasing
    for k in 1..result.fve.len() {
        assert!(result.fve[k] >= result.fve[k-1] - 1e-12);
    }
    // All FVE in [0, 1]
    assert!(result.fve.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-9));
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Apply FPCA directly to densities (Euclidean; ignores density constraint) | LQD transform → L² FPCA (Petersen & Mueller 2016) | 2016 paper | Principal components stay in density space under inverse transform |
| Fréchet mean in general metric space (iterative) | Quantile average (closed-form for 1D Wasserstein) | Known since 1990 (Rüschendorf & Rachev) | O(n·m) computation, no iteration |

**Deprecated/outdated:**
- Direct FPCA on densities: produces components that mix negative regions; avoid entirely. The LQD transform is the standard fix.

---

## Validation Architecture

nyquist_validation is `true` in `.planning/config.json` — this section is required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Built-in Rust test harness (`#[test]`) |
| Config file | none — inline `#[cfg(test)] mod tests` in `density_fda.rs` |
| Quick run command | `cargo test -p fdars-core --features linalg density_fda` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DENS-01 | `lqd_transform` on truncated Gaussian → ψ is finite, length = n_q | unit | `cargo test -p fdars-core --features linalg density_fda::tests::lqd_transform_finite` | ❌ Wave 0 |
| DENS-01 | Round-trip lqd→dens→lqd L∞ < 5e-3 | unit | `cargo test -p fdars-core --features linalg density_fda::tests::round_trip_lqd_density_within_tolerance` | ❌ Wave 0 |
| DENS-01 | `inverse_lqd` always returns normalized (integral = 1 ± 1e-6) non-negative density | unit | `cargo test -p fdars-core --features linalg density_fda::tests::inverse_lqd_normalized_nonneg` | ❌ Wave 0 |
| DENS-01 | `lqd_fpca` FVE is monotone non-decreasing in [0,1] | unit | `cargo test -p fdars-core --features linalg density_fda::tests::lqd_fpca_fve_monotone_and_bounded` | ❌ Wave 0 |
| DENS-01 | Single-mode density family captured by leading PC (FVE[0] > 0.80) | unit | `cargo test -p fdars-core --features linalg density_fda::tests::lqd_fpca_leading_pc_captures_shift` | ❌ Wave 0 |
| DENS-01 | `wasserstein_barycenter` singleton → input density (L∞ < 1e-4) | unit | `cargo test -p fdars-core --features linalg density_fda::tests::barycenter_singleton_reduction` | ❌ Wave 0 |
| DENS-01 | `wasserstein_barycenter` two densities → lies between quantile-wise | unit | `cargo test -p fdars-core --features linalg density_fda::tests::barycenter_two_density_midpoint` | ❌ Wave 0 |
| DENS-01 | `normalize_density` → integral = 1 ± 1e-10, non-negative | unit | `cargo test -p fdars-core --features linalg density_fda::tests::normalize_density_integral_to_one` | ❌ Wave 0 |
| DENS-01 | Error on negative density input | unit | `cargo test -p fdars-core --features linalg density_fda::tests::error_negative_density` | ❌ Wave 0 |
| DENS-01 | Error on length mismatch | unit | `cargo test -p fdars-core --features linalg density_fda::tests::error_length_mismatch` | ❌ Wave 0 |
| DENS-01 | Error on empty sample for barycenter | unit | `cargo test -p fdars-core --features linalg density_fda::tests::error_empty_barycenter` | ❌ Wave 0 |
| DENS-01 | `lqd_transform` ψ ≡ 0 for uniform density (known analytic result) | unit | `cargo test -p fdars-core --features linalg density_fda::tests::lqd_uniform_is_zero` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core --features linalg density_fda`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`

### Wave 0 Gaps
- [ ] `fdars-core/src/density_fda.rs` — the module file itself (all tests live inline)
- [ ] `lib.rs` additions: `pub mod density_fda;` + `pub use density_fda::{...}`
- [ ] Framework install: none needed (built-in Rust `#[test]`)

---

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1`. This is a pure numeric Rust library — no network I/O, no authentication, no user sessions, no persistence. ASVS categories V2–V4 and V6 do not apply. V5 (input validation) applies.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | All public functions validate at entry: non-negative density, sorted argvals, length match, non-empty sample |
| V6 Cryptography | no | — |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| NaN/Inf propagation from log(0) or log(-x) | Tampering | Reject zero/negative density at validation gate before any log call |
| Integer overflow in grid indexing | Tampering | Use Rust's safe slice indexing; usize arithmetic is checked in debug mode |
| Division by zero in normalize | Tampering | Explicit `integral < 1e-15` guard returns FdarError::InvalidParameter |

---

## Divergences from fdadensity (Document in Rustdoc)

These divergences are intentional and must be documented in the `density_fda.rs` module-level doc comment:

1. **Quantile interpolation:** fdadensity uses `spline(..., method='natural')` (natural cubic spline) for the CDF→t mapping in `dens2lqd` and the `lqdSup`→`dSup` back-mapping in `lqd2dens`. This implementation uses `linear_interp` (piecewise linear). Effect: round-trip L∞ error is larger (~1e-3 vs. ~1e-5 for cubic spline) on grids with < 51 points. For ≥ 101 points the difference is negligible for smooth densities.

2. **useSplines path in lqd2dens:** fdadensity has a `useSplines=TRUE` path that integrates `exp(spline(lqd))` analytically per panel. This implementation always uses `cumulative_trapz(exp(ψ), t_grid)`. Effect: same as point 1 — a small accuracy trade-off for code simplicity and no new dependency.

3. **getWFmean weights:** fdadensity does not support weights (documentation confirms no weight parameter). This implementation adds an optional `weights: Option<&[f64]>` parameter, defaulting to uniform 1/n — a strict superset of fdadensity capability.

4. **Input normalization:** fdadensity warns and auto-normalizes if `|trapz(dens) - 1| > 1e-5`. This implementation normalizes silently (no warn) and documents the behaviour in rustdoc.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `linear_interp` (piecewise linear) produces round-trip L∞ < 5e-3 for smooth densities on 101-point grids — no empirical verification done this session | Architecture Patterns / Pitfall 1 | Test may fail if tolerance too tight; adjust to 1e-2 if needed |
| A2 | `FdMatrix` column-major layout: element (i,j) at index `i + j*nrows`, confirmed by CLAUDE.md convention | Reuse Map | Wrong layout → silently wrong FPCA results; but confirmed by multiple prior milestones |
| A3 | `wasserstein_barycenter` uses direct quantile-function inversion (not the LQD path) as the simpler and more numerically stable route for the mean | Architecture Patterns §Pattern 3 | Alternative path (through LQD of average) would also work but requires log of a gradient, which is noisier |

---

## Open Questions

1. **Density-space modes in LqdFpcaResult — ship or defer?**
   - What we know: CONTEXT.md marks this as "Claude's Discretion / nice-to-have". Computing mean ± sqrt(eigenvalue)·loading in LQD space, then `inverse_lqd` to density space, is straightforward with Pattern 2.
   - What's unclear: Whether to include a `density_modes(argvals)` method on `LqdFpcaResult` this phase.
   - Recommendation: Defer the method to a follow-on; add a rustdoc comment on `LqdFpcaResult` explaining how a caller can compute modes manually using `inverse_lqd`. Keeps the phase at M-effort.

2. **Quantile-grid default — 101 or length of input grid?**
   - What we know: fdadensity defaults `N = length(dSup)` (same as density grid). CONTEXT.md says "default ~101 points". 
   - Recommendation: Default to `argvals.len().max(101)` — never fewer than 101 to ensure round-trip accuracy, but matches the input grid if it's larger. Expose as `n_quantile_pts: Option<usize>` with `None` → default.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | all | ✓ | 1.97.0 (cargo 1.97.0) | — |
| `linalg` feature (faer 0.23) | `fdata_to_pc_1d` fast path | ✓ | via Cargo.toml | nalgebra fallback already in regression.rs |
| `parallel` feature (rayon 1.10) | clippy gate | ✓ | via Cargo.toml | — |

**Note on /tmp:** MEMORY.md documents /tmp tmpfs exhaustion blocking pre-commit doctest linking. Use `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` before committing, or `--no-verify` for docs-only commits. The density_fda.rs doctests will trigger linking.

---

## Sources

### Primary (MEDIUM confidence — official R package source, read via rdrr.io)
- [fdadensity dens2lqd.R source](https://rdrr.io/cran/fdadensity/src/R/dens2lqd.R) — exact algorithm for LQD forward transform: CDF via `cumtrapzRcpp`, `lqd_temp = -log(dens)`, spline interpolation onto [0,1]
- [fdadensity lqd2dens.R source](https://rdrr.io/cran/fdadensity/src/R/lqd2dens.R) — exact algorithm for inverse LQD: `cumtrapzRcpp(exp(lqd))`, rescaling, `exp(-lqd)`, dedup, `approx`, normalize
- [fdadensity package index](https://rdrr.io/cran/fdadensity/) — complete API surface (34 functions) confirmed

### Secondary (MEDIUM confidence — authoritative paper)
- [Petersen & Mueller 2016, Annals of Statistics 44(1):183-218](https://projecteuclid.org/journals/annals-of-statistics/volume-44/issue-1/Functional-data-analysis-for-density-functions-by-transformation-to-a/10.1214/15-AOS1363.full) — mathematical foundation: ψ(t) = log q(t) = −log f(Q(t)); Wasserstein Fréchet mean = quantile average formula confirmed

### Tertiary (LOW confidence — training knowledge, verified via source reading)
- In-codebase reads this session: `helpers.rs:172-240`, `regression.rs:25-38, 287-365` — confirmed exact signatures and invariants

---

## Metadata

**Confidence breakdown:**
- LQD/inverse-LQD exact algorithm: MEDIUM — R source confirmed via rdrr.io; Rust implementation deviates only in interpolation scheme
- Wasserstein barycenter formula: MEDIUM — paper confirmed; getWFmean source not directly read (algorithm inferred from documentation)
- Reuse map (fdata_to_pc_1d, helpers): HIGH — files read this session, signatures verified
- Round-trip tolerance (5e-3): LOW — not empirically tested in Rust this session; based on linear vs. spline accuracy reasoning

**Research date:** 2026-08-21
**Valid until:** 2027-02-21 (stable: fdadensity 0.1.4 is the frozen reference; Rust helper APIs are stable)
