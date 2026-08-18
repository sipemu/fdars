# Phase 26: PACE Sparse FPCA - Context

**Gathered:** 2026-08-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a **unified PACE FPCA estimator for sparse / irregularly-sampled functional data** as a
new public entry point in `fdars-core/src/pace_fpca.rs`, re-exported at the crate root. The
estimator produces, in one call: a kernel-smoothed mean, an eigendecomposition of the smoothed
covariance surface (eigenvalues + eigenfunctions), **conditional-expectation (BLUP/PACE) FPC
scores per curve**, and **fitted continuous trajectories with pointwise confidence bands**.
Additive/non-breaking: no existing public signature changes; no new crate dependency.

**Explicitly out of scope:** automatic measurement-error-variance (σ²) estimation, plotting, and
the sparse/PACE concurrent-regression variant (REG-01 sparse path — enabled by this phase but
deferred to a future milestone).

</domain>

<decisions>
## Implementation Decisions

### PACE Estimation Pipeline & Conventions
- **Eigenstructure:** obtained by a **symmetric eigendecomposition of the smoothed covariance
  surface** produced by `irreg_fdata::cov_irreg` on the work grid (Simpson-weighted inner
  product), taking the top-`ncomp` eigenpairs. Do NOT route through `fdata_to_pc_1d` — that
  SVDs a *data* matrix, whereas PACE eigendecomposes the *covariance surface*.
- **FPC scores:** **conditional-expectation (BLUP/PACE) scores**, newly implemented (no existing
  helper — the backlog's `spm::partial::conditional_expectation` reference is INACCURATE; that
  function does not exist). Formula per curve i:
  `ξ_ik = λ_k · φ_ik^T · Σ_yi^{-1} · (Y_i − μ_i)`, where `Σ_yi = Φ_i diag(λ) Φ_i^T + σ² I_{n_i}`
  evaluated on curve i's observed points (Φ_i = eigenfunctions interpolated to those points).
- **Measurement-error variance σ²:** **caller-supplied** (`sigma2` in the config, with a
  documented small default). Automatic σ² estimation (raw-vs-smoothed diagonal) is DEFERRED.
- **Confidence bands:** pointwise bands from the **BLUP prediction variance** (Yao et al. Ω:
  `Var(x̂_i(t)) = Φ(t) (diag(λ) − diag(λ)Φ_i^T Σ_yi^{-1} Φ_i diag(λ)) Φ(t)^T`), 95% Gaussian
  (`alpha` configurable).

### API Shape & Signature
- Entry point: **`pace_fpca(data: &IrregFdata, config: &PaceFpcaConfig) -> Result<PaceFpcaResult, FdarError>`**,
  crate-root re-exported.
- Input: **`&IrregFdata`** (the existing sparse/irregular container — `offsets` + `argvals` + `values` + `rangeval`).
- Work grid: **caller-supplied `work_grid: &[f64]`** in the config (the grid on which mean,
  covariance, eigenfunctions, and fitted trajectories are represented).
- Params: **`PaceFpcaConfig` builder struct** (`ncomp`, `bandwidth`, `sigma2`, `work_grid`,
  `alpha` for band level) — mirrors project convention (`GmmClusterConfig`, `StlConfig`,
  `ElasticConfig`) for methods with ≥4 knobs. Derive `Debug, Clone, PartialEq`; conditional serde.

### Result Struct
- New **`PaceFpcaResult`** (`FpcaResult` cannot carry BLUP scores, eigenvalues-as-such, or bands).
- Fields: **`{ mean (Vec, len m), eigenvalues (Vec, len ncomp), eigenfunctions (FdMatrix m×ncomp),
  scores (FdMatrix n×ncomp, conditional-expectation), fitted (FdMatrix n×m), fitted_lower
  (FdMatrix n×m), fitted_upper (FdMatrix n×m), argvals (work grid), sigma2 (echoed), ncomp }`**.
- Bands as **separate `fitted_lower` / `fitted_upper` FdMatrix** (explicit, directly consumable).
- Derive `Debug, Clone, PartialEq`; `#[non_exhaustive]`; conditional serde. `#[must_use]` on `pace_fpca`.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets (VERIFIED present)
- `irreg_fdata::IrregFdata` (`irreg_fdata/mod.rs:38`) — sparse container: `offsets` (len n+1),
  `argvals`, `values` (concatenated), `rangeval [min,max]`.
- `irreg_fdata::mean_irreg` (`kernels.rs:58`) — kernel-smoothed mean on a grid.
- `irreg_fdata::cov_irreg(ifd, s_grid, t_grid, bandwidth) -> FdMatrix` (`smoothing.rs:111`) —
  smoothed covariance surface. **Core reuse for the eigendecomposition.**
- `irreg_fdata::to_regular_grid(ifd, target_grid) -> FdMatrix` (`smoothing.rs:285`) — per-curve
  regularization (may help build Φ_i / interpolate).
- `helpers::simpsons_weights` (integration weights for the functional inner product / eigen-normalization).
- `nalgebra` symmetric eigendecomposition (`SymmetricEigen`) for the covariance surface — same
  linalg path the crate already uses for SVD.

### MUST implement (NOT reuse — verified absent)
- The **conditional-expectation (BLUP) score** solve and the **prediction-variance bands** — no
  `conditional_expectation` / `blup` / `pace` symbol exists anywhere in `src/`. This is the real
  algorithmic core of the phase; treat "orchestration" claims in the backlog with that caveat.

### Established Patterns
- All public fns return `Result<T, FdarError>`; dimension/parameter checks at entry
  (`InvalidDimension`, `InvalidParameter`).
- Config structs are builder-style with serde (`GmmClusterConfig`, `StlConfig`, `ElasticConfig`).
- Public result structs derive `Debug, Clone, PartialEq`, `#[non_exhaustive]`, conditional serde.
- Inline `#[cfg(test)] mod tests`; crate-root re-export in `src/lib.rs` (`pub mod` + `pub use`).

</code_context>

<specifics>
## Specific Ideas

- Recovery test: synthetic sparse data from a KNOWN generative model (known mean + eigenfunctions
  φ_k + eigenvalues λ_k + score distribution + per-curve random sampling density + σ² noise) →
  recovered eigenstructure and per-curve BLUP scores/trajectories match ground truth within a
  documented tolerance (sign/rotation-aligned comparison of eigenfunctions).
- Consistency test: `fitted` lies within `[fitted_lower, fitted_upper]`; bands widen where data is
  sparse. Reproducibility: same inputs → identical output (deterministic).
- Error-path tests: empty `IrregFdata`, a curve with too few points, `ncomp` too large,
  non-positive `bandwidth`/`sigma2`, mismatched `work_grid` → appropriate `FdarError`, no panic.
- R baseline matched by capability: `fdapace::FPCA` (Yao–Müller–Wang PACE). Document the
  conditional-expectation formulation + band definition in rustdoc (as prior milestones documented
  divergences from R baselines).

</specifics>

<deferred>
## Deferred Ideas

- Automatic σ² (measurement-error variance) estimation from the raw-vs-smoothed diagonal.
- GCV/CV bandwidth selection for the covariance surface (use caller-supplied bandwidth).
- The REG-01 sparse/PACE concurrent-regression variant (enabled by this phase, deferred).
- Functional-fragment completion / trajectory extrapolation beyond the observed range.

</deferred>
