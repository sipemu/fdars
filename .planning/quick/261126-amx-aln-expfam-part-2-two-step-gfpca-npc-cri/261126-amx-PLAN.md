# Plan — ALN-EXPFAM part 2 (GH #60): two-step GFPCA, npc_criterion, periodic, piecewise_linear2

Continues quick 261126-8kh (commit 00fe89fc, variational-EM route).

## Goal
Finish the registr 2.2.1 `register_fpca` port: `fpca_type = "two-step"` (`gfpca_twoStep`),
`npc_criterion`, `periodic = TRUE`, `warping = "piecewise_linear2"` (+ priors). Skip the
>1e7-row template-GLM subsampling (memory guard, not needed).

## Public API (additive; defaults unchanged)
- `gfpca_two_step(data, family, &GfpcaConfig) -> GfpcaResult`
- `FpcaType { VariationalEm (default), TwoStep }` → `RegisterFpcaConfig::fpca_type`
- `NpcCriterion { var_explained, min_component_share }` → `GfpcaConfig::npc_criterion`
- `GfpcaConfig::periodic`, `GfpcaConfig::index_significant_digits` (Some(4))
- `WarpingType { Nonparametric (default), PiecewiseLinear2 }`, `RegisterFpcaConfig::{warping, prior_sd}`
- `GfpcaResult::{fpca_type, evalues_sum, variance_components, yhat}`

## Algorithm (from registr / mgcv / lme4 source)
1. `coarsen_index` → common grid; `cov_hall`: mgcv `ps` P-spline GLM mean (GCV / UBRE),
   centred cross products (regular / irregular branch), `te(i1, i2, k = 10, bs = "ps")` with
   mgcv's `np` reparameterisation and GCV, latent transform, diagonal floor, `nearPD`.
2. Eigen-decomposition, positive filter, `determine_npc`.
3. GLMM `value ~ s(index, ps, Kt) + (0 + psi_k || id)`: mgcv sum-to-zero constraint +
   `smooth2random` split, block (Schur) penalised IRLS, lme4 REML (gaussian) / two-stage
   Laplace with `aic()` (other families), L-BFGS on θ (and β in stage 2).
4. Registration step with per-curve spline fits of `Yhat` (gamma: Gamma-log GLM).
5. npc_criterion (variational: Kt = 20 pre-run with npc = 20); pbs basis; pl2 polygon warps
   with an exact triangle parameterisation, L-BFGS + Nelder–Mead, grid multistart.

## Validation
R goldens (`golden_twostep.R`): cov_hall, gfpca_twoStep for 4 families, register_fpca two-step
(3 families), npc choice (both elements), fpca_gauss npc_varExplained, register_fpca with
npc_criterion, pbs basis, registr / fpca_gauss / register_fpca periodic, registr pl2 (with and
without priors). Invariants: orthonormality, yhat identity, LS normal equations, known-component
recovery, finite-difference gradients (pl2 incl. priors / gamma scale), constraint feasibility.

## Tasks
1. basis: PeriodicBs + TemplateBasis; warp.rs uses TemplateBasis.
2. gfpca: FpcaType, NpcCriterion, periodic, npc-criterion EM.
3. mgcv.rs, glmm.rs, twostep.rs (two-step GFPCA).
4. pl2.rs + mod.rs integration (fpca dispatch, Kt = 20, scaled domain, priors).
5. Goldens + tests; docs; CHANGELOG; gates.
