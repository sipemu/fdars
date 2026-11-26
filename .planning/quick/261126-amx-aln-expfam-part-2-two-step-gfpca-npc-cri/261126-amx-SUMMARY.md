---
status: complete
quick_id: 261126-amx
issue: 60
---
# Summary — ALN-EXPFAM part 2 (GH #60): two-step GFPCA, npc_criterion, periodic, piecewise_linear2

Completes GH #60 (part 1: quick 261126-8kh, `00fe89fc`).

## Commits
- `004c7141` feat(quick-261126-amx): two-step GFPCA, npc_criterion, periodic templates and
  piecewise-linear warps for register_fpca (closes #60)

## Public API added (additive; all defaults reproduce part 1)
- `gfpca_two_step(data, family, &GfpcaConfig) -> GfpcaResult` (registr `gfpca_twoStep`).
- `FpcaType { VariationalEm (default), TwoStep }`, `RegisterFpcaConfig::fpca_type`.
- `NpcCriterion { var_explained, min_component_share }` (`new`, `with_min_share`),
  `GfpcaConfig::npc_criterion`.
- `GfpcaConfig::periodic` (pbs basis), `GfpcaConfig::index_significant_digits` (default `Some(4)`).
- `WarpingType { Nonparametric (default), PiecewiseLinear2 }`, `RegisterFpcaConfig::{warping, prior_sd}`.
- `GfpcaResult::{fpca_type, evalues_sum, variance_components, yhat}`.
- Re-exported from `alignment` and the crate root.

## Implementation (new files `mgcv.rs`, `glmm.rs`, `twostep.rs`, `pl2.rs`)
- `cov_hall` natively: mgcv `ps` basis (0.1 %-widened evenly spaced knots, 2nd-order difference
  penalty), PIRLS with GCV (gaussian, Tweedie p=2 ≡ gamma) / UBRE (binomial, poisson) smoothing
  selection; registr's regular / irregular cross products; `te(bs = "ps", k = 10)` with mgcv's
  `np = TRUE` margin reparameterisation and two-parameter GCV; registr's latent transform
  (binomial `1/μ'(η)`, gamma/poisson `η` as registr); 0.01 diagonal floor; `Matrix::nearPD`
  (Higham/Dykstra + `do2eigen`) port.
- GLMM replacing gamm4: mgcv sum-to-zero constraint + `smooth2random` fixed/random split of the
  P-spline, block (Schur-complement) penalised IRLS (Newton weights for the gamma log link),
  lme4 profiled REML (gaussian) and two-stage Laplace with lme4's `aic()` (`nAGQ = 0` then
  `(θ, β)` jointly), L-BFGS with central finite-difference gradients, warm-started modes.
- `register_fpca(two-step)`: re-registration to per-curve LS (gamma: Gamma-log GLM) spline fits
  of the latent `Yhat` on the link scale; variance parameters warm-start the next GFPCA (registr
  `start_params`).
- npc_criterion: `determine_npc` (≥ first element, `< second` element bound) for two-step;
  variational EM: `Kt = 20`, npc = 20 pre-run, strict `>` rule, evalues/evalues_sum from the
  pre-run, Kt = 20 for the registration templates too (registr `register_fpca`).
- periodic: `pbs::pbs` port (`getPeriodBsplineKnots`, wrapped columns), `Kt − 1` quantile knots,
  analytic derivatives (R falls back to Nelder–Mead).
- piecewise_linear2: registr polygon on the per-curve scaled domain, exact parameterisation of
  the constraint triangles (softmax), analytic gradients, L-BFGS + Nelder–Mead polish, coarse
  knot-grid multistart on the initial pass; normal priors (binomial only, as R).

## Validation (`alignment/expfam/tests_twostep.rs`, 17 new tests + 1 doctest)
R goldens from `.planning/quick/261126-8kh-.../golden_twostep.R` (registr 2.2.1, mgcv 1.9-4,
gamm4 0.3.0, lme4 2.0.1, Matrix 1.7-5, pbs) → `golden_twostep.rs`:
- `coarsen_index` = R on hand cases.
- cov_hall (4 families): rel. error ≤ 4e-6 (observed 2e-7 gaussian … 4e-6 binomial).
- gfpca_twoStep (4 families): eigenvectors |cos| > 1 − 1e-8, eigenvalues / sum ≤ 2e-5;
  gaussian mean & scores ~5e-8, θ ~1e-6; binomial / poisson / gamma mean & scores ≤ 1.9e-4,
  θ ≤ 1e-4 (lme4 Nelder–Mead stage-2 tolerance).
- two-step on R's registered data (sparse irregular grids): binomial subspace 1 − 1e-6, evalues
  8e-4, α 6e-4; poisson 1 − 1e-13, 1e-6, 7e-6.
- register_fpca(two-step), 3 significant digits: gaussian |Δβ| 1.8e-6, losses 3e-6, subspace
  1 − 7e-9, iterations equal; binomial / poisson iterations equal, losses after the first step
  ≤ R's (−0.06 %/−0.24 %, −0.011 %), |Δβ| < 0.05 (inherited from the initial registration where
  R uses Nelder–Mead / barrier; see part 1).
- npc: hand cases of both rules; two-step `c(0.95, 0.1)` → 2 (second element binds) and `0.95`
  → 3 as R; fpca_gauss(npc_varExplained = 0.9) with R's 20×20 start: npc 3, evalues 1e-6, μ 1e-6,
  efunctions |cos| > 1 − 1e-8; register_fpca(npc_criterion = 0.9): npc 1, Kt = 20, |Δβ| < 0.05,
  final loss within 0.5 %.
- periodic: pbs values 1e-12, derivatives vs FD, periodicity at the ends; registr periodic loss
  ≤ R's, |Δβ| < 0.02; fpca_gauss(periodic) from R's start on R's registered data: knots 1e-12,
  μ 1e-7, evalues 1e-6, |cos| > 1 − 1e-10; register_fpca periodic loss within 1 %, |Δβ| < 0.05.
- piecewise_linear2: hinv = registr on hand cases; FD gradients (4 families, prior, gamma scale);
  fdars' loss at R's β = R's loss to 1e-8 (gaussian, binomial, binomial + priors); our losses
  ≤ R's (gaussian 36.06 vs 36.78, binomial 1152.08 vs 1158.59, priors 990.8307 vs 990.8310 with
  |Δβ| 0.011); feasibility, monotone t̂ with fixed ends; validation errors; register_fpca runs.
- Invariants: orthonormal eigenvectors, `yhat = α + sφᵀ`, `fitted`, LS normal equations of the
  subject coefficients, recovery of two known components (subspace > 0.995).

## Gates
- `cargo fmt --all` clean.
- `cargo test -p fdars-core --lib expfam`: 34/34 (17 part 1 unchanged + 17 new), ~55 s.
- `cargo test -p fdars-core --doc expfam`: 5/5; `--lib hygiene` 3/3.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`: clean.
- `cargo check -p fdars-core --features serde` and `--no-default-features`: clean.
- pmat: max cognitive complexity in `alignment/expfam` 21 (≤ 23).
- `RUSTDOCFLAGS=-D warnings cargo doc` fails crate-wide on pre-existing `[module documentation](self)`
  links to private modules (affine.rs etc.); expfam uses the same pattern (incl. one new link in
  twostep.rs) — not a gate.

## Deviations (documented in the module docs)
- No mgcv / gamm4: same GCV/UBRE, REML and Laplace criteria, different optimisers (grid +
  golden-section / L-BFGS, FD gradients); smooth variance component in fdars' penalty scaling.
- Two-step templates fit on grid points clamped to `[t_min, t_max]`; final `Yhat` not truncated.
- `periodic` also sets the two-step knots (R keeps non-periodic knots there).
- Piecewise-linear warps evaluated at the scaled observation times (R: equally spaced grid;
  identical for equally spaced complete curves); multistart on the initial pass.
- `coarsen_index`: integer-digit count without R's scientific-notation quirk (≥ 1e5).
- Not ported: > 1e7-row subsampling of the template GLM (memory guard, not needed).

## Remaining
Nothing for #60.
