---
status: complete
quick_id: 261126-8kh
issue: 60
---
# Summary — ALN-EXPFAM (GH #60): exponential-family registration with generalized FPCA

## Commits
- `00fe89fc` feat(quick-261126-8kh): exponential-family registration with variational-EM GFPCA (registr register_fpca) (refs #60)

## Public API added (module `alignment::expfam`, re-exported from `alignment` and the crate root)
- `ExpFamily { Gaussian (default), Binomial, Gamma, Poisson }`, `CurveIncompleteness { Leading, Trailing, Full }`.
- `RegisterFpcaConfig` (`#[non_exhaustive]`, serde; registr defaults: kh 4, max_iterations 10,
  lambda_inc 0, `fpca: GfpcaConfig`, warp_max_iter 500, warp_tol 1e-12, gamma_scale_init 1.5).
- `register_fpca(data: &IrregFdata, template: Option<&IrregFdata>, &cfg) -> RegisterFpcaResult`
  (registration + fpca + converged/iterations/delta_index/registration_loss, `registered_data()`).
- `register_fpca_regular(&FdMatrix, argvals, &cfg)`.
- `register_expfam(data, template, &cfg) -> ExpFamRegistration` (one registr pass to the GLM mean:
  t_hat, hinv_beta, hinv_inner_knots, curve_loss, loss, gamma_scales, template knots/coefs).
- `gfpca_variational(data, family, &GfpcaConfig) -> GfpcaResult` (registr `fpca_gauss` / `bfpca`;
  `GfpcaConfig`: npc 1, kt 8, max_iter 50, error_thresh 1e-4, seed 1988, psi_init, xi_init).
- `evaluate_inverse_warp(beta, inner_knots, t_lo, t_hi, t)`.

## Algorithm (mirrors registr 2.2.1 source)
- Template basis = `bs(knots = quantile(t, seq(0,1,len=Kt-2))[-ends], intercept=TRUE)` on the pooled
  range (type-7 quantiles); initial template = IRLS GLM (identity / logit / log / Gamma-log).
- Inverse warp t̂ = Θ_h(t*)β, cubic B-splines on each curve's own range, Kh−4 quantile inner knots,
  start = Greville abscissae (= registr's `ginv(Θ_h) t*`).
- Loss = registr `loss_h` (gaussian ½RSS, binomial, poisson incl. log y!, gamma with scale) + `n_i·λ_inc·pen`.
- Optimiser: exact reparameterisation β_l = A + S Σ_{k<l} softmax(0,u)_k, A/S through logistic maps
  for incomplete modes, gamma scale on log scale; L-BFGS (in-crate two-loop from soft-DTW) with
  Armijo backtracking, max-norm step cap 1, warm starts moved 1e-3 inside the boundary (vanishing
  gradients of the softmax at ties).
- GFPCA: exact port of `fpca_gauss_optimization` / `bfpca_optimization` (Kronecker M-step,
  Jaakkola–Jordan ξ, SVD of the evaluated FPC basis on a 100-grid, orthogonalised scores, subject
  coefs); gamma/poisson use the gaussian EM and the `*-varEM` mean-scale loss as R's default.
- Outer loop: until mean((Δt̂/t_max)²) ≤ 1e-4 or max_iterations, then final GFPCA.

## Validation (`alignment/expfam/tests.rs`, 17 tests + 4 doctests)
R registr 2.2.1 installed into `~/.cache/R-lib` (stats::glm, fastglm not installed). Goldens from
`golden_registr.R` → `alignment/expfam/golden.rs`, data from a portable LCG generated identically in
R and Rust (`sim`, mimicking `simulate_unregistered_curves`: cubic Bernstein warps, same mean curve).
- Data reproduction: sums equal to 1e-9 for all 7 golden data sets.
- Template GLM (converged MLE, R `epsilon=1e-14`): coefficients ≤ 1e-7 for all four families.
- `loss_h` exactness: fdars' loss at R's β (with R's template coefs, and R's `bs` boundary-stretch
  quirk for slightly infeasible β) equals R's total loss to 1e-10 relative for gaussian, binomial,
  poisson, gamma (with R's scales), trailing, leading, full.
- Our initial registration: total loss ≤ R's·(1+1e-6); on every curve where R's β is feasible,
  our curve loss ≤ R's (+1e-6 rel., R's glm tolerance) and |Δβ| < 1e-3 (gaussian/binomial/incomplete),
  1e-2 (poisson/gamma; R uses Nelder–Mead); R-infeasible curves (R's constrOptim allows 1e-6
  violations of monotonicity / bounds) |Δβ| < 0.05.
- `fpca_gauss` (npc 2, R's psi start) and `bfpca` (R's psi start, ξ = 1): knots 1e-12, mean /
  subject coefs / alpha 1e-7, evalues 1e-6 rel, efunctions |cos| > 1 − 1e-10, iteration count equal.
- Full `register_fpca` vs R (different EM starts): observed max |Δβ| gaussian 1e-4, binomial 0.015,
  poisson 7e-4, trailing 0.011; R's own seed sensitivity (seeds 1, 7, 42;
  `registr_seed_sensitivity.R`) is 0.046 / 0.010 / 2e-4 / 3e-3. Mean within 0.005, FPC subspace
  cos > 0.9999, iteration counts equal; final registration losses ≤ R's (binomial +0.001 %, poisson
  −0.001 %) or within 0.2 % (gaussian, trailing).
- Invariants: finite-difference gradients for all families × link/mean template scale × 4
  incompleteness modes (incl. gamma scale); parameterisation monotone with exact fixed endpoints;
  digamma closed forms (1e-13); recovery of known warps (centred RMSE ×0.41 binomial n=80,
  ×0.33 poisson n=40, and ×<0.6 raw on irregular 60 % subsampled grids); identity warps on identical
  curves; λ_inc = 1e4 pins endpoints (< 1e-3) while λ_inc = 0 frees them; FdMatrix front end ==
  IrregFdata; template = data reproduces the default; validation errors.

## Gates
- `cargo fmt --all` clean.
- `cargo test -p fdars-core --lib expfam` 17/17; `--doc expfam` 4/4; `--lib hygiene` 3/3.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
- `cargo check -p fdars-core --features serde` and `--no-default-features` clean.
- pmat: no function in `alignment/expfam` above cognitive 23.

## Deviations (documented in the module docs)
- Optimiser (reparameterisation + L-BFGS, analytic gradients for all families) instead of
  constrOptim; strictly monotone warps (R allows ties / 1e-6 violations).
- Rust RNG for the EM start (`psi_init` / `xi_init` reproduce R's start).
- IRLS converged to 1e-12 relative deviance (R glm: 1e-8).
- `ln Γ` instead of `log(gamma(·))` in the gamma loss.

## Not done (remaining for #60) — all done in quick 261126-amx (`004c7141`, closes #60)
- `fpca_type = "two-step"` (`gfpca_twoStep`: Hall covariance binning + gamm4 GLMM) — the only
  route R offers with a family-specific (non-gaussian) GFPCA for gamma/poisson.
- `npc_criterion` (variance-explained choice of npc with the Kt = 20 override).
- `periodic = TRUE` (pbs basis), `warping = "piecewise_linear2"` with `priors` / `prior_sd`.
- > 10⁷-row subsampling of the template GLM (irrelevant for in-memory sizes).
