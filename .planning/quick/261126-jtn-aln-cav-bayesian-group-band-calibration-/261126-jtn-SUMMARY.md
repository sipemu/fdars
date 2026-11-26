---
quick_id: 261126-jtn
issue: 64
backlog_id: ALN-CAV
status: complete
issue_fully_done: true
---
# ALN-CAV (#64) summary: Bayesian group alignment credible-band calibration

## Commits
- `51a9a762` fix(quick-261126-jtn): exact template centring + in-model calibration study for
  Bayesian group alignment bands (`closes #64`)

## Findings
1. **In-model study** (`bayesian_group_calibration_tests.rs`). Exact in-model data: template q* in
   the 12-dim B-spline span, warps in the 1-harmonic warp span (pre-centred so the truth is a
   fixed point of the sampler's centring), curve i's SRSF observed at abscissae γ_i(t_j) with values
   (q*_j + ε_ij)/ψ_ij, so the sampler's likelihood `interp(q_i, γ_i(t_j))ψ_ij = q*_j + ε_ij` holds
   exactly at the truth (verified to 1e-12 noise-free). Data on the sampling grid itself cannot be
   exactly in-model: the collocation matrix is singular unless |γ(t_j) − t_j| < h
   (Schoenberg–Whitney) and ill-conditioned near it (first attempt: residual sd 6800).
   Needed an internal `Problem::q_grid: Option<Vec<Vec<f64>>>` (None in production; HMC slope uses
   the old uniform-index formula when None → bit-identical).
   Coverage of 95% pointwise bands, 5 curves × 31 points, 40 data sets × 4 chains, mean ± MC SE,
   σ² prior IG(1e-6,1e-6):
   | sampler | σ | warps | template (fixed) | template pre-fix | σ̂²/σ² |
   |---|---|---|---|---|---|
   | infHMC 8000 it | 0.01 | 0.964±0.008 | 0.945±0.008 | 0.899±0.012 | 1.04 |
   | infHMC 8000 it | 0.1 | 0.945±0.011 | 0.915±0.011 | 0.827±0.014 | 0.90 |
   | pCN injective 40000 it | 0.1 | 0.923±0.014 | 0.924±0.010 | 0.909±0.012 | 1.00 |
   | infHMC 8000 it (20 sets) | 0.3 | 0.926±0.023 | 0.909±0.016 | 0.847±0.021 | 0.74 |
   → samplers calibrated; the original 0.83–0.87 warp coverage (f-space noise) is misfit.
   Raw draws vs raw truth with center_every = 0 (exact target): warp 0.951±0.015, template
   0.946±0.015.
2. **Bug (fixed)**: `realise` (center_draws) warped the linear interpolant of the template grid
   values (`warp_srsf`); chord error up to 0.12 vs 4.6e-4 exact (unit test), biasing template draws.
   Now `Problem::warp_template` evaluates the B-spline at γ_I(t). Draws store `tmpl_coef`.
3. **center_draws** on/off: centring does not remove valid variability; raw bands over-cover
   (warps 0.96–0.985) because they include the non-identified common phase.
4. **Pseudo-likelihood**: interpolating noisy SRSF nodes at warped points averages noise, so warps
   can reduce SSE → σ² under-estimated (0.90× at σ=0.1, 0.74× at 0.3), mild under-coverage
   growing with noise.
5. **σ² prior floor**: fdasrsf default β0 = 0.1 adds ≈ 2β0/(nT) (≈0.0013 for nT=155) to σ²; at
   small noise this dominated (σ̂²/σ² = 16 at σ=0.01) → over-coverage. Documented.
6. Without in-chain centring (center_every = 0) the non-identified ridge mixes slowly (R̂ up to 1.6
   at σ=0.1); with centring every 100 its R̂ ≤ 1.02.
7. Burn-in/thin/pooling/quantile code reviewed: no bugs.

## Not done (deliberate)
- Function-space likelihood option: not added (pCN only, nonlinear in the template, needs inverse
  warps per proposal; not needed since the samplers are calibrated). Misfit documented instead.
- In-chain `center` still warps the template interpolant then re-projects (left unchanged to keep
  chains bit-identical; HMC re-draws the template from its full conditional right after).

## Behaviour change
Default (`center_draws = true`) template summaries only. `warp_hash` (all template-independent
fields) and the full `center_draws = false` hash equal the pre-change values computed on c038fd46;
full default hash moved 0x17c8a7166fdb441d → 0x423ab6bafaaecb16. CHANGELOG "Fixed" (both files).

## Tests
- New: `in_model_simulation_satisfies_the_likelihood_exactly`, `centred_template_uses_the_exact_spline`,
  `credible_bands_calibrated_on_in_model_data` (short: warp 0.959 > 0.92, template 0.897 > 0.86),
  `#[ignore]` `credible_bands_calibrated_on_in_model_data_long_study` (~7 min release, asserts the table).
- Tightened: f-noise short study > 0.85 (measured 0.874), long > 0.80 (measured 0.829).
- Hash test now pins warp hash + raw hash (pre-change) + new full hash.

## Gates
- `cargo fmt --all` clean; `cargo test -p fdars-core --lib bayesian` 39 passed, 3 ignored;
  `alignment::hygiene` 3 passed; `--doc bayesian` 2 passed; long studies run in release (pass);
  `cargo clippy -p fdars-core --all-targets --features linalg,parallel -D warnings` clean;
  `cargo check --no-default-features` clean. No serde types added.
