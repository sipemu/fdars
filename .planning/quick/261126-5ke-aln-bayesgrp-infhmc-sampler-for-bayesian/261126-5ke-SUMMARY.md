---
quick_id: 261126-5ke
issue: 52
backlog_id: ALN-BAYESGRP
status: complete
issue_fully_done: true
---
# ALN-BAYESGRP (GH #52) part 2: infHMC sampler, ESS + multimodality diagnostics

Completes #52 on top of the pCN implementation (065279f0, quick 261126-4r5).

## Commits
- `735f5e3e` feat(quick-261126-5ke): infHMC sampler for Bayesian group alignment, ESS and
  multimodality diagnostics (`closes #52`)

## Public API (additive; `BayesianSampler` re-exported from `alignment` and crate root)
- `enum BayesianSampler { Pcn (default), InfHmc }` (`#[non_exhaustive]`).
- `BayesianGroupConfig`: `sampler`, `hmc_step_size` (0.1 = fdasrsf √h), `hmc_max_leapfrog` (10),
  `hmc_adapt` (true), `hmc_target_accept` (0.65), `injective_warps` (false; forced on for infHMC).
- `BayesianGroupResult`: `ess_sigma2`, `ess_log_likelihood`, `ess_warps` (Geyer initial monotone
  sequence, summed over chains), `warp_bimodality` (Sarle coefficient of pooled per-curve warp-area
  draws), `multimodal_curves` (BC > 5/9).
- `BayesianGroupChain::warp_step_sizes` (adapted ε per curve; empty for pCN).

## Algorithm (fdasrsf 2.7.5 source read: `bayesian_functions.infhmc_step(_batch)`,
`dual_avg_step`, `bayes_warp_geom`, `time_warping.time_warping_bayes_infHMC`)
- Per curve, target `exp(−SSE_i/2σ²) N(c; 0, C) 1{valid}`, `C` = warp prior covariance
  (preconditioner); momentum `N(0, C)`; half kick `v −= ε/2 C∇Φ`, exact rotation
  `(c, v) ← (c cos ε + v sin ε, v cos ε − c sin ε)`, half kick; exact-H MH; trajectory leaving
  the support rejected; steps uniform on `1..=L`.
- Dual averaging per curve during burn-in (t0 = 10, γ = 0.05, κ = 0.75, μ = log 10ε₀, ε ≤ π/2),
  frozen at exp(log ε̄) at the end of burn-in — as fdasrsf.
- Exact reverse-mode gradient of SSE through shooting map → trapezoidal cumulative integral →
  normalisation → linear interpolant of q_i (new file `alignment/bayesian_group_hmc.rs`).
- Template: exact conjugate Gaussian Gibbs draw (as fdasrsf's infHMC), with this module's
  template prior so both samplers target the same posterior; σ² Gibbs; centring schedule shared.
- Curves updated sequentially, each with own leapfrog count (fdasrsf batches curves with one
  shared count — both valid).

## Key finding: shooting-map periodicity (→ `injective_warps`)
`exp_1(v)` is 2π-periodic in ‖v‖, so a warp has preimages at ‖v‖ = r and 2π − r (antipodal),
… The pCN model (fdasrsf `time_warping_bayes`, prior sd 4) puts *most* coefficient-posterior mass
on the ‖v‖ ≈ 2π shell (larger ring area), and pCN's β = 0.5 jumps visit it; local HMC cannot.
The induced γ-posterior differs slightly between shells (measured: curve-2 warp-area mean
−0.09318 untruncated pCN vs −0.09326 infHMC, ≈ 4 SE). fdasrsf's infHMC restricts to ‖v‖ < π;
we do the same (always for infHMC, opt-in `injective_warps` for pCN, default off → pCN unchanged).
With it, a 2·10⁶-iteration pCN gives −0.093272 ± 1.5e-5, matching infHMC.

## Validation (12 new tests + 1 new `#[ignore]` long study + doctest)
- **Bit-identity**: FNV hash over every pre-existing result field of a default (pCN) run pinned
  at `0x17c8a7166fdb441d`, computed on 065279f0 before any change.
- **Gradient vs central FD** (h = 1e-6), 3 curves × 4 coefficient vectors × 6 coefficients,
  relative 1e-6 (c = 0 itself skipped: γ = id puts every γ_j on a knot of the piecewise-linear
  q, a kink).
- **Dual averaging** = fdasrsf 2.7.5 `dual_avg_step` literals (1e-12).
- **Kernel invariance vs exact quadrature**: one curve, fixed template/σ² = 0.5, 1 harmonic;
  20 000 HMC draws vs 401² quadrature: means within 1 SE, variances within 4 rel-SE
  (measured 1.475e-4 vs 1.425e-4; with 4 × 400 000 draws in release: ratios 0.997–1.008).
- **Full-sampler invariance** (`#[ignore]`, ~3 s release): 2 curves, no centring, infHMC 4×5·10⁴
  vs injective pCN 4×10⁶ (thin 20): σ² 0.25355 vs 0.25296 (SE 5e-4), warp areas 0.081351 vs
  0.081356 (SE 1.3e-5), −0.093264 vs −0.093261 (SE 1.8e-5); variances within 1.2–2.3 batch-means SE.
- **Mixing at equal iterations** (2 curves, 4 chains × 3000): infHMC ESS/draw σ² 0.47, loglik
  0.33, warps 0.46, max R̂ 1.0004; pCN 0.035 / 0.008 / 0.022, max R̂ 4.28 (asserted ≥ 5× and
  R̂ < 1.05). Earlier 5000-it exploration: pCN warps ESS/draw 0.002 vs infHMC 0.4 (≈ 200×);
  per-iteration cost of infHMC ≈ 1.3–1.6× pCN.
- **Acceptance** 0.67–0.72 for target 0.65 (asserted (0.5, 0.85)); template acceptance 1 (Gibbs).
- **Determinism**: bit-identical reruns; chain k == single chain seeded seed + k; differs from pCN.
- **Recovery**: exponential warps (identity baseline > 0.1) recovered to 0.023 max abs (< 0.05).
- **Multimodality**: 5-curve problem where one of four infHMC chains settles in a poor mode
  (σ² 0.66 vs 0.22) → R̂(σ²) 3.4, all curves BC 0.83–0.97 > 5/9, mode_spread > 0.01 (test asserts
  flags whenever chains disagree). Converged runs give BC ≈ 0.27–0.36.
- ESS vs AR(1) theory (ρ = 0, 0.5, 0.9, within 15%); Sarle BC vs scipy 1.18.1 literals (1e-12),
  Gaussian → 1/3, N(±3,1) mixture → 0.725.
- Validation errors for HMC settings (ignored by pCN; pCN mixture ignored by infHMC).
- Periodicity: v and −v(2π − r)/r give the same γ; rejected under `injective_warps`.

## Gates
- `cargo fmt --all` clean.
- `cargo test -p fdars-core --lib bayesian`: 36 passed, 2 ignored (4.8 s debug);
  `alignment::hygiene` 3 passed; `cargo test --doc bayesian` 2 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
- `cargo check -p fdars-core --no-default-features` clean.
- pmat: new file max cognitive 13; module max 19 (≤ 23).
- `cargo doc`: 94 warnings before and after (pre-existing, none new).
- No serde derives added.

## Deviations / notes
- infHMC always applies `‖v‖ < π` (documented); pCN default model unchanged.
- infHMC moves locally: chains can settle in poor, well-separated local modes (observed with 5
  curves); diagnostics (`rhat_*`, `mode_spread`, `multimodal_curves`) flag this. No hybrid
  mode-jumping move was added.
- fdasrsf's infHMC differs in model details we do not copy (cubic B-spline template with
  `N(0, 1e3 I)` prior, `C_j = 1/⌈j/2⌉`, centring every 20); we keep this module's model so the
  two samplers share a posterior. No direct fdasrsf infHMC golden comparison (MCMC output is not
  seed-comparable across implementations); validated distributionally instead.
- ESS is summed per-chain Geyer ESS (not the rank-normalised multi-chain ESS of Vehtari et al.).

## Remaining
Nothing for #52.
