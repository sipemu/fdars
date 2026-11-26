---
quick_id: 261126-lfb
issue: 69
backlog_id: ALN-CAV
status: complete
issue_fully_done: true
---
# ALN-CAV (#69) summary: mode escape for infHMC (parallel tempering)

## Commits
- `576acc52` feat(quick-261126-lfb): parallel tempering for Bayesian group alignment
  (mode escape for infHMC) (`closes #69`)

## Public API (additive)
- `BayesianGroupConfig::n_temperatures` (default 1 = off), `pt_min_inv_temp` (default 0.1).
  Works with both samplers.
- `BayesianGroupChain::swap_acceptance: Vec<f64>` (per adjacent replica pair; empty when off).

## Algorithm
Per chain K replicas, β_k = β_min^{k/(K−1)}, target L^β·prior. Tempered steps: pCN warp /
template log-ratio ×β, infHMC Hamiltonian likelihood scale β/(2σ²), Gibbs template precision
β n BᵀB/σ² (+prior) with mean term ×β, σ² ~ IG(α₀ + β nT/2, β₀ + β SSE/2). After each sweep,
adjacent pairs (k, k+1) with k ≡ it (mod 2) swap states with probability
min(1, exp((β_k − β_{k+1})(ℓ_{k+1} − ℓ_k))) (MH on the product space with a symmetric
proposal → product of tempered targets invariant → β = 1 marginal is the posterior). Step
sizes / adaptation and acceptance counters stay with the level; one RNG per chain, replicas
sequential → deterministic, chain k still seeded seed + k.

## Evidence for the choice (5-curve stuck problem, `tiny_data(5)`, 4 chains × 3000 it, debug)
- Plain infHMC stuck for seeds 3, 13, 33, 53 of {3,13,23,33,43,53} (one chain σ² 0.65–0.68 vs
  0.224; log-lik −126 vs −69.5; template of the stuck chain is a different shape — a joint mode).
- (a) pCN jumps mixed into infHMC (`p` per curve, step b): b=1 p=.1 and b=.5 p=.1 fixed 0/3 seeds
  and made seed 23 *worse* (a chain jumped into the poor mode); b=.3 p=.3 fixed 2/3. Rejected and
  removed (single-curve moves with fixed template cannot leave a joint mode reliably).
- (c) PT: geometric ladder too coarse near β=1 with K=3–4 (pair-0 swap ≈ 0); K=6 β_min=.1 fixed
  6/6, K=8 β_min=.1 6/6, K=10 β_min=.05 6/6, K=8 β_min=.05 5/6 (one at R̂ 1.08). Chosen default
  β_min = 0.1. Bottleneck pair near β≈0.3 (rate 0.00–0.05) = the phase transition between modes.
- (b) DP-seeded restarts / best-of-M starts: not implemented — not a kernel, and selection would
  make all chains agree on one mode even when the posterior is genuinely multimodal, silencing R̂.

## Results (seed 3)
| | chain σ² means | R̂(σ²) | max R̂ | ESS/draw (σ², loglik, min warp) | time debug / release |
|---|---|---|---|---|---|
| plain | 0.224 0.224 0.224 0.656 | 3.383 | 9.76 | 0.49 0.22 0.34 | 2.5 s / 0.4 s |
| PT6 β_min 0.1 | 0.223 0.222 0.225 0.224 | 1.0003 | 1.0015 | 0.61 0.36 0.52 | 7.3 s / 1.1 s |
Swap rates [0.24, 0.16, 0.03, 0.33, 0.35]. Seeds 13/33/53: R̂(σ²) ≤ 1.002, flags cleared.
Cost ≈ 3× run time for K = 6 (linear in K per iteration; summaries unchanged).

## Invariance validation
- `tempered_hmc_kernel_leaves_tempered_conditional_invariant`: β = 0.3 single-curve HMC vs 401²
  quadrature of exp(−β SSE/2σ²)·prior (means within 4 SE, variances within 4 rel-SE).
- `parallel_tempering_preserves_the_posterior`: 2-curve tiny problem, PT4 vs plain infHMC,
  4×6000 it: σ² 0.25341 vs 0.25299 (SE 1.1e-3), areas 0.081374 vs 0.081382 (SE 3.6e-5),
  −0.093297 vs −0.093293 (SE 4.5e-5); variances within 2 SE.
- `#[ignore]` `parallel_tempering_and_long_pcn_agree_on_posterior_moments` (release 16 s total):
  PT4 infHMC 4×5e4 and PT3 injective pCN 4×6e5 vs plain injective pCN 4×1e6 (thin 20):
  σ² 0.25313 / 0.25350 vs 0.25296 (SE 5e-4 / 7e-4); areas 0.081354 / 0.081350 vs 0.081356,
  −0.093258 / −0.093258 vs −0.093261; variances within ≈1 SE. All R̂ < 1.001.
- `replica_exchange_acceptance_matches_formula`: empirical swap rate 0.67093 vs
  min(1, e^{0.6Δℓ}) = 0.67160 (SE 0.0023); levels keep their β; uphill swap always accepted.
- `temperature_ladder_and_validation`, `parallel_tempering_escapes_poor_mode` (short, asserts plain
  stuck + PT escape), `#[ignore]` multi-seed version (seeds 3/13/33/53).
- Bit-identity: new `default_infhmc_sampler_is_bit_identical_to_pre_mode_jump_release` (two infHMC
  configs, hashes incl. ESS/BC/step sizes, pinned on 80fb0916 before any code edit); existing pCN
  hash test unchanged and passing.

## Docs
Module docs "Multimodality": reading rhat_sigma2 / mode_spread / multimodal_curves / per-chain
log-likelihood, when discarding a chain is justified (large log-lik gap, other chains agree) and
when not (similar-likelihood modes = genuine multimodality), ladder sizing from swap_acceptance
(levels needed grow ~ √(nT)), runnable doctest. CHANGELOG (root + fdars-core) "Added".

## Gates
- `cargo fmt --all` clean; `cargo test -p fdars-core --lib bayesian_group` 35 passed, 5 ignored;
  ignored PT studies pass in release; `--doc bayesian` 3 passed; `alignment::hygiene` 3 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
- `cargo check -p fdars-core --no-default-features` clean. pmat max cognitive in module 19.
- No serde types added.

## Deviations / notes
- Mechanism (a) implemented, measured and removed; (b) not implemented (reasons above).
- Ladder is fixed geometric (no adaptive ladder); docs explain sizing. Large problems need more
  replicas (∝ √(nT)) — possible follow-up: adaptive ladder (Syed et al. 2021 / Vousden 2016).
- Swap rates are counted over all iterations (incl. burn-in).
