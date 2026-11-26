---
quick_id: 261126-lfb
issue: 69
backlog_id: ALN-CAV
---
# ALN-CAV (#69): mode escape for infHMC Bayesian group alignment

## Goal
infHMC chains can settle in poor, well-separated joint (template + warps) modes (5 curves:
one of four chains at σ² ≈ 0.66 vs 0.22). Add an opt-in, exactly invariant mode-escape
mechanism; defaults bit-identical.

## Candidates (chosen by evidence on the stuck example, several seeds)
(a) pCN/independence jumps mixed into the per-curve infHMC kernel (`hmc_jump_prob`,
    `hmc_jump_beta`); valid mixture kernel (pCN reversible w.r.t. the prior → MH ratio =
    likelihood ratio).
(c) Parallel tempering within each chain (`n_temperatures`, `pt_min_inv_temp`): K replicas
    targeting L^β p, every sampler step tempered (pCN ratio, HMC scale, Gibbs template and σ²),
    even/odd adjacent swaps with exact MH acceptance; only β = 1 reported;
    `BayesianGroupChain::swap_acceptance`.
(b) DP-seeded restarts rejected a priori: restart selection would hide genuine multimodality
    from R̂ (all chains pick the same mode) and is not a kernel.

## Validation
- Bit-identity: infHMC hashes pinned on 80fb0916 before edits; pCN hashes unchanged.
- Stuck-chain reproduction → with the chosen option all chains agree, R̂(σ²) ≈ 1.
- Invariance: tempered HMC kernel vs exact quadrature of the tempered conditional; full
  sampler with PT vs long injective pCN on the 2-curve tiny problem (within MC error).
- Swap ratio hand check; ESS/R̂ and runtime reported.
- Docs: module "Multimodality" guidance on the flags.
