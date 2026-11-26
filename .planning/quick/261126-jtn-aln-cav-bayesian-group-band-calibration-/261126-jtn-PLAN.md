---
quick_id: 261126-jtn
issue: 64
backlog_id: ALN-CAV
---
# ALN-CAV (#64): calibration of Bayesian group alignment credible bands

## Goal
Separate sampler error from model misfit behind the 83–87% coverage of nominal-95% warp bands
of `bayesian_align_group`, fix any summary bug, and set test thresholds to the calibrated case.

## Approach
1. In-model simulation harness (tests only): known template SRSF q* in the B-spline span, known
   warps in the 1-harmonic warp space, data SRSFs q_i constructed so that the discrete likelihood
   `interp(q_i, γ_i(t_j))·ψ_i(t_j) = q*_j + ε_ij`, ε iid N(0, σ²), holds exactly (linear collocation
   solve; Schoenberg–Whitney condition |γ_i(t_j) − t_j| < h makes it non-singular). Problem built
   from these q_i directly (bypassing f → SRSF).
2. Coverage of 95% pointwise warp bands and template (SRSF) bands vs centred truth, over many
   seeds, pCN (injective_warps) and infHMC, center_draws on/off; MC SE across seeds. Same harness
   on the original f-noise design for comparison.
3. Audit summaries (quantiles, centring, pooling, burn-in/thin).
4. Fix bugs / document misfit / optional f-space likelihood depending on findings; tighten the
   coverage tests; CHANGELOG.
