---
status: complete
quick_id: 261126-al4
issue: 61
---
# Summary — ALN-WFDA (GH #61): Tang–Müller pairwise curve synchronisation (fdapace `WFDA`)

## Commits
- `1236c5fd` feat(quick-261126-al4): Tang–Müller pairwise curve synchronisation (fdapace WFDA) (closes #61)

## Public API added (module `alignment::wfda`, re-exported from `alignment` and the crate root)
- `pairwise_synchronization(data: &FdMatrix, argvals: &[f64], &WfdaConfig) -> Result<WfdaResult, FdarError>`
- `WfdaConfig` (`#[non_exhaustive]`, serde): `choice` (Truncated), `subset_prop` 0.5, `nknots` 2,
  `is_pwl` true, `seed` 666, `lambda: Option<f64>` (None → fdapace default), `cost_measure`
  (PairwiseFit), `truncation_quantile` 0.9, `max_evals` 4000, `optim_tol` 1e-12; `validate()`.
- `WfdaChoice { Truncated, Weighted }`, `WfdaCostMeasure { PairwiseFit, FdapaceLegacy }`.
- `WfdaResult`: `warps`, `inverse_warps`, `aligned` (n × m, argvals scale), `costs`, `lambda`,
  `partners`, `pairwise_warps` (per curve K × m), `pairwise_costs`.

## Algorithm (mirrors fdapace 0.6.0 source)
Amplitude normalisation y/max|y|; λ = 1e-4·√(Σ∫(ỹ−μ̃)²/(n−1)) (trapezoid over argvals);
K = round_half_even(subsetProp·(n−1)); partners via StdRng(seed + i); PWL warps with equally spaced
interior knots, heights optimised in [1e-6, 1−1e-6] on Σ(ỹ_k∘h − ỹ_i)² + λΣ(h − w)²; random search
over K² sorted-uniform candidates (shared across pairs, as fdapace); truncated (type-7 quantile) or
1/cost-weighted mean → h⁻¹; h by grid inversion (R `approx`, ties = mean); aligned = y∘h.

## Validation
- **Golden (R fdapace 0.6.0, minqa 1.2.8)** — `golden_wfda.R` (this dir) generates
  `fdars-core/src/alignment/wfda_golden.rs`. N = 8, M = 41 on [0, 10], subsetProp = 1 (RNG-free).
  Five cases: unmodified WFDA truncated / weighted (`FdapaceLegacy`), and WFDA with the one-line
  cost fix truncated / weighted / nknots 3 + λ 1e-3 (`PairwiseFit`). Observed agreement: λ to 1e-12
  rel, h / hInv ≤ 2.9e-5, aligned ≤ 4.7e-4, costs ≤ 3.7e-5 rel; asserted with 1e-4 / 2e-3 / 1e-4
  (difference is BOBYQA's loose stop; fdars reaches equal or lower pairwise objectives — checked for
  curve 1's 7 pairs against bobyqa `fval`).
- Invariants: identity warps / aligned = data on identically shaped curves; exact recovery of a
  PWL pairwise warp (n = 2, λ = 0, error < 2e-3); group recovery of h_i⁻¹ = mean_{k≠i} g_k∘γ_i for
  quadratic warps (nknots 7, error < 1e-2) and the predicted registered curves; monotone warps with
  fixed endpoints (PWL and random search); h is the grid inverse of h⁻¹; h⁻¹ equals the truncated /
  weighted average of the returned pairwise warps; fdapace's K rounding (half-even); seeded,
  deterministic, distinct partners; R `quantile` type 7 and `approx` ties = mean hand values;
  zero-cost weighted limit; input / config errors. 14 unit tests + doc test.

## Gates
- `cargo fmt --all` clean.
- `cargo test -p fdars-core --lib wfda` 14/14 (default features and `--no-default-features`).
- `cargo test -p fdars-core --doc pairwise_synchronization` 1/1.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
- `cargo check -p fdars-core --features serde` clean.
- pmat complexity on `wfda.rs`: max cognitive 16 (≤ 23).

## Deviations from fdapace (documented in the module docs)
- **fdapace bug:** its pairwise cost compares against `ymatNormalised[candidateKcurves[k]]`, a single
  matrix element (linear indexing), not the partner curve. Default `PairwiseFit` uses the intended
  data-fit cost; `FdapaceLegacy` reproduces fdapace exactly. Worth reporting upstream.
- Optimiser: box-projected Nelder–Mead with restarts, multi-started from the identity (fdapace's
  start) and power warps u^p (p ∈ {0.5, 0.75, 1.5, 2}); identity-start NM alone fell into a poor
  local minimum for the largest warps of the golden data where BOBYQA did not.
- RNG: partner subsets (subset_prop < 1) and random-search candidates use StdRng, so they differ
  from R for the same seed (hence golden tests use subsetProp = 1; random search is validated by
  invariants only).
- PWL warps are evaluated on the work grid (fdapace: M equally spaced points) — identical for
  equally spaced argvals.
- Weighted mean with zero pairwise costs uses the limit (mean over zero-cost partners) instead of NaN.

## Remaining
None for #61. STATE.md not updated (orchestrator does it at merge, per instructions).
