---
quick_id: 261126-vzj
issue: 82
---
# Fix GH #82 — wcr/wnet β(t) inconsistent with fitted values (non-orthogonal DWT)

## Goal
`wcr` / `wnet` must return β(t) such that `fitted_i = intercept + Σ_t x_i(t) β(t)` under
every `BoundaryMode` and signal length.

## Root cause
The fit is linear in the coefficient design `C = X Aᵀ` (A: analysis operator, m → P).
Hence the time-domain coefficient is `β = Aᵀ w` (adjoint). `coeff_weights_to_beta_t`
used `reconstruct` (= A⁻¹ on the range), which equals Aᵀ only when A is orthogonal
(Periodic with every per-level length even). Symmetric (mirror to 2n; Aᵀ folds the
mirror half back, reconstruct truncates it → factor 1/2 per level, 0.125 at level 3)
and Periodic with odd per-level lengths (duplicated sample) are not orthogonal.

## Fix
Add `wavelet::analysis_adjoint(&WaveletCoeffs)` (crate-private): per level
`core_synthesis` (transpose of the even periodic core) followed by the transpose of the
boundary extension (`fold_extension`: symmetric → `e[i] + e[2n-1-i]`; periodic odd →
add the duplicated tail sample into `n-1`). Route `coeff_weights_to_beta_t` through it.
For the orthogonal case the fold is the identity → bit-identical to the old output.

## Validation
- Invariant test: intercept + X·β(t) == fitted for wcr (PCR+PLS) and wnet, Symmetric and
  Periodic, m = 64 and m = 37 (odd pyramid lengths).
- Planted-β recovery: y = 1 + Xβ_true, full-column-rank X (n ≫ m), symmetric wcr
  recovers β_true and α = 1.
- Adjoint identity test ⟨A x, w⟩ = ⟨x, Aᵀ w⟩ in the wavelet module.
- Existing periodic tests unchanged.
