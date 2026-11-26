---
quick_id: 261126-vzj
issue: 82
status: complete
commits: [a3cd5839]
---
# Summary — GH #82 wcr/wnet β(t) vs fitted values

**Root cause:** β(t) must be `Aᵀw` (adjoint of the DWT analysis operator A, since the
design is `C = X Aᵀ`). `coeff_weights_to_beta_t` used `reconstruct` (inverse DWT),
which equals `Aᵀ` only for the orthogonal case (periodic, even per-level lengths).
Symmetric mode mirrors to length 2n per level; reconstruct truncates the mirror half
instead of folding it back → β shrunk by ½ per level (0.125 at db4 level 3, as
reported). Periodic with odd per-level lengths (duplicated tail sample) was also
slightly inconsistent — found by the new odd-m test.

**Fix (a3cd5839):** crate-private `wavelet::analysis_adjoint` (per level:
`core_synthesis` + `fold_extension`, the transpose of `extend_signal`); shared
`validate_pyramid` helper with `reconstruct`; `coeff_weights_to_beta_t` now calls the
adjoint. Orthogonal case bit-identical to `reconstruct` (asserted in test). No public
API change. CHANGELOG [Unreleased] Fixed (root + fdars-core).

**Tests (red before fix, green after):**
- `wcr_beta_t_reproduces_fitted_under_every_boundary_mode` (PCR+PLS; Sym/Per; m=64,37)
  — before: gap 5.38 at m=64 Symmetric.
- `wnet_beta_t_reproduces_fitted_under_every_boundary_mode` — before: gap 6.13.
- `wcr_symmetric_recovers_planted_time_domain_beta` (y = 1 + Xβ, full-rank X) —
  before: rel L2 err 0.75.
- `analysis_adjoint_is_exact_transpose_of_decompose` (⟨Ax,c⟩ = ⟨x,Aᵀc⟩; Haar/db4,
  both modes, n=37/64; equality with reconstruct in the orthogonal case).
Reference: mathematical invariant (adjoint identity + unique β for full-rank X);
R `refund` was not installed/used; the invariants determine the correct β(t)
uniquely, so no external reference values were needed.

**Gates:** fmt ✓; `cargo test --lib wavelet` 88/88 ✓ (also `--no-default-features
--features linalg`); doctests wavelet ✓; clippy --all-targets linalg,parallel,serde
-D warnings ✓; rustdoc -Dwarnings ✓; pmat quality-gate 0 violations ✓.

**Behaviour change:** `beta_t` of wcr/wnet changes for Symmetric mode and for
Periodic mode with odd per-level lengths; fitted values, coeff_weights, intercept,
predict unchanged.
