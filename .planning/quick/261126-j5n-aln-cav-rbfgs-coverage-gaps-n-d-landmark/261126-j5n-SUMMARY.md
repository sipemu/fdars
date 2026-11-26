---
status: complete
quick_id: 261126-j5n
issue: 67
commits: [4dbb5a88]
---
# 261126-j5n — ALN-CAV RBFGS optimiser coverage gaps (#67)

## Public API added (all additive)
- `elastic_align_pair_nd_with_config`, `elastic_distance_nd_with_config`,
  `karcher_mean_nd_with_config` (R^d shared warp).
- `elastic_align_pair_constrained_with_config` (landmarks).
- `karcher_median_with_config(data, argvals, &RobustKarcherConfig, &WarpOptimConfig)`,
  `robust_karcher_mean_with_config(..)` — penalty weight is `warp.lambda`;
  `config.lambda` must be 0 or equal (else `InvalidParameter`).
- `align_to_target_with_config`, `karcher_mean_closed_with_config`.
- `WarpOptimConfig::band_frac` (default 0), `WarpOptimConfig::lambda_scale:
  WarpLambdaScale {Native (default), Dp, Fdasrsf}`, `WarpOptimConfig::rbfgs_lambda_factor`.
- Re-exported from `alignment` and the crate root. `closed_karcher`-with-config
  was missing and is added.

## Design decisions
- **Band + RBFGS:** banded DP seed on the *input* grid (O(m·R), with the DP
  options), unconstrained RBFGS, pointwise clamp projection onto
  |γ(t) − t| ≤ R/(m−1) (monotone, boundary-exact), then the better of projected
  RBFGS / seed in the RBFGS objective ⇒ always in band, never worse than the
  banded DP. Iterates are not projected (documented). Curve shapes and landmark
  alignment reject `band_frac > 0`.
- **λ scale:** derived exactly via the isometry of γ ↦ (q∘γ)√γ̇:
  E_dp/S = (a−b)² + ab·(D_rb + λ_dp/(ab)·P) ⇒ λ_rb = φ·λ_dp,
  φ = scl·E1E2/(e1e2) (1-D; → 1 + O(1/m)), φ = scl/(e1e2) for curve shapes
  (raw SRVFs, ≈ 1/(m‖q1‖‖q2‖)), total-norm φ/d for the per-channel-normalised
  R^d DP (exact for equal channel norms). Finding: for 1-D the two native
  conventions already nearly coincide (the old doc's "not interchangeable" was
  overstated); for curve shapes they differ by ~(m−1).
- **Landmarks + RBFGS:** segmented DP first (with options), each segment mapped
  affinely to [0,1]² (q̃2 carries √κ, κ = L_r/L_c), RBFGS seeded by the DP
  segment; penalty `L2Gam/L2Psi` on κγ̃' (new `pen_slope` in `RbfgsProblem`),
  Roughness scaled by κ²/L², λ normalised with the whole curves' Euclidean norms
  (new `lam_euclid` override) so a segment's objective is the global one
  restricted. Geodesic + λ>0 rejected (non-additive).
- **Seed options:** `RbfgsParams` carries `DpSeed` (nbhd_dim/dp_scl/dp_cost/dp_prune);
  classic options run the original seed code (bit-identical). Curve shapes too.

## Validation (fdars-core/src/alignment/optim_coverage_tests.rs, 21 tests)
- Pre-change FNV hashes (10 groups: N-d pair+Karcher, constrained, median+trimmed+
  align_to_target, closed Karcher, RBFGS defaults 1-D/Karcher/curve shape; λ∈{0,0.05})
  recorded on 79430c46 before any edit — unchanged after.
- Default `_with_config` == classic for every new entry point (assert_eq);
  `band_frac` DP == `elastic_align_pair_banded` / `karcher_mean_banded` exactly.
- RBFGS objective ≤ DP objective: N-d (4.6e-4 vs 2.8e-3), landmarks (7.1e-4 vs
  3.8e-3; penalised L2Gam/L2Psi global objective too), align_to_target per curve,
  banded (all bands), Weiszfeld objective of the median (0.12 vs 0.69).
- Known-warp recovery: N-d |γ−γ0⁻¹|∞ 0.0010 (DP 0.0044); landmarks 0.0039 (DP 0.0112).
- Landmarks exact (assert_eq of γ at snapped grid points) for DP options,
  RBFGS L2Gam/L2Psi/Roughness.
- Symmetric case where the free optimum passes through the landmark: constrained
  RBFGS == free RBFGS to 7e-8 (λ=0.3) / 1e-10 (λ=2) while ×4 / ×¼ λ moves the
  warp by ~1e-2 — validates the segment λ normalisation.
- λ conversion on a fixed warp: penalty ratio ΔE_dp/(ab·ΔE_rb) = 1.003 (m=241),
  1.03 (m=25, vs 1.13 without φ), wrong scale 0.004; data ratio 1.0002;
  `lambda_scale` applied exactly (bit-equal to Native with λφ / λ/φ);
  curve shapes: converted RBFGS pull 0.0506 vs DP 0.0500 (native RBFGS 0.0067).
- Seed: max_iter = 0 RBFGS reproduces the DP-with-options warp (1.3e-3 vs 2.0e-2).
- FD gradient check of the κ-slope penalties.
- No external reference values: semantics are fdars-internal (band, λ
  conversion, landmark RBFGS have no fdasrsf counterpart); checked against
  derived identities and invariants instead.

## Gates
- `cargo fmt --all` ✓; `cargo clippy -p fdars-core --all-targets --features linalg,parallel -D warnings` ✓;
  `cargo check --features serde` ✓; `cargo doc -D warnings` ✓.
- `cargo test -p fdars-core --lib alignment::` 581 passed, 3 ignored;
  `cargo test --doc alignment` 87 passed.
- pmat cognitive max in touched files 22 (≤ 23).

## Deviations / not done
- RBFGS band: final-warp projection + seed safeguard rather than projected
  iterates (documented).
- R^d shared-warp λ conversion exact only for equal channel norms (documented).
- Curve-shape and landmark paths reject `band_frac`.
