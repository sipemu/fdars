---
quick_id: 261126-m5e
status: complete
issue: 72
commits:
  - a389c1cd feat(quick-261126-m5e): periodic continuous registration + R-default W basis preset (refs #72)
  - 1b4466f4 feat(quick-261126-m5e): simultaneous bands, held-out bandwidth tuning, fragmented/sparse observation for partial conformal (closes #72)
---
# Summary — ALN-CAV missing options (#72)

## A. continuous_registration (alignment/continuous.rs)
- `ContinuousRegistrationConfig::periodic` (default false) — R `register.fd(periodic = TRUE)`:
  W coefficient 0 becomes the free shift δ (as R), registered curve `y(wrap(h(t)+δ))`, Jacobian
  column ∂/∂δ = Dy(h+δ) (exact; R's gradient is approximate). Discrete curves are interpolated by a
  natural spline over three concatenated periods (≈ periodic spline). `initial_coefficients` col 0
  = initial shift when periodic (as R). Returned warps = h + δ (R `warpfd`), `w_coefficients` col 0
  stays 0, new `ContinuousRegistrationResult::shift`.
- `ContinuousRegistrationConfig::r_defaults()` — 2 linear B-splines, lfd_order 0, max_iter 20, crit 2.
  `default()` unchanged.
- Golden: R fda 6.3.0 script `golden_register_fd_periodic.R` (this dir). Periodic (Fourier yfd, 21
  basis, default W, both crit): fdars criterion ≤ R's at R's optimum, |Δshift| ≤ 5.7e-4,
  |Δc2| ≤ 4.6e-3, registered values within 1e-2. R-default W basis on the #47 data: criterion ≤ R,
  within 1 %, |Δc2| < 2e-2. Plus FD gradient incl. shift column (and wrap past b), shift recovery
  with wraparound (δ up to ±0.3, exact to 1e-4), period invariance, spline periodicity.
- Bit identity: FNV hash of default outputs pinned on 9c856e31 (pre-change, via git stash) and
  asserted in `default_results_bit_identical_to_pre_periodic_pin`.

## B. partial_conformal_* (conformal/partial.rs)
1. **Bandwidth tuning (held-out).** The paper (Sec. on tuning) adapts Liang, Zhu & Foygel Barber
   (2024, ModSel-CP) selecting h by minimum average PI length on the same augmented data. ModSel's
   exact guarantee requires re-selecting h per trial response; with our affine full-conformal
   scores this has no closed form, so we implement the validity-preserving held-out form:
   `tuning_fraction` reserves a seeded part T of the calibration curves; h* = argmin mean band width
   over leave-one-out rotations within T (under the new curve's observation pattern); h* depends on T
   only ⇒ exact finite-sample coverage. `partial_conformal_select_bandwidth(tuning_model, obs, cfg)`
   exposes it; `PartialBandwidthSelection` in the prediction. Candidates: user list or 5–100 %
   quantiles of tuning-set predictor distances. Effect (one-off experiment, n=60, α=0.2, prefix 12/25):
   mean pointwise width 0.304 (median rule) → 0.163 (tuned).
2. **Simultaneous bands.** `PartialAmplitudeBand::Simultaneous`, `PartialBandScale::{Constant,
   TrainingSd, Custom}`. Score sup_t |Y−Ŷ|/σ; band ĉ ± Rσ with R = k-th smallest of A_i/(1−b_i),
   A_i = sup|a_i − b_i ĉ|/σ, which provably contains the full-conformal set (triangle inequality) ⇒
   simultaneous finite-sample coverage; b=0 reduces to the split sup-norm band (tested).
3. **Fragmented / sparse.** `PartialObservation::{Prefix, Fragments, Sparse}` +
   `partial_conformal_predict_observed`. Paper: fragments use d_prod = Σ λ_j d(f|J_j, f'|J_j),
   Σλ_j = 1 (paper leaves λ free; fdars uses λ_j ∝ fragment length); sparse uses Euclidean distance in
   ℝ^N (fdars linearly interpolates calibration curves at arbitrary times). Fragments are located with
   `elastic_partial_match` (`fragment_registrations`; `registration` = longest fragment; sparse: match
   of the point set).

Validation: FNV pin of defaults (pre-change hash), single fragment ≡ prefix bitwise, exact rotation
counts ≥ k (simultaneous ×3 patterns/scales, fragments L2/SRSF, sparse), brute-force containment of
the simultaneous bound (4000 random accepted trials, bound reached ≥ 70 %), d_prod/sparse distance by
hand, LOO tuning criterion recomputed via the public API for every candidate (1e-9), seeded MC:
simultaneous const/modulated 0.81/0.84 (200 reps, 1−α=0.8, n2=19), sparse pointwise 0.806,
tuned pointwise/simultaneous 0.826/0.867 (150 reps, n2=20).

## Gates
- `cargo fmt --all` clean; `cargo test -p fdars-core --lib alignment::continuous` 21/21,
  `--lib conformal::` 60/60; doctests `continuous` 2/2, `conformal` 9/9.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
- `cargo check -p fdars-core --features serde` clean; `--no-default-features` clean.
- pmat cognitive max: partial.rs 18, continuous.rs 16.

## Deviations / remaining
- Held-out tuning instead of the paper's same-data rule (documented reason above).
- λ_j ∝ fragment length is a choice (paper unspecified).
- Phase band envelope/radius remain display approximations (exact membership via
  `phase_set_contains`), unchanged — the issue lists it as a caveat, not a missing option.
