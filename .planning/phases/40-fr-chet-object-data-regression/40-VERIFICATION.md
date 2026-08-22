---
phase: 40-fr-chet-object-data-regression
verified: 2026-08-22T12:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 40: Fréchet / Object-Data Regression Verification Report

**Phase Goal:** A user can perform metric-space (object-data) regression and statistics via a new `fdars-core/src/frechet/` module — a MetricSpace abstraction (distance + weighted-Fréchet-mean solver) with a 1D-Wasserstein density backend, Fréchet mean/variance, 1D 2-Wasserstein distance, global + local Fréchet regression over Euclidean predictors, density-response regression, and a Fréchet ANOVA group-difference test — reusing `density_fda.rs`, additive/non-breaking, no new crate dependency. R baseline: `frechet`.

**Verified:** 2026-08-22
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | MetricSpace abstraction (distance + weighted-Fréchet-mean solver) with a 1D-Wasserstein density backend; all new entry points Result-returning + crate-root re-exported in `fdars-core/src/frechet/` | VERIFIED | `frechet/space.rs`: `pub trait MetricSpace: Send + Sync` with `distance()` + `weighted_frechet_mean()`; `WassersteinDensitySpace` impl confirms delegation to `wasserstein2_distance` / `wasserstein_barycenter`; all 10 symbols in `pub use frechet::{...}` block at `lib.rs:152–156`. |
| 2 | Fréchet mean (weighted barycenter) + variance (mean squared distance to mean): density mean agrees with DENS-01's Wasserstein mean within tolerance; variance ≈0 for identical-object sample and grows with dispersion | VERIFIED | `frechet/mean.rs`: `frechet_mean` is generic `<S: MetricSpace>`, delegates to `space.weighted_frechet_mean`. Tests: `mean_of_identical_recovers_object` (w2 < 0.15), `variance_zero_for_identical_sample` (var < 0.02), `variance_grows_with_dispersion` (v_wide > v_tight) — all 23 frechet tests pass. |
| 3 | 1D 2-Wasserstein distance (quantile-based, reusing DENS-01 quantile machinery): 0 for identical, matches a hand-computed shift reference | VERIFIED | `frechet/space.rs`: `pub fn wasserstein2_distance` reuses `density_to_quantile` (same CDF→quantile step as `density_fda`). Tests: `w2_identical_is_zero` (< 1e-8), `w2_matches_location_shift` (|w2−0.5| < 0.05 for N(0,1) vs N(0.5,1)) — both pass. |
| 4 | Global (global-linear-weight) + local (local-linear/kernel-weighted) Fréchet regression over Euclidean predictors tracking a known predictor→object relationship within tolerance; density-response variant predicts a conditional density | VERIFIED | `frechet/regression.rs`: `frechet_global_reg` uses Petersen–Müller signed weights via `signed_quantile_average` (never `wasserstein_barycenter`); `frechet_local_reg` uses product Gaussian kernel + local-linear correction. Tests: `global_tracks_known_relationship` (W2 < 0.25 at x=0.5), `local_tracks_known_relationship` (W2 < 0.25 at x=0.0). Density-response variant confirmed: density-row responses fed directly to both functions. Both confirmed via grep: `wasserstein_barycenter(` is absent from `regression.rs`. |
| 5 | Fréchet ANOVA returns a numeric statistic (+ p-value) that flags genuine between-group difference and does not flag homogeneous sample; all entry points reuse `density_fda.rs`; no new crate dependency; seeded RNG; invalid inputs return FdarError; existing public signatures unchanged; full suite + clippy --all-targets green | VERIFIED | `frechet/anova.rs`: Dubey–Müller Tn + χ²(k−1) asymptotic + seeded permutation p-value; reuses `frechet_mean`/`frechet_variance` (Wave 1) + `inference::dist::chi_square_sf` (no `statrs`); per-iteration `StdRng::seed_from_u64(seed.wrapping_add(perm))`. Tests: `anova_flags_shifted_groups` (perm p < 0.05), `anova_ignores_homogeneous_sample` (perm p > 0.05), `anova_permutation_is_seed_reproducible` (same result for same seed). `fdars-core/Cargo.toml` unchanged (git diff HEAD~15 returns empty). Clippy `--all-targets --features linalg,parallel -- -D warnings` exits 0. |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/frechet/mod.rs` | Module barrel; result structs; re-exports | VERIFIED | Declares `mod anova; mod mean; mod regression; mod space;`; defines `FrechetGlobalRegResult`, `FrechetLocalRegResult`, `FrechetAnovaResult` with correct derives + serde-gated + `#[non_exhaustive]`; re-exports all 10 public symbols. |
| `fdars-core/src/frechet/space.rs` | `MetricSpace` trait + `WassersteinDensitySpace` + `wasserstein2_distance` + `signed_quantile_average` | VERIFIED | 427 lines, substantive: trait, struct, `pub fn wasserstein2_distance`, `pub(crate) fn signed_quantile_average`, private `density_to_quantile`, inline tests. |
| `fdars-core/src/frechet/mean.rs` | Generic `frechet_mean` + `frechet_variance` | VERIFIED | `#[must_use]` on `frechet_mean`; generic over `<S: MetricSpace>`; validates empty sample, weight-length mismatch, zero-sum weights; inline tests cover all behaviors. |
| `fdars-core/src/frechet/regression.rs` | `frechet_global_reg` + `frechet_local_reg` | VERIFIED | `#[must_use]` on both; Cholesky via `linalg::{cholesky_factor, cholesky_forward_back, cholesky_solve}`; 1e-6 ridge on Σ̂ and μ₂; `signed_quantile_average` called, `wasserstein_barycenter` absent; inline tests. |
| `fdars-core/src/frechet/anova.rs` | `frechet_anova` + Dubey–Müller Tn | VERIFIED | `#[must_use]`; reuses `frechet_mean`/`frechet_variance`; `inference::dist::chi_square_sf` for asymptotic p-value; per-iteration seeded permutation; σ̂ₗ² flagged `[ASSUMED]` in rustdoc (two occurrences); 5 inline tests. |
| `fdars-core/src/lib.rs` | `pub mod frechet;` + all 10 symbol re-exports | VERIFIED | `pub mod frechet;` at line 92; `pub use frechet::{frechet_anova, frechet_global_reg, frechet_local_reg, frechet_mean, frechet_variance, wasserstein2_distance, FrechetAnovaResult, FrechetGlobalRegResult, FrechetLocalRegResult, MetricSpace, WassersteinDensitySpace};` at lines 152–156. |
| `fdars-core/src/inference/mod.rs` | `pub(crate) mod dist;` | VERIFIED | Line 30: `pub(crate) mod dist;` — additive widening, non-breaking, no public API change. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `MetricSpace::distance` (WassersteinDensitySpace) | `wasserstein2_distance` | direct delegation | VERIFIED | `space.rs:87`: `wasserstein2_distance(a, b, &self.argvals)` |
| `MetricSpace::weighted_frechet_mean` (WassersteinDensitySpace) | `density_fda::wasserstein_barycenter` | assembles FdMatrix, calls barycenter | VERIFIED | `space.rs:119`: `wasserstein_barycenter(&mat, &self.argvals, Some(weights))` — confirmed `signed_quantile_average` absent from this path |
| `frechet_global_reg` | `signed_quantile_average` | direct call, never `wasserstein_barycenter` | VERIFIED | `regression.rs:137`: `signed_quantile_average(responses, argvals, &weights, n_q)?`; grep confirms `wasserstein_barycenter(` absent from regression.rs |
| `frechet_local_reg` | `signed_quantile_average` | direct call after product-kernel + local-linear correction | VERIFIED | `regression.rs:243`: `signed_quantile_average(responses, argvals, &weights, n_q)?` |
| `frechet_anova` | `frechet_mean` / `frechet_variance` | for per-group and pooled variances | VERIFIED | `anova.rs:9`: `use super::mean::{frechet_mean, frechet_variance};`; called at lines 47, 48, 57 |
| `frechet_anova` | `inference::dist::chi_square_sf` | asymptotic p-value — in-crate, no statrs | VERIFIED | `anova.rs:14`: `use crate::inference::dist::chi_square_sf;`; called at `anova.rs:164` |
| `signed_quantile_average` | `density_fda::{dedup_adjacent, quantile_density_from_q}` | density quantile back-map (pub(crate) widening) | VERIFIED | `space.rs:12`: `use crate::density_fda::{dedup_adjacent, quantile_density_from_q, wasserstein_barycenter};`; used in `signed_quantile_average` at lines 267–270 |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 23 frechet tests pass | `cargo test -p fdars-core --features linalg,parallel frechet` | 23 passed, 0 failed | PASS |
| Clippy --all-targets clean | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Finished with no warnings | PASS |
| `wasserstein2_distance` = 0 for identical distribution | `frechet::space::tests::w2_identical_is_zero` | w2 < 1e-8 | PASS |
| Location-shift W2 matches δ=0.5 | `frechet::space::tests::w2_matches_location_shift` | |w2−0.5| < 0.05 | PASS |
| Global regression tracks truth | `frechet::regression::tests::global_tracks_known_relationship` | W2 < 0.25 | PASS |
| Negative signed weights return Ok | `frechet::regression::tests::global_accepts_negative_weights` | Ok with valid density | PASS |
| ANOVA flags shifted groups | `frechet::anova::tests::anova_flags_shifted_groups` | perm p < 0.05 | PASS |
| ANOVA ignores homogeneous sample | `frechet::anova::tests::anova_ignores_homogeneous_sample` | perm p > 0.05 | PASS |
| Permutation reproducible with same seed | `frechet::anova::tests::anova_permutation_is_seed_reproducible` | Identical p-values | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| FRE-01-01 | 40-01 | MetricSpace trait + WassersteinDensitySpace backend | SATISFIED | `frechet/space.rs`: trait + struct; `WassersteinDensitySpace::new` validates grid |
| FRE-01-02 | 40-01 | Fréchet mean (agrees with DENS-01 barycenter within tolerance) | SATISFIED | `frechet/mean.rs`: `frechet_mean` delegates to `weighted_frechet_mean`; test passes w2 < 0.15 |
| FRE-01-03 | 40-01 | Fréchet variance (~0 for identical, grows with dispersion) | SATISFIED | `frechet/mean.rs`: `frechet_variance` = Σ wᵢ d²(objectsᵢ, mean); tests pass |
| FRE-01-04 | 40-02 | Global Fréchet regression (Petersen–Müller) | SATISFIED | `frechet/regression.rs`: `frechet_global_reg` with Σ̂⁻¹ via Cholesky, signed weights |
| FRE-01-05 | 40-02 | Local (kernel-weighted local-linear) Fréchet regression | SATISFIED | `frechet/regression.rs`: `frechet_local_reg` with product Gaussian kernel + μ₂⁻¹μ₁ correction |
| FRE-01-06 | 40-01 | Public `wasserstein2_distance` W₂ distance function | SATISFIED | `frechet/space.rs`: `#[must_use] pub fn wasserstein2_distance`; crate-root re-exported |
| FRE-01-07 | 40-02 | Density-response Fréchet regression | SATISFIED | Both `frechet_global_reg` / `frechet_local_reg` accept density-row responses (the density-response variant is delivered by the same entry points) |
| FRE-01-08 | 40-03 | Fréchet ANOVA (Dubey–Müller Tn + seeded permutation p-value) | SATISFIED | `frechet/anova.rs`: `frechet_anova` complete with all required components |

---

### Documented Divergences (Verified in Source)

Both divergences cited in the phase specification are confirmed present in rustdoc:

1. **Isotonic projection vs R's osqp QP**: Both `frechet_global_reg` and `frechet_local_reg` carry explicit rustdoc stating they use `signed_quantile_average` (sort-based isotonic projection), NOT `wasserstein_barycenter`, because Petersen–Müller / local-linear correction weights can be negative. Confirmed at `regression.rs:70–75` and `regression.rs:158–161`.

2. **σ̂ₗ² estimator [ASSUMED] note**: Present at `anova.rs:24–31` (`compute_tn` doc) and referenced again at `anova.rs:111`. Cites Dubey & Müller (2019, Biometrika 106(4)) and R `frechet::DenANOVA` provenance; explicitly states permutation p-value is primary inference and robust to this assumption.

---

### No New Crate Dependency

`fdars-core/Cargo.toml` is unchanged from 15 commits prior (git diff HEAD~15 returns empty output). No `statrs`, `osqp`, or any other new dependency was added. The ANOVA asymptotic p-value reuses the in-crate `inference::dist::chi_square_sf`.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | No TBD/FIXME/XXX markers, no dead_code attributes, no stubs in frechet/ |

---

### Human Verification Required

None. All observable truths are programmatically verifiable via inline tests that actually run and pass.

---

### Gaps Summary

No gaps. All 5 ROADMAP success criteria are met with direct codebase evidence:
- 23/23 frechet module tests pass
- All 10 symbols crate-root re-exported
- Signed regression path confirmed separated from barycenter path
- No new crate dependency
- Clippy --all-targets clean
- Both documented divergences (isotonic projection, [ASSUMED] σ̂ₗ²) verified in source

---

_Verified: 2026-08-22T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
