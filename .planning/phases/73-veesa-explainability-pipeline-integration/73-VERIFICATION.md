---
phase: 73-veesa-explainability-pipeline-integration
verified: 2026-09-05T21:00:00Z
status: passed
score: 7/7 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 73: VEESA Explainability Pipeline Integration Verification Report

**Phase Goal:** Users can explain any trained predictor over jfPCA scores via permutation feature importance and reconstruct interpretable principal directions, driven end-to-end from fit -> transform -> PFI.
**Verified:** 2026-09-05T21:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | elastic_pfi is deterministic under seed: two runs with the same seed return byte-identical importance (VEE-03) | VERIFIED | `pfi_seed_determinism` test asserts `run1.importance == run2.importance` (Vec<f64> equality); test passes (4/4 elastic_pfi::tests ok) |
| 2 | On a known-signal design where one PC carries the response, that PC's importance ranks strictly above every noise PC (VEE-03) | VERIFIED | `pfi_known_signal_ranking` test with Custom(-MSE) metric asserts `importance[0] > importance[1]` AND `importance[0] > importance[2]` (strict inequalities); test passes on spanning 4-harmonic fixture (n=10, m=14) |
| 3 | principal_directions at c=0 reproduces the jfPCA mean: amplitude curve == karcher_mean within 1e-10 (VEE-04) | VERIFIED | `principal_directions_c0_mean` asserts `(amp - km).abs() < 1e-10` for each element j in 0..m; passes (4/4 jfpca_model::tests::principal_directions* ok). Implementation uses `mean_srsf` (mu_q_centered) as base + `karcher_mean[0]` as f0 — the exact SRSF pair that built karcher_mean |
| 4 | Reconstruction scales amplitude perturbation by sigma_j = eigenvalues[j].sqrt() (std-dev), NOT the raw eigenvalue (VEE-04) | VERIFIED | `principal_directions_sigma_sqrt_scaling` test: deviation at c=1 is closer to sqrt-scale than raw-scale; dedicated test distinct from c=0 gate. NaN guard added: `eigenvalues[pc_index].max(0.0).sqrt()`. Augmented-dim range tightened to `0..m` (IN-01 fix, commit 02884894) |
| 5 | PrincipalDirections returns amplitude_curves and phase_curves each shaped (n_c, m) (VEE-04) | VERIFIED | `principal_directions_shapes` asserts `pd.amplitude_curves.shape() == (n_c, m)` and `pd.phase_curves.shape() == (n_c, m)`; passes |
| 6 | veesa_pipeline ties fit -> score_training -> elastic_pfi end-to-end and runs as a module doctest under cargo test --doc (VEE-05) | VERIFIED | `veesa_pipeline.rs` line 371-372 chains `jfpca_fit -> score_training -> elastic_pfi`; doctests at lines 9 and 326 both pass under `cargo test --doc --features linalg,parallel`: `2 passed; 0 failed` |
| 7 | All changes are additive and non-breaking: existing public signatures unchanged; new items re-exported from lib.rs and prelude.rs (VEE-05) | VERIFIED | 2809 lib tests pass (baseline 2801 + 8 new); `lib.rs` has `pub mod elastic_pfi;` + re-export block (lines 138, 513-516); `PrincipalDirections` re-exported at line 516; `prelude.rs` lines 76-77 include all 6 new public items; clippy `--all-targets -D warnings` clean; `fmt --check` clean |

**Score:** 7/7 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/elastic_pfi.rs` | elastic_pfi, PfiMetric, ElasticPfiResult, veesa_pipeline, VeesaPipelineResult + inline tests + module doctest | VERIFIED | File exists, 574 lines, all 5 public items present + 4 inline tests + 2 doctests |
| `fdars-core/src/jfpca_model.rs` | JfpcaModel::principal_directions + PrincipalDirections + inline tests | VERIFIED | `principal_directions` method at line 429, `PrincipalDirections` struct at line 77, 4 gate tests at lines 778-915 |
| `fdars-core/src/lib.rs` | pub mod elastic_pfi; + re-export block | VERIFIED | `pub mod elastic_pfi;` at line 138; `pub use elastic_pfi::{elastic_pfi, veesa_pipeline, ElasticPfiResult, PfiMetric, VeesaPipelineResult};` at lines 513-515; `pub use jfpca_model::PrincipalDirections;` at line 516 |
| `fdars-core/src/prelude.rs` | prelude re-export line | VERIFIED | Lines 76-77 export `elastic_pfi, veesa_pipeline, ElasticPfiResult, PfiMetric, PrincipalDirections, VeesaPipelineResult` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `elastic_pfi` | `model.score_training().scores` (Phase 72 seam) | `veesa_pipeline` calls `model.score_training()?` and passes `training_scores.scores` to `elastic_pfi` | WIRED | `elastic_pfi.rs` line 371-372; doctest uses same seam |
| `principal_directions` | `model.mean_srsf / eigenvalues / vert_component / mean_psi / horiz_component` | reads model fields; calls `srsf_inverse` + `exp_map_sphere` + `psi_to_gam` | WIRED | All field reads at `jfpca_model.rs` lines 452, 466, 471-474, 482-487; helpers imported at line 58+64 |
| `elastic_pfi` | `explain::helpers::{shuffle_global, clone_scores_matrix}` | `use crate::explain::helpers::{clone_scores_matrix, shuffle_global};` | WIRED | `elastic_pfi.rs` line 43; called at lines 281-282 without body modification |
| `veesa_pipeline` | `jfpca_fit` + `elastic_pfi` | `jfpca_fit(...)?; model.score_training()?; elastic_pfi(...)` | WIRED | `elastic_pfi.rs` lines 370-372 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `elastic_pfi` | `importance[k]` | `baseline_metric - mean_perm` from real permutation loop over `scores` (FdMatrix from caller) | Yes — computed from caller's real score matrix | FLOWING |
| `principal_directions` | `amplitude_curves[(ci,j)]` | `srsf_inverse(q_perturbed, argvals, f0)` where `q_perturbed` uses stored `mean_srsf` + eigenvector perturbation | Yes — real SRSF inversion | FLOWING |
| `principal_directions` | `phase_curves[(ci,j)]` | `psi_to_gam(exp_map_sphere(mean_psi, v_perturbed, time), time)` scaled to argvals domain | Yes — real sphere exponential map + warp conversion | FLOWING |
| `veesa_pipeline` | `VeesaPipelineResult.pfi` | Fully delegated to `elastic_pfi` via training scores from `score_training()` | Yes — no static returns | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| VEE-03: elastic_pfi inline tests (seed-determinism, known-signal ranking, zero-repeats rejection) | `cargo test -p fdars-core --lib --features linalg,parallel elastic_pfi::tests` | 4 passed; 0 failed | PASS |
| VEE-04: principal_directions gate tests (c=0, sigma-sqrt, shapes, bad-pc rejection) | `cargo test -p fdars-core --lib --features linalg,parallel jfpca_model::tests::principal_directions` | 4 passed; 0 failed | PASS |
| VEE-05: module doctest + veesa_pipeline doctest | `cargo test -p fdars-core --features linalg,parallel --doc elastic_pfi` | 2 passed; 0 failed | PASS |
| Additive/non-breaking: full suite regression | `cargo test -p fdars-core --features linalg,parallel --lib` | 2809 passed; 0 failed (baseline 2801 + 8 new) | PASS |
| Additive/non-breaking: clippy --all-targets | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Finished (no warnings) | PASS |
| Additive/non-breaking: fmt | `cargo fmt --check` | No output (clean) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| VEE-03 | 73-01-PLAN | Model-agnostic PFI over jfPCA scores; deterministic under seed; informative PC ranks above noise PCs | SATISFIED | `elastic_pfi` function accepts `impl Fn(&FdMatrix) -> Vec<f64>`; seed-determinism test passes (bit-identical); known-signal ranking test passes (strict inequality) |
| VEE-04 | 73-01-PLAN | Principal-direction reconstruction: amplitude+phase split, c=0 reproduces jfPCA mean, sigma_j=sqrt(eigenvalue) | SATISFIED | `JfpcaModel::principal_directions` + `PrincipalDirections`; c=0 gate <1e-10 passes; sigma-sqrt test passes; shapes (n_c, m) pass |
| VEE-05 | 73-01-PLAN | Cohesive veesa_pipeline tying fit->transform->PFI; crate-root + prelude re-exports; running module doctest | SATISFIED | `veesa_pipeline` function present and wired; all 7 new public items re-exported in lib.rs + prelude.rs; 2 doctests pass under `cargo test --doc` |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | No debt markers, no stubs, no placeholder implementations found in phase-modified files |

### Code Review Fixes Verified (commit 02884894)

All 4 warnings and 2 info items from 73-REVIEW.md are confirmed resolved:

| Finding | Fix | Verified |
|---------|-----|---------|
| WR-01: PfiMetric missing Debug/Clone/PartialEq | Manual impls added; Custom variant uses `Arc<dyn Fn>` (enables Clone via `Arc::clone`) instead of plan's `Box<dyn Fn>` | PASS — impls at elastic_pfi.rs lines 91-124 |
| WR-02: PfiMetric::Mse doc wrong sign direction | Doc corrected: informative PCs yield **negative** importance with Mse/Mae | PASS — lines 59-76 |
| WR-03: principal_directions missing #[must_use] | `#[must_use = "..."]` added | PASS — jfpca_model.rs line 428 |
| WR-04: compute_metric silently truncates on short predict() output | Validation added: `if baseline_pred.len() != n { return Err(InvalidDimension) }` | PASS — elastic_pfi.rs lines 263-269 |
| IN-01: sigma-scaling test uses augmented-dim index in max_vert | Range changed from `0..=m` to `0..m` (matches perturbation range) | PASS — jfpca_model.rs line 870 |
| IN-02: eigenvalues.sqrt() not guarded against numerical negatives | `.max(0.0).sqrt()` clamping added | PASS — jfpca_model.rs line 452 |

**Notable deviation from PLAN spec:** PLAN specified `Custom(Box<dyn Fn>)`. Implementation uses `Custom(Arc<dyn Fn>)`. This is a strictly better choice: `Arc` enables `Clone` for `PfiMetric::Custom` (satisfying the project's CLAUDE.md requirement for `Clone` on all public types) without requiring `Box`. The behavior is identical; the deviation is a quality improvement prompted by WR-01.

### Human Verification Required

None. All must-haves are mechanically verifiable and have been verified by running test commands.

---

_Verified: 2026-09-05T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
