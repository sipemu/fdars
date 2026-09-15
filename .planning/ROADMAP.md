# Roadmap: fdars

## Milestones

- ✅ **v0.41.0 1.0 API Stabilization Pass** — Phases 81–85 (shipped 2026-09-07)
- ✅ **v0.42.0 1.0 API Finalization** — Phases 86–89 (shipped 2026-09-08)
- ✅ **v0.43.0 Test Determinism & Release Hardening** — Phases 90–93 (shipped 2026-09-09)
- ✅ **v0.44.0 Differentiable Core — Reverse-Mode & Broadened Subset** — Phases 94–100 (shipped 2026-09-12)
- ✅ **v0.45.0 Soft-DTW Barycenter — Global Optimizer (SDTW-O1)** — Phases 101–105 (shipped 2026-09-13)
- 🚧 **v0.46.0 Quality Hardening — pmat Complexity Pass & CI Guardrail** — Phases 106–111 (in progress)

## Overview

**v0.46.0 Quality Hardening — pmat Complexity Pass & CI Guardrail** is an extra quality cycle inserted before the remaining 1.0-checklist items (`fdars-j75`, `1.0-CUT`). A **pmat 3.18.0** audit (2026-09-13) surfaced concentrated technical debt in `fdars-core`: **120 complexity violations** (median cognitive 39, max cyclomatic 59 at `funfem_cluster`), **11 entropy violations** (repeated `DataTransformation` patterns), **2 SATD doc-comment false-positives**, and **1 critical `.unwrap()`**. This milestone drives `pmat quality-gate` to **PASS** (0 violations) by behavior-preservingly decomposing every hotspot into helpers, extracting the entropy patterns, rewording the SATD false-positives, and fixing the unwrap — then wires a **pmat CI guardrail** so debt can't silently regress.

**Governing contract:** strictly **non-breaking + numeric-preserving**. All 2942 lib tests + 215 doctests + 28 examples stay green; refactoring is pure internal decomposition (extract helpers), never a behavior or numeric change (the established crate norm — private impl bodies stay behavior-identical). `fdars-core` only (protects R + WASM bindings). **No new *crate* dependency** — pmat is a dev/CI tool invoked externally, never added to `Cargo.toml` (CI adds a pmat install step). Exit criteria = normal gates (`cargo fmt --check`, `clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, `--features serde` build guard, 28 examples + doctests, `cargo package`) **plus** `pmat quality-gate` passing.

With **API** (v0.42.0), **Quality** (v0.43.0), **differentiable-core** (v0.44.0, DIF-F1/F2/F3), and the **algorithm blocker `SDTW-O1`** (v0.45.0) all cleared on `documentation/ROADMAP-TO-1.0.md`, the only remaining 1.0 blockers are **fdars-j75** (R ecosystem — `fdars-r` FdMatrix migration) and the terminal **1.0-CUT**. This milestone does **not** touch either — it is an extra quality cycle before them.

## Phases

**Phase Numbering:**

- Integer phases (106, 107, …): Planned milestone work (continues from v0.45.0 — no reset)
- Decimal phases (106.1, 106.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 106: Top-Level Complexity Hotspots** - Decompose the 8 worst-ranked functions under pmat thresholds (QUAL-01) (completed 2026-09-14)
- [x] **Phase 107: SPM Complexity Cluster** - Decompose the dense `src/spm/` cluster under pmat thresholds (QUAL-02) (completed 2026-09-14)
- [x] **Phase 108: Entropy Extraction & SATD/Unwrap Cleanup** - Extract repeated patterns; reword SATD false-positives; fix the critical unwrap (QUAL-04 + QUAL-05) (completed 2026-09-14)
- [ ] **Phase 109: Complexity Tail & Quality-Gate Green** - Clear all remaining complexity violations so `pmat quality-gate` reports 0 findings (QUAL-03)
- [ ] **Phase 110: pmat CI Guardrail** - Add a GitHub Actions pmat quality-gate job that fails the build if violations return (CI-01)
- [ ] **Phase 111: Release Preparation & Verification** - Bump 0.45.0 → 0.46.0, CHANGELOG, all gates green incl. `pmat quality-gate` + `cargo package` (REL-01)

## Phase Details

### Phase 106: Top-Level Complexity Hotspots

**Goal**: Decompose the worst top-level complexity hotspots into smaller helper functions so each falls under pmat's cognitive/cyclomatic thresholds — behavior and numeric output unchanged.
**Depends on**: Nothing (first phase of this milestone; independent of 107/108)
**Requirements**: QUAL-01
**Success Criteria** (what must be TRUE):

  1. Each of `funfem_cluster` (cyc 59), `elastic_net_cd` (cog 54), `wnet_cv_lambda` (cog 45), `wnet` (cog 31), `boost_fofr` (cyc 39), `pace_fpca` (cyc 37), `bayesian_fosr` (cyc 36), and `kcfc_cluster` (cyc 28) is refactored into smaller helpers, each under pmat's cognitive/cyclomatic thresholds.
  2. `pmat analyze complexity` shows these 8 functions no longer appear as violations (the complexity count drops by the corresponding number of findings).
  3. The full `cargo test` suite (2942 lib + 215 doctests) passes with identical numeric output — no behavior or numeric drift.
  4. `cargo clippy --all-targets --features linalg,parallel -- -D warnings` and `cargo fmt --check` stay clean; the public API surface is unchanged (no signature or visibility change).

**Plans**: 4 plans

- [x] 106-01-PLAN.md — TRACER: decompose wavelet/regression.rs hotspots (elastic_net_cd, wnet_cv_lambda, wnet) below pmat thresholds
- [x] 106-02-PLAN.md — decompose clustering_advanced.rs hotspots (funfem_cluster, kcfc_cluster) below pmat thresholds
- [x] 106-03-PLAN.md — decompose boosting_regression hotspots (boost_fofr, bayesian_fosr) below pmat thresholds
- [x] 106-04-PLAN.md — decompose pace_fpca (BLUP closure extraction) below pmat thresholds

### Phase 107: SPM Complexity Cluster

**Goal**: Decompose the concentrated complexity cluster inside `src/spm/` into smaller helpers under pmat thresholds — behavior and numeric output unchanged.
**Depends on**: Nothing (independent of 106/108; may run in parallel)
**Requirements**: QUAL-02
**Success Criteria** (what must be TRUE):

  1. The SPM cluster — `mfpca`, `spm_phase1_iterative`, `spm_mewma_monitor`, `select_ncomp`, `conditional_expectation` (partial), `spm_monitor_from_fields` (phase), `profile_phase1`, `spe_multivariate` (stats) — is refactored so each falls under pmat's cognitive/cyclomatic thresholds.
  2. `pmat analyze complexity` shows the `src/spm/` functions no longer appear as violations.
  3. The full `cargo test` suite passes with identical numeric output — no behavior or numeric drift in any SPM code path.
  4. `cargo clippy --all-targets --features linalg,parallel -- -D warnings` and `cargo fmt --check` stay clean; the public SPM API surface is unchanged.

**Plans**: 4 plans

- [x] 107-01-PLAN.md — Tracer: amewma.rs + iterative.rs (spm_amewma_monitor cog 52, spm_phase1_iterative cog 42)
- [x] 107-02-PLAN.md — MFPCA cluster: mfpca.rs + ncomp.rs + contrib.rs (mfpca, select_ncomp, t2_contributions_mfpca, spe_contributions)
- [x] 107-03-PLAN.md — Monitors: ewma.rs + mewma.rs + stats.rs + phase.rs (spm_ewma_monitor, spm_mewma_monitor, spe_multivariate, spm_monitor_from_fields)
- [x] 107-04-PLAN.md — Remaining monitors: partial.rs + frcc.rs + profile.rs (conditional_expectation, frcc_phase1, profile_phase1)

### Phase 108: Entropy Extraction & SATD/Unwrap Cleanup

**Goal**: Extract the repeated `DataTransformation` patterns flagged by pmat entropy into shared helpers, reword the two SATD doc-comment false-positives, and replace the one critical `.unwrap()` with proper error handling — all behavior-preserving.
**Depends on**: Nothing (independent of the complexity refactors; may run early or in parallel)
**Requirements**: QUAL-04, QUAL-05
**Success Criteria** (what must be TRUE):

  1. The 11 repeated `DataTransformation` patterns (across `metric/tests.rs`, `fem_smoothing.rs`, `explain_generic/friedman.rs`, `seasonal/tests.rs`, `regression.rs`, `covariance.rs`, `detrend/tests.rs`, `scalar_on_function/tests.rs`, plus the project-level pattern-diversity finding) are extracted into shared helpers, and `pmat analyze entropy` reports 0 actionable violations.
  2. The 2 SATD doc-comment false-positives (`depth/dispatch.rs:281`, `metric/lp.rs:175`) are reworded so keyword matchers no longer flag them, and `pmat analyze satd` reports 0 violations.
  3. The critical `.unwrap()` at `depth/rpd.rs:229` is replaced with proper error handling (`?` or `.expect()` with a descriptive message), and `pmat analyze defects` reports 0 violations.
  4. The full `cargo test` suite passes with identical numeric output; clippy `--all-targets` and fmt stay clean; the public API surface is unchanged.

**Plans**: 4 plans

- [x] 108-01-PLAN.md — TRACER: reword 2 SATD doc-comment false-positives + fix the critical `.unwrap()` at rpd.rs:229 (satd→0, defects→0)
- [x] 108-02-PLAN.md — entropy in `#[cfg(test)]` modules: shared `#[cfg(test)]` helpers in test_helpers.rs + call-site substitutions
- [x] 108-03-PLAN.md — entropy in production files: private per-file helpers (fem_smoothing, friedman, soft_dtw, scalar_on_shape, classification/fit)
- [x] 108-04-PLAN.md — entropy in `///` doc-comment examples + STABLE 3× entropy 0 confirmation / escalation

### Phase 109: Complexity Tail & Quality-Gate Green

**Goal**: Resolve every remaining complexity violation across `fdars-core` so `pmat quality-gate` reports 0 findings of any kind — the milestone's central verifiable target.
**Depends on**: Phase 106, Phase 107, Phase 108 (the count is only verifiable once the hotspot, SPM, entropy, SATD, and defect work has landed)
**Requirements**: QUAL-03
**Success Criteria** (what must be TRUE):

  1. All remaining complexity violations (beyond the Phase 106/107 hotspots) are refactored into helpers under pmat thresholds — behavior and numeric output unchanged.
  2. `pmat analyze complexity` reports 0 complexity findings across `fdars-core`.
  3. `pmat quality-gate` reports 0 violations overall (0 complexity + 0 entropy + 0 SATD + 0 defect) — the gate PASSES.
  4. The full `cargo test` suite passes with identical numeric output; clippy `--all-targets` and fmt stay clean; the public API surface is unchanged.

**Plans**: 14 plans

- [x] 109-01-PLAN.md — TRACER: alignment fpns/closed/bayesian (4 fns) — prove extract→pmat→commit loop
- [x] 109-02-PLAN.md — alignment clustering/nd/generative (8 fns)
- [x] 109-03-PLAN.md — peer.rs (3 fns, heaviest cognitive scores)
- [x] 109-04-PLAN.md — clustering_advanced.rs (5 fns, incl. funfem_cluster residual)
- [x] 109-05-PLAN.md — coclustering/kernel_kmeans/kshape (6 fns)
- [x] 109-06-PLAN.md — helpers.rs shared infra (4 fns)
- [x] 109-07-PLAN.md — scalar_on_function multi/additive/fregre_lm (7 fns)
- [x] 109-08-PLAN.md — fpca_variants/famm/pace_fpca (7 fns, incl. pace_fpca_validate_config residual)
- [x] 109-09-PLAN.md — fof_regression + boosting gamlss/boost_fosr (5 fns)
- [x] 109-10-PLAN.md — gmm/subspace, density_fda, conformal/cv (6 fns)
- [x] 109-11-PLAN.md — smooth_basis, soft_dtw, fts/spectral, depth/dispatch (8 fns)
- [x] 109-12-PLAN.md — alignment singletons geodesic/lambda_cv/mod/multires/phase_boxplot/warp_stats (6 fns)
- [x] 109-13-PLAN.md — regression/depth/cv/2D-FOSR singleton batch (9 fns)
- [ ] 109-14-PLAN.md — FINAL: explainability/shapelet/simulation/etc. singleton batch (15 fns) + whole-crate `pmat quality-gate` → 0

### Phase 110: pmat CI Guardrail

**Goal**: Add a GitHub Actions CI job that installs pmat and runs `pmat quality-gate` against `fdars-core`, failing the build if violations return — mirroring the v0.43.0 `determinism-guardrail` pattern — so quality can't silently regress.
**Depends on**: Phase 109 (the gate must be green before it is enforced in CI, or it would fail the build on introduction)
**Requirements**: CI-01
**Success Criteria** (what must be TRUE):

  1. A new GitHub Actions job installs pmat and runs `pmat quality-gate` (and/or a complexity-threshold check) against `fdars-core`, failing the build if pmat violations are present.
  2. The job passes against the current (post-Phase-109) green tree, and would fail if a complexity/entropy/SATD/defect violation were reintroduced.
  3. The guardrail is documented (e.g., in the workflow and/or contributor docs) so contributors can reproduce the check locally.

**Plans**: TBD

### Phase 111: Release Preparation & Verification

**Goal**: Bump the crate to 0.46.0, record the milestone in both CHANGELOGs, and verify every whole-crate gate — including `pmat quality-gate` — is green, leaving the crate release-ready.
**Depends on**: Phase 106, Phase 107, Phase 108, Phase 109, Phase 110 (terminal — all prior work must be complete)
**Requirements**: REL-01
**Success Criteria** (what must be TRUE):

  1. The crate version is bumped 0.45.0 → 0.46.0 with a `[0.46.0]` entry in both CHANGELOGs (repository root + `fdars-core`).
  2. All whole-crate gates are green: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, the `--features serde` build guard, all 28 examples + doctests, and `cargo package`.
  3. `pmat quality-gate` passes on the release tree (0 violations) — the milestone's defining exit criterion.
  4. The `git tag v0.46.0` → crates.io publish is left as the DEFERRED operator step (GSD `git.create_tag` is off because `release.yml` couples tag-push to publish) — never performed inside a phase.

**Plans**: TBD

## Progress

**Execution Order:**
Phases 106, 107, and 108 are independent and may run in parallel (or in any order). Phase 109 depends on 106+107+108; Phase 110 depends on 109; Phase 111 (terminal) depends on all prior phases: (106, 107, 108) → 109 → 110 → 111.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 106. Top-Level Complexity Hotspots | 4/4 | Complete    | 2026-09-14 |
| 107. SPM Complexity Cluster | 4/4 | Complete    | 2026-09-14 |
| 108. Entropy Extraction & SATD/Unwrap Cleanup | 4/4 | Complete    | 2026-09-14 |
| 109. Complexity Tail & Quality-Gate Green | 0/TBD | Not started | - |
| 110. pmat CI Guardrail | 0/TBD | Not started | - |
| 111. Release Preparation & Verification | 0/TBD | Not started | - |

## Shipped Milestones

<details>
<summary>✅ v0.45.0 Soft-DTW Barycenter — Global Optimizer (SDTW-O1) (Phases 101–105) — SHIPPED 2026-09-13</summary>

Implementation milestone clearing the last in-core **algorithm-quality** item (`SDTW-O1`) on the 1.0 gap checklist. Replaced the single-init fixed-step MM/soft-DBA descent in `soft_dtw_barycenter` with a hand-rolled in-crate limited-memory **L-BFGS** optimizer (two-loop recursion, m=10 `VecDeque` history, Armijo backtracking line search + safeguards) driven by the exact soft-DTW gradient, wrapped in a **deterministically-seeded multi-restart** loop (pointwise mean + seeded perturbations, lowest-total-cost winner). Exposed via an additive `SoftDtwBarycenterConfig` + `soft_dtw_barycenter_with_config` entry point with the legacy 4-arg `soft_dtw_barycenter` preserved byte-for-byte as a thin non-breaking wrapper. Strictly additive/non-breaking, **no new crate dependency**, wasm32-safe. Full detail: [`milestones/v0.45.0-ROADMAP.md`](milestones/v0.45.0-ROADMAP.md); archived phase artifacts: `milestones/v0.45.0-phases/`.

- [x] Phase 101: In-Crate L-BFGS Barycenter Optimizer (1/1 plan) — completed 2026-09-13
- [x] Phase 102: Deterministic Multi-Restart Wrapper (1/1 plan) — completed 2026-09-13
- [x] Phase 103: `SoftDtwBarycenterConfig` API & Legacy Wrapper (1/1 plan) — completed 2026-09-13
- [x] Phase 104: Quality Evidence — Escapes-Local-Minimum & f64 Parity (1/1 plan) — completed 2026-09-13
- [x] Phase 105: Release Preparation & Verification (1/1 plan) — completed 2026-09-13

**Requirements:** 5/5 satisfied (SDTW-01/02/03/04, REL-01). Audit: passed; integration INTEGRATED. Whole-crate gates green (fmt, clippy `--all-targets`, 2942 tests, serde build, 215 doctests, 28 examples, `cargo package`). Release-ready — operator `git tag v0.45.0` → crates.io publish pending.

</details>

<details>
<summary>✅ v0.44.0 Differentiable Core — Reverse-Mode & Broadened Subset (Phases 94–100) — SHIPPED 2026-09-12</summary>

Implementation milestone completing the entire **differentiable-core** section of the 1.0 gap checklist (DIF-F1/F2/F3). Added a hand-written in-crate reverse-mode (VJP) tape (`Var`/Wengert-list, `vjp` entry point) alongside the v0.39.0 forward-mode `Dual`; broadened the differentiable operation set across four families; generalized targeted f64 hot-path signatures over the scalar type via defaulted type params (`T = f64`). Strictly additive/non-breaking, no new crate dependency. Full detail: [`milestones/v0.44.0-ROADMAP.md`](milestones/v0.44.0-ROADMAP.md); archived phase artifacts: `milestones/v0.44.0-phases/`.

- [x] Phase 94: Reverse-Mode Autodiff Core (VJP Tape) (4/4 plans) — completed 2026-09-11
- [x] Phase 95: Generic Scalar Hot-Path Signatures (3/3 plans) — completed 2026-09-11
- [x] Phase 96: Differentiable Basis Evaluation & Inner Products (3/3 plans) — completed 2026-09-11
- [x] Phase 97: Differentiable Regression Prediction & Smoothing Penalties (3/3 plans) — completed 2026-09-11
- [x] Phase 98: Differentiable Depth & Curve Distances (1/1 plan) — completed 2026-09-11
- [x] Phase 99: End-to-End Autodiff Flow & Gradient API (1/1 plan) — completed 2026-09-11
- [x] Phase 100: Release Preparation & Verification (1/1 plan) — completed 2026-09-11

**Requirements:** 11/11 satisfied (RAD-01/02/03, DOP-01/02/03/04, GEN-01/02, API-01, REL-01). Audit: passed; integration COMPLETE. Release-ready — operator `git tag v0.44.0` → crates.io publish pending.

</details>

<details>
<summary>✅ v0.43.0 Test Determinism & Release Hardening (Phases 90–93) — SHIPPED 2026-09-09</summary>

Quality + release-hardening milestone. Root-caused the long-standing `co_cluster`/`svd_sign` golden flake as a **deterministic faer-vs-nalgebra SVD backend divergence** and fixed it with a least-invasive test-side `#[cfg_attr(not(feature = "linalg"), ignore)]` guard. A suite-wide robustness audit found zero further fragile tests; a new `determinism-guardrail` CI job locks the green baseline. Cleared the **Quality** blocker on the 1.0 gap checklist. Full detail: [`milestones/v0.43.0-ROADMAP.md`](milestones/v0.43.0-ROADMAP.md); archived phase artifacts: `milestones/v0.43.0-phases/`.

- [x] Phase 90: Golden-Flake Root-Cause & Deterministic Fix (1/1 plan) — completed 2026-09-09
- [x] Phase 91: Suite-Wide Robustness Sweep (1/1 plan) — completed 2026-09-09
- [x] Phase 92: CI Determinism Guardrail (1/1 plan) — completed 2026-09-09
- [x] Phase 93: Release Preparation & Readiness Verification (1/1 plan) — completed 2026-09-09

**Requirements:** 7/7 satisfied (FLAKE-01/02, ROBUST-01/02, CI-01, REL-01/02). Audit: passed. Release-ready — operator `git tag v0.43.0` → crates.io publish pending.

</details>

<details>
<summary>✅ v0.42.0 1.0 API Finalization (Phases 86–89) — SHIPPED 2026-09-08</summary>

The second breaking milestone — clears the entire **API section** of the 1.0 gap checklist: sealed `wire` `pub(crate)`, `#[non_exhaustive]` on 38 config structs + the construction-idiom change, and the full `_1d`/`_2d` → `Dim`-dispatch naming unification. API-shape-only — no numeric/behavioral change. Full detail: [`milestones/v0.42.0-ROADMAP.md`](milestones/v0.42.0-ROADMAP.md); archived phase artifacts: `milestones/v0.42.0-phases/`.

- [x] Phase 86: Surface Sealing (1/1 plan) — completed 2026-09-08
- [x] Phase 87: Targeted Renames (1/1 plan) — completed 2026-09-08
- [x] Phase 88: Large Suffix Batch (1/1 plan) — completed 2026-09-08
- [x] Phase 89: Release Preparation & Verification (1/1 plan) — completed 2026-09-08

**Requirements:** 9/9 satisfied (SEAL-01/02/03, NAME-01/02/03/04/05, REL-01). Audit: passed. Release-ready — operator `git tag v0.42.0` → crates.io publish pending.

</details>

<details>
<summary>✅ v0.41.0 1.0 API Stabilization Pass (Phases 81–85) — SHIPPED 2026-09-07</summary>

First breaking milestone after a long additive-only run. Audit → user-approved breaking-change inventory → per-category cleanups → stability deliverables → release prep. Full detail: [`milestones/v0.41.0-ROADMAP.md`](milestones/v0.41.0-ROADMAP.md); archived phase artifacts: `milestones/v0.41.0-phases/`.

- [x] Phase 81: API Audit & Deprecated-Form Removal (2/2 plans) — completed 2026-09-07
- [x] Phase 82: Public-Surface Sealing & Non-Exhaustive Coverage (2/2 plans) — completed 2026-09-07
- [x] Phase 83: Naming Unification (3/3 plans) — completed 2026-09-07
- [x] Phase 84: Stability Deliverables (1/1 plan) — completed 2026-09-07
- [x] Phase 85: Release Preparation & Verification (1/1 plan) — completed 2026-09-07

**Requirements:** 9/9 satisfied (AUDIT-01, API-01/02/03/04, STAB-01/02/03, REL-01). Audit: passed, integration INTEGRATED.

</details>

**Shipped:** v0.14.0 → v0.45.0 (see the Milestones list above and the `milestones/` archives).
