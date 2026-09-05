# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (shipped 2026-08-11) — [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 — Elastic Feasibility + Parity Quick Wins** — Phases 12–13 (shipped 2026-08-12, PR #40) — [archive](milestones/v0.16.0-ROADMAP.md)
- ✅ **v0.17.0 — Registration Parity & Elastic-FPCA Performance** — Phases 14–15 (shipped 2026-08-12, PR #41) — [archive](milestones/v0.17.0-ROADMAP.md)
- ✅ **v0.18.0 — R-Ecosystem Gap Audit** — Phases 16–19 (shipped 2026-08-15) — [archive](milestones/v0.18.0-ROADMAP.md)
- ✅ **v0.19.0 — Functional Inference Suite** — Phases 20–21 (shipped 2026-08-16) — [archive](milestones/v0.19.0-ROADMAP.md)
- ✅ **v0.20.0 — Table-Stakes Quick Wins** — Phases 22–23 (shipped 2026-08-16) — [archive](milestones/v0.20.0-ROADMAP.md)
- ✅ **v0.21.0 — Functional Regression Completeness** — Phases 24–25 (shipped 2026-08-17) — [archive](milestones/v0.21.0-ROADMAP.md)
- ✅ **v0.22.0 — PACE Sparse FPCA & Elastic Multinomial** — Phases 26–27 (shipped 2026-08-19) — [archive](milestones/v0.22.0-ROADMAP.md)
- ✅ **v0.23.0 — Depth, Outliers & Interval Inference** — Phases 28–30 (shipped 2026-08-20) — [archive](milestones/v0.23.0-ROADMAP.md)
- ✅ **v0.24.0 — Functional Regression & Clustering Breadth** — Phases 31–33 (shipped 2026-08-20) — [archive](milestones/v0.24.0-ROADMAP.md)
- ✅ **v0.25.0 — Serial Dependence, Representation & Density Breadth** — Phases 34–36 (shipped 2026-08-21) — [archive](milestones/v0.25.0-ROADMAP.md)
- ✅ **v0.26.0 — FPCA Breadth & Sparse Covariance** — Phases 37–38 (shipped 2026-08-21) — [archive](milestones/v0.26.0-ROADMAP.md)
- ✅ **v0.27.0 — Functional Time Series & Fréchet Regression** — Phases 39–40 (shipped 2026-08-22) — [archive](milestones/v0.27.0-ROADMAP.md)
- ✅ **v0.28.0 — Spectral Functional Time Series & Object-Data Fréchet Regression** — Phases 41–42 (shipped 2026-08-23) — [archive](milestones/v0.28.0-ROADMAP.md)
- ✅ **v0.29.0 — Boosting/Bayesian Regression, FEM/PDE Smoothing & Functional Co-Clustering** — Phases 43–45 (shipped 2026-08-30) — [archive](milestones/v0.29.0-ROADMAP.md)
- ✅ **v0.30.0 — Performance & Consolidation Pass** — Phases 46–51 (shipped 2026-09-01) — [archive](milestones/v0.30.0-ROADMAP.md)
- ✅ **v0.31.0 — Multi-Ecosystem Gap Audit** — Phases 52–53 (shipped 2026-09-02) — [archive](milestones/v0.31.0-ROADMAP.md)
- ✅ **v0.32.0 — Global Alignment Kernel & Kernel Clustering** — Phases 54–56 (shipped 2026-09-02) — [archive](milestones/v0.32.0-ROADMAP.md)
- ✅ **v0.33.0 — Shapelet Transform & Classification** — Phases 57–60 (shipped 2026-09-02) — [archive](milestones/v0.33.0-ROADMAP.md)
- ✅ **v0.34.0 — k-Shape Clustering & Shape-Based Distance** — Phases 61–63 (shipped 2026-09-02) — [archive](milestones/v0.34.0-ROADMAP.md)
- ✅ **v0.35.0 — Optimal Experimental Design for Sparse FDA (FOptDes)** — Phases 64–65 (shipped 2026-09-03) — [archive](milestones/v0.35.0-ROADMAP.md)
- ✅ **v0.36.0 — PEER: Structured-Penalty & Longitudinal Scalar-on-Function Regression** — Phases 66–68 (shipped 2026-09-04) — [archive](milestones/v0.36.0-ROADMAP.md)
- ✅ **v0.37.0 — WAV: Wavelet-Domain Functional Regression** — Phases 69–71 (shipped 2026-09-04) — [archive](milestones/v0.37.0-ROADMAP.md)
- 🚧 **v0.38.0 — VEESA: Elastic Shape Explainability & Conformal Anomaly Detection** — Phases 72–74 (in progress)

## Phases

### 🚧 v0.38.0 — VEESA: Elastic Shape Explainability & Conformal Anomaly Detection (Phases 72–74) — IN PROGRESS

**Milestone Goal:** Close the specific capability gaps between fdars and three Tucker-affiliated elastic-shape-analysis works — the VEESA paper (Goode, Tucker & Ries, *J. Data Science*), the R package `sandialabs/veesa`, and arXiv 2504.01172 (elastic conformal anomaly detection) — additively and reuse-first, on top of fdars' existing jfPCA / elastic-distance / conformal machinery. Real `fdars-core/src/` code, additive/non-breaking (protects R + WASM bindings + 28 examples), **no new crate dependency**, normal test/clippy (`--all-targets --features linalg,parallel`)/fmt gates. Crate ships on the `v0.38.0` tag (deferred operator step, per the release-decoupling convention). Phase numbering continues from v0.37.0 (…71) → Phase 72 onward.

- [x] **Phase 72: jfPCA Fit/Transform Seam** - Public jfPCA fit transformer + out-of-sample projection onto the trained joint-FPCA basis (completed 2026-09-05)
- [x] **Phase 73: VEESA Explainability Pipeline & Integration** - Model-agnostic PFI over jfPCA scores + principal-direction reconstruction, wired end-to-end with re-exports + doctest (completed 2026-09-05)
- [ ] **Phase 74: Elastic Conformal Anomaly Detection** - Elastic nonconformity scores + inductive conformal p-values/flags catching magnitude and shape outliers

#### Phase 72: jfPCA Fit/Transform Seam

**Goal**: Users can fit a reusable jfPCA transformer on training curves and project *new* out-of-sample curves onto the *trained* joint-FPCA basis, in the trained coordinate system.
**Depends on**: Phase 71 (prior milestone); builds on shipped `elastic_fpca.rs` (`joint_fpca`/`vert_fpca`/`horiz_fpca`, `JointFpcaResult`, private `project_onto_eigenvectors`)
**Requirements**: VEE-01, VEE-02
**Success Criteria** (what must be TRUE):

  1. A public jfPCA fit step returns a reusable transformer (a model object or `JointFpcaResult` extension) that stores the trained Karcher-mean template, `mean_psi`, joint eigenvector components (`vert_component`/`horiz_component`), `balance_c`, and `argvals`.
  2. Training scores produced by the fit reproduce `joint_fpca`'s scores within 1e-8.
  3. A public out-of-sample transform (`prep_testing_data` equivalent) aligns new raw curves to the trained Karcher-mean template and projects them onto the trained joint-FPCA basis, returning scores in the trained coordinate system.
  4. Transforming the original training curves through the transform reproduces the training scores within tolerance (fit→transform round-trip), reusing the existing private `project_onto_eigenvectors` now exposed via this seam.
  5. All new public surface is additive/non-breaking (existing signatures unchanged); whole-crate `cargo test`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, and `cargo fmt --check` pass.

**Plans**: 1 plan

Plans:

- [x] 72-01-PLAN.md — jfPCA fit/transform seam: JfpcaModel + jfpca_fit + .transform() (tracer → numerical gates → error paths → doctest/full gates)

#### Phase 73: VEESA Explainability Pipeline & Integration

**Goal**: Users can explain any trained predictor over jfPCA scores via permutation feature importance and reconstruct interpretable principal directions, driven end-to-end from fit → transform → PFI.
**Depends on**: Phase 72 (consumes the fit/transform seam)
**Requirements**: VEE-03, VEE-04, VEE-05
**Success Criteria** (what must be TRUE):

  1. Model-agnostic permutation feature importance (PFI) computes per-PC importance for a caller-supplied trained predictor (generic over a predictor trait / scoring closure, not tied to `elastic_pcr`) by permuting each PC-score column and measuring degradation in a user-selected error metric, reusing `elastic_explain.rs` permutation machinery.
  2. PFI is deterministic under a seed, and on a known-signal design the informative PC ranks above the noise PCs.
  3. Principal-direction reconstruction returns functions at μ ± c·σⱼ along each joint PC, split into amplitude (warped-function) and phase (warping-function) parts suitable for plotting; `c = 0` reproduces the jfPCA mean function.
  4. An end-to-end convenience path ties fit (align + jfPCA) → out-of-sample transform → PFI, with the full public surface re-exported at crate root and in the prelude.
  5. A running end-to-end module doctest passes under `cargo test --doc`; whole-crate `cargo test`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, and `cargo fmt --check` pass; additive/non-breaking.

**Plans**: 1 plan
**UI hint**: yes

Plans:

- [ ] 73-01-PLAN.md — Model-agnostic PFI (VEE-03), principal-direction reconstruction (VEE-04), and end-to-end veesa_pipeline + re-exports + doctest (VEE-05)

#### Phase 74: Elastic Conformal Anomaly Detection

**Goal**: Users can flag both magnitude and shape outliers in functional data via an inductive conformal anomaly detector that scores curves against a reference template using elastic distance.
**Depends on**: Nothing within this milestone (independent of the VEE group; sequenced last for a clean linear execution order). Builds on shipped `tolerance/conformal.rs` (`conformal_prediction_band`, `NonConformityScore`) and `alignment/` distances.
**Requirements**: ECA-01, ECA-02, ECA-03
**Success Criteria** (what must be TRUE):

  1. `NonConformityScore` (currently `{SupNorm, L2}`) is extended with elastic variants — amplitude distance, phase distance, and combined elastic distance — each scoring a curve against a reference template, non-negative and zero for a curve identical to the reference (reusing `amplitude_distance`/`phase_distance`/`elastic_distance`).
  2. An inductive conformal anomaly detector calibrates elastic nonconformity scores on a reference/calibration set (against a template such as the calibration Karcher mean), then returns per-curve conformal p-values and anomaly flags at a chosen level α.
  3. On exchangeable clean data the flag rate is approximately α (marginal validity), and injected magnitude and shape outliers are flagged.
  4. A `ConformalAnomalyResult` carries per-curve conformal p-values, nonconformity scores, boolean flags, and the calibrated threshold; the full surface is re-exported at crate root and in the prelude with a running module doctest.
  5. Additive/non-breaking — the existing `conformal_prediction_band` path is unchanged; whole-crate `cargo test`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, and `cargo fmt --check` pass.

**Plans**: TBD

Plans:

- [ ] 74-01: TBD

<details>
<summary>✅ v0.37.0 — WAV: Wavelet-Domain Functional Regression (Phases 69–71) — SHIPPED 2026-09-04</summary>

- [x] Phase 69: Discrete Wavelet Transform Primitive (2/2 plans) — WAV-01, WAV-02 — completed 2026-09-04
- [x] Phase 70: Wavelet-Domain Regressors (`wcr` + `wnet`) (2/2 plans) — WAV-03, WAV-04 — completed 2026-09-04
- [x] Phase 71: Prediction, Diagnostics & Integration (1/1 plans) — WAV-05, WAV-06 — completed 2026-09-04

Shipped a from-scratch in-crate discrete wavelet transform (`wavelet/`: Haar + Daubechies db2–db10, multi-level Mallat pyramid, periodic + symmetric boundaries, arbitrary lengths, perfect reconstruction ≤1e-10) plus two Gaussian-response wavelet-domain scalar-on-function regressors — `wcr` (PCR/PLS in wavelet-coefficient space) and `wnet` (per-coefficient elastic-net with deterministic cross-validated λ and sparse-support recovery) — with out-of-sample `predict`, β(t)/fitted accessors, and the full 14-symbol surface re-exported at crate root + prelude with a running end-to-end module doctest. Additive/non-breaking, no new dependency; whole-crate 2794 lib + 199 doc tests green, clippy `--all-targets` clean. Audit PASSED 6/6; integration SOUND. Promotes GAP-07. Full detail: [milestones/v0.37.0-ROADMAP.md](milestones/v0.37.0-ROADMAP.md).

**Ship step remaining (operator):** crate version bump 0.36.0 → 0.37.0 + `v0.37.0` git tag + crates.io publish (release.yml publishes on the tag). Deferred out of the autonomous run — the crate source is code-complete but still versioned 0.36.0.

</details>

<details>
<summary>✅ v0.36.0 — PEER: Structured-Penalty & Longitudinal Scalar-on-Function Regression (Phases 66–68) — SHIPPED 2026-09-04</summary>

- [x] Phase 66: Core PEER Estimator & Penalty Families (1/1 plans) — PER-01, PER-02 — completed 2026-09-04
- [x] Phase 67: Automatic λ Selection — GCV + REML (1/1 plans) — PER-03 — completed 2026-09-04
- [x] Phase 68: Longitudinal PEER, Prediction & Integration (1/1 plans) — PER-04, PER-05 — completed 2026-09-04

Shipped `peer()` (three penalty families: Ridge / 2nd-difference / caller-supplied Decree Q), automatic λ selection (GCV grid + self-contained REML EM), longitudinal `lpeer()` (subject random effects via `famm::fit_scalar_mixed_model`, structured penalty applied in FPC-score space), out-of-sample `predict`, and full crate-root + prelude exports with a running end-to-end doctest. Additive/non-breaking; crate 0.35.0 → 0.36.0. Audit PASSED 5/5. Full detail: [milestones/v0.36.0-ROADMAP.md](milestones/v0.36.0-ROADMAP.md).

</details>

## Progress

**Execution Order (v0.38.0):**
Phases execute in numeric order: 72 → 73 → 74. Phase 73 depends on the Phase 72 seam; Phase 74 (ECA) is independent of the VEE group and may be sequenced anywhere, placed last here for a clean linear order.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 72. jfPCA Fit/Transform Seam | v0.38.0 | 1/1 | Complete    | 2026-09-05 |
| 73. VEESA Explainability Pipeline & Integration | v0.38.0 | 1/1 | Complete    | 2026-09-05 |
| 74. Elastic Conformal Anomaly Detection | v0.38.0 | 0/TBD | Not started | - |

## Status

**v0.38.0 VEESA in progress** (Phases 72–74) — closes the gaps against the VEESA paper + `sandialabs/veesa` R package + arXiv 2504.01172 (elastic conformal anomaly detection). Reuse-first on shipped `elastic_fpca.rs` / `alignment/` / `elastic_explain.rs` / `tolerance/conformal.rs`; no new crate dependency. VEE group (Phases 72–73) is a dependency chain; the ECA group (Phase 74) is independent.

**v0.37.0 WAV shipped** (Phases 69–71) — promoted GAP-07 (wavelet-domain functional regression) from `.planning/research/GAP-BACKLOG.md`. After GAP-07, only **GAP-08** (differentiable / autodiff-compatible FDA core — invasive generics refactor, score 1.73, L-effort) remains in the backlog.

**Remaining operator ship steps:** bump + tag + publish for each un-shipped crate version (v0.29.0–v0.38.0 sit unreleased against the 0.28.0-published crate); each `v*` tag triggers `release.yml` → crates.io. The autonomous runs intentionally do not tag/publish.

Next: `/gsd-plan-phase 72`.
