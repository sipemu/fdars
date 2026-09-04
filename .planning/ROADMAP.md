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
- 🚧 **v0.37.0 — WAV: Wavelet-Domain Functional Regression** — Phases 69–71 (in progress) — promotes GAP-07

## Phases

**Active milestone: v0.37.0 WAV — Wavelet-Domain Functional Regression (Phases 69–71)**

- [x] **Phase 69: Discrete Wavelet Transform Primitive** — Forward/inverse orthogonal DWT (Haar + Daubechies db2–dbN), multi-level, periodic + symmetric boundaries, arbitrary lengths (completed 2026-09-04)
- [ ] **Phase 70: Wavelet-Domain Regressors (`wcr` + `wnet`)** — PCR/PLS and elastic-net scalar-on-function regression on wavelet coefficients (Gaussian response)
- [ ] **Phase 71: Prediction, Diagnostics & Integration** — Out-of-sample `predict` for both fits, accessors, crate-root/prelude re-exports, end-to-end doctest

<details>
<summary>✅ v0.36.0 — PEER: Structured-Penalty & Longitudinal Scalar-on-Function Regression (Phases 66–68) — SHIPPED 2026-09-04</summary>

- [x] Phase 66: Core PEER Estimator & Penalty Families (1/1 plans) — PER-01, PER-02 — completed 2026-09-04
- [x] Phase 67: Automatic λ Selection — GCV + REML (1/1 plans) — PER-03 — completed 2026-09-04
- [x] Phase 68: Longitudinal PEER, Prediction & Integration (1/1 plans) — PER-04, PER-05 — completed 2026-09-04

Shipped `peer()` (three penalty families: Ridge / 2nd-difference / caller-supplied Decree Q), automatic λ selection (GCV grid + self-contained REML EM), longitudinal `lpeer()` (subject random effects via `famm::fit_scalar_mixed_model`, structured penalty applied in FPC-score space), out-of-sample `predict`, and full crate-root + prelude exports with a running end-to-end doctest. Additive/non-breaking; crate 0.35.0 → 0.36.0. Audit PASSED 5/5. Full detail: [milestones/v0.36.0-ROADMAP.md](milestones/v0.36.0-ROADMAP.md).

</details>

## Phase Details

### Phase 69: Discrete Wavelet Transform Primitive

**Goal**: The crate can transform a signal (and back) via an orthogonal discrete wavelet transform — Haar (db1) and Daubechies db2–dbN, multi-level, with selectable boundary handling — as a reusable in-crate primitive that the wavelet-domain regressors build on.
**Depends on**: Nothing new (first phase of milestone; reuses only `error.rs`/`FdMatrix` infrastructure). Foundational — must precede Phases 70 and 71.
**Requirements**: WAV-01, WAV-02
**Success Criteria** (what must be TRUE):

  1. Forward→inverse round-trip reconstructs the input within numerical tolerance (e.g. ≤1e-10 relative) for Haar and Daubechies db2–dbN across multiple decomposition levels.
  2. The transform runs on arbitrary (non-power-of-2) signal lengths under both periodic and symmetric boundary modes without panicking, and both boundary modes independently satisfy the perfect-reconstruction round-trip.
  3. A known-answer Haar single-level decomposition matches the hand-computed (sum/difference over √2) approximation and detail coefficients.
  4. Invalid inputs (unsupported family/order, empty signal, level exceeding max decomposable depth) return a descriptive `FdarError` rather than panicking or producing NaN.

**Plans**: 2 plans

- [x] 69-01-PLAN.md — Wavelet module + db1–db10 filter tables + single-level analysis/synthesis engine (periodic + symmetric, arbitrary length); Haar known-answer + single-level round-trip gates
- [x] 69-02-PLAN.md — Multi-level Mallat pyramid (decompose/reconstruct, auto/explicit level), WaveletCoeffs result struct, FdMatrix batch path; full round-trip + non-power-of-2 + invalid-input gates

### Phase 70: Wavelet-Domain Regressors (`wcr` + `wnet`)

**Goal**: Users can fit two wavelet-domain scalar-on-function regressors on Gaussian responses — `wcr` (PCR/PLS in wavelet-coefficient space) and `wnet` (elastic-net with cross-validated λ) — each transforming curves to wavelet coefficients via the Phase 69 DWT, then reusing fdars' existing FPCR/PLS and coordinate-descent machinery.
**Depends on**: Phase 69 (both regressors consume the DWT primitive). `wcr` and `wnet` are independent of each other.
**Requirements**: WAV-03, WAV-04
**Success Criteria** (what must be TRUE):

  1. `wcr` fits via both PCR and PLS on wavelet coefficients and recovers a known coefficient function β(t) on synthetic data (spanning, full-rank predictor design) within tolerance; the result struct carries β(t), intercept, and fitted values.
  2. `wnet` recovers a sparse coefficient pattern on synthetic data where the true signal is localized in a few wavelet coefficients — the selected/nonzero coefficients concentrate on the true support — and its result struct carries β(t), intercept, fitted values, and the selected coefficients.
  3. `wnet` cross-validated λ selection is deterministic across runs and, at the selected λ, yields a non-degenerate fit that tracks the injected signal (β(t) recovery on synthetic SNR data).
  4. Both regressors validate inputs (dimension/parameter mismatches → descriptive `FdarError`, never panic) and produce finite, NaN-free β(t) and fitted values.

**Plans**: 2 plans
- [ ] 70-01-PLAN.md — `wcr` (PCR + PLS) on wavelet coefficients + shared curves→coeff-design & β(t)-reconstruction helpers (WAV-03)
- [ ] 70-02-PLAN.md — `wnet` per-coefficient elastic-net CD adapter + deterministic CV-λ + sparse-support recovery (WAV-04)

### Phase 71: Prediction, Diagnostics & Integration

**Goal**: Both fitted regressors predict on new curves and expose their coefficient function and fitted values; the full public wavelet surface (DWT + `wcr` + `wnet` + config/result types + `predict`) is reachable from the crate root and prelude, with a running end-to-end doctest — all additive and non-breaking.
**Depends on**: Phase 70 (prediction and exports consume the fitted `wcr`/`wnet` results). Final phase of the chain.
**Requirements**: WAV-05, WAV-06
**Success Criteria** (what must be TRUE):

  1. `predict` is self-consistent for both `wcr` and `wnet`: re-passing the training curves reproduces the training fitted values within tolerance, and prediction on new curves returns finite values.
  2. Coefficient-function and fitted-value accessors return the expected outputs from both fitted result structs.
  3. The full public surface (DWT forward/inverse, `wcr`, `wnet`, config/result types, `predict`) is reachable via both the crate root and `prelude`, and a running end-to-end module doctest passes under `cargo test --doc`.
  4. The change is additive/non-breaking — no existing public signature changes; R + WASM bindings and all 28 examples remain unaffected (whole-crate `cargo test` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check` green).

**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 69. Discrete Wavelet Transform Primitive | 2/2 | Complete    | 2026-09-04 |
| 70. Wavelet-Domain Regressors (`wcr` + `wnet`) | 0/? | Not started | - |
| 71. Prediction, Diagnostics & Integration | 0/? | Not started | - |

## Status

Active milestone **v0.37.0 WAV** (Phases 69–71) — promotes GAP-07 (wavelet-domain functional regression) from `.planning/research/GAP-BACKLOG.md`. Strict dependency chain: DWT primitive (69) → wavelet-domain regressors `wcr`/`wnet` (70) → prediction + integration (71). Implementation milestone; publishes to crates.io on the `v0.37.0` tag (crate bump 0.36.0 → 0.37.0). After GAP-07, only GAP-08 (differentiable core) remains in the backlog.

Next: `/gsd-plan-phase 69`
