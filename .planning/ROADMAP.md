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

## Phases

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

## Status

**v0.37.0 WAV shipped** (Phases 69–71) — promoted GAP-07 (wavelet-domain functional regression) from `.planning/research/GAP-BACKLOG.md`. After GAP-07, only **GAP-08** (differentiable / autodiff-compatible FDA core — invasive generics refactor, score 1.73, L-effort) remains in the backlog.

**Remaining operator ship step for v0.37.0:** bump crate `fdars-core` 0.36.0 → 0.37.0, tag `v0.37.0`, push → `release.yml` publishes to crates.io. (The autonomous run intentionally did not tag/publish; a `v0.37.0` tag on the un-bumped 0.36.0 crate would trigger a version-mismatched publish.)

Next: `/gsd-new-milestone` (candidate: promote GAP-08, or a new gap audit).
