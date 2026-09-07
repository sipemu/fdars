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
- ✅ **v0.38.0 — VEESA: Elastic Shape Explainability & Conformal Anomaly Detection** — Phases 72–74 (shipped 2026-09-05) — [archive](milestones/v0.38.0-ROADMAP.md)
- ✅ **v0.39.0 — DIFF: Differentiable FDA Core (Forward-Mode Autodiff)** — Phases 75–77 (shipped 2026-09-06) — [archive](milestones/v0.39.0-ROADMAP.md)
- ✅ **v0.40.0 — Correctness & Release Hardening** — Phases 78–80 (shipped 2026-09-07) — [archive](milestones/v0.40.0-ROADMAP.md)

## Phases

_No active milestone. v0.40.0 shipped 2026-09-07 — all prior milestone phase detail lives in `milestones/vX.Y.Z-ROADMAP.md`. Run `/gsd-new-milestone` to start the next milestone._

## Status

**v0.40.0 Correctness & Release Hardening — SHIPPED 2026-09-07 (release-ready; operator tag/publish pending).** Fixed the `soft_dtw_backward` zero-gradient bug + stabilized the barycenter optimizer (CORR-01), swept all hand-written gradient passes clean (CORR-02, Phase 78); repaired the `--features serde` build broken since Phase 60 (BUILD-01, Phase 79); Nyquist-signed-off phases 75/76/77 and bumped version/CHANGELOG/docs to 0.40.0 with all whole-crate gates green (REL-01/02, Phase 80). Milestone audit: 5/5 requirements satisfied, integration clean.

**Ship:** `fdars-core` is at version 0.40.0 and release-ready. The final step is operator-driven: `git tag v0.40.0` → `git push origin v0.40.0` → `release.yml` runs `cargo publish` (folds the unpublished v0.39.0 forward-mode AD core + these fixes into fdars' first crates.io release since v0.38.0). Backlog: SDTW-O1 (proper L-BFGS/multi-restart soft-DTW barycenter optimizer).

**Next:** `/gsd-new-milestone`

