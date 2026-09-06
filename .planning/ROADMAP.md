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

## Phases

<details>
<summary>✅ v0.39.0 — DIFF: Differentiable FDA Core (Forward-Mode Autodiff) (Phases 75–77) — SHIPPED 2026-09-06</summary>

Promoted **GAP-08** (score 1.73, L-effort) — **the last item in the v0.31.0 `GAP-BACKLOG.md`, now exhausted.** Added an in-crate forward-mode automatic-differentiation core (`Scalar` trait + `Dual{value,tangent}`) and made a scoped subset of FDA ops (**elastic distance + FPCA scores**) generic over the scalar type, so exact gradients compose through arbitrary op chains. Additive/non-breaking (existing f64 signatures untouched; generic versions alongside), **no new crate dependency** (in-crate dual numbers, forward-mode only). Milestone audit: 4/4 requirements, integration INTEGRATED. Whole-crate gates green (2859 lib + 209 doc tests, clippy `--all-targets` clean, fmt clean). Full detail: [milestones/v0.39.0-ROADMAP.md](milestones/v0.39.0-ROADMAP.md).

- [x] Phase 75: Scalar Trait & Forward-Mode Dual Substrate (1/1) — DIF-01 — completed 2026-09-06
- [x] Phase 76: Differentiable Elastic Distance & FPCA Scores (2/2) — DIF-02, DIF-03 — completed 2026-09-06
- [x] Phase 77: Gradient API, Composition Demo & Integration (1/1) — DIF-04 — completed 2026-09-06

**Scope note:** the warp-*searched* `elastic_distance` was intentionally NOT made differentiable (its discrete DP argmin is piecewise-constant → non-differentiable) — DIF-02 delivered the differentiable `soft_dtw_distance_generic` + fixed-warp `amplitude_distance_at_warp_generic` instead. A **pre-existing** `soft_dtw_backward` zero-gradient bug was discovered and logged to backlog (not this milestone's defect; left unfixed under additive scope). Nyquist VALIDATION.md files remain `draft` (run `/gsd-validate-phase 75|76|77` to reconcile).

**Ship step remaining (operator):** crate version bump → 0.39.0 + `v0.39.0` git tag + crates.io publish (release.yml publishes on the tag). Deferred out of the autonomous run — crate source is code-complete but still versioned 0.38.0 in Cargo.toml.

</details>

<details>
<summary>✅ v0.38.0 — VEESA: Elastic Shape Explainability & Conformal Anomaly Detection (Phases 72–74) — SHIPPED 2026-09-05</summary>

Closed the capability gaps against three Tucker-affiliated elastic-shape works (the VEESA paper, `sandialabs/veesa`, arXiv 2504.01172), additively and reuse-first with no new crate dependency: a public jfPCA fit→transform seam (VEE-01/02), a model-agnostic VEESA explainability layer (`elastic_pfi`, `principal_directions`, `veesa_pipeline`; VEE-03/04/05), and an inductive elastic conformal anomaly detector (`elastic_nonconformity`, `elastic_conformal_anomaly`, `ConformalAnomalyResult`; ECA-01/02/03). Whole-crate 2821 lib + 203 doc tests green. Audit PASSED 8/8; integration SOUND. Full detail: [milestones/v0.38.0-ROADMAP.md](milestones/v0.38.0-ROADMAP.md).

</details>

<details>
<summary>✅ v0.37.0 — WAV: Wavelet-Domain Functional Regression (Phases 69–71) — SHIPPED 2026-09-04</summary>

Shipped a from-scratch in-crate discrete wavelet transform (`wavelet/`: Haar + Daubechies db2–db10, multi-level Mallat pyramid, periodic + symmetric boundaries, perfect reconstruction ≤1e-10) plus `wcr` (PCR/PLS in wavelet space) and `wnet` (per-coefficient elastic-net) scalar-on-function regressors. Promotes GAP-07. Audit PASSED 6/6. Full detail: [milestones/v0.37.0-ROADMAP.md](milestones/v0.37.0-ROADMAP.md).

</details>

Earlier milestones (v0.14.0–v0.36.0) are shipped and archived — see the Milestones list above and `milestones/`.

## Status

**v0.39.0 DIFF shipped** (Phases 75–77, 2026-09-06) — promoted **GAP-08**, the last item in the v0.31.0 `GAP-BACKLOG.md`. **All parity/gap backlogs (scikit-fda, R core, multi-ecosystem) are now exhausted.** An in-crate forward-mode autodiff core (`Scalar`/`Dual`, `grad`/`jacobian`/`directional_derivative`) with a scoped differentiable subset (elastic soft-DTW + amplitude-at-warp + FPCA scores) whose gradients compose end-to-end; additive/non-breaking, no new crate dependency.

**Remaining operator ship steps:** bump + tag + publish for each un-shipped crate version — the crate source is code-complete through v0.39.0 but Cargo.toml is versioned 0.38.0; each `v*` tag triggers `release.yml` → crates.io. The autonomous runs intentionally do not tag/publish.

**Next:** `/gsd-new-milestone`. With the gap backlogs exhausted, strong candidates are a crate-release-hardening / 1.0-readiness pass (bump/tag/publish, remove the v0.30.0 `#[deprecated]` forms, fix the pre-existing `soft_dtw_backward` bug + the serde build break), or promoting a deferred item (DIF-F1 reverse-mode AD, DIF-F2 broaden the differentiable subset).
