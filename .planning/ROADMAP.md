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
- 🚧 **v0.39.0 — DIFF: Differentiable FDA Core (Forward-Mode Autodiff)** — Phases 75–77 (in progress) — promotes GAP-08 (last v0.31.0 backlog item)

## Phases

### 🚧 v0.39.0 — DIFF: Differentiable FDA Core (Forward-Mode Autodiff) — Phases 75–77

Promotes **GAP-08** (score 1.73, L-effort) — the last remaining item in the v0.31.0 `GAP-BACKLOG.md`. Adds an in-crate forward-mode automatic-differentiation core (a `Scalar` trait + `Dual<T>` number) and makes a scoped subset of FDA operations (**elastic distance + FPCA scores**) generic over the scalar type, so exact gradients flow through arbitrary compositions into optimization/ML pipelines. Reference baseline: the Julia generic-programming idiom (ElasticFDA.jl + ForwardDiff). Additive/non-breaking (existing f64 signatures untouched; generic versions live alongside — protects R + WASM bindings + 28 examples), **no new crate dependency** (in-crate dual numbers, forward-mode only). Ships to crates.io on the `v0.39.0` tag.

- [x] **Phase 75: Scalar Trait & Forward-Mode Dual Substrate** — DIF-01 — the `Scalar` trait + `Dual<T>` number every generic op is written against; foundational, blocks 76 & 77 (completed 2026-09-06)
- [ ] **Phase 76: Differentiable Elastic Distance & FPCA Scores** — DIF-02, DIF-03 — the two scoped ops made generic-over-`Scalar`; exact forward-mode gradients at `Dual`, f64 parity preserved — **2 plans** (Wave 1, parallel: disjoint code)
  - [ ] 76-01-PLAN.md — DIF-02: `soft_dtw_distance_generic` (pilot, vs oracle+FD+f64-parity) + `amplitude_distance_at_warp_generic` (fixed-warp SRSF; warp-searched `elastic_distance` deferred — non-differentiable DP argmin)
  - [ ] 76-02-PLAN.md — DIF-03: `project_scores_generic` FPCA score projection (analytic gradient rotation·weights ≤1e-12 + FD + f64-parity)
- [ ] **Phase 77: Gradient API, Composition Demo & Integration** — DIF-04 — ergonomic `(value, gradient)`/Jacobian entry point + end-to-end composition example + crate-root/prelude re-exports + module doctest

<details>
<summary>✅ v0.38.0 — VEESA: Elastic Shape Explainability & Conformal Anomaly Detection (Phases 72–74) — SHIPPED 2026-09-05</summary>

Closed the capability gaps against three Tucker-affiliated elastic-shape works (the VEESA paper, `sandialabs/veesa`, arXiv 2504.01172), additively and reuse-first with no new crate dependency. Shipped: a public jfPCA **fit → transform** seam (`jfpca_fit`/`JfpcaModel`/`.transform()`/`.score_training()`, 1e-8 training-score reproduction; VEE-01/02); a **VEESA explainability** layer — model-agnostic permutation feature importance over jfPCA scores (`elastic_pfi`/`PfiMetric`), principal-direction reconstruction (`principal_directions`/`PrincipalDirections`, c=0 reproduces the mean, σ=√eigenvalue), and an end-to-end `veesa_pipeline` (VEE-03/04/05); and an **inductive elastic conformal anomaly detector** — `NonConformityScore` extended with amplitude/phase/combined elastic variants, `elastic_nonconformity` + `elastic_conformal_anomaly` (p-values, flags, calibrated threshold catching magnitude AND shape outliers), `ConformalAnomalyResult` (ECA-01/02/03). Full crate-root + prelude re-exports + running doctests. Whole-crate **2821 lib + 203 doc tests green**, clippy `--all-targets` clean, fmt clean. Audit PASSED 8/8; integration SOUND. Full detail: [milestones/v0.38.0-ROADMAP.md](milestones/v0.38.0-ROADMAP.md).

- [x] Phase 72: jfPCA Fit/Transform Seam (1/1) — completed 2026-09-05
- [x] Phase 73: VEESA Explainability Pipeline & Integration (1/1) — completed 2026-09-05
- [x] Phase 74: Elastic Conformal Anomaly Detection (1/1) — completed 2026-09-05

**Ship step remaining (operator):** crate version bump 0.36.0 → 0.38.0 + `v0.38.0` git tag + crates.io publish (release.yml publishes on the tag). Deferred out of the autonomous run — the crate source is code-complete but still versioned 0.36.0.

</details>

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

## Phase Details

### Phase 75: Scalar Trait & Forward-Mode Dual Substrate

**Goal**: The numeric substrate exists — a `Scalar` trait plus a forward-mode `Dual<T>` number carrying value + tangent — that the differentiable subset is written against, with all arithmetic/transcendental ops the subset needs and gradient seed/extract helpers, verified against analytical derivatives.
**Depends on**: Nothing (first phase of the milestone)
**Requirements**: DIF-01
**Success Criteria** (what must be TRUE):

  1. A user can construct a `Dual` value, seed one input's tangent to 1, run a composed expression using ±, ×, ÷, `sqrt`, `exp`, `ln`, `sin`, `cos`, `powf`, `abs`, and partial comparisons, and extract both the value and the derivative.
  2. Dual arithmetic reproduces the analytical derivative of composed elementary functions to ≤1e-10 (known-answer tests).
  3. The `Scalar` trait is implemented for `f64`, so f64-instantiated generic code compiles and runs identically to the current numerics.
  4. The substrate adds no new crate dependency (in-crate dual numbers only) and existing f64 public signatures are untouched.

**Plans**: 1 plan

- [x] 75-01-PLAN.md — Scalar trait + forward-mode Dual substrate (op set, seed/extract, three-tier tests), registered in lib.rs, no new dependency

### Phase 76: Differentiable Elastic Distance & FPCA Scores

**Goal**: The two scoped FDA operations — elastic distance and FPCA score projection — are generic over `Scalar`, so at `Dual` they yield exact forward-mode gradients w.r.t. a curve's input values, while at `f64` they reproduce the existing numerics.
**Depends on**: Phase 75 (both ops are written against the DIF-01 `Scalar` substrate)
**Requirements**: DIF-02, DIF-03
**Success Criteria** (what must be TRUE):

  1. A user can take the forward-mode gradient of the elastic (soft-DTW / amplitude+phase) distance w.r.t. a curve's values; it matches central finite differences AND the existing hand-written `soft_dtw` gradient.
  2. The f64 instantiation of the elastic-distance path reproduces the current `elastic_distance` / `amplitude_distance` outputs within 1e-12.
  3. A user can take the forward-mode gradient of FPC scores w.r.t. input-curve values; it matches central finite differences.
  4. The f64 instantiation of the FPCA score projection reproduces the existing FPCA scores within tolerance.
  5. Both generic paths live alongside the existing f64 functions (additive/non-breaking) with no new crate dependency.

**Plans**: TBD

### Phase 77: Gradient API, Composition Demo & Integration

**Goal**: An ergonomic public gradient entry point over the `Scalar`-generic subset ships, together with a worked end-to-end composition example, full crate-root + prelude re-exports, and a running module doctest — proving AD flows through arbitrary compositions of the differentiable ops.
**Depends on**: Phase 76 (consumes the differentiable elastic-distance + FPCA-score ops)
**Requirements**: DIF-04
**Success Criteria** (what must be TRUE):

  1. A user can call one public `(value, gradient)` / directional-derivative / Jacobian entry point over the differentiable subset and get back both the objective value and its gradient.
  2. A worked example composes the differentiable ops into a single scalar objective and takes its gradient, demonstrating AD flowing through the composition end-to-end.
  3. The full differentiable surface (Scalar/Dual, generic ops, gradient API) is reachable via crate-root and prelude re-exports.
  4. The module doctest runs green under `cargo test --doc`.

**Plans**: TBD

## Progress

**Execution Order (v0.39.0):**
Phases execute in numeric order: 75 → 76 → 77. This is a hard dependency chain — DIF-01 (Phase 75) is the substrate DIF-02/DIF-03 (Phase 76) are generic over; DIF-04 (Phase 77) consumes the differentiable ops from Phase 76. Phase 75 must land first; Phase 77 must land last. Within Phase 76 the two ops (elastic distance, FPCA scores) are independent of each other (disjoint code areas: `metric/soft_dtw.rs`+`elastic_*` vs `regression.rs` FPCA) and may be planned/executed in either order.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 75. Scalar Trait & Forward-Mode Dual Substrate | v0.39.0 | 1/1 | Complete    | 2026-09-06 |
| 76. Differentiable Elastic Distance & FPCA Scores | v0.39.0 | 0/? | Not started | - |
| 77. Gradient API, Composition Demo & Integration | v0.39.0 | 0/? | Not started | - |

## Status

**v0.39.0 DIFF planning** (Phases 75–77) — promotes **GAP-08** (score 1.73, L-effort), the last remaining item in the v0.31.0 `GAP-BACKLOG.md`. Adds an in-crate forward-mode autodiff core (`Scalar` trait + `Dual<T>`) and makes elastic distance + FPCA scores generic over the scalar type, so exact gradients compose through op chains. Reuse-first pilot: `metric/soft_dtw.rs` (existing hand-written gradient to validate against), `elastic_*`, `regression.rs` FPCA. Additive/non-breaking (existing f64 signatures untouched; generic versions alongside — protects R + WASM bindings + 28 examples), no new crate dependency, forward-mode only. Deferred: reverse-mode/VJP (DIF-F1), broadening beyond elastic+FPCA (DIF-F2), refactoring f64 hot-path signatures to be generic (DIF-F3). Ships to crates.io on the `v0.39.0` tag.

**v0.38.0 VEESA shipped** (Phases 72–74, 2026-09-05) — closed the gaps against the VEESA paper + `sandialabs/veesa` R package + arXiv 2504.01172 (elastic conformal anomaly detection). Audit PASSED 8/8; integration SOUND; whole-crate 2821 lib + 203 doc tests green.

**Remaining operator ship steps:** bump + tag + publish for each un-shipped crate version (v0.29.0–v0.38.0 sit unreleased against the 0.28.0-published crate); each `v*` tag triggers `release.yml` → crates.io. The autonomous runs intentionally do not tag/publish.

Next: `/gsd-plan-phase 75`. After GAP-08 (this milestone), the v0.31.0 `GAP-BACKLOG.md` is exhausted — the following milestone will most likely be a crate-release-hardening / 1.0-readiness pass or a fresh audit.
