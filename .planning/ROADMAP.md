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
- 🚧 **v0.40.0 — Correctness & Release Hardening** — Phases 78–80 (in progress)

## Phases

### 🚧 v0.40.0 — Correctness & Release Hardening (In Progress)

**Milestone Goal:** Fix the correctness bugs and build breakage discovered during recent milestones, formally validate the outstanding v0.39.0 phases, then bump/tag/publish — folding the unpublished v0.39.0 forward-mode AD core plus these fixes into fdars' first crates.io release since v0.38.0. Implementation milestone — real `fdars-core/src/` changes scoped to fixes/hardening (no new algorithms); additive/non-breaking (protects R + WASM bindings + 28 examples); no new crate dependency; behavior-preserving except where correcting the acknowledged `soft_dtw` bug. Real code → this milestone **does** get a `v0.40.0` git tag. Phase numbering continues from v0.39.0 (…77) → **Phase 78 onward**.

- [x] **Phase 78: Gradient Correctness — soft_dtw Fix & Backward-Pass Audit** - Fix the `soft_dtw_backward` endpoint-seed bug, add a regression test, and sweep every hand-written gradient pass for analogous boundary-seed defects (CORR-01, CORR-02) (completed 2026-09-07)
- [x] **Phase 79: Serde Feature Repair** - Restore `cargo build --features serde` by adding conditional serde derives to `ClassifFit` and embedded types, with a CI guard against re-breakage (BUILD-01) (completed 2026-09-07)
- [x] **Phase 80: Release Hardening & Ship v0.40.0** - Nyquist sign-off of phases 75/76/77, crate bump 0.38.0 → 0.40.0, CHANGELOG + docs refresh, all whole-crate gates green — release-ready for the operator `v0.40.0` tag → crates.io publish (REL-01, REL-02) (completed 2026-09-07)

## Phase Details

### Phase 78: Gradient Correctness — soft_dtw Fix & Backward-Pass Audit

**Goal**: Every hand-written backward/gradient pass in the crate produces a correct (non-zero, boundary-seeded) gradient — starting with the concrete `soft_dtw_backward` endpoint-seed fix and extending to a full audited sweep of the sibling gradient passes.
**Depends on**: Nothing (independent of Phase 79)
**Requirements**: CORR-01, CORR-02
**Success Criteria** (what must be TRUE):

  1. `soft_dtw_backward` no longer overwrites the `E[n][m]=1.0` endpoint seed — on non-identical input it returns a non-zero soft-alignment matrix, and a new regression test asserts both a non-zero `soft_dtw_backward`/`soft_dtw_accumulate_gradient` gradient AND that `soft_dtw_barycenter` on non-identical curves converges to a barycenter measurably different from the pointwise mean, cross-checked against the v0.39.0 `Dual` path / `corrected_oracle_gradient` reference within tolerance.
  2. The existing `test_soft_dtw_barycenter_*` tests are tightened so they can no longer pass on an all-zero gradient.
  3. Every hand-written backward/gradient pass named in CORR-02 (`alignment/differentiable`, `autodiff`, `boosting_regression/gamlss`, `elastic_regression/logistic`, `explain_generic/counterfactual`, `regression`, `seasonal/mod`, `smooth_basis`, `metric/soft_dtw`) is audited and given a "clean" (with a one-line rationale) or "fixed" (with a regression test) disposition, all traceable in the phase artifact.
  4. Whole-crate gates stay green after the fixes: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, and `cargo test` all pass; the change is behavior-preserving except for the intended `soft_dtw` gradient correction.

**Plans**: 2 plans

- [x] 78-01-PLAN.md — CORR-01: fix soft_dtw_backward endpoint-seed bug + SC#1 regression tests + tighten the three barycenter tests + update oracle doc comment
- [x] 78-02-PLAN.md — CORR-02: audit sweep of the 9 hand-written gradient passes + write 78-AUDIT.md disposition table

### Phase 79: Serde Feature Repair

**Goal**: `cargo build --features serde` compiles cleanly again and cannot silently re-break.
**Depends on**: Nothing (independent of Phase 78)
**Requirements**: BUILD-01
**Success Criteria** (what must be TRUE):

  1. `cargo build --features serde` compiles cleanly (broken since Phase 60) — `ClassifFit` and any types it embeds that lacked serde support gain conditional `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` derives consistent with the crate's existing serde convention.
  2. A `ClassifFit` (or embedding type) can round-trip through serde serialization/deserialization under `--features serde`.
  3. A guard — a `--features serde` build/round-trip check runnable in CI — is in place to prevent silent re-breakage.
  4. The default-feature build and existing gates (`cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`) remain green; the change is additive/non-breaking.

**Plans**: 1 plan

- [x] 79-01-PLAN.md — Add conditional serde derives to ClassifFit (+ embedded ClassifResult/ClassifMethod), NonConformityScore, JointFpcaResult; land a serde round-trip test; green both build configs

### Phase 80: Release Hardening & Ship v0.40.0

**Goal**: The crate is validated and release-ready — outstanding v0.39.0 phases formally signed off, version/CHANGELOG/docs updated, and all whole-crate gates green — so the operator's `v0.40.0` tag → crates.io publish is the only remaining step.
**Depends on**: Phase 78, Phase 79 (validates + folds in all prior fixes; must land last)
**Requirements**: REL-01, REL-02
**Success Criteria** (what must be TRUE):

  1. Phases 75/76/77 `VALIDATION.md` are moved from `status: draft` to `validated` via the validate-phase flow, reflecting the green test suite; any genuine coverage gaps surfaced during sign-off are filled or explicitly recorded.
  2. `fdars-core` version is bumped 0.38.0 → 0.40.0 and `CHANGELOG.md` carries both the v0.39.0 (AD core) and v0.40.0 (this milestone) entries.
  3. README and tracked `documentation/` (the `docs/` dir is gitignored) are refreshed wherever they reference the version or the corrected `soft_dtw` behavior.
  4. The whole-crate gates pass: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, and `cargo test`.
  5. The final `git tag v0.40.0` push → crates.io publish via `release.yml` is documented in the phase SUMMARY as the operator-driven step, gated on all prior phases being green (the phase prepares and verifies release-readiness, it does not itself tag/publish).

**Plans**: 2 plans

- [x] 80-01-PLAN.md — REL-01: lightweight Nyquist sign-off of phases 75/76/77 VALIDATION.md (draft → validated where coverage holds; record any gap)
- [x] 80-02-PLAN.md — REL-02: version bump 0.38.0 → 0.40.0, CHANGELOG [0.39.0]+[0.40.0], README/documentation refresh, whole-crate gates green, operator ship-steps in SUMMARY

<details>
<summary>✅ v0.39.0 — DIFF: Differentiable FDA Core (Forward-Mode Autodiff) (Phases 75–77) — SHIPPED 2026-09-06</summary>

Promoted **GAP-08** (score 1.73, L-effort) — **the last item in the v0.31.0 `GAP-BACKLOG.md`, now exhausted.** Added an in-crate forward-mode automatic-differentiation core (`Scalar` trait + `Dual{value,tangent}`) and made a scoped subset of FDA ops (**elastic distance + FPCA scores**) generic over the scalar type, so exact gradients compose through arbitrary op chains. Additive/non-breaking (existing f64 signatures untouched; generic versions alongside), **no new crate dependency** (in-crate dual numbers, forward-mode only). Milestone audit: 4/4 requirements, integration INTEGRATED. Whole-crate gates green (2859 lib + 209 doc tests, clippy `--all-targets` clean, fmt clean). Full detail: [milestones/v0.39.0-ROADMAP.md](milestones/v0.39.0-ROADMAP.md).

- [x] Phase 75: Scalar Trait & Forward-Mode Dual Substrate (1/1) — DIF-01 — completed 2026-09-06
- [x] Phase 76: Differentiable Elastic Distance & FPCA Scores (2/2) — DIF-02, DIF-03 — completed 2026-09-06
- [x] Phase 77: Gradient API, Composition Demo & Integration (1/1) — DIF-04 — completed 2026-09-06

**Scope note:** the warp-*searched* `elastic_distance` was intentionally NOT made differentiable (its discrete DP argmin is piecewise-constant → non-differentiable) — DIF-02 delivered the differentiable `soft_dtw_distance_generic` + fixed-warp `amplitude_distance_at_warp_generic` instead. A **pre-existing** `soft_dtw_backward` zero-gradient bug was discovered and logged to backlog (now being fixed in v0.40.0 Phase 78 / CORR-01). Nyquist VALIDATION.md files remain `draft` (reconciled in v0.40.0 Phase 80 / REL-01).

**Ship step remaining (operator):** crate version bump + `v0.40.0` git tag + crates.io publish now folds the v0.39.0 AD core forward — see v0.40.0 Phase 80.

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

## Progress

**Execution Order:** 78 and 79 are independent (either order / parallelizable); 80 lands last (validates + folds in everything).

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 78. Gradient Correctness — soft_dtw Fix & Backward-Pass Audit | v0.40.0 | 2/2 | Complete    | 2026-09-07 |
| 79. Serde Feature Repair | v0.40.0 | 1/1 | Complete    | 2026-09-07 |
| 80. Release Hardening & Ship v0.40.0 | v0.40.0 | 2/2 | Complete    | 2026-09-07 |

## Status

**v0.40.0 Correctness & Release Hardening in progress** (Phases 78–80). A small fixup/hardening milestone: fix the pre-existing `soft_dtw_backward` zero-gradient bug + sweep sibling gradient passes (CORR-01/02, Phase 78), repair the `--features serde` build broken since Phase 60 (BUILD-01, Phase 79), then Nyquist-validate phases 75/76/77 and bump/CHANGELOG/docs to release-ready (REL-01/02, Phase 80). All parity/gap backlogs (scikit-fda, R core, multi-ecosystem) remain exhausted; this milestone is correctness/release-hardening only, no new algorithms.

**Ship:** real code changes → this milestone gets a `v0.40.0` git tag. The crate source is code-complete through v0.39.0 but Cargo.toml is still versioned 0.38.0; the `v0.40.0` tag folds the v0.39.0 AD core + these fixes into fdars' first crates.io release since v0.38.0. Tag/publish is the final operator-driven step (Phase 80 prepares + verifies release-readiness).

**Next:** `/gsd-plan-phase 78` (or `79` — independent).
