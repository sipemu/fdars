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
- 🔨 **v0.23.0 — Depth, Outliers & Interval Inference** — Phases 28–30 (in progress)

## Phases

- [ ] **Phase 28: Depth-Measure Long Tail** - Add the missing univariate functional depth measures (HRD/MHRD, HI/MHI/EI, extremal, ERL, L∞, TVD+MSSI) to `depth/`, each registered in the `DepthMethod` dispatcher (DEPTH-01)
- [ ] **Phase 29: Outlier-Detector Suite** - Add `tvdmss`, `muod`, `sequential_transform_outliers`, and the `depthgram` statistic to `outliers.rs`, reusing DEPTH-01's TVD+MSSI depth (OUT-01, depends on Phase 28)
- [ ] **Phase 30: Interval Testing Procedure Family** - Implement one-/two-population interval-wise tests (B-spline & Fourier) with domain-selective adjusted p-values + interval-wise FLM coefficient testing in new `inference/itp.rs` (INF-03, independent of Phases 28/29)

## Phase Details

### Phase 28: Depth-Measure Long Tail
**Goal**: A user can compute every canonical batch univariate functional depth measure that `roahd`/`fdaoutlier` expose but fdars was missing — half-region & modified half-region depth, hypograph/modified-hypograph & epigraph indices, extremal depth, extreme-rank-length depth, L∞ depth, and total-variation depth with MSSI — over a column-major `FdMatrix`, and select any of them through the existing unified `DepthMethod` dispatcher, without any existing depth code changing.
**Depends on**: Nothing (independent of Phase 30; Phase 29 depends on this phase)
**Requirements**: DEPTH-01
**Success Criteria** (what must be TRUE):

  1. User can call a new `Result`-returning public function per measure in `fdars-core/src/depth/` (crate-root re-exported) — half-region depth (HRD), modified half-region depth (MHRD), hypograph index (HI), modified-hypograph index (MHI), un-modified epigraph index (EI), extremal depth, extreme-rank-length depth (ERL), L∞ depth, and total-variation depth with MSSI — each computing the measure over a column-major `FdMatrix` and returning per-curve depth values.
  2. Each new measure is registered as a `DepthMethod` variant so `functional_depth(data, method)` dispatches to it, and the existing `DepthMethod` variants and dispatcher signature keep working unchanged.
  3. On synthetic data with a known depth ordering (a clearly-central curve plus injected magnitude/shape outliers), each measure ranks the central curve deepest and the outliers shallowest, matching the documented reference behavior within tolerance (inline `#[cfg(test)]` tests).
  4. Invalid inputs (empty matrix / single curve / mismatched argvals vs values / degenerate columns) return `FdarError` rather than panicking, with dimension/parameter checks at each entry point.
  5. No new crate dependency is added, existing depth-measure and `DepthMethod` public signatures are untouched (additive/non-breaking), and the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 3 plans
- [ ] 28-01-PLAN.md — hypo_epi.rs (HI/MHI/EI, tracer) + half_region.rs (HRD/MHRD) wired to DepthMethod
- [ ] 28-02-PLAN.md — extremal.rs + erl.rs + linf.rs (extremal / ERL / L∞ depths) wired to DepthMethod
- [ ] 28-03-PLAN.md — tvd.rs (TvdMssResult + TVD/MSSI, pinned for Phase 29) + all-9-variant dispatcher round-trip

### Phase 29: Outlier-Detector Suite
**Goal**: A user can flag magnitude and shape outliers in a functional sample with the four `fdaoutlier`/`roahd` detectors fdars was missing — `tvdmss` (TVD+MSSI), `muod` (Massive Unsupervised Outlier Detection), sequential-transformation detection, and the `depthgram` statistic — as numeric outputs (indices/scores, no rendering), reusing the DEPTH-01 depths and the existing MS-plot / outliergram machinery, without any existing outlier code changing.
**Depends on**: Phase 28 (tvdmss reuses DEPTH-01's total-variation depth + MSSI; must be sequenced after Phase 28)
**Requirements**: OUT-01
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning public functions in `fdars-core/src/outliers.rs` (crate-root re-exported) — `tvdmss`, `muod`, `sequential_transform_outliers`, and a `depthgram` statistic — each returning numeric outputs (flagged outlier indices and/or per-curve scores), with no plotting/rendering.
  2. On synthetic data with injected magnitude outliers and injected shape outliers, `tvdmss` flags both classes and the other detectors return the expected outlier index sets within a documented tolerance (inline `#[cfg(test)]` tests).
  3. `tvdmss` computes its detector from DEPTH-01's total-variation depth + MSSI (built on Phase 28), and the suite reuses the existing MS-plot / outliergram machinery rather than reimplementing it — adding no new crate dependency.
  4. Invalid inputs (empty / single-curve / mismatched dimensions / degenerate columns) return `FdarError` rather than panicking, with checks at each entry point.
  5. Existing outlier detectors (`magnitude_shape_outlyingness`, `outliergram`) and all DEPTH-01 depth functions keep their public signatures unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 2 plans
- [ ] 29-01-PLAN.md — tvdmss (TRACER: TvdMssResult + iqr_fence + functional_boxplot + re-export) + muod (pointwise-mean regression, upper IQR fence)
- [ ] 29-02-PLAN.md — sequential_transform_outliers (T0/T1/T2/D1 cumulative + functional_boxplot base) + depthgram (MBD/MEI parabola, upper fence)

### Phase 30: Interval Testing Procedure Family
**Goal**: A user can run the Interval Testing Procedure (ITP) family that `fdatest` provides — one-population and two-population interval-wise tests over B-spline and Fourier bases with domain-selective adjusted p-values, plus interval-wise FLM coefficient testing — via a new `inference/itp.rs`, reusing the shipped INF-01 permutation infrastructure and `basis/` projection, without any existing inference code changing.
**Depends on**: Nothing (independent of Phases 28/29; depends only on already-shipped INF-01 permutation infrastructure + existing `basis/` — may run in parallel with Phases 28/29)
**Requirements**: INF-03
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning public entry points in `fdars-core/src/inference/itp.rs` (crate-root re-exported) for a one-population interval-wise test, a two-population interval-wise test, and an interval-wise FLM coefficient test, each accepting a basis choice (B-spline or Fourier) and returning per-component/per-domain adjusted p-values.
  2. Each test returns domain-selective adjusted p-values (the ITP interval-wise closure adjustment), so a user can identify which sub-intervals of the domain drive a significant result — not just a single global p-value.
  3. On synthetic data with a localized between-group difference confined to a known sub-interval (and a null case with no difference), the adjusted p-values are small on the true differing interval and non-significant elsewhere / everywhere in the null case, within a documented tolerance (inline `#[cfg(test)]` tests).
  4. The ITP family reuses the INF-01 permutation infrastructure and `basis/` projection, adds no new crate dependency, and invalid inputs (empty / mismatched group sizes / incompatible basis parameters) return `FdarError` rather than panicking.
  5. Existing `inference/` entry points (INF-01/INF-02 tests) and `basis/` projection keep their public signatures unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 2 plans
- [ ] 30-01-PLAN.md — pval_correct closure helper (hand-computed test) + itp_one_pop wired end-to-end + ItpResult + serde + mod.rs/lib.rs re-export (tracer)
- [ ] 30-02-PLAN.md — itp_two_pop (pool + relabel) + itp_flm (response permutation) reusing the plan-01 closure helpers

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 28. Depth-Measure Long Tail | 0/TBD | Not started | - |
| 29. Outlier-Detector Suite | 0/2 | Not started | - |
| 30. Interval Testing Procedure Family | 0/2 | Not started | - |

**Execution order:** Phase 29 (OUT-01) has a **hard dependency on Phase 28** (DEPTH-01) — `tvdmss` reuses DEPTH-01's total-variation depth + MSSI, so **28 must complete before 29**. Phase 30 (INF-03) is **independent** of Phases 28/29 (it depends only on the already-shipped INF-01 permutation infrastructure + existing `basis/`) and **may run in parallel** with them. Default sequence: 28 → 29, with 30 free to run alongside. First differentiator milestone (P1 table-stakes exhausted after v0.22.0); all three items score 2.31 (P2, M-effort) in `R-BACKLOG.md`. Additive/non-breaking, reuse-first, no new crate dependency.

<details>
<summary>✅ v0.22.0 PACE Sparse FPCA & Elastic Multinomial (Phases 26–27) — SHIPPED 2026-08-19</summary>

Final P1 table-stakes item (FPCA-01) + elastic-family completion (REG-03), each reuse-first (orchestrate/extend existing code). Additive/non-breaking, no new dependency; full suite 2107 lib tests green, clippy `--all-targets` clean, serde-feature build verified. Milestone audit passed (2/2). Disjoint independent modules. Code review caught + fixed real bugs both phases (Phase 26: NaN-mean guard + BLUP/band ridge/error consistency + n_i≥2; Phase 27: serde-feature compile break + normalization Inf/NaN guard). After this milestone the P1 table-stakes tier is exhausted.

- [x] Phase 26: PACE Sparse FPCA (1/1 plans) — completed 2026-08-19 (FPCA-01: `pace_fpca` + `PaceFpcaConfig` + `PaceFpcaResult` in new `pace_fpca.rs`)
- [x] Phase 27: Elastic Multinomial Regression (1/1 plans) — completed 2026-08-19 (REG-03: `elastic_multinomial` + `predict_elastic_multinomial` + `ElasticMultinomialResult` in `elastic_regression/logistic.rs`)

Full phase detail: [milestones/v0.22.0-ROADMAP.md](milestones/v0.22.0-ROADMAP.md)

</details>
<details>
<summary>✅ v0.14.0 Performance & scikit-fda Gap Audit (Phases 1–9) — SHIPPED 2026-08-09</summary>

Audit-only milestone — every phase produced analysis artifacts, zero `fdars-core/src/` edits. Deliverables: `.planning/research/AUDIT-REPORT.md` (consolidated report) + `.planning/research/BACKLOG.md` (32-item value-ranked backlog).

- [x] Phase 1: Measurement Discipline & Baselines (2/2 plans) — completed 2026-08-07
- [x] Phase 2: Static Hot-Path Analysis (2/2 plans) — completed 2026-08-07
- [x] Phase 3: Elastic Alignment Hot Path (2/2 plans) — completed 2026-08-08
- [x] Phase 4: FPCA/SVD & Allocation Audit (3/3 plans) — completed 2026-08-08
- [x] Phase 5: Parallelism Gap Assessment (3/3 plans) — completed 2026-08-08
- [x] Phase 6: Conditional SVD Library Comparison (1/1 plans) — completed 2026-08-09
- [x] Phase 7: scikit-fda Capability Enumeration (2/2 plans) — completed 2026-08-09
- [x] Phase 8: Capability Parity Matrix & Categorization (3/3 plans) — completed 2026-08-09
- [x] Phase 9: Consolidated Report & Prioritized Backlog (3/3 plans) — completed 2026-08-09

Full phase detail: [milestones/v0.14.0-ROADMAP.md](milestones/v0.14.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.15.0 Top-Backlog Quick Wins (Phases 10–11) — SHIPPED 2026-08-11</summary>

First implementation milestone — the top-4 audit-backlog quick wins delivered as real `fdars-core/src/` code, each with inline tests and numerical verification. Full suite green; milestone audit passed (4/4); shipped via PR #38, `fdars-core` 0.15.0 on crates.io.

- [x] Phase 10: Capability Gaps — Spline Interpolation & Functional Summary Statistics (2/2 plans) — completed 2026-08-10 (FEAT-01, FEAT-02)
- [x] Phase 11: Performance Wins — Parallel CV Folds & faer FPCA SVD (2/2 plans) — completed 2026-08-11 (PERF-01, PERF-02)

Full phase detail: [milestones/v0.15.0-ROADMAP.md](milestones/v0.15.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.16.0 Elastic Feasibility + Parity Quick Wins (Phases 12–13) — SHIPPED 2026-08-12 (PR #40)</summary>

Second implementation milestone — the next tier of the v0.14.0 audit backlog: the P1 elastic-feasibility headline plus three effort-S scikit-fda parity gaps, all additive/non-breaking. Milestone audit passed (4/4 requirements, cross-phase integration clean, 2663 tests green). Released via PR #40 (crate 0.16.0, tag v0.16.0).

- [x] Phase 12: Elastic Feasibility — Banded Alignment Default & `band_frac` (1/1 plans) — completed 2026-08-12 (PERF-03: opt-in `*_with_band` wrappers, large grids feasible)
- [x] Phase 13: Parity Quick Wins — Imputation, Extrapolation Policy & Scoring Metrics (2 plans + 1 gap-closure) — completed 2026-08-12 (FEAT-03 imputation, FEAT-04 `ExtrapolationPolicy` both interp paths, FEAT-05 five scoring metrics)

Full phase detail: [milestones/v0.16.0-ROADMAP.md](milestones/v0.16.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.17.0 Registration Parity & Elastic-FPCA Performance (Phases 14–15) — SHIPPED 2026-08-12 (PR #41)</summary>

Third implementation milestone — the next tier of the v0.14.0 audit backlog: the P1 shift-registration gap + its scikit-fda quality diagnostics, plus a targeted elastic-FPCA parallelization. All additive/non-breaking. Milestone audit passed (3/3 requirements, integration clean; full suite green: 2727 tests `linalg,parallel` / 2718 default). Released via PR #41 (crate 0.17.0, tag v0.17.0).

- [x] Phase 14: Shift Registration (2/2 plans) — completed 2026-08-12 (FEAT-06 `least_squares_shift_registration` + `ShiftRegistrationResult` in new `alignment/shift.rs`; FEAT-07 three registration-quality scores in `alignment/quality.rs`)
- [x] Phase 15: Elastic-FPCA Performance (1/1 plans) — completed 2026-08-12 (PERF-04 parallelize `:701/:720/:764` via `iter_maybe_parallel!` collect-then-assign, N≥50 guard, bit-identical equivalence)

Full phase detail: [milestones/v0.17.0-ROADMAP.md](milestones/v0.17.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.18.0 R-Ecosystem Gap Audit (Phases 16–19) — SHIPPED 2026-08-15</summary>

Audit-only milestone — the R-ecosystem analog of v0.14.0: zero `fdars-core/src/` edits. The R FDA ecosystem (35 packages surveyed) replaced scikit-fda as the yardstick now that the actionable scikit-fda backlog is exhausted. Milestone audit passed (7/7 requirements). Deliverables: `.planning/research/R-AUDIT-REPORT.md` (inventory → parity matrix → strengths → consolidated findings) + `.planning/research/R-BACKLOG.md` (26-item value-ranked, promotion-ready backlog).

- [x] Phase 16: R Ecosystem Inventory (2/2 plans) — completed 2026-08-15 (INV-01, INV-02: 35 pkgs, 275 caps, 248 in-scope)
- [x] Phase 17: Parity Matrix & Categorization (1/1 plans) — completed 2026-08-15 (GAP-01, GAP-02: 250 rows mapped, 162 actionable gaps, 18 table-stakes / 144 differentiator)
- [x] Phase 18: Reverse-Parity Strengths Sweep (1/1 plans) — completed 2026-08-15 (GAP-03: 42 modules walked, 12 R-honest strengths)
- [x] Phase 19: Consolidated Report & Ranked Backlog (1/1 plans) — completed 2026-08-15 (RPT-01, RPT-02: 26 ranked GSD-ready items, completeness gate PASS)

Full phase detail: [milestones/v0.18.0-ROADMAP.md](milestones/v0.18.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.19.0 Functional Inference Suite (Phases 20–21) — SHIPPED 2026-08-16</summary>

First implementation milestone from the v0.18.0 R-ecosystem backlog — the two P1 table-stakes inference items (INF-01, INF-02) closing fdars' dominant table-stakes deficit: R-parity Area 5 (Inference), previously 0/22 present. New `fdars-core/src/inference/` module (8 public entry points), additive/non-breaking, 29 inline tests; full suite green (2039 lib tests), clippy clean. Milestone audit passed (2/2). No new crate dependency (self-contained χ²/F survival functions).

- [x] Phase 20: Two-Sample Functional Tests & `inference/` Module (1/1 plans) — completed 2026-08-16 (INF-01: `t_perm_test`, `f_perm_test`, `two_sample_mean_test`, `mean_scb`, `scb_two_sample_test` + `TestResult`)
- [x] Phase 21: Functional-Linear-Model Inference (1/1 plans) — completed 2026-08-16 (INF-02: `flm_f_test`, `flm_gof_test`, `oneway_anova_vstat` alongside unchanged `fanova`)

Full phase detail: [milestones/v0.19.0-ROADMAP.md](milestones/v0.19.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.20.0 Table-Stakes Quick Wins (Phases 22–23) — SHIPPED 2026-08-16</summary>

Second batch of R-backlog items — the two top-ranked (score 5.00, P1 table-stakes, S-effort) quick wins, each wrapping existing infrastructure. Additive/non-breaking; full suite green (2061 lib tests), clippy clean, no new dependency. Milestone audit passed (2/2). Phases independent (disjoint modules).

- [x] Phase 22: Constant Basis & AIC Smoothing Selection (1/1 plans) — completed 2026-08-16 (T-01: `constant_basis`, `CvCriterion::Aic` + `aic_smoother`, `smooth_basis_aic`)
- [x] Phase 23: Functional Boxplot & Depth Dispatcher (1/1 plans) — completed 2026-08-16 (T-02: `DepthMethod` + `functional_depth` dispatcher, `functional_boxplot` + `FunctionalBoxplotResult`)

Full phase detail: [milestones/v0.20.0-ROADMAP.md](milestones/v0.20.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.21.0 Functional Regression Completeness (Phases 24–25) — SHIPPED 2026-08-17</summary>

Third batch of R-backlog items — the two remaining P1 table-stakes functional-regression gaps (REG-01 rank 6, REG-02 rank 7), each reusing existing scalar-on-function design machinery. Additive/non-breaking, `Result`-returning, **zero changes to existing public signatures**, no new crate dependency. Full suite green (2081 lib tests + doctests), clippy `--all-targets` clean. Milestone audit passed (2/2). Both phases independent (disjoint modules).

- [x] Phase 24: Concurrent / Varying-Coefficient Regression (1/1 plans) — completed 2026-08-17 (REG-01: `concurrent_regression` + `ConcurrentRegrResult` in new `concurrent_regression.rs`)
- [x] Phase 25: Functional GLM (Exponential Family) (1/1 plans) — completed 2026-08-17 (REG-02: `functional_glm` + `GlmFamily{Binomial,Poisson,Gamma,Gaussian}` + `FunctionalGlmResult` in new `scalar_on_function/glm.rs`)

Full phase detail: [milestones/v0.21.0-ROADMAP.md](milestones/v0.21.0-ROADMAP.md)

</details>
