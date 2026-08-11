# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (shipped 2026-08-11) — [archive](milestones/v0.15.0-ROADMAP.md)
- 🚧 **v0.16.0 — Elastic Feasibility + Parity Quick Wins** — Phases 12–13 (in progress)

## Overview

Second implementation milestone. Promotes the next tier of the v0.14.0 audit backlog after v0.15.0 shipped ranks 1–4. It pairs the single highest-*value* remaining item — banded elastic alignment (PERF-ELASTIC-BAND, P1, value 5), which makes currently-infeasible large-grid elastic workloads (N=500, M=200) tractable — with three effort-S scikit-fda parity gaps (in-grid NaN imputation, a composable extrapolation-policy enum, and functional scoring metrics). All four items are additive/non-breaking, respect the column-major `FdMatrix` layout and the feature-gated parallelism model, and expose only `Result<T, FdarError>`-returning public APIs. The audit already produced exact file locations, root causes, and proposed API signatures (see `.planning/research/BACKLOG.md`) — reuse, do not re-derive.

## Phases

**Phase Numbering:**

- Integer phases (12, 13): Planned milestone work (continuing from v0.15.0's Phase 11)
- Decimal phases (12.1, 12.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 12: Elastic Feasibility — Banded Alignment Default & `band_frac`** - Expose a banded DP path (`band_frac`) through the elastic alignment API so large grids become tractable; retain the exact unbanded path
- [ ] **Phase 13: Parity Quick Wins — Imputation, Extrapolation Policy & Scoring Metrics** - Add in-grid NaN imputation, a composable `ExtrapolationPolicy` enum, and functional scoring metrics (MAE/MSE/MAPE/MSLE/explained-variance)

## Phase Details

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

First implementation milestone (v0.14.0 was audit-only) — the top-4 audit-backlog quick wins delivered as real `fdars-core/src/` code (two table-stakes capability gaps, two measured performance wins), each with inline tests and numerical-equivalence/accuracy verification. Full suite 1948 tests green under both feature configs; milestone audit passed (4/4 requirements, cross-phase integration clean); shipped via PR #38.

- [x] Phase 10: Capability Gaps — Spline Interpolation & Functional Summary Statistics (2/2 plans) — completed 2026-08-10 (FEAT-01 spline interpolation, FEAT-02 functional summary statistics)
- [x] Phase 11: Performance Wins — Parallel CV Folds & faer FPCA SVD (2/2 plans) — completed 2026-08-11 (PERF-01 parallel `fclassif_cv`, PERF-02 faer `thin_svd` FPCA)

Full phase detail: [milestones/v0.15.0-ROADMAP.md](milestones/v0.15.0-ROADMAP.md)

</details>

### 🚧 v0.16.0 Elastic Feasibility + Parity Quick Wins (In Progress)

**Milestone Goal:** Make large-grid elastic alignment feasible (the audit's top bottleneck) and close three more scikit-fda parity gaps — the next tier of the v0.14.0 audit backlog. All changes are additive/non-breaking.

### Phase 12: Elastic Feasibility — Banded Alignment Default & `band_frac`

**Goal**: Users can run elastic alignment (`karcher_mean`, `elastic_self_distance_matrix`, `elastic_cross_distance_matrix`) on a banded Sakoe-Chiba dynamic-programming path via an exposed `band_frac`, so previously-infeasible large grids (e.g. N=500, M=200) complete in tractable time — while the full unbanded path remains available and exact.
**Depends on**: Nothing (first phase of this milestone; independent of prior milestones — the banded DP variants already exist and are correct)
**Requirements**: PERF-03
**Success Criteria** (what must be TRUE):

  1. The high-level elastic alignment API exposes a `band_frac` (Sakoe-Chiba band width as a fraction of M) threaded into the DP warp search across `alignment/` (`karcher.rs`, `elastic_self_distance_matrix` / `elastic_cross_distance_matrix`); the banded path is reachable through the public API without callers having to invoke the internal `_banded` variants directly.
  2. The full (unbanded) path remains available and unchanged — a caller can still request exact unbanded alignment (`band_frac = 0.0` / `None`), and existing callers are not broken (additive parameter or config, no positional-argument breakage).
  3. An inline `#[cfg(test)]` test confirms that alignment output with a sufficiently wide band matches the unbanded result within numerical tolerance (banded ≈ unbanded at small M where exact comparison is feasible).
  4. A benchmark or timing test demonstrates the feasibility improvement at a large (N, M) — the banded path completes an alignment that the unbanded default previously made infeasible (audit measured ~4–6× at representative cells).
  5. `cargo test -p fdars-core --features linalg` and `cargo clippy -p fdars-core --features linalg` pass with the new/changed alignment API covered; every new/changed public function returns `Result<T, FdarError>` and respects the column-major `FdMatrix` layout.

**Plans**: 1/1 plans executed

- [x] 12-01-PLAN.md — Opt-in `*_with_band(band_frac: Option<f64>)` wrappers for `karcher_mean` / self+cross distance matrices, crate-root re-exports, equivalence + feasibility tests

**Note:** PERF-ELASTIC-BAND is the P1 headline (value 5) — the banded implementations already exist; the work is API surfacing/defaulting + equivalence and feasibility tests, not a new algorithm. Isolated to `alignment/`, so it stands alone as its own phase and can execute concurrently with Phase 13 (no shared files).

### Phase 13: Parity Quick Wins — Imputation, Extrapolation Policy & Scoring Metrics

**Goal**: Callers can impute missing (`NaN`) values in a regular-grid `FdMatrix`, control out-of-range interpolation/evaluation behavior via a composable `ExtrapolationPolicy` enum, and score functional predictions against observations with standard metrics (MAE, MSE, MAPE, MSLE, explained-variance) — closing three scikit-fda parity gaps.
**Depends on**: Nothing (independent of Phase 12; can run in parallel with it)
**Requirements**: FEAT-03, FEAT-04, FEAT-05
**Success Criteria** (what must be TRUE):

  1. A new public imputation function returning `Result<FdMatrix, FdarError>` fills `NaN` entries in a regular-grid `FdMatrix` with at least mean and linear-interpolation strategies over each curve; input validation rejects all-missing curves and unsupported configurations. Inline tests verify imputation reproduces known values on synthetic gaps and errors on invalid input. *(FEAT-03)*
  2. A composable `ExtrapolationPolicy` enum — `Boundary` (clamp to nearest edge), `Exception` (return `FdarError` for out-of-range queries), `Fill(value)` (constant fill), `Periodic` (wrap) — threads through `spline_interpolate` and the existing linear interpolation/evaluation path to control behavior for query points outside `argvals`; inline tests exercise each variant, including the error path for `Exception`. *(FEAT-04)*
  3. Public scoring functions over `FdMatrix` — `functional_mae`, `functional_mse`, `functional_mape`, `functional_msle`, `functional_explained_variance` — each return `Result<_, FdarError>` with dimension validation; inline tests verify each against a hand-computed reference. *(FEAT-05)*
  4. All three additions are additive (new functions/enum, no removals or breaking changes), respect the column-major `FdMatrix` layout, and never panic on input (always `Result<_, FdarError>`); the existing linear-interpolation path and `spline_interpolate` remain available.
  5. `cargo test -p fdars-core --features linalg` and `cargo clippy -p fdars-core --features linalg` pass with the new imputation, extrapolation-policy, and scoring functions covered.

**Plans**: 2 plans
- [ ] 13-01-PLAN.md — FEAT-03 imputation + FEAT-04 extrapolation policy (both in helpers.rs, serialized)
- [ ] 13-02-PLAN.md — FEAT-05 functional scoring metrics (new scoring.rs)

**Note:** FEAT-03, FEAT-04, FEAT-05 are independent effort-S additive parity gaps. FEAT-03 (imputation) and FEAT-04 (extrapolation policy) both touch `helpers.rs`; FEAT-05 (scoring) lands in a new `scoring.rs`. The three plans are largely parallelizable, but FEAT-03 and FEAT-04 share `helpers.rs` — sequence or serialize those two writes (e.g. same wave with careful non-overlapping edits, or adjacent waves) to avoid a merge collision on that file.

## Progress

**Execution Order:**
Phases execute in numeric order: 12 → 13. Phases 12 and 13 are mutually independent (disjoint files: `alignment/` vs `helpers.rs`/`scoring.rs`) and may be executed in either order or concurrently.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 12. Elastic Feasibility (Banded Alignment & `band_frac`) | v0.16.0 | 1/1 | Complete    | 2026-08-11 |
| 13. Parity Quick Wins (Imputation, Extrapolation Policy & Scoring) | v0.16.0 | 0/2 | Not started | - |

## Coverage

All 4 v1 requirements mapped to exactly one phase — no orphans, no duplicates.

| Requirement | Phase | Audit Backlog ID |
|-------------|-------|------------------|
| PERF-03 | Phase 12 | PERF-ELASTIC-BAND (P1/M, score 2.89; also P5-4) |
| FEAT-03 | Phase 13 | PREP-03 (P2/S, score 3.00) |
| FEAT-04 | Phase 13 | REPR-03 (P2/S, score 3.00) |
| FEAT-05 | Phase 13 | MISC-04 (P2/S, score 3.00) |

**Coverage:** 4/4 v1 requirements mapped ✓

---
*Milestone v0.16.0 started 2026-08-11 — second implementation milestone. Phases 12–13 promoted from the v0.14.0 ranked backlog: PERF-03 (banded elastic alignment, P1 headline), FEAT-03 (imputation), FEAT-04 (extrapolation policy), FEAT-05 (scoring metrics). Prior: v0.15.0 shipped 2026-08-11 (Phases 10–11, 4/4 requirements). v0.14.0 audit milestone shipped 2026-08-09 (9 phases, 13/13 requirements).*
