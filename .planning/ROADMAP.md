# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- 🚧 **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (in progress)

## Overview

First implementation milestone (v0.14.0 was audit-only). Executes the four highest-value, low-effort (score 3.00–4.00, effort S) items from the v0.14.0 audit backlog with real `fdars-core/src/` changes: two table-stakes capability gaps (spline interpolation, functional summary statistics) and two measured performance wins (parallel CV fold loop, faer SVD swap). All four items are independent with no cross-dependencies — the audit already produced exact file locations, root causes, and proposed API signatures (see `.planning/research/BACKLOG.md`). Every new/changed public function returns `Result<T, FdarError>`, respects the column-major `FdMatrix` layout and the feature-gated parallelism model, and ships with inline `#[cfg(test)]` unit tests plus numerical-equivalence/accuracy verification.

## Phases

**Phase Numbering:**
- Integer phases (10, 11): Planned milestone work (continuing from v0.14.0's Phase 9)
- Decimal phases (10.1, 10.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 10: Capability Gaps — Spline Interpolation & Functional Summary Statistics** - Add off-grid spline interpolation and the missing descriptive-statistics functions
- [ ] **Phase 11: Performance Wins — Parallel CV Folds & faer FPCA SVD** - Parallelize the classification CV fold loop and swap the FPCA SVD to faer behind `linalg`

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

### 🚧 v0.15.0 Top-Backlog Quick Wins (In Progress)

**Milestone Goal:** Ship the top-4 audit-backlog quick wins as real `fdars-core` code — two capability gaps closed, two performance wins landed — each with tests and numerical verification.

### Phase 10: Capability Gaps — Spline Interpolation & Functional Summary Statistics
**Goal**: Callers can interpolate functional data at arbitrary off-grid query points with cubic/order-k splines, and compute the standard functional descriptive statistics (trimmed mean, depth-based median, covariance, variance, std) directly over an `FdMatrix`.
**Depends on**: Nothing (first phase of this milestone; independent of the v0.14.0 audit phases)
**Requirements**: FEAT-01, FEAT-02
**Success Criteria** (what must be TRUE):
  1. A new public `spline_interpolate(data: &FdMatrix, argvals: &[f64], query_points: &[f64], order: usize) -> Result<FdMatrix, FdarError>` in `helpers.rs` fits a B-spline (reusing the existing `basis/` B-spline system) per curve and evaluates at the query points; an inline `#[cfg(test)]` test shows it reproduces known cubic-spline values within 1e-10 at off-grid points, and reproduces the input exactly (within 1e-10) when query points coincide with `argvals`.
  2. Five new public functions — `trim_mean` (depth-trimmed mean), `depth_based_median` (index of deepest curve), `functional_covariance` (M×M sample covariance), `functional_variance` (pointwise), `functional_std` (pointwise) — accept an `FdMatrix` and return `Result<_, FdarError>`; inline unit tests verify each against a hand-computed reference (variance = squared std pointwise; covariance diagonal equals `functional_variance`; `depth_based_median` returns the argmax-depth curve index).
  3. Every new function validates inputs and returns `FdarError` (never panics) on dimension/parameter mismatch (e.g. `query_points` out of range, `order` too large for the grid, `alpha` outside [0,1)), exercised by inline tests.
  4. `cargo test -p fdars-core --features linalg` and `cargo clippy -p fdars-core --features linalg` pass with the new functions covered; the existing linear-interpolation path remains available (spline is additive, not a removal).
**Plans**: 2 plans

Plans:
- [ ] 10-01-spline-interpolate-PLAN.md — `spline_interpolate` in `helpers.rs` (B-spline fit-then-evaluate) + inline tests + crate-root re-export (FEAT-01, wave 1, independent)
- [ ] 10-02-functional-summary-statistics-PLAN.md — `trim_mean`, `depth_based_median`, `functional_covariance`, `functional_variance`, `functional_std` in `fdata.rs` + inline tests + crate-root re-exports (FEAT-02, wave 1, independent)

**Note:** FEAT-01 and FEAT-02 are independent additive features with no shared code — the two plans are fully parallelizable.

### Phase 11: Performance Wins — Parallel CV Folds & faer FPCA SVD
**Goal**: The classification cross-validation fold loop runs in parallel under the `parallel` feature with identical results, and FPCA computes its SVD via faer `thin_svd` on a zero-copy view under the `linalg` feature, producing results equivalent to the retained nalgebra path.
**Depends on**: Nothing (independent of Phase 10; can run in parallel with it)
**Requirements**: PERF-01, PERF-02
**Success Criteria** (what must be TRUE):
  1. The `for fold in 0..nfold` loop in `fclassif_cv` (`classification/cv.rs:76`) is wrapped in `iter_maybe_parallel!(0..nfold)` and collects fold results into a `Vec`; an integration/inline test asserts parallel CV fold accuracy (and per-fold outputs) is bit-for-bit identical to sequential execution for a fixed seed, and the change compiles and passes with the `parallel` feature both on and off.
  2. `fdata_to_pc_1d` (`regression.rs:298`) computes its SVD via `faer` `thin_svd` on a zero-copy `MatRef::from_column_major_slice` view under `#[cfg(feature = "linalg")]`, with singular-vector sign conventions reconciled to match the existing output; a test confirms the faer-path `FpcaResult` matches the nalgebra-path `FpcaResult` within numerical tolerance (significant components within ~1e-8·σ₁, near-zero components treated as noise).
  3. The nalgebra SVD path is retained under `#[cfg(not(feature = "linalg"))]` so the `""` and `parallel` (non-`linalg`) builds are unchanged and still pass their tests.
  4. `cargo test -p fdars-core --features linalg`, `cargo test -p fdars-core --features parallel`, and `cargo clippy -p fdars-core --features linalg` all pass; no new external dependency is added (faer and rayon are already present).
**Plans**: TBD

Plans:
- [ ] 11-01: Parallelize `fclassif_cv` fold loop via `iter_maybe_parallel!(0..nfold)` + sequential-vs-parallel equivalence test (PERF-01, independent)
- [ ] 11-02: Swap FPCA SVD to faer `thin_svd` behind `linalg`, reconcile sign conventions, retain nalgebra path + equivalence test (PERF-02, independent)

**Note:** PERF-01 and PERF-02 touch different files (`classification/cv.rs` vs `regression.rs`) with no shared code — the two plans are fully parallelizable.

## Progress

**Execution Order:**
Phases execute in numeric order: 10 → 11. Phases 10 and 11 are mutually independent and may be executed in either order or concurrently.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 10. Capability Gaps (Spline Interp & Summary Stats) | v0.15.0 | 0/2 | Not started | - |
| 11. Performance Wins (Parallel CV & faer SVD) | v0.15.0 | 0/2 | Not started | - |

## Coverage

All 4 v1 requirements mapped to exactly one phase — no orphans, no duplicates.

| Requirement | Phase | Audit Backlog ID |
|-------------|-------|------------------|
| FEAT-01 | Phase 10 | REPR-02 (P1/S, score 4.00) |
| FEAT-02 | Phase 10 | EXPL-02 (P1/S, score 4.00) |
| PERF-01 | Phase 11 | PERF-PAR-CV (P2/S, score 4.00) |
| PERF-02 | Phase 11 | P6-1 (P2/S, score 3.00) |

**Coverage:** 4/4 v1 requirements mapped ✓

---
*Milestone v0.15.0 started 2026-08-10 — first implementation milestone (real `fdars-core/src/` changes). Phases 10–11 promoted from the v0.14.0 ranked backlog: FEAT-01 (spline interp), FEAT-02 (summary stats), PERF-01 (parallel CV), PERF-02 (faer SVD swap). Prior: v0.14.0 audit milestone shipped 2026-08-09 (9 phases, 13/13 requirements, milestone audit passed).*
