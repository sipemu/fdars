# Requirements: fdars — v0.15.0 Top-Backlog Quick Wins

**Defined:** 2026-08-10
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against scikit-fda — top audit-backlog items first.

> This milestone implements the four highest-value, low-effort items (score 3.00–4.00, effort S) from the v0.14.0 audit backlog (`.planning/milestones/v0.14.0-*`). Unlike v0.14.0 (audit-only), this milestone makes real `fdars-core/src/` changes. Each REQ links back to its audit backlog ID for traceability.

## v1 Requirements

Requirements for the v0.15.0 release. Each maps to exactly one roadmap phase.

### Capabilities

- [x] **FEAT-01**: A caller can interpolate functional data at arbitrary off-grid query points using cubic / order-k spline interpolation — a new `spline_interpolate(data, argvals, query_points, order) -> Result<FdMatrix, FdarError>` in `helpers.rs` that fits a B-spline (reusing the existing `basis/` B-spline system) per curve and evaluates at the query points. Replaces the current linear-only interpolation. *(audit: REPR-02, P1/S)*
- [x] **FEAT-02**: A caller can compute functional descriptive statistics via public functions — `trim_mean` (depth-trimmed mean), `depth_based_median` (index of deepest curve), `functional_covariance` (M×M sample covariance), `functional_variance` (pointwise), and `functional_std` (pointwise) — over an `FdMatrix`. *(audit: EXPL-02, P1/S)*

### Performance

- [ ] **PERF-01**: `fclassif_cv` executes its cross-validation fold loop in parallel via `iter_maybe_parallel!(0..nfold)` when the `parallel` feature is enabled, producing fold results identical to sequential execution. *(audit: PERF-PAR-CV, P2/S — ~4–5× projected)*
- [ ] **PERF-02**: `fdata_to_pc_1d` computes its FPCA SVD via faer `thin_svd` on a zero-copy `MatRef` view under the `linalg` feature, producing an `FpcaResult` equivalent (within numerical tolerance, sign conventions reconciled) to the existing nalgebra path, which is retained for non-`linalg` builds. *(audit: P6-1, P2/S — 1.8–4.1× measured)*

## v2 Requirements

Deferred to future milestones — remaining audit backlog items, tracked but not in this roadmap.

### Capabilities (deferred)

- **REPR-03**: `ExtrapolationPolicy` enum (Boundary / Error / Fill / Periodic) threaded through interpolation (audit: REPR-03, P2/S).
- **EXPL-03**: Stahel-Donoho outlyingness for functional data (audit: EXPL-03, P3/M).
- **PREP-04 / PREP-06**: Shift-only (LeastSquaresShift) registration; LDO-regularized FPCA (audit: P1/M each).

### Performance (deferred)

- **PERF-ELASTIC-BAND**: Default/expose banded elastic alignment (audit: PERF-ELASTIC-BAND, P1/M).
- **PERF-PAR-ELFPCA**: Parallelize the three elastic-FPCA inner N-loops (audit: PERF-PAR-ELFPCA, P2/M).
- Remaining ranked backlog items in `.planning/research/BACKLOG.md`.

## Out of Scope

Explicitly excluded from v0.15.0 to keep the milestone a tight, high-value slice.

| Feature | Reason |
|---------|--------|
| Any backlog item below the top-4 quick wins | Deferred to future milestones; this milestone is scoped to the four score-3.00–4.00 / effort-S items |
| New external dependencies | faer already present; no new crates needed for these 4 items |
| Breaking API changes | All four are additive (new functions) or behavior-preserving (SVD backend swap, internal parallelism) |
| Plotting / visualization parity | A numeric Rust crate does not need matplotlib-style output (carried over from v0.14.0 scope) |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FEAT-01 | Phase 10 | Complete |
| FEAT-02 | Phase 10 | Complete |
| PERF-01 | Phase 11 | Pending |
| PERF-02 | Phase 11 | Pending |

**Coverage:**

- v1 requirements: 4 total
- Mapped to phases: 4 (100%) ✓
- Unmapped: 0

---
*Requirements defined: 2026-08-10*
*Last updated: 2026-08-10 after roadmap creation (Phases 10–11 mapped)*
