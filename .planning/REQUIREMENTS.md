# Requirements: fdars — v0.16.0 Elastic Feasibility + Parity Quick Wins

**Defined:** 2026-08-11
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against scikit-fda — top audit-backlog items first.

> This milestone promotes the next tier of the v0.14.0 audit backlog (`.planning/milestones/v0.14.0-*`, ranked in `.planning/research/BACKLOG.md`) after v0.15.0 shipped ranks 1–4. It pairs the single highest-*value* remaining item — banded elastic alignment (P1, value 5), which makes currently-infeasible large-grid workloads tractable — with three effort-S scikit-fda parity gaps. Each REQ links back to its audit backlog ID for traceability. All changes are additive/non-breaking, respect the column-major `FdMatrix` layout and feature-gated parallelism model, and every public function returns `Result<T, FdarError>`.

## v1 Requirements

Requirements for the v0.16.0 release. Each maps to exactly one roadmap phase.

### Performance / Feasibility

- [x] **PERF-03**: Elastic alignment can run on a **banded** dynamic-programming path so large grids that are currently infeasible (e.g. N=500, M=200) complete in tractable time. A `band_frac` (Sakoe-Chiba-style band width as a fraction of M) is exposed through the elastic alignment config and threaded into the DP warp search in `alignment/` (`karcher.rs`, `elastic_self_distance_matrix` / `elastic_cross_distance_matrix`); the full (unbanded) path remains available. Alignment output with a sufficiently wide band matches the unbanded result within numerical tolerance, verified by an inline test; a benchmark or timing test demonstrates the feasibility improvement at a large (N, M). *(audit: PERF-ELASTIC-BAND, rank 10, P1/M — ~4–6× measured; also P5-4)*

### Capabilities

- [ ] **FEAT-03**: A caller can impute missing values (`NaN`) in a regular-grid `FdMatrix` via a new public function returning `Result<FdMatrix, FdarError>`, with at least mean/linear-interpolation strategies over each curve; input validation rejects all-missing curves or unsupported configurations. Inline tests verify imputation reproduces known values on synthetic gaps and errors on invalid input. *(audit: PREP-03, rank 5, P2/S — `helpers.rs`, `irreg_fdata/`)*
- [ ] **FEAT-04**: Interpolation/evaluation of functional data accepts a composable **`ExtrapolationPolicy`** enum — `Boundary` (clamp to nearest edge), `Exception` (return `FdarError` for out-of-range queries), `Fill(value)` (constant fill), and `Periodic` (wrap) — controlling behavior for query points outside `argvals`. It threads through `spline_interpolate` (and the existing linear path); inline tests exercise each variant, including the error path for `Exception`. *(audit: REPR-03, rank 6, P2/S — `helpers.rs` interpolation/evaluation paths)*
- [ ] **FEAT-05**: A caller can score functional predictions against observations via public metric functions over `FdMatrix` — `functional_mae`, `functional_mse` (and `functional_mape`, `functional_msle`, `functional_explained_variance`) — each returning `Result<_, FdarError>` with dimension validation. Inline tests verify each against a hand-computed reference. *(audit: MISC-04, rank 9, P2/S — new `scoring.rs`)*

## v2 Requirements

Deferred to future milestones — remaining audit backlog items, tracked but not in this roadmap.

### Capabilities (deferred)

- **PREP-04 / PREP-06**: Shift-only (LeastSquaresShift) registration; LDO-regularized FPCA (audit: P1/M each).
- **INF-01 / INF-02**: Asymptotic functional ANOVA V-statistic; two-sample Hotelling T² inference wrapper (audit: P2/S each).
- **REPR-01 / EXPL-01 / ML-01 / ML-02**: New basis types; pluggable-metric depth; extra classifiers; LDO-regularized regression (audit: P2/M each).

### Performance (deferred)

- **PERF-PAR-ELFPCA**: Parallelize the three elastic-FPCA inner N-loops (audit: P2/M).
- **PERF-FPCA-TRUNCSVD**: Truncated SVD computing only `ncomp` components in FPCA (audit: P2/L).
- Remaining ranked backlog items in `.planning/research/BACKLOG.md`.

## Out of Scope

Explicitly excluded from v0.16.0 to keep the milestone a tight, high-value slice.

| Feature | Reason |
|---------|--------|
| Backlog items below this tier | Deferred; this milestone is scoped to the P1 elastic-feasibility item plus three effort-S parity gaps |
| New external dependencies | The four items use existing crates (rayon/faer/nalgebra already present) |
| Breaking API changes | All four are additive (new functions/enums) or behavior-preserving (banded path is opt-in via `band_frac`, full path retained) |
| Plotting / visualization parity | A numeric Rust crate does not need matplotlib-style output (carried over from v0.14.0 scope) |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PERF-03 | Phase 12 | Complete |
| FEAT-03 | Phase 13 | Pending |
| FEAT-04 | Phase 13 | Pending |
| FEAT-05 | Phase 13 | Pending |
