# Requirements: fdars — Milestone v0.17.0 Registration Parity & Elastic-FPCA Performance

**Defined:** 2026-08-12
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against scikit-fda — driven by the evidence-backed v0.14.0 audit backlog, top items first.

Continues the audit→implementation pipeline (v0.15.0, v0.16.0). All three items are the next tier of `.planning/research/BACKLOG.md` and carry exact file locations, root causes, and proposed API signatures from the v0.14.0 audit — reuse, do not re-derive. All are additive/non-breaking, following the established convention (inline `#[cfg(test)]` tests, `Result`-returning public functions, equivalence-tested performance changes, column-major `FdMatrix`, feature-gated parallelism via `iter_maybe_parallel!`).

## v1 Requirements

Requirements for milestone v0.17.0. Each maps to exactly one roadmap phase.

### Registration

- [ ] **FEAT-06**: A user can register a set of curves by rigid horizontal shift — `least_squares_shift_registration(data, argvals, ...)` aligns each curve to the sample mean by minimizing the L2 distance under a per-curve constant shift `δᵢ` (golden-section / ternary search over the objective, evaluated via linear interpolation), returning the registered curves plus the per-curve shift values. Fills the "simplest registration method" gap — fdars currently jumps from landmark shifts to full elastic SRSF warping. _(Backlog: PREP-04, rank 11, P1/M)_
- [ ] **FEAT-07**: A user can quantify registration quality with the three standard scikit-fda diagnostics — `least_squares_score` (∑‖registeredᵢ − mean‖²/n), `pairwise_correlation_score` (mean pairwise correlation between registered curves), and `sobolev_least_squares_score` (derivative-penalized LS) — added to `alignment/quality.rs` alongside the existing `alignment_quality` / `warp_complexity` / `warp_smoothness`. _(Backlog: PREP-05, rank 13, P2/S)_

### Performance

- [ ] **PERF-04**: The three per-curve elastic-FPCA loops in `elastic_fpca.rs` (lines 701 `shooting_vectors_from_psis`, 720 `build_augmented_srsfs`, 764 `svd_scores_and_eigenvalues`) run in parallel under the `parallel` feature via `iter_maybe_parallel!(0..n)`, producing output numerically equivalent to the sequential path (scores + eigenvalues within tolerance); the light `:764` body is guarded by a size threshold (N ≳ 50) or accepts a documented small-N regression. Projected ~4–5× at N≥50. _(Backlog: PERF-PAR-ELFPCA, rank 17, P2/M)_

## v2 Requirements

Deferred to future milestones (next-tier audit backlog items, not in this roadmap):

### Registration / Preprocessing

- **PREP-06**: Derivative-penalty (LDO) regularized FPCA — generalized eigenvalue problem via the existing `bspline_penalty_matrix` (rank 12, P1/M).
- **ACC-VALIDATE**: Comparative fdars-vs-scikit-fda numerical-accuracy validation, incl. the elastic level-encoding fix (GH #34) (rank 18, P2/M).

## Out of Scope

Explicitly excluded for this milestone.

| Feature | Reason |
|---------|--------|
| `FiniteElementBasis` / mesh-based registration | Requires a mesh data structure; large-effort, out of the registration-parity scope (REPR-01 defers FEBasis) |
| Regularized (LDO) FPCA | Deferred to v2 — larger generalized-eigensolver work; this milestone is registration + a targeted elastic-FPCA perf win |
| Numerical-accuracy validation vs scikit-fda (ACC-VALIDATE) | Deferred to v2 — a cross-cutting validation harness, separable from the registration capability gaps |
| Plotting/visualization parity with scikit-fda | A numeric Rust library does not need matplotlib-style output (standing project boundary) |
| Releasing v0.16.0 (version bump + PR + tag) | Separate ship step, not a milestone requirement — tracked in PROJECT.md Current State |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FEAT-06 | Phase 14 | Pending |
| FEAT-07 | Phase 14 | Pending |
| PERF-04 | Phase 15 | Pending |

**Coverage:**
- v1 requirements: 3 total
- Mapped to phases: 3
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-12*
*Last updated: 2026-08-12 after initial definition (milestone v0.17.0)*
