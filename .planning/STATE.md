---
gsd_state_version: 1.0
milestone: v0.15.0
milestone_name: Top-Backlog Quick Wins
current_phase: 10
current_phase_name: Capability Gaps — Spline Interpolation & Functional Summary Statistics
status: "Milestone v0.15.0 shipped — PR #38 (phases 10+11)"
stopped_at: Completed 11-02-faer-fpca-svd-PLAN.md
last_updated: "2026-08-11T16:31:20.481Z"
last_activity: 2026-08-11
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 4
  completed_plans: 4
  percent: 100
last_activity_desc: Phase 11 execution started
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-10)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against scikit-fda — top audit-backlog items first.
**Current focus:** Phase 11 — performance-wins-parallel-cv-folds-faer-fpca-svd

## Current Position

Phase: 10 — Capability Gaps — Spline Interpolation & Functional Summary Statistics
Plan: Not started
Status: Milestone v0.15.0 shipped — PR #38 (phases 10+11)
Last activity: 2026-08-11

## Milestone Roadmap (v0.15.0)

Two phases, four independent effort-S backlog items — no cross-dependencies. Each phase's two plans are fully parallelizable.

| Phase | Requirements | Backlog IDs | Plans |
|-------|--------------|-------------|-------|
| 10 — Capability Gaps | FEAT-01, FEAT-02 | REPR-02, EXPL-02 | 10-01 (spline interp), 10-02 (summary stats) |
| 11 — Performance Wins | PERF-01, PERF-02 | PERF-PAR-CV, P6-1 | 11-01 (parallel CV), 11-02 (faer SVD swap) |

Phases 10 and 11 are mutually independent and may be executed in either order or concurrently.

## Performance Metrics

**Velocity:**

- Total plans completed: 25 (v0.14.0)
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase (v0.14.0):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | - | - |
| 02 | 2 | - | - |
| 03 | 2 | - | - |
| 04 | 3 | - | - |
| 05 | 3 | - | - |
| 06 | 1 | - | - |
| 07 | 2 | - | - |
| 08 | 3 | - | - |
| 09 | 3 | - | - |
| 10 | 2 | - | - |
| 11 | 2 | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 10 P01 | 8 | 3 tasks | 2 files |
| Phase 10 P02 | 9 | 3 tasks | 2 files |
| Phase 11 P01 | 7m | 2 tasks | 1 files |
| Phase 11 P02 | 15 | 3 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Relevant to current work (v0.15.0 implementation):

- v0.15.0 is the first implementation milestone — real `fdars-core/src/` changes (v0.14.0 was audit-only). Each phase requires inline `#[cfg(test)]` tests and numerical-equivalence/accuracy verification.
- Phase numbering continues from v0.14.0 (ended at Phase 9): v0.15.0 starts at Phase 10.
- Roadmap grouping: 2 capability items (FEAT-01/FEAT-02) → Phase 10; 2 performance items (PERF-01/PERF-02) → Phase 11. All four are independent effort-S items with no cross-dependencies (parallelizable within and across phases).
- All four items carry exact file locations, root causes, and proposed API signatures from the v0.14.0 audit (`.planning/research/BACKLOG.md`) — reuse, do not re-derive.

Conventions carried from the v0.14.0 audit that constrain implementation:

- Column-major `FdMatrix`; all public functions return `Result<T, FdarError>` (never panic on input); feature-gated parallelism via `iter_maybe_parallel!` (5 macros in `parallel.rs`); `#[must_use]` on expensive computations; inline `#[cfg(test)]` tests.
- PERF-02: faer `thin_svd` path under `#[cfg(feature = "linalg")]`; retain the nalgebra `SVD::new` path under `#[cfg(not(feature = "linalg"))]`. faer measured 1.8–4.1× over nalgebra at fdars' real FPCA sizes.
- PERF-02 sign conventions: singular-vector signs may differ from nalgebra — a one-time equivalence check with a significant-values filter (~1e-8·σ₁; near-zero values are noise in both backends) is required. `svd_equivalence` integration-test pattern already exists from Phase 6.
- PERF-01: `fclassif_cv` fold loop (`classification/cv.rs:76`) folds are fully independent; fold-assignment RNG runs once before the loop, so no per-thread seeding is needed. Swap `for fold in 0..nfold` → `iter_maybe_parallel!(0..nfold)` with a `.map().collect()` result pattern.
- FEAT-01: `spline_interpolate` reuses the existing `basis/` B-spline system (de Boor evaluation over stored coefficients); ~80–120 lines in `helpers.rs`.
- FEAT-02: `functional_variance`/`functional_std` are one-pass O(n·m); `functional_covariance` is O(n·m²); `depth_based_median`/`trim_mean` call the existing `depth/` functions.
- TMPDIR=/home/simonm/.cache/fdars-bench-tmp required for bench/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" — use `--no-verify` for docs, free /tmp before executing (MEMORY.md).
- [Phase ?]: Removed #[must_use] from spline_interpolate: Result<T, E> already carries must_use; clippy double_must_use lint confirms removal is correct
- [Phase ?]: spline_interpolate uses nalgebra SVD pseudoinverse inline (not pub(super) svd_pseudoinverse from basis/helpers.rs) to avoid crossing module visibility boundaries
- [Phase ?]: functional_std delegates to functional_variance: guarantees std^2==var by construction
- [Phase ?]: Symmetric M×M covariance computes upper triangle only and mirrors (halves inner-product work)
- [Phase ?]: alpha guard uses Range::contains per clippy::manual_range_contains lint
- [Phase ?]: Use iter_maybe_parallel!(0..nfold).map(...).collect() for parallel CV folds in fclassif_cv; no direct rayon import for iteration, only ParallelIterator trait under cfg(feature=parallel)
- [Phase ?]: PERF-02: faer thin_svd FPCA path (zero-copy MatRef) under linalg; fix_svd_signs shared across both backends; nalgebra retained under cfg(not linalg)

### Pending Todos

None yet.

### Blockers/Concerns

- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer).
- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer).

## Deferred Items

Items acknowledged at v0.14.0 milestone close (2026-08-09):

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| uat | 07-UAT.md | passed, 0 open scenarios — flagged by artifact audit only for file presence; no action required | 2026-08-09 |

## Session Continuity

Last session: 2026-08-11T05:29:31.149Z
Stopped at: Completed 11-02-faer-fpca-svd-PLAN.md
Resume file: None

## Operator Next Steps

- Review the roadmap: `.planning/ROADMAP.md` (Phases 10–11).
- Plan the first phase: `/gsd-plan-phase 10`.
