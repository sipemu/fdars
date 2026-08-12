---
gsd_state_version: 1.0
milestone: v0.17.0
milestone_name: Registration Parity & Elastic-FPCA Performance
status: planning
last_updated: "2026-08-12T07:43:15.979Z"
last_activity: 2026-08-12
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-11)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against scikit-fda — top audit-backlog items first.
**Current focus:** Phase 13 — Parity Quick Wins (FEAT-03/04 done; FEAT-05 scoring metrics remaining)

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-08-12 — Milestone v0.17.0 started

## Milestone Roadmap (v0.16.0)

Two phases, four backlog items: one P1 elastic-feasibility item (its own phase) plus three effort-S parity gaps (one phase, parallelizable plans).

| Phase | Requirements | Backlog IDs | Notes |
|-------|--------------|-------------|-------|
| 12 — Elastic Feasibility | PERF-03 | PERF-ELASTIC-BAND (P1/M; also P5-4) | Isolated to `alignment/`; banded variants exist — surface `band_frac`, keep unbanded exact |
| 13 — Parity Quick Wins | FEAT-03, FEAT-04, FEAT-05 | PREP-03, REPR-03, MISC-04 (all P2/S) | Imputation + `ExtrapolationPolicy` enum + scoring metrics; FEAT-03/04 share `helpers.rs` |

Phases 12 and 13 are mutually independent (disjoint files) and may be executed in either order or concurrently.

## Performance Metrics

**Velocity:**

- Total plans completed: 29 (25 in v0.14.0 + 4 in v0.15.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10 | v0.15.0 | 2 |
| 11 | v0.15.0 | 2 |

**Recent Trend:**

- Last 5 plans: 10-01, 10-02, 11-01, 11-02 (v0.15.0, all completed + verified)
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 12-elastic-feasibility-banded-alignment-default-band-frac P01 | 22 | 3 tasks | 6 files |
| Phase 13-parity-quick-wins P01 (FEAT-03 + FEAT-04) | 30 min | 3 tasks | 2 files |
| Phase 13-parity-quick-wins P02 (FEAT-05 scoring) | 10 min | 3 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Relevant to current work (v0.16.0 implementation):

- v0.16.0 is the second implementation milestone — real `fdars-core/src/` changes. Each phase requires inline `#[cfg(test)]` tests and numerical-equivalence/feasibility verification.
- Phase numbering continues from v0.15.0 (ended at Phase 11): v0.16.0 starts at Phase 12.
- Roadmap grouping: PERF-03 (banded elastic alignment, P1 headline) → its own Phase 12 (heavier, isolated to `alignment/`); the three effort-S parity gaps FEAT-03/FEAT-04/FEAT-05 → Phase 13 (parallelizable plans). Phases 12 and 13 are mutually independent (disjoint files).
- All four items carry exact file locations, root causes, and proposed API signatures from the v0.14.0 audit (`.planning/research/BACKLOG.md` — PERF-ELASTIC-BAND, PREP-03, REPR-03, MISC-04) — reuse, do not re-derive.

Conventions carried from prior milestones that constrain implementation:

- Column-major `FdMatrix`; all public functions return `Result<T, FdarError>` (never panic on input); feature-gated parallelism via `iter_maybe_parallel!` (5 macros in `parallel.rs`); `#[must_use]` on expensive computations (note: `Result<T, E>` already carries must_use — do not double-annotate); inline `#[cfg(test)]` tests.
- All four v0.16.0 items are additive/non-breaking: PERF-03's banded path is opt-in via `band_frac` (full unbanded path retained and exact); FEAT-03/04/05 are new functions/enum.

Phase-specific implementation notes from the audit:

- PERF-03 (`alignment/karcher.rs:300`, `elastic_self/cross_distance_matrix`): the banded variants (`karcher_mean_banded`, `elastic_self_distance_matrix_banded`, `elastic_cross_distance_matrix_banded`) already exist and are correct — `karcher_mean` currently hard-codes `band_frac=0.0` (→ `band_radius(0.0, m) = None` → full unbanded DP). Work is API surfacing/defaulting (`band_frac ≈ 0.1`), not a new algorithm. API-compat risk: adding a positional parameter is breaking — prefer `band_frac: Option<f64>` or an `ElasticConfig` field. Measured ~4–6× at representative cells; N=500,M=200 unbanded is infeasible (~700 s/iter).
- FEAT-03 (`helpers.rs` / `irreg_fdata/`): compose existing `helpers::linear_interp` into `impute_missing_values(data, argvals, method)` with `ImputationMethod` = Linear (mean/linear-interp) / Constant; scan each row for NaN, interpolate between bounding non-NaN neighbors; reject all-missing curves. ~1 week.
- FEAT-04 (`helpers.rs`): add `ExtrapolationPolicy` enum (`Boundary` clamp / `Exception` → `Err(FdarError)` / `Fill(f64)` / `Periodic` wrap); thread through `spline_interpolate` (from v0.15.0) and the existing linear path via a match at the boundary-check point. Small enum + dispatch, no new algorithm.
- FEAT-05 (new `scoring.rs`): `functional_mae`, `functional_mse`, `functional_mape`, `functional_msle`, `functional_explained_variance` over `FdMatrix` residuals — each a one-pass formula (~5–10 lines) with dimension validation; existing `r_squared`/`r_squared_adj` live in `helpers.rs` for reference.
- Wave/serialization: FEAT-03 and FEAT-04 both edit `helpers.rs` — sequence or serialize those two plan writes to avoid a merge collision; FEAT-05 (new `scoring.rs`) is fully independent.
- TMPDIR=/home/simonm/.cache/fdars-bench-tmp required for bench/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" — use `--no-verify` for docs, free /tmp before executing (MEMORY.md).
- [Phase ?]: Used *_with_band(Option<f64>) wrappers (not positional parameter addition) to keep non-breaking API per LOCKED decision in CONTEXT.md
- [Phase ?]: Added _banded variants for self/cross distance matrices to crate-root re-exports for complete discoverability
- [Phase 13-01]: Result<T,E>-returning fns must NOT have #[must_use] — clippy::double_must_use fires under -D warnings; confirmed by STATE.md convention note

### Pending Todos

None yet.

### Blockers/Concerns

- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer).
- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer).
- PERF-03 benchmarking is sensitive to OS scheduler jitter (unpinned governor); the audit flagged some elastic cells LOW-CONFIDENCE — prefer a stable feasibility demonstration (banded completes where unbanded is infeasible) over a precise speedup number.

## Deferred Items

Items acknowledged at v0.15.0 milestone close (2026-08-11):

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| tech-debt | Weakened MEWMA test assertion | Advisory code-review item carried forward | 2026-08-11 |
| tech-debt | `fix_svd_signs` NaN no-op | Advisory code-review item carried forward | 2026-08-11 |
| tech-debt | Over-broad test name (Phase 11) | Advisory code-review item carried forward | 2026-08-11 |
| validation | Phase 10 & 11 VALIDATION.md remain `draft` (Nyquist coverage TODO) | Carried forward | 2026-08-11 |

## Session Continuity

Last session: 2026-08-11T21:13:00.000Z
Stopped at: Completed 13-02-PLAN.md
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
