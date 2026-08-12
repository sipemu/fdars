---
gsd_state_version: 1.0
milestone: v0.17.0
milestone_name: Registration Parity & Elastic-FPCA Performance
status: planning
last_updated: "2026-08-12T08:15:00.000Z"
last_activity: 2026-08-12
progress:
  total_phases: 2
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-12)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against scikit-fda — top audit-backlog items first.
**Current focus:** Phase 14 — Shift Registration (FEAT-06 least-squares shift registration + FEAT-07 registration-quality scores). Roadmap defined; ready to plan Phase 14.

## Current Position

Phase: 14 — Shift Registration (not started)
Plan: —
Status: Roadmap created; awaiting phase planning
Last activity: 2026-08-12 — Roadmap for v0.17.0 created (Phases 14–15, 3 requirements, 100% coverage)

## Milestone Roadmap (v0.17.0)

Two phases, three backlog items. Phase 14 pairs the two registration items (they share `alignment/`; FEAT-07 scores the FEAT-06 output). Phase 15 isolates the elastic-FPCA perf win.

| Phase | Requirements | Backlog IDs | Notes |
|-------|--------------|-------------|-------|
| 14 — Shift Registration | FEAT-06, FEAT-07 | PREP-04 (P1/M), PREP-05 (P2/S) | New `least_squares_shift_registration` in `alignment/` (golden-section L2-to-mean, returns registered curves + per-curve δᵢ) + three quality scores (`least_squares_score` / `pairwise_correlation_score` / `sobolev_least_squares_score`) in `alignment/quality.rs`. FEAT-07 scores FEAT-06 output — natural pair. |
| 15 — Elastic-FPCA Performance | PERF-04 | PERF-PAR-ELFPCA (P2/M) | Parallelize the three per-curve loops in `elastic_fpca.rs:701/720/764` via `iter_maybe_parallel!(0..n)`; numerically equivalent to sequential; light `:764` guarded by N ≳ 50 threshold. Isolated to `elastic_fpca.rs`. |

Phases 14 and 15 are mutually independent (disjoint files: `alignment/` vs `elastic_fpca.rs`) and may be executed in either order or concurrently.

## Performance Metrics

**Velocity:**

- Total plans completed: 33 (25 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10 | v0.15.0 | 2 |
| 11 | v0.15.0 | 2 |
| 12 | v0.16.0 | 1 |
| 13 | v0.16.0 | 3 |

**Recent Trend:**

- Last 5 plans: 11-01, 11-02 (v0.15.0), 12-01, 13-01, 13-02 (v0.16.0) — all completed + verified
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Relevant to current work (v0.17.0 implementation):

- v0.17.0 is the third implementation milestone — real `fdars-core/src/` changes. Each phase requires inline `#[cfg(test)]` tests and numerical-equivalence / feasibility verification.
- Phase numbering continues from v0.16.0 (ended at Phase 13): v0.17.0 starts at Phase 14.
- Roadmap grouping: FEAT-06 (shift registration) + FEAT-07 (registration-quality scores) → Phase 14 (both live in `alignment/`; FEAT-07 scores the FEAT-06 output — natural pair). PERF-04 (parallelize elastic-FPCA loops) → Phase 15 (isolated to `elastic_fpca.rs`). Phases 14 and 15 are mutually independent (disjoint files).
- All three items carry exact file locations, root causes, and proposed API signatures from the v0.14.0 audit (`.planning/research/BACKLOG.md` — PREP-04, PREP-05, PERF-PAR-ELFPCA) — reuse, do not re-derive.
- Research intentionally skipped for this milestone — the v0.14.0 audit already researched these items; the backlog carries signatures + scikit-fda references (no SUMMARY.md).

Conventions carried from prior milestones that constrain implementation:

- Column-major `FdMatrix`; all public functions return `Result<T, FdarError>` (never panic on input); feature-gated parallelism via `iter_maybe_parallel!` (5 macros in `parallel.rs`); inline `#[cfg(test)]` tests; crate-root re-export of new public functions.
- `Result<T, E>`-returning fns must NOT carry `#[must_use]` — `clippy::double_must_use` fires under `-D warnings` (confirmed convention note, Phase 13-01).
- All three v0.17.0 items are additive/non-breaking: FEAT-06/FEAT-07 add new functions (existing `alignment/` signatures untouched); PERF-04 is an internal parallelization (public `vert_fpca`/`joint_fpca` signatures unchanged, no new deps).

Phase-specific implementation notes from the audit:

- FEAT-06 / Phase 14 (`alignment/`, PREP-04): implement `least_squares_shift_registration(data, argvals, ...)` — minimize `‖curveᵢ(t − δᵢ) − mean(t)‖²` per curve via golden-section / ternary search (each objective eval via linear interpolation); mean via `fdata::functional_mean`. Return registered curves + per-curve shifts `δᵢ`. `landmark_shift_deltas` already exists internally inside `landmark_register` (reference, not returned). ~1 wk equiv (M with scikit-fda comparison).
- FEAT-07 / Phase 14 (`alignment/quality.rs`, PREP-05): add `least_squares_score` (∑‖registeredᵢ − mean‖²/n), `pairwise_correlation_score` (mean pairwise correlation), `sobolev_least_squares_score` (derivative-penalized LS) alongside existing `alignment_quality` / `warp_complexity` / `warp_smoothness`. Effort S. Scores the FEAT-06 output — verify direction (registration lowers LS score, raises correlation).
- PERF-04 / Phase 15 (`elastic_fpca.rs:701/720/764`, PERF-PAR-ELFPCA): wrap each of the three per-curve `for i in 0..n` loops in `iter_maybe_parallel!(0..n)`. `:701` shooting-vector row / `:720` augmented-SRSF row → `.collect::<Vec<_>>()` + row-assignment; `:764` score extraction is a light body → guard behind N ≳ 50 threshold (per SC1 payback rule) or accept documented small-N regression. Equivalence-test scores + eigenvalues (elastic geometry is FP-order-sensitive). No RNG in any of the three bodies → no per-thread seeding.
- Wave/serialization: FEAT-06 (new registration fn in `alignment/`) and FEAT-07 (`alignment/quality.rs`) touch different files within `alignment/` — mostly parallelizable, but if both add crate-root re-exports, serialize the `lib.rs`/`mod.rs` re-export edits to avoid a merge collision. PERF-04 (`elastic_fpca.rs`) is fully independent.
- TMPDIR=/home/simonm/.cache/fdars-bench-tmp required for bench/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" — use `--no-verify` for docs, free /tmp before executing (MEMORY.md).

### Pending Todos

None yet.

### Blockers/Concerns

- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer).
- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer).
- PERF-04 benchmarking is sensitive to OS scheduler jitter (unpinned governor); the audit flagged elastic cells LOW-CONFIDENCE — prefer an equivalence/feasibility demonstration (parallel matches sequential within tolerance at N ≥ 50) over a precise speedup number.
- v0.16.0 release still pending (version bump 0.15.0 → 0.16.0 + PR to protected `main` + tag) — tracked in PROJECT.md Current State, not a v0.17.0 requirement.

## Deferred Items

Items acknowledged at v0.15.0 milestone close (2026-08-11):

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| tech-debt | Weakened MEWMA test assertion | Advisory code-review item carried forward | 2026-08-11 |
| tech-debt | `fix_svd_signs` NaN no-op | Advisory code-review item carried forward | 2026-08-11 |
| tech-debt | Over-broad test name (Phase 11) | Advisory code-review item carried forward | 2026-08-11 |
| validation | Phase 10 & 11 VALIDATION.md remain `draft` (Nyquist coverage TODO) | Carried forward | 2026-08-11 |

v2 backlog items deferred at v0.17.0 definition (2026-08-12): PREP-06 (LDO-regularized FPCA), ACC-VALIDATE (fdars-vs-scikit-fda accuracy validation) — see REQUIREMENTS.md v2 section.

## Session Continuity

Last session: 2026-08-12T08:15:00.000Z
Stopped at: Roadmap for v0.17.0 created (Phases 14–15)
Resume file: None

## Operator Next Steps

- Plan Phase 14 with `/gsd-plan-phase 14` (Shift Registration — FEAT-06 + FEAT-07)
