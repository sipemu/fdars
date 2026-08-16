---
gsd_state_version: 1.0
milestone: v0.19.0
milestone_name: Functional Inference Suite
current_phase: 21
current_phase_name: Functional-Linear-Model Inference
status: executing
stopped_at: Completed 21-01-PLAN.md
last_updated: "2026-08-16T07:35:24.358Z"
last_activity: 2026-08-16
last_activity_desc: Phase 21 plan 01 executed (INF-02 FLM inference) — flm_f_test, flm_gof_test, oneway_anova_vstat
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 2
  completed_plans: 2
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-15)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone gives fdars its first standalone functional-inference surface (R-parity Area 5, currently 0/22 present), promoted top-first from the v0.18.0 R-ecosystem backlog.
**Current focus:** Phase 20 — Two-Sample Functional Tests & `inference/` Module

## Current Position

Phase: 21 — Functional-Linear-Model Inference
Plan: 01 — complete (INF-02)
Status: Phase 21 complete — ready to verify / close milestone
Last activity: 2026-08-16 — Phase 21 plan 01 executed (flm_f_test, flm_gof_test, oneway_anova_vstat)

## Milestone Roadmap (v0.19.0)

Two sequential phases, two requirements. First **implementation** milestone from the R-ecosystem backlog — real `fdars-core/src/` code. All additions are additive/non-breaking, `Result`-returning, with inline `#[cfg(test)]` tests and crate-root re-exports; **zero changes to existing public signatures.** Closes the two P1 table-stakes items in R-parity Area 5 (Inference).

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 20 — Two-Sample Functional Tests & `inference/` Module | INF-01 | Create new `fdars-core/src/inference/` module with standalone two-sample tests: `t_perm_test`, `f_perm_test` (`tperm.fd`/`Fperm.fd` analogs), a two-sample mean/covariance-equality test, and `mean_scb` (simultaneous confidence bands) + an SCB-based two-sample test. **Reuse-first:** lift `function_on_scalar` permutation machinery, expose `spm::stats::hotelling_t2` as the two-sample mean test, reuse `tolerance/degras` bootstrap-band code for the SCB. Builds the module scaffolding INF-02 depends on. |
| 21 — Functional-Linear-Model Inference | INF-02 | Add formal FLM inference on a fitted `FregreLmResult`: `flm_gof_test` (goodness-of-fit) + `flm_f_test` (F-test), both residual-based against the FLM null; plus an asymptotic `oneway_anova_vstat` one-way functional ANOVA V-statistic **alongside** the existing permutation ANOVA (`function_on_scalar::fanova`, not replaced). **Reuse-first:** consume fitted-model residuals + integration weights already available. **Depends on Phase 20** (shares the `inference/` module scaffolding). |

**Execution order:** 20 → 21 (strict — INF-02's FLM inference reuses the `inference/` module created in Phase 20).

## Performance Metrics

**Velocity:**

- Total plans completed: 36 (25 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10–11 | v0.15.0 | 4 |
| 12–13 | v0.16.0 | 4 |
| 14–15 | v0.17.0 | 3 |
| 16–19 | v0.18.0 | 5 |

**Recent Trend:**

- Last 5 plans: 16-01, 16-02, 17-01, 18-01, 19-01 (v0.18.0) — all completed + verified
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 20 P01 | 1 session | 3 tasks | 6 files |
| Phase 21 P01 | 1h20m | 4 tasks | 7 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Relevant to current work (v0.19.0 implementation):

- v0.19.0 is an **implementation** milestone — real `fdars-core/src/` code (the first drawn from the R-ecosystem backlog). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is the **two P1 table-stakes inference items** (INF-01, INF-02). INF-03 (ITP family, P2 differentiator) is **deferred** to v2.
- Phase numbering **continues** from v0.18.0 (ended at Phase 19) → v0.19.0 starts at Phase 20.
- **INF-01 first, INF-02 second** — INF-01 creates the `fdars-core/src/inference/` module scaffolding + two-sample tests; INF-02's FLM inference reuses that module (strict 20 → 21 order).
- **Reuse-first mandate (from R-BACKLOG.md):**
  - INF-01 reuses `function_on_scalar` permutation machinery (lift into standalone `t_perm_test`/`f_perm_test`), exposes `spm::stats::hotelling_t2` as the two-sample mean test, and reuses `tolerance/degras` bootstrap-band code for `mean_scb`.
  - INF-02 reuses fitted `FregreLmResult` residuals + integration weights, and adds the asymptotic V-statistic ANOVA **alongside** the existing permutation ANOVA in `function_on_scalar.rs` (not a replacement).
- R baselines matched by **capability** (test statistic + p-value + decision), not R's exact signatures: `fda::Fperm.fd`/`tperm.fd`, `fda.usc` mean/cov equality + FLM GoF/F-test, `SCBmeanfd`, `fdatest`/`fdANOVA` V-statistic.
- Crate is at 0.17.0 (v0.18.0 was audit-only); the crate-version bump to 0.19.0 is a **ship-time** decision, not part of the implementation phases.

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` for reproducible permutation tests.
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- TMPDIR=/home/simonm/.cache/fdars-bench-tmp required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code, so this pointer matters again.
- [Phase ?]: Phase 20 INF-01: inference/ module reuses fanova/hotelling_t2/scb_mean_degras; self-contained chi-square SF avoids a statrs dependency
- [Phase ?]: flm_gof_test uses F-form Ramsey-RESET residual lack-of-fit; oneway_anova_vstat uses scaled-chi2 (Box/Satterthwaite) V-null; F-dist SF self-contained via incomplete beta (no new dep)

### Pending Todos

None yet.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer).
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant again now that the milestone compiles real code + doctests.
- Permutation tests are randomized — inline `#[cfg(test)]` assertions must seed deterministically (per-thread RNG convention) and assert on p-value **direction/threshold** (reject vs. fail-to-reject), not exact values, to avoid flaky tests.

## Deferred Items

v2 backlog items deferred at v0.19.0 definition (2026-08-15): INF-03 (Interval Testing Procedure / ITP family — P2 differentiator; depends on the INF-01 `inference/` scaffolding). Larger R-parity clusters (T-01/T-02 quick wins, REG-01 concurrent regression, REG-02 functional GLM families, FPCA-01 PACE sparse FPCA, FTS-*, FRE-*, etc.) remain ranked in `.planning/research/R-BACKLOG.md` — see REQUIREMENTS.md v2 section.

Advisory tech-debt carried forward (not v0.19.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; Phases 10/11/14/15 VALIDATION.md remain `draft` (Nyquist TODO).

## Session Continuity

Last session: 2026-08-16T07:35:18.365Z
Stopped at: Completed 21-01-PLAN.md
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 20`
