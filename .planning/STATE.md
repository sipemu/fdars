---
gsd_state_version: 1.0
milestone: v0.20.0
milestone_name: Table-Stakes Quick Wins
status: Awaiting next milestone
stopped_at: Completed 22-01-PLAN.md
last_updated: "2026-08-16T20:04:41.782Z"
last_activity: 2026-08-16
last_activity_desc: Milestone v0.20.0 roadmap created (Phases 22–23)
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 2
  completed_plans: 2
  percent: 100
current_phase: 23
current_phase_name: Functional Boxplot & Depth Dispatcher
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-16)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone ships the two top-ranked (score 5.00, P1 table-stakes, S-effort) R-parity quick wins from `.planning/research/R-BACKLOG.md`, each closing a baseline capability gap by wrapping existing fdars infrastructure.
**Current focus:** Phase 22 — Constant Basis & AIC Smoothing Selection (T-01)

## Current Position

Phase: Milestone v0.20.0 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-08-16 — Milestone v0.20.0 completed and archived

## Milestone Roadmap (v0.20.0)

Two **independent** phases, two requirements. Second implementation milestone from the R-ecosystem backlog — real `fdars-core/src/` code. All additions are additive/non-breaking, `Result`-returning, with inline `#[cfg(test)]` tests and crate-root re-exports; **zero changes to existing public signatures.** Both items are S-effort and **wrap existing infrastructure** (low risk); mirrors the v0.15.0 quick-wins pattern.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 22 — Constant Basis & AIC Smoothing Selection | T-01 | Add a named `constant_basis` / `ConstantBasis` constructor to `basis/` (single intercept column + zero roughness penalty, usable in regression design matrices), and add an AIC criterion (AIC = n·log(RSS/n) + 2·tr(H)) to the automatic smoothing-parameter selector (`smooth_basis` / `smoothing`, which do GCV/CV only today). **Reuse-first:** reuses the existing basis system + the hat-matrix trace already computed for GCV. No new algorithm. |
| 23 — Functional Boxplot & Depth Dispatcher | T-02 | Add the López-Pintado depth-fence `functional_boxplot` (numeric outputs only: median curve, central region = inner 50% by depth, 1.5×IQR-of-depths whisker/fence, per-curve outlier flags — no plotting) in `outliers`/`depth`, and a unified `DepthMethod` enum + `functional_depth(data, method)` dispatcher over the existing depth functions (`fraiman_muniz_1d`, `band_1d`, `modified_band_1d`, `random_projection_1d`, …). **Reuse-first:** wraps existing depth code; dispatcher mirrors the `CovType`/`ProjectionBasisType` enum-dispatch convention. |

**Execution order:** 22 and 23 are **mutually independent** (disjoint modules: `basis/`+`smoothing` vs `depth/`+`outliers`). Either may execute first; they may run in parallel. Default order 22 → 23 by backlog rank (both score 5.00; T-01 rank 1, T-02 rank 2).

## Performance Metrics

**Velocity:**

- Total plans completed: 38 (21 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0 + 2 in v0.19.0)
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
| 20–21 | v0.19.0 | 2 |

**Recent Trend:**

- Last 5 plans: 17-01, 18-01, 19-01, 20-01, 21-01 — all completed + verified
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 22 P01 | 18min | 3 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Relevant to current work (v0.20.0 implementation):

- v0.20.0 is an **implementation** milestone — real `fdars-core/src/` code (the second drawn from the R-ecosystem backlog, after v0.19.0). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is the **two top-ranked (score 5.00) P1 table-stakes S-effort quick wins** (T-01, T-02). Both **wrap existing infrastructure** — low risk; mirrors the v0.15.0 quick-wins pattern.
- Phase numbering **continues** from v0.19.0 (ended at Phase 21) → v0.20.0 starts at Phase 22. No reset.
- **T-01 and T-02 are mutually independent** — they touch **disjoint modules** (`basis/`+`smoothing`/`smooth_basis` vs `depth/`+`outliers`). Two phases (22 = T-01, 23 = T-02); no cross-phase dependency, so they may execute in either order or in parallel. Default order 22 → 23 by backlog rank.
- **Reuse-first mandate (from R-BACKLOG.md T-01/T-02 blocks):**
  - T-01 reuses the existing `basis/` system (constant basis = single-column design + zero roughness penalty) and the hat-matrix trace already computed for GCV (AIC = n·log(RSS/n) + 2·tr(H)); no new algorithm.
  - T-02 wraps the existing depth functions (`fraiman_muniz_1d`, `band_1d`, `modified_band_1d`, `random_projection_1d`, …) behind a `DepthMethod`-dispatched `functional_depth`, and adds the canonical depth-fence `functional_boxplot` as numeric outputs (central region + 1.5×IQR-of-depths whisker + outlier flags).
- R baselines matched by **capability**, not R's exact signatures: `fda`/`fda.usc` (constant basis, `akaike_information_criterion` smoothing); `roahd`/`fdaoutlier`/`fda.usc` (functional boxplot fences + general depth dispatcher).
- **Plotting is out of scope** — T-02 delivers the *numeric* central-region/whisker/outlier outputs only (consistent with the R audit's plotting exclusion). T-01 adds **AIC only** (mainstream criterion); FPE/Shibata/Rice are a separate lower-ranked backlog item.
- Crate is at 0.17.0 (v0.18.0 was audit-only; v0.19.0 shipped code but the release is still pending). The crate-version bump to 0.20.0 (and the pending 0.19.0 release) is a **ship-time** decision, not part of the implementation phases.

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` where randomness is involved.
- Enum-dispatch convention already established (`CovType`, `ProjectionBasisType`, `DepthMethod`-style) — the T-02 `DepthMethod` dispatcher follows it.
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- TMPDIR=/home/simonm/.cache/fdars-bench-tmp required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code + doctests, so this pointer matters.

### Pending Todos

None yet.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer).
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant since the milestone compiles real code + doctests.
- The functional-boxplot fence definition (López-Pintado 1.5×IQR-of-depths) has minor R-implementation variants (`roahd` vs `fdaoutlier` vs `fda.usc`); pin one convention during planning and document the choice in rustdoc (as prior milestones documented divergences).

## Deferred Items

v2 backlog items (from `.planning/research/R-BACKLOG.md`, see REQUIREMENTS.md v2 section): REG-01 (concurrent/varying-coefficient regression), REG-02 (functional GLM families), REG-03 (elastic multinomial regression), INF-03 (Interval Testing Procedure / ITP family — deferred from v0.19.0), FPCA-01 (PACE sparse FPCA), plus the larger differentiator clusters (DEPTH-01, OUT-01, FTS-*, FRE-*, DENS-*, CLUS-*, REP-*, SPARSE-*).

Advisory tech-debt carried forward (not v0.20.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; Phases 10/11/14/15 VALIDATION.md remain `draft` (Nyquist TODO).

**Also pending (not a backlog item):** a crate release for v0.19.0 — version bump + PR + tag, since v0.19.0 shipped real code.

## Session Continuity

Last session: 2026-08-16T19:43:29.313Z
Stopped at: Completed 22-01-PLAN.md
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
