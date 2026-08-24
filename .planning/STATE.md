---
gsd_state_version: 1.0
milestone: v0.29.0
current_phase: 43
current_phase_name: Boosting / Bayesian Functional Regression
status: executing
stopped_at: Completed 43-02-boosted-fofr-PLAN.md
last_updated: "2026-08-24T07:00:50.508Z"
last_activity: 2026-08-23
last_activity_desc: Phase 43 execution started
state_head: 954db9c4b9a0ffb66da7f363d84837374769d468
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 5
  completed_plans: 2
milestone_name: Boosting/Bayesian Regression, FEM/PDE Smoothing & Functional Co-Clustering
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-23)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone (v0.29.0) draws the **final three** items from the v0.18.0 `R-BACKLOG.md` (all score 0.67, L-effort), **exhausting the R-parity backlog**: boosting/Bayesian functional regression (REG-06), FEM/PDE smoothing on irregular domains (REP-02), and functional co-clustering (CLUS-02), each by adding `fdars-core/src/` code additively.
**Current focus:** Phase 43 — Boosting / Bayesian Functional Regression

## Current Position

Phase: 43 (Boosting / Bayesian Functional Regression) — EXECUTING
Plan: 3 of 5
Status: Ready to execute
Last activity: 2026-08-23 — Phase 43 execution started

## Milestone Roadmap (v0.29.0)

Three phases, three requirements — the **final three** `R-BACKLOG.md` items (all score 0.67, L-effort): REG-06 (rank 24, boosting/Bayesian functional regression), REP-02 (rank 25, FEM/PDE smoothing on irregular 2D domains), CLUS-02 (rank 26, functional co-clustering). #1–23 shipped through v0.28.0. **This milestone exhausts `R-BACKLOG.md`.** Real `fdars-core/src/` code — additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports; **zero changes to existing public signatures.** Numeric outputs only — plotting/rendering out of scope. **Unlike prior reuse-first milestones, all three are large standalone estimation subsystems — the heaviest milestone in the sequence; each phase likely needs a careful multi-plan decomposition.** The **no-new-crate-dependency** convention carries forward, with one caveat: **REP-02 (Phase 44)** is the phase where the planner MAY revisit it if an in-house triangulated-mesh/FEM implementation proves impractical (must flag at plan time).

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 43 — Boosting / Bayesian Functional Regression | REG-06 (5 reqs) | new `boosting_regression.rs`: component-wise gradient boosting with functional base-learners for function-on-scalar (boosted FOSR, one base-learner selected per iteration) + function-on-function (boosted FoFR); GAMLSS-style distributional regression (multiple distributional parameters, e.g. location + scale); Bayesian FOSR via Gibbs/VB with posterior-mean + credible-band coefficient summaries; FDboost-style stability selection (per-learner selection frequencies / stable predictor set). fdars regression is penalized/kernel/PLS/elastic only today. R baseline: `FDboost` 1.1-4 + `refund`. Rank 24, score 0.67, P3 differentiator, L-effort. Covers ~4 absent Area-4 gaps. |
| 44 — FEM/PDE Smoothing on Irregular 2D Domains | REP-02 (4 reqs) | new `fem_smoothing.rs`: linear finite-element basis over a user-supplied triangulated 2D mesh (nodes + triangle connectivity), basis-function evaluation + mass/stiffness assembly; PDE-regularized (Laplacian-penalty) surface smoothing of scattered observations over an irregular 2D domain returning a fitted surface + diagnostics. Plus **additive** positive (log-domain, nonnegative-guaranteed) and Ramsay integral-of-exp monotone smoothers into `smooth_basis.rs`. Does **not** overlap fdars' A-6 strength (regular-grid 2D FOSR / `function_on_scalar_2d`) — this is irregular-mesh FEM. R baseline: `fdaPDE` 1.1-24. Rank 25, score 0.67, P3 differentiator, L-effort. Covers ~5 absent/partial Area-1 gaps. **v1 scope is 2D triangulated meshes only (3D tetrahedral FEM out of scope).** Planner MAY revisit the no-new-crate-dependency constraint here. |
| 45 — Functional Co-Clustering (funLBM latent-block) | CLUS-02 (3 reqs) | new `coclustering.rs`: a functional latent block model (funLBM) — block-wise-Gaussian EM on FPC scores simultaneously assigning curves to row-clusters and argument points to column-clusters given a target (row, column) block count; result exposing row labels, column labels, per-block parameters, converged log-likelihood / model criterion (e.g. ICL); slope-heuristic model selection over a range of candidate (row, column) block counts. fdars' existing `clustering.rs`/`gmm/` cluster curves only. R baseline: `funLBM` 2.3.1 + `funHDDC` (slope heuristic). Rank 26, score 0.67, P3 differentiator, L-effort. Covers 2 absent Area-4 gaps. |

**Execution order:** All three phases are **independent** — REG-06 (Phase 43), REP-02 (Phase 44), CLUS-02 (Phase 45) have **no cross-phase hard dependency** and each touches a disjoint area of the codebase (`boosting_regression.rs` vs `fem_smoothing.rs` + additive `smooth_basis.rs` smoothers vs `coclustering.rs`). They may be planned and executed in **any order or in parallel**. No forced sequence.

## Performance Metrics

**Velocity:**

- Total plans completed: 79 (21 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0 + 2 in v0.19.0 + 2 in v0.20.0 + 2 in v0.21.0 + 2 in v0.22.0 + 7 in v0.23.0 + 7 in v0.24.0 + 10 in v0.25.0 + 4 in v0.26.0 + 6 in v0.27.0 + 5 in v0.28.0)
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
| 22–23 | v0.20.0 | 2 |
| 24–25 | v0.21.0 | 2 |
| 26–27 | v0.22.0 | 2 |
| 28–30 | v0.23.0 | 7 |
| 31–33 | v0.24.0 | 7 |
| 34–36 | v0.25.0 | 10 |
| 37–38 | v0.26.0 | 4 |
| 39–40 | v0.27.0 | 6 |
| 41–42 | v0.28.0 | 5 |
| 43 | v0.29.0 | TBD (REG-06) |
| 44 | v0.29.0 | TBD (REP-02) |
| 45 | v0.29.0 | TBD (CLUS-02) |

**Recent Trend:**

- Last milestone: v0.28.0 phases 41–42 (5 plans) — both completed + verified (5/5 each), milestone audit PASSED 12/12, released as crate `fdars-core` 0.28.0 on crates.io
- Trend: consistent ~45min per plan, 7 tests/plan average for implementation phases. **v0.29.0 is heavier** — three large standalone estimation subsystems (boosting/Bayesian, mesh/FEM, latent-block EM), each likely a multi-plan decomposition.

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 43 P01 | 13 | 3 tasks | 8 files |
| Phase 43 P02 | 6 | 2 tasks | 1 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.29.0 implementation):

- v0.29.0 is an **implementation** milestone — real `fdars-core/src/` code (drawn top-first from the R-ecosystem backlog, after v0.19.0–v0.28.0). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is the **final three score-0.67 (L-effort)** `R-BACKLOG.md` items: **REG-06** (rank 24, boosting/Bayesian functional regression), **REP-02** (rank 25, FEM/PDE smoothing on irregular 2D domains), **CLUS-02** (rank 26, functional co-clustering). **This milestone exhausts `R-BACKLOG.md`** (#1–23 shipped through v0.28.0).
- **Unlike prior reuse-first milestones, all three are large standalone estimation subsystems** — the heaviest milestone in the sequence. Each phase likely needs a careful multi-plan decomposition at plan time (not thin API-surfacing additions).
- Phase numbering **continues** from v0.28.0 (ended at Phase 42) → v0.29.0 starts at Phase 43. No reset.
- **One requirement-category per phase, three phases:** Phase 43 = REG-06, Phase 44 = REP-02, Phase 45 = CLUS-02.
- **All three phases are independent** — no cross-phase hard dependency. Each touches a disjoint area of the codebase (`boosting_regression.rs` vs `fem_smoothing.rs` + additive `smooth_basis.rs` smoothers vs `coclustering.rs`), so 43/44/45 may run in **any order or in parallel**.
- **REG-06 scope (from R-BACKLOG.md block):** new `fdars-core/src/boosting_regression.rs` — component-wise gradient boosting with functional base-learners for function-on-scalar (boosted FOSR, one base-learner selected per iteration) + function-on-function (boosted FoFR); GAMLSS-style distributional functional regression (models >1 distributional parameter, e.g. location + scale); a Bayesian function-on-scalar regression Gibbs/VB sampler producing coefficient posterior summaries (mean + credible bands); FDboost-style stability selection (per-learner selection frequencies / stable predictor set). R baseline: `FDboost` 1.1-4 + `refund`. Covers ~4 absent Area-4 gaps.
- **REP-02 scope (from R-BACKLOG.md block):** new `fdars-core/src/fem_smoothing.rs` — a linear finite-element basis over a user-supplied triangulated 2D mesh (nodes + triangle connectivity) with basis-function evaluation + mass/stiffness assembly, plus PDE (Laplacian) -regularized surface smoothing of scattered observations over an irregular 2D domain (fitted surface + diagnostics). **Additively** add positive (log-domain, nonnegative-guaranteed) + Ramsay integral-of-exp monotone smoothers to `smooth_basis.rs`. Does **not** overlap fdars' A-6 strength (regular-grid 2D FOSR / `function_on_scalar_2d`) — this is irregular-mesh FEM. **v1 scope is 2D triangulated meshes only** (3D tetrahedral FEM out of scope). R baseline: `fdaPDE` 1.1-24. Covers ~5 absent/partial Area-1 gaps.
- **CLUS-02 scope (from R-BACKLOG.md block):** new `fdars-core/src/coclustering.rs` — a functional latent block model (funLBM): block-wise-Gaussian EM on FPC scores that **simultaneously** clusters curves (rows) and argument points (columns) given a target (row, column) block count, exposing row labels, column labels, per-block parameters, and a converged log-likelihood / model criterion (e.g. ICL); plus a slope-heuristic model-selection helper over a range of candidate (row, column) block counts. fdars' `clustering.rs`/`gmm/` cluster curves only. R baseline: `funLBM` 2.3.1 + `funHDDC`. Covers 2 absent Area-4 gaps.
- **No-new-crate-dependency convention carries forward**, with one explicit caveat: **REP-02 (Phase 44)** is the phase where the planner MAY revisit it if an in-house triangulated-mesh/FEM implementation proves impractical at plan time — and must flag it explicitly. All other phases (43, 45) keep the strict no-new-dependency rule.
- R baselines matched by **capability**, not R's exact signatures. Document any divergence from the R baseline in rustdoc (as prior milestones documented divergences).
- **After v0.29.0 ships, `R-BACKLOG.md` is exhausted** — the next milestone requires a fresh yardstick (a new gap-audit against another reference ecosystem, a performance/consolidation pass, or a crate-release-hardening milestone), decided via `/gsd-new-milestone`.

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` where randomness is involved — relevant here: REG-06's Bayesian Gibbs/VB sampler + stability-selection resampling and CLUS-02's EM initialization must be seeded/deterministic; FPC scores reuse `fdata_to_pc_1d`.
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code + doctests, so this pointer matters.
- [Phase 43]: Beta matrix in BoostFosrResult stores mean fitted-value proxy (p × m_t) not B-spline K-vectors — consistent with FosrResult.beta convention
- [Phase 43]: BaseLearner struct pre-factors Cholesky once per learner — amortized over mstop iterations (only back-solves per iteration)
- [Phase 43]: boost_fofr uses bfpc FPC-score compression (fdata_to_pc_1d) rather than FDboost bsignal B-spline joint expansion — simpler, dependency-free, documented divergence in rustdoc

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** — preserved from the removed `.beads` issue tracker (issue `fdars-j75`, the only OPEN issue of 15; the other 14 were closed R-vs-fdars validation tasks, already reflected in the test suite). The full historical export remains in git history at `.beads/issues.jsonl` if needed.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer). All three phases are independent and could otherwise parallelize, but sequential fallback may serialize them regardless. Prior milestones (v0.26.0–v0.28.0) executed phases inline (not via gsd-executor subagents) for this reason.
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant since the milestone compiles real code + doctests. Also: `target/` grows to 100+GB and fills /home; `rm -rf target/debug/{incremental,examples}` frees ~108G if example LINK fails (MEMORY.md pointer).
- Executor subagents trip the 600s stream watchdog on long fdars cargo builds; `--no-verify` commits leave fmt drift → run `cargo fmt` per commit + a whole-crate sweep at milestone end (MEMORY.md pointers).
- **Heaviest-milestone flag:** all three phases are large standalone estimation subsystems (boosting/Bayesian machinery, mesh/FEM subsystem, latent-block EM) — not the thin reuse-first additions of prior milestones. Expect each phase to decompose into multiple plans; budget accordingly at plan time.
- REG-06: pin the boosting recurrence (functional base-learner family, per-iteration base-learner selection rule, step size / stopping), the GAMLSS distributional parameterization (which distribution + which parameters modelled), the Bayesian FOSR sampler (Gibbs vs VB, prior structure, credible-band construction), and the stability-selection resampling scheme against `FDboost`/`refund`; document any divergence in rustdoc.
- REP-02: pin the linear-FE basis + mass/stiffness assembly over triangles, the Laplacian PDE-penalty smoothing normal equations, the log-domain positive smoother, and the Ramsay integral-of-exp monotone smoother against `fdaPDE`; document any divergence in rustdoc. **This is the phase where the planner may flag the no-new-dependency constraint** if an in-house triangulated-mesh/FEM implementation proves impractical.
- CLUS-02: pin the funLBM block-wise-Gaussian EM on FPC scores (block model, E-/M-steps, ICL criterion, convergence + initialization/seeding) and the slope-heuristic selection criterion against `funLBM`/`funHDDC`; document any divergence in rustdoc.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260822-pvk | Update README: link Python package pyfda, note R package outdated | 2026-08-22 | 0c634e55 | [260822-pvk-update-readme-link-python-package-pyfda-](./quick/260822-pvk-update-readme-link-python-package-pyfda-/) |
| 260823-bds | Remove unused .beads issue tracker + stale AGENTS.md (conserved open issue fdars-j75 → Pending Todos) | 2026-08-23 | 198b5566 | [260823-bds-remove-beads-issue-tracker](./quick/260823-bds-remove-beads-issue-tracker/) |

## Deferred Items

`R-BACKLOG.md` is **exhausted** once v0.29.0 ships — no further ranked R-parity items remain. Explicit v0.29.0 exclusions: new crate dependency for boosting, mesh/FEM, or co-clustering machinery (REP-02 is the one place the planner may revisit this at plan time); plotting/rendering of boosting paths, FE meshes/surfaces, or co-cluster blocks; changes to existing public signatures (`smooth_basis.rs`, `clustering.rs`, `gmm/`, regression modules, …); 3D tetrahedral-mesh FEM (only 2D triangulated meshes in v1 scope); full mgcv/BayesX-grade sampler diagnostics (multiple chains, R̂, convergence tests) beyond posterior summaries + credible bands.

Advisory tech-debt carried forward (not v0.29.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; prior VALIDATION.md files (incl. Phases 28–42) remain `draft` (Nyquist TODO); intentional R-baseline divergences documented in rustdoc across prior milestones.

**Release status:** `fdars-core` **0.28.0 published to crates.io** (tag `v0.28.0`, 2026-08-23). Milestone v0.28.0 fully shipped and archived under `.planning/milestones/`. v0.29.0 is in planning (roadmap created; not yet executed).

## Session Continuity

Last session: 2026-08-24T07:00:50.498Z
Stopped at: Completed 43-02-boosted-fofr-PLAN.md
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 43` (or 44 / 45 — any order; all three are independent). Expect multi-plan decomposition per phase (heaviest milestone in the sequence).
