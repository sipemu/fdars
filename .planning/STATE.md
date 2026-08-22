---
gsd_state_version: 1.0
milestone: v0.28.0
milestone_name: Spectral Functional Time Series & Object-Data Fréchet Regression
status: planning
last_updated: "2026-08-22T16:47:35.938Z"
last_activity: 2026-08-22
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-22)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone (v0.28.0) draws the two now-unblocked score-1.00 (L-effort) items from the v0.18.0 `R-BACKLOG.md`: spectral functional time series (FTS-03) and object-data Fréchet regression (FRE-02), each by adding `fdars-core/src/` code additively.
**Current focus:** v0.28.0 started — defining requirements

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-08-22 — Milestone v0.28.0 started

## Milestone Roadmap (v0.27.0)

Two phases, two requirements — the two score-1.33 (L-effort) `R-BACKLOG.md` items: FTS-01 (rank 20), FRE-01 (rank 21). Opens the two largest gap zones (Area 6 functional time series, 2/25 present; Area 7 density/object data, 0/25 present), exhausting the 1.33 tier. Real `fdars-core/src/` code — additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports; **zero changes to existing public signatures.** Both are reuse-first (new `fts/forecast.rs` reusing `fdata_to_pc_1d` + `scoring.rs` + FTS-02's `fts/acf.rs`; new `frechet/` module sharing DENS-01's `density_fda.rs` quantile/Wasserstein machinery); no new algorithm subsystem beyond the two new modules, **no new crate dependency.** Numeric outputs only — plotting/rendering out of scope.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 39 — Functional Time-Series Forecasting | FTS-01 (5 reqs) | new `fts/forecast.rs`: `ftsm` (FPCA-based functional time-series model — mean + FPC loadings + score-time-series + fitted curves), FPC-score-regression forecasting (scalar AR/ARIMA-style models per score sequence → reconstruct forecast curves), `fplsr` (functional PLS forecasting variant), dynamic forecast updating, iterative multi-step (h > 1) forecasting. Reuses `fdata_to_pc_1d` + `scoring.rs`, builds on the shipped FTS-02 (`fts/acf.rs`, Phase 34). R baseline: `ftsa`. Rank 20, score 1.33, P2 differentiator. |
| 40 — Fréchet / Object-Data Regression | FRE-01 (8 reqs) | new `frechet/` module: metric-space abstraction (distance + weighted-Fréchet-mean solver) with a 1D-Wasserstein (density-response) backend, Fréchet mean/variance, global & local (kernel-weighted) Fréchet regression over Euclidean predictors, 1D 2-Wasserstein distance, density-response Fréchet regression, Fréchet ANOVA. Shares DENS-01's (`density_fda.rs`, Phase 36) quantile/Wasserstein machinery. R baseline: `frechet`. Rank 21, score 1.33, P2 differentiator. |

**Execution order:** Both phases are **independent** — FTS-01 (Phase 39) and FRE-01 (Phase 40) have **no cross-phase hard dependency** (as in prior implementation milestones), and each touches a disjoint area of the codebase (new `fts/forecast.rs` vs new `frechet/` module). They may be planned and executed in **any order or in parallel**. No forced sequence.

## Performance Metrics

**Velocity:**

- Total plans completed: 72 (21 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0 + 2 in v0.19.0 + 2 in v0.20.0 + 2 in v0.21.0 + 2 in v0.22.0 + 7 in v0.23.0 + 7 in v0.24.0 + 10 in v0.25.0 + 4 in v0.26.0)
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
| 39 | v0.27.0 | TBD (FTS-01) |
| 40 | v0.27.0 | TBD (FRE-01) |

**Recent Trend:**

- Last milestone: v0.26.0 phases 37–38 (4 plans) — both completed + verified (5/5 each), milestone audit PASSED 8/8
- Trend: consistent ~45min per plan, 7 tests/plan average for implementation phases

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.27.0 implementation):

- v0.27.0 is an **implementation** milestone — real `fdars-core/src/` code (drawn top-first from the R-ecosystem backlog, after v0.19.0–v0.26.0). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is the **two score-1.33 (L-effort)** `R-BACKLOG.md` items: **FTS-01** (rank 20, functional time-series forecasting) + **FRE-01** (rank 21, Fréchet/object-data regression). Shipping these two exhausts the 1.33 tier. Both are **reuse-first** (new `fts/forecast.rs` reusing `fdata_to_pc_1d` + `scoring.rs` + FTS-02's `fts/acf.rs`; new `frechet/` module sharing DENS-01's `density_fda.rs` quantile/Wasserstein machinery); no new algorithm subsystem beyond the two new modules, **no new crate dependency**. Plotting/rendering out of scope (numeric outputs only).
- Phase numbering **continues** from v0.26.0 (ended at Phase 38) → v0.27.0 starts at Phase 39. No reset.
- **One requirement per phase, two phases:** Phase 39 = FTS-01, Phase 40 = FRE-01.
- **Both phases are independent** — no cross-phase hard dependency. Each touches a disjoint area of the codebase (new `fts/forecast.rs` vs new `frechet/` module), so 39/40 may run in **any order or in parallel**.
- **FTS-01 scope (from R-BACKLOG.md block):** new `fdars-core/src/fts/forecast.rs` — decompose a time-ordered curve series via `fdata_to_pc_1d` (`ftsm`: mean + FPC loadings + score-time-series + fitted curves), fit scalar (AR/ARIMA-style) time-series models to each FPC-score sequence and reconstruct h-step-ahead forecast curves, a functional PLS forecasting variant (`fplsr`), a dynamic-updating path (update forecast on new observation without full refit), and iterative multi-step (h > 1) forecasting. Reuses `fdata_to_pc_1d` + `scoring.rs`; **depends on the shipped FTS-02** (`fts/acf.rs`, Phase 34) for score-model order/inference. R baseline: `ftsa`. Largest single-area gap zone by capability count (Area 6, 2/25 present).
- **FRE-01 scope (from R-BACKLOG.md block):** new `fdars-core/src/frechet/` module — a metric-space abstraction (distance + weighted-Fréchet-mean solver) with a 1D-Wasserstein (density-response) backend as the first concrete space, Fréchet mean (weighted barycenter) + variance (mean squared distance to the mean), global Fréchet regression (weighted global/linear weight scheme over Euclidean predictors), local (local-linear / kernel-weighted) Fréchet regression, the 1D 2-Wasserstein distance (quantile-based), density-response Fréchet regression (conditional density from Euclidean predictors in 2-Wasserstein space), and Fréchet ANOVA (group-difference test on metric-space responses via means/variances). **Shares DENS-01's** (`density_fda.rs`, Phase 36) quantile/Wasserstein machinery — start from the density (2-Wasserstein) response space. R baseline: `frechet`. Single largest all-absent zone (Area 7, 0/25 present).
- R baselines matched by **capability**, not R's exact signatures. Document any divergence from the R baseline in rustdoc (as prior milestones documented divergences).

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` where randomness is involved (any Monte-Carlo / bootstrap / permutation paths need seeded reproducibility — relevant to FRE-01's Fréchet ANOVA if permutation-based, mirroring INF-01's 999-perm default).
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code + doctests, so this pointer matters.

### Pending Todos

None yet.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer). Both phases are independent and could otherwise parallelize, but sequential fallback may serialize them regardless. Prior milestone (v0.26.0) executed phases inline (not via gsd-executor subagents) for this reason.
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant since the milestone compiles real code + doctests. Also: `target/` grows to 100+GB and fills /home; `rm -rf target/debug/{incremental,examples}` frees ~108G if example LINK fails (MEMORY.md pointer).
- Executor subagents trip the 600s stream watchdog on long fdars cargo builds; `--no-verify` commits leave fmt drift → run `cargo fmt` per commit + a whole-crate sweep at milestone end (MEMORY.md pointers).
- FTS-01: the `ftsm` reconstruction/truncation convention, the choice of scalar time-series model (AR/ARIMA-style) fit to each FPC-score sequence and how its order is selected (reuse FTS-02's ACF/PACF), the `fplsr` PLS-score formulation, the dynamic-updating update rule, and the iterative multi-step reconstruction each have specific `ftsa` reference formulations; pin the score-model family + order selection, the PLS forecasting variant, the dynamic-update rule, and the multi-step iteration during planning, and document any divergence from `ftsa` in rustdoc. Reuse `fdata_to_pc_1d` + `scoring.rs` + `fts/acf.rs`.
- FRE-01: the metric-space abstraction shape (distance + weighted-Fréchet-mean solver trait/struct), the global-vs-local Fréchet-regression weight schemes (global linear weights vs local-linear/kernel weights), the 1D 2-Wasserstein distance formulation (quantile-based, must reuse `density_fda.rs`), the density-response prediction path, and the Fréchet-ANOVA test statistic (+ p-value derivation, seeded if permutation-based) each have specific `frechet` reference definitions; pin the abstraction, the weight schemes, the Wasserstein formula, and the ANOVA statistic during planning; reuse `density_fda.rs`'s quantile/Wasserstein machinery rather than re-deriving. Document any divergence in rustdoc.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260822-pvk | Update README: link Python package pyfda, note R package outdated | 2026-08-22 | 0c634e55 | [260822-pvk-update-readme-link-python-package-pyfda-](./quick/260822-pvk-update-readme-link-python-package-pyfda-/) |

## Deferred Items

v2 backlog items (from `.planning/research/R-BACKLOG.md`): FTS-03 (spectral functional time series — DPCA/spectral-density/VAR-VMA/FARMA, score 1.00, L-effort — **depends on FTS-01**, reuses `rustfft`); FRE-02 (object-data Fréchet spaces — covariance/correlation/spherical/network/point-process, score 1.00, L-effort — **depends on FRE-01**'s solver framework); REG-06, REP-02, CLUS-02 (score ≤ 1.00, L). These form the next tier once v0.27.0 exhausts the 1.33 tier. Explicit v0.27.0 exclusions: new crate dependency for time-series / metric-space machinery; plotting/rendering of forecasts, prediction bands, or Fréchet fits; changes to existing public signatures (`fdata_to_pc_1d`, `fts/acf.rs`, `density_fda.rs`, `scoring.rs`); spectral/frequency-domain FTS (FTS-03); object-space Fréchet backends beyond 1D density/Wasserstein (FRE-02); Bayesian/boosting functional regression (REG-06).

Advisory tech-debt carried forward (not v0.27.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; prior VALIDATION.md files (incl. Phases 28–38) remain `draft` (Nyquist TODO); intentional R-baseline divergences documented in rustdoc across prior milestones.

**Release status:** `fdars-core` **0.27.0 published to crates.io** (tag `v0.27.0`, 2026-08-22; `release.yml` `cargo publish` succeeded). The 0.27.0 crate folds in all additive code shipped since 0.24.0 (v0.25.0/v0.26.0 were never published separately). `CHANGELOG.md` added at repo root.

## Session Continuity

Last session: 2026-08-22T00:00:00.000Z
Stopped at: v0.27.0 roadmap created (Phases 39–40) — ready to plan
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
