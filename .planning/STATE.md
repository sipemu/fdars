---
gsd_state_version: 1.0
milestone: v0.28.0
milestone_name: Spectral Functional Time Series & Object-Data Fréchet Regression
current_phase: 42
current_phase_name: Object-Data Fréchet Regression
status: executing
stopped_at: Phase 41 complete, ready to plan Phase 42
last_updated: "2026-08-22T22:05:28.952Z"
last_activity: 2026-08-23
last_activity_desc: Phase 42 execution started
state_head: 59b00ad9f7f92c4b449db31040919bfab51dae09
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 5
  completed_plans: 2
  percent: 40
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-22)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone (v0.28.0) draws the two now-unblocked score-1.00 (L-effort) items from the v0.18.0 `R-BACKLOG.md`: spectral functional time series (FTS-03) and object-data Fréchet regression (FRE-02), each by adding `fdars-core/src/` code additively.
**Current focus:** Phase 42 — Object-Data Fréchet Regression

## Current Position

Phase: 42 (Object-Data Fréchet Regression) — EXECUTING
Plan: 1 of 3
Status: Executing Phase 42
Last activity: 2026-08-23 — Phase 42 execution started

## Milestone Roadmap (v0.28.0)

Two phases, two requirements — the two now-unblocked score-1.00 (L-effort) `R-BACKLOG.md` items: FTS-03 (rank 22, spectral functional time series), FRE-02 (rank 23, object-data Fréchet regression). Both dependencies (FTS-01, FRE-01) shipped in v0.27.0. Draws from Area 6 (functional time series) + Area 7 (density/object data), exhausting the 1.00 tier's dependency-satisfied items. Real `fdars-core/src/` code — additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports; **zero changes to existing public signatures.** Both are reuse-first (FTS-03 reuses the existing `rustfft` + FTS-01/FTS-02's `fts/forecast.rs`/`fts/acf.rs`; FRE-02 plugs new metric backends into the shipped FRE-01 solver); no new algorithm subsystem beyond the new `fts/spectral.rs` module + `simulation.rs` additions + `frechet/` backends, **no new crate dependency.** Numeric outputs only — plotting/rendering out of scope.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 41 — Spectral Functional Time Series | FTS-03 (5 reqs) | new `fts/spectral.rs`: spectral density operator (frequency-domain long-run covariance via `rustfft` over lagged autocovariance operators at Fourier frequencies), dynamic FPCA (DPCA — dynamic eigen-filters + dynamic scores from the spectral density), curve reconstruction from dynamic scores via inverse dynamic filtering; plus a functional VAR/VMA simulator + functional ARMA (FARMA) simulator in `simulation.rs`. Reuses the existing `rustfft` dependency; builds on the shipped FTS-01/FTS-02 (`fts/forecast.rs`, `fts/acf.rs`) foundation. R baseline: `freqdom`/`ftsa`. Rank 22, score 1.00, P3 differentiator. |
| 42 — Object-Data Fréchet Regression | FRE-02 (7 reqs) | extends `frechet/` with pluggable non-density `MetricSpace` backends — SPD covariance-matrix (Frobenius/power/log-Cholesky), correlation-matrix, spherical (geodesic exp/log), network, and point-process response spaces (each: distance + weighted-Fréchet-mean solver), consumed generically by the shipped FRE-01 `frechet_global_reg`/`frechet_local_reg`/`frechet_anova`. R baseline: `frechet` 0.3.0. Rank 23, score 1.00, P3 differentiator. |

**Execution order:** Both phases are **independent** — FTS-03 (Phase 41) and FRE-02 (Phase 42) have **no cross-phase hard dependency** (as in prior implementation milestones), and each touches a disjoint area of the codebase (new `fts/spectral.rs` + `simulation.rs` additions vs `frechet/` metric backends). They may be planned and executed in **any order or in parallel**. No forced sequence.

## Performance Metrics

**Velocity:**

- Total plans completed: 74 (21 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0 + 2 in v0.19.0 + 2 in v0.20.0 + 2 in v0.21.0 + 2 in v0.22.0 + 7 in v0.23.0 + 7 in v0.24.0 + 10 in v0.25.0 + 4 in v0.26.0 + 6 in v0.27.0)
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
| 41 | v0.28.0 | TBD (FTS-03) |
| 42 | v0.28.0 | TBD (FRE-02) |

**Recent Trend:**

- Last milestone: v0.27.0 phases 39–40 (6 plans) — both completed + verified (5/5 each), milestone audit PASSED 13/13, released as crate `fdars-core` 0.27.0 on crates.io
- Trend: consistent ~45min per plan, 7 tests/plan average for implementation phases

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.28.0 implementation):

- v0.28.0 is an **implementation** milestone — real `fdars-core/src/` code (drawn top-first from the R-ecosystem backlog, after v0.19.0–v0.27.0). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is the **two now-unblocked score-1.00 (L-effort)** `R-BACKLOG.md` items: **FTS-03** (rank 22, spectral functional time series) + **FRE-02** (rank 23, object-data Fréchet regression). Both dependencies (FTS-01, FRE-01) shipped in v0.27.0. Both are **reuse-first** (FTS-03 reuses the existing `rustfft` + FTS-01/FTS-02's `fts/forecast.rs`/`fts/acf.rs`; FRE-02 plugs new metric backends into the shipped FRE-01 solver); no new algorithm subsystem beyond `fts/spectral.rs` + `simulation.rs` additions + `frechet/` backends, **no new crate dependency**. Plotting/rendering out of scope (numeric outputs only).
- Phase numbering **continues** from v0.27.0 (ended at Phase 40) → v0.28.0 starts at Phase 41. No reset.
- **One requirement per phase, two phases:** Phase 41 = FTS-03, Phase 42 = FRE-02.
- **Both phases are independent** — no cross-phase hard dependency. Each touches a disjoint area of the codebase (new `fts/spectral.rs` + `simulation.rs` additions vs `frechet/` metric backends), so 41/42 may run in **any order or in parallel**.
- **FTS-03 scope (from R-BACKLOG.md block):** new `fdars-core/src/fts/spectral.rs` — spectral density operator estimation (frequency-domain long-run covariance formed via `rustfft` over lagged autocovariance operators at Fourier frequencies), dynamic functional PCA (DPCA — dynamic eigen-filters + dynamic scores from the spectral density), and curve reconstruction from dynamic scores via inverse dynamic filtering; plus a functional VAR/VMA simulator and a functional ARMA (FARMA) simulator in `simulation.rs`. Reuses the existing `rustfft`; **builds on the shipped FTS-01/FTS-02** (`fts/forecast.rs`, `fts/acf.rs`). R baseline: `freqdom`/`ftsa`. Covers ~9 absent Area-6 gaps.
- **FRE-02 scope (from R-BACKLOG.md block):** extend `fdars-core/src/frechet/` with per-space metric + geodesic operations implemented as pluggable non-density `MetricSpace` backends — SPD covariance-matrix responses (Frobenius / power / log-Cholesky metrics), correlation matrices, spherical data (geodesic exp/log maps), network responses, and point-process responses (each: distance + weighted-Fréchet-mean solver) — consumed generically by the shipped FRE-01 `frechet_global_reg`/`frechet_local_reg`/`frechet_anova`. **Depends on the shipped FRE-01** solver framework. R baseline: `frechet` 0.3.0. Covers ~8 absent Area-7 gaps.
- R baselines matched by **capability**, not R's exact signatures. Document any divergence from the R baseline in rustdoc (as prior milestones documented divergences).

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` where randomness is involved — relevant to both phases: FTS-03's VAR/VMA/FARMA simulators must be seeded/deterministic, and FRE-02's Fréchet-ANOVA over object spaces reuses FRE-01's seeded-permutation p-value (999-perm default from INF-01).
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code + doctests, so this pointer matters.

### Pending Todos

None yet.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer). Both phases are independent and could otherwise parallelize, but sequential fallback may serialize them regardless. Prior milestones (v0.26.0/v0.27.0) executed phases inline (not via gsd-executor subagents) for this reason.
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant since the milestone compiles real code + doctests. Also: `target/` grows to 100+GB and fills /home; `rm -rf target/debug/{incremental,examples}` frees ~108G if example LINK fails (MEMORY.md pointer).
- Executor subagents trip the 600s stream watchdog on long fdars cargo builds; `--no-verify` commits leave fmt drift → run `cargo fmt` per commit + a whole-crate sweep at milestone end (MEMORY.md pointers).
- FTS-03: the spectral-density-operator estimator (which lag window / kernel weighting over autocovariance operators, which set of Fourier frequencies, how `rustfft` is applied over the operator sequence), the DPCA formulation (dynamic eigen-filters + dynamic scores from the spectral density, and the inverse-dynamic-filtering reconstruction convention), and the functional VAR/VMA + FARMA simulator recurrences (operator-kernel parameterization, burn-in, seeding) each have specific `freqdom`/`ftsa` reference formulations; pin the spectral-density estimator, the DPCA filter/score/reconstruction convention, and the simulator recurrences during planning, and document any divergence from `freqdom`/`ftsa` in rustdoc. Reuse `rustfft` + `fts/acf.rs` autocovariance machinery + `simulation.rs`.
- FRE-02: each non-density `MetricSpace` backend has a specific reference geometry — SPD covariance-matrix metrics (Frobenius vs power vs log-Cholesky distance + their weighted-Fréchet-mean solvers), the correlation-manifold distance + mean, spherical geodesic exp/log maps + spherical Fréchet mean, the network (graph-Laplacian/adjacency) distance + mean, and the point-process (intensity/count) distance + mean — plus how each plugs into the FRE-01 generic solver and `frechet_anova`. Pin each backend's distance + weighted-mean solver during planning; reuse the FRE-01 `MetricSpace` trait + regression/ANOVA machinery rather than re-deriving. Document any divergence from `frechet` 0.3.0 in rustdoc.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260822-pvk | Update README: link Python package pyfda, note R package outdated | 2026-08-22 | 0c634e55 | [260822-pvk-update-readme-link-python-package-pyfda-](./quick/260822-pvk-update-readme-link-python-package-pyfda-/) |

## Deferred Items

v2 backlog items (from `.planning/research/R-BACKLOG.md`): REG-06 (boosting/Bayesian functional regression — FDboost/GAMLSS/Gibbs-VB FOSR, score 0.67, L-effort); REP-02 (FEM/PDE smoothing on irregular 2D/3D domains — fdaPDE, score 0.67, L-effort); CLUS-02 (functional co-clustering — funLBM latent-block + slope-heuristic, score 0.67, L-effort). These form the next tier once v0.28.0 exhausts the 1.00 tier's dependency-satisfied items. Explicit v0.28.0 exclusions: new crate dependency for spectral / object-space machinery; plotting/rendering of spectra, DPCA filters, or object-space Fréchet fits; changes to existing public signatures (`fdata_to_pc_1d`, `fts/acf.rs`, `fts/forecast.rs`, `frechet/`, `simulation.rs`, …); boosting/Bayesian functional regression (REG-06); FEM/PDE smoothing (REP-02); functional co-clustering (CLUS-02).

Advisory tech-debt carried forward (not v0.28.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; prior VALIDATION.md files (incl. Phases 28–40) remain `draft` (Nyquist TODO); intentional R-baseline divergences documented in rustdoc across prior milestones (incl. v0.27.0's signed-weight isotonic projection, Fréchet-ANOVA σ̂ₗ² `[ASSUMED]` estimator, Wasserstein-barycenter reconstruction floor).

**Release status:** `fdars-core` **0.27.0 published to crates.io** (tag `v0.27.0`, 2026-08-22; `release.yml` `cargo publish` succeeded). The 0.27.0 crate folds in all additive code shipped since 0.24.0 (v0.25.0/v0.26.0 were never published separately). `CHANGELOG.md` at repo root. v0.28.0 ship (version bump 0.27.0 → 0.28.0 + tag + crates.io) is a pending operator ship-time step once both phases complete.

## Session Continuity

Last session: 2026-08-22T00:00:00.000Z
Stopped at: Phase 41 complete, ready to plan Phase 42
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 41` (or `/gsd-plan-phase 42` — the two phases are independent and may be planned in any order or in parallel)
