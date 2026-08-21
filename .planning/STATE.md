---
gsd_state_version: 1.0
milestone: v0.25.0
current_phase: 35
current_phase_name: Basis-System Completions
status: planning
stopped_at: Phase 34 complete, ready to plan Phase 35
last_updated: "2026-08-21T06:37:25.286Z"
last_activity: 2026-08-21
last_activity_desc: Phase 34 complete, transitioned to Phase 35
state_head: 86598bd60ff65b50aeff3a994cf8080ff2c6e204
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
milestone_name: Serial Dependence, Representation & Density Breadth
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-20)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone draws the next three top-ranked items from the v0.18.0 `R-BACKLOG.md` (score 1.73 each): functional serial-dependence tooling (FTS-02), basis-system completions (REP-01), and density object-data FDA (DENS-01), each by adding or extending `fdars-core/src/` modules additively.
**Current focus:** Roadmap created (Phases 34–36) — next: `/gsd-plan-phase 34` (or 35 / 36, in any order). Deferred operator ship-step: crate release for v0.23.0 + v0.24.0 + v0.25.0 (bump `fdars-core/Cargo.toml` from 0.23.0, tag/publish).

## Current Position

Phase: 35 — Basis-System Completions
Plan: Not started
Status: Ready to plan
Last activity: 2026-08-21 — Phase 34 complete, transitioned to Phase 35

## Milestone Roadmap (v0.25.0)

Three phases, three requirements — the next three top-ranked `R-BACKLOG.md` items (all tied at score 1.73, M-effort): FTS-02 (rank 14), REP-01 (rank 16), DENS-01 (rank 17). Broadens fdars' functional-time-series diagnostics, representation layer, and density-FDA families. Real `fdars-core/src/` code — additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports; **zero changes to existing public signatures.** All three are reuse-first (new `fts/acf.rs`; extend `basis/` + new `multi_fdata.rs`; new `density_fda.rs` reusing `fdata_to_pc_1d`); no new algorithm subsystem, **no new crate dependency.** Numeric outputs only — plotting/rendering out of scope.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 34 — Functional Serial-Dependence Tooling | FTS-02 | new `fts/acf.rs`: L2-norm functional ACF/PACF with strong-white-noise confidence bands, a functional stationarity test, a long-run-covariance kernel-sandwich estimator, and a functional differencing operator — reusing `helpers` quadrature + `covariance.rs`. R baseline: `ftsa` / `fdaACF`. Foundational for deferred FTS-01/FTS-03 (build before FTS-01). Rank 14, score 1.73. |
| 35 — Basis-System Completions | REP-01 | extend `basis/`: `monomial_basis`/`exponential_basis`/`power_basis`/`polygonal_basis` factories (with penalty matrices); a `MultiFunData` multivariate/multi-domain container in new `multi_fdata.rs`; a composable `Lfd`/linear-differential-operator object; a `principal_differential_analysis` (PDA) estimator. Constant basis already handled (T-01). R baseline: `fda` / `funData` / `tf`. Rank 16, score 1.73. |
| 36 — Density Object-Data FDA | DENS-01 | new `density_fda.rs`: the log-quantile-density (LQD) transform + inverse, LQD-FPCA (reuse `fdata_to_pc_1d` in LQD space, with FVE), a 1D Wasserstein Fréchet-mean (quantile-average) barycenter, and density normalization/regularization. Numeric only. 1D-density subset of R-audit Area 7 (simpler than FRE-01/FRE-02). R baseline: `fdadensity`. Rank 17, score 1.73. |

**Execution order:** All three phases are **independent** — FTS-02, REP-01, and DENS-01 have **no cross-phase hard dependency** (as in v0.24.0's REG-04/REG-05/CLUS-01), and each touches a disjoint area of the codebase. They may be planned and executed in **any order or in parallel**. No forced sequence.

## Performance Metrics

**Velocity:**

- Total plans completed: 58 (21 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0 + 2 in v0.19.0 + 2 in v0.20.0 + 2 in v0.21.0 + 2 in v0.22.0 + 7 in v0.23.0 + 7 in v0.24.0)
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
| 34 | v0.25.0 | TBD (FTS-02) |
| 35 | v0.25.0 | TBD (REP-01) |
| 36 | v0.25.0 | TBD (DENS-01) |

**Recent Trend:**

- Last milestone: v0.24.0 phases 31–33 (7 plans) — all completed + verified
- Trend: consistent ~45min per plan, 7 tests/plan average for implementation phases

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.25.0 implementation):

- v0.25.0 is an **implementation** milestone — real `fdars-core/src/` code (drawn top-first from the R-ecosystem backlog, after v0.19.0–v0.24.0). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is the **next three top-ranked** `R-BACKLOG.md` items (all tied at score 1.73, M-effort): **FTS-02** (rank 14), **REP-01** (rank 16), **DENS-01** (rank 17) — broadening the functional-time-series diagnostics, representation, and density-FDA families. All three are **reuse-first** (new `fts/acf.rs`; extend `basis/` + new `multi_fdata.rs`; new `density_fda.rs` reusing `fdata_to_pc_1d`); no new algorithm subsystem, **no new crate dependency**. Plotting/rendering out of scope (numeric outputs only).
- Phase numbering **continues** from v0.24.0 (ended at Phase 33) → v0.25.0 starts at Phase 34. No reset.
- **One requirement per phase, three phases:** Phase 34 = FTS-02, Phase 35 = REP-01, Phase 36 = DENS-01.
- **All three phases are independent** — no cross-phase hard dependency. Each touches a disjoint area of the codebase, so 34/35/36 may run in **any order or in parallel**.
- **FTS-02 scope (from R-BACKLOG.md block):** new `fts/acf.rs` — L2-norm functional ACF/PACF with the strong-white-noise limiting distribution for confidence bands; a functional `stationarity_test`; a `long_run_covariance` kernel-sandwich estimator; a functional differencing operator. Reuses `helpers` quadrature + `covariance.rs`. R baseline: `ftsa` (facf, T_stationary, long-run covariance) / `fdaACF` (L2-norm fACF, partial fACF, white-noise distribution). Foundational for FTS-01/FTS-03 forecasting — build before FTS-01.
- **REP-01 scope (from R-BACKLOG.md block):** extend `basis/` with `monomial_basis`, `exponential_basis`, `power_basis`, `polygonal_basis` factories (each with penalty matrices); a `MultiFunData` multivariate/multi-domain container in new `multi_fdata.rs`; a composable `Lfd`/`LinearDifferentialOperator` object; a `principal_differential_analysis` (PDA, linear-ODE estimation) estimator. The constant basis is already handled by T-01 (do not re-add). R baseline: `fda` (monomial/exponential/power/polygonal bases; Lfd/PDA) / `funData` (multiFunData) / `tf` (tidy vector).
- **DENS-01 scope (from R-BACKLOG.md block):** new `density_fda.rs` — the log-quantile-density (LQD) transform + inverse (compositional-geometry map), LQD-FPCA for probability densities (reuse `fdata_to_pc_1d` in LQD space, with FVE), a 1D Wasserstein Fréchet mean (quantile-average barycenter), and density normalization/regularization. Numeric outputs only. The 1D-density subset of Area 7 — simpler than the general Fréchet items (FRE-01/FRE-02). R baseline: `fdadensity`.
- R baselines matched by **capability**, not R's exact signatures.

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` where randomness is involved (the FTS-02 white-noise-band / stationarity tests need seeded reproducibility, mirroring INF-01's 999-perm default).
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code + doctests, so this pointer matters.

### Pending Todos

None yet.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer). All three phases are independent and could otherwise parallelize, but sequential fallback may serialize them regardless.
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant since the milestone compiles real code + doctests. Also: `target/` grows to 100+GB and fills /home; `rm -rf target/debug/{incremental,examples}` frees ~108G if example LINK fails (MEMORY.md pointer).
- FTS-02: the L2-norm fACF/fPACF definition, the strong-white-noise limiting distribution for the confidence bands, the functional stationarity test statistic, and the long-run-covariance kernel/bandwidth choice each have specific reference formulations (`ftsa`/`fdaACF`); pin the exact fACF normalization, the white-noise band construction, the kernel-sandwich weighting, and the differencing convention during planning, and document any divergence from the R baseline in rustdoc (as prior milestones documented divergences).
- REP-01: the monomial/exponential/power/polygonal penalty-matrix definitions, the `MultiFunData` container semantics (per-component argvals, shared observation count), the composable `Lfd` construction, and the PDA linear-ODE estimation each have distinct `fda`/`funData` reference definitions; pin each basis's penalty formula and the PDA estimation approach during planning. Do NOT re-add the constant basis (T-01 already ships it).
- DENS-01: the LQD forward/inverse transform (compositional-geometry map), the LQD-FPCA in transformed space (reuse `fdata_to_pc_1d`, define FVE), the 1D Wasserstein barycenter (quantile-average), and the density normalization/regularization each have specific `fdadensity` reference definitions; pin the LQD numerics (grid, integration, boundary handling), the round-trip tolerance, and the barycenter quantile-average during planning; reuse `fdata_to_pc_1d` + `helpers` quadrature.

## Deferred Items

v2 backlog items (from `.planning/research/R-BACKLOG.md`, see REQUIREMENTS.md v2 section): FPCA-02, SPARSE-01 (score 1.73, P3); FTS-01, FRE-01 (score 1.33, L); FTS-03, FRE-02, REG-06, REP-02, CLUS-02 (score ≤ 1.00, L). FTS-01/FTS-03 build on FTS-02's serial-dependence foundation; FRE-01/FRE-02 are the general Fréchet/object-data items (DENS-01 covers only the tractable 1D-density subset of Area 7). REP-02 (tidy-vector API beyond the `MultiFunData` container) is deferred. Explicit v0.25.0 exclusions: plotting/rendering of fACF/PACF diagnostics, basis functions, or density curves; full FTS forecasting (FTS-01); spectral/dynamic FTS (FTS-03); general Fréchet regression / object-data statistics (FRE-01/FRE-02); multivariate density FPCA / general metric-space barycenters; tidyfun-style tidy-vector semantics beyond `MultiFunData`; new crate dependencies; changes to existing public signatures.

Advisory tech-debt carried forward (not v0.25.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; prior VALIDATION.md files (incl. Phases 28–33) remain `draft` (Nyquist TODO); intentional R-baseline divergences documented in rustdoc across prior milestones.

**Also pending (not a backlog item):** a crate release for v0.23.0, v0.24.0 **and** v0.25.0 — version bump (Cargo.toml still 0.23.0) + PR + tag, since all three shipped/ship real code (operator-driven ship-time step; a `v*` tag push triggers the crates.io publish).

## Session Continuity

Last session: 2026-08-21T00:00:00.000Z
Stopped at: Phase 34 complete, ready to plan Phase 35
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 34` (or 35 / 36 — all independent, any order).
