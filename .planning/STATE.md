---
gsd_state_version: 1.0
milestone: v0.26.0
status: Awaiting next milestone
stopped_at: v0.26.0 roadmap created (Phases 37–38) — ready to plan
last_updated: "2026-08-21T20:38:24.585Z"
last_activity: 2026-08-21
last_activity_desc: Milestone v0.26.0 completed and archived
state_head: ea4dd7e0d3646090cd6985362b1c6a991e751dbf
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 4
  completed_plans: 4
milestone_name: FPCA Breadth & Sparse Covariance
current_phase: 38
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-21)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone draws the two remaining top-ranked items from the v0.18.0 `R-BACKLOG.md` (score 1.73, M-effort): specialized FPCA variants (FPCA-02) and fast sparse/irregular covariance (SPARSE-01), each by adding `fdars-core/src/` code additively.
**Current focus:** Phase 38 — Sparse Fast Covariance & Trajectory Bands

## Current Position

Phase: Milestone v0.26.0 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-08-21 — Milestone v0.26.0 completed and archived

## Milestone Roadmap (v0.26.0)

Two phases, two requirements — the two remaining top-ranked `R-BACKLOG.md` items (both tied at score 1.73, M-effort): FPCA-02 (rank 18), SPARSE-01 (rank 19). Completes the FPCA long tail and adds fast sparse-covariance estimation, exhausting the 1.73 tier. Real `fdars-core/src/` code — additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports; **zero changes to existing public signatures.** Both are reuse-first (extend `regression.rs` / new `fpca_variants.rs` reusing `fdata_to_pc_1d` + `covariance.rs`; extend `irreg_fdata/` reusing `cov_irreg` + `pace_fpca`); no new algorithm subsystem, **no new crate dependency.** Numeric outputs only — plotting/rendering out of scope.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 37 — Specialized FPCA Variants | FPCA-02 | extend `regression.rs` (or new `fpca_variants.rs`): `fpca_der` (FPCA of derivatives), `fsvd` (functional SVD / cross-FPCA), `cross_covariance` (cross-covariance surface between two samples), `dynamical_correlation` (scalar association), and a sandwich-smoother / sparse-SVD (ssvd) FPCA path. Reuses `fdata_to_pc_1d` + `covariance.rs`. Complements the shipped PACE core (FPCA-01). R baseline: `fdapace` (FPCAder, FSVD, GetCrCov, DynCorr, FCCor) / `refund` (fpca.sc sandwich, fpca.ssvd). Rank 18, score 1.73. |
| 38 — Sparse Fast Covariance & Trajectory Bands | SPARSE-01 | extend `irreg_fdata/`: `face_covariance` (FACE fast-sandwich sparse-data covariance surface), `mface_covariance` (multivariate `mfaces` extension), and fitted continuous trajectories with pointwise confidence bands integrated with the FACE path. Builds on `cov_irreg`, integrates with the shipped PACE `pace_fpca` (FPCA-01). R baseline: `face` / `mfaces` / `fdapace` (trajectory bands). Rank 19, score 1.73. |

**Execution order:** Both phases are **independent** — FPCA-02 (Phase 37) and SPARSE-01 (Phase 38) have **no cross-phase hard dependency** (as in prior implementation milestones), and each touches a disjoint area of the codebase (extend `regression.rs`/new `fpca_variants.rs` vs extend `irreg_fdata/`). They may be planned and executed in **any order or in parallel**. No forced sequence.

## Performance Metrics

**Velocity:**

- Total plans completed: 68 (21 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0 + 2 in v0.19.0 + 2 in v0.20.0 + 2 in v0.21.0 + 2 in v0.22.0 + 7 in v0.23.0 + 7 in v0.24.0 + 10 in v0.25.0)
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
| 37 | v0.26.0 | TBD (FPCA-02) |
| 38 | v0.26.0 | TBD (SPARSE-01) |

**Recent Trend:**

- Last milestone: v0.25.0 phases 34–36 (10 plans) — all completed + verified (5/5, 13/13, 7/7), milestone audit PASSED
- Trend: consistent ~45min per plan, 7 tests/plan average for implementation phases

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.26.0 implementation):

- v0.26.0 is an **implementation** milestone — real `fdars-core/src/` code (drawn top-first from the R-ecosystem backlog, after v0.19.0–v0.25.0). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is the **two remaining top-ranked** `R-BACKLOG.md` items (both tied at score 1.73, M-effort): **FPCA-02** (rank 18), **SPARSE-01** (rank 19) — completing the FPCA long tail and adding fast sparse-covariance estimation. Shipping these two exhausts the 1.73 tier. Both are **reuse-first** (extend `regression.rs` / new `fpca_variants.rs`; extend `irreg_fdata/`); no new algorithm subsystem, **no new crate dependency**. Plotting/rendering out of scope (numeric outputs only).
- Phase numbering **continues** from v0.25.0 (ended at Phase 36) → v0.26.0 starts at Phase 37. No reset.
- **One requirement per phase, two phases:** Phase 37 = FPCA-02, Phase 38 = SPARSE-01.
- **Both phases are independent** — no cross-phase hard dependency. Each touches a disjoint area of the codebase, so 37/38 may run in **any order or in parallel**.
- **FPCA-02 scope (from R-BACKLOG.md block):** extend `regression.rs` (or new `fpca_variants.rs`) — `fpca_der` (differentiate loadings / FPCA of the differentiated process), `fsvd` (bivariate SVD / cross-FPCA between two samples, paired left/right singular functions + singular values), `cross_covariance` (cross-covariance surface between two samples over their argument grids), `dynamical_correlation` (scalar functional-correlation association measure), and a sandwich-smoother / sparse-SVD (ssvd) FPCA path (smoothed-covariance estimator as an alternative to the raw thin-SVD). Reuses `fdata_to_pc_1d` + `covariance.rs`. Complements the shipped PACE core (FPCA-01, already handles the table-stakes PACE path). R baseline: `fdapace` (FPCAder, FSVD, GetCrCov, DynCorr, FCCor) / `refund` (fpca.sc sandwich, fpca.ssvd).
- **SPARSE-01 scope (from R-BACKLOG.md block):** extend `irreg_fdata/` — `face_covariance` (FACE fast-sandwich covariance estimator for sparse/irregular functional data), `mface_covariance` (the multivariate `mfaces` extension for multiple simultaneously-observed sparse variables), and integrated fitted continuous trajectories with pointwise confidence bands for sparse curves. Builds on `irreg_fdata::cov_irreg` and integrates with the shipped PACE `pace_fpca` (FPCA-01) machinery where applicable. R baseline: `face` (FACE) / `mfaces` (multivariate FACE) / `fdapace` (trajectory bands).
- R baselines matched by **capability**, not R's exact signatures.

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` where randomness is involved (any Monte-Carlo band / bootstrap paths need seeded reproducibility, mirroring INF-01's 999-perm default and FTS-02's white-noise bands).
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code + doctests, so this pointer matters.

### Pending Todos

None yet.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer). Both phases are independent and could otherwise parallelize, but sequential fallback may serialize them regardless.
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant since the milestone compiles real code + doctests. Also: `target/` grows to 100+GB and fills /home; `rm -rf target/debug/{incremental,examples}` frees ~108G if example LINK fails (MEMORY.md pointer).
- FPCA-02: the FPCAder differentiation convention (which loadings/scores are of the differentiated process), the `fsvd` bivariate-SVD normalization + sign convention, the `cross_covariance` surface weighting/integration, the `dynamical_correlation` definition + range, and the sandwich-smoother / ssvd smoothing choice each have specific `fdapace`/`refund` reference formulations; pin the derivative estimator, the cross-SVD normalization, the cross-covariance integration, the dynamical-correlation formula, and the sandwich-smoother bandwidth during planning, and document any divergence from the R baseline in rustdoc (as prior milestones documented divergences). Reuse `fdata_to_pc_1d` + `covariance.rs`.
- SPARSE-01: the FACE fast-sandwich smoother (tensor-product spline basis + sandwich weighting), the `mfaces` multivariate block-covariance construction (within-variable + cross-variable blocks), and the trajectory-band integration with `pace_fpca` (BLUP scores → fitted trajectory → pointwise Ω bands) each have specific `face`/`mfaces`/`fdapace` reference definitions; pin the FACE basis + sandwich weighting, the multivariate block layout, and the band construction during planning; reuse `cov_irreg` + `pace_fpca` rather than re-deriving. Note: `cov_irreg` gives a kernel-smoothed empirical covariance, not the FACE sandwich specifically — FACE is the new estimator.

## Deferred Items

v2 backlog items (from `.planning/research/R-BACKLOG.md`): FTS-01, FRE-01 (score 1.33, L-effort — the next tier after the 1.73 tier is exhausted by this milestone); FTS-03, FRE-02, REG-06, REP-02, CLUS-02 (score ≤ 1.00, L). FTS-01 builds on the shipped FTS-02's serial-dependence foundation; FRE-01/FRE-02 are the general Fréchet/object-data items (DENS-01 covered only the tractable 1D-density subset of Area 7; FRE-01 shares DENS-01's Wasserstein/quantile machinery). Explicit v0.26.0 exclusions: new crate dependency for FACE/SVD/covariance; plotting/rendering of FPCA loadings, cross-covariance surfaces, or trajectory bands; changes to existing public signatures (`fdata_to_pc_1d`, `pace_fpca`, `cov_irreg`); general object-space Fréchet machinery (FRE-01/FRE-02); the FTS forecasting subsystem (FTS-01).

Advisory tech-debt carried forward (not v0.26.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; prior VALIDATION.md files (incl. Phases 28–36) remain `draft` (Nyquist TODO); intentional R-baseline divergences documented in rustdoc across prior milestones.

**Also pending (not a backlog item):** a crate release for v0.23.0, v0.24.0 **and** v0.25.0 — version bump (Cargo.toml still 0.23.0) + PR + tag, since all three shipped real code (operator-driven ship-time step; a `v*` tag push triggers the crates.io publish).

## Session Continuity

Last session: 2026-08-21T13:00:00.000Z
Stopped at: v0.26.0 roadmap created (Phases 37–38) — ready to plan
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
