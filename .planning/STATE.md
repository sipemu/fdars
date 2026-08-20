---
gsd_state_version: 1.0
milestone: v0.24.0
status: Awaiting next milestone
stopped_at: Phase 33 complete — all phases complete
last_updated: "2026-08-20T21:05:42.367Z"
last_activity: 2026-08-20
last_activity_desc: Milestone v0.24.0 completed and archived
state_head: a0611f52fd6c9b43adc73e311987dd81c7b74969
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 7
  completed_plans: 7
milestone_name: Functional Regression & Clustering Breadth
current_phase: 33
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-20)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone draws the next three top-ranked P2 differentiator gaps from the v0.18.0 `R-BACKLOG.md` (score 1.73 each): additive functional regression + variable selection (REG-04), flexible mixed-effects regression (REG-05), and model-based/density functional clustering (CLUS-01), each by extending existing `fdars-core/src/` modules additively.
**Current focus:** Planning next milestone (v0.24.0 shipped & archived; run `/gsd-new-milestone`). Deferred operator ship-step: crate release for v0.23.0 + v0.24.0 (bump `fdars-core/Cargo.toml` from 0.23.0, tag/publish).

## Current Position

Phase: Milestone v0.24.0 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-08-20 — Milestone v0.24.0 completed and archived

## Milestone Roadmap (v0.24.0)

Three phases, three requirements — the next three top-ranked P2 differentiators from `R-BACKLOG.md` (all tied at score 1.73, M-effort): REG-04 (rank 12), REG-05 (rank 13), CLUS-01 (rank 15). Broadens fdars' functional-regression and functional-clustering families. Real `fdars-core/src/` code — additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports; **zero changes to existing public signatures.** All three are reuse-first (extend `scalar_on_function/` / `famm.rs` + `fof_regression.rs` / `clustering.rs` + `gmm/`); no new algorithm subsystem, **no new crate dependency.** Numeric outputs only — plotting/rendering out of scope.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 31 — Additive Functional Regression & Variable Selection | REG-04 | new `scalar_on_function/additive.rs`: FAM (backfitting over FPC-score components), GKAM + GSAM variants, a group-penalized `variable_selection` helper, a permutation-test wrapper, and a history-index (lagged-predictor-window) estimator — reusing `smoothing.rs` kernels + `fdata_to_pc_1d`. R baseline: `fdapace`/`fda.usc`/`refund`. Rank 12, score 1.73. |
| 32 — Flexible Mixed-Effects Regression | REG-05 | extend `famm.rs` (today only `fmm_test_fixed`) to random-effects estimation (denseFLMM-style mixed-model equations over FPC scores/basis coefficients, multiFAMM, fastFMM) + wire a flexible-RE function-on-function path into `fof_regression.rs` (base FoF already present at parity). R baseline: `denseFLMM`/`multifamm`/`fastFMM`/`refund` (pffr). Rank 13, score 1.73. |
| 33 — Model-Based & Density Functional Clustering | CLUS-01 | extend `clustering.rs` + `gmm/`: funHDDC per-group subspace covariance model, funFEM discriminative-subspace variant, DBSCAN over functional distances (reuse `distance.rs`), a kCFC subspace-embedding loop, and a joint align-and-cluster estimator (reuse `alignment/`). Numeric assignments/model outputs only. R baseline: `funHDDC`/`funFEM`/`fdacluster`/`fdapace`/`fdasrvf`. Rank 15, score 1.73. |

**Execution order:** All three phases are **independent** — REG-04, REG-05, and CLUS-01 have **no cross-phase hard dependency** (unlike v0.23.0's DEPTH→OUT chain), and each extends a disjoint area of the codebase. They may be planned and executed in **any order or in parallel**. No forced sequence.

## Performance Metrics

**Velocity:**

- Total plans completed: 51 (21 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0 + 2 in v0.19.0 + 2 in v0.20.0 + 2 in v0.21.0 + 2 in v0.22.0 + 7 in v0.23.0)
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
| 31 | v0.24.0 | TBD (REG-04) |
| 32 | v0.24.0 | TBD (REG-05) |
| 33 | v0.24.0 | TBD (CLUS-01) |

**Recent Trend:**

- Last 5 plans: v0.23.0 phases 28–30 (7 plans) — all completed + verified
- Trend: consistent ~45min per plan, 7 tests/plan average for implementation phases

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 31 P02 | 563 | 3 tasks | 3 files |
| Phase 32-flexible-mixed-effects-regression P02 | 15 | 2 tasks | 2 files |
| Phase 33 P00 | 7 | 3 tasks | 4 files |
| Phase 33 P02 | 5m | 3 tasks | 2 files |
| Phase 33-model-based-density-functional-clustering P03 | 11 | 3 tasks | 2 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.24.0 implementation):

- v0.24.0 is an **implementation** milestone — real `fdars-core/src/` code (drawn top-first from the R-ecosystem backlog, after v0.19.0–v0.23.0). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is the **next three P2 differentiators** from `R-BACKLOG.md` (all tied at score 1.73, M-effort): **REG-04** (rank 12), **REG-05** (rank 13), **CLUS-01** (rank 15) — broadening the functional-regression and functional-clustering families. All three are **reuse-first** (extend `scalar_on_function/` / `famm.rs` + `fof_regression.rs` / `clustering.rs` + `gmm/`); no new algorithm subsystem, **no new crate dependency**. Plotting/rendering out of scope (numeric outputs only).
- Phase numbering **continues** from v0.23.0 (ended at Phase 30) → v0.24.0 starts at Phase 31. No reset.
- **One requirement per phase, three phases:** Phase 31 = REG-04, Phase 32 = REG-05, Phase 33 = CLUS-01.
- **All three phases are independent** — no cross-phase hard dependency (unlike v0.23.0's DEPTH→OUT chain). Each touches a disjoint area of the codebase, so 31/32/33 may run in **any order or in parallel**.
- **REG-04 scope (from R-BACKLOG.md block):** new `scalar_on_function/additive.rs` — FAM (backfitting additive model over FPC-score components), a GKAM (generalized kernel additive) and GSAM (generalized spectral additive) variant, a group-penalized scalar-on-function `variable_selection` helper, a permutation-test wrapper, and a history-index (lagged-predictor-window) estimator. Reuses `smoothing.rs` kernels + `fdata_to_pc_1d`. R baseline: `fdapace` (FAM) / `fda.usc` (GKAM, GSAM) / `refund` (fosr.vs, fosr.perm, history-index).
- **REG-05 scope (from R-BACKLOG.md block):** extend `famm.rs` (today only `fmm_test_fixed`) with a dense functional linear mixed model (denseFLMM-style mixed-model equations over FPC scores / basis coefficients), a multivariate functional additive mixed variant (multiFAMM), fast functional mixed-model inference (fastFMM), and wire a flexible random-effects function-on-function path into `fof_regression.rs`. NOTE: the base function-on-function capability is **already present at parity** — this item extends only the flexible/RE variant, not the base capability. R baseline: `denseFLMM` / `multifamm` / `fastFMM` / `refund` (pffr).
- **CLUS-01 scope (from R-BACKLOG.md block):** extend `clustering.rs` + `gmm/` with a funHDDC-style per-group subspace covariance model, a funFEM discriminative-subspace clustering variant, a DBSCAN density clusterer over functional distances (reuse `distance.rs`), a kCFC subspace-embedding loop, and a joint align-and-cluster estimator (reuse `alignment/`). Numeric cluster assignments + model outputs only. Excludes co-clustering (funLBM, split out as CLUS-02, deferred). R baseline: `funHDDC` / `funFEM` / `fdacluster` (DBSCAN, Sangalli joint) / `fdapace` (kCFC) / `fdasrvf` (elastic k-means).
- R baselines matched by **capability**, not R's exact signatures.

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` where randomness is involved (the REG-04 permutation-test wrapper needs seeded reproducibility, mirroring INF-01's 999-perm default).
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code + doctests, so this pointer matters.
- [Phase ?]: GroupLasso only this phase; GroupMcp/GroupScad return FdarError::InvalidParameter (future work)
- [Phase ?]: Single shared StdRng::seed_from_u64(seed) for permutation_test_fam — NOT per-iteration seed+k
- [Phase ?]: fof_re_regression: x_scores passed to fit_scalar_mixed_model WITHOUT h.sqrt() rescaling — L2 weighting already in fpca_x.project() (Pitfall 2)
- [Phase ?]: predict_fof_re: fixed-effect-only prediction, no RE for new subjects (matches fmm_predict convention)
- [Phase ?]: AkBk single-model funHDDC (not full 6-model R family) — documented in rustdoc
- [Phase ?]: nalgebra SVD in gmm/subspace.rs for MSRV 1.81 compliance (faer would require 1.84+)
- [Phase ?]: DBSCAN uses Vec<Option<usize>> for type-safe noise labeling in clustering_advanced
- [Phase ?]: kCFC uses k-means++ init + per-cluster fdata_to_pc_1d reconstruction error reassignment
- [Phase ?]: Empty-cluster fallback in kCFC: keep prior FPCA model to avoid NaN
- [Phase 33]: funFEM uses W^{-1}B via Cholesky+SVD (simplified Fisher-EM, no generalized-eigenvalue crate)
- [Phase 33]: align-cluster init: Fisher-Yates shuffle + strided pick to avoid degenerate same-group templates
- [Phase 33]: resp bootstrap from hard labels required to seed first Fisher-EM E-step

### Pending Todos

None yet.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer). All three phases are independent and could otherwise parallelize, but sequential fallback may serialize them regardless.
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant since the milestone compiles real code + doctests. Also: `target/` grows to 100+GB and fills /home; `rm -rf target/debug/{incremental,examples}` frees ~108G if example LINK fails (MEMORY.md pointer).
- REG-04: the FAM/GKAM/GSAM additive-model estimators and the history-index (lagged-window) definition have subtle reference formulations (`fdapace`/`fda.usc`/`refund`); pin the exact backfitting/kernel/spectral construction and the group-penalty for `variable_selection` during planning, and document any divergence from the R baseline in rustdoc (as prior milestones documented divergences).
- REG-05: `famm.rs` today provides only `fmm_test_fixed` — pin the denseFLMM mixed-model-equation formulation (over FPC scores vs basis coefficients), the multiFAMM/fastFMM variant shapes, and the flexible-RE `fof_regression.rs` wiring during planning. The base FoF capability must stay untouched (extend only the flexible/RE variant).
- CLUS-01: the funHDDC per-group subspace covariance model, funFEM discriminative-subspace step, DBSCAN eps/min-points semantics over functional distances, kCFC subspace-embedding loop, and the joint align-and-cluster estimator each have distinct reference definitions; pin each estimator's numeric outputs + a cluster-agreement (adjusted-Rand/accuracy) test metric during planning; reuse `distance.rs`/`gmm/`/`alignment/`.

## Deferred Items

v2 backlog items (from `.planning/research/R-BACKLOG.md`, see REQUIREMENTS.md v2 section): REP-01, FTS-02 (score 1.73, P2); DENS-01, FPCA-02, SPARSE-01 (score 1.73, P3); FTS-01, FRE-01 (score 1.33, L); FTS-03, FRE-02, REG-06, REP-02, CLUS-02 (score ≤ 1.00, L). CLUS-02 (funLBM co-clustering) is the L-effort item split out of CLUS-01. Also from prior milestones: REG-01 sparse/PACE variant, configurable PACE bandwidth-selection subsystem, extra GLM families + configurable links. Explicit v0.24.0 exclusions: plotting/rendering of regression fits, cluster assignments, or mixed-model diagnostics; boosting/Bayesian functional regression (REG-06); functional co-clustering (funLBM, CLUS-02); the base function-on-function regression (already present — REG-05 extends only the flexible/RE variant); new crate dependencies; changes to existing public signatures.

Advisory tech-debt carried forward (not v0.24.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; prior VALIDATION.md files (incl. Phases 28–30) remain `draft` (Nyquist TODO); intentional R-baseline divergences documented in rustdoc (Fast-MUOD, univariate depthgram, response-permutation FLM, symmetric extremal/ERL).

**Also pending (not a backlog item):** a crate release for v0.23.0 — version bump (Cargo.toml still 0.22.0 → 0.23.0) + PR + tag, since v0.23.0 shipped real code (operator-driven ship-time step); and, later, the same for v0.24.0.

## Session Continuity

Last session: 2026-08-20T20:37:59.488Z
Stopped at: Phase 33 complete — all phases complete
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
