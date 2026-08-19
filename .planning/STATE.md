---
gsd_state_version: 1.0
milestone: v0.22.0
milestone_name: PACE Sparse FPCA & Elastic Multinomial
current_phase: 27
current_phase_name: elastic-multinomial-regression
status: verifying
stopped_at: "Completed 27-01-PLAN.md: elastic multinomial OvR (REG-03) — logistic.rs, 7 tests, 2 commits"
last_updated: "2026-08-19T08:29:12.693Z"
last_activity: 2026-08-19
last_activity_desc: Phase 26 complete, transitioned to Phase 27
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 2
  completed_plans: 2
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-18)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone ships the final P1 table-stakes item (FPCA-01, unified PACE sparse FPCA) and completes fdars' elastic-regression family (REG-03, elastic multinomial), each by orchestrating/extending existing `fdars-core/src/` code.
**Current focus:** Phase 27 — elastic-multinomial-regression

## Current Position

Phase: 27 (elastic-multinomial-regression) — EXECUTING
Plan: 1 of 1
Status: Phase complete — ready for verification
Last activity: 2026-08-19 — Phase 27 execution started

## Milestone Roadmap (v0.22.0)

Two **independent** phases, two requirements. Fourth implementation milestone drawn top-first from the R-ecosystem backlog (after v0.19.0 INF-01/INF-02, v0.20.0 T-01/T-02, v0.21.0 REG-01/REG-02) — real `fdars-core/src/` code. All additions are additive/non-breaking, `Result`-returning, with inline `#[cfg(test)]` tests and crate-root re-exports; **zero changes to existing public signatures.** Both items are reuse-first (orchestrate/extend existing code); no new algorithm subsystem, no new crate dependency. **After this milestone the P1 table-stakes tier is exhausted** — the remaining backlog is all P2/P3 differentiators.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 26 — PACE Sparse FPCA | FPCA-01 | New `pace_fpca.rs`: a unified PACE sparse FPCA estimator for sparse/irregular curves, chaining existing pieces in one call — kernel-smoothed mean + `irreg_fdata::cov_irreg` covariance surface → eigendecomposition (`regression::fdata_to_pc_1d`) → conditional-expectation (BLUP/PACE) FPC scores per curve (`spm::partial::conditional_expectation`) → fitted trajectories + pointwise confidence bands. `Result`-returning, crate-root re-export. **Reuse-first:** orchestrates `irreg_fdata` + `spm::partial` + `regression`. The **last P1 table-stakes** item (rank 8, score 2.31, M-effort). Also unblocks REG-01's deferred sparse/PACE path + SPARSE-01 (future). |
| 27 — Elastic Multinomial Regression | REG-03 | `elastic_regression/logistic.rs`: extend the binary `elastic_logistic` to multinomial (multi-class, K ≥ 2) elastic logistic over SRSF/SRVF space (one-vs-rest or softmax) + a `predict_elastic_multinomial` companion returning class probabilities / labels. `Result`-returning, crate-root re-export. **Reuse-first:** reuses the existing SRVF representation + warping machinery; binary `elastic_logistic` signature retained unchanged. Rank 3, score 3.00, S-effort — closes the sole partial in fdars' otherwise-complete elastic-regression family. |

**Execution order:** 26 and 27 are **mutually independent** (disjoint modules: new `pace_fpca.rs` orchestrating `irreg_fdata`/`spm::partial`/`regression` vs `elastic_regression/logistic.rs` + SRVF/warping machinery). Either may execute first; they may run in parallel. Default order 26 → 27 by backlog rank (FPCA-01 is the last P1 table-stakes item; REG-03 is a high-score S-effort differentiator). Mirrors the v0.19.0/v0.20.0/v0.21.0 two-independent-phases structure.

## Performance Metrics

**Velocity:**

- Total plans completed: 42 (21 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0 + 2 in v0.19.0 + 2 in v0.20.0 + 2 in v0.21.0)
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
| 26 | v0.22.0 | TBD (FPCA-01) |
| 27 | v0.22.0 | TBD (REG-03) |

**Recent Trend:**

- Last 5 plans: 21-01, 22-01, 23-01, 24-01, 25-01 — all completed + verified
- Trend: consistent ~45min per plan, 7 tests/plan average for implementation phases

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 26 P01 | 120 | 3 tasks | 2 files |
| Phase 27 P27-01 | 20 | 4 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.22.0 implementation):

- v0.22.0 is an **implementation** milestone — real `fdars-core/src/` code (the fourth drawn top-first from the R-ecosystem backlog, after v0.19.0, v0.20.0, v0.21.0). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is **FPCA-01** (rank 8, score 2.31, P1 table-stakes, M-effort — the *last* P1 table-stakes item) + **REG-03** (rank 3, score 3.00, P2 differentiator, S-effort — completes the elastic family). Both are **reuse-first** (orchestrate/extend existing code); no new algorithm subsystem, no new crate dependency.
- Phase numbering **continues** from v0.21.0 (ended at Phase 25) → v0.22.0 starts at Phase 26. No reset.
- **FPCA-01 and REG-03 are mutually independent** — disjoint modules (new `pace_fpca.rs` orchestrating `irreg_fdata`/`spm::partial`/`regression` vs `elastic_regression/logistic.rs` + SRVF/warping). Two phases (26 = FPCA-01, 27 = REG-03); no cross-phase dependency, so they may execute in either order or in parallel. Default order 26 → 27 by backlog rank. Mirrors the v0.19.0/v0.20.0/v0.21.0 two-independent-phases structure.
- **FPCA-01 scope decision (locked):** orchestrate existing pieces only — kernel-smoothed mean + `irreg_fdata::cov_irreg` covariance surface → eigendecompose (`regression::fdata_to_pc_1d`) → conditional-expectation (BLUP/PACE) scores per curve (`spm::partial::conditional_expectation`) → fitted trajectories + pointwise bands. A configurable/non-canonical PACE bandwidth-selection subsystem (new GCV/CV layer for the covariance surface) is **out of scope** — reuse existing smoothing bandwidth machinery / defaults / caller-supplied bandwidth. The sparse/PACE variant of REG-01 is *enabled* by this work but is a distinct capability, **deferred** to a future milestone.
- **REG-03 scope decision (locked):** extend `elastic_logistic` to multinomial (multi-class, K ≥ 2) logistic over SRSF/SRVF space (one-vs-rest or softmax) + `predict_elastic_multinomial`. Elastic multinomial beyond logistic (multinomial elastic PCR, ordinal) is **out of scope** — REG-03 closes only the single elastic-logistic multi-class partial. The existing binary `elastic_logistic` public signature is retained unchanged.
- **Reuse-first mandate (from R-BACKLOG.md FPCA-01/REG-03 blocks):**
  - FPCA-01 orchestrates `irreg_fdata` (+ `cov_irreg`) + `spm::partial::conditional_expectation` + `regression::fdata_to_pc_1d`; validates inputs via the existing `validation` module; returns the mean, eigenstructure, per-curve BLUP scores, and fitted trajectories with bands.
  - REG-03 reuses the existing SRVF representation + warping machinery; the binary `elastic_logistic` path stays unchanged and (at K = 2) should agree with the multinomial path.
- R baselines matched by **capability**, not R's exact signatures: `fdapace`/`fda` (FPCA-01); `fdasrvf` (REG-03).
- **Plotting is out of scope** — no rendering of FPCA trajectories, confidence bands, or class boundaries (consistent with the R-audit plotting exclusion).
- Crate is at **0.21.0 released** on crates.io (v0.21.0, tag pushed). The crate-version bump + PR + tag for this milestone is a **ship-time** step, decoupled from the implementation phases.

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` where randomness is involved.
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code + doctests, so this pointer matters.
- [Phase ?]: Do NOT subtract sigma2 from cov_irreg surface before eigendecomposition — sigma2 enters only as ridge term in Sigma_yi (Yao et al. 2005 §2.2)
- [Phase ?]: nalgebra symmetric_eigen() returns ASCENDING eigenvalues — collect pairs, sort descending; fix_svd_signs convention: largest-magnitude element positive
- [Phase ?]: helpers::linear_interp (public) for eigenfunction interpolation; irreg_fdata::linear_interp is pub(super) and inaccessible from pace_fpca.rs
- [Phase ?]: No new crate dependency: Beasley-Springer-Moro rational approximation replaces statrs for standard normal quantile
- [Phase ?]: OvR multinomial: reuse binary elastic_logistic K times unchanged (maximal reuse, no new dep)
- [Phase ?]: Row-normalise OvR sigmoid scores to class posteriors; zero-row guard assigns uniform 1/K

### Pending Todos

None yet.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer).
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant since the milestone compiles real code + doctests.
- FPCA-01: the PACE estimator wires together four existing pieces (`irreg_fdata::cov_irreg`, `spm::partial::conditional_expectation`, `regression::fdata_to_pc_1d`, kernel-smoothed mean) whose interfaces/conventions (grid alignment, measurement-error variance σ² for the BLUP, band construction) must be reconciled during planning; pin the conditional-expectation formulation + confidence-band definition and document them in rustdoc (as prior milestones documented divergences from R baselines).
- REG-03: choice of one-vs-rest vs softmax, the multinomial IRLS/optimizer convergence policy, and the K = 2 ↔ binary-`elastic_logistic` agreement condition should be pinned during planning and documented; input guards needed for fewer-than-2-classes and label/curve-count mismatch.

## Deferred Items

v2 backlog items (from `.planning/research/R-BACKLOG.md`, see REQUIREMENTS.md v2 section): **REG-01 sparse/PACE variant** (now *enabled* by FPCA-01's PACE infra, but a distinct capability — deferred), a configurable/non-canonical PACE bandwidth-selection subsystem, elastic multinomial beyond logistic (multinomial elastic PCR, ordinal), extra GLM families (inverse-Gaussian, negative-binomial) + configurable links, and the remaining P2/P3 differentiators (DEPTH-01, OUT-01, INF-03, REG-04/05/06, FTS-*, FRE-*, DENS-*, CLUS-*, REP-*, SPARSE-*, FPCA-02).

Advisory tech-debt carried forward (not v0.22.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; prior VALIDATION.md files remain `draft` (Nyquist TODO).

**Also pending (not a backlog item):** a crate release for v0.22.0 — version bump (0.21.0 → next) + PR + tag, since this milestone ships real code (operator-driven ship-time step).

## Session Continuity

Last session: 2026-08-19T08:29:12.683Z
Stopped at: Completed 27-01-PLAN.md: elastic multinomial OvR (REG-03) — logistic.rs, 7 tests, 2 commits
Resume file: None

## Operator Next Steps

- Plan the first phase with /gsd-plan-phase 26 (or 27 — they are independent)
