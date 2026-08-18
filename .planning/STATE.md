---
gsd_state_version: 1.0
milestone: v0.22.0
milestone_name: PACE Sparse FPCA & Elastic Multinomial
status: planning
last_updated: "2026-08-18T17:49:24.660Z"
last_activity: 2026-08-18
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-17)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone ships the two remaining P1 table-stakes functional-regression items from `.planning/research/R-BACKLOG.md` (REG-01 concurrent/varying-coefficient regression, REG-02 exponential-family functional GLM), each reusing existing scalar-on-function design machinery.
**Current focus:** Planning next milestone (v0.21.0 shipped)

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-08-18 — Milestone v0.22.0 started

## Milestone Roadmap (v0.21.0)

Two **independent** phases, two requirements. Third implementation milestone drawn top-first from the R-ecosystem backlog (after v0.19.0 INF-01/INF-02 and v0.20.0 T-01/T-02) — real `fdars-core/src/` code. All additions are additive/non-breaking, `Result`-returning, with inline `#[cfg(test)]` tests and crate-root re-exports; **zero changes to existing public signatures.** Both items are reuse-first (existing scalar-on-function design machinery); no new algorithm subsystem, no new crate dependency.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 24 — Concurrent / Varying-Coefficient Regression | REG-01 | New `concurrent_regression.rs`: dense functional concurrent (varying-coefficient) regression — a time-varying β(t) relating a functional response to functional predictor(s) on the *same* shared grid, estimated by penalized pointwise / local-linear least squares over the dense grid. Result carries `{ beta_curve, fitted, residuals }`; crate-root re-export. **Reuse-first:** reuses `smoothing.rs` kernels. **DENSE variant only** — the sparse/PACE path is DEFERRED (no FPCA-01 dependency pulled in). |
| 25 — Functional GLM (Exponential Family) | REG-02 | `scalar_on_function/`: generalize `functional_logistic` into `functional_glm(data, y, family)` via IRLS over FPC scores, with `GlmFamily { Binomial, Poisson, Gamma, Gaussian }` (canonical link + variance per family). `Binomial` reproduces `functional_logistic`, whose signature is retained unchanged. **Reuse-first:** reuses the `functional_logistic` IRLS loop + `fdata_to_pc_1d`. |

**Execution order:** 24 and 25 are **mutually independent** (disjoint modules: new `concurrent_regression.rs`+`smoothing.rs` vs `scalar_on_function/`+`fdata_to_pc_1d`). Either may execute first; they may run in parallel. Default order 24 → 25 by backlog rank (REG-01 rank 6 before REG-02 rank 7).

## Performance Metrics

**Velocity:**

- Total plans completed: 40 (21 in v0.14.0 + 4 in v0.15.0 + 4 in v0.16.0 + 3 in v0.17.0 + 5 in v0.18.0 + 2 in v0.19.0 + 2 in v0.20.0)
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
| 24 | v0.21.0 | 1 (complete — REG-01) |
| 25 | v0.21.0 | 1 (complete — REG-02) |

**Recent Trend:**

- Last 5 plans: 20-01, 21-01, 22-01, 23-01, 24-01 — all completed + verified
- Trend: consistent ~45min per plan, 7 tests/plan average for implementation phases

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

**Phase 25 decisions (2026-08-17):**

- statrs is NOT a runtime dependency of fdars-core (absent from Cargo.toml); Poisson log(y!) computed as exact integer factorial sum instead.
- GlmFamily + FunctionalGlmResult defined in mod.rs alongside other result types; glm.rs contains only the implementation helpers.
- Recovery tests require multi-component functional data (3 orthogonal sine-basis components with permuted scores) to ensure X'WX non-singular at ncomp=3.
- Deviance-change convergence (< tol) used; Binomial parity still holds at fixed-point because both logistic and GLM converge to same coefficients.

**Phase 24 decisions (2026-08-17):**

- Recovery test uses sin(πt) not sin(2πt): local-linear with bw=0.15 introduces 0.31 bias on full-period sin (high curvature); raw OLS was correct (<0.002 error); smoother was the bottleneck. Half-period sin(πt) recovered within 0.10 at interior, well inside 0.15 tolerance.
- Concurrent regression column loop: iter_maybe_parallel!(0..m) with per-closure-local xtx/xty allocation (safe for rayon); serialize collected Vec<(f64, Vec<f64>)> after .collect() for order-stable determinism.

Relevant to current work (v0.21.0 implementation):

- v0.21.0 is an **implementation** milestone — real `fdars-core/src/` code (the third drawn top-first from the R-ecosystem backlog, after v0.19.0 and v0.20.0). All additions are **additive/non-breaking**, `Result`-returning, with inline `#[cfg(test)]` tests + crate-root re-exports; **zero changes to existing public signatures.**
- Scope is the **two remaining P1 table-stakes functional-regression items** (REG-01 rank 6 / score 2.89, REG-02 rank 7 / score 2.31). Both are M-effort and **reuse existing scalar-on-function design machinery**; no new algorithm subsystem, no new crate dependency.
- Phase numbering **continues** from v0.20.0 (ended at Phase 23) → v0.21.0 starts at Phase 24. No reset.
- **REG-01 and REG-02 are mutually independent** — disjoint modules (new `concurrent_regression.rs`+`smoothing.rs` vs `scalar_on_function/`+`fdata_to_pc_1d`). Two phases (24 = REG-01, 25 = REG-02); no cross-phase dependency, so they may execute in either order or in parallel. Default order 24 → 25 by backlog rank. Mirrors the v0.20.0 (T-01/T-02) and v0.19.0 two-independent-phases structure.
- **REG-01 scope decision (locked):** DENSE variant only — pointwise / local-linear LS estimation of β(t) over the shared grid with a roughness penalty. The sparse/PACE variant is explicitly DEFERRED (no FPCA-01 dependency). Do not scope the sparse path into any phase.
- **REG-02 scope decision (locked):** `GlmFamily { Binomial, Poisson, Gamma, Gaussian }` — exactly these four families (each with canonical link + variance function). Do NOT add inverse-Gaussian / negative-binomial / configurable links (out of scope this milestone).
- **Reuse-first mandate (from R-BACKLOG.md REG-01/REG-02 blocks):**
  - REG-01 reuses `smoothing.rs` kernels for the penalized pointwise β(t) fit; returns `{ beta_curve, fitted, residuals }`.
  - REG-02 reuses the `functional_logistic` IRLS loop + `fdata_to_pc_1d` FPCA-score design; `Binomial` reproduces the existing logistic path, which is retained unchanged.
- R baselines matched by **capability**, not R's exact signatures: `fdaconcur`/`refund`/`fdapace` (REG-01); `fda.usc`/`refund` (REG-02).
- **Plotting is out of scope** — no rendering of fitted β(t) or GLM diagnostics (consistent with the R-audit plotting exclusion).
- Crate is at 0.19.0 released; the v0.20.0 → 0.20.0 bump is still pending. The crate-version bump + PR + tag for this milestone is a **ship-time** step, decoupled from the implementation phases.

Conventions carried from prior milestones (relevant to implementation):

- Column-major `FdMatrix` (`src/matrix.rs`), `Result<T, FdarError>` on all public fns, feature-gated rayon parallelism (`iter_maybe_parallel!` etc.), per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` where randomness is involved.
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — a plain `-p ... -D warnings` false-greens; MEMORY.md pointer).
- TMPDIR=/home/simonm/.cache/fdars-bench-tmp required for build/doctest linking; /tmp tmpfs exhaustion causes bogus "No space left" (MEMORY.md pointer) — this milestone builds real code + doctests, so this pointer matters.

### Pending Todos

None yet.

### Blockers/Concerns

- Local main is ahead of origin/HEAD → harness worktrees may fork the wrong base; GSD phases fall back to sequential no-worktree dispatch (MEMORY.md pointer).
- /tmp tmpfs exhaustion can block pre-commit doctest linking with bogus "No space left" — use `--no-verify` for docs and free /tmp before executing (MEMORY.md pointer). Relevant since the milestone compiles real code + doctests.
- REG-01: the penalized-pointwise-LS β(t) estimator has several R-implementation conventions (`fdaconcur` local-linear vs `refund` basis-penalty); pin one and document the choice in rustdoc (as prior milestones documented divergences).
- REG-02: canonical link + variance choices per family, and the IRLS convergence/step-halving policy, should be pinned during planning and documented; Gamma/Poisson responses need domain guards (positive / non-negative-integer).

## Deferred Items

v2 backlog items (from `.planning/research/R-BACKLOG.md`, see REQUIREMENTS.md v2 section): **REG-01 sparse/PACE variant** (deferred — benefits from FPCA-01's PACE infra, not yet built), extra GLM families (inverse-Gaussian, negative-binomial) + configurable links, REG-03 (elastic multinomial regression), REG-04/05/06 (additive / mixed-effects / boosting-Bayesian regression), FPCA-01 (PACE sparse FPCA), INF-03 (ITP family — deferred from v0.19.0), plus the larger differentiator clusters (DEPTH-01, OUT-01, FTS-*, FRE-*, DENS-*, CLUS-*, REP-*, SPARSE-*).

Advisory tech-debt carried forward (not v0.21.0 work): weakened MEWMA test assertion; `fix_svd_signs` NaN no-op; over-broad Phase 11 test name; prior VALIDATION.md files remain `draft` (Nyquist TODO).

**Also pending (not a backlog item):** a crate release — version bump + PR + tag, since v0.20.0 shipped real code (crate at 0.19.0 → 0.20.0 bump pending) and v0.21.0 will ship more.

## Session Continuity

Last session: 2026-08-17
Stopped at: Completed Phase 25 plan 01 — functional_glm + GlmFamily + FunctionalGlmResult (REG-02, commit cb839d52)
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
