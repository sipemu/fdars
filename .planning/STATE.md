---
gsd_state_version: 1.0
milestone: v0.35.0
milestone_name: Optimal Experimental Design for Sparse FDA (FOptDes)
status: planning
last_updated: "2026-09-02T15:00:00.000Z"
last_activity: 2026-09-02
progress:
  total_phases: 2
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-02)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against reference ecosystems — this milestone promotes GAP-05 (Optimal Experimental Design for Sparse FDA / FOptDes), rank 4 in the v0.31.0 `GAP-BACKLOG.md`.
**Current focus:** Roadmap created for v0.35.0 (Phases 64–65). Next: `/gsd-plan-phase 64`.

## Current Position

Phase: 64 of 65 (Criterion Machinery Core) — not started
Plan: —
Status: Ready to plan
Last activity: 2026-09-02 — Roadmap created for v0.35.0 (Phases 64–65), 5 requirements mapped

Progress: [░░░░░░░░░░] 0%

## Milestone Roadmap (v0.35.0)

Two phases, 5 requirements (FOD-01..05) — an implementation milestone promoting GAP-05 (score 2.12, M-effort), the first drawing from the *design* front of the backlog, built directly on the shipped `pace_fpca` estimator. All new code lives in **ONE** new top-level file `src/optimal_design.rs` (peer of `kshape.rs`/`kernel_kmeans.rs`) plus additive `lib.rs`/`prelude.rs` re-exports. Additive/non-breaking, **no new crate dependency** (MSRV 1.81 preserved, `linalg` NOT required); **publishes to crates.io on the `v0.35.0` tag**. All four researchers converged (HIGH confidence) on a **strict sequential dependency chain** (criterion machinery core → greedy selection + integration) that cannot be reordered or parallelized — the greedy loop delegates entirely to the criterion evaluator. Fine granularity + a clean two-phase build (mirrors v0.34.0 SBD-core → k-Shape and the general criterion-primitive → greedy-wrapper precedent). Phase numbering continues from v0.34.0 (ended at 63) → Phase 64.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 64 — Criterion Machinery Core | FOD-01, FOD-02, FOD-03 | New `src/optimal_design.rs`. Shared private `build_sigma_design` (p×p `Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p`, mirror `pace_fpca.rs:461–474`) + trajectory-reconstruction criterion (integrated Simpson-weighted BLUP-MSE, FOD-01) + FPC-score A-/D-optimality criterion (K×K posterior `Cov(ξ|Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ`, the `pace_fpca.rs:547–558` A_mat/Ω_i pattern, FOD-02) + public `#[must_use]` `design_criterion` evaluator with `DesignCriterion`/`OptimalityKind` enums (FOD-03). **NO greedy loop.** Make-or-break numerical gates (all known-answer testable): Σ_d assembly + σ²I ridge correct (shape `\|S\|×\|S\|`, not K×K); Simpson-weighted → `MSE(∅) ≈ Σ_k λ_k`, grid-invariant; score prior recovery `Cov(ξ|∅)=diag(λ)`; optimality sign — criterion monotone NON-increasing as points added; ridge-retry (`1e-8`) on near-singular Σ_d, never panic. Additive `lib.rs` re-export of enums + `design_criterion`. |
| 65 — Greedy Selection & Integration | FOD-04, FOD-05 | Greedy sequential forward-selection `optimal_design(model, config)` wrapper (start empty, add the candidate that most reduces the criterion until budget `p`, FOD-04) + `OptDesConfig` (Default, no `#[non_exhaustive]`) / `OptDesResult` (`#[non_exhaustive]`, selected indices + argvals + achieved-criterion trace) + two-stage `&PaceFpcaResult` entry point (read-only, no re-estimation, FOD-05) + additive crate-root/prelude re-exports + criterion benchmark. Thin orchestration — no new math. Gates: greedy **determinism WITH and WITHOUT `--features parallel`** (parallel evaluate, sequential argmin, smallest-index tie-break); duplicate-candidate exclusion; monotone achieved-criterion trace; input validation (`budget==0`, `budget>\|grid\|`, off-grid candidate, `ncomp==0`, `sigma2<=0`); additive/non-breaking (28 examples + WASM + R unaffected); whole-crate fmt/`clippy --all-targets --features linalg,parallel`/test gates; module doctest + benchmark. |

**Execution order (dependency-driven — strict chain):** 64 → 65. No reordering or parallelization is possible — the greedy loop in Phase 65 delegates entirely to Phase 64's `design_criterion`. Phase 64 front-loads every numerical make-or-break gate; Phase 65 is a thin deterministic greedy wrapper + config/result types + re-exports + benchmark.

## Performance Metrics

**Velocity:**

- Total plans completed: 105+ (across v0.14.0–v0.34.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10–45 | v0.15.0–v0.29.0 | 63 |
| 46–51 | v0.30.0 | 23 |
| 52–53 | v0.31.0 | 7 |
| 54–56 | v0.32.0 | 3 |
| 57–60 | v0.33.0 | 4 |
| 61–63 | v0.34.0 | 3 |
| 64–65 | v0.35.0 | 0/TBD |

**Recent Trend:**

- Last milestone: v0.34.0 phases 61–63 (3 plans) — audit PASSED 5/5, shipped `v0.34.0` (crate 0.33.0 → 0.34.0). Promoted GAP-03 (k-Shape / SBD).
- Trend: v0.35.0 stays in implementation shape — real code, normal test/clippy/fmt gates, crate publish on tag. Reuse-heavy (`pace_fpca.rs` Σ_yi/posterior-covariance machinery, `linalg::cholesky_solve`, `helpers::{simpsons_weights, linear_interp}`, `iter_maybe_parallel!`), effort M for a mature codebase, **ONE new file** (`optimal_design.rs`). Two phases driven by the strict criterion-core → greedy-wrapper dependency chain, not padding.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.35.0):

- **Implementation milestone, publishes on tag** — v0.35.0 makes real `fdars-core/src/` changes and **will** bump the crate version + publish to crates.io on the `v0.35.0` tag. Normal test/clippy/fmt gates apply. (audit-milestone-no-tag does NOT apply.)
- **Two-stage workflow, no re-estimation** — FOptDes is a pure design step over a supplied, already-estimated `PaceFpcaResult` (eigenfunctions, eigenvalues, σ²). The design step never re-estimates the covariance surface — that would couple estimation into design and duplicate `pace_fpca`. Re-estimation is explicitly out of scope.
- **ONE new file, `src/optimal_design.rs`** — self-contained algorithm (config + result + two enums + two public fns + ~3 private helpers, ~300 lines). Top-level peer of `kshape.rs`/`kernel_kmeans.rs`; not a submodule directory (no second-file primitive to separate). NOT under `metric/` (FOptDes is an experimental-design algorithm, not a distance metric).
- **Criterion-evaluator → greedy-wrapper split** — `design_criterion` (Phase 64) is a pure, public, `#[must_use]` function that scores a caller-supplied index set; `optimal_design` (Phase 65) is a thin greedy forward-selection wrapper that delegates entirely to it. Keeps `design_criterion` independently useful (evaluate a hand-chosen/historical design against the PACE prior) and mirrors the sbd.rs → kshape.rs precedent.
- **Both criteria share `build_sigma_design`** — trajectory-reconstruction (FOD-01) and score-prediction (FOD-02) both need `Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p`; a single shared private helper builds it (row-major, mirroring `pace_fpca.rs:461–474`), then each branch differs only in post-solve usage. Keeping both criteria in the core phase (64) lets the greedy wrapper stay pure orchestration.
- **Reuse-first, no new dependency** — `pace_fpca::PaceFpcaResult` (read-only borrow of eigenfunctions/eigenvalues/sigma2/argvals/ncomp; Σ_yi assembly `461–474`, A_mat/Ω_i posterior-covariance pattern `547–558`, ridge-retry `480–490`), `linalg::cholesky_solve` (p×p solves, row-major — always available, NOT behind `linalg` feature), `helpers::simpsons_weights` (quadrature — pass `&model.argvals`), `helpers::linear_interp` (only if off-grid candidates ever needed; MVP constrains candidates to the work grid), `iter_maybe_parallel!` (parallel candidate sweep, no RNG — criterion is deterministic). No `Cargo.toml` change; MSRV stays 1.81.
- **Trajectory criterion contract (Phase 64)** — integrated Simpson-weighted conditional BLUP-MSE `Σ_j w_j (Σ_k λ_k φ_k(t_j)² − φ_d(t_j)ᵀ Σ_d⁻¹ φ_d(t_j))`; MUST use `simpsons_weights(&model.argvals)` (never uniform 1.0/1/m — else grid-scale-wrong); quadratic form includes Ω off-diagonals; known-answer `MSE(∅) ≈ Σ_k λ_k` (grid-invariant).
- **Score criterion contract (Phase 64)** — K×K posterior `Cov(ξ|Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ` via `cholesky_solve`; A-opt = `trace(Cov)`, D-opt = `log det(Cov)` (NEGATIVE — posterior eigenvalues ≤ prior λ_k; minimize log-det covariance = maximize information); known-answer `Cov(ξ|∅)=diag(λ)` → `A(∅)=Σλ_k`, `D(∅)=Σ log λ_k`. Missing the `diag(1/λ)` prior term / forgetting σ²I are the classic silent-singularity bugs.
- **Optimality-sign / monotonicity gate** — Trajectory, A-opt, and D-opt must all be monotone NON-increasing as points are added (`criterion(S∪{t}) ≤ criterion(S) + 1e-12`). This guarantees the Phase 65 greedy loop minimizes (never maximizes) the objective. Assert in tests — O(1) overhead, catches make-or-break sign flips.
- **Greedy determinism (Phase 65)** — parallelize candidate EVALUATION via `iter_maybe_parallel!`, but take a SEQUENTIAL argmin with smallest-index tie-break (rayon `min_by` is not stable). Two same-config calls → byte-identical `selected_indices`, and seq == parallel builds agree. Mirror the `pace_fpca.rs` determinism-test pattern. Exclude already-selected indices each step (no duplicate points).
- **Candidates constrained to the work grid (MVP)** — every `config.candidate_grid` value must appear (within FP tolerance) in `model.argvals`; return `InvalidParameter` otherwise. Keeps criterion evaluation exact (index arithmetic, no interpolation). Off-grid interpolated candidates (via `linear_interp`) deferred to FOD-B5.
- **Crate-root re-exports deferred to Phase 65** — Phase 64 additively re-exports the enums + `design_criterion`; the full surface (`pub mod optimal_design` + `optimal_design`/`OptDesConfig`/`OptDesResult` + `prelude`) + benchmark land in the final phase, to avoid exposing a partial public API mid-milestone.
- **Additive/non-breaking** — zero changes to existing public signatures (protects R + WASM bindings + 28 examples); only the new `optimal_design.rs` + additive `lib.rs`/`prelude.rs` re-exports.
- **Phase numbering continues** — v0.34.0 ended at Phase 63 → v0.35.0 starts at Phase 64. No reset.
- **5 requirements → 2 phases** (fine granularity, strict dependency chain): Phase 64 FOD-01/02/03, Phase 65 FOD-04/05. All 5 mapped, no orphans, no duplicates. (FOD-02 kept in the core phase so the greedy wrapper stays pure orchestration.)

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; the additive FOptDes surface should be exposed to R/WASM bindings in a follow-up, not this milestone.

### Blockers/Concerns

- **Research flags (from SUMMARY.md)** — NEITHER phase needs a `--research-phase` pass (well-documented Ji & Müller 2017 / fdapace patterns; HIGH confidence). Both phases' numerical make-or-break gates warrant known-answer tests cross-checked against the formulas (Phase 64: Σ_yi assembly, score posterior covariance, Simpson-weighted integration, optimality sign; Phase 65: greedy determinism seq==parallel, duplicate exclusion). Non-blocking for the roadmap.
- **O(G·p·cost) greedy blowup** — brute-force per-candidate re-solve is O(budget · G · K³); fine for typical G ≤ 51, K ≤ 5, budget ≤ 10 (seconds). Rank-1 Cholesky / Sherman-Morrison update (FOD-B4) deferred until profiling shows a bottleneck at large grids (m ≫ 200). Correctness first. No design impact now.
- **Grid-constrained candidates only** — v0.35.0 MVP constrains candidate points to the model's work grid (exact index arithmetic, no interpolation). Non-grid candidates via eigenfunction interpolation (FOD-B5) deferred. Document in rustdoc. Non-blocking.
- **`!Send` in the parallel sweep** — the candidate closure must capture only immutable references to the pre-computed `M_S`; each closure allocates its own local Σ_d/M copy (no shared mutable state). `PaceFpcaResult` is `Send + Sync`; no FFT/`FftPlanner` here. Verify `cargo build/test --features parallel` compiles. Non-blocking.
- Historical build/CI hazards (MEMORY.md) apply this implementation milestone: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); watch `/tmp` and `target/` disk pressure on full builds (adding a benchmark grows `target/debug/`); prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds; audit-milestone-no-tag does NOT apply (this ships code → tag as normal).

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Optimal-design | FOD-BREADTH (FOD-B1 SR-criterion, FOD-B2 exhaustive/branch-and-bound, FOD-B3 CV-ridge selection, FOD-B4 rank-1 Cholesky update, FOD-B5 off-grid interpolated candidates) | Deferred | v0.35.0 | future milestone |
| Shape-clustering | KSH-BREADTH (multivariate/variable-length SBD, hierarchical/other clustering families) | Deferred | v0.34.0 | future milestone |
| Shapelets | LSH-01 (gradient learning-shapelets) — needs autodiff through the distance; ties to GAP-08 | Deferred | v0.33.0 | future milestone |
| Shapelets | SHP-BREADTH (multivariate/DTW-shapelet/ROCKET) | Deferred | v0.33.0 | future milestone |
| Kernel-methods | SVM-01 (native in-crate kernel-SVM / QP solver) — Gram export (GAK-05/06) covers the use case in the interim | Deferred | v0.32.0 | future milestone |
| Backlog | GAP-06/07/08 (PEER/lpeer, wavelet regression, differentiable core) — carry forward, drawn top-first | Deferred | v0.32.0 | future milestones |
| API-breaking | APIB-01 — breaking removal of the 6 `#[deprecated]` forms from v0.30.0 | Deferred | v0.30.0 | future 1.0-readiness |

## Session Continuity

Last session: 2026-09-02T15:00:00.000Z
Stopped at: Roadmap created for v0.35.0 (Phases 64–65); 5 requirements (FOD-01..05) mapped, traceability updated
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 64`
