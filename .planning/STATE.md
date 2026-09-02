---
gsd_state_version: 1.0
milestone: v0.32.0
milestone_name: Global Alignment Kernel & Kernel Clustering
current_phase: 56
status: completed
stopped_at: Phase 56 complete — all phases complete
last_updated: "2026-09-02T08:08:26.262Z"
last_activity: 2026-09-02
last_activity_desc: Phase 56 complete
state_head: f2224f5f70fdd961ff62d6ea62576b2a9926bfc0
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 3
  completed_plans: 3
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-02)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against reference ecosystems — this milestone promotes GAP-01 (Global Alignment Kernel), the top-ranked item from the v0.31.0 `GAP-BACKLOG.md`.
**Current focus:** Roadmap created for v0.32.0 (Phases 54–56). Next: `/gsd-plan-phase 54`.

## Current Position

Phase: 56
Plan: Not started
Status: All phases complete
Last activity: 2026-09-02 — Phase 56 complete

## Milestone Roadmap (v0.32.0)

Three phases, 8 requirements — first **implementation** milestone after three audit/consolidation cycles. Real `fdars-core/src/` changes, additive/non-breaking, no new crate dependency; **publishes to crates.io on the `v0.32.0` tag**. Strictly sequential dependency spine (all four researchers converged): GAK kernel core → Gram-matrix export → kernel-k-means. All algorithmic risk front-loaded into Phase 54. Fine granularity + distinct per-phase correctness gates → three small phases (not one phase / three plans). Phase numbering continues from v0.31.0 (ended at 53) → Phase 54.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 54 — GAK Kernel Core | GAK-01, GAK-02, GAK-03, GAK-04 | New `src/metric/gak.rs` (sibling to `soft_dtw.rs`). Log-domain (log-sum-exp) forward DP → triangular normalization (PSD) → n×n Gram (symmetric by assignment, parallel via `iter_maybe_parallel!`) → median-heuristic σ. Resolves the underflow / non-PSD / asymmetry / σ-degeneracy / NaN pitfalls. Required tests: no-underflow (m ≥ 100–400), unit diagonal, PSD eigenvalue ≥ −1e-8, bit-exact symmetry, σ-sensitivity, tslearn@0.9.0 regression < 1e-6. |
| 55 — Gram-Matrix Export | GAK-05, GAK-06 | Split train/predict API in `metric/gak.rs`. `gak_gram_train` (n×n, carries training self-kernels) + `gak_gram_predict` (n_test × n_train, cross-normalized against **stored training** self-kernels). Enforces the `SVC(kernel='precomputed')` contract; prevents the silent self-kernel-normalization bug. Rustdoc handoff example; O(n²) with diagonal computed once. Mechanical wrapping over the proven Phase-54 kernel. |
| 56 — Kernel-k-means | GAK-07, GAK-08 | New top-level `src/kernel_kmeans.rs` (peer of `clustering.rs`). Purely Gram-based (NO centroid-curve field), `n_init` random-partition restarts (Gram computed once, reused), empty-cluster recovery, deterministic seeding (`seed + restart_idx`), out-of-sample `predict` reusing the same kernel + normalization. Tests: 2-group purity 1.0, empty-cluster no-panic (k > natural), same-seed determinism, predict routes correctly. |

**Execution order (dependency-driven):** 54 → 55 → 56. Phase 54 must produce a correct log-domain PSD kernel before 55 can wrap it and 56 can consume it. No algorithmic risk in 55/56 — all risk is in 54.

## Performance Metrics

**Velocity:**

- Total plans completed: 95+ (across v0.14.0–v0.31.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10–45 | v0.15.0–v0.29.0 | 63 |
| 46–51 | v0.30.0 | 23 |
| 52–53 | v0.31.0 | 7 |
| 54–56 | v0.32.0 | 0/TBD |

**Recent Trend:**

- Last milestone: v0.31.0 phases 52–53 (7 plans) — audit PASSED 7/7, archived. Produced `GAP-BACKLOG.md`.
- Trend: v0.32.0 **returns to implementation shape** (like v0.24.0–v0.29.0) — real code, normal test/clippy/fmt gates, crate publish on tag. Reuse-heavy (builds on `metric/soft_dtw.rs` + `distance.rs`), effort S/M, 2 new files + 2 minor modifications.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.32.0):

- **Implementation milestone, publishes on tag** — unlike the three preceding audit/consolidation cycles, v0.32.0 makes real `fdars-core/src/` changes and **will** bump the crate version + publish to crates.io on the `v0.32.0` tag. Normal test/clippy/fmt gates apply.
- **Reuse-first, no new dependency** — GAK reuses the 2-row rolling-buffer DP + `softmin3` stabilization pattern from `metric/soft_dtw.rs`, the `self_distance_matrix`/`cross_distance_matrix` parallel pattern from `metric/mod.rs`, the `l2_distance_matrix` from `distance.rs`, and the `StdRng::seed_from_u64(seed + k)` seeding pattern. No new `Cargo.toml` entry.
- **Module placement** — GAK in `src/metric/gak.rs` (sibling to `soft_dtw.rs`, the 11th metric submodule); kernel-k-means in top-level `src/kernel_kmeans.rs` (peer of `clustering.rs`, NOT inside `metric/`). Gram export is functions in `gak.rs`, not a separate file.
- **Additive/non-breaking** — zero changes to existing public signatures (protects R + WASM bindings + 28 examples); only `pub mod gak;` / `pub mod kernel_kmeans;` additions + crate-root re-exports.
- **Log-domain is mandatory (Pitfall 1)** — the forward DP accumulates in log space (log-sum-exp); a `test_gak_no_underflow` with m ≥ 100–400 must pass before Phase 54 closes.
- **Normalized triangular form only (Pitfall 2/10)** — only `k(x,y)/sqrt(k(x,x)·k(y,y))` is PSD; the public API exposes only the normalized kernel. A PSD eigenvalue test (min eig ≥ −1e-8) is required. NOT `exp(-soft_dtw)` (that is not PSD).
- **Symmetrize by assignment (Pitfall 3)** — `G[j][i] = G[i][j]` after the upper-triangle fill; bit-exact symmetry test.
- **Split train/predict Gram API (Pitfall 8)** — `gak_gram_train` carries training self-kernels; `gak_gram_predict` returns n_test × n_train normalized against the **stored training** self-kernels. No monolithic single-matrix function.
- **Kernel-k-means: no centroid, random restarts (Pitfalls 5/6/7)** — Gram-based assignment (no centroid-curve field), `n_init` random-partition restarts (NOT k-means++ misapplied to similarities), empty-cluster recovery, deterministic per-restart seeding.
- **Phase numbering continues** — v0.31.0 ended at Phase 53 → v0.32.0 starts at Phase 54. No reset.
- **8 requirements → 3 phases** (fine granularity, dependency spine): Phase 54 GAK-01/02/03/04, Phase 55 GAK-05/06, Phase 56 GAK-07/08. All 8 mapped, no orphans, no duplicates.

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; the additive GAK/kernel-k-means surface should be exposed to R/WASM bindings in a follow-up, not this milestone.

### Blockers/Concerns

- **σ-heuristic sensitivity on real fdars curves** (research flag, Phase 54) — the Cuturi formula assumes normalized unit-variance series; FDA curves may have different amplitude scales. Phase 54 planning includes a σ-sensitivity check on representative data. Non-blocking for the roadmap.
- **Kernel-k-means initialization** (research flag, Phase 56) — random uniform restarts (current plan) vs kernel-k-means++; a lightweight Phase-56-planning experiment settles it. Non-blocking.
- **Series-length ratio guard (2:1)** and **PSD eigenvalue test scope** — Phase 54 planning confirms whether to hard-guard the ratio and clarifies the eigendecomposition dependency (nalgebra symmetric eigen). Non-blocking.
- Historical build/CI hazards (MEMORY.md) still apply this implementation milestone: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); watch `/tmp` and `target/` disk pressure on full builds; prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Kernel-methods | SVM-01 (native in-crate kernel-SVM / QP solver) — Gram export (GAK-05/06) covers the use case in the interim | Deferred | v0.32.0 | future milestone |
| Kernel-methods | KRN-01 (additional curve kernels + kernel-PCA/SVM consumers reusing GAK Gram) | Deferred | v0.32.0 | future milestone |
| Backlog | GAP-02/03/05/06/07/08 (shapelets, k-Shape, FOptDes, PEER, wavelet regression, differentiable core) — carry forward, drawn top-first | Deferred | v0.32.0 | future milestones |
| API-breaking | APIB-01 — breaking removal of the 6 `#[deprecated]` forms from v0.30.0 | Deferred | v0.30.0 | future 1.0-readiness |

## Session Continuity

Last session: 2026-09-02T00:00:00.000Z
Stopped at: Phase 56 complete — all phases complete
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 54`
