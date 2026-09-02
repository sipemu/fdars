---
gsd_state_version: 1.0
milestone: v0.34.0
milestone_name: k-Shape Clustering & Shape-Based Distance
current_phase: 63
current_phase_name: SBD-based k-medoids & Wrap-up
status: planning
stopped_at: Phase 62 complete, ready to plan Phase 63
last_updated: "2026-09-02T12:54:53.740Z"
last_activity: 2026-09-02
last_activity_desc: Phase 62 complete, transitioned to Phase 63
state_head: bca287d60f287aa1d39248be9ef22f31e1a33691
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 2
  completed_plans: 2
  percent: 67
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-02)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against reference ecosystems — this milestone promotes GAP-03 (k-Shape clustering & Shape-Based Distance), the third-ranked item from the v0.31.0 `GAP-BACKLOG.md`.
**Current focus:** Roadmap created for v0.34.0 (Phases 61–63). Next: `/gsd-plan-phase 61`.

## Current Position

Phase: 63 — SBD-based k-medoids & Wrap-up
Plan: Not started
Status: Ready to plan
Last activity: 2026-09-02 — Phase 62 complete, transitioned to Phase 63

## Milestone Roadmap (v0.34.0)

Three phases, 5 requirements (KSH-01..05) — an implementation milestone promoting GAP-03 (score 2.12, M-effort), rounding out the curve-clustering family alongside v0.32.0's GAK kernel-k-means. Real `fdars-core/src/` changes: SBD in a new `src/metric/sbd.rs` (peer of `gak.rs`/`soft_dtw.rs`), k-Shape in a new top-level `src/kshape.rs` (peer of `kernel_kmeans.rs`), the SBD-k-medoids convenience a thin adapter at the bottom of `kshape.rs`. Additive/non-breaking, no new crate dependency; **publishes to crates.io on the `v0.34.0` tag**. All four researchers converged on a **strict sequential dependency chain** (SBD core → k-Shape fit+predict → SBD-k-medoids) that cannot be reordered or parallelized. Fine granularity + disjoint per-phase correctness gates → three phases. Phase numbering continues from v0.33.0 (ended at 60) → Phase 61.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 61 — SBD Distance Core | KSH-01, KSH-02 | New `src/metric/sbd.rs`. FFT normalized-cross-correlation `sbd(x,y) -> (distance, shift)` + public n×n `sbd_distance_matrix`. Five make-or-break FFT/NCC gates: zero-pad to `next_power_of_two(2m−1)` (else circular wrap), explicit IFFT scale (rustfft unnormalized — divide by `fft_len`), coefficient-normalized NCC (÷ `‖x‖·‖y‖`), signed-lag extraction (not `fft_len − k` wrap), mandatory z-normalization. Verify: `sbd(x,x) ≈ 0`, symmetry, shifted-copy at correct shift, offset/scale invariance, constant-series guard, NCC ∈ [−1,1]. Pure `&[f64]` — no `to_dmatrix` in hot loop; `FftPlanner` is `!Send` → one per rayon task. |
| 62 — k-Shape Clustering & Predict | KSH-03, KSH-04 | New top-level `src/kshape.rs`. `kshape_fd` (iterative SBD assignment + shape-extraction centroids, `n_init` restarts default 10, in-place empty-cluster recovery, deterministic per-restart seeding, non-increasing objective) + `KShapeResult::predict`. **Main algorithmic phase** — shape extraction is the only genuinely-new numerical piece: TOP eigenvector (largest eigenvalue; nalgebra `SymmetricEigen` returns ascending → take the last) of `M = Qᵀ Sᵀ S Q`, members shift-aligned + z-normalized, SIGN-fixed by correlation to members, centroid re-z-normalized. Everything else (restart loop, seeding, recovery, predict) mirrors `kernel_kmeans.rs`. Verify: two-shifted-motif recovery at high purity (centroid corr > 0.99), determinism (same seed = identical labels; seq == parallel), empty-cluster no-panic, inertia monotone, `predict(train) == cluster`. |
| 63 — SBD-based k-medoids & Wrap-up | KSH-05 | `sbd_kmedoids` convenience at the bottom of `kshape.rs` (build SBD matrix → feed existing `kmedoids_from_distances`) — a REAL public deliverable, not just a doc example. Crate-root re-exports (`pub mod kshape`; `kshape_fd`, `KShapeConfig`, `KShapeResult`, `sbd_kmedoids`; `metric::{sbd, sbd_distance_matrix, SbdResult}`) + `prelude` additions + criterion benchmark. Deferred to the final phase to avoid partial public API exposure. Verify: SBD-matrix-not-L2 integration test + doctest; additive/non-breaking (28 examples + WASM + R unaffected); whole-crate fmt/clippy/test gates. |

**Execution order (dependency-driven — strict chain):** 61 → 62 → 63. No reordering or parallelization is possible. Phase 61 front-loads the five FFT/NCC numerical make-or-break gates; Phase 62 is the main algorithmic phase (shape-extraction centroids); Phase 63 is the thin k-medoids adapter + crate-root re-exports + benchmark.

## Performance Metrics

**Velocity:**

- Total plans completed: 101+ (across v0.14.0–v0.33.0)
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
| 61–63 | v0.34.0 | 0/TBD |

**Recent Trend:**

- Last milestone: v0.33.0 phases 57–60 (4 plans) — audit PASSED 7/7, shipped `v0.33.0` (crate 0.32.0 → 0.33.0). Promoted GAP-02 (shapelets).
- Trend: v0.34.0 stays in implementation shape — real code, normal test/clippy/fmt gates, crate publish on tag. Reuse-heavy (`rustfft`, `nalgebra` `SymmetricEigen`, the v0.33.0 `shapelet::z_normalize_window`, `kernel_kmeans.rs` patterns, `alignment::clustering::kmedoids_from_distances`), effort M for a mature codebase, ~2 new files (`metric/sbd.rs`, `kshape.rs`). Three phases driven by the strict SBD → k-Shape → k-medoids dependency chain, not padding.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.34.0):

- **Implementation milestone, publishes on tag** — v0.34.0 makes real `fdars-core/src/` changes and **will** bump the crate version + publish to crates.io on the `v0.34.0` tag. Normal test/clippy/fmt gates apply. (audit-milestone-no-tag does NOT apply.)
- **SBD in `src/metric/sbd.rs`, k-Shape in top-level `src/kshape.rs`** — SBD is a distance primitive (peer of `gak.rs`/`soft_dtw.rs`), reusable independently by k-medoids and future consumers; k-Shape is a full clustering algorithm (peer of `kernel_kmeans.rs`). One-way dependency: `kshape.rs` imports `metric::sbd`, never the reverse. Mirrors the v0.32.0 GAK→kernel-k-means precedent.
- **`sbd` returns `(distance, optimal_shift)`** — the shift is mandatory: the centroid update shift-aligns members using the SBD-returned shift before shape extraction. Discarding the shift breaks convergence (Pitfall 7).
- **Reuse-first, no new dependency** — `rustfft` (FFT NCCc, idiom from `fts/spectral.rs`/`seasonal/mod.rs`), `nalgebra` `SymmetricEigen` (shape-extraction eigenproblem), `shapelet::z_normalize_window`/`z_normalize_into` (v0.33.0, population std + `STD_EPS=1e-12` constant guard), `metric::self_distance_matrix`, `kernel_kmeans.rs` patterns (n_init, seeding, empty-cluster recovery, predict), `alignment::clustering::{kmedoids_from_distances, KMedoidsConfig, KMedoidsResult}`. No `Cargo.toml` dependency change; MSRV stays 1.81 (`SymmetricEigen` is nalgebra core, not behind `linalg`).
- **FFT correctness contract (Phase 61, silent-correctness killers)** — zero-pad to `next_power_of_two(2m−1)` (a too-short FFT wraps cross-correlation circularly); rustfft IFFT is UNNORMALIZED → divide by `fft_len` explicitly; NCC is COEFFICIENT-normalized (÷ `‖x‖·‖y‖`, from the pre-FFT z-normed vectors), not count-normalized; signed-lag extraction (`shift = if idx ≤ m−1 { idx } else { idx − fft_len }`), not a `fft_len − k` wrap; z-normalize both series inside `sbd()` unconditionally; constant-series (std ≈ 0) → distance 1.0, shift 0.
- **Shape-extraction centroid contract (Phase 62)** — centroid = TOP eigenvector (largest eigenvalue) of `M = Qᵀ Sᵀ S Q` with `Q = I − O/n_k` (mean-centering projection — skipping it silently degrades to k-means); nalgebra `SymmetricEigen` returns eigenvalues ascending → take the LAST column; members shift-aligned (SBD shift) + z-normalized before building `S`; SIGN-fix by `dot(v, mean_of_members) < 0 → negate` (NOT `dominant_sign_negative` from `regression.rs` — wrong convention); re-z-normalize the eigenvector before storing.
- **`n_init` default 10** — fdars convention (matches `KernelKmeansConfig`), exceeds tslearn's 1; k-Shape is init-sensitive. Deterministic per-restart seeding `StdRng::seed_from_u64(seed.wrapping_add(restart as u64))`; best restart by inertia.
- **In-place empty-cluster recovery** — farthest-point reassignment mirroring `kernel_kmeans.rs::recover_empty_clusters`; a documented divergence from tslearn's full-restart. Never panic on an empty cluster.
- **`sbd_kmedoids` is a real public function** — the user explicitly chose KSH-05 as a public deliverable (thin convenience over `kmedoids_from_distances`), not merely a doc example. Lives at the bottom of `kshape.rs` (no dedicated module file).
- **Crate-root re-exports deferred to Phase 63** — `pub mod kshape` + all public re-exports + `prelude` additions land only in the final phase, to avoid exposing a partial public API mid-milestone.
- **Additive/non-breaking** — zero changes to existing public signatures (protects R + WASM bindings + 28 examples); only the new `metric/sbd.rs` + `kshape.rs` + crate-root re-exports.
- **Phase numbering continues** — v0.33.0 ended at Phase 60 → v0.34.0 starts at Phase 61. No reset.
- **5 requirements → 3 phases** (fine granularity, strict dependency chain): Phase 61 KSH-01/02, Phase 62 KSH-03/04, Phase 63 KSH-05. All 5 mapped, no orphans, no duplicates.

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; the additive SBD/k-Shape surface should be exposed to R/WASM bindings in a follow-up, not this milestone.

### Blockers/Concerns

- **Research flags (from SUMMARY.md)** — Phase 61 MEDIUM (FFT zero-padding / NCC normalization / lag indexing are numerical fundamentals; recommend `/gsd-plan-phase --research-phase 61` with known-answer tests cross-checked against tslearn/aeon); Phase 62 MEDIUM (shape-extraction eigenvector formulation, nalgebra ascending order, sign-fix criterion — `/gsd-plan-phase --research-phase 62`, two-group known-answer test before implementation); Phase 63 NONE (standard wrapper + re-exports + benchmark). Non-blocking for the roadmap.
- **Approximate eigendecomposition for large m (>500)** — shape-extraction eigh is O(m³); for typical m ≤ 200 negligible. Deferred to a follow-up if profiling shows a bottleneck. No design impact now.
- **1D-curves-only scope** — v0.34.0 is univariate curves only; multivariate SBD, variable-length series, and other clustering families (KSH-BREADTH) deferred. Document in rustdoc. Non-blocking.
- Historical build/CI hazards (MEMORY.md) apply this implementation milestone: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); watch `/tmp` and `target/` disk pressure on full builds; prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds; audit-milestone-no-tag does NOT apply (this ships code → tag as normal).

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Shape-clustering | KSH-BREADTH (multivariate/variable-length SBD, hierarchical/other clustering families) | Deferred | v0.34.0 | future milestone |
| Shapelets | LSH-01 (gradient learning-shapelets) — needs autodiff through the distance; ties to GAP-08 | Deferred | v0.33.0 | future milestone |
| Shapelets | SHP-BREADTH (multivariate/DTW-shapelet/ROCKET) | Deferred | v0.33.0 | future milestone |
| Kernel-methods | SVM-01 (native in-crate kernel-SVM / QP solver) — Gram export (GAK-05/06) covers the use case in the interim | Deferred | v0.32.0 | future milestone |
| Kernel-methods | KRN-01 (additional curve kernels + kernel-PCA/SVM consumers reusing GAK Gram) | Deferred | v0.32.0 | future milestone |
| Backlog | GAP-05/06/07/08 (FOptDes, PEER, wavelet regression, differentiable core) — carry forward, drawn top-first | Deferred | v0.32.0 | future milestones |
| API-breaking | APIB-01 — breaking removal of the 6 `#[deprecated]` forms from v0.30.0 | Deferred | v0.30.0 | future 1.0-readiness |

## Session Continuity

Last session: 2026-09-02T14:30:00.000Z
Stopped at: Phase 62 complete, ready to plan Phase 63
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 61`
