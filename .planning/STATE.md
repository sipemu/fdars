---
gsd_state_version: 1.0
milestone: v0.34.0
milestone_name: k-Shape Clustering & Shape-Based Distance
status: planning
last_updated: "2026-09-02T12:00:21.032Z"
last_activity: 2026-09-02
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-02)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against reference ecosystems — this milestone promotes GAP-02 (discovery-based shapelet transform & classification), the second-ranked item from the v0.31.0 `GAP-BACKLOG.md`.
**Current focus:** Roadmap created for v0.33.0 (Phases 57–60). Next: `/gsd-plan-phase 57`.

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-09-02 — Milestone v0.34.0 started

## Milestone Roadmap (v0.33.0)

Four phases, 7 requirements (SHP-01..07) — an implementation milestone promoting GAP-02, the only backlog gap corroborated across three reference libraries (sktime, pyts, tslearn). Real `fdars-core/src/` changes in a new `src/shapelet/` submodule, additive/non-breaking, no new crate dependency; **publishes to crates.io on the `v0.33.0` tag**. All four researchers converged on a **strict compile-time dependency chain** (distance core → discovery → transform → classifier) that cannot be reordered or parallelized — each of the four `src/shapelet/` files depends one-way on the previous, so phase boundaries mirror file boundaries. Fine granularity + disjoint per-phase correctness gates → four phases (not one phase / four plans). Phase numbering continues from v0.32.0 (ended at 56) → Phase 57.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 57 — Shapelet Distance Core | SHP-01, SHP-02 | New `src/shapelet/distance.rs` + `mod.rs` skeleton. Per-window z-normalization (ddof=0, constant-window guard `std.max(1e-10)` → finite), `sdist` = **min** over sliding windows of z-normalized Euclidean distance with explicit `best_so_far` early-abandon, the `Shapelet` type. Pure `&[f64]` arithmetic, lowest risk — but two make-or-break gates: scale/offset-invariance (per-window norm) + known-motif recovery (min-not-mean). No `to_dmatrix` in the hot loop; use `row_to_buf` for row slices. |
| 58 — Discovery & Ranking | SHP-03, SHP-04, SHP-05 | New `src/shapelet/discovery.rs` (`ShapeletConfig`, `QualityMeasure`). Candidate generation (exhaustive OR contracted/random via `max_candidates`, seeded), information-gain with **optimal split threshold** (orderline sort + all-gap scan) + F-statistic quality, top-K + self-similarity pruning → `ShapeletSet`. **HIGHEST-risk phase** (most pitfalls): combinatorial tractability (naive O(n²·M³) intractable — contract designed in from the start), deterministic seeding, `total_cmp` + `(series_idx, offset)` tie-break for float ties. |
| 59 — Shapelet Transform | SHP-06 | New `src/shapelet/transform.rs` (`ShapeletTransformFit`, `shapelet_transform_fit`, `shapelet_transform`). Apply a fitted `ShapeletSet` → n×K distance-feature matrix (train + out-of-sample). Transform consistency: `transform(train)` reproduces fit-time distances within 1e-12; short-series guard → `Err(InvalidDimension)`; `all(is_finite())`. Crate-root transform re-exports land here. |
| 60 — Bundled Classifier | SHP-07 | New `src/shapelet/classifier.rs` — `ShapeletTransformClassifier` fit (discover → transform → classify) + predict; **kNN default** (`fclassif_knn_fit`), LDA optional via config enum. `classification/` consumed unmodified. The crate-root `pub mod shapelet` + all re-exports are **deferred to this final phase** to avoid partial public API exposure. Train/test leakage discipline in doctest + integration test; criterion benchmark added. |

**Execution order (dependency-driven — strict chain):** 57 → 58 → 59 → 60. No reordering or parallelization is possible; each `src/shapelet/` file depends one-way on the previous. Phase 57 front-loads the two numerical make-or-break gates; Phase 58 is the highest-risk phase; 59 owns transform consistency; 60 owns the user-facing pipeline + crate-root re-exports.

## Performance Metrics

**Velocity:**

- Total plans completed: 98+ (across v0.14.0–v0.32.0)
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
| 57–60 | v0.33.0 | 0/TBD |

**Recent Trend:**

- Last milestone: v0.32.0 phases 54–56 (3 plans) — audit PASSED 8/8, shipped + tagged `v0.32.0`. First implementation milestone after three audit cycles; promoted GAP-01 (GAK).
- Trend: v0.33.0 stays in implementation shape — real code, normal test/clippy/fmt gates, crate publish on tag. Reuse-heavy (builds on `matrix.rs`, `distance.rs`, `classification/`, `parallel.rs`), effort L for a mature codebase, ~800 LOC across four new files in `src/shapelet/`. Four phases (one file each) driven by the strict dependency chain, not padding.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.33.0):

- **Implementation milestone, publishes on tag** — v0.33.0 makes real `fdars-core/src/` changes and **will** bump the crate version + publish to crates.io on the `v0.33.0` tag. Normal test/clippy/fmt gates apply.
- **New `src/shapelet/` submodule, not inside `classification/`** — shapelets are a feature-engineering family (curves → distance features → classification) that produces a matrix *consumed by* classifiers, not implemented alongside them. Placing them in `classification/` would break its `#[non_exhaustive]` `ClassifMethod` contract and block future non-classifier consumers. Four files: `distance.rs`, `discovery.rs`, `transform.rs`, `classifier.rs` + `mod.rs` barrel.
- **Reuse-first, no new dependency** — reuses `FdMatrix`/`row_to_buf`/`row_l2_sq` (`matrix.rs`), `cross_distance_matrix` (`distance.rs`), `iter_maybe_parallel!` + `seed_for_thread` (`parallel.rs`), `fclassif_knn_fit`/`ClassifFit`/`ClassifResult` (`classification/fit.rs`), `FdarError` (`error.rs`). z-normalization + sliding-window Euclidean + IG are pure arithmetic. No new `Cargo.toml` entry.
- **Discovery-based only** — Ye & Keogh 2009 + Hills–Lines 2014; **not** learning-shapelets (LSH-01, deferred; needs autodiff through the distance, ties to GAP-08).
- **Bundled classifier defaults to kNN** — `fclassif_knn_fit` (canonical Hills/Lines; avoids the FPCA-on-distance-features oddity of LDA), with LDA selectable via a `ShapeletClassifier` config enum.
- **Quality measure defaults to information gain with optimal split threshold** — orderline sort + all-n−1-gap midpoint scan (never a fixed threshold); F-statistic selectable via a `QualityMeasure` enum (better under class imbalance; reuses existing `pub(crate) integrated_f_statistic`, subject to a Phase-58 adaptation check for the 2-group scalar case).
- **Per-window z-normalization is mandatory (Pitfall 1/2)** — normalize each length-L window independently (NOT the whole series); population std (ddof=0, pyts convention); clamp `std.max(1e-10)` so constant windows return a finite zero-ish vector. Scale/offset-invariance test must pass in Phase 57.
- **`sdist` is the min, not mean (Pitfall 3)** — strict `fold(INFINITY, min)` over sliding windows; known-motif recovery test is the Phase-57/58 key integration gate.
- **Early-abandon via explicit `best_so_far` (Pitfall 5)** — sequential inner loop short-circuits when the partial sum-of-squares exceeds `best_so_far`; answer identical to non-abandoned; window scan stays sequential (rayon only at the series/candidate level).
- **Combinatorial contract designed in from the start (Pitfall 4)** — `max_candidates` contracted/random search with deterministic seeding; naive O(n²·M³) exhaustive is intractable and must never be the only path. Tractability test: n=100, m=200 returns in seconds.
- **Self-similarity pruning (Pitfall 8)** — drop candidates whose source series + position range overlaps an already-selected shapelet; selected K span ≥ min(K, n_train) distinct series; no two transform columns correlate > ~0.95.
- **Transform consistency (Pitfall 9)** — store shapelets already-z-normalized in `ShapeletTransformFit`; `transform` uses them verbatim (no re-discovery, no re-normalization against test statistics); `transform(train)` reproduces fit-time distances within 1e-12. Short-series guard → `Err(InvalidDimension)` (Pitfall 10).
- **Deterministic ordering (Pitfall 11/13)** — `config.seed` threaded through; `total_cmp` + `(series_idx, start_offset)` tie-break (never `partial_cmp(...).unwrap()`); two same-seed fits produce identical shapelets + thresholds.
- **Crate-root re-exports deferred to Phase 60** — `pub mod shapelet` + all public re-exports land only in the final phase, to avoid exposing a partial public API mid-milestone.
- **Additive/non-breaking** — zero changes to existing public signatures (protects R + WASM bindings + 28 examples); only the new `src/shapelet/` module + crate-root re-exports.
- **Phase numbering continues** — v0.32.0 ended at Phase 56 → v0.33.0 starts at Phase 57. No reset.
- **7 requirements → 4 phases** (fine granularity, strict dependency chain): Phase 57 SHP-01/02, Phase 58 SHP-03/04/05, Phase 59 SHP-06, Phase 60 SHP-07. All 7 mapped, no orphans, no duplicates.

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; the additive shapelet surface should be exposed to R/WASM bindings in a follow-up, not this milestone.

### Blockers/Concerns

- **F-statistic 2-group adaptation** (research flag, Phase 58) — verify the existing `pub(crate) integrated_f_statistic` (full-curve) adapts to the 2-group scalar-distance case or needs a small new implementation. Non-blocking for the roadmap.
- **Contracted-search determinism assumes single-process** (research flag, Phase 58) — current `seed_for_thread` seeding is single-machine; document the assumption, defer distributed reproducibility. Non-blocking.
- **1D-curves-only scope** (research flag) — v0.33.0 is univariate curves only; multivariate/DTW-shapelet/ROCKET breadth (SHP-BREADTH) explicitly deferred. Document in rustdoc. Non-blocking.
- **Elastic-distance choice** (research flag) — implementation uses z-normalized Euclidean; note elastic alignment as a preprocessing alternative. Non-blocking.
- Historical build/CI hazards (MEMORY.md) apply this implementation milestone: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); watch `/tmp` and `target/` disk pressure on full builds; prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds; audit-milestone-no-tag does NOT apply (this ships code → tag as normal).

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Shapelets | LSH-01 (gradient learning-shapelets) — needs autodiff through the distance; ties to GAP-08 | Deferred | v0.33.0 | future milestone |
| Shapelets | SHP-BREADTH (multivariate/DTW-shapelet/ROCKET) | Deferred | v0.33.0 | future milestone |
| Kernel-methods | SVM-01 (native in-crate kernel-SVM / QP solver) — Gram export (GAK-05/06) covers the use case in the interim | Deferred | v0.32.0 | future milestone |
| Kernel-methods | KRN-01 (additional curve kernels + kernel-PCA/SVM consumers reusing GAK Gram) | Deferred | v0.32.0 | future milestone |
| Backlog | GAP-03/05/06/07/08 (k-Shape, FOptDes, PEER, wavelet regression, differentiable core) — carry forward, drawn top-first | Deferred | v0.32.0 | future milestones |
| API-breaking | APIB-01 — breaking removal of the 6 `#[deprecated]` forms from v0.30.0 | Deferred | v0.30.0 | future 1.0-readiness |

## Session Continuity

Last session: 2026-09-02T00:00:00.000Z
Stopped at: Phase 60 complete — all phases complete
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
