# Phase 46: Whole-Crate Profiling & Measurement - Context

**Gathered:** 2026-08-30
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase produces the **evidence base** for the whole v0.30.0 milestone: three ranked
inventories that drive Phases 47–51 concretely — a hot-path optimization target list, a
duplication/consolidation inventory, and an API-inconsistency inventory. It is **audit/measure
only**: zero behavior-changing edits to `fdars-core/src/` algorithms, no new crate dependency
(existing dev-deps only: criterion, feature-gated `dhat-heap`). Deliverables are report documents,
not production code changes.

</domain>

<decisions>
## Implementation Decisions

### Profiling Methodology & Scope
- Reuse the existing profiling harness (`benches/audit_hotpaths.rs` + the 10 module benches);
  add **throwaway** probe benches for the 9 currently-unbenchmarked subsystems. Permanent
  `[[bench]]` registration is explicitly Phase 51 (BENCH-01) — do not register them here.
- Use a 2–3 point N×M scaling grid (e.g. n ∈ {50, 200, 1000} curves × m ∈ {50, 200} eval points)
  to expose scaling behavior, not a single mid-size point.
- Allocation profiling via feature-gated `dhat-heap` on the **top candidate** workloads per
  subsystem (not exhaustive) — enough to surface `FdMatrix`↔`DMatrix` copies and per-iteration allocs.
- Record the benchmark **environment** (CPU governor, `RAYON_NUM_THREADS`) in the report to keep
  before/after comparisons honest (v0.14.0 audit governor/CPU-pinning caveat).

### Inventory Artifacts & Location
- The three inventories live as report docs under
  `.planning/phases/46-whole-crate-profiling-measurement/` (PROF-01/02/03 artifacts). No committed
  `src/` changes.
- Structure: **three separate ranked inventory docs** (hot-path, duplication, API-inconsistency)
  plus one short summary tying them together.
- Every inventory item carries a real criterion/allocation number and a `file:line` source anchor
  (required by the success criteria).
- Throwaway probe benches are **discarded / not registered**; only their measurements survive in
  the report. Permanent bench coverage is Phase 51's job.

### Ranking Criteria & Depth
- Hot-path ranking metric: **wall-time × representativeness** of the workload, with allocation
  count as a secondary signal.
- Dedup-leverage ranking: **(# call sites × complexity/drift-risk)** with `file:line` anchors —
  not raw duplicate-LOC count.
- API-inconsistency ranking: **user-facing impact + breadth** (config pattern vs result pattern vs
  redundant function), with a proposed **canonical form noted per item** (drives Phase 50).
- Depth bound: prioritize the 9 named reuse-first v0.19–v0.29 subsystems (`inference`, `fts`,
  `frechet`, `density_fda`, `fpca_variants`, `face`, `boosting_regression`, `fem_smoothing`,
  `coclustering`); time-box exhaustiveness to "enough to drive Phases 47–50 concretely" rather than
  an exhaustive whole-crate sweep.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/benches/`: 10 existing criterion benches — `audit_hotpaths.rs`, `matrix_benchmarks.rs`,
  `regression_benchmarks.rs`, `classification_benchmarks.rs`, `depth_benchmarks.rs`,
  `alignment_benchmarks.rs`, `seasonal_benchmarks.rs`, `smoothing_benchmarks.rs`,
  `basis_benchmarks.rs`, `explain_benchmarks.rs` — all `harness = false`.
- `dhat = "0.3"` dev-dependency already present; `dhat-heap` feature already declared in `[features]`
  (gates dhat allocation profiling, used in a prior Phase 4 audit).
- `criterion = { version = "0.5", features = ["html_reports"] }` dev-dependency.

### Established Patterns
- Column-major `FdMatrix` (`src/matrix.rs`) with efficient row methods (`row_to_buf`, `row_dot`,
  `row_l2_sq`) — allocation analysis should note where these are bypassed via `to_dmatrix()` copies.
- Feature-gated rayon via `parallel.rs` macros — parallelism-gap detection feeds PERF-03 (Phase 48).
- The 9 reuse-first subsystems (v0.19–v0.29) are the priority profiling targets.

### Integration Points
- Outputs (three ranked inventories) are consumed by: Phase 47 (PROF-01→PERF), Phase 49
  (PROF-02→CONS), Phase 50 (PROF-03→API), Phase 51 (module list→BENCH-01).

</code_context>

<specifics>
## Specific Ideas

- Honor the MEMORY.md operational pointers during any bench/build runs:
  `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` to avoid /tmp tmpfs exhaustion; watch `target/`
  growth (`rm -rf target/debug/{incremental,examples}` frees ~108G if example LINK fails);
  full clippy gate is `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- Audit-only milestone phase: per the "audit-milestone-no-git-tag" pointer, this phase makes no
  behavior change; no version bump / publish here.

</specifics>

<deferred>
## Deferred Ideas

- Permanent `[[bench]]` registration for the 9 unbenchmarked modules → Phase 51 (BENCH-01).
- Committing PERF-proof benchmarks as regression guards → Phase 51 (BENCH-02).
- Actually optimizing / dedup'ing / unifying anything surfaced here → Phases 47–50.

</deferred>
