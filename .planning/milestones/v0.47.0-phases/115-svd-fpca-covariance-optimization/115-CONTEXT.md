# Phase 115: SVD/FPCA & Covariance Optimization - Context

**Gathered:** 2026-09-23
**Status:** Ready for planning
**Mode:** Smart discuss — optimization phase (constraints pre-locked; targets steered by the Phase 113 ranking; specific technique at Claude's discretion under strict numeric-identity). Research ENABLED (like Phase 114, the hot paths may already be near-optimal — profile before assuming leverage).

<domain>
## Phase Boundary

Optimize the next-ranked hot paths (SVD/FPCA + covariance surface, conditional on the Phase 113
ranking) with the same non-breaking + numeric-identity proof pipeline (OPT-02), AND land the
consolidated cross-cutting proof (OPT-03). This is the LAST optimization phase; report is 116,
release is 117.

Requirements: OPT-02, OPT-03.

</domain>

<decisions>
## Implementation Decisions

### Target selection — driven by the Phase 113 ranking (committed hotspot_ranking.md)
- **Covariance surface (`face_covariance` in `fdars-core/src/irreg_fdata/face.rs`) is the CLEAN PRIMARY target** — it ranked #4 by leverage and PASSED the noise gate (noise-stable criterion cells). Optimize this directly (numeric-identity-safe), with committed before/after criterion + golden-value proof.
- **SVD-proper (`audit_p6_svd_nalgebra` / `_faer_seq` / `_conversion`) is NOISE-EXCLUDED** (CI-width ≥5%, some RSD ≥5%). Per the operator-approved Phase-113 hand-off (STATE.md), Phase 115 MUST **re-benchmark the SVD paths at higher criterion sample counts** (or longer measurement time) so CI-width <5% BEFORE selecting an SVD path as an optimization target. Only optimize an SVD path if, after re-benchmarking, a clean numeric-identity-safe win with meaningful leverage exists; otherwise document it as near-optimal (the honest Phase-114 precedent). FPCA has noise-stable cells + cross-language comparators — a secondary candidate.
- **Honesty clause (Phase 114 precedent):** if a target is already near-optimal (obvious wins taken) and no numeric-identity-safe change has statistically-significant leverage, report that honestly — a numeric-identical cleanup + "already near-optimal, here is the before/after proof" is a valid OPT-02 outcome. Do NOT relax a golden tolerance or fabricate a speedup.

### Non-negotiable constraints (milestone-wide)
- **Strictly non-breaking + numeric-identical** (≤1e-10 / bit-exact). Float-reassociation / parallel-reduction drift is the primary risk — keep accumulation order; no new `par_iter().sum()` reductions. Optimizations live in private internals; no public API/signature/behavior change (protects R + WASM + 28 examples). Gate behind `linalg`/`parallel` only if default-feature behavior is unchanged.
- **NO new crate dependency.** Reuse `nalgebra` / `faer`(linalg) / `rayon`(parallel) / criterion.
- **Committed BEFORE-baseline first** (Anti-Pattern 3), then optimize, then committed AFTER proving the change.

### OPT-03 — the consolidated cross-cutting proof (lands with this phase)
After the optimization(s), prove and commit the consolidated battery:
- `cargo test -p fdars-core --release --features linalg` green (release-build numeric-identity proof).
- The v0.43.0 **determinism guardrail** green (equivalence_phase47/48/49/50 + any golden fixtures added in 114/115).
- **All FIVE build variants compile:** default / `linalg` / `serde` / `linalg,serde,parallel` / WASM (`wasm32-unknown-unknown`). NOTE: the `serde` build was verified GREEN on 2026-09-23 (the old pre-existing ShapeletTransformClassifier/ClassifFit break is repaired) — do not regress it.
- **pmat quality-gate green:** `pmat quality-gate --checks complexity,satd,dead-code,duplicates` (the curated gate from v0.46.0 — entropy excluded as non-deterministic). Any decomposition helpers added must stay ≤23 cognitive complexity.
- `cargo fmt --check` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.

### Claude's Discretion
- The specific covariance optimization technique (allocation reuse, loop tightening, cache-friendlier layout, avoiding redundant symmetric-matrix work), the golden-value fixtures (seed/size), and whether SVD gets optimized or documented-near-optimal after re-benchmarking — all at Claude's discretion within the numeric-identity + non-breaking constraints, driven by the profiling evidence.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/irreg_fdata/face.rs` — `face_covariance` (FACE fast covariance estimation) — the clean primary target.
- `fdars-core/src/covariance.rs`, `src/fpca_variants.rs`, `src/regression.rs` (`fdata_to_pc_1d`) — SVD/FPCA + covariance code.
- `fdars-core/benches/audit_hotpaths.rs` — `audit_p6_svd_nalgebra`/`_faer_seq`/`_conversion` (the noise-excluded SVD benches — re-benchmark here) + the covariance/face benches (before/after timing source).
- Phase-114 pattern: golden-value equivalence test (`fdars-core/tests/equivalence_phase114.rs`) + committed before/after criterion dirs under `.planning/phases/.../{before,after}-criterion/`. Reuse for 115.
- The v0.43.0 determinism guardrail (equivalence_phase47/48/49/50) + the v0.46.0 pmat quality-gate (`pmat quality-gate --checks complexity,satd,dead-code,duplicates`).

### Established Patterns
- Column-major `FdMatrix`; accumulation-order-sensitive numeric code; feature-gated parallelism macros in `parallel.rs`.

### Integration Points
- `benchmarks/results/hotspot_ranking.md` (Phase 113) is the target-selection authority + the before-baseline reference. Phase 116 re-runs the harness to show the post-optimization gains.

</code_context>

<specifics>
## Specific Ideas

- Start with `face_covariance` (clean, ranked) — profile it (like 114) to find the concrete numeric-identity-safe opportunity before assuming leverage.
- Re-benchmark the SVD paths at higher criterion sample count FIRST; decide target vs document-near-optimal from the tightened numbers.
- OPT-03 is the consolidated proof for the WHOLE optimization pass (114 + 115) — run the full battery once at the end of 115.

</specifics>

<deferred>
## Deferred Ideas

- Publishing the head-to-head speedup narrative — Phase 116 report.
- Release bump/CHANGELOG/tag — Phase 117.
- asv/continuous perf tracking (PERF-CI-01) — future milestone.

</deferred>
