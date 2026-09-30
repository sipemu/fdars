# Phase 114: Elastic Alignment Optimization - Context

**Gathered:** 2026-09-23
**Status:** Ready for planning
**Mode:** Smart discuss — optimization phase (constraints pre-locked by milestone; specific technique at Claude's discretion under strict numeric-identity). Research skipped per STATE (114–115 use the established v0.30.0 optimize pipeline + v0.43.0 determinism guardrail).

<domain>
## Phase Boundary

Optimize the **confirmed #1 bottleneck** — elastic alignment O(N²·M²) in `src/alignment/` — so it is
measurably faster while producing **numerically identical** output. Requirement: OPT-01.

The Phase 113 ranking confirmed the anchor as `audit_p3_elastic_cross` (elastic cross-alignment /
self-distance-matrix), **37.96 s @ N=500,M=50** (noise-stable, RSD 0.29%), rank #1 by both absolute
time and leverage. The banded variant `audit_p3_elastic_cross_banded` is 7.84 s @ N=500,M=50.

Boundary: this phase optimizes ONLY the elastic-alignment hot path. SVD/FPCA + covariance are
Phase 115. Report is Phase 116. No release/bump here.

</domain>

<decisions>
## Implementation Decisions

### Target hot path (from Phase 113 ranking + code inspection)
- `fdars-core/src/alignment/pairwise.rs`: `elastic_align_pair` (line 40) + `self_distance_matrix_impl` (line 243, O(N²) all-pairs). Public entries: `elastic_self_distance_matrix`/`_banded`; criterion group `audit_p3_elastic_cross` (+ `_banded`).
- Inner DP: `dp_alignment_core_banded` (`src/alignment/mod.rs:604`) → `dp_grid_solve_banded` with per-edge `dp_edge_weight_core`/`dp_edge_weight` + `dp_lambda_penalty` (the m²·~35 edge evaluations).

### ⚠ The obvious wins are ALREADY taken (code inspection 2026-09-23)
This hot path was optimized in prior milestones. `self_distance_matrix_impl` ALREADY: computes only the upper triangle N(N−1)/2 and mirrors (symmetry exploited), uses `iter_maybe_parallel!` (feature-gated parallelism), and precomputes all SRSF transforms once. `dp_alignment_core_banded` ALREADY: has a uniform-grid fast path with a precomputed √(dr/dc) slope table (avoids per-edge sqrt/division), banding support, and normalized-SRSF reuse. **So the low-hanging fruit (symmetry, pair-parallelism, SRSF precompute, DP slope table) is gone.** The remaining opportunity is subtle and must be found by profiling/inspection, not assumed — likely inside `dp_grid_solve_banded` (memory layout / allocation churn of the DP cost+traceback grids, inner-loop bounds, the edge-weight inner sum `dp_edge_weight_core`), or `reparameterize_curve`, or SIMD-friendly restructuring of the edge-weight accumulation. **This is why research is NOT skipped for this phase** (the STATE "skip research for 114–115" note assumed an obvious target; it isn't). If profiling finds NO numeric-identity-safe speedup with meaningful leverage, that is itself a valid, honest OPT-01 outcome to surface (report "hot path already near-optimal") rather than forcing a risky reassociation.

### Non-negotiable constraints (milestone-wide, apply to this phase)
- **Strictly non-breaking:** NO public signature/behavioral change. The optimization lives in private internals. Optimizations may be gated behind `linalg`/`parallel` only if they don't change default-feature behavior — the R binding's default-only feature set + WASM + 28 examples must compile and pass unchanged.
- **Numeric-identical (the hard constraint):** output must be bit-identical (or ≤1e-10) before/after. **Float-reassociation / parallel-reduction drift is the primary risk** (MEMORY.md): refactoring accumulation loops or introducing `par_iter().sum()` silently changes release-build results. KEEP accumulation order. Exploiting distance-matrix symmetry (compute upper triangle, mirror) is safe (same per-pair computation, just skipping redundant pairs) — that is the lowest-risk high-leverage optimization. Parallelizing independent pairs (each pair's DP is deterministic and independent) is safe if per-pair results are unchanged and written to fixed matrix positions.
- **Proof required (OPT-01):** committed before/after criterion evidence on the target cells (`audit_p3_elastic_cross` at N≥100, esp. N=500,M=50) showing a statistically significant speedup; a golden-value assertion (bit-exact / ≤1e-10) proving identical output; the full `cargo test` suite green + `cargo test --release --features linalg` (the numeric-identity proof) + the v0.43.0 determinism guardrail green after EVERY optimization commit.

### Verification pipeline (v0.30.0 optimize + v0.43.0 determinism)
- Capture the committed BEFORE-baseline first (criterion on the target cells) — never skip the before-baseline commit (Anti-Pattern 3). Then optimize. Then capture AFTER + prove speedup + numeric identity.
- Gates after each optimization commit: `cargo fmt --check`; `cargo clippy --all-targets --features linalg,parallel -- -D warnings`; full `cargo test`; `cargo test --release --features linalg`; the determinism guardrail; golden-value elastic assertion (≤1e-10). Reuse existing `nalgebra`/`faer`(linalg)/`rayon`(parallel)/criterion — NO new crate dependency.

### Claude's Discretion
- The exact optimization technique (symmetry exploitation, allocation reuse, DP inner-loop tightening, safe pair-parallelism), the golden-value fixture (seed/size), and whether to add a dedicated golden-value test module vs extend an existing one — all at Claude's discretion within the numeric-identity + non-breaking constraints.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/alignment/pairwise.rs` — `elastic_align_pair`, `elastic_align_pair_banded`, `elastic_self_distance_matrix`(_banded / _with_band), `self_distance_matrix_impl`.
- `fdars-core/benches/alignment_benchmarks.rs` — existing criterion groups incl. `audit_p3_elastic_cross`(_banded) — the before/after timing source.
- `parallel.rs` macros (`slice_maybe_parallel!`, etc.) — the feature-gated parallelism model to reuse.
- `metric.rs` `self_distance_matrix` pattern (referenced by pairwise.rs:185) — precedent for symmetric distance-matrix computation.
- The v0.43.0 determinism guardrail + existing golden-value test patterns (search tests for `golden`/`≤1e-10`/`assert_relative_eq`).

### Established Patterns
- Column-major `FdMatrix`; per-thread RNG seeding; `Result<T, FdarError>` (no panics); accumulation-order-sensitive numeric code.

### Integration Points
- `benchmarks/results/comparison.csv` + `hotspot_ranking.md` (Phase 113) are the before-baseline reference; Phase 116 re-runs the harness to show the post-optimization gain.

</code_context>

<specifics>
## Specific Ideas

- Distance-matrix symmetry is the most obvious numeric-identity-safe win (compute N(N−1)/2 pairs instead of N², mirror) — verify `self_distance_matrix_impl` isn't already exploiting it before assuming leverage.
- If pair-parallelism is added, ensure it's gated behind `parallel` and per-pair results are written to fixed [i][j] positions (no cross-pair reduction) so release-build output is bit-identical to sequential.
- Golden-value fixture should be a fixed seed/size elastic self-distance matrix asserted bit-exact (or ≤1e-10) before vs after.

</specifics>

<deferred>
## Deferred Ideas

- SVD/FPCA + covariance optimization — Phase 115 (incl. the SVD re-benchmark-at-higher-sample-count note).
- Any public-API change or new algorithm — out of scope (strictly non-breaking).
- Publishing the speedup narrative — Phase 116 report.

</deferred>
