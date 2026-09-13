# Phase 104: Quality Evidence — Escapes-Local-Minimum & f64 Parity - Context

**Gathered:** 2026-09-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the SDTW-04 quality evidence: a committed, deterministic test (or tests) proving that
the multi-restart optimizer (Phase 102/103) reaches a strictly lower total soft-DTW barycenter
cost than the fixed-step MM baseline on a case where MM stalls in a local optimum, AND that the
legacy-wrapper path preserves f64 behavior — converging to a bounded barycenter within the data's
convex hull with no divergence. No production-code change beyond what's needed to expose the
comparison; this is an evidence/test phase. Release prep (version bump, CHANGELOG, gates) is
Phase 105.

</domain>

<decisions>
## Implementation Decisions

### Evidence Design
- **Evidence form:** committed deterministic test(s) in an inline `#[cfg(test)]` module (fixed
  seed) — primary, gate-checked. NO new runnable example (keeps the milestone's "28 examples"
  count untouched — avoids churn in the Phase 105 gate references).
- **MM-stall dataset:** a crafted bimodal / multi-cluster set of series where a single
  unperturbed-mean restart lands in a local optimum but at least one seeded perturbed restart
  escapes it. Assert `multi_restart_total_cost < single_init_total_cost` by a clear margin
  (not within f64 rounding). Reconstruct/obtain the MM (single-init) baseline via the existing
  4-arg `soft_dtw_barycenter` (n_restarts=1) and compare to `soft_dtw_barycenter_with_config`
  (or the multirestart driver) with n_restarts>1.
- **f64-parity / no-divergence evidence:** a test asserting the legacy `soft_dtw_barycenter`
  path returns a finite barycenter (no NaN/inf) with every coordinate within [min, max] of the
  input data (convex-hull containment), on representative data.
- **Test location:** inline `#[cfg(test)]` module in `fdars-core/src/metric/soft_dtw.rs`,
  consistent with Phases 101/102 (`barycenter_lbfgs_tests`, `barycenter_multirestart` tests).

### Claude's Discretion
- Exact construction of the MM-stall dataset (number of clusters, offsets, series count) — must
  be reproducible and genuinely exhibit a single-init stall that multi-restart beats.
- Test module name and how the cost is computed (reuse `soft_dtw_distance` summed over rows).
- Whether to add one or two tests (escape-evidence + parity) or fold parity into an existing
  test — as long as SDTW-04's two claims are both covered by committed deterministic tests.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/metric/soft_dtw.rs` — `soft_dtw_barycenter` (4-arg legacy = single-init via
  multirestart n_restarts=1), `soft_dtw_barycenter_with_config` + `SoftDtwBarycenterConfig`
  (Phase 103), `soft_dtw_barycenter_multirestart` (`pub(crate)`), `soft_dtw_distance`
  (per-row cost for the total-cost objective), `init_barycenter_mean`.
- Phase 101's `lbfgs_cost_le_mm` test already reconstructs an MM baseline inside a test — the
  fuller escape-from-local-minimum demonstration (a genuine MM stall) is THIS phase's job.

### Established Patterns
- Deterministic tests with fixed seeds; `StdRng::seed_from_u64(seed + k)` seeding.
- Total soft-DTW cost = Σ_rows soft_dtw_distance(barycenter, row, gamma).
- Column-major `FdMatrix`; `data.row(i)`.

### Integration Points
- Tests only touch the existing public/`pub(crate)` API — no new production symbols required.

</code_context>

<specifics>
## Specific Ideas

- The escape-evidence test must be deterministic (fixed seed) and pass under full `cargo test`
  and the CI determinism guardrail (RAYON_NUM_THREADS=1).
- Assert a clear cost margin so the test is robust to cross-platform f64 rounding (not a
  knife-edge strict inequality).
- Convex-hull containment: for each evaluation point j, barycenter[j] ∈ [min_i data[i][j],
  max_i data[i][j]] (or a small epsilon slack) — proves no divergence.

</specifics>

<deferred>
## Deferred Ideas

- Release prep: version bump 0.44.0→0.45.0, CHANGELOG entries, ROADMAP-TO-1.0.md SDTW-O1
  checkoff, full gate battery, cargo package (REL-01) → Phase 105.
- A runnable escape-from-local-minimum example — intentionally NOT added (keeps 28-example count).

</deferred>
