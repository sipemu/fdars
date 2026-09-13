# Phase 102: Deterministic Multi-Restart Wrapper - Context

**Gathered:** 2026-09-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Wrap the Phase 101 single-init L-BFGS soft-DTW barycenter refiner in a deterministic
multi-restart loop (SDTW-02): run the local optimizer from multiple initializations
(pointwise mean + deterministically-seeded perturbations) and return the barycenter with
the lowest total soft-DTW cost across restarts, reproducibly for a fixed seed. This is an
internal/additive capability — the public `SoftDtwBarycenterConfig` builder + new
config-taking entry point exposing `n_restarts`/`seed` is Phase 103; the escape-from-local-
minimum quality evidence is Phase 104. This phase must not break the existing 4-arg
`soft_dtw_barycenter` signature.

</domain>

<decisions>
## Implementation Decisions

### Multi-Restart Design
- **Restart 0 is the unperturbed pointwise mean** (`init_barycenter_mean`) — guarantees the
  multi-restart result is never worse than Phase 101's single-init path.
- **Perturbation scheme:** additive Gaussian noise per coordinate, scaled to the per-column
  data spread (per-evaluation-point std of the input series), applied to the mean init.
  Each restart k (k ≥ 1) is seeded with `StdRng::seed_from_u64(seed + k)` (project
  convention) for reproducibility.
- **Default restart count:** `n_restarts = 10` as the internal function's default; the
  public config field + its default is finalized in Phase 103. The internal multi-restart
  function takes `n_restarts` (and `seed`) as parameters.
- **Winner selection:** the restart with the lowest total soft-DTW cost
  `Σ_rows soft_dtw_distance(barycenter, row)`; ties broken by lowest restart index for
  determinism.

### Claude's Discretion
- Internal function name/shape of the multi-restart wrapper and whether Phase 101's
  single-init descent is refactored into a reusable private helper (recommended) that both
  the 4-arg entry and the multi-restart wrapper call.
- Exact RNG distribution helper (`rand_distr::Normal`) usage, consistent with existing crate
  usage.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/metric/soft_dtw.rs` — Phase 101 delivered `eval_obj_and_grad`,
  `two_loop_lbfgs`, `backtrack_armijo`, and the single-init L-BFGS body inside
  `soft_dtw_barycenter`; `init_barycenter_mean` (pointwise mean); `soft_dtw_distance`
  (per-row cost for the total-cost objective). Refactor the single-init descent into a
  private helper the wrapper can call from each init.
- `SoftDtwBarycenterResult { barycenter, n_iter, converged }` (`#[non_exhaustive]`).

### Established Patterns
- Per-thread/per-restart RNG seeding convention: `StdRng::seed_from_u64(seed + k as u64)`
  (used across the crate for reproducibility).
- `rand` 0.8 + `rand_distr` 0.4 are existing dependencies (no new dep needed).
- Column-major `FdMatrix`; rows via `data.row(i)`.

### Integration Points
- The existing 4-arg `soft_dtw_barycenter(data, gamma, max_iter, tol)` stays byte-compatible
  (it can internally delegate to a single-restart call). The config-taking public entry that
  surfaces `n_restarts`/`seed` is Phase 103.

</code_context>

<specifics>
## Specific Ideas

- Determinism + correctness are the goal. Same seed + inputs → bit-identical barycenter.
- The kept barycenter's total cost must be ≤ any individual restart's cost (winner selection).
- Restart parallelism (feature-gated rayon) is explicitly OUT OF SCOPE this milestone — keep
  the loop sequential and deterministic.

</specifics>

<deferred>
## Deferred Ideas

- Public `SoftDtwBarycenterConfig` builder + config-taking entry point (SDTW-03) → Phase 103.
- Escape-from-local-minimum quality evidence vs MM (SDTW-04) → Phase 104.
- Parallel restarts (rayon) — out of scope this milestone.

</deferred>
