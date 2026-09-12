# Phase 101: In-Crate L-BFGS Barycenter Optimizer - Context

**Gathered:** 2026-09-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the fixed-step soft-DBA majorization-minimization (MM / diagonal-Newton)
descent inside `soft_dtw_barycenter` with a hand-rolled limited-memory L-BFGS
optimizer (two-loop recursion + backtracking Armijo line search) driven by the
exact soft-DTW gradient the crate already computes. Single-initialization path
only — the deterministic multi-restart wrapper (SDTW-02) is Phase 102, and the
`SoftDtwBarycenterConfig` builder + new entry point (SDTW-03) is Phase 103. This
phase delivers SDTW-01: the local optimizer, wired behind the existing 4-arg
`soft_dtw_barycenter(data, gamma, max_iter, tol)` signature (unchanged), compiling
and running in the default feature set and on `wasm32` with no `linalg`/`argmin`
gating and no new crate dependency.

</domain>

<decisions>
## Implementation Decisions

### L-BFGS Optimizer Design
- History size: **m = 10** memory (s, y) pairs — standard L-BFGS default; cheap,
  robust for this smooth objective. Two-loop recursion with the standard
  γ_k = (sᵀy)/(yᵀy) initial-Hessian scaling.
- Line search: **backtracking Armijo**, c1 = 1e-4, backtracking factor β = 0.5,
  initial step = 1.0, ~20-backtrack cap; on line-search failure take a safeguarded
  fallback step so the iteration cannot diverge.
- Stopping rule: **reuse the existing `max_iter` and `tol` parameters** — stop when
  the gradient ∞-norm falls below `tol` OR the relative change in total soft-DTW
  cost falls below `tol`, or when `max_iter` is reached.
- Behavior vs current MM path: **L-BFGS replaces the MM/diagonal-Newton descent as
  the single-init refiner.** Same objective and gradient; the 4-arg entry point is
  unchanged in this phase. (Config-taking API arrives in Phase 103.)

### Claude's Discretion
- Exact internal module layout / helper function names for the L-BFGS routine.
- Whether to keep a tiny MM safeguard internally if the line search stalls (user
  chose pure replacement, but a safeguarded fallback step to guarantee
  non-divergence is at Claude's discretion as an implementation detail).
- Memory-buffer representation for the (s, y) history ring buffer.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/metric/soft_dtw.rs` — home of `soft_dtw_barycenter` (line 453),
  the MM step `update_barycenter` (line 413, the code being replaced),
  `init_barycenter_mean` (line 428, the pointwise-mean init to reuse),
  `soft_dtw_accumulate_gradient_and_weight` (per-row gradient + diagonal weight),
  `soft_dtw_distance` (total-cost objective term), and `soft_dtw_backward`.
- `SoftDtwBarycenterResult { barycenter, n_iter, converged }` — existing result
  type; keep its shape (do not break it).

### Established Patterns
- Column-major `FdMatrix`; rows via `data.row(i)`.
- The **exact** soft-DTW gradient is correct as of Phase 78 / CORR-01 (v0.40.0) —
  the historical v0.39.0 zero-endpoint-seed bug is FIXED and cross-checked against
  a `corrected_oracle_gradient` + forward-mode `Dual` + finite differences in
  `soft_dtw.rs` tests. Do NOT touch the gradient/objective definitions (out of scope).
- Objective for L-BFGS = Σ_rows soft_dtw_distance(bary, row); gradient =
  Σ_rows accumulate_gradient(bary, row). Both already available in-module.
- No logging; `Result`/plain return; deterministic; pure-Rust numerics.

### Integration Points
- `soft_dtw_barycenter` is re-exported at crate root (`lib.rs`) and in
  `metric/mod.rs`; the 4-arg signature must stay byte-compatible for R + WASM
  bindings + 28 examples.

</code_context>

<specifics>
## Specific Ideas

- Milestone constraint (from REQUIREMENTS.md): no new crate dependency (hand-roll
  L-BFGS in-crate, as the reverse-mode tape was hand-rolled in v0.44.0); must
  compile + run in default features and on `wasm32` (no `linalg`/`argmin` gating).
- The two-loop recursion and Armijo line search are textbook (Nocedal & Wright);
  the only inputs are the objective value and gradient vector at a point.

</specifics>

<deferred>
## Deferred Ideas

- Deterministic multi-restart wrapper (SDTW-02) → Phase 102.
- `SoftDtwBarycenterConfig` builder + config-taking entry point (SDTW-03) → Phase 103.
- Quality-evidence test proving escape-from-local-minimum vs MM (SDTW-04) → Phase 104.
- Optional rayon-parallel restarts — explicitly out of scope this milestone.

</deferred>
