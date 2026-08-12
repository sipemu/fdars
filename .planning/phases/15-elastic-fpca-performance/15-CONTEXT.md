# Phase 15: Elastic-FPCA Performance - Context

**Gathered:** 2026-08-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver PERF-04 (backlog PERF-PAR-ELFPCA): parallelize the three per-curve loops on the elastic-FPCA critical path under the `parallel` feature, producing output numerically equivalent to the sequential path. Additive/non-breaking, isolated to `fdars-core/src/elastic_fpca.rs` (disjoint from Phase 14's `alignment/` work).

Target loop sites (verified against live source — line numbers current as of this milestone):
- `shooting_vectors_from_psis` — `elastic_fpca.rs:701` `for i in 0..n` (HEAVY body: `inv_exp_map_sphere(mu_psi, &psis[i], time)` per curve, writes disjoint row i).
- `build_augmented_srsfs` — `elastic_fpca.rs:720` `for i in 0..n` (MEDIUM body: copy SRSF row + one `signum·sqrt`, writes disjoint row i).
- `svd_scores_and_eigenvalues` — `elastic_fpca.rs:764` `for i in 0..n` (VERY LIGHT body: single multiply `scores[(i,k)] = u[(i,k)] * sv`, nested inside `for k in 0..ncomp`).

`elastic_fpca.rs` currently uses zero `iter_maybe_parallel!` (all-sequential). Out of scope: the other per-curve loops in the file (`:734/:740/:800/:829/:878/:921/:964`), any algorithmic change, and the `fdata_to_pc_1d` (non-elastic) FPCA path.
</domain>

<decisions>
## Implementation Decisions

### Area 1 — Parallelization scope & pattern
- **`:701` and `:720`:** parallelize via `iter_maybe_parallel!(0..n).map(|i| …).collect::<Vec<_>>()` producing per-curve row data, then a sequential assignment pass into the column-major `FdMatrix` (mirrors the collect-then-assemble pattern already used by `align_to_target` in `alignment/set.rs`). Do NOT attempt parallel writes directly into the column-major buffer — collect owned per-row results first, assign sequentially.
- **`:764`:** parallelize the `for i in 0..n` loop via `iter_maybe_parallel!` **guarded by an N ≥ 50 threshold** — below the threshold, run the existing sequential path (the body is a single multiply; parallel dispatch only pays back at N ≥ 50 per the audit's streaming-sentinel payback point). This satisfies PERF-04's literal "all three loops" requirement without a small-N regression.
- **Threshold:** `N ≥ 50` (a named constant, documented in a comment).

### Area 2 — Equivalence testing & conventions
- **Tolerance:** these loops are pure disjoint per-index writes with NO cross-iteration reduction, so parallel and sequential produce **bit-identical** output (no floating-point summation reordering). Assert exact equality where the loop is a pure write; fall back to a tight `1e-12` only if a reduction is unavoidable.
- **Entry points tested:** inline `#[cfg(test)]` tests on both `vert_fpca` and `joint_fpca`, comparing the (feature-gated) parallel result's scores + eigenvalues against a sequentially-computed reference within tolerance. Tests run under the `linalg` feature (elastic FPCA needs SVD).
- **Conventions:** use the existing `iter_maybe_parallel!` macro from `parallel.rs`; additive/non-breaking (no public signature changes — `shooting_vectors_from_psis`/`build_augmented_srsfs` are `pub(crate)`, `svd_scores_and_eigenvalues` is private); no new dependencies; sequential when the `parallel` feature is off (macro handles this).

### Claude's Discretion
- Exact naming of the threshold constant, the collect-closure body factoring, whether `:764` is parallelized over `i` for a fixed `k` or restructured, and test fixture sizes (N large enough to exercise the ≥50 path) — all at Claude's discretion, guided by codebase conventions and the existing `iter_maybe_parallel!` call sites.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `iter_maybe_parallel!` (and 4 sibling macros) in `fdars-core/src/parallel.rs` — the feature-gated parallelism primitive; `elastic_fpca.rs` does not use it yet.
- `alignment/set.rs::align_to_target` — canonical "parallel-collect into `Vec`, then sequential row-assign into `FdMatrix`" pattern; the model for `:701`/`:720`.
- `inv_exp_map_sphere(mu_psi, &psis[i], time)` — the heavy per-curve computation in `:701` (returns a `Vec<f64>` row).

### Established Patterns
- Column-major `FdMatrix` (element (i,j) at index i + j*nrows): parallel writes into the shared buffer are unsafe/awkward — collect owned per-row `Vec`s first, then assign sequentially.
- Per-thread determinism is a non-issue here: no RNG in any of the three loops.

### Integration Points
- All three functions are internal (`pub(crate)`/private) to the elastic-FPCA path; `vert_fpca` / `joint_fpca` are the public entry points that call them. No re-export changes needed.
</code_context>

<specifics>
## Specific Ideas

- Backlog anchor: PERF-PAR-ELFPCA (rank 17, P2/M) in `.planning/research/BACKLOG.md` — exact line numbers, static-independence argument (SC2), and the ~4–5× projection at N≥50 from the Phase-5 thread-scaling table.
- Audit caveat: elastic cells were flagged LOW-CONFIDENCE under an unpinned governor — demonstrate **feasibility + numerical equivalence**, not a pinned speedup number.
</specifics>

<deferred>
## Deferred Ideas

- Parallelizing the other per-curve loops in `elastic_fpca.rs` (`:734/:740/:800/:829/:878/:921/:964`) — out of PERF-04 scope.
- Truncated/thin SVD in the elastic path (PERF-FPCA-TRUNCSVD) — separate backlog item, not this phase.
</deferred>
