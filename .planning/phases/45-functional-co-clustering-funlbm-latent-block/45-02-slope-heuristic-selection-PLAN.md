---
phase: 45-functional-co-clustering-funlbm-latent-block
plan: 02
type: execute
wave: 2
depends_on: ["45-01"]
files_modified:
  - fdars-core/src/coclustering.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [CLUS-02-03]
estimate:
  tokens: 55000
  raw_tokens: 32000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "co_cluster_select(data, argvals, k_range, l_range, &CoClusterConfig) fits every (K,L) on the grid and selects one via the Birgé–Massart slope heuristic, returning the selected fit plus grid diagnostics (CLUS-02-03)."
    - "On well-separated synthetic (K=2,L=2) data the slope heuristic selects the true (or near-true) (K,L)."
    - "Grid sweep is deterministic under seed and reuses co_cluster (45-01) per (K,L)."
  artifacts:
    - fdars-core/src/coclustering.rs
  key_links:
    - "co_cluster_select -> co_cluster (per-(K,L) fit) via iter_maybe_parallel! grid sweep."
    - "lib.rs + prelude.rs re-export co_cluster_select + CoClusterSelectResult."
---

<objective>
Deliver slope-heuristic model selection over a (K,L) grid (CLUS-02-03): a `co_cluster_select` public fn
that fits every candidate (K,L) via `co_cluster` (45-01), collects (model_dimension, log_likelihood) pairs,
estimates the Birgé–Massart slope by OLS over the large-model region, and selects
`argmax_{(K,L)} [ LL − 2·|slope|·dim ]`. Returns the selected fit plus full grid diagnostics for inspection.

Purpose: automatic block-count selection is CLUS-02-03 and the required selector (ICL is diagnostic).
R baseline: `funHDDC` slope heuristic, matched by capability.

Output: `co_cluster_select` + `CoClusterSelectResult` added to `coclustering.rs`; crate-root + prelude
re-exports; inline correctness + edge-case tests.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/45-functional-co-clustering-funlbm-latent-block/45-CONTEXT.md
@.planning/phases/45-functional-co-clustering-funlbm-latent-block/45-RESEARCH.md
@.planning/phases/45-functional-co-clustering-funlbm-latent-block/45-01-funlbm-cem-core-PLAN.md
</context>

<critical_semantics>
This plan depends on 45-01. The model dimension for a fitted (K,L) is (per the RESOLVED columns=m semantics):
`dim(K,L) = p_kl = (K-1) + (L-1) + 2*K*L*eff_ncomp` where eff_ncomp is the effective FPC count from the
fitted result (read from `block_params[0].mean.len()`, which equals eff_ncomp). Columns partition the m
argument points (col proportion term (L-1)); this matches 45-01's ICL. Reuse `co_cluster` unchanged per
(K,L) — do NOT re-derive the fit.
</critical_semantics>

<artifacts_this_phase_produces>
New public symbols (this plan):
- `pub struct CoClusterSelectResult` — `{ best: CoClusterResult, best_k: usize, best_l: usize, grid_scores: Vec<(usize,usize,f64,usize,f64)> /* (K,L,log_lik,model_dim,penalised_score) */, slope_estimate: f64, penalty_rate: f64 }`, derives `Debug, Clone` + `#[non_exhaustive]` + serde cfg_attr.
- `pub fn co_cluster_select(data: &FdMatrix, argvals: &[f64], k_range: &[usize], l_range: &[usize], config: &CoClusterConfig) -> Result<CoClusterSelectResult, FdarError>`.
</artifacts_this_phase_produces>

<verified_api_grounding>
- `co_cluster(data, argvals, &CoClusterConfig) -> Result<CoClusterResult, FdarError>` (45-01). Set `config.n_row_blocks`/`n_col_blocks` per grid cell by cloning the config and overriding the two fields.
- `iter_maybe_parallel!($expr)` (parallel.rs) — under `parallel` returns a rayon ParallelIterator, else sequential; the closure captures `data`/`argvals`/`config` by shared ref (FdMatrix + CoClusterConfig are Send+Sync). Collect into `Vec<Result<...>>` then propagate the first Err.
- `FdarError::InvalidParameter { parameter, message }` for empty grids.
</verified_api_grounding>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end "select over a (K,L) grid" — one path only</name>
  <files>fdars-core/src/coclustering.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - fdars-core/src/coclustering.rs (45-01 output: co_cluster signature, CoClusterResult fields, eff_ncomp = block_params[0].mean.len())
    - fdars-core/src/parallel.rs:41-55 (iter_maybe_parallel! expansion + Send+Sync requirement)
    - fdars-core/src/gmm/em.rs:100-115 (iter_maybe_parallel! usage pattern collecting results)
    - fdars-core/src/lib.rs:465-479 (re-export block), fdars-core/src/prelude.rs (tail)
    - This PLAN's <critical_semantics> (model dimension formula)
  </read_first>
  <action>
Define `CoClusterSelectResult` with the fields in <artifacts_this_phase_produces> (derives Debug, Clone,
`#[non_exhaustive]`, serde cfg_attr; `#[must_use]` on the producer). Implement `co_cluster_select`:
validate `k_range` and `l_range` are non-empty (else `FdarError::InvalidParameter`); build the grid of
(K,L) pairs (`k_range × l_range`); sweep with `iter_maybe_parallel!` over the grid, cloning `config` per
cell and overriding `n_row_blocks=K`, `n_col_blocks=L`, calling `co_cluster`; collect
`Vec<Result<CoClusterResult, FdarError>>` and propagate the first Err (skip cells that would violate K>n or
L>m only if you choose to pre-filter — otherwise let co_cluster's error propagate; document the choice).
For each successful fit compute `model_dim = (K-1)+(L-1)+2*K*L*eff_ncomp` where `eff_ncomp =
result.block_params[0].mean.len()`, and record `(K, L, log_likelihood, model_dim)`.

Tracer selection (thin but real): the full slope estimation is Task 2 — for the tracer, wire the full
pipeline but if the grid has fewer than 4 points OR the slope estimation is not yet in place, fall back to
`argmax log_likelihood` (this fallback is ALSO a documented edge case in Task 2, so it is production code,
not a stub). Populate `best`, `best_k`, `best_l`, `grid_scores` (with penalised_score = LL for the fallback),
`slope_estimate = 0.0`, `penalty_rate = 0.0` for the tracer.

Register: add `co_cluster_select` and `CoClusterSelectResult` to the crate-root `pub use coclustering::{...}`
re-export and add `CoClusterSelectResult` to the prelude re-export.

Add one inline smoke test `test_co_cluster_select_smoke`: on a small (n=8,m=6) uniform-grid FdMatrix, call
`co_cluster_select` with `k_range=&[2,3]`, `l_range=&[2]`, assert it returns a CoClusterSelectResult whose
`grid_scores.len()==2` and `best.row_labels.len()==8` and `best.col_labels.len()==6`.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib coclustering::tests::test_co_cluster_select_smoke 2>&1 | tail -20</automated>
  </verify>
  <done>co_cluster_select compiles, is re-exported, sweeps the grid via co_cluster, and the smoke test proves it returns a populated CoClusterSelectResult with one grid_scores entry per (K,L). Committed.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Birgé–Massart slope estimation + selection + edge cases</name>
  <files>fdars-core/src/coclustering.rs</files>
  <read_first>
    - fdars-core/src/coclustering.rs (Task 1 grid sweep; make_block_data oracle from 45-01 test module)
    - fdars-core/src/test_helpers.rs:24-93 (adjusted_rand_index / uniform_grid for oracle)
    - This PLAN's <critical_semantics>
    - 45-RESEARCH.md §4.2 (slope recipe) and §4.3 (edge cases)
  </read_first>
  <behavior>
    - Test: on well-separated synthetic (K=2,L=2) block data (reuse 45-01's make_block_data), co_cluster_select over k_range=[2,3,4], l_range=[2,3] selects best_k==2 && best_l==2 (or ARI(best.row_labels, true)>0.8 as a tolerance).
    - Test: single-cell grid (k_range=[2], l_range=[2]) returns that cell directly (no slope estimation), slope_estimate handled without panic.
    - Test: empty k_range or l_range → FdarError::InvalidParameter (no panic).
    - Test: determinism — same seed → identical best_k/best_l/grid_scores.
  </behavior>
  <action>
Replace the tracer fallback with the full Birgé–Massart slope heuristic (45-RESEARCH.md §4.2):
(1) collect `(dim, ll, k, l)` for all successful fits; (2) sort by dim descending, take the top 50%
(at least 4 points when available) as the large-model region; (3) OLS slope of ll on dim over that subset:
`slope = Σ(dim_i−d̄)(ll_i−l̄) / Σ(dim_i−d̄)²`; (4) `penalty_rate = 2.0 * slope.abs()`;
(5) select `(K*,L*) = argmax_{grid} [ ll − penalty_rate·dim ]`; populate `grid_scores` with the penalised
score per cell, `slope_estimate`, `penalty_rate`, and re-fit or reuse the stored best `CoClusterResult` for
that (K*,L*). Handle the documented edge cases WITHOUT erroring (45-RESEARCH.md §4.3): grid with <4 points
or all-equal dims in the large-model subset (OLS denominator ≈ 0) → fall back to argmax LL; single-cell grid
→ return that cell (slope_estimate=0, penalty_rate=0); penalty_rate ≤ 0 → fall back to argmax LL. Keep
`grid_scores` fully populated in every branch so the caller can inspect all candidates (Pitfall 4 —
boundary selections are returned, never errored). Document the heuristic + its data-separation caveat in
rustdoc; document divergence from funHDDC (OLS-over-top-50% calibration).

Add inline tests: `test_slope_heuristic_selects_correct_kl` (make_block_data, grid [2,3,4]×[2,3],
best_k==2 && best_l==2 or ARI tolerance), `test_select_single_cell` (1×1 grid returns that cell),
`test_select_empty_range_errors` (empty k_range and empty l_range each → InvalidParameter),
`test_select_determinism` (same seed → identical best_k/best_l/grid_scores).
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib coclustering 2>&1 | tail -25</automated>
  </verify>
  <done>Slope heuristic selects the true (K,L) on well-separated data; single-cell + empty-grid + determinism edge cases pass; grid_scores populated in every branch. Full coclustering module green. Committed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Pure in-process numeric library: no I/O, network, untrusted input, auth. Attack surface: none. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-45-06 | Tampering | Empty k_range/l_range → empty grid, invalid arithmetic | low | mitigate | Validate non-empty ranges before the sweep; return FdarError::InvalidParameter. |
| T-45-07 | Tampering | OLS denominator ≈ 0 (all-equal dims) → div-by-zero / NaN slope | low | mitigate | Guard the denominator; fall back to argmax LL when near-zero (§4.3 edge case). |
| T-45-08 | Denial of Service | Boundary-model selection on poorly separated data | low | accept | Return grid_scores for inspection; slope heuristic is documented as data-separation dependent — never error on boundary picks. |

No package-manager installs in this phase — no supply-chain threat.
</threat_model>

<verification>
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib coclustering` — all inline tests green.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
- `cargo fmt` run before each commit.
</verification>

<success_criteria>
- CLUS-02-03: `co_cluster_select` selects the block count via the slope-heuristic criterion over a user (K,L) grid, returning the selected fit + grid diagnostics; deterministic under seed; edge cases (single cell, empty range, flat slope) handled without panic.
- Additive/non-breaking: no existing public signature changed; crate-root + prelude re-exports added.
</success_criteria>

<output>
Create `.planning/phases/45-functional-co-clustering-funlbm-latent-block/45-02-SUMMARY.md` when done.
</output>
