---
phase: 45-functional-co-clustering-funlbm-latent-block
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/coclustering.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [CLUS-02-01, CLUS-02-02]
estimate:
  tokens: 95000
  raw_tokens: 55000
  tasks: 3
  confidence: low
must_haves:
  truths:
    - "co_cluster(data, argvals, &CoClusterConfig) fits a funLBM via block-wise-Gaussian CEM on FPC scores and returns row_labels (len n) + col_labels (len m) simultaneously (CLUS-02-01)."
    - "col_labels ranges over the m ARGUMENT POINTS (col_labels.len() == m), NOT the ncomp FPC components — the RESOLVED funLBM semantics."
    - "The classification log-likelihood is non-decreasing across CEM iterations and identical calls with the same seed produce identical labels/log-lik/ICL (CLUS-02-01)."
    - "CoClusterResult exposes row_labels, col_labels, block_params, row_props, col_props, log_likelihood, and a finite icl (CLUS-02-02)."
    - "Error paths (K > n, L > m, ncomp < 1, ncomp > min(n,m), data/argvals mismatch) return FdarError without panicking (CLUS-02-01)."
  artifacts:
    - fdars-core/src/coclustering.rs
  key_links:
    - "coclustering.rs -> fdata_to_pc_1d (global FPCA loadings + weights + mean) for the block-score projection."
    - "coclustering.rs -> kmeans_fd for seeded row-cluster initialization."
    - "lib.rs pub mod coclustering + crate-root re-exports; prelude.rs re-exports CoClusterConfig/CoClusterResult."
---

<objective>
Deliver the funLBM co-clustering foundation (CLUS-02-01, CLUS-02-02): a new `fdars-core/src/coclustering.rs`
module implementing a functional latent block model via Classification EM (CEM) on FPC scores, where
row-clusters partition the n curves and column-clusters partition the **m argument points**. Fitting is
exposed through a single `Result`-returning public fn `co_cluster` driven by a builder-style
`CoClusterConfig`; the returned `CoClusterResult` carries hard row/col labels, per-block Gaussian
parameters, mixing proportions, the converged classification log-likelihood, and the ICL criterion.

Purpose: fdars' existing `clustering.rs`/`gmm/` cluster curves only. This adds the row×column
co-clustering paradigm (funLBM, R baseline `funLBM` 2.3.1), matched by capability. This plan is the
foundation the model-selection plan (45-02) builds on.

Output: `coclustering.rs` with `CoClusterConfig`, `CoClusterResult`, `BlockParams`, `co_cluster`;
crate-root + prelude re-exports; full inline `#[cfg(test)]` oracle + error-path tests.
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
@.planning/phases/45-functional-co-clustering-funlbm-latent-block/45-VALIDATION.md
</context>

<critical_semantics>
## RESOLVED column-cluster semantics (OVERRIDES 45-RESEARCH.md §1.1 / A1)

45-RESEARCH.md assumed `col_labels` ranges over the `ncomp` FPC components. **That was REJECTED.**
The LOCKED CONTEXT.md decision (and CLUS-02-01) is: **column-clusters range over the m ARGUMENT
POINTS.** `col_labels` has length **m**. The m evaluation points are partitioned into L column-clusters
(need not be contiguous). This is true funLBM. Everything else in 45-RESEARCH.md (CEM alternating hard
row/col assignment, non-decreasing classification log-lik, diagonal block covariance, log-sum-exp guard,
empty-cluster guard, ICL, seeded restarts, reuse of `fdata_to_pc_1d`/`kmeans_fd`/`data_scaled_reg`
patterns) still applies — but adapted to **columns = argument points**.

### Global-FPCA reuse reconciled with columns-over-argument-points (LOCKED)

Compute ONE global FPCA: `fdata_to_pc_1d(data, ncomp, argvals)` → `FpcaResult { rotation (m×ncomp),
mean (len m), weights (len m), scores (n×ncomp) }`. The **block score** of curve i for column-cluster l
on FPC component k is the projection of `Y_i` **restricted to column-cluster l's argument points** onto
the global loadings restricted to those same points:

    block_score[i][l][k] = Σ_{j : col_labels[j] == l}  weights[j] · (data[(i,j)] - mean[j]) · rotation[(j,k)]

This restricts the standard weighted FPC inner product to a column-block's argument-point subset, keeping
columns = argument points while reusing a single global FPCA (no per-iteration FPCA recompute). The
per-(row-cluster k, column-cluster l) block Gaussian is a diagonal `ncomp`-dim Gaussian fit on
`{ block_score[i][l][·] : row_labels[i] == k }`.

So there are TWO "component" axes: the L column-clusters (over m argument points) and the ncomp FPC
components INSIDE each block score. Do not conflate them. The block Gaussian dimension is `ncomp`
(the same for every block); the column-cluster only changes WHICH argument points enter the projection.
</critical_semantics>

<artifacts_this_phase_produces>
New public symbols (this plan):
- `pub struct CoClusterConfig` — builder-style config (fields: `n_row_blocks`, `n_col_blocks`, `ncomp`, `max_iter`, `tol`, `n_init`, `seed`) with `impl Default`.
- `pub struct CoClusterResult` — `{ row_labels: Vec<usize> (len n), col_labels: Vec<usize> (len m), n_row_blocks, n_col_blocks, block_params: Vec<BlockParams> (len K*L, indexed k*L+l), row_props: Vec<f64> (len K), col_props: Vec<f64> (len L), log_likelihood: f64, icl: f64, iterations: usize, converged: bool }`.
- `pub struct BlockParams` — `{ mean: Vec<f64> (len ncomp), variance: Vec<f64> (len ncomp) }` (diagonal block Gaussian on block scores).
- `pub fn co_cluster(data: &FdMatrix, argvals: &[f64], config: &CoClusterConfig) -> Result<CoClusterResult, FdarError>`.

(45-02 adds `co_cluster_select` + `CoClusterSelectResult`.)
</artifacts_this_phase_produces>

<verified_api_grounding>
All signatures confirmed by direct file read this planning session:
- `fdata_to_pc_1d(data: &FdMatrix, ncomp: usize, argvals: &[f64]) -> Result<FpcaResult, FdarError>` (regression.rs:287). `FpcaResult { singular_values, rotation: FdMatrix (m×ncomp), scores: FdMatrix (n×ncomp), mean: Vec<f64> (len m), centered, weights: Vec<f64> (len m) }` (regression.rs:22-38). Note `fdata_to_pc_1d` internally clips `ncomp = ncomp.min(n).min(m)` — read the ACTUAL returned `scores.ncols()`/`rotation.ncols()` as the effective ncomp, do not assume the requested value.
- `kmeans_fd(data: &FdMatrix, argvals: &[f64], k, max_iter, tol, seed) -> Result<KmeansResult, FdarError>`; `KmeansResult.cluster: Vec<usize>` len n (clustering.rs:545).
- `FdMatrix`: `nrows()`, `ncols()`, `shape() -> (usize,usize)`, `column(j) -> &[f64]` (contiguous, len n), `zeros(nrows,ncols)`, `Index<(usize,usize)>` + `IndexMut` (matrix.rs). Column-major: `[(i,j)]` at `i + j*nrows`; `column(j)` is the contiguous slice for fixed j (use it in E-col inner loops over i).
- `FdarError::InvalidParameter { parameter: &'static str, message: String }`, `InvalidDimension { parameter: &'static str, expected: String, actual: String }`, `ComputationFailed { operation: &'static str, detail: String }` (error.rs:8-25).
- `iter_maybe_parallel!` macro (parallel.rs) — NOT needed this plan (single fit); used in 45-02.
- Config builder pattern to mirror: `GmmClusterConfig` + `impl Default` (gmm/cluster.rs:49-86).
- `data_scaled_reg` regularization pattern (gmm/covariance.rs:20-45) — reimplement inline over block scores (`reg = 1e-6 * mean_k Var(block_score[·][·][k])`).
- lib.rs registration: `pub mod` list is alphabetical (matrix.rs read shows clustering, clustering_advanced, concurrent_regression …) — insert `pub mod coclustering;` between `clustering_advanced;` and `concurrent_regression;`. Crate-root re-export block near clustering re-exports (lib.rs:469). prelude.rs re-export at tail.
</verified_api_grounding>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end "co_cluster fits and returns labels" — one path only</name>
  <files>fdars-core/src/coclustering.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - fdars-core/src/regression.rs:22-38 (FpcaResult fields) and :287-320 (fdata_to_pc_1d validation + ncomp clipping)
    - fdars-core/src/clustering.rs:545-607 (kmeans_fd signature, K>n guard to mirror)
    - fdars-core/src/gmm/cluster.rs:49-86 (GmmClusterConfig + impl Default builder pattern to mirror)
    - fdars-core/src/matrix.rs:84-160,279,328,422-455 (FdMatrix zeros/nrows/ncols/column/Index)
    - fdars-core/src/error.rs:1-52 (FdarError variants + field types)
    - fdars-core/src/lib.rs:80-110 (pub mod ordering) and :465-479 (clustering re-export block)
    - fdars-core/src/prelude.rs (tail re-export block)
    - This PLAN's <critical_semantics> block (columns = m argument points; block-score projection formula)
  </read_first>
  <action>
Create `fdars-core/src/coclustering.rs` with the module doc comment (//!) explaining funLBM co-clustering,
that col_labels range over the m argument points, that ONE global FPCA is reused via a block-score
projection, and documenting divergences from R funLBM (global vs per-block FPCA; CEM vs SEM-Gibbs;
diagonal vs full block covariance). Define `CoClusterConfig` with fields `n_row_blocks: usize`,
`n_col_blocks: usize`, `ncomp: usize`, `max_iter: usize`, `tol: f64`, `n_init: usize`, `seed: u64`, all
derives (`Debug, Clone, PartialEq`), `#[cfg_attr(feature="serde", derive(Serialize,Deserialize))]`,
`#[non_exhaustive]`, and `impl Default` (n_row_blocks 2, n_col_blocks 2, ncomp 5, max_iter 200, tol 1e-6,
n_init 3, seed 42) mirroring GmmClusterConfig. Define `BlockParams { mean: Vec<f64>, variance: Vec<f64> }`
and `CoClusterResult` with the exact fields in <artifacts_this_phase_produces>, all derives +
`#[non_exhaustive]` + serde cfg_attr + `#[must_use]` on the result struct's expensive producer.

Wire the single tracer path in `co_cluster`: validate inputs, run ONE global FPCA, run ONE CEM fit
(n_init = 1 for the tracer path is acceptable — the multi-restart loop is Task 2), and return a populated
CoClusterResult. Validation up front (return FdarError, never panic): reject `ncomp < 1`
(InvalidParameter), `config.n_row_blocks > n` (InvalidParameter, mirror kmeans_fd K>n message),
`config.n_col_blocks > m` (InvalidParameter — L>m, the RESOLVED columns=argument-points guard); let the
data/argvals length mismatch and empty-matrix errors propagate from `fdata_to_pc_1d`. After
`fdata_to_pc_1d`, read `let eff_ncomp = fpca.scores.ncols();` and use eff_ncomp everywhere (it may be < requested).

Implement the block-score projection helper `fn block_scores(data, fpca_rotation, mean, weights, col_labels, l, eff_ncomp) -> Vec<f64>` (or a flat n×L×eff_ncomp buffer builder) computing, per <critical_semantics>:
`block_score[i][l][k] = Σ_{j: col_labels[j]==l} weights[j] * (data[(i,j)] - mean[j]) * rotation[(j,k)]`.
Iterate j via the argument-point index (0..m), accumulate into the (i,l,k) block-score buffer only for the
j's whose col_labels[j]==l. Store block scores as a flat `Vec<f64>` indexed `(i*L + l)*eff_ncomp + k`.

For the tracer CEM: seed row_labels from `kmeans_fd(data, argvals, K, 100, 1e-4, seed)?.cluster`; seed
col_labels by an inline k-means++-style pass over the m argument-point PROFILES (each argument point j has
an n-dim profile = column(j) of data; run L-center k-means++ on those m profiles in R^n, ~10 assign-update
iters, seeded `StdRng::seed_from_u64(seed.wrapping_add(1))`), producing col_labels of length m. Then run the
CEM loop for `config.max_iter` iterations: (E-row) recompute block scores under current col_labels, then for
each curve i pick argmax_k [ ln row_props[k] + Σ_l Σ_k' log_gaussian_1d(block_score[i][l][k'], block_mean[k][l][k'], block_var[k][l][k']) ];
(E-col) for each argument point j pick argmax_l of the classification log-density gain from assigning j to l
(reproject / accumulate per <critical_semantics>); (M-step) recompute row_props, col_props, and per-block
diagonal mean+variance from the block scores; compute the classification log-likelihood; break when
`|ll - prev_ll| < config.tol` after iter 0 (set converged=true) or at max_iter. Add `reg` (inline
data_scaled_reg over block scores) to every block variance to avoid variance collapse. Guard empty clusters
(row_props[k] or col_props[l] near 0 → skip in density accumulation, `continue` on `< 1e-15`). Use the
max-shifted log-sum-exp form for any normalization. Compute ICL at the end (Task 2 finalizes the formula;
tracer may store `icl = ll` as a placeholder that Task 2 replaces — DO NOT ship the placeholder, Task 2 is
in the same plan/file).

Register the module: add `pub mod coclustering;` to lib.rs (alphabetical position between
`clustering_advanced;` and `concurrent_regression;`), add a crate-root
`pub use coclustering::{co_cluster, CoClusterConfig, CoClusterResult, BlockParams};` re-export near the
clustering re-export block, and add `pub use crate::coclustering::{CoClusterConfig, CoClusterResult};` to
prelude.rs. `log_gaussian_1d(x, mu, var) = -0.5 * ((x-mu).powi(2)/var + var.ln() + (2.0*PI).ln())`.

Add ONE inline smoke test `test_co_cluster_smoke` that builds a tiny (n=8, m=6) FdMatrix with a uniform
grid via `crate::test_helpers::uniform_grid`, calls `co_cluster` with K=2,L=2,ncomp=3, and asserts
`row_labels.len()==8 && col_labels.len()==6 && block_params.len()==4 && log_likelihood.is_finite()`.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib coclustering::tests::test_co_cluster_smoke 2>&1 | tail -20</automated>
  </verify>
  <done>co_cluster compiles, is re-exported at crate root + prelude, and the smoke test proves the single fit path returns row_labels(len n)+col_labels(len m)+K*L block_params with finite log-likelihood. Committed.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: CEM correctness — multi-restart, monotone log-lik, ICL, determinism</name>
  <files>fdars-core/src/coclustering.rs</files>
  <read_first>
    - fdars-core/src/gmm/cluster.rs:11-33 (run_multiple_inits: base_seed.wrapping_add(init*1000) restart pattern, keep best by log-lik)
    - fdars-core/src/gmm/em.rs:17-40 (diagonal Gaussian log-density), :55-83 (log-sum-exp + zero-weight skip), :214-226 (compute_icl pattern)
    - fdars-core/src/gmm/covariance.rs:20-45 (data_scaled_reg), :162-186 (empty-cluster fallback)
    - fdars-core/src/test_helpers.rs:24-93 (adjusted_rand_index — test oracle), uniform_grid
    - This PLAN's <critical_semantics> (ICL column term is over the L argument-point clusters)
  </read_first>
  <behavior>
    - Test: classification log-likelihood is non-decreasing across CEM iterations (record the per-iter LL vector; assert each step >= previous minus a tiny epsilon).
    - Test: on synthetic (K=2,L=2) block-structured data, adjusted_rand_index(true_row_labels, row_labels) > 0.8 AND adjusted_rand_index(true_col_labels, col_labels) > 0.8.
    - Test: two co_cluster calls with the same seed produce byte-identical row_labels, col_labels, log_likelihood, and icl.
    - Test: icl is finite (not NaN/±∞) on well-conditioned data.
  </behavior>
  <action>
Wrap the single-fit CEM (Task 1) in an `n_init`-restart loop: for `init in 0..config.n_init`, fit with
seed `config.seed.wrapping_add(init as u64 * 1000)`, keep the result with the highest `log_likelihood`
(mirror gmm run_multiple_inits). Finalize the ICL:
`p_kl = (K-1) + (L-1) + 2*K*L*eff_ncomp` and
`icl = log_likelihood - 0.5 * (p_kl as f64) * ((n as f64).ln() + (m as f64).ln())`
— the RESOLVED symmetric penalty with the column dimension = m (the L column-clusters partition the m
argument points, so the column proportion term is (L-1) and the column-dimension log is ln m per
Govaert-Nadif). Document the ICL formula + this columns=m choice in the rustdoc for CoClusterResult.icl.

Ensure the M-step block variance always includes `reg` (inline data_scaled_reg over the block-score buffer)
so variance never collapses; ensure empty row/col clusters are skipped with the `< 1e-15` proportion guard
so no `ln(0)` reaches the accumulator. Confirm the classification LL is computed identically inside the loop
and at the end so the non-decreasing property holds within a single fit.

Add an inline test-only oracle `make_block_data(n, m, ncomp, seed) -> (FdMatrix, Vec<f64>, Vec<usize>, Vec<usize>)`
generating n curves at m argument points with a known (K=2,L=2) block structure: partition the m argument
points into two contiguous halves (true_col_labels), give row-group-0 curves a large positive offset on the
first argument-point half and row-group-1 a large negative offset there (weak signal on the second half), add
small Normal noise via `rand_distr::Normal` + `StdRng::seed_from_u64(seed)`. Return (data, uniform_grid(m),
true_row_labels, true_col_labels).

Add inline tests: `test_classification_ll_nondecreasing` (expose the per-iter LL vector via a thin internal
`#[cfg(test)]` helper or return it from an internal fit fn), `test_coclustering_recovers_block_structure`
(ARI > 0.8 on both axes using make_block_data), `test_determinism_under_seed` (two calls identical),
`test_icl_is_finite`.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib coclustering::tests::test_classification_ll_nondecreasing coclustering::tests::test_coclustering_recovers_block_structure coclustering::tests::test_determinism_under_seed coclustering::tests::test_icl_is_finite 2>&1 | tail -25</automated>
  </verify>
  <done>All four correctness tests pass: log-lik monotone, ARI > 0.8 on both row and col axes, same-seed determinism, finite ICL. Multi-restart keeps best by log-lik. ICL uses the symmetric (ln n + ln m) penalty. Committed.</done>
</task>

<task type="auto">
  <name>Task 3: Error paths + result-surface accessors (CLUS-02-02)</name>
  <files>fdars-core/src/coclustering.rs</files>
  <read_first>
    - fdars-core/src/clustering.rs:563-581 (K>n and argvals-mismatch guard messages to mirror)
    - fdars-core/src/regression.rs:293-319 (fdata_to_pc_1d error variants that propagate)
    - fdars-core/src/error.rs:1-52 (FdarError matching in tests)
  </read_first>
  <action>
Ensure `co_cluster` returns (never panics) `FdarError::InvalidParameter` for: `config.ncomp < 1`;
`config.n_row_blocks > n` (K>n); `config.n_col_blocks > m` (L>m — the RESOLVED columns=argument-points
guard); and let the data/argvals length mismatch propagate as the `fdata_to_pc_1d`
`FdarError::InvalidDimension`. Verify the CoClusterResult surface is fully populated and documented:
row_labels (len n), col_labels (len m), block_params (len K*L, indexed k*L+l), row_props (len K, sums to 1
over non-empty), col_props (len L), log_likelihood, icl, iterations, converged. Add per-field rustdoc noting
col_labels.len()==m and block_params indexing.

Add inline tests matching on the exact FdarError variant (use `matches!`):
`test_error_k_exceeds_n` (K=99 on n=8 → InvalidParameter),
`test_error_l_exceeds_m` (L=99 on m=6 → InvalidParameter),
`test_error_zero_ncomp` (ncomp=0 → InvalidParameter),
`test_error_argvals_mismatch` (argvals.len() != m → InvalidDimension propagated),
`test_result_surface_populated` (fit a small model; assert col_labels.len()==m, row_labels.len()==n,
block_params.len()==K*L, row_props.len()==K, col_props.len()==L, and that every block_params[i].mean.len()
and .variance.len() equal eff_ncomp).
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib coclustering 2>&1 | tail -25</automated>
  </verify>
  <done>All error-path tests return the correct FdarError variant with no panic; result-surface test confirms col_labels.len()==m and every accessor is populated with correct lengths. Full coclustering test module green. Committed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Pure in-process numeric library: no I/O, network, untrusted input, auth, or session state. Attack surface: none. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-45-01 | Tampering | Degenerate params: empty row/col cluster during CEM (n_k=0 or d_l=0) | low | mitigate | Skip empty clusters via `< 1e-15` proportion guard; keep previous params; never `ln(0)` into the accumulator. |
| T-45-02 | Tampering | Block variance collapse (σ²→0 in small block) | low | mitigate | Add inline `reg = 1e-6 * mean_k Var(block_score)` to every block variance (data_scaled_reg pattern). |
| T-45-03 | Denial of Service | Log-sum-exp underflow across n terms in the E-col accumulation | low | mitigate | Accumulate in log-density space; never exponentiate before summing; max-shifted normalization. |
| T-45-04 | Tampering | Integer overflow / invalid geometry: K>n, L>m, ncomp<1, ncomp>min(n,m) | low | mitigate | Validate before any arithmetic; return FdarError::InvalidParameter/InvalidDimension (no panic). |
| T-45-05 | Repudiation | Label switching across restarts (non-reproducible labels) | low | accept | Compare fits by log-lik (max is best); tests use adjusted_rand_index, not raw label equality — inherent to the model, not a defect. |

No package-manager installs in this phase (no new crate dependency) — no supply-chain (T-*-SC) threat.
</threat_model>

<verification>
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib coclustering` — all inline tests green.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean (CI lints test/bench code; a plain `-p ... -D warnings` false-greens).
- `cargo fmt` run before each commit (MEMORY.md: --no-verify commits otherwise leave fmt drift).
- col_labels.len() == m verified by test (RESOLVED semantics), NOT ncomp.
</verification>

<success_criteria>
- CLUS-02-01: `co_cluster` fits a funLBM via block-wise-Gaussian CEM on FPC scores, simultaneously assigning n curves to row-clusters and m argument points to column-clusters, given (K,L); log-lik non-decreasing; deterministic under seed; error paths return FdarError.
- CLUS-02-02: `CoClusterResult` exposes row_labels, col_labels, per-block parameters, converged log_likelihood, and a finite ICL.
- Additive/non-breaking: no existing public signature changed; crate-root + prelude re-exports added.
</success_criteria>

<output>
Create `.planning/phases/45-functional-co-clustering-funlbm-latent-block/45-01-SUMMARY.md` when done.
</output>
