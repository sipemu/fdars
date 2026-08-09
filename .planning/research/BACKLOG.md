# fdars Prioritized Backlog

**Crate:** fdars-core v0.14.0
**Audit milestone:** v0.14.0 — audit-only deliverable; no production code changes included
**Source report:** [AUDIT-REPORT.md](AUDIT-REPORT.md) (Phases 1–9)
**Produced by:** Phase 9 consolidation (Plans 01–03)

This file is the standalone prioritized backlog for the v0.14.0 audit milestone.
It is intended to be consumed directly by `/gsd-new-milestone` to promote items into
future milestone requirements. Each item is phrased as a GSD-ready candidate
requirement or phase.

---

## Ranking Methodology

### Formula

```
score = value / sqrt(effort)
```

A higher score means more user value delivered per unit of effort. Items are ordered by
descending score in the Ranked Backlog table. The formula rewards high-value items and
penalizes high-effort items non-linearly (large efforts are more than proportionally
expensive to deliver).

### Value Scale (1–5)

Value measures **user value**, not ease of implementation.

| Value | Anchor |
|-------|--------|
| 5 | Table-stakes capability blocking real workloads — absent capability that scikit-fda users rely on daily, or P1 default-path performance cost affecting every caller |
| 4 | High-value capability widely used in practice; present partial implementation needs significant work; or P1 hot-path saving >2× at common workload sizes |
| 3 | Meaningful capability or performance improvement; important but not blocking; commonly requested in FDA toolkits |
| 2 | Useful addition or moderate performance gain; niche use-case or limited to uncommon workload sizes |
| 1 | Niche differentiator, cosmetic improvement, or very minor performance gain with limited real-world impact |

### Effort Map (S / M / L)

| Effort | Numeric | sqrt(effort) | Definition |
|--------|---------|--------------|------------|
| S | 1 | 1.000 | Small — approximately 1 week of implementation including tests |
| M | 3 | 1.732 | Medium — approximately 2–4 weeks including integration and validation |
| L | 9 | 3.000 | Large — approximately 1–3 months or cross-cutting architectural change |

### Severity Scale

| Severity | Meaning |
|----------|---------|
| P1 | Default-path performance cost affecting every caller of a common function, or a table-stakes capability gap blocking real workloads |
| P2 | Meaningful but not blocking — measurable performance win or useful missing capability that sophisticated users notice |
| P3 | Niche or cosmetic-adjacent — minor gain limited to uncommon workload sizes or rare use-cases |

**Note:** Severity (P1/P2/P3) and Value (1–5) are correlated but independent. Severity
describes the category of impact; Value quantifies user benefit for ranking purposes.
A P2 item can have Value=4 if the improvement is significant for a moderately common
workload. A P1 item with wide reach but low absolute gain may have Value=3.

---

## Ranked Backlog

**FINAL MASTER TABLE — sorted by descending `score = value / sqrt(effort)` (Plan 03 sort). Rank column filled 1..31.**

Items with the same computed score are sub-ordered: P1 before P2 before P3 (higher severity first within the same score tier).

| Rank | ID | Title | Severity | Value (1–5) | Effort (S/M/L) | Score (value/sqrt(effort)) | Area / Location |
|------|----|-------|----------|------------|----------------|---------------------------|-----------------|
| 1 | REPR-02 | Implement spline (cubic/order-k) interpolation at off-grid points | P1 | 4 | S | 4.00 | Representation / `helpers.rs`, `basis/` |
| 2 | EXPL-02 | Add functional summary statistics: trim_mean, depth_median, cov, var, std | P1 | 4 | S | 4.00 | Exploratory / `fdata.rs`, `covariance.rs` |
| 3 | PERF-PAR-CV | Parallelize the classification CV fold loop | P2 | 4 | S | 4.00 | Classification CV / `classification/cv.rs:76` |
| 4 | P6-1 | Swap nalgebra SVD for faer thin_svd in `fdata_to_pc_1d` | P2 | 3 | S | 3.00 | FPCA / `regression.rs:298` |
| 5 | PREP-03 | Implement missing-value imputation for regular FdMatrix grids | P2 | 3 | S | 3.00 | Preprocessing / `helpers.rs`, `irreg_fdata/` |
| 6 | REPR-03 | Add composable extrapolation-policy enum (Boundary/Exception/Fill/Periodic) | P2 | 3 | S | 3.00 | Representation / `helpers.rs` interpolation/evaluation paths |
| 7 | INF-01 | Add asymptotic functional ANOVA V-statistic (oneway_anova + v_sample_stat) | P2 | 3 | S | 3.00 | Inference / `function_on_scalar.rs`, new `inference.rs` |
| 8 | INF-02 | Expose two-sample Hotelling T² as standalone inference function | P2 | 3 | S | 3.00 | Inference / `spm/stats.rs` → thin `inference` wrapper |
| 9 | MISC-04 | Add functional MAE, MSE scoring metrics (+ MAPE, MSLE, explained_variance) | P2 | 3 | S | 3.00 | Misc / `helpers.rs` or new `scoring.rs` |
| 10 | PERF-ELASTIC-BAND | Default elastic alignment to a banded path / expose band_frac | P1 | 5 | M | 2.89 | Elastic alignment / `alignment/karcher.rs:300`, `elastic_self/cross_distance_matrix` |
| 11 | PREP-04 | Implement shift-only (LeastSquaresShift) registration | P1 | 4 | M | 2.31 | Preprocessing / `alignment/`, no current shift estimator |
| 12 | PREP-06 | Implement derivative-penalty (LDO) regularized FPCA | P1 | 4 | M | 2.31 | Preprocessing / `regression.rs`, `smooth_basis.rs` |
| 13 | PREP-05 | Add registration-quality validation scores (LS, PairwiseCorrelation, Sobolev) | P2 | 2 | S | 2.00 | Preprocessing / `alignment/quality.rs` |
| 14 | PREP-01 | Add AIC/FPE/Shibata/Rice bandwidth-selection criteria to smoothing | P3 | 2 | S | 2.00 | Preprocessing / `smoothing.rs` |
| 15 | PREP-08 | Expose local_averages, occupation_measure, number_crossings as public APIs | P3 | 2 | S | 2.00 | Preprocessing / `helpers.rs` or new `feature_construction.rs` |
| 16 | MISC-01 | Add Mahalanobis, NormInduced, Transformation metrics + angular/cosine functions | P3 | 2 | S | 2.00 | Misc / `distance.rs`, `utility.rs` |
| 17 | PERF-PAR-ELFPCA | Parallelize the three elastic-FPCA inner N-loops | P2 | 3 | M | 1.73 | Elastic FPCA / `elastic_fpca.rs:701/720/764` |
| 18 | ACC-VALIDATE | Comparative fdars-vs-scikit-fda numerical-accuracy validation | P2 | 3 | M | 1.73 | Cross-cutting (Preprocessing / Misc / ML) |
| 19 | PREP-02 | Implement generic SmootherConfig abstraction + SmoothingParameterSearch | P2 | 3 | M | 1.73 | Preprocessing / `smoothing.rs`, kernel smoother variants |
| 20 | REPR-01 | Add MonomialBasis/ConstantBasis (and advanced: TensorBasis/FEBasis) to basis/ | P2 | 3 | M | 1.73 | Representation / `basis/` |
| 21 | EXPL-01 | Add pluggable-metric depth (DistanceBased) and OutlyingnessBased combinator | P2 | 3 | M | 1.73 | Exploratory / `depth/`, `distance.rs` |
| 22 | ML-01 | Add MaximumDepthClassifier, NearestCentroid, RadiusNeighbors variants | P2 | 3 | M | 1.73 | ML / `classification/` |
| 23 | ML-02 | Implement LDO-regularized linear regression + HistoricalLinearRegression | P2 | 3 | M | 1.73 | ML / `scalar_on_function/`, `smooth_basis.rs` |
| 24 | MISC-02 | Implement composable LinearDifferentialOperator and L2Regularization objects | P2 | 3 | M | 1.73 | Misc / `smooth_basis.rs`, new `operator.rs` trait |
| 25 | PREP-09 | Implement diffusion-map manifold embedding for functional data | P3 | 2 | M | 1.15 | Preprocessing / `distance.rs`, `regression.rs` (truncated eigen) |
| 26 | EXPL-03 | Implement Stahel-Donoho outlyingness for functional data | P3 | 2 | M | 1.15 | Exploratory / new in `depth/` or `outliers.rs` |
| 27 | MISC-03 | Add make_gaussian wrapper and make_sinusoidal_process dedicated generator | P3 | 2 | M | 1.15 | Misc / `simulation.rs`, `covariance.rs` |
| 28 | PERF-FPCA-TRUNCSVD | Truncated/thin SVD computing only ncomp components in FPCA | P2 | 3 | L | 1.00 | FPCA / `regression.rs:298` via `nalgebra::SVD::new` |
| 29 | PERF-PAR-CENTER | Parallelize center_columns on FPCA path | P3 | 1 | S | 1.00 | FPCA / `regression.rs:167` |
| 30 | REPR-04 | Implement EM and Minimize mixed-effects irregular-to-basis converters | P3 | 2 | L | 0.67 | Representation / `irreg_fdata/`, `famm.rs` |
| 31 | PREP-07 | Implement functional variable-selection methods (MaximaHunting, RKHS, mRMR) | P3 | 2 | L | 0.67 | Preprocessing / no current module — new `variable_selection.rs` |
| 32 | PERF-FPCA-CLONE | Eliminate redundant centered.clone() + zero-copy to_dmatrix() bridge | P3 | 1 | M | 0.58 | FPCA / `regression.rs:291/298` |

---

## Backlog Items

### P6-1 — Swap nalgebra SVD for faer thin_svd in `fdata_to_pc_1d`

**Candidate requirement / phase phrasing:** "Replace the nalgebra `SVD::new(weighted.to_dmatrix(), true, true)` call in `fdata_to_pc_1d` with faer `thin_svd` on a zero-copy `MatRef` view, gated behind the existing `linalg` feature."

- **Location / area:** `fdars-core/src/regression.rs`, line 298. The `fdata_to_pc_1d` function is the primary FPCA entry point called by scalar-on-function regression, functional logistic regression, and classification CV loops — every FPCA-backed computation in the library routes through this site.

- **Current cost or gap:** nalgebra SVD at the primary audit cell (N=500, M=200): **41.026 ms** (run1 median). This SVD step accounts for approximately **99.8–99.9%** of total `fdata_to_pc_1d` wall-clock. The `to_dmatrix()` bridge at the same line allocates an ~800 KB DMatrix copy (N×M×8 bytes) on every call; its copy-share is ~0.17% of wall-clock (negligible, but eliminable). At N=1000, M=200: 95.6 ms per FPCA call.

- **Root cause:** `nalgebra::SVD::new` requires a `DMatrix<f64>` input, which forces a `to_dmatrix()` column-major memcopy from `FdMatrix` before the SVD can begin. nalgebra's SVD implementation is always sequential regardless of the `parallel` feature flag. faer's `thin_svd` accepts a `MatRef` — a zero-copy view constructed directly from the `FdMatrix` column-major slice via `MatRef::from_column_major_slice` — and executes a faster SVD algorithm that consistently outperforms nalgebra at fdars' tall-thin (N >> M) rectangular matrix sizes.

- **Proposed direction:** Under `#[cfg(feature = "linalg")]`, replace the `weighted.to_dmatrix()` + `SVD::new` block with:
  ```
  let mat_ref = faer::MatRef::<f64>::from_column_major_slice(weighted.as_slice(), n, m);
  let fa_svd = mat_ref.thin_svd();
  ```
  Extract U, S, Vt using faer accessors (`.U()`, `.S()`, `.V()`). Retain the existing nalgebra path under `#[cfg(not(feature = "linalg"))]` so the `""` and `parallel` builds are unaffected. Add a CI regression test verifying `FpcaResult` output agrees with the nalgebra path within numerical tolerance. Evaluate faer parallel SVD (not measured in Phase 6) — if it offers additional speedup at M≥200, surface as a follow-on candidate.

- **Severity (P1/P2/P3):** **P2** — The measured speedup at the primary cell (N=500, M=200) is **1.8×** (run1) / **1.9×** (run2), below the research-defined "clearly worth it" threshold of ≥2×. The speedup is consistently positive at all 7 measured cells (3.6×, 4.1×, 2.7×, 1.8×, 3.7×, 3.1×, 1.9× — N∈{100,100,500,500,1000,1000,500} × M∈{50,200,50,200,50,200,500}). The absolute saving at N=1000, M=200 is ~27 ms/call — meaningful for FPCA-heavy workflows (pipeline loops, cross-validation grids). Downgrade to P3 if a pinned-governor re-run shows speedup < 1.5× at the primary cell.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. faer is already a dependency of `fdars-core` (no new Cargo.toml additions). Code change is ~20 lines in `fdata_to_pc_1d`. Output extraction requires mapping faer accessors to the existing `FpcaResult` fields (singular values, U, Vt). Singular vector sign conventions may differ from nalgebra — a one-time equivalence check is required. Numerical equivalence already confirmed by the `svd_equivalence` integration test.

- **Evidence link:** [bench/p6_svd_nalgebra_linalg_run1.txt](bench/p6_svd_nalgebra_linalg_run1.txt) (N=500, M=200: 41.026 ms) · [bench/p6_svd_faer_seq_linalg_run1.txt](bench/p6_svd_faer_seq_linalg_run1.txt) (N=500, M=200: 23.084 ms) · speedup: **1.8×**. Wall-clock source for copy-share derivation: [bench/p4_fpca_linalg,parallel_run1.txt](bench/p4_fpca_linalg,parallel_run1.txt) (N=1000, M=200: 38.307 ms; copy-share derived as 0.14% from 53.3 µs / 38,307 µs). Full comparison grid and narrative in AUDIT-REPORT.md §Phase 6 SC2.

---

### PERF-ELASTIC-BAND — Default elastic alignment to a banded path / expose band_frac

**Candidate requirement / phase phrasing:** "Change the default of `karcher_mean`, `elastic_self_distance_matrix`, and `elastic_cross_distance_matrix` to use the banded DP path (band_frac ≈ 0.1), and expose `band_frac` as an optional parameter on the high-level API. The banded implementations already exist and are correct — this is an API default change, not a new algorithm."

- **Location / area:** `fdars-core/src/alignment/karcher.rs` line 300 (`karcher_mean` hard-codes `band_frac=0.0`); `fdars-core/src/alignment/` — `elastic_self_distance_matrix` and `elastic_cross_distance_matrix` and their `_banded` variants. Affects every caller of the three high-level elastic alignment functions, which include the karcher mean, all elastic distance matrix computations, and downstream clustering/classification pipelines that call these entry points.

  **P5-4 cross-reference note:** This item is also the elastic-banding default-path cost identified as P5-4 in the Phase-5 SC3(b) / SC4 parallelism gap analysis. P5-4 is NOT a separate item — it is the same measured ~4–6× cost as PERF-ELASTIC-BAND. Any reference to P5-4 in the report resolves here.

- **Current cost or gap:** `karcher_mean` N=500,M=200 unbanded: ~18.9–28.8 s per iteration (LOW CONFIDENCE — OS scheduler jitter; stable baseline needed). `elastic_self_distance_matrix` N=500,M=50 unbanded: ~24–26 s (OK confidence). `elastic_cross_distance_matrix` N=500,M=50 unbanded: ~37–38 s (EXCELLENT confidence, 0% two-run variance). N=500,M=200 elastic distance matrices are **INFEASIBLE** to run (~384–700 s/iteration, estimated from extrapolation of N=500,M=50 trend) — this infeasibility is itself the primary bottleneck evidence. With banding at `band_frac=0.1`, the measured reduction is **4–6× at representative cells** (karcher N=500,M=200: ~4–5.9×; elastic_self N=500,M=50: 5.7×; elastic_cross N=100,M=200: 4.5×; elastic_cross N=100,M=50 banded: 322.73 ms vs ~1.55 s unbanded → 4.8×). Theoretical expectation at M=200: ~7× (m/band = 200/20 = 10× minus overhead).

- **Root cause:** `karcher_mean()` at `karcher.rs:300` calls `karcher_mean_impl(.., 0.0)` — the `band_frac=0.0` hard-code passes `band_radius(0.0, m) = None` (since `band_frac <= 0`), triggering the full O(m²) unbanded DP path per alignment pair. All three target functions follow the same opt-in pattern: banded variants (`karcher_mean_banded`, `elastic_self_distance_matrix_banded`, `elastic_cross_distance_matrix_banded`) exist and are correct, but users must explicitly call them. The default API gives no path to the faster banded implementation. Complexity: karcher is O(max_iter·N·m²) unbanded vs O(max_iter·N·m·band) banded; distance matrices are O(N²·m²) vs O(N²·m·band).

- **Proposed direction:** (1) Change `karcher_mean()` at `karcher.rs:300` to pass `band_frac=0.1` (or add a `band_frac: f64 = 0.1` parameter) instead of hard-coding 0.0. (2) Add a `band_frac: Option<f64>` (defaulting to `Some(0.1)`) or `band_frac: f64 = 0.1` parameter to `elastic_self_distance_matrix` and `elastic_cross_distance_matrix`, promoting the banded path as the default. (3) Document the performance tradeoff in rustdoc: `band_frac=0.0` gives exact unbanded results; the default `band_frac=0.1` gives 4–6× speedup with small band-approximation error. GSD-ready as candidate Phase: "Set elastic alignment API defaults to banded path (band_frac ≈ 0.1) to make N=500+, M=200+ workloads tractable."

- **Severity (P1/P2/P3):** **P1** — The default-path cost makes N=500,M=200 elastic distance matrices entirely infeasible (~700 s/iteration), blocking any real production workload with more than ~100 curves at M=200 evaluation points. `karcher_mean` is fdars' primary elastic shape analysis entry point, called by users who want registration/shape analysis. This is a table-stakes capability gap for typical real-data FDA workloads (N=200–1000 curves is standard in practice). Every caller of the default API pays this cost.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. The banded implementations already exist and are correct; no new algorithm is needed. Work involves: (a) changing the default parameter values or adding a `band_frac` optional parameter to the three public functions; (b) updating rustdoc on all three; (c) adding regression tests confirming numeric equivalence (banded vs unbanded at small M where exact comparison is feasible); (d) deciding on API shape (new parameter vs default change with deprecation note). The main risk is API compatibility — adding a parameter is a breaking change if callers use positional arguments; a `band_frac: Option<f64>` or a new `ElasticConfig` wrapper is the safer approach and adds a few more days.

- **Evidence link:** [bench/p3_elastic_cross_linalg,parallel_run1.txt](bench/p3_elastic_cross_linalg,parallel_run1.txt) (N=100,M=200: 27.85 s unbanded vs [banded](bench/p3_elastic_cross_banded_linalg,parallel_run1.txt) 6.16 s → **4.5×**; N=500,M=50: 37.82 s unbanded vs [banded](bench/p3_elastic_cross_banded_linalg,parallel_run1.txt) 8.01 s → **4.7×**) · [bench/p3_karcher_linalg,parallel_run1.txt](bench/p3_karcher_linalg,parallel_run1.txt) (karcher N=500,M=200 unbanded; banded comparison at [bench/p3_karcher_banded_linalg,parallel_run1.txt](bench/p3_karcher_banded_linalg,parallel_run1.txt)) · Infeasibility evidence at N=500,M=200: `elastic_cross` ~700 s/iter estimated, `elastic_self` ~384 s/iter estimated. Banded-vs-unbanded analysis: AUDIT-REPORT.md §Phase 3 → Banded-vs-Unbanded Analysis (SC2, D-03/D-04) and §D-05 Source Fact. Phase-5 cross-reference: AUDIT-REPORT.md §Phase 5 SC3(b) (banding opt-in cost) and SC4 P5-4.

---

### PERF-FPCA-CLONE — Eliminate redundant centered.clone() + zero-copy to_dmatrix() bridge

**Candidate requirement / phase phrasing:** "Eliminate the `centered.clone()` at `regression.rs:291` and evaluate a zero-copy `to_dmatrix()` bridge at `:298` to reduce per-FPCA-call heap traffic from three O(n·m) allocations to one."

- **Location / area:** `fdars-core/src/regression.rs` lines 291 (`.clone()` call) and 298 (`weighted.to_dmatrix()` call) inside `fdata_to_pc_1d`. This function is the primary FPCA entry point called by scalar-on-function regression, functional logistic regression, and classification CV loops.

- **Current cost or gap:** At N=500, M=200: **21 total_blocks**, **3,574,424 total_bytes** (35.74 bytes/n·m, corrected CR-02), **3,531,192 peak_bytes**. Three O(n·m) allocations of 800,000 bytes each: `FdMatrix::zeros(n,m)` at `:167` (necessary), `centered.clone()` at `:291` (zero-copy candidate — 800 KB redundant clone), `weighted.to_dmatrix()` at `:298` (nalgebra DMatrix bridge — 800 KB copy contributing ~0.17% of wall-clock at N=500,M=200; 0.14% at N=1000,M=200). The clone at `:291` is the primary target: the original `centered` is stored in `FpcaResult.centered`, but the weight-scaling step could use a pre-allocated buffer rather than a full clone.

- **Root cause:** The `centered.clone()` at `:291` creates a second 800 KB `FdMatrix` solely to apply integration-weight scaling before SVD, then discards it after SVD. The original `centered` is retained in `FpcaResult.centered` (needed for projection), so the clone cannot be avoided without restructuring the data flow. `to_dmatrix()` at `:298` (`DMatrix::from_column_slice`) is a column-major memcopy into nalgebra format required by `SVD::new`; it is eliminable if SVD is moved to a library (faer) that accepts a zero-copy `MatRef` view.

- **Proposed direction:** (a) Replace `centered.clone()` at `:291` with a pre-allocated output buffer: compute the weighted values directly into a fresh `FdMatrix` without retaining the intermediary, or restructure to share the buffer with `FpcaResult.centered`. (b) Evaluate zero-copy `to_dmatrix()` bridge: if SVD is moved to faer (`linalg` feature, per PERF-FPCA-TRUNCSVD / P6-1), `faer::MatRef::from_column_major_slice` constructs a zero-copy view from the `FdMatrix` slice, eliminating the `:298` copy entirely. GSD-ready as candidate Phase: "Eliminate `centered.clone()` at regression.rs:291 and evaluate zero-copy DMatrix bridge at `:298`."

- **Severity (P1/P2/P3):** **P3** — The copy-share of the `to_dmatrix()` bridge is **~0.14–0.17% of wall-clock** at all measured cells — a negligible fraction. The `centered.clone()` saves one 800 KB allocation per call, improving heap traffic but not measurably improving wall-clock (SVD at ~99.8–99.9% dominates). This is a cleanliness improvement, not a performance bottleneck. Useful for memory-constrained environments but not a priority for compute throughput.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. Restructuring `fdata_to_pc_1d` to avoid the clone while correctly retaining `FpcaResult.centered` requires careful data-flow analysis; incorrect restructuring could invalidate the stored `centered` field used by `project()`. The zero-copy bridge depends on first landing PERF-FPCA-TRUNCSVD or P6-1 (faer SVD adoption) — it cannot eliminate the `to_dmatrix()` copy without changing the SVD backend.

- **Evidence link:** [bench/p4_dhat_fpca_n500_m200.txt](bench/p4_dhat_fpca_n500_m200.txt) (N=500,M=200: 21 total_blocks, 3,574,424 total_bytes, 3,531,192 peak_bytes — three O(n·m) allocations of 800 KB each) · [bench/p4_fpca_linalg,parallel_run1.txt](bench/p4_fpca_linalg,parallel_run1.txt) (N=1000,M=200: 38.307 ms wall-clock; copy-share derivation: 53.3 µs / 38,307 µs = 0.14%) · Full allocation analysis and copy-share derivation: AUDIT-REPORT.md §Phase 4 → Allocation Audit and §Phase 4 → SVD-Compute vs Copy Split.

---

### PERF-FPCA-TRUNCSVD — Truncated/thin SVD computing only ncomp components in FPCA

**Candidate requirement / phase phrasing:** "Replace `nalgebra::SVD::new(dmatrix, true, true)` (full thin SVD of N×M matrix) in `fdata_to_pc_1d` with a truncated-SVD routine that computes only the top ncomp singular components, targeting O(N·M·ncomp) vs current O(min(N,M)²·max(N,M)) cost."

- **Location / area:** `fdars-core/src/regression.rs` line 298 (`nalgebra::SVD::new(weighted.to_dmatrix(), true, true)`) inside `fdata_to_pc_1d`. Every FPCA-backed computation in the library routes through this site: scalar-on-function regression, functional logistic regression, classification CV loops, and elastic FPCA.

- **Current cost or gap:** At N=1000, M=200: **38.307 ms** wall-clock; at N=500, M=200: **16.011 ms**. SVD is **~99.8–99.9% of total wall-clock** at every measured cell. The full SVD at M=200 computes all 200 singular values/vectors; only the top `ncomp` (typically 3–10) are retained after a `[:ncomp]` slice. At M=200 with ncomp=5, the full SVD computes **~40× more components than needed**. M-scaling (N=100): n100_m200 (1.690 ms) vs n100_m50 (213.3 µs) — **~7.9× slower for 4× more M**, consistent with O(m²) SVD scaling, confirming the M-scaling bottleneck is the full decomposition. A truncated SVD could reduce the SVD cost by up to O(M/ncomp) = 40× at M=200,ncomp=5.

- **Root cause:** `nalgebra::SVD::new` always computes the full SVD — all min(N,M) singular values/vectors at cost O(min(N,M)² · max(N,M)). For the typical FPCA use case where ncomp « M (e.g., 5 « 200 « 1000), the algorithm computes far more than needed. Iterative truncated-SVD methods (randomized SVD, Lanczos, LOBPCG) compute only the top k components at O(N·M·k) — a ~M/ncomp reduction in SVD cost. faer `thin_svd` (P6-1) is still full thin SVD; true truncation requires a different algorithm.

- **Proposed direction:** Evaluate and implement a truncated-SVD routine for `fdata_to_pc_1d`: (1) **Randomized SVD** (Halko-Martinsson-Tropp algorithm) — a standard approach, implementable in pure Rust (~200 lines), computes top-k singular vectors in O(N·M·k) with controllable approximation error via oversampling parameter. (2) **LAPACK DGESDD partial request** via `ndarray-linalg` — exact but requires LAPACK linkage (additional dependency). (3) **faer iterative SVD** — evaluate if faer 0.23+ exposes a truncated variant. Add an `ncomp`-vs-M guard: when ncomp/M > 0.5 (half or more components requested), fall back to full SVD (truncated is more expensive than full for large ncomp/M ratios). GSD-ready as candidate Phase: "Implement truncated/thin SVD in fdata_to_pc_1d (top ncomp components only)."

- **Severity (P1/P2/P3):** **P2** — SVD is the dominant cost at ~99.8–99.9% of wall-clock; a truncated SVD could halve or better the FPCA runtime for typical ncomp « M usage. Meaningful for FPCA-heavy workflows (pipeline loops, cross-validation grids at large N×M). Not P1 because FPCA itself is 2–3 orders of magnitude cheaper than elastic alignment (16–38 ms vs 24+ s per call), so this is not the primary production bottleneck for most workloads.

- **Effort estimate (S/M/L):** **L** — approximately 1–3 months. Replacing or wrapping nalgebra SVD requires evaluating numerical stability of truncated methods for FPCA (sign convention, convergence, oversampling), implementing or integrating a truncated-SVD library, adding an ncomp-vs-M guard, and adding regression tests for the FPCA numerical path across all downstream consumers. Risk: truncated SVD introduces approximation error controlled by oversampling — incorrect parameterization could produce silently inaccurate FPCA results.

- **Evidence link:** [bench/p4_fpca_linalg,parallel_run1.txt](bench/p4_fpca_linalg,parallel_run1.txt) (N=1000,M=200: 38.307 ms; N=100,M=200: 1.6896 ms; N=100,M=50: 213.33 µs — M-scaling ~7.9× for 4× more M confirming O(m²) SVD) · [bench/p4_fpca_linalg,parallel_run2.txt](bench/p4_fpca_linalg,parallel_run2.txt) · Full SVD-dominance derivation: AUDIT-REPORT.md §Phase 4 → SVD-Compute vs Copy Split · Phase 4 Draft Backlog entry 2.

---

### PERF-PAR-CV — Parallelize the classification CV fold loop

**Candidate requirement / phase phrasing:** "Wrap the `for fold in 0..nfold` loop in `fclassif_cv` (`classification/cv.rs:76`) in `iter_maybe_parallel!(0..nfold)` to enable parallel CV fold execution, targeting ~4–5× speedup at machine-default threads."

- **Location / area:** `fdars-core/src/classification/cv.rs` line 76 — the outer `for fold in 0..nfold` fold loop in `fclassif_cv`. Called by classification cross-validation workflows; each fold executes a full FPCA fit + classifier fit + predict sequence.

- **Current cost or gap:** `fclassif_cv` (lda, 5-fold) at N=100, M=50: **~948–952 µs/iteration** (OK confidence, 0.5% two-run variance). The fold loop runs `nfold` sequential FPCA+fit+predict passes; per-fold FPCA SVD dominates each fold's cost. With 5 folds and ~190–200 µs per fold (estimated from the 950 µs total), parallelizing 5 folds onto 5+ threads could reduce total CV time proportionally.

- **Root cause:** `cv.rs:76` uses a plain `for fold in 0..nfold` with no parallelism macro. Folds are fully independent (disjoint train/test splits, no shared mutable state). The fold-assignment RNG (`assign_folds`) runs once before the loop, producing a deterministic `Vec<usize>` fold map — the loop body has no RNG calls, so no per-thread seeding is needed. The `iter_maybe_parallel!` macro infrastructure already exists in `parallel.rs` and would gate this on the `parallel` feature flag identically to existing parallel loops.

- **Proposed direction:** Replace `for fold in 0..nfold` with `iter_maybe_parallel!(0..nfold)` (per SC2). Each fold body is heavy (a full FPCA SVD + classifier fit), so it tracks the karcher heavy-body regime from SC1 (payback N≤10) — worth parallelizing at any realistic nfold (5–20 is typical). The results currently assembled via a sequential push would need to be collected into a `Vec` via `.map().collect()` pattern. GSD-ready as candidate Phase: "Parallelize fclassif_cv fold loop via iter_maybe_parallel!(0..nfold)."

- **Severity (P1/P2/P3):** **P2** — CV fold parallelism is a meaningful throughput improvement for any workflow that performs hyperparameter search or repeated cross-validation (a common FDA workflow). Not P1 because the CV loop is not on the default single-call path (users must explicitly invoke `fclassif_cv`), and the absolute time per 5-fold run is small (~950 µs) at N=100,M=50 — though it scales linearly with N, M, nfold, and ncomp.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. `iter_maybe_parallel!` is already available in `parallel.rs`; the change is a one-line macro substitution + updating the result collection pattern. Integration test to confirm identical fold accuracy under parallel and sequential execution.

- **Evidence link:** [bench/p1_cv_linalg,parallel_run1.txt](bench/p1_cv_linalg,parallel_run1.txt) (N=100,M=50 `fclassif_cv` lda 5-fold: **947–952 µs/iteration**, OK confidence) · SC1 thread-scaling table (karcher heavy-body: ~4.7× ceiling at 8 threads, payback at N≤10) → projected **~4–5× speedup** at machine-default threads for any realistic nfold. Static independence argument: AUDIT-REPORT.md §Phase 5 SC2 (`cv.rs:76` row). Phase-5 backlog entry P5-1: AUDIT-REPORT.md §Phase 5 SC4.

---

### PERF-PAR-ELFPCA — Parallelize the three elastic-FPCA inner N-loops

**Candidate requirement / phase phrasing:** "Wrap the three per-curve `for i in 0..n` loops in `shooting_vectors_from_psis` (`elastic_fpca.rs:701`), `build_augmented_srsfs` (`elastic_fpca.rs:720`), and `svd_scores_and_eigenvalues` (`elastic_fpca.rs:764`) in `iter_maybe_parallel!(0..n)` to parallelize the elastic-FPCA critical path."

- **Location / area:** `fdars-core/src/elastic_fpca.rs` lines 701, 720, and 764 — three per-curve `for i in 0..n` loops on the elastic-FPCA computation path. These loops are called sequentially on every `vert_fpca` / `joint_fpca` call.

- **Current cost or gap:** No dedicated benchmark measurement exists for these three loops individually (no `p5_elfpca` artifact). The `vert_fpca` reference cell (N=100,M=50) is **300.64 µs** and `joint_fpca` is **1.8850 ms** (both single-run reference measurements, `linalg,parallel` build). These are secondary reference points, not primary variance-tracked cells. The three loops are on the critical path of elastic-FPCA and are SEQUENTIAL (confirmed by `grep` of `iter_maybe_parallel` returning zero hits at these lines — §Parallelism Gap List §SEQUENTIAL rows). **Current cost is PROJECTED, not measured directly for these loops.**

- **Root cause:** `elastic_fpca.rs:701/720/764` all use plain `for i in 0..n` with no parallelism macro. Each iteration writes a disjoint per-curve row or score entry: `:701` writes a shooting-vector row for curve `i`; `:720` builds an augmented SRSF row for curve `i`; `:764` extracts a score for curve `i` from pre-computed SVD factors. No cross-iteration dependency exists; all three are safe to parallelize (static independence argument confirmed in §Phase 5 SC2). No RNG appears in any of the three loop bodies.

- **Proposed direction:** Wrap each of the three loops in `iter_maybe_parallel!(0..n)` (per SC2 static analysis). For `:701` and `:720` (per-curve row writes), the result collection may need a `.collect::<Vec<_>>()` + row-assignment pass. For `:764` (score extraction), the light per-iteration body (a dot-product-scale computation) sits nearer the streaming-sentinel payback regime — consider guarding behind a size threshold (N ≳ 50) or accept a small-N regression. GSD-ready as candidate Phase: "Parallelize the three elastic-FPCA per-curve loops in elastic_fpca.rs:701/720/764."

- **Severity (P1/P2/P3):** **P2** — Elastic FPCA is the registration-aware FPCA path and is typically called with N ≥ 50 curves where the speedup pays back. The improvement is meaningful for elastic-FPCA-heavy workflows but does not affect the common `fdata_to_pc_1d` path used by most users.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. Three separate loops to parallelize; `:764` light-body needs a size-threshold guard. Integration tests must confirm numeric equivalence of elastic-FPCA output (scores, eigenvalues) between parallel and sequential execution — important because elastic geometry computations can be sensitive to floating-point order.

- **Evidence link:** SC1 thread-scaling table: AUDIT-REPORT.md §Phase 5 SC1 (karcher `iter_maybe_parallel!(0..n)` scales ~4.7× at 8 threads; projected **~4–5× speedup** for `:701/720` at N≥50; `:764` light-body may need N≳50 threshold per SC1 payback rule). SC2 static-independence analysis: AUDIT-REPORT.md §Phase 5 SC2 (`elastic_fpca.rs:701/720/764` rows). Phase 5 backlog entry P5-2: AUDIT-REPORT.md §Phase 5 SC4. Allocation context: AUDIT-REPORT.md §Phase 4 Allocation Hotspot List (`elastic_fpca.rs:214/317/483/584/930`).

---

### PERF-PAR-CENTER — Parallelize center_columns on the FPCA path

**Candidate requirement / phase phrasing:** "Wrap the outer-M loop in `center_columns` (`regression.rs:167`) in `iter_maybe_parallel!(0..m)` to parallelize column-centering on the `fdata_to_pc_1d` path."

- **Location / area:** `fdars-core/src/regression.rs` line 167 — the `center_columns` function called inside `fdata_to_pc_1d` before SVD. The outer `for j in 0..m` loop subtracts each column's mean (m iterations; each column `j` independently updated). **Distinct from the already-parallel `fdata.rs:center_1d`** (RESEARCH Pitfall 1) — this is the sequential centering function on the FPCA path only.

- **Current cost or gap:** `center_columns` is on the `fdata_to_pc_1d` path (`fdata_to_pc_1d` at N=500,M=200: **16.011 ms** total, but SVD dominates at ~99.8–99.9% of wall-clock). Centering is O(N·M) with a trivial per-element body — its share of the 16 ms total is negligible next to the SVD step. At N=1000,M=200 (38.307 ms total): centering ≈ O(200,000) simple arithmetic operations — estimated at < 0.5% of wall-clock. **No dedicated centering benchmark exists** (the FPCA benchmark measures the full `fdata_to_pc_1d` including SVD).

- **Root cause:** `regression.rs:167` uses a plain `for j in 0..m` / `for i in 0..n` double loop with no parallelism macro (confirmed zero `iter_maybe_parallel` hits at `:167` — §Parallelism Gap List §SEQUENTIAL rows). Column-major layout makes each column `j` independent (centering subtracts that column's mean only). No shared mutable state across columns; no RNG in loop body.

- **Proposed direction:** Wrap the outer `for j in 0..m` loop in `iter_maybe_parallel!(0..m)` (per SC2). This is the lowest-priority parallelism candidate: the per-element body is very light (streaming regime, sitting well below the SC1 payback crossover except at large M) and SVD — not centering — is the FPCA M-scaling bottleneck. Net FPCA-call speedup would be small even with ideal centering parallelism. GSD-ready as candidate Phase: "Parallelize center_columns at regression.rs:167 via iter_maybe_parallel!(0..m)."

- **Severity (P1/P2/P3):** **P3** — Centering is a negligible fraction of FPCA wall-clock (SVD at ~99.8–99.9% dominates). Parallelizing centering would improve heap allocation patterns marginally but would not move the needle on FPCA throughput. Useful only as a code-consistency improvement (all heavy loops parallelized via `iter_maybe_parallel!`) not a performance win.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. One-line macro substitution analogous to the CV fold change; update result collection pattern; add integration test confirming identical centered output between parallel and sequential execution.

- **Evidence link:** [bench/p1_fpca_linalg,parallel_run1.txt](bench/p1_fpca_linalg,parallel_run1.txt) (N=500,M=200: 16.155 ms — total `fdata_to_pc_1d`; centering share not separately measured, estimated negligible vs SVD). SC1 payback rule: AUDIT-REPORT.md §Phase 5 SC1 §Payback-Threshold N (light-body rule — streaming regime, pays back only at large M). SC2 independence analysis: AUDIT-REPORT.md §Phase 5 SC2 (`regression.rs:167` row). Phase 5 backlog entry P5-3: AUDIT-REPORT.md §Phase 5 SC4.

---

### ACC-VALIDATE — Comparative fdars-vs-scikit-fda numerical-accuracy validation

**Candidate requirement / phase phrasing:** "Run a comparative numerical-accuracy validation of fdars against scikit-fda 0.10.1 across four fragile/known-bug areas: B-spline round-trip (GH #33), elastic level-encoding (GH #34), Lomb-Scargle NaN handling, and GMM over-split fix (commit ec17d138). Use the existing scikit-fda venv at `.planning/research/skfda-verify/venv`."

- **Location / area:** Cross-cutting — four areas:
  1. `fdars-core/src/smooth_basis.rs` / `src/basis/` — `fdata_to_basis()` / `basis_to_fdata()` B-spline round-trip (GH #33, commit `2fb6d3c9`, FIXED in v0.14.0)
  2. `fdars-core/src/alignment/karcher.rs` / `elastic_fpca.rs` — elastic level-encoding midpoint anchor (`gauss_model()` / `joint_gauss_model()`, GH #34, commit `6ed62398`, FIXED in v0.14.0)
  3. `fdars-core/src/seasonal/lomb_scargle.rs` — Lomb-Scargle NaN/Inf silently dropped via post-hoc `filter(|x| x.is_finite())` (CONCERNS.md §Fragile Areas — not yet fixed)
  4. `fdars-core/src/gmm/` — GMM over-split covariance-floor-scaling fix (commit `ec17d138`, v0.13.2, FIXED)

- **Current cost or gap:** This milestone (Phase 8) flags accuracy concerns but does NOT run numeric comparisons (D-02 / D-02a flag-only policy). All four areas carry "present — accuracy NOT verified" flags in the Phase-8 parity tables. A comparative validation pass is needed before these capabilities can be reported as fully correct against scikit-fda. The scikit-fda 0.10.1 venv is already present at `.planning/research/skfda-verify/venv` — no environment setup is required.

- **Root cause:** Deferred by D-02 / D-02a decision during Phase 8 audit planning. The audit-only milestone fence (RPT-01 scope constraint) excluded comparative accuracy testing — only the parity survey and gap identification were in scope. The four fragile areas require: two known-bug fixes that are reported as fixed but have narrow regression coverage, one unfixed silent-NaN issue, and one fixed over-split bug whose fix has no independent benchmark comparison.

- **Proposed direction:** (a) Add a Rust test binary that exercises the four fragile areas and outputs CSV comparison data. (b) Add a Python comparison script that imports the fdars CSV output and scikit-fda 0.10.1 output, computes residuals, and reports pass/fail. (c) Integrate as a new `tests/validate_against_skfda.rs` integration test (or a `benches/` accuracy benchmark). Required comparisons: (1) `fdata_to_basis` → `basis_to_fdata` round-trip residuals vs scikit-fda `BasisSmoother` on Berkeley growth, Aemet, and synthetic-step datasets; (2) elastic registration sample-mean vs data-mean on Growth/Aemet/synthetic-bumps datasets; (3) Lomb-Scargle self-consistency: constant signal → zero power spectrum; white noise → no dominant period; (4) GMM cluster assignments vs scikit-learn `GaussianMixture` at n=200, k=3, varying covariance types. GSD-ready as candidate Phase: "Run comparative fdars-vs-scikit-fda numerical-accuracy validation for the four fragile areas (B-spline, elastic level-encoding, Lomb-Scargle, GMM)."

- **Severity (P1/P2/P3):** **P2** — Unverified accuracy for known-bug areas is a meaningful correctness risk. The two FIXED bugs (GH #33, GH #34, GMM over-split) could still have edge-case regressions with narrow test coverage; the Lomb-Scargle NaN issue is unfixed and actively silences errors. Not P1 because the fixes are claimed in the codebase and narrow regressions exist; the risk is edge-case failure, not systematic incorrectness.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. The scikit-fda venv is already installed; the standard datasets (Berkeley growth, Aemet) are available via scikit-fda's built-in data loaders. Work involves: Python comparison harness (~200 lines), Rust test binary outputting CSV (~300 lines), integration of the comparison as a CI-optional test (Python required in CI environment). Main risk: scikit-fda 0.10.1 API surface — some functions may have moved since the milestone was planned.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → D-02a: Comparative Numerical-Accuracy Validation (ACC-01) — complete description of the four fragile areas, recommended approach, and the deferred decision rationale. scikit-fda venv: `.planning/research/skfda-verify/venv` (present — confirmed during Phase 8 setup). Phase 8 parity table flags: AUDIT-REPORT.md §Phase 8 → Preprocessing Parity Table rows for `BasisSmoother` and the elastic-level-encoding area; §Phase 8 → ML Parity Table row for GMM; §Phase 8 → Seasonal Parity Table row for Lomb-Scargle.

---

### PREP-01 — Add AIC/FPE/Shibata/Rice bandwidth-selection criteria to smoothing

**Candidate requirement / phase phrasing:** "Add AIC (`akaike_information_criterion`), FPE (`finite_prediction_error`), Shibata (`shibata`), and Rice (`rice`) bandwidth-selection criteria to `smoothing.rs`, alongside the existing CV/GCV `CvCriterion` variants."

- **Location / area:** `fdars-core/src/smoothing.rs` — the `CvCriterion` enum and associated bandwidth-selection logic. scikit-fda area: `smoothing` module (`akaike_information_criterion`, `finite_prediction_error`, `shibata`, `rice` standalone functions).

- **Current cost or gap:** fdars' `CvCriterion` enum offers only `Cv` and `Gcv`. scikit-fda provides four additional analytical bandwidth criteria — AIC, FPE, Shibata, and Rice — as named functions. All four are absent in fdars. Category: differentiator.

- **Root cause:** `smoothing.rs` implements the CV/GCV path only. The four additional criteria are analytical (closed-form formulas over the hat matrix trace): AIC = log(RSS/n) + 2·tr(H)/n; FPE = RSS·(1 + tr(H)/n) / (n − tr(H)/n); Shibata and Rice are similar hat-matrix-based expressions. No new algorithm is required — each formula is O(n²) at most (hat matrix computation dominates).

- **Proposed direction:** Add `AIC`, `FPE`, `Shibata`, and `Rice` variants to the `CvCriterion` enum in `smoothing.rs` and implement their corresponding criterion-value computations (using the same hat-matrix trace already computed for GCV). GSD-ready as candidate Phase: "Add AIC/FPE/Shibata/Rice bandwidth-selection criteria to smoothing.rs."

- **Severity (P1/P2/P3):** **P3** — These are alternative bandwidth criteria; CV and GCV already cover the most common use cases. The additional criteria are useful for users who prefer information-theoretic bandwidth selection, but their absence does not block any common workflow. Category: differentiator.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. No new algorithm: each criterion is a formula over the hat-matrix trace already computed in the GCV path. Implementation involves adding four enum variants and their formula evaluations (~30–50 lines). Tests: verify criterion values against known analytical results on synthetic smooth data.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Preprocessing Parity Table → `akaike_information_criterion` / `finite_prediction_error` / `shibata` / `rice` rows (all verdict: absent, differentiator). Phase 8 SUMMARY: [09-01-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-01-SUMMARY.md).

---

### PREP-02 — Implement generic SmootherConfig abstraction + SmoothingParameterSearch

**Candidate requirement / phase phrasing:** "Implement a `SmootherConfig` enum (or trait object) that swaps smoothing hat-matrix strategies uniformly, and a `SmoothingParameterSearch` wrapper for grid-search over arbitrary smoothing parameters — matching scikit-fda's `KernelSmoother` abstraction pattern."

- **Location / area:** `fdars-core/src/smoothing.rs` — the NW, local-linear, local-poly, and kNN smoother variants. scikit-fda area: `smoothing` module (`KernelSmoother` with pluggable `LinearSmootherLeaveOneOutScorer` and `SmoothingParameterSearch`).

- **Current cost or gap:** No single strategy-object abstraction that swaps smoothing hat-matrix strategies uniformly. No generic grid-search wrapper over arbitrary smoothing parameters. fdars uses free functions per smoother variant; users must call each function separately with different bandwidth parameters and compare results manually. Category: table-stakes (the abstraction is expected in any FDA toolkit that users want to tune by bandwidth).

- **Root cause:** fdars uses free functions per smoother variant (NW, local-linear, local-poly, kNN). The abstraction layer that would let users swap strategies by config is absent. Implementing a `SmootherConfig` enum or a `Smoother` trait with `hat_matrix()` method would unblock `SmoothingParameterSearch` (a grid search over bandwidths that evaluates a `CvCriterion` for each parameter value).

- **Proposed direction:** (1) Define a `SmootherConfig` enum (`NadarayaWatson { bandwidth }`, `LocalLinear { bandwidth }`, `LocalPolynomial { bandwidth, degree }`, `KNearestNeighbors { k }`) or a `Smoother` trait with a `smooth(data, argvals)` method and a `hat_matrix()` method. (2) Implement `SmoothingParameterSearch` as a generic grid-search struct that accepts a `SmootherConfig` and a `CvCriterion`, runs leave-one-out scoring for each parameter value, and returns the optimal bandwidth. GSD-ready as candidate Phase: "Implement SmootherConfig abstraction and SmoothingParameterSearch for fdars."

- **Severity (P1/P2/P3):** **P2** — The strategy abstraction is table-stakes for users who want to compare smoothing approaches or perform bandwidth tuning in a pipeline. Its absence requires manual looping over smoother variants — not blocking but meaningfully inconvenient for common FDA preprocessing workflows.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. Defining the trait/enum and implementing hat-matrix computation for each smoother variant (or adapting existing free functions). The `SmoothingParameterSearch` logic is straightforward (grid-search loop + criterion evaluation); the main work is API design and connecting existing smoother implementations.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Preprocessing Parity Table → `KernelSmoother` / `SmoothingParameterSearch` rows (verdict: absent, table-stakes). Phase 8 SUMMARY: [09-01-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-01-SUMMARY.md).

---

### PREP-03 — Implement missing-value imputation for regular FdMatrix grids

**Candidate requirement / phase phrasing:** "Add an `impute_missing_values(data: &mut FdMatrix, argvals: &[f64])` public function to `helpers.rs` (or a new `imputation.rs` module) that fills NaN entries in a regular-grid `FdMatrix` using linear or spline interpolation along each curve's evaluation axis."

- **Location / area:** `fdars-core/src/helpers.rs` (candidate location for `impute_missing_values`) and `fdars-core/src/irreg_fdata/` (existing irregular→regular conversion infrastructure). scikit-fda area: `preprocessing` module (`MissingValuesInterpolation` transformer).

- **Current cost or gap:** No dedicated in-grid NaN-imputation transformer. fdars has `irreg_fdata::to_regular_grid` (irregular→regular kernel fill) and `helpers::linear_interp` (point-to-point linear interpolation), but no named function that walks an `FdMatrix` detecting NaN entries and filling them in-place via interpolation between adjacent non-NaN observations. Category: table-stakes (users with missing or sensor-dropout data need imputation before any analysis).

- **Root cause:** The irregular-data and interpolation pieces exist. `linear_interp` computes a linearly interpolated value between two `(x, y)` pairs. Composing these into `impute_missing_values(data: &mut FdMatrix)` — scan each row for NaN entries, find their bounding non-NaN neighbors, apply linear interpolation — is the missing step. No new algorithm is required.

- **Proposed direction:** Add `impute_missing_values(data: &mut FdMatrix, argvals: &[f64], method: ImputationMethod)` where `ImputationMethod` is `Linear` (using `helpers::linear_interp`) or `Constant(f64)` (fill with a fixed value). For each row `i` in `data`, scan for NaN entries, find the nearest non-NaN neighbors in `argvals`, interpolate. GSD-ready as candidate Phase: "Add MissingValuesInterpolation (in-grid NaN imputation) to fdars helpers."

- **Severity (P1/P2/P3):** **P2** — Missing-value imputation is table-stakes for real datasets (sensor dropouts, incomplete registrations). The absence requires users to pre-fill NaNs outside fdars before analysis. Not P1 because the `irreg_fdata` path provides a workaround for irregular data; the specific in-grid NaN case is the gap.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. The linear interpolation logic is already in `helpers::linear_interp`. The new function scans `FdMatrix` rows for NaN entries and calls the existing interpolator. Adding the constant-fill variant and the `ImputationMethod` enum adds minor complexity.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Preprocessing Parity Table → `MissingValuesInterpolation` row (verdict: absent, table-stakes). Phase 8 SUMMARY: [09-01-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-01-SUMMARY.md).

---

### PREP-04 — Implement shift-only (LeastSquaresShift) registration

**Candidate requirement / phase phrasing:** "Implement `least_squares_shift_registration(data: &FdMatrix, argvals: &[f64]) -> FregreResult` that aligns each curve to the sample mean by minimizing the L2-distance under a constant horizontal shift — scikit-fda's `LeastSquaresShiftRegistration`."

- **Location / area:** `fdars-core/src/alignment/` — no current shift-only registration function. scikit-fda area: `preprocessing` module (`LeastSquaresShiftRegistration`).

- **Current cost or gap:** No shift-only LS registration in fdars. fdars jumps from landmark shifts (full landmark warping) to full elastic SRSF warping — the simple rigid-shift estimator is absent. `landmark_shift_deltas` is computed internally inside `landmark_register` but not returned as a standalone output. Category: table-stakes (shift registration is the simplest registration method and is widely expected as a starting-point alternative to full elastic alignment).

- **Root cause:** fdars has no `least_squares_shift_registration` function. The algorithm minimizes `‖curve_i(t − δ_i) − mean(t)‖²` over scalar shift `δ_i` per curve — a 1D optimization with a simple golden-section or ternary-search solver (per-curve, O(m) each evaluation). The mean function is `fdata::functional_mean`. The infrastructure for all components is present; the function itself is missing.

- **Proposed direction:** Implement `least_squares_shift_registration(data: &FdMatrix, argvals: &[f64], max_shift: f64) -> Result<RegistrationResult, FdarError>` where `RegistrationResult` carries the registered curves and the per-curve shift values `δ_i`. The 1D optimization over `δ_i` uses golden-section search on `‖data_i(t − δ_i) − mean(t)‖²` evaluated via linear interpolation. GSD-ready as candidate Phase: "Implement LeastSquaresShiftRegistration for fdars alignment module."

- **Severity (P1/P2/P3):** **P1** — Shift registration is the table-stakes entry-level registration method: it is faster than elastic alignment by 2–3 orders of magnitude, and many FDA workflows start with shift registration before graduating to full elastic alignment. Its absence forces users who only need simple alignment to use the computationally expensive elastic path or to implement shift registration themselves. This is a meaningful capability gap for the default (non-elastic) user.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. The golden-section optimizer (~50 lines), the L2-to-mean objective (using `fdata::functional_mean` and linear interpolation), and the result type. Integration tests against scikit-fda `LeastSquaresShiftRegistration` output on standard datasets (Berkeley growth, Aemet) are recommended but require the scikit-fda venv from ACC-VALIDATE.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Preprocessing Parity Table → `LeastSquaresShiftRegistration` row (verdict: absent, table-stakes). Phase 8 SUMMARY: [09-01-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-01-SUMMARY.md).

---

### PREP-05 — Add registration-quality validation scores (LS, PairwiseCorrelation, Sobolev)

**Candidate requirement / phase phrasing:** "Add `least_squares_score`, `pairwise_correlation_score`, and `sobolev_least_squares_score` functions to `alignment/quality.rs` — scikit-fda's `LeastSquares`, `PairwiseCorrelation`, and `SobolevLeastSquares` validation statistics."

- **Location / area:** `fdars-core/src/alignment/quality.rs` — existing quality functions (`alignment_quality`, `warp_complexity`, `warp_smoothness`). scikit-fda area: `preprocessing.registration.validation` module (`LeastSquares`, `SobolevLeastSquares`, `PairwiseCorrelation`).

- **Current cost or gap:** `alignment::quality::alignment_quality` / `warp_complexity` / `warp_smoothness` exist but do not match the specific sum-of-squares-to-mean LS score. `LeastSquares` computes `∑_i ‖registered_i − mean‖² / n`. `PairwiseCorrelation` computes the mean pairwise correlation between registered curves. `SobolevLeastSquares` adds a derivative-penalty term. Category: `LeastSquares` / `PairwiseCorrelation` = table-stakes; `SobolevLeastSquares` = differentiator.

- **Root cause:** New score functions could be added to `alignment/quality.rs` without structural change. `LeastSquares` is a single O(n·m) formula over `fdata::functional_mean` output. `PairwiseCorrelation` is O(n²·m). `SobolevLeastSquares` requires derivative approximation (already in `helpers.rs`).

- **Proposed direction:** Add three functions to `alignment/quality.rs`: `least_squares_score(registered: &FdMatrix, argvals: &[f64]) -> f64`, `pairwise_correlation_score(registered: &FdMatrix, argvals: &[f64]) -> f64`, `sobolev_least_squares_score(registered: &FdMatrix, argvals: &[f64], lambda: f64) -> f64`. GSD-ready as candidate Phase: "Add LS, PairwiseCorrelation, SobolevLS registration-quality scores to alignment/quality.rs."

- **Severity (P1/P2/P3):** **P2** — Registration quality scores are standard diagnostic tools for validating that registration improved curve alignment; their absence requires users to implement their own post-registration diagnostics. Table-stakes for the LS and PairwiseCorrelation variants (commonly reported in FDA papers); differentiator for SobolevLS.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. All three are formula implementations over existing `FdMatrix` operations. No new data structures. Tests: verify against scikit-fda output on standard datasets.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Preprocessing Parity Table → `LeastSquares` / `SobolevLeastSquares` / `PairwiseCorrelation` rows (verdict: absent, table-stakes / differentiator). Phase 8 SUMMARY: [09-01-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-01-SUMMARY.md).

---

### PREP-06 — Implement derivative-penalty (LDO) regularized FPCA

**Candidate requirement / phase phrasing:** "Add a `fdata_to_pc_1d_regularized(data: &FdMatrix, argvals: &[f64], ncomp: usize, basis_type: BasisType, lambda: f64) -> Result<FpcaResult, FdarError>` function that solves the LDO-penalized FPCA generalized eigenvalue problem (K·w = λ·(M + αP)·w), using the existing penalty matrix from `smooth_basis::bspline_penalty_matrix`."

- **Location / area:** `fdars-core/src/regression.rs` (`fdata_to_pc_1d` — unregularized FPCA entry point) and `fdars-core/src/smooth_basis.rs` (`bspline_penalty_matrix` / `fourier_penalty_matrix` — penalty matrix infrastructure). scikit-fda area: `decomposition` module (`FPCA` with `LinearDifferentialOperator` regularization).

- **Current cost or gap:** `fdata_to_pc_1d` uses Simpson-weighted FPCA without any derivative-penalty regularizer. scikit-fda's `FPCA` supports `LinearDifferentialOperator` regularization — the generalized eigenvalue problem (K·w = λ·(M + αP)·w, where P is the penalty matrix). This is important for noisy functional data where unregularized FPCA loadings overfit high-frequency noise. Category: table-stakes.

- **Root cause:** Regularized FPCA requires solving a generalized eigenvalue problem instead of a standard eigenvalue problem. The penalty matrix (`bspline_penalty_matrix` in `smooth_basis.rs`) is already implemented; the generalized-eigenvalue solver path is the missing piece. The `linalg` feature adds `faer`; the Cholesky factorization of (M + αP) followed by a transformed standard eigenvalue problem is the standard approach.

- **Proposed direction:** Add `fdata_to_pc_1d_regularized(data, argvals, ncomp, basis_type, lambda) -> Result<FpcaResult, FdarError>` that: (1) computes the B-spline or Fourier penalty matrix P via `smooth_basis::bspline_penalty_matrix`; (2) forms M + λP (adding penalty to the integration weight matrix); (3) solves the generalized eigenvalue problem K·w = λ·(M + λP)·w via Cholesky factorization of (M + λP) and transformation to a standard eigenvalue problem. GSD-ready as candidate Phase: "Implement LDO-penalized regularized FPCA in regression.rs."

- **Severity (P1/P2/P3):** **P1** — Regularized FPCA is table-stakes for noisy functional data. Unregularized FPCA overfits high-frequency noise when the grid is dense; regularization via a smoothness penalty is standard practice in FDA and is a core feature of scikit-fda's FPCA. Its absence means fdars users working with noisy data have no direct regularization path for FPCA.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. The penalty matrix is already implemented. Work involves: implementing the generalized eigenvalue solution (Cholesky + transform, or direct `faer` generalized eigensolver), connecting to `FpcaResult`, and adding tests validating that increasing `lambda` produces smoother loadings on noisy data.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Preprocessing Parity Table → `FPCA` row (verdict: partial, table-stakes — LDO regularization absent). Phase 8 SUMMARY: [09-01-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-01-SUMMARY.md). Penalty matrix infrastructure: `fdars-core/src/smooth_basis.rs:82` (`bspline_penalty_matrix`).

---

### PREP-07 — Implement functional variable-selection methods (MaximaHunting, RKHS, mRMR)

**Candidate requirement / phase phrasing:** "Implement a `variable_selection` module with `maxima_hunting`, `recursive_maxima_hunting`, `rkhs_variable_selection`, and `minimum_redundancy_maximum_relevance` functions — matching scikit-fda's `MaximaHunting`, `RecursiveMaximaHunting`, `RKHSVariableSelection`, and `MinimumRedundancyMaximumRelevance`."

- **Location / area:** New `fdars-core/src/variable_selection.rs` module (no current functional variable-selection module). scikit-fda area: `preprocessing.dim_reduction.variable_selection` module (all four methods).

- **Current cost or gap:** Four scikit-fda variable-selection methods are absent in fdars: `MaximaHunting` (iterative peak search on a relevance curve), `RecursiveMaximaHunting` (recursive variant with decorrelation), `RKHSVariableSelection` (kernel-based relevance measure), and `MinimumRedundancyMaximumRelevance` (mutual-information optimization). Category: all four are differentiator.

- **Root cause:** No functional variable-selection module in fdars. Each method is a distinct algorithm: maxima-hunting uses peak detection on the marginal correlation curve (internal peak finder in `seasonal::detect_threshold_crossings` is a related tool); RKHS uses kernel-based relevance (related to `covariance::CovKernel`); mRMR uses mutual information (new computation). No shared infrastructure to reuse — each is an independent implementation.

- **Proposed direction:** Add a `variable_selection.rs` module with: `maxima_hunting(data: &FdMatrix, y: &[f64], argvals: &[f64], max_features: usize) -> Result<Vec<usize>, FdarError>` (returns selected evaluation-point indices); similar signatures for the recursive, RKHS, and mRMR variants. GSD-ready as candidate Phase: "Implement MaximaHunting and RKHS functional variable-selection methods."

- **Severity (P1/P2/P3):** **P3** — All four methods are differentiator-category: useful for researchers wanting to identify the most informative time points in functional data, but not required for basic FDA workflows. Their absence does not block any common use case.

- **Effort estimate (S/M/L):** **L** — approximately 1–3 months. Four separate algorithms with no shared infrastructure to reuse. MaximaHunting is simplest (~100 lines including peak detection); mRMR requires mutual-information estimation which is a non-trivial standalone computation. Each method needs its own numerical stability analysis and test suite.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Preprocessing Parity Table → `MaximaHunting` / `RecursiveMaximaHunting` / `RKHSVariableSelection` / `MinimumRedundancyMaximumRelevance` rows (all verdict: absent, differentiator). Phase 8 SUMMARY: [09-01-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-01-SUMMARY.md).

---

### PREP-08 — Expose local_averages, occupation_measure, number_crossings as public feature APIs

**Candidate requirement / phase phrasing:** "Expose `local_averages`, `occupation_measure`, and `number_crossings` as public functions in a `feature_construction.rs` module (or `helpers.rs`), wrapping existing internal logic from `seasonal::detect_threshold_crossings` and `landmark::detect_zero_crossings`."

- **Location / area:** `fdars-core/src/helpers.rs` or new `fdars-core/src/feature_construction.rs` — internal crossing logic in `seasonal::detect_threshold_crossings` and `landmark::detect_zero_crossings`. scikit-fda area: `representation.extrapolation` and `preprocessing` modules (`LocalAveragesTransformer`, `OccupationMeasureTransformer`, `NumberCrossingsTransformer`).

- **Current cost or gap:** Three feature-construction transformers are absent as public APIs: `LocalAveragesTransformer` / `local_averages` (average curve value over specified intervals), `OccupationMeasureTransformer` / `occupation_measure` (proportion of time curve spends above a threshold), `NumberCrossingsTransformer` / `number_crossings` (count of threshold crossings). Crossing logic exists internally in `seasonal::detect_threshold_crossings` and `landmark::detect_zero_crossings` but is not a public feature API. Category: differentiator for all three.

- **Root cause:** Internal crossing logic is per-module private. Local averages and occupation measure are straightforward integral operations over fdars' `FdMatrix` (one pass each). Exposing them as public feature extractors requires wrapping in a new module or adding to `helpers.rs`. No new algorithm is required.

- **Proposed direction:** Add to `helpers.rs` or a new `feature_construction.rs`: `local_averages(data: &FdMatrix, argvals: &[f64], intervals: &[(f64, f64)]) -> FdMatrix` (returns per-curve averages over each interval); `occupation_measure(data: &FdMatrix, argvals: &[f64], level: f64) -> Vec<f64>` (fraction of argvals range where curve ≥ level); `number_crossings(data: &FdMatrix, argvals: &[f64], level: f64) -> Vec<usize>` (count level crossings per curve, wrapping the existing `detect_threshold_crossings`). GSD-ready as candidate Phase: "Expose local_averages, occupation_measure, number_crossings as public feature-extraction APIs."

- **Severity (P1/P2/P3):** **P3** — These are differentiator-category feature constructors: useful for users building scalar feature sets from functional data (e.g., for classification or regression after feature extraction), but not required for any core FDA algorithm.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. All three are wrapper functions over existing operations. `local_averages` is a trapezoidal integral over an interval; `occupation_measure` is a count divided by interval length; `number_crossings` wraps `detect_threshold_crossings`. Minimal new code (~50–80 lines total).

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Preprocessing Parity Table → `LocalAveragesTransformer` / `OccupationMeasureTransformer` / `NumberCrossingsTransformer` rows (all verdict: absent, differentiator). Phase 8 SUMMARY: [09-01-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-01-SUMMARY.md).

---

### PREP-09 — Implement diffusion-map manifold embedding for functional data

**Candidate requirement / phase phrasing:** "Implement `diffusion_map(data: &FdMatrix, argvals: &[f64], ncomp: usize, sigma: f64) -> Result<FdMatrix, FdarError>` in a new `manifold.rs` module, using fdars' existing L2/Lp distances (from `distance.rs`) as the pairwise kernel basis."

- **Location / area:** New `fdars-core/src/manifold.rs` (no current manifold-learning module). Dependencies: `fdars-core/src/distance.rs` (L2 pairwise distances), `fdars-core/src/regression.rs` (truncated eigendecomposition via `fdata_to_pc_1d`). scikit-fda area: `preprocessing.dim_reduction` module (`DiffusionMap`).

- **Current cost or gap:** No diffusion-map or manifold-learning embedding for functional data. Category: differentiator. The building blocks exist — pairwise distance computation (`distance.rs`), normalization via matrix ops, and truncated eigendecomposition (analogous to `fdata_to_pc_1d`) — but the diffusion-map step sequence is unimplemented.

- **Root cause:** Diffusion maps require: (1) compute pairwise kernel matrix K_ij = exp(−d(i,j)² / σ²) using `distance::lp_distance` or `distance::l2_distance`; (2) normalize to a Markov matrix (row normalization + symmetric normalization); (3) apply truncated eigendecomposition (analogous to SVD in `fdata_to_pc_1d`). All building blocks exist; the composition is the gap.

- **Proposed direction:** Add `diffusion_map(data, argvals, ncomp, sigma, n_steps) -> Result<DiffusionMapResult, FdarError>` where `DiffusionMapResult` carries the embedding coordinates (N×ncomp) and eigenvalues. GSD-ready as candidate Phase: "Implement DiffusionMap manifold embedding using fdars' existing distance infrastructure."

- **Severity (P1/P2/P3):** **P3** — Diffusion maps are a differentiator-category capability: powerful for nonlinear dimensionality reduction of functional data, but not required for basic FDA workflows. Users who need manifold-learning for functional data currently have no fdars path; they must convert to a feature-vector representation first.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. Building the pairwise kernel matrix (O(N²·m) distance evaluations — feasible for N ≤ 500 with existing distance infrastructure) and the Markov normalization (~40 lines). Eigendecomposition reuses `fdata_to_pc_1d`'s SVD path but on the normalized kernel matrix (a symmetric dense matrix). Testing: verify on synthetic manifold data (e.g., Swiss roll projected into functional form).

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Preprocessing Parity Table → `DiffusionMap` row (verdict: absent, differentiator). Phase 8 SUMMARY: [09-01-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-01-SUMMARY.md).

---

### REPR-01 — Add MonomialBasis/ConstantBasis (and advanced: TensorBasis/FEBasis)

**Candidate requirement / phase phrasing:** "Add `MonomialBasis` (polynomial) and `ConstantBasis` (intercept) to `fdars-core/src/basis/`, and expose the internal tensor-product logic from `function_on_scalar_2d.rs` as a public `TensorBasis` API. `FiniteElementBasis` (irregular meshes) is a longer-term large-effort item."

- **Location / area:** `fdars-core/src/basis/` — currently exposes B-spline and Fourier only. Internal tensor-product logic in `function_on_scalar_2d.rs`. scikit-fda area: `representation.basis` module (`MonomialBasis`, `ConstantBasis`, `FiniteElementBasis`, `VectorValuedBasis`, `TensorBasis`, `CustomBasis`).

- **Current cost or gap:** Missing basis types: `MonomialBasis` (polynomial), `ConstantBasis` (intercept-only), `FiniteElementBasis` (irregular meshes), `VectorValuedBasis` (multivariate output), `TensorBasis` (multivariate domain, tensor product of 1D bases), `CustomBasis` (user-supplied function set). Only B-spline and Fourier are publicly exposed. Categories: MonomialBasis/ConstantBasis = table-stakes; TensorBasis/FiniteElementBasis/VectorValuedBasis/CustomBasis = differentiator.

- **Root cause:** `basis/` only exposes B-spline and Fourier constructors. The internal tensor-product logic exists in `function_on_scalar_2d.rs` (2D FOSR) but is not a public `TensorBasis` API. Adding `MonomialBasis` and `ConstantBasis` is low-cost (simple polynomial evaluation). `FiniteElementBasis` requires a mesh data structure and is high-cost.

- **Proposed direction:** (a) Add `MonomialBasis { n_basis: usize, domain_range: (f64, f64) }` and `ConstantBasis { domain_range: (f64, f64) }` to `basis/mod.rs` with polynomial evaluation kernels. (b) Extract the tensor-product logic from `function_on_scalar_2d.rs` into a public `TensorBasis { basis1, basis2 }` type. (c) Defer `FiniteElementBasis` (requires mesh data structure) to a separate large-effort item. GSD-ready as candidate Phase: "Add MonomialBasis, ConstantBasis, and TensorBasis to fdars basis module."

- **Severity (P1/P2/P3):** **P2** — `MonomialBasis` and `ConstantBasis` are table-stakes (widely used in functional regression as the simplest basis systems); their absence forces users to approximate polynomial bases with B-splines. TensorBasis is needed for 2D domain functional data. FEBasis is differentiator.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks for MonomialBasis + ConstantBasis + TensorBasis extraction. `FiniteElementBasis` (mesh structure) is excluded from this estimate.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Representation Parity Table → `MonomialBasis` / `ConstantBasis` / `TensorBasis` rows (table-stakes/differentiator, absent/partial). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### REPR-02 — Implement spline (cubic/order-k) interpolation at off-grid points

**Candidate requirement / phase phrasing:** "Implement `spline_interpolate(data: &FdMatrix, argvals: &[f64], query_points: &[f64], order: usize) -> Result<FdMatrix, FdarError>` in `helpers.rs`, using the B-spline basis already in `basis/` for de Boor evaluation of stored coefficients at arbitrary query points."

- **Location / area:** `fdars-core/src/helpers.rs` (`fdata_interpolate`, `linear_interp` — currently linear-only interpolation). `fdars-core/src/basis/` — B-spline basis evaluation already implemented. scikit-fda area: `representation` module (`SplineInterpolation` — cubic or order-k spline interpolation at arbitrary off-grid evaluation points).

- **Current cost or gap:** `helpers::fdata_interpolate` and `helpers::linear_interp` provide only linear interpolation. scikit-fda's `SplineInterpolation` provides spline (cubic or order-k) interpolation at arbitrary off-grid evaluation points — standard for functional data resampling and evaluation. Category: table-stakes (spline interpolation is the standard functional data evaluation method; linear interpolation produces visible artefacts for smooth curves).

- **Root cause:** B-spline evaluation at arbitrary query points requires computing the de Boor algorithm on the existing knot grid. The B-spline basis in `basis/` can already evaluate basis functions; composing this with stored coefficients for interpolation is the missing step. The B-spline basis system is already present — the "interpolate at query points" wrapper is absent.

- **Proposed direction:** Add `spline_interpolate(data: &FdMatrix, argvals: &[f64], query_points: &[f64], order: usize) -> Result<FdMatrix, FdarError>` that: (1) fits a B-spline of the given order to each curve's `(argvals, data_row)` pair (using `basis::bspline_fit` or equivalent); (2) evaluates at `query_points` using the de Boor algorithm. GSD-ready as candidate Phase: "Implement cubic/order-k spline interpolation at arbitrary off-grid query points."

- **Severity (P1/P2/P3):** **P1** — Spline interpolation at off-grid points is table-stakes for functional data evaluation: resampling curves to a common grid, evaluating at query points for prediction, and numerical integration all require smooth interpolation. Linear interpolation produces visible kinks on smooth curves; spline interpolation is the standard method. Its absence is a meaningful workflow gap.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. The B-spline basis is already implemented in `basis/`. The new function fits a B-spline to each row and evaluates at query points using the existing B-spline evaluation kernel. Total new code ~80–120 lines.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Representation Parity Table → `SplineInterpolation` row (verdict: partial, table-stakes — only linear interpolation available). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### REPR-03 — Add composable extrapolation-policy enum (Boundary/Exception/Fill/Periodic)

**Candidate requirement / phase phrasing:** "Add an `ExtrapolationPolicy` enum (`Boundary` / `Exception` / `Fill(f64)` / `Periodic`) to `helpers.rs` and thread it through `fdata_interpolate` and `fdata_to_basis` so users can control out-of-range evaluation behavior."

- **Location / area:** `fdars-core/src/helpers.rs` — `fdata_interpolate` (currently silently clamps to grid boundary). scikit-fda area: `representation.extrapolation` module (`BoundaryExtrapolation`, `ExceptionExtrapolation`, `FillExtrapolation`, `PeriodicExtrapolation`).

- **Current cost or gap:** No named extrapolation-policy objects. `fdata_interpolate` silently clamps to the grid boundary; there is no composable extrapolation-policy type. Named policy objects: `BoundaryExtrapolation` (clamp to boundary value), `ExceptionExtrapolation` (raise error on out-of-range query), `FillExtrapolation` (constant fill value), `PeriodicExtrapolation` (periodic wrap). Categories: BoundaryExtrapolation/ExceptionExtrapolation/FillExtrapolation = table-stakes; PeriodicExtrapolation = differentiator.

- **Root cause:** `fdata_interpolate` silently clamps to the grid boundary; there is no composable extrapolation-policy type. Implementing these as a Rust enum (`ExtrapolationPolicy`) passed to the interpolation/evaluation functions is low-cost; the policy dispatch logic is a small addition to `helpers.rs`.

- **Proposed direction:** Add `ExtrapolationPolicy` enum: `Boundary` (clamp to last known value), `Error` (return `Err(FdarError::InvalidParameter)` on out-of-range query), `Fill(f64)` (return constant fill value), `Periodic` (wrap query point modulo the domain range). Thread through `fdata_interpolate(data, argvals, query_points, policy: ExtrapolationPolicy)`. GSD-ready as candidate Phase: "Add ExtrapolationPolicy enum to fdars interpolation/evaluation path."

- **Severity (P1/P2/P3):** **P2** — Extrapolation policies are table-stakes for users who evaluate functional curves outside the observation range (common in prediction and visualization). Silent boundary clamping is a correctness footgun: users may not notice they are receiving boundary values for out-of-range queries. Making the behavior explicit is a meaningful correctness improvement.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. Small enum + match dispatch at the boundary-check point in `fdata_interpolate`. No new algorithm.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Representation Parity Table → `BoundaryExtrapolation` / `ExceptionExtrapolation` / `FillExtrapolation` / `PeriodicExtrapolation` rows (table-stakes/differentiator, absent). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### REPR-04 — Implement EM and Minimize mixed-effects irregular-to-basis converters

**Candidate requirement / phase phrasing:** "Implement `em_mixed_effects_converter` and `minimize_mixed_effects_converter` for converting `IrregFdata` to a basis representation via functional mixed-effects models — scikit-fda's `EMMixedEffectsConverter` and `MinimizeMixedEffectsConverter`."

- **Location / area:** `fdars-core/src/irreg_fdata/` (existing irregular→regular conversion via kernel smooth) and `fdars-core/src/famm.rs` (ANOVA mixed models — related but distinct). scikit-fda area: `preprocessing.smoothing` module (`EMMixedEffectsConverter`, `MinimizeMixedEffectsConverter`).

- **Current cost or gap:** Both scikit-fda mixed-effects converters are absent. `MinimizeMixedEffectsConverter` converts `FDataIrregular` to `FDataBasis` by minimizing a mixed-effects criterion (MAP estimate of basis coefficients given random-effects prior). `EMMixedEffectsConverter` uses the EM algorithm alternating between E-step (posterior scores) and M-step (basis coefficient update). A two-step workaround (irreg→grid→basis) is possible but not statistically equivalent. Category: differentiator.

- **Root cause:** Requires a functional mixed-effects solver: each curve is modeled as a random effect plus a fixed-effect basis expansion. The EM variant alternates between E-step (posterior coefficient scores) and M-step (basis coefficient + variance update). `famm.rs` handles a related but distinct model (ANOVA mixed models with fixed effects, not basis-conversion EM). No shared infrastructure — this is a new solver.

- **Proposed direction:** Implement `minimize_mixed_effects_converter(irreg_data: &IrregFdata, basis: &BsplineBasis, lambda: f64) -> Result<FdMatrix, FdarError>` using iterative optimization (argmin solver already in fdars' dependency tree). Implement `em_mixed_effects_converter(irreg_data: &IrregFdata, basis: &BsplineBasis, max_iter: usize) -> Result<EmConverterResult, FdarError>`. GSD-ready as candidate Phase: "Implement EM and Minimize irregular-to-basis converters for IrregFdata."

- **Severity (P1/P2/P3):** **P3** — Differentiator category: these converters are statistically superior to the two-step workaround for irregularly sampled data, but the workaround is available. Users with highly irregular sampling patterns (e.g., clinical trial data with patient-specific observation times) would benefit most.

- **Effort estimate (S/M/L):** **L** — approximately 1–3 months. The EM variant requires implementing the E-step (closed-form posterior given Gaussian assumptions) and M-step (basis coefficient update via ridge-regression structure). Requires careful numerical stability analysis (covariance estimation for the random-effects prior can be ill-conditioned with sparse data).

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Representation Parity Table → `EMMixedEffectsConverter` / `MinimizeMixedEffectsConverter` rows (verdict: absent, differentiator). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### EXPL-01 — Add pluggable-metric depth (DistanceBased) and OutlyingnessBased combinator

**Candidate requirement / phase phrasing:** "Add a `distance_based_depth(data: &FdMatrix, argvals: &[f64], distance_fn: &dyn Fn(&[f64], &[f64]) -> f64) -> Result<Vec<f64>, FdarError>` function to `depth/` enabling user-supplied distance metrics, and an `outlyingness_based_depth` wrapper computing depth = 1/(1 + outlyingness)."

- **Location / area:** `fdars-core/src/depth/` — `functional_spatial_1d` (hard-wired to L2/kernel variants). scikit-fda area: `exploratory.depth` module (`DistanceBasedDepth`, `OutlyingnessBasedDepth`, `SimplicialDepth` exact).

- **Current cost or gap:** `depth::functional_spatial_1d` is hard-wired to L2/kernel variants; it is not parameterizable by an arbitrary user-supplied metric (`DistanceBasedDepth` gap). No `OutlyingnessBasedDepth` combinator wrapping any outlyingness measure into depth = 1/(1+outlyingness). `SimplicialDepth` exact (combinatorial) is absent (fdars has random-Tukey approximation only). Category: DistanceBasedDepth = table-stakes; OutlyingnessBasedDepth/SimplicialDepth-exact = differentiator.

- **Root cause:** `depth/` uses concrete distance functions. Adding a trait parameter (`DistanceFn: Fn(&[f64], &[f64]) -> f64`) to a new `distance_based_depth` function would enable pluggable metrics without a new algorithm. The outlyingness combinator is a formula wrapper (no algorithm). Exact simplicial depth is combinatorially O(n^d) and impractical for d > 2; the approximation is already present.

- **Proposed direction:** Add two functions to `depth/`: (1) `distance_based_depth(data, argvals, distance_fn: impl Fn(&[f64], &[f64]) -> f64) -> Result<Vec<f64>, FdarError>` — computes per-curve depth using the supplied distance metric (L1, L∞, or any user function). (2) `outlyingness_based_depth(outlyingness: &[f64]) -> Vec<f64>` — applies depth = 1/(1+outlyingness) pointwise. GSD-ready as candidate Phase: "Add DistanceBased and OutlyingnessBased depth combinators to fdars depth module."

- **Severity (P1/P2/P3):** **P2** — `DistanceBasedDepth` is table-stakes for users who want to compute depth under non-L2 metrics (e.g., DTW distance, elastic distance). The hard-wired L2 variant covers many use cases, but pluggable metrics are a meaningful extension. `OutlyingnessBasedDepth` is a differentiator combinator.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. `distance_based_depth` requires computing pairwise distances under a user-supplied metric (O(n²) calls to the closure) and summing per-curve depth scores. The generic closure parameter adds complexity to the function signature (lifetime + trait bound). Tests: verify against L2 baseline (should match `functional_spatial_1d` with L2 distance).

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Exploratory Parity Table → `DistanceBasedDepth` / `OutlyingnessBasedDepth` rows (verdict: absent, table-stakes/differentiator). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### EXPL-02 — Add functional summary statistics: trim_mean, depth_median, cov, var, std

**Candidate requirement / phase phrasing:** "Add `trim_mean`, `depth_based_median`, `functional_covariance`, `functional_variance`, and `functional_std` to `fdata.rs` or a new `summary.rs` module — filling the table-stakes descriptive-statistics gap vs scikit-fda's `trim_mean`, `depth_based_median`, `cov`, `var`, `std`."

- **Location / area:** `fdars-core/src/fdata.rs` — currently has `functional_mean` and `geometric_median`. `fdars-core/src/covariance.rs` — kernel-based GP covariance (not the sample covariance of a regular-grid dataset). scikit-fda area: `exploratory.stats` module (`trim_mean`, `depth_based_median`, `cov`, `var`, `std`).

- **Current cost or gap:** Missing functional descriptive statistics: `trim_mean` (trimmed mean — exclude deepest-α or shallowest-α curves), `depth_based_median` (depth-weighted median — the curve maximizing depth in the reference set), functional `cov` (full sample covariance function — an N×N or M×M matrix from the observed curves, not a parametric kernel), `var` (functional variance — pointwise variance across curves), `std` (functional standard deviation — pointwise std). Category: all table-stakes (these are standard descriptive statistics for any data, elevated to FDA).

- **Root cause:** `fdata.rs` has `functional_mean` and `geometric_median` but not trimmed-mean, depth-weighted-median, or pointwise variance/std. `covariance.rs` has kernel-based GP covariance but not the sample covariance matrix of a regular-grid dataset as a standalone function. These are straightforward numerical operations on `FdMatrix` — the missing piece is named public functions.

- **Proposed direction:** Add to `fdata.rs`: `trim_mean(data: &FdMatrix, argvals: &[f64], alpha: f64, depth_fn: DepthFn) -> Result<Vec<f64>, FdarError>` (exclude lowest-depth fraction α); `depth_based_median(data: &FdMatrix, argvals: &[f64], depth_fn: DepthFn) -> Result<usize, FdarError>` (index of deepest curve). Add to `covariance.rs` or new `summary.rs`: `functional_covariance(data: &FdMatrix, argvals: &[f64]) -> FdMatrix` (M×M sample covariance); `functional_variance(data: &FdMatrix) -> Vec<f64>` (pointwise variance); `functional_std(data: &FdMatrix) -> Vec<f64>` (pointwise std). GSD-ready as candidate Phase: "Add trim_mean, depth_median, functional cov/var/std to fdars exploratory module."

- **Severity (P1/P2/P3):** **P1** — These are table-stakes descriptive statistics for any FDA library. Variance, standard deviation, and sample covariance are the building blocks of functional data analysis; their absence means users must compute them outside fdars from the raw `FdMatrix`. Trimmed mean and depth-based median are standard robust statistics for functional data.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. `functional_variance` and `functional_std` are one-pass O(n·m) operations. `functional_covariance` is O(n·m²) — the outer product of centered curves. `depth_based_median` calls the existing depth functions and takes argmax. `trim_mean` calls depth functions and filters rows. No new algorithms.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Exploratory Parity Table → `trim_mean` / `depth_based_median` / `cov` / `var` / `std` rows (all verdict: absent, table-stakes). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### EXPL-03 — Implement Stahel-Donoho outlyingness for functional data

**Candidate requirement / phase phrasing:** "Implement `stahel_donoho_outlyingness(data: &FdMatrix, argvals: &[f64], n_projections: usize, seed: u64) -> Result<Vec<f64>, FdarError>` in `depth/` or a new `outliers.rs` module — scikit-fda's `StahelDonohoOutlyingness`."

- **Location / area:** New implementation in `fdars-core/src/depth/` or `fdars-core/src/outliers.rs`. Related existing: `outliers::magnitude_shape_outlyingness` (directional outlyingness for MS-plot, different method). scikit-fda area: `exploratory.outliers` module (`StahelDonohoOutlyingness`).

- **Current cost or gap:** `StahelDonohoOutlyingness` — projection-based outlyingness for functional data. fdars has `outliers::magnitude_shape_outlyingness` (directional outlyingness for MS-plot) and LRT outlyingness; Stahel-Donoho outlyingness uses random projection directions and max absolute-deviation scoring (a distinct method). Category: differentiator.

- **Root cause:** Stahel-Donoho outlyingness uses random projection directions and max absolute-deviation scoring. It is distinct from fdars' current methods and would require a new implementation: sample `n_projections` random unit directions in function space, project each curve onto each direction, compute per-curve outlyingness as the max normalized absolute deviation across all projections.

- **Proposed direction:** Implement `stahel_donoho_outlyingness(data: &FdMatrix, argvals: &[f64], n_projections: usize, seed: u64) -> Result<Vec<f64>, FdarError>`. Steps: (1) sample `n_projections` random unit L2 directions in R^m (using existing RNG infrastructure); (2) project each curve onto each direction (`data_i · direction_j` — a dot product); (3) for each direction, compute median and MAD of the projections; (4) per-curve outlyingness = max over directions of `|proj_ij − median_j| / MAD_j`. GSD-ready as candidate Phase: "Implement Stahel-Donoho outlyingness for fdars."

- **Severity (P1/P2/P3):** **P3** — Differentiator category: Stahel-Donoho outlyingness is a robust alternative to MS-plot outlyingness for users who want to detect outliers robust to the choice of projection direction. Its absence is acceptable for most workflows (fdars already has MS-plot and LRT outlyingness).

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. Random projection generation is straightforward (using existing `rand` infrastructure); the per-direction MAD computation is O(n·m) per direction. Integration tests: verify that the method detects synthetic outliers in simulated functional data.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Exploratory Parity Table → `StahelDonohoOutlyingness` row (verdict: absent, differentiator). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### ML-01 — Add MaximumDepthClassifier, NearestCentroid, RadiusNeighbors variants

**Candidate requirement / phase phrasing:** "Add `maximum_depth_classifier` (argmax over per-class depth measures), `nearest_centroid_classifier` (assign to class with nearest functional mean), and `radius_neighbors_classifier` / `radius_neighbors_regressor` (kNN with distance threshold ε) to `fdars-core/src/classification/`."

- **Location / area:** `fdars-core/src/classification/` — existing classifiers: LDA, QDA, kNN, kernel, DD. scikit-fda area: `ml.classification` module (`MaximumDepthClassifier`, `NearestCentroid`, `RadiusNeighborsClassifier`, `DTMClassifier`, `DDGClassifier`) and `ml.regression` (`RadiusNeighborsRegressor`).

- **Current cost or gap:** `MaximumDepthClassifier` (classify by maximum depth under each class's empirical depth measure) and `NearestCentroid` as a named nearest-centroid classifier are absent. `RadiusNeighborsClassifier` and `RadiusNeighborsRegressor` (classify/regress by all neighbors within radius ε) absent. `DTMClassifier` (distance-to-measure) and `DDGClassifier` (DD-plot generalized) absent. `NearestNeighbors` index (general structure for neighbor queries) absent. Category: MaximumDepthClassifier/NearestCentroid = table-stakes; Radius/DTM/DDG/NearestNeighbors = differentiator.

- **Root cause:** `MaximumDepthClassifier` is a thin wrapper over `depth/` (already present): fit computes per-class depth measures; predict returns argmax. `NearestCentroid` is also thin over `fdata.rs::functional_mean`. `RadiusNeighbors*` requires a threshold variant of fdars' existing kNN infrastructure. `DTMClassifier` / `DDGClassifier` are more advanced and require distance-to-measure computation and DD-plot projection respectively.

- **Proposed direction:** (a) Add `maximum_depth_classifier_fit(data, labels, argvals, depth_method)` + `maximum_depth_classifier_predict(model, new_data)` to `classification/`. (b) Add `nearest_centroid_fit(data, labels, argvals)` + `nearest_centroid_predict(model, new_data)`. (c) Add `radius_neighbors_classifier_fit(data, labels, argvals, radius)` + `radius_neighbors_classifier_predict(model, new_data, radius)` (threshold variant of kNN). Defer DTM/DDG to a follow-on phase. GSD-ready as candidate Phase: "Add MaximumDepthClassifier, NearestCentroid, and RadiusNeighbors classifiers to fdars classification module."

- **Severity (P1/P2/P3):** **P2** — MaximumDepthClassifier and NearestCentroid are table-stakes classifiers expected in any FDA toolkit; their thin-wrapper nature (over existing depth and mean infrastructure) makes them low-effort additions. RadiusNeighbors is a differentiator classifier.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks for MaximumDepthClassifier + NearestCentroid + RadiusNeighbors. DTM/DDG deferred to separate items.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → ML Parity Table → `MaximumDepthClassifier` / `NearestCentroid` / `RadiusNeighborsClassifier` / `RadiusNeighborsRegressor` rows (absent, table-stakes/differentiator). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### ML-02 — Implement LDO-regularized linear regression + HistoricalLinearRegression

**Candidate requirement / phase phrasing:** "Add `fregre_lm_ldo(data, y, argvals, basis_type, lambda) -> Result<FregreLmResult, FdarError>` (LDO-penalized functional linear regression) and `historical_linear_regression(data, y, argvals) -> Result<HistoricalLmResult, FdarError>` — scikit-fda's LDO `LinearRegression` and `HistoricalLinearRegression`."

- **Location / area:** `fdars-core/src/scalar_on_function/` — `fregre_lm.rs` (unregularized FPCA-based regression). `fdars-core/src/smooth_basis.rs` (`bspline_penalty_matrix` — penalty matrix infrastructure). scikit-fda area: `ml.regression` module (`LinearRegression` with `LinearDifferentialOperator` and `HistoricalLinearRegression`).

- **Current cost or gap:** scikit-fda's `LinearRegression` with `LinearDifferentialOperator` regularization (unified LDO form) is partially matched by `fregre_lm` but the LDO-penalty variant is absent. `HistoricalLinearRegression` (function-on-function regression where future values predict past values via historical kernel integral) is absent. `RadiusNeighborsRegressor` also absent. Category: LDO-LinearRegression = table-stakes; HistoricalLinearRegression/RadiusNeighbors = differentiator.

- **Root cause:** LDO-regularized regression requires the same penalty matrix from `smooth_basis.rs` (already present) folded into the regression normal equations — analogous to PREP-06 (LDO-FPCA). The penalty matrix P is available; adding it to the normal equations `(X'WX + λP)β = X'Wy` is the gap. `HistoricalLinearRegression` requires implementing the historical-integral kernel and its numerical quadrature (new algorithm).

- **Proposed direction:** (a) Add `fregre_lm_ldo(data, y, argvals, basis_type, lambda) -> Result<FregreLmResult, FdarError>` to `scalar_on_function/` using the existing penalty matrix infrastructure. (b) Implement `historical_linear_regression(data, y, argvals) -> Result<HistoricalLmResult, FdarError>` with the historical-integral kernel `∫₀ᵗ β(s,t)·x(s)ds`. GSD-ready as candidate Phase: "Add LDO-regularized functional regression and HistoricalLinearRegression to fdars."

- **Severity (P1/P2/P3):** **P2** — LDO-regularized regression is table-stakes for noisy functional data (same argument as PREP-06 LDO-FPCA — users need regularization). HistoricalLinearRegression is a differentiator (specialized model used in longitudinal FDA). Both are meaningful for common FDA regression workflows.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks for LDO-regression (penalty matrix folding into normal equations, same approach as PREP-06). HistoricalLinearRegression adds the historical-integral kernel (new algorithm, separate implementation) — could be a follow-on item.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → ML Parity Table → `LinearRegression` (LDO variant) row (verdict: partial, table-stakes — LDO regularization absent) and `HistoricalLinearRegression` row (verdict: absent, differentiator). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### INF-01 — Add asymptotic functional ANOVA V-statistic (oneway_anova + v_sample_stat)

**Candidate requirement / phase phrasing:** "Add `oneway_anova_asymptotic(groups: &[&FdMatrix], argvals: &[f64]) -> Result<AnovaResult, FdarError>` implementing the V-statistic asymptotic test — scikit-fda's `oneway_anova` with `v_sample_stat` and `v_asymptotic_stat`."

- **Location / area:** `fdars-core/src/function_on_scalar.rs` — `fanova` (permutation-based ANOVA already present, no asymptotic path). New module candidate: `fdars-core/src/inference.rs`. scikit-fda area: `inference` module (`oneway_anova`, `v_sample_stat`, `v_asymptotic_stat`).

- **Current cost or gap:** No asymptotic one-way functional ANOVA using the V-statistic. fdars' `fanova` tests group-mean differences via permutation only; the asymptotic V-statistic path is absent. scikit-fda provides the asymptotic distribution (V compared to χ² or F-approximation) as a standalone alternative to permutation testing. Category: all table-stakes (asymptotic tests are faster than permutation for large n; both methods are expected in a complete inference module).

- **Root cause:** fdars' `fanova` tests via permutation. The asymptotic V-statistic path requires: (1) compute V = ∑_{i<j} n_i·n_j·‖mean_i − mean_j‖²_L2 / (∑n_k)²; (2) compare to an asymptotic χ² approximation based on the covariance structure. The mean and L2-norm infrastructure is present (`fdata.rs`, `distance.rs`); only the V-statistic formula and its asymptotic approximation (using `statrs` for χ² distribution) are missing.

- **Proposed direction:** Add `v_sample_stat(groups: &[&FdMatrix], argvals: &[f64]) -> Result<f64, FdarError>` (compute V statistic) and `oneway_anova_asymptotic(groups, argvals) -> Result<AnovaResult, FdarError>` (compute V, approximate p-value via χ² distribution from `statrs`). GSD-ready as candidate Phase: "Add asymptotic V-statistic functional ANOVA to fdars inference module."

- **Severity (P1/P2/P3):** **P2** — Asymptotic tests are table-stakes for large datasets where permutation testing is computationally expensive. The permutation fallback (`fanova`) already exists; the asymptotic variant is a meaningful addition for practical scalability.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. V-statistic formula is O(k²·m) (k groups, m evaluation points — all in existing infrastructure). Asymptotic approximation requires the covariance eigenvalue estimation — the simplest version uses a χ² distribution with a moment-matched degree-of-freedom estimate (~30 lines using `statrs`).

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Inference Parity Table → `oneway_anova` / `v_sample_stat` / `v_asymptotic_stat` rows (all verdict: absent, table-stakes). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### INF-02 — Expose two-sample Hotelling T² as standalone inference function

**Candidate requirement / phase phrasing:** "Expose `hotelling_test_ind(group1: &FdMatrix, group2: &FdMatrix, argvals: &[f64]) -> Result<HotellingResult, FdarError>` as a standalone two-independent-sample inference function — scikit-fda's `hotelling_test_ind`."

- **Location / area:** `fdars-core/src/spm/stats.rs` — `hotelling_t2` function (already computes Hotelling T² for SPM single-sample control-chart use). New thin wrapper candidate: `fdars-core/src/inference.rs`. scikit-fda area: `inference` module (`hotelling_test_ind`).

- **Current cost or gap:** `hotelling_test_ind` (two-independent-sample functional Hotelling T²) is absent as a public inference function. `spm::stats::hotelling_t2` exists but is designed for single-sample control-chart use (scores vs. control limits), not as a two-sample hypothesis test. Category: table-stakes (two-sample Hotelling T² is a standard multivariate hypothesis test widely used in FDA to compare two groups).

- **Root cause:** The Hotelling T² computation is already present in `spm/stats.rs`; wrapping it into a two-sample test requires: (1) pool covariance matrices from both groups; (2) apply degrees-of-freedom correction; (3) compute p-value via F-distribution (`statrs`). This is a thin `inference` module re-exporting the SPM statistic with two-sample semantics.

- **Proposed direction:** Add `hotelling_test_ind(group1: &FdMatrix, group2: &FdMatrix, argvals: &[f64]) -> Result<HotellingResult, FdarError>` to a new `inference.rs` module (or `function_on_scalar.rs`). Internally: compute group means and pooled covariance, call `spm::stats::hotelling_t2`-equivalent formula, compute p-value via F-distribution from `statrs`. GSD-ready as candidate Phase: "Add two-sample Hotelling T² test to fdars inference module."

- **Severity (P1/P2/P3):** **P2** — Two-sample Hotelling T² is table-stakes for comparing two groups of functional curves. It is the standard starting point for FDA group comparison (e.g., treated vs. control). Its absence forces users to reconstruct the test from SPM internals.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. The T² computation already exists in `spm/stats.rs`; the wrapper adds pooled-covariance pooling and F-distribution p-value via `statrs` (~40 lines). Tests: verify against scikit-fda `hotelling_test_ind` on synthetic two-group data.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Inference Parity Table → `hotelling_test_ind` row (verdict: absent, table-stakes). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### MISC-01 — Add Mahalanobis, NormInduced, Transformation metrics + angular/cosine functions

**Candidate requirement / phase phrasing:** "Add `mahalanobis_distance`, `norm_induced_metric`, `transformation_metric`, `angular_distance`, `cosine_similarity`, and `cosine_similarity_matrix` to `fdars-core/src/distance.rs` — filling the scikit-fda `MahalanobisDistance`, `NormInducedMetric`, `TransformationMetric`, `angular_distance`, `cosine_similarity`, `cosine_similarity_matrix` gap."

- **Location / area:** `fdars-core/src/distance.rs` — existing distance/metric functions (Lp, Hausdorff, DTW, Fisher-Rao, inner-product, amplitude, phase). `fdars-core/src/utility.rs` — related utilities. scikit-fda area: `misc.metrics` module (`MahalanobisDistance`, `NormInducedMetric`, `TransformationMetric`, `angular_distance`, `cosine_similarity`, `cosine_similarity_matrix`).

- **Current cost or gap:** `MahalanobisDistance`, `NormInducedMetric`, `TransformationMetric`, `angular_distance`, `cosine_similarity`, `cosine_similarity_matrix` all absent in fdars. Category: differentiator (these are alternative distance/similarity measures useful for specific analysis scenarios but not required for core FDA workflows).

- **Root cause:** `distance.rs` implements Lp, Hausdorff, DTW, Fisher-Rao, inner-product, amplitude, phase distances. Mahalanobis requires a covariance matrix (available from `covariance.rs` or `linalg.rs::mahalanobis`); `NormInducedMetric` and `TransformationMetric` are composable wrappers. Angular/cosine are derivable from inner products (present in `utility.rs`).

- **Proposed direction:** Add to `distance.rs`: `mahalanobis_distance(x, y, cov_inv: &FdMatrix) -> f64` (using existing matrix inverse); `angular_distance(x, y, argvals) -> f64` (= arccos(cosine_similarity)); `cosine_similarity(x, y, argvals) -> f64` (= inner_product / (norm(x)·norm(y))); `cosine_similarity_matrix(data, argvals) -> FdMatrix` (n×n cosine similarity matrix). GSD-ready as candidate Phase: "Add Mahalanobis, angular, and cosine distance/similarity functions to fdars distance module."

- **Severity (P1/P2/P3):** **P3** — All six are differentiator-category; Lp and Fisher-Rao distances (already present) cover the most common FDA distance use cases. These additions enable alternative similarity computations that some research applications require.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. All six are formula implementations over existing operations (inner products, norms already computed by existing distance functions). No new algorithms.

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Misc Parity Table → `MahalanobisDistance` / `NormInducedMetric` / `TransformationMetric` / `angular_distance` / `cosine_similarity` / `cosine_similarity_matrix` rows (all verdict: absent, differentiator). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### MISC-02 — Implement composable LinearDifferentialOperator and L2Regularization objects

**Candidate requirement / phase phrasing:** "Define a `DifferentialOperator` trait with a `penalty_matrix(basis, argvals) -> FdMatrix` method in a new `operator.rs` module, and provide `LinearDifferentialOperator { order: usize }` and `L2Regularization { lambda: f64 }` implementations — unblocking PREP-06 and ML-02 without code duplication."

- **Location / area:** New `fdars-core/src/operator.rs` module. `fdars-core/src/smooth_basis.rs` (`bspline_penalty_matrix` / `fourier_penalty_matrix` — existing penalty matrix implementations to wrap). scikit-fda area: `misc.operators` module (`LinearDifferentialOperator`, `L2Regularization`, `Identity`).

- **Current cost or gap:** `LinearDifferentialOperator` (LDO) composable object absent — the penalty matrix computation exists in `smooth_basis::bspline_penalty_matrix` / `fourier_penalty_matrix` but not as a composable `LinearDifferentialOperator` object that can be passed to any smoother, FPCA, or regression estimator. `L2Regularization` (scalar-weight ridge regularization) composable object absent. `Identity` operator composable object absent. Category: LDO/L2Reg = table-stakes (required by PREP-06, ML-02, REPR-03 to avoid code duplication).

- **Root cause:** fdars implements penalty matrices as standalone functions. Making them composable objects (a `DifferentialOperator` trait with `penalty_matrix()` method) would enable the LDO-FPCA (PREP-06) and LDO-regression (ML-02) paths to share the same operator abstraction without duplicating penalty-matrix logic. This is an API-ergonomics enhancement that also enables code reuse across three gap items.

- **Proposed direction:** Define `pub trait DifferentialOperator: Send + Sync { fn penalty_matrix(&self, basis: &BasisType, argvals: &[f64], n: usize) -> Result<FdMatrix, FdarError>; }`. Implement: `LinearDifferentialOperator { pub order: usize }` (calls `bspline_penalty_matrix` or `fourier_penalty_matrix` with the given derivative order), `L2Regularization { pub lambda: f64 }` (returns lambda·I), `IdentityOperator` (returns I). GSD-ready as candidate Phase: "Implement composable DifferentialOperator trait and LDO/L2Reg/Identity implementations as shared regularization abstraction."

- **Severity (P1/P2/P3):** **P2** — Table-stakes for code architecture: without this trait, PREP-06 (LDO-FPCA) and ML-02 (LDO-regression) would duplicate penalty-matrix logic. The trait is also required for scikit-fda-API-compatible LDO usage patterns. Meaningful architectural improvement that blocks two other gap items from being implemented cleanly.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks. Trait definition and three implementations are straightforward (~100 lines). The complexity is in threading the operator through PREP-06 and ML-02 function signatures (requires updating those APIs to accept `&dyn DifferentialOperator`).

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Misc Parity Table → `LinearDifferentialOperator` / `L2Regularization` / `Identity` rows (verdict: partial/absent, table-stakes). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### MISC-03 — Add make_gaussian wrapper and make_sinusoidal_process dedicated generator

**Candidate requirement / phase phrasing:** "Add `make_gaussian(n_samples, n_features, cov_kernel, mean, noise, seed) -> Result<FdMatrix, FdarError>` as a one-call GP generator wrapper and `make_sinusoidal_process(n_samples, n_features, amplitude_range, period, phase_std, noise, seed) -> Result<FdMatrix, FdarError>` as a dedicated sinusoidal functional-data generator."

- **Location / area:** `fdars-core/src/simulation.rs` — `sim_fundata` (KL expansion generator), `generate_gaussian_process` (GP trajectory generator). scikit-fda area: `datasets` module (`make_gaussian`, `make_sinusoidal_process`, `make_multimodal_samples`, `make_multimodal_landmarks`, `make_random_warping`, `make_sde_trajectories`).

- **Current cost or gap:** `make_gaussian` (exact Gaussian-process functional-data generator matching scikit-fda interface) is absent as a one-call wrapper. `make_sinusoidal_process` (sinusoidal functional data with amplitude/frequency/noise parameters) is absent as a dedicated generator — achievable via `sim_fundata` with Fourier eigenfunctions but not as a named function. `make_multimodal_samples`, `make_multimodal_landmarks`, `make_random_warping`, `make_sde_trajectories` also absent. Category: `make_gaussian` / `make_sinusoidal_process` = table-stakes (widely used for test data generation); multimodal/warping/SDE = differentiator.

- **Root cause:** `simulation.rs` generates Gaussian data via KL expansion (`sim_fundata`) and GP trajectories via `generate_gaussian_process`. The scikit-fda `make_gaussian` interface is a one-call wrapper with specific parameter semantics (n_samples, n_features, cov_kernel, noise); adapting the existing `generate_gaussian_process` to that interface is low-cost. Sinusoidal data is achievable via `sim_fundata` with Fourier eigenfunctions but requires a dedicated wrapper for a clean interface. Random warpings and SDE trajectories are new algorithms.

- **Proposed direction:** (a) Add `make_gaussian(n_samples, n_features, argvals, cov_kernel: &CovKernel, noise: f64, seed: u64) -> Result<FdMatrix, FdarError>` as a thin wrapper over `generate_gaussian_process`. (b) Add `make_sinusoidal_process(n_samples, n_features, argvals, amplitude_range: (f64, f64), period: f64, phase_std: f64, noise: f64, seed: u64) -> Result<FdMatrix, FdarError>`. Defer multimodal/warping/SDE generators to separate items. GSD-ready as candidate Phase: "Add make_gaussian and make_sinusoidal_process wrappers to fdars simulation module."

- **Severity (P1/P2/P3):** **P3** — The underlying functionality exists; these are convenience wrappers. Not a capability gap — users can call `generate_gaussian_process` or `sim_fundata` directly. Useful for documentation examples and test data generation parity with scikit-fda tutorials.

- **Effort estimate (S/M/L):** **M** — approximately 2–4 weeks including sinusoidal generator implementation (generating per-curve sinusoidal signals with random phase drawn from Normal(0, phase_std²) and optional Gaussian noise). `make_gaussian` itself is a one-day wrapper; `make_sinusoidal_process` requires the sinusoidal generation logic (~50 lines).

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Misc Parity Table → `make_gaussian` / `make_sinusoidal_process` rows (verdict: partial/absent, table-stakes). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

### MISC-04 — Add functional MAE, MSE scoring metrics (+ MAPE, MSLE, explained_variance)

**Candidate requirement / phase phrasing:** "Add `functional_mae`, `functional_mse`, `functional_mape`, `functional_msle`, and `functional_explained_variance` to `fdars-core/src/helpers.rs` or a new `scoring.rs` module — scikit-fda's `mean_absolute_error`, `mean_squared_error`, `mean_absolute_percentage_error`, `mean_squared_log_error`, `explained_variance_score`."

- **Location / area:** `fdars-core/src/helpers.rs` — existing scoring functions `r_squared`, `r_squared_adj`, and `cv::metric_r_squared`. New module candidate: `fdars-core/src/scoring.rs`. scikit-fda area: `misc.metrics` module (`mean_absolute_error`, `mean_squared_error`, `mean_absolute_percentage_error`, `mean_squared_log_error`, `explained_variance_score`).

- **Current cost or gap:** `mean_absolute_error` and `mean_squared_error` absent. `mean_absolute_percentage_error`, `mean_squared_log_error`, `explained_variance_score` absent. `helpers.rs` has `r_squared` and `r_squared_adj`; the standard regression scoring metrics (MAE, MSE) are not present as named functions. Category: MAE/MSE = table-stakes; MAPE/MSLE/explained-variance = differentiator.

- **Root cause:** `helpers.rs` has `r_squared` and `r_squared_adj`; MAE and MSE are equally straightforward (one-pass integral or pointwise average over residuals). Adding a `scoring.rs` module with `functional_mae`, `functional_mse`, `functional_mape`, `functional_msle`, `functional_explained_variance` is low algorithmic complexity — each is a one-pass formula over the residual matrix.

- **Proposed direction:** Add to `helpers.rs` or new `scoring.rs`: `functional_mae(y_true: &FdMatrix, y_pred: &FdMatrix, argvals: &[f64]) -> f64` (mean absolute error integrated over argvals); `functional_mse(y_true, y_pred, argvals) -> f64` (mean squared error); `functional_mape(y_true, y_pred, argvals) -> f64` (mean absolute percentage error); `functional_msle(y_true, y_pred, argvals) -> f64` (mean squared log error); `functional_explained_variance(y_true, y_pred, argvals) -> f64` (= 1 - Var(y_true - y_pred)/Var(y_true)). GSD-ready as candidate Phase: "Add MAE/MSE/MAPE/MSLE/explained_variance scoring metrics to fdars helpers or scoring module."

- **Severity (P1/P2/P3):** **P2** — MAE and MSE are table-stakes regression metrics universally expected in any predictive modeling library. Their absence from fdars is a meaningful gap for users evaluating functional regression model quality. MAPE/MSLE/explained_variance are differentiator additions that extend the scoring vocabulary.

- **Effort estimate (S/M/L):** **S** — approximately 1 week. Each metric is a one-pass formula over `FdMatrix` residuals (~5–10 lines per metric). Integration tests: verify MAE/MSE against manually computed values on synthetic data; verify R² consistency (MAE/MSE and R² should agree on relative model ranking).

- **Evidence link:** AUDIT-REPORT.md §Phase 8 → Misc Parity Table → `mean_absolute_error` / `mean_squared_error` / `mean_absolute_percentage_error` / `mean_squared_log_error` / `explained_variance_score` rows (verdict: absent, table-stakes/differentiator). Phase 8 SUMMARY: [09-02-SUMMARY.md](../phases/09-consolidated-report-prioritized-backlog/09-02-SUMMARY.md).

---

## Completeness Gate

This section documents the checklist every backlog item MUST pass before a plan is marked
complete, and records the three phase-level assertions Plan 03 will finalize over the full
item set.

### 7-Field Item Checklist

Every item under `## Backlog Items` must carry all seven of the following fields, each with
substantive content (not a placeholder):

1. **Location / area** — file, function, and/or module path; characterizes scope
2. **Current cost or gap** — a real measurement (benchmark number, allocation count) or a
   documented capability absence; no invented figures
3. **Root cause** — the algorithmic or architectural reason the cost or gap exists
4. **Proposed direction** — a concrete, GSD-ready candidate fix or feature description
5. **Severity (P1/P2/P3)** — severity classification with a brief rationale
6. **Effort estimate (S/M/L)** — effort classification with a brief rationale
7. **Evidence link** — a Markdown link to a real file under `.planning/research/bench/` or
   a phase SUMMARY / AUDIT-REPORT section; must be resolvable

### Tracer Item Status

**P6-1** passes the 7-field checklist as of Plan 01:
- Location / area: `regression.rs:298` (fdata_to_pc_1d) — present
- Current cost: 41.026 ms at N=500,M=200; 99.8–99.9% SVD share — present (real benchmark number)
- Root cause: nalgebra requires DMatrix allocation; always sequential — present
- Proposed direction: faer MatRef zero-copy + thin_svd, linalg-gated — present (GSD-ready wording)
- Severity: P2 with rationale (1.8× at primary cell, below 2× threshold) — present
- Effort: S (~1 week, faer already vendored, ~20 lines) — present
- Evidence link: two bench artifacts with real numbers linked — present

Computed score in Ranked Backlog: value=3, effort=S(1), score=3/sqrt(1)=**3.00** — present.

### Phase-Level Assertions — CONFIRMED (Plan 03)

The Ranked Backlog table is now fully sorted (32 items, Rank 1..32). The three phase-level
assertions are confirmed below.

---

#### Assertion 1: At least one P1 item exists — PASSED

**P1 items in the backlog (5 total):**

| Rank | ID | Title | Severity | Score |
|------|----|-------|----------|-------|
| 1 | REPR-02 | Implement spline (cubic/order-k) interpolation at off-grid points | **P1** | 4.00 |
| 2 | EXPL-02 | Add functional summary statistics: trim_mean, depth_median, cov, var, std | **P1** | 4.00 |
| 10 | PERF-ELASTIC-BAND | Default elastic alignment to a banded path / expose band_frac | **P1** | 2.89 |
| 11 | PREP-04 | Implement shift-only (LeastSquaresShift) registration | **P1** | 2.31 |
| 12 | PREP-06 | Implement derivative-penalty (LDO) regularized FPCA | **P1** | 2.31 |

**Rationale for P1 assignments:**
- **REPR-02 (P1):** Spline interpolation at off-grid points is table-stakes for functional data evaluation — resampling, prediction, and numerical integration all require smooth evaluation. Linear interpolation produces visible kinks on smooth curves. Its absence forces workarounds for any real evaluation workflow.
- **EXPL-02 (P1):** Functional summary statistics (variance, std, covariance, trimmed mean, depth median) are the building blocks of any FDA analysis. Their absence means users must compute them from raw `FdMatrix` outside fdars — a table-stakes gap.
- **PERF-ELASTIC-BAND (P1):** The default elastic alignment path makes N=500,M=200 distance matrices entirely infeasible (~700 s/iteration). Every caller of the default API pays this cost. Promoted to P1 in Plan 02 with explicit evidence and rationale.
- **PREP-04 (P1):** Shift-only registration is the entry-level alignment method (2–3 orders of magnitude faster than elastic). Its absence forces users who only need simple alignment to use the computationally expensive elastic path or implement it themselves.
- **PREP-06 (P1):** LDO-regularized FPCA is table-stakes for noisy functional data. Unregularized FPCA overfits high-frequency noise; regularization is standard practice in FDA and a core feature of scikit-fda's FPCA.

**Assertion 1 PASSED** — 5 P1 items present in the backlog.

---

#### Assertion 2: No top-10 item is cosmetic-only — PASSED

Top-10 review (Ranks 1–10), one line per item confirming non-cosmetic nature:

| Rank | ID | Severity | Non-cosmetic verdict |
|------|----|----------|----------------------|
| 1 | REPR-02 | P1 | **Real capability gap** — spline interpolation required for off-grid evaluation in prediction, resampling, and integration workflows. |
| 2 | EXPL-02 | P1 | **Real capability gap** — functional variance/std/covariance are table-stakes building blocks for any FDA analysis. |
| 3 | PERF-PAR-CV | P2 | **Real performance win** — ~4–5× speedup on a commonly repeated workflow (cross-validation hyperparameter search) via one-line macro substitution. |
| 4 | P6-1 | P2 | **Real performance win** — 1.8–4.1× SVD speedup measured at 7 real FPCA sizes; SVD is 99.8–99.9% of FPCA wall-clock. |
| 5 | PREP-03 | P2 | **Real capability gap** — missing-value imputation is table-stakes for datasets with sensor dropouts; no NaN-imputation entry point in fdars. |
| 6 | REPR-03 | P2 | **Real correctness issue** — silent boundary clamping on out-of-range queries is a footgun; named extrapolation policies make behavior explicit and prevent silent incorrect values. |
| 7 | INF-01 | P2 | **Real capability gap** — asymptotic V-statistic ANOVA is table-stakes for large-n group comparisons where permutation testing is computationally expensive. |
| 8 | INF-02 | P2 | **Real capability gap** — two-sample Hotelling T² is the standard starting point for functional group comparison (treated vs. control); its absence forces users to reconstruct the test from SPM internals. |
| 9 | MISC-04 | P2 | **Real capability gap** — MAE and MSE are universally expected regression metrics; their absence means users cannot evaluate functional regression model quality without computing them externally. |
| 10 | PERF-ELASTIC-BAND | P1 | **Real performance bottleneck** — N=500,M=200 elastic distance matrices are entirely infeasible (~700 s/iteration) on the default API; banded path already exists but is opt-in only. |

**No cosmetic convenience-only item found in Ranks 1–10.** All ten top items address: real capability gaps that block documented scikit-fda workflows (REPR-02, EXPL-02, PREP-03, REPR-03, INF-01, INF-02, MISC-04), real measured performance wins (PERF-PAR-CV, P6-1), or a real blocking production bottleneck (PERF-ELASTIC-BAND).

**Assertion 2 PASSED** — top-10 confirmed non-cosmetic.

---

#### Assertion 3: Ordering is strictly descending by score — PASSED

Score sequence (Rank 1..32):

```
4.00, 4.00, 4.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00,
2.89, 2.31, 2.31, 2.00, 2.00, 2.00, 2.00,
1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73, 1.73,
1.15, 1.15, 1.15, 1.00, 1.00, 0.67, 0.67, 0.58
```

Each score is ≤ the score of the row above it. The sequence is monotonically non-increasing.

Score verification: 4.00 → 4.00 → 4.00 → 3.00 → 3.00 → 3.00 → 3.00 → 3.00 → 3.00 → 2.89 → 2.31 → 2.31 → 2.00 → 2.00 → 2.00 → 2.00 → 1.73 → 1.73 → 1.73 → 1.73 → 1.73 → 1.73 → 1.73 → 1.73 → 1.15 → 1.15 → 1.15 → 1.00 → 1.00 → 0.67 → 0.67 → 0.58 — all steps are ≤ 0 (non-increasing). No inversions found.

**Assertion 3 PASSED** — ordering is strictly descending by score (non-increasing, confirmed across all 32 rows).

---

### Completeness Gate: ALL THREE ASSERTIONS PASSED

**Gate status: PASSED (Plan 03)**

- [x] At least one P1 item exists — 5 P1 items: REPR-02, EXPL-02, PERF-ELASTIC-BAND, PREP-04, PREP-06
- [x] No top-10 item is cosmetic-only — all 10 confirmed non-cosmetic (real gaps or real perf wins)
- [x] Ordering is strictly descending by score — 4.00 → … → 0.58, non-increasing confirmed

The backlog is promotion-ready for `/gsd-new-milestone`.
