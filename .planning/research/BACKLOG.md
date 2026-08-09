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

Items ordered by descending `score = value / sqrt(effort)`. Computed score shown in Score column.

| Rank | ID | Title | Severity | Value (1–5) | Effort (S/M/L) | Score (value/sqrt(effort)) | Area / Location |
|------|----|-------|----------|------------|----------------|---------------------------|-----------------|
| — | P6-1 | Swap nalgebra SVD for faer thin_svd in `fdata_to_pc_1d` | P2 | 3 | S | 3.00 | FPCA / `regression.rs:298` |
| — | PERF-ELASTIC-BAND | Default elastic alignment to a banded path / expose band_frac | P1 | 5 | M | 2.89 | Elastic alignment / `alignment/karcher.rs:300`, `elastic_self/cross_distance_matrix` |
| — | PERF-PAR-CV | Parallelize the classification CV fold loop | P2 | 4 | S | 4.00 | Classification CV / `classification/cv.rs:76` |
| — | PERF-FPCA-TRUNCSVD | Truncated/thin SVD computing only ncomp components in FPCA | P2 | 3 | L | 1.00 | FPCA / `regression.rs:298` via `nalgebra::SVD::new` |
| — | PERF-PAR-ELFPCA | Parallelize the three elastic-FPCA inner N-loops | P2 | 3 | M | 1.73 | Elastic FPCA / `elastic_fpca.rs:701/720/764` |
| — | PERF-PAR-CENTER | Parallelize center_columns on FPCA path | P3 | 1 | S | 1.00 | FPCA / `regression.rs:167` |
| — | PERF-FPCA-CLONE | Eliminate redundant centered.clone() + zero-copy to_dmatrix() bridge | P3 | 1 | M | 0.58 | FPCA / `regression.rs:291/298` |
| — | ACC-VALIDATE | Comparative fdars-vs-scikit-fda numerical-accuracy validation | P2 | 3 | M | 1.73 | Cross-cutting (Preprocessing / Misc / ML) |

*Rows appended by Plan 02. Final global sort (by descending score) deferred to Plan 03.*

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

### Phase-Level Assertions (Deferred to Plan 03)

The following three assertions require the full item set (all performance + gap backlog items)
and are explicitly deferred to Plan 03, which performs the final sort and completeness sweep:

1. **P1-existence:** At least one P1 item exists in the backlog. (Cannot be asserted with
   a single P6-1 tracer item; deferred until Plans 02/03 add all performance and gap items.)

2. **No top-10 cosmetic items:** No item in the top 10 ranked rows is a cosmetic
   convenience-only entry (i.e., all top-10 items affect correctness, performance on a real
   workload, or a documented scikit-fda capability gap). (Deferred to Plan 03 final-sort pass.)

3. **Descending-score order:** The `## Ranked Backlog` table rows are ordered by descending
   `score = value / sqrt(effort)` after the final sort. (Deferred to Plan 03; Plan 01 seeds
   one row and Plans 02/03 append rows before Plan 03 performs the final sort.)

Plan 03 will confirm all three phase-level assertions and mark the gate as PASSED or flag
any remaining open items.
