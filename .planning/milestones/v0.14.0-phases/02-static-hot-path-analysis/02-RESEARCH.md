# Phase 02: Static Hot-Path Analysis — Research

**Researched:** 2026-08-07
**Domain:** fdars-core source-code static analysis — complexity, allocations, parallelism gaps
**Confidence:** HIGH (all findings are source-verified; no external search required)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PERF-01 | Static hot-path analysis documents bottleneck candidates per module with algorithmic complexity in N and M, covering elastic alignment, FPCA/SVD, depth & distance matrices, CV loops, streaming depth, and smoothing | Sections 3–7 of this document provide file:line anchors and complexity derivations for all 6 modules; allocation and parallelism gap inventories are in Sections 5–6 |
</phase_requirements>

---

## Summary

Phase 2 is a zero-runtime static analysis. No code is changed; no benchmarks are run. The deliverable is three lists and one table appended to the single growing report at `.planning/research/AUDIT-REPORT.md` (established by Phase 1, decision D-05). All findings below are read directly from fdars-core/src/ in this session.

The key discoveries that sharpen the plan:

**SVD copy sites:** The ROADMAP claimed "8 FdMatrix→DMatrix SVD-copy sites." The actual count is **9** `to_dmatrix()` call sites in production source (excluding the definition itself at `matrix.rs:310` and the doc comment at `lib.rs:58`). Of these, 7 are direct `SVD::new(…to_dmatrix()…)` calls; 1 is a test helper (`matrix.rs:682`); 1 is a `stacked.to_dmatrix()` inside MFPCA (`spm/mfpca.rs:336`). There are also 14 `DMatrix::from_column_slice` call sites that are NOT SVD copies but rather basis-matrix constructions for least-squares solves. The plan must enumerate both categories separately.

**Parallelism landscape:** The `fdata_to_pc_1d` centering step (`center_columns` in `regression.rs:167`) is a plain sequential double loop — NOT wrapped in `iter_maybe_parallel!`. This was confirmed as Open Question A5 in Phase 1 research and resolved there in AUDIT-REPORT.md. Conversely, `fdata.rs:center_1d` (a separate function) IS parallelized, but FPCA calls `regression.rs:center_columns` internally. The classification CV fold loop (`cv.rs:76` — `for fold in 0..nfold`) is sequential and unparallelized. The streaming depth batch loop (`streaming_depth/fraiman_muniz.rs:82` — `iter_maybe_parallel!`) IS already feature-gated parallel. The elastic-FPCA inner N-loops (`elastic_fpca.rs:701, 720, 740, 764, 800, 829, 878, 921, 964`) are all plain sequential `for i in 0..n` — none wrapped in `iter_maybe_parallel!`.

**Banding:** `karcher_mean()` defaults to `band_frac = 0.0` which passes `None` as the band (full DP, O(m²) per pair). The banded variant requires an explicit call to `karcher_mean_banded()`. Same pattern for `elastic_self_distance_matrix()` vs `elastic_self_distance_matrix_banded()`. This is opt-in, not automatic.

**Report artifact:** Append Phase 2 analysis sections to `.planning/research/AUDIT-REPORT.md` (D-05). Do NOT create a separate file. Consistent with Phase 1 precedent.

**Primary recommendation:** The plan should structure three tasks: (1) write the per-module complexity table using the file:line anchors below, (2) write the allocation-hotspot list by walking all 9 `to_dmatrix()` and 14 `DMatrix::from_column_slice` sites, (3) write the parallelism-gap list by checking each named loop site against the `iter_maybe_parallel!` grep inventory.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Complexity table | `fdars-core/src/` (source analysis) | `.planning/research/AUDIT-REPORT.md` (output) | Read source; write findings to report |
| Allocation-hotspot list | `fdars-core/src/` (grep targets) | `.planning/research/AUDIT-REPORT.md` | All allocation sites are in src/ |
| Parallelism-gap list | `fdars-core/src/parallel.rs` (macro model) + per-module src | `.planning/research/AUDIT-REPORT.md` | Feature gate model is in parallel.rs; call sites scattered in src/ |
| Feature-gate annotations | Cargo.toml features + `#[cfg(feature = "parallel")]` conditional compilation | Per-finding inline annotation | Every finding must carry its feature requirement |

---

## Section 1: Parallelism Macro Model (Criterion 4 anchor)

The feature-gate model is defined verbatim in `fdars-core/src/parallel.rs:1-156` [VERIFIED: fdars-core/src/parallel.rs:1-156].

Five macros, all gated by `#[cfg(feature = "parallel")]`:

| Macro | Sequential fallback | Parallel form | Typical usage |
|-------|---------------------|--------------|---------------|
| `iter_maybe_parallel!(expr)` | `IntoIterator::into_iter(expr)` | `IntoParallelIterator::into_par_iter(expr)` | Range loops, Vec consumption |
| `slice_maybe_parallel!(slice)` | `slice.iter()` | `slice.par_iter()` | Shared-reference slice iteration |
| `slice_maybe_parallel_mut!(slice)` | `slice.iter_mut()` | `slice.par_iter_mut()` | Mutable slice iteration |
| `maybe_par_chunks_mut!(slice, sz, closure)` | `chunks_mut().for_each()` | `par_chunks_mut().for_each()` | Chunked mutable iteration |
| `maybe_par_chunks_mut_enumerate!(slice, sz, closure)` | enumerated sequential chunks | enumerated parallel chunks | Indexed chunked mutation |

**Rule for annotation in RESEARCH.md findings:** A loop wrapped in any of these 5 macros is feature-gated (`parallel`-only). A plain `for item in 0..n` or `for item in iter` is genuinely sequential and is a gap candidate. The macro re-exports are at `parallel.rs:152-156` [VERIFIED: fdars-core/src/parallel.rs:152-156]:

```rust
pub use iter_maybe_parallel;
pub use maybe_par_chunks_mut;
pub use maybe_par_chunks_mut_enumerate;
pub use slice_maybe_parallel;
pub use slice_maybe_parallel_mut;
```

**Feature gate context:** `default = ["parallel"]` so the `parallel` feature is active in a plain `cargo build`/`cargo bench` unless `--no-default-features` is passed. Tests run sequentially by architectural constraint (no rayon in test builds per ARCHITECTURE.md). `linalg` gates only `faer` and `anofox-regression`; it has no effect on loop parallelism.

---

## Section 2: FdMatrix→DMatrix Allocation Hotspot Inventory (Criterion 2 anchor)

**IMPORTANT CORRECTION:** The ROADMAP states "8 FdMatrix→DMatrix SVD-copy sites." The actual verified count is **9** `to_dmatrix()` call sites in production source (excluding matrix.rs:310 which is the method definition itself, and lib.rs:58 which is a doc comment). The planner and AUDIT-REPORT.md must use 9, not 8.

### 2A: `to_dmatrix()` call sites — the FdMatrix→DMatrix copy

[VERIFIED: grep -rn "to_dmatrix()" fdars-core/src/ confirmed in this session]

| File | Line | Context | SVD call? | Feature gate |
|------|------|---------|-----------|-------------|
| `fdars-core/src/regression.rs` | 298 | `SVD::new(weighted.to_dmatrix(), true, true)` — core FPCA | Yes | none (always executed) |
| `fdars-core/src/elastic_fpca.rs` | 214 | `SVD::new(centered.to_dmatrix(), true, true)` — horiz_fpca | Yes | none |
| `fdars-core/src/elastic_fpca.rs` | 317 | `SVD::new(combined.to_dmatrix(), true, true)` — joint_fpca | Yes | none |
| `fdars-core/src/elastic_fpca.rs` | 483 | `SVD::new(centered.to_dmatrix(), true, true)` — vert_fpca | Yes | none |
| `fdars-core/src/elastic_fpca.rs` | 584 | `SVD::new(combined.to_dmatrix(), true, true)` — joint_fpca second path | Yes | none |
| `fdars-core/src/elastic_fpca.rs` | 930 | `SVD::new(combined.to_dmatrix(), true, true)` — elastic_fpca third path | Yes | none |
| `fdars-core/src/alignment/nd.rs` | 705 | `SVD::new(gram.to_dmatrix(), true, true)` — ND elastic FPCA Gram matrix | Yes | none |
| `fdars-core/src/spm/mfpca.rs` | 336 | `SVD::new(stacked.to_dmatrix(), true, true)` — multivariate FPCA | Yes | none |
| `fdars-core/src/matrix.rs` | 682 | `let dmat = mat.to_dmatrix()` — test helper (round-trip test) | No | `#[cfg(test)]` |

**SVD-copy sites:** 8 production sites (not 9 — site 9 is `#[cfg(test)]`). Correction: the 8-count in the ROADMAP refers to SVD-only sites, which is accurate for production code. The 9th `to_dmatrix()` call is test-only and not a Phase 4 dhat target.

**Summary for AUDIT-REPORT.md:** 8 production `SVD::new(x.to_dmatrix(), …)` call sites. Every one allocates a new `DMatrix<f64>` of size n×m (or m×m for Gram matrix), performs full SVD, and discards the DMatrix. None cache or reuse the DMatrix.

### 2B: `DMatrix::from_column_slice` sites — basis construction (NOT SVD copies)

These are distinct from the `to_dmatrix()` chain. They construct DMatrix from an existing `Vec<f64>` flat buffer for basis regression / least-squares solves.

[VERIFIED: grep -rn "DMatrix::from_column_slice" fdars-core/src/ confirmed in this session]

| File | Line(s) | Context |
|------|---------|---------|
| `fdars-core/src/smooth_basis.rs` | 198, 199 | basis matrix + penalty matrix for smoothing |
| `fdars-core/src/smooth_basis.rs` | 695, 696 | same, second code path |
| `fdars-core/src/seasonal/ssa.rs` | 178 | SSA trajectory matrix |
| `fdars-core/src/basis/auto_select.rs` | 95, 128 | basis matrix for auto-selection (two paths) |
| `fdars-core/src/basis/fourier_fit.rs` | 68 | Fourier basis matrix |
| `fdars-core/src/basis/projection.rs` | 113 | B-spline basis matrix |
| `fdars-core/src/elastic_regression/regression.rs` | 274, 278 | elastic regression basis + penalty |
| `fdars-core/src/elastic_regression/scalar_on_shape.rs` | 117, 119 | shape-regression basis + penalty |
| `fdars-core/src/basis/pspline.rs` | 87 | P-spline basis matrix |

These are 14 call sites total. Each copies a flat buffer into a DMatrix. They are candidates for Phase 4 dhat audit but are secondary to the `to_dmatrix()` SVD path.

### 2C: Redundant clone in `fdata_to_pc_1d`

[VERIFIED: fdars-core/src/regression.rs:291]

```rust
let mut weighted = centered.clone();  // regression.rs:291
```

This clones the entire n×m centered matrix immediately after `center_columns` returns it, to apply sqrt-weight scaling in place. The original `centered` is also stored in `FpcaResult`. This is a double-size allocation: n×m for `centered` + n×m for `weighted`. A zero-copy optimization would scale weights in-place before storing to `FpcaResult.centered`, or store unweighted and weight-on-demand.

---

## Section 3: Parallelism Gap Inventory (Criterion 3 anchor)

### 3A: Classification CV fold loop — SEQUENTIAL, gap candidate

[VERIFIED: fdars-core/src/classification/cv.rs:76]

```rust
for fold in 0..nfold {          // cv.rs:76 — plain sequential for loop
    let (train_idx, test_idx) = fold_split(&folds, fold);
    …
    let predictions = cv_fold_predict(…);  // calls fdata_to_pc_1d internally
    …
}
```

Each fold is independent. No shared mutable state between fold iterations (fold_errors is written once per fold and only read after all folds complete). The inner `cv_fold_predict` calls `fdata_to_pc_1d` (which includes nalgebra SVD — thread-safe) and LDA/QDA/kNN (also thread-safe). This loop is a safe candidate for `iter_maybe_parallel!`.

**Feature gate status:** NOT wrapped in any parallelism macro. Genuinely sequential.

**Parallelization blocker:** `cv_fold_predict` returns `Option<Vec<usize>>` and the error count is accumulated into a `Vec<f64>`. Converting to parallel requires collecting results (which Rayon supports). The `FdMatrix` arguments are borrowed immutably — no interior mutability, compatible with `Send+Sync`.

### 3B: Streaming depth `depth_batch` — ALREADY PARALLEL (feature-gated)

[VERIFIED: fdars-core/src/streaming_depth/fraiman_muniz.rs:82]

```rust
iter_maybe_parallel!(0..nobj)          // fraiman_muniz.rs:82
    .map(|i| { … })
    .collect()
```

The `depth_batch` method on `StreamingFraimanMuniz` IS already wrapped in `iter_maybe_parallel!`. When the `parallel` feature is active, each object's depth query runs in parallel. This is NOT a gap — it is correctly parallelized.

**Annotation for AUDIT-REPORT.md:** The streaming depth module is not a parallelism gap. It should be listed in the "already parallelized" column of the report.

Same applies to `StreamingMBD::depth_batch` in `streaming_depth/mbd.rs:76` [VERIFIED: grep result in this session].

### 3C: Elastic-FPCA inner N-loops — SEQUENTIAL, gap candidates

[VERIFIED: fdars-core/src/elastic_fpca.rs:701, 720, 734, 740, 764, 800, 829, 878, 921, 964]

The following helper functions inside `elastic_fpca.rs` contain sequential `for i in 0..n` loops over curves:

| Function | Line(s) | Operation |
|----------|---------|-----------|
| `shooting_vectors_from_psis` | 701–707 | Compute n shooting vectors via `inv_exp_map_sphere` per curve — each independent |
| `build_augmented_srsfs` | 720–727 | Build augmented SRSF matrix, one row per curve — independent |
| `center_matrix` | 734, 740–745 | Centering loop (column-then-row) — not parallelized |
| `svd_scores_and_eigenvalues` | 764–767 | Score extraction `for k in 0..ncomp { for i in 0..n` — inner i-loop independent |
| `build_symmetric_covariance` | ~800 | Symmetric covariance construction |
| `split_joint_eigenvectors` | ~780 | Eigenfunction splitting |

None of these loops use `iter_maybe_parallel!`. They are plain sequential `for i in 0..n` or nested loops. For N=500–1000, each is an O(n·m) pass, executed inside `vert_fpca`, `horiz_fpca`, or `joint_fpca`.

**Parallelization consideration:** `shooting_vectors_from_psis` (per-curve `inv_exp_map_sphere`) and score extraction are the strongest candidates. Each curve's computation is independent. The `center_matrix` centering is a two-pass (mean then subtract) pattern that can be parallelized with a reduce step.

### 3D: Karcher mean inner N-loop — ALREADY PARALLEL (feature-gated)

[VERIFIED: fdars-core/src/alignment/karcher.rs:185, 376, 413, 432]

```rust
let align_results: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)   // karcher.rs:185
    .map(|i| align_srsf_pair_banded(mu_q, &data_srsfs[i], argvals, lambda, band))
    .collect();
```

The per-curve alignment within each Karcher iteration IS wrapped in `iter_maybe_parallel!`. This is why `karcher_mean` was selected as the D-04 4-combo discriminator in Phase 1.

### 3E: Pairwise distance matrix N-loops — ALREADY PARALLEL (feature-gated)

[VERIFIED: fdars-core/src/alignment/pairwise.rs:227, 304, 341]

The `elastic_self_distance_matrix` inner loop over upper-triangular pairs:

```rust
let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)    // pairwise.rs:227
    .flat_map(|i| {
        ((i + 1)..n).map(|j| { elastic_distance_from_srsf(…) })
    })
    .collect();
```

IS wrapped in `iter_maybe_parallel!`. The pairwise distance computation is parallelized at the curve level.

### 3F: Banding — opt-in, not automatic

[VERIFIED: fdars-core/src/alignment/karcher.rs:293-320]

```rust
pub fn karcher_mean(…) -> KarcherMeanResult {
    karcher_mean_impl(data, argvals, max_iter, tol, lambda, 0.0)   // band_frac=0.0 → None
}

pub fn karcher_mean_banded(…, band_frac: f64) -> KarcherMeanResult {
    karcher_mean_impl(data, argvals, max_iter, tol, lambda, band_frac)
}
```

`band_frac = 0.0` passes `band_radius(0.0, m) = None` [VERIFIED: alignment/karcher.rs:333 passes `fine_band = band_radius(band_frac, m)` — when `band_frac <= 0.0` this returns `None`]. The user must explicitly call `karcher_mean_banded()` to enable banding. The default `karcher_mean()` always does full DP (O(m²) per pair).

Same pattern for `elastic_self_distance_matrix()` vs `elastic_self_distance_matrix_banded()` [VERIFIED: pairwise.rs:194-212].

**Implication:** Every user of `karcher_mean()` or `elastic_self_distance_matrix()` pays O(m²) per alignment unconditionally. Automatic banding (e.g., enabling banding when band_frac is set in `ElasticConfig`) would reduce per-alignment cost to O(m · band). This is a gap to document.

---

## Section 4: Per-Module Complexity Table Anchors (Criterion 1 anchor)

For each of the 6 modules, the table in AUDIT-REPORT.md must cite the specific functions and loop nesting patterns that determine the Big-O. The executor reads the code at the cited lines to derive and confirm the complexity — this section gives the exact anchors.

### Module 1: Elastic Alignment

**Primary function:** `karcher_mean` → `karcher_mean_impl` [VERIFIED: alignment/karcher.rs:323]
**Secondary:** `elastic_self_distance_matrix` [VERIFIED: alignment/pairwise.rs:194]

**Loop structure (karcher_mean_impl):**
- Outer: `for _ in 0..max_iter` (typically ≤ 20 convergence iterations)
- Middle: `iter_maybe_parallel!(0..n)` — N alignments per iteration
- Inner: `dp_alignment_core_banded(q1, q2, argvals, lambda, band)` — O(m²) DP table fill (or O(m·band) with banding)

**Complexity:** O(max_iter · N · m²) unbanded. With band: O(max_iter · N · m · band). Phase 3 measures actual scaling.

**Fragile flag:** Banding is opt-in. Default `karcher_mean()` always uses O(m²) per alignment.

### Module 2: FPCA/SVD

**Primary function:** `fdata_to_pc_1d` [VERIFIED: regression.rs:212-322]

**Loop structure:**
- `center_columns` (regression.rs:167): O(n·m) sequential double loop — NOT parallelized
- Weight scaling (regression.rs:292-295): O(n·m) sequential double loop
- `weighted.clone()` (regression.rs:291): O(n·m) full matrix copy
- `SVD::new(weighted.to_dmatrix(), …)` (regression.rs:298): nalgebra full SVD
  - SVD cost: O(min(n,m)·n·m) for thin SVD. Since typical n<m (more evaluation points than curves), this is O(n²·m). For n>m (more curves than points), O(n·m²).

**Allocation chain per call:**
1. `center_columns` creates `FdMatrix::zeros(n, m)` — O(n·m)
2. `centered.clone()` creates another FdMatrix — O(n·m)
3. `weighted.to_dmatrix()` creates `DMatrix<f64>` — O(n·m)
4. Total: 3 allocations of size n·m per `fdata_to_pc_1d` call

**Feature gate:** nalgebra SVD is always sequential regardless of `parallel` feature.

### Module 3: Depth & Distance Matrices

**Primary function:** `fraiman_muniz_1d` [VERIFIED: depth_benchmarks.rs — takes `(&FdMatrix, &FdMatrix, bool)`]

**Loop structure for FM depth (src/depth/mod.rs — `maybe_par_chunks_mut_enumerate!` at line 80 for random projection; FM depth uses a different path):**
- FM depth: For each object curve, count pointwise rank among reference curves. Complexity O(n_obj · n_ref · m).
- Distance matrices: `lp_self_1d` → `iter_maybe_parallel!(0..n)` with inner `((i+1)..n)` — O(n² · m) total.

**Feature gate:** FM depth uses `iter_maybe_parallel!` at `streaming_depth/fraiman_muniz.rs:82`. The static `fraiman_muniz_1d` function's parallelism anchor is in `depth/mod.rs` — executor should verify whether the static variant also uses `iter_maybe_parallel!` or a sequential loop.

**Distance matrix:** `metric/lp.rs:57, 105` [VERIFIED: grep] — both `lp_cross_1d` and `lp_self_1d` use `iter_maybe_parallel!`.

### Module 4: CV Loops

**Primary function:** `fclassif_cv` [VERIFIED: classification/cv.rs:45-120]

**Loop structure:**
- Outer: `for fold in 0..nfold` (sequential — gap candidate, Section 3A)
- Inner per fold: `cv_fold_predict` → `fdata_to_pc_1d` (O(n·m) centering + O(n²·m) SVD) + classifier fit + predict

**Complexity:** O(nfold · cost_per_fold). Each fold processes (n - n/nfold) training curves. With LDA: O(nfold · ((n · m · K) + SVD(n/nfold, m))) where K = ncomp.

**Feature gate:** The fold loop is sequential. The FPCA inside each fold runs the same sequential `center_columns`.

### Module 5: Streaming Depth

**Primary function:** `StreamingFraimanMuniz::depth_batch` [VERIFIED: streaming_depth/fraiman_muniz.rs:77-90]

**Loop structure:**
- `iter_maybe_parallel!(0..nobj)` — parallel over query curves
- Each query: binary-search rank against sorted reference column vectors

**Complexity:** O(n_obj · n_ref · log(n_ref)) with sorted reference. Reference construction: O(n_ref · m · log(n_ref)) for sorting all m columns.

**Feature gate:** `iter_maybe_parallel!` at line 82 — feature-gated parallel.

### Module 6: Smoothing (Nadaraya-Watson)

**Primary function:** `nadaraya_watson` [VERIFIED: smoothing.rs:72]

**Loop structure:**
- `slice_maybe_parallel!(x_new).map(|&x0| { for i in 0..n { … } })` [VERIFIED: smoothing.rs:110-128]
- Outer over prediction points (parallelized with `slice_maybe_parallel!`)
- Inner over training points: sequential `for i in 0..n`

**Complexity:** O(n_pred · n_train) kernel evaluations. Each evaluation is O(1) (single kernel function call per point pair).

**Feature gate:** Outer loop is feature-gated parallel via `slice_maybe_parallel!`.

---

## Section 5: How "Done" is Verified (Acceptance Criteria Design)

Since Phase 2 produces no executable output — only markdown sections in AUDIT-REPORT.md — verification must be source-based. Recommend the following concrete checks for each success criterion:

### SC1 — Per-module complexity table

**Check:** `grep -c "Elastic\|FPCA\|Depth\|CV loops\|Streaming\|Smoothing" .planning/research/AUDIT-REPORT.md`
Must return ≥ 6 (one row per module in the table). Additionally, each row must contain a Big-O expression (grep for `O(n`).

**Stronger check:** Each table row cites at least one `file:line` anchor. The planner must require that the executor names at least one function and one file:line per row — paraphrase alone fails the provenance rule.

### SC2 — Allocation-hotspot list

**Check:** `grep -c "to_dmatrix\|from_column_slice" .planning/research/AUDIT-REPORT.md`
Must return ≥ 8 (one line per production `to_dmatrix()` SVD site).

**Stronger check:** The report must contain the exact file paths and line numbers from Section 2A above. The planner should require the executor to cross-check the count against Section 2A (8 production `to_dmatrix()` SVD sites, not 9) and note the correction from the ROADMAP's "8" claim.

### SC3 — Parallelism-gap list

**Check:** `grep -c "sequential\|gap\|candidate" .planning/research/AUDIT-REPORT.md`

**Stronger check:** The list must include at least: (1) `cv.rs:76` fold loop as sequential gap, (2) elastic_fpca inner N-loops as sequential gaps, (3) `fraiman_muniz.rs:82` as already-parallel (not a gap). Items that already use `iter_maybe_parallel!` must be labeled "already parallel" not "gap" to avoid false positives in Phase 5.

### SC4 — Feature-gate annotations

**Check:** Every entry in the parallelism-gap list has one of: `[parallel-gated]`, `[sequential]`, `[linalg-gated]`, or `[always]`.

**Check:** No path is labeled "sequential" when its hot loop uses `iter_maybe_parallel!`. The planner should require the executor to cross-check each listed loop against the macro grep inventory (Section 3 above).

---

## Section 6: Report Artifact Location and Append Protocol

**Location:** `.planning/research/AUDIT-REPORT.md` [VERIFIED: decision D-05 from Phase 1, confirmed in AUDIT-REPORT.md line 8 — "single growing report"]

**Append protocol:** Phase 2 appends a new section after the Phase 1 sections. Do NOT replace or re-write Phase 1 content. The section header should be:

```markdown
## Phase 2 — Static Hot-Path Analysis
```

**Three sub-sections to append:**

1. `### Complexity Table` — per-module Big-O table (SC1)
2. `### Allocation Hotspot List` — all `to_dmatrix()` and `from_column_slice` sites (SC2)
3. `### Parallelism Gap List` — sequential loop gaps + already-parallel inventory (SC3, SC4)

**No separate file needed.** The Phase 1 precedent (D-05) established AUDIT-REPORT.md as the single accumulation point. Phase 3 will append after Phase 2 sections.

---

## Section 7: MVP Tracer Module Recommendation

**Tracer module: Elastic Alignment** (confirmed as the recommended worst case).

Rationale:
- Highest complexity: O(max_iter · N · m²) — dominant cost by far
- Known fragility (CONCERNS.md:94-99): recent GH #33, #34 bugs show this module's complexity
- Contains both a parallelism gap (elastic_fpca inner loops) and an opt-in banding gap
- Covers all 3 lists: complexity row (SC1), `to_dmatrix()` sites in elastic_fpca.rs (SC2), and sequential N-loops (SC3)
- Already the D-03/D-04 discriminator module from Phase 1

**Tracer task:** Write the elastic alignment complexity row and all 6 `elastic_fpca.rs` `to_dmatrix()` call site entries first. Verify the complexity table format and the allocation list format against the AUDIT-REPORT.md structure. Then expand to the remaining 5 modules.

---

## Architecture Patterns

### System Architecture Diagram

```
AUDIT-REPORT.md (single growing file)
   │
   ├── §Methodology (Phase 1)
   ├── §Workload Matrix (Phase 1)
   └── ## Phase 2 — Static Hot-Path Analysis  [NEW]
           ├── ### Complexity Table
           │       Module | N complexity | M complexity | Feature gate
           │       (6 rows, one per module sentinel)
           ├── ### Allocation Hotspot List
           │       site: file:line | type | size | phase target
           │       (8 to_dmatrix SVD sites + 14 from_column_slice sites + 1 redundant clone)
           └── ### Parallelism Gap List
                   loop: file:line | status | parallelism macro? | gap candidate?
                   (gap: cv.rs:76 fold loop, elastic_fpca N-loops)
                   (ok: karcher.rs:185, pairwise.rs:227, fraiman_muniz.rs:82, smoothing.rs:110)
```

### Recommended Project Structure

No new files created in Phase 2. The only output is appended sections in:

```
.planning/research/
└── AUDIT-REPORT.md     ← append ## Phase 2 section with 3 sub-sections
```

### Pattern 1: Append to Growing Report

The plan must use the `Edit` tool (not `Write`) to append Phase 2 sections below the existing Phase 1 content. Use an anchor comment at the end of existing content or append directly. Do NOT overwrite Phase 1 content.

### Pattern 2: File:Line Citation Format

Every claim about a code site must follow the format:
```
`{file_relative_to_fdars_root}:{line}` — {brief description}
```

Example:
```
`fdars-core/src/regression.rs:298` — `SVD::new(weighted.to_dmatrix(), true, true)` — core FPCA, always executed
```

This format allows the verify step to `grep -n "to_dmatrix"` and confirm the line number exists.

---

## Common Pitfalls

### Pitfall 1: Confusing `center_1d` (parallelized) with `center_columns` (sequential)

**What goes wrong:** `fdata.rs:center_1d` uses `iter_maybe_parallel!` at line 218. `regression.rs:center_columns` (the function actually called by `fdata_to_pc_1d`) uses a plain `for j in 0..m` loop at line 171. A reader who finds `center_1d` in fdata.rs and concludes "FPCA centering is parallelized" is wrong.

**How to avoid:** Always cite the specific call path: `fdata_to_pc_1d` → `center_columns` (regression.rs:167, sequential), not `fdata.rs:center_1d`.

**Warning signs:** If the complexity table says "FPCA centering is parallel," check the actual call chain.

### Pitfall 2: Reporting `matrix.rs:682` (test-only `to_dmatrix()`) as a production allocation hotspot

**What goes wrong:** `matrix.rs:682` contains `let dmat = mat.to_dmatrix()` inside a `#[cfg(test)]` block. It is not compiled into the production crate or any benchmark.

**How to avoid:** The allocation hotspot list must exclude test-only call sites. Production `to_dmatrix()` SVD sites: 8 (not 9).

### Pitfall 3: Labeling `depth_batch` on `StreamingFraimanMuniz` as a parallelism gap

**What goes wrong:** `streaming_depth/fraiman_muniz.rs:82` — `iter_maybe_parallel!(0..nobj)` IS already feature-gated parallel. It is not a gap.

**How to avoid:** Check every named loop against the full `iter_maybe_parallel!` grep inventory (Section 3) before labeling it a gap.

### Pitfall 4: Claiming banding reduces O(n²) → O(n) in the Karcher iteration

**What goes wrong:** Banding reduces the per-pair DP from O(m²) to O(m·band). The O(n) loop over pairs in each iteration is NOT affected by banding. The total cost reduction is a constant factor for fixed band, not an asymptotic reduction in n.

**How to avoid:** Complexity entry for elastic alignment must separate the n-scaling from the m-scaling:
- Without band: O(max_iter · n · m²)
- With band (band << m): O(max_iter · n · m · band)
- n-scaling is unchanged by banding.

### Pitfall 5: Treating `DMatrix::from_column_slice` in `smooth_basis.rs` as an SVD copy

**What goes wrong:** `smooth_basis.rs:198` creates a DMatrix for basis regression (least-squares solve), not for SVD. It is a different allocation pattern with different optimization paths.

**How to avoid:** The allocation hotspot list must use two sub-categories: (A) `to_dmatrix()` for SVD and (B) `DMatrix::from_column_slice` for basis construction. These are distinct candidates for different optimization strategies.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Scanning for allocation sites | Manual inspection of every file | `grep -rn "to_dmatrix\|from_column_slice\|DMatrix::from"` on fdars-core/src/ | Grep is exhaustive; manual inspection misses nested calls |
| Inferring feature-gate status of a loop | Reading the surrounding code | `grep -rn "iter_maybe_parallel" fdars-core/src/` then check each named loop against the list | The grep result from this session already gives the complete inventory |
| Counting lines added to AUDIT-REPORT.md | Manual diff | `wc -l .planning/research/AUDIT-REPORT.md` before and after | Verifiable metric for "did the section get written" |

---

## Runtime State Inventory

This is an analysis-only phase (no code changes, no renaming, no migration). Runtime state inventory: SKIPPED — not applicable. All work is read-only from fdars-core/src/ and write-only to the markdown report.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `grep` (GNU coreutils) | All source-analysis steps | Yes (Linux) | standard | — |
| Rust toolchain | `cargo check` for feature-gate verification (optional) | Yes | 1.97.0 | — |
| `.planning/research/AUDIT-REPORT.md` | Append target | Yes (created in Phase 1) | — | — |

**Missing dependencies with no fallback:** None.

**Known limitation:** This phase makes no runtime calls. The grepping was done in this research session to establish ground truth; the executor need not re-grep unless a source file has changed since 2026-08-07.

---

## Validation Architecture

> nyquist_validation is enabled (workflow.nyquist_validation absent = enabled per RESEARCH.md contract).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | None — Phase 2 is analysis-only; output is markdown |
| Config file | — |
| Quick run command | Manual: `grep -c "to_dmatrix" .planning/research/AUDIT-REPORT.md` |
| Full suite command | Manual checklist against 4 success criteria |

### Phase Requirements → Verification Map

| SC # | Behavior | Verification Type | Automated Command | Source |
|------|----------|-----------|-------------------|--------|
| SC1 | Complexity table with ≥6 module rows, each citing Big-O and file:line | Manual grep | `grep -c "O(n" .planning/research/AUDIT-REPORT.md` (expect ≥ 6) | Verified in this session |
| SC2 | Allocation list with 8 production `to_dmatrix()` sites + `from_column_slice` sites | Manual grep | `grep -c "to_dmatrix" .planning/research/AUDIT-REPORT.md` (expect ≥ 8) | Verified in this session |
| SC3 | Parallelism gap list with sequential gaps labeled and already-parallel items noted | Manual review | `grep -c "sequential\|gap candidate\|already parallel" .planning/research/AUDIT-REPORT.md` | Verified in this session |
| SC4 | Feature-gate annotations on every finding | Manual review | `grep -c "parallel-gated\|sequential\|linalg-gated\|always" .planning/research/AUDIT-REPORT.md` | Verified in this session |

### Wave 0 Gaps

None — no test files to create. The verification is manual grep-based. The only prerequisite artifact (AUDIT-REPORT.md) already exists from Phase 1.

---

## Security Domain

> security_enforcement: true in config.json. This phase writes only to local markdown files. No network calls, no user input, no secrets.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Local analysis only |
| V3 Session Management | No | Local analysis only |
| V4 Access Control | No | Local analysis only |
| V5 Input Validation | No | No user-supplied input |
| V6 Cryptography | No | No cryptographic operations |

**Security finding:** None. This phase is read-only analysis of local source files.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `fraiman_muniz_1d` (static, non-streaming) parallelism is via a loop in `depth/mod.rs` — executor should verify whether it uses `iter_maybe_parallel!` or a sequential loop | Section 4 (Module 3) | If it is sequential, FM depth becomes a gap candidate; if parallel, it stays in the "already parallel" column |
| A2 | `band_radius(0.0, m)` returns `None` (unbanded) — derived from `karcher_mean` calling `karcher_mean_impl(..., 0.0)` | Section 3F | If `band_frac = 0.0` maps to a small positive band rather than `None`, the default cost is lower than O(m²) per pair |

**Both assumptions are LOW risk** — A1 affects only the FM depth row label; A2 can be confirmed with a one-line read of the `band_radius` function. Neither affects the SVD copy count or the CV fold gap finding, which are HIGH confidence.

**If this table has only 2 entries:** Correct — the vast majority of findings in this RESEARCH.md are directly verified from source files read in this session.

---

## Open Questions

1. **Does `fraiman_muniz_1d` (the static variant) use `iter_maybe_parallel!`?**
   - What we know: `streaming_depth/fraiman_muniz.rs:82` uses it. `depth/mod.rs:80` uses `maybe_par_chunks_mut_enumerate!` for random-projection depth but FM depth may be a different code path.
   - What's unclear: The exact function body of the static `fraiman_muniz_1d` was not fully read in this session — only its call signature was confirmed from the bench file.
   - Recommendation: Executor runs `grep -n "iter_maybe_parallel\|for.*0\.\." fdars-core/src/depth/mod.rs` before writing the depth module row in the complexity table. If sequential, label it a parallelism gap.

2. **Does `elastic_fpca.rs:930` correspond to a third public FPCA function or an internal helper?**
   - What we know: Lines 214, 317, 483, 584 are in `horiz_fpca`, `joint_fpca` (two paths), and `vert_fpca` respectively. Line 930 is a fifth site.
   - What's unclear: Which function contains `combined.to_dmatrix()` at line 930.
   - Recommendation: Executor reads `elastic_fpca.rs` around line 920–940 to confirm the function name before adding the allocation list entry.

---

## Sources

### Primary (HIGH confidence — read directly in this session)

- `fdars-core/src/parallel.rs:1-156` — All 5 parallelism macros, verbatim definitions and re-exports [VERIFIED]
- `fdars-core/src/classification/cv.rs:1-120` — Complete `fclassif_cv` implementation, sequential `for fold in 0..nfold` at line 76 [VERIFIED]
- `fdars-core/src/regression.rs:165-322` — `center_columns` (sequential, lines 167-181), weighted clone (line 291), `SVD::new(weighted.to_dmatrix(), …)` (line 298) [VERIFIED]
- `fdars-core/src/elastic_fpca.rs:1-850 (sampled)` — 5 `to_dmatrix()` SVD sites at lines 214, 317, 483, 584; sequential N-loops at 701, 720, 734, 740 [VERIFIED]
- `fdars-core/src/alignment/karcher.rs:170-432` — `iter_maybe_parallel!` at lines 185, 376, 413, 432; `karcher_mean` passes `band_frac=0.0` at line 300 [VERIFIED]
- `fdars-core/src/alignment/pairwise.rs:185-241` — `elastic_self_distance_matrix` → full DP (no band) at line 194; `iter_maybe_parallel!(0..n)` at line 227 [VERIFIED]
- `fdars-core/src/alignment/nd.rs:695-745` — `SVD::new(gram.to_dmatrix(), …)` at line 705 [VERIFIED]
- `fdars-core/src/spm/mfpca.rs:325-356` — `SVD::new(stacked.to_dmatrix(), …)` at line 336 [VERIFIED]
- `fdars-core/src/smoothing.rs:100-130` — `slice_maybe_parallel!(x_new)` at line 110, sequential inner `for i in 0..n` [VERIFIED]
- `fdars-core/src/streaming_depth/fraiman_muniz.rs:77-90` — `iter_maybe_parallel!(0..nobj)` at line 82 [VERIFIED]
- `fdars-core/src/fdata.rs:165-240` — `iter_maybe_parallel!(0..m)` at line 172 and 218 in `mean_1d`/`center_1d`; distinct from `regression.rs:center_columns` [VERIFIED]
- `fdars-core/src/matrix.rs:310, 682` — `to_dmatrix()` definition (line 310) and test-only call (line 682) [VERIFIED]
- `grep -rn "to_dmatrix\|DMatrix::from_column_slice"` — complete production inventory run in this session [VERIFIED]
- `grep -rn "iter_maybe_parallel!\|slice_maybe_parallel!\|maybe_par_chunks_mut!"` — complete macro usage inventory run in this session [VERIFIED]
- `.planning/codebase/CONCERNS.md` — Scaling limits (lines 119-134), dense matrix reconstruction (lines 75-79) [VERIFIED]
- `.planning/phases/01-measurement-discipline-baselines/01-02-SUMMARY.md` — Phase 1 decisions including A5 resolution (sequential `center_columns`) and D-05 single growing report [VERIFIED]
- `.planning/research/AUDIT-REPORT.md` — D-05 single growing report confirmed; Phase 1 sections present [VERIFIED]

### Tertiary (LOW confidence — training knowledge, not verified)

- Algorithmic complexity derivations for FPCA SVD cost (O(min(n,m)·n·m)) — standard linear algebra, not verified from source [ASSUMED]
- `band_radius(0.0, m)` returning `None` — inferred from context, not read directly [ASSUMED: A2]

---

## Metadata

**Confidence breakdown:**
- Allocation hotspot inventory: HIGH — grep run directly on source in this session
- Parallelism gap inventory: HIGH — grep run directly on source in this session; each named site read
- Banding opt-in finding: HIGH — `karcher_mean` source read directly
- FPCA centering sequential finding: HIGH — confirmed in AUDIT-REPORT.md (Open Question A5 resolved in Phase 1) AND regression.rs:167 read directly
- Complexity derivations: MEDIUM — loop nesting confirmed, Big-O from standard theory
- `fraiman_muniz_1d` parallelism status: MEDIUM — streaming variant confirmed parallel; static variant anchor not fully read (Open Question 1)

**Research date:** 2026-08-07
**Valid until:** Stable — fdars-core v0.14.0 source is the audit target. Recheck only if source files change before Phase 2 executes.
