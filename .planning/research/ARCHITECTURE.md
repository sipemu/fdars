# Architecture Research — Performance Hot-Spot Map

**Domain:** Functional Data Analysis (Rust library, fdars-core v0.14.0)
**Researched:** 2026-08-07
**Confidence:** HIGH (derived from direct source analysis of the codebase)

---

## Purpose

This document maps the algorithmic cost structure of fdars-core so that the
performance audit can aim static analysis and benchmarks at the true hot paths.
Each section identifies a computational subsystem, gives its complexity in N
(number of curves) and M (grid resolution), characterises the dominant cost
driver, and assigns an audit priority.

---

## Component Boundaries and Cost Map

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  AUDIT PRIORITY MAP                                                      │
│                                                                          │
│  HIGH   alignment/        elastic DP  O(N·M²) Karcher / O(N²·M²) dist  │
│         elastic_fpca.rs   7× SVD+copy per elastic-FPCA call             │
│         regression.rs     FdMatrix→DMatrix copy before every SVD        │
│                                                                          │
│  MED    depth/            band/BD  O(N²·M) pairwise; FM/MBD O(N·M·logN)│
│         spm/mfpca.rs      stacked SVD on wide matrix                    │
│         classification/cv K-fold FPCA re-computed per fold              │
│         clustering.rs     k-means assignment O(K·N·M) per iteration     │
│                                                                          │
│  LOW    smoothing.rs      kernel smoother O(n·M) per curve, already //  │
│         basis/            P-spline normal-eq once, then O(n·B) per curve│
│         seasonal/         FFT O(M log M), rustfft already optimised     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | File(s) | Responsibility |
|-----------|---------|----------------|
| FPCA engine | `src/regression.rs:fdata_to_pc_1d()` | Full SVD on N×M weighted matrix; returns FpcaResult |
| Elastic FPCA | `src/elastic_fpca.rs` | Vertical / horizontal / joint FPCA after alignment; 7 SVD calls with copies |
| Elastic alignment (pairwise) | `src/alignment/pairwise.rs` | Pairwise DP warp, SRSF distance, distance matrices |
| Karcher mean | `src/alignment/karcher.rs` | Iterative alignment-and-mean; inner loop is O(N) DP warps per iteration |
| DP core | `src/alignment/mod.rs:dp_alignment_core_banded()` | M×M grid fill; banded variant exists but requires opt-in |
| Depth measures | `src/depth/` | Band BD O(N²·M), FM/MBD O(N·M·log N) via sorted columns |
| MFPCA | `src/spm/mfpca.rs` | SVD on horizontally stacked N×(sum M_p) matrix |
| Classification CV | `src/classification/cv.rs` | K-fold loop, FPCA re-run per fold |
| Clustering | `src/clustering.rs` | k-means assignment + centroid O(K·N·M) per iteration |
| P-spline / B-spline | `src/basis/pspline.rs`, `basis/bspline.rs` | Normal-equation solve once; per-curve solve is O(n·B²) |
| Kernel smoothing | `src/smoothing.rs` | Nadaraya-Watson / local-linear per output point; rayon-parallelised |
| FdMatrix ↔ DMatrix copy | `src/matrix.rs:to_dmatrix()` | Full N×M copy into nalgebra heap allocation |

---

## Architectural Patterns

### Pattern 1: FdMatrix-to-DMatrix Round-Trip (Dense Copy Before Every SVD)

**What:** `fdata_to_pc_1d()` in `src/regression.rs` builds a weighted copy of
the centered matrix (`weighted.clone()` followed by `weighted.to_dmatrix()`),
then calls `SVD::new(...)`. This is two allocations of size N×M before any
factorisation work begins. In `src/elastic_fpca.rs` this pattern is repeated
for all seven SVD call sites (lines 122, 214, 317, 399, 483, 584, 930).

**Complexity of the copy:** O(N·M) time and memory per SVD call.
At N=500 curves × M=200 points, that is 100 000 f64 values (800 KB) copied on
every FPCA invocation.

**When to use (acceptable):** Once at fitting time if the result is cached. The
`FpcaResult::project()` method avoids re-running SVD, which is the correct
pattern already followed by downstream callers.

**Problem:** The copy is unnecessary when nalgebra works on column-major data
and FdMatrix is already column-major. A zero-copy path via
`DMatrix::from_column_slice` (which `to_dmatrix()` already uses internally)
eliminates the intermediate `weighted` clone. Alternatively, switching to
`faer`'s `Mat` (already a dev dependency) or a randomised/truncated SVD would
remove the full-decomposition cost entirely when only `ncomp << min(N,M)`
components are needed.

**Audit action:** Grep all `to_dmatrix()` call sites and verify each one is
preceded by a necessary transformation (weighting, centering) rather than a
redundant clone. Count the total allocation budget per FPCA call.

---

### Pattern 2: Full O(M²) DP Grid Per Pairwise Alignment

**What:** `dp_alignment_core_banded()` (`src/alignment/mod.rs`, line 567) fills
an M×M DP grid with a coprime-neighbourhood of 35 possible moves. Without a
band the cost is O(M²) per pair.

**Complexity:**

| Operation | Complexity |
|-----------|------------|
| Single pair alignment (unbanded) | O(M²) |
| Single pair alignment (banded, radius r) | O(M·r) |
| Karcher mean, K iterations | O(K·N·M²) unbanded, O(K·N·M·r) banded |
| Self-distance matrix (N curves) | O(N²·M²) / O(N²·M·r) banded |
| Cross-distance matrix (N1 × N2) | O(N1·N2·M²) / O(N1·N2·M·r) banded |

At M=200, N=100, K=30, the unbanded Karcher mean requires ≈ 120 000 000 000
elementary operations. With a 15% band (r≈30) this drops 7× to ≈ 18 000 000 000.

**Existing mitigation:** `elastic_align_pair_banded`, `karcher_mean_banded`,
`elastic_self_distance_matrix_banded`, and `elastic_cross_distance_matrix_banded`
are all exposed. The thread-local `DP_SCRATCH` eliminates per-call heap
allocation for the DP grid (correctly reuses across rayon worker threads).

**Gap:** The banded variants require the caller to pass `band_frac`; there is no
auto-selection heuristic. For Karcher-mean-based pipelines (elastic FPCA, elastic
regression) the Karcher step calls `align_srsf_pair_banded` through
`iter_maybe_parallel!` but `band` is propagated from the caller — the default
unlocked path in `karcher_mean()` passes `band = None`. SRSF computation itself
(`srsf_transform`) is O(N·M) and not a bottleneck.

**Audit action:** Profile `dp_grid_solve_banded` call count and total wall time
under realistic (N=100, M=200) inputs. Verify the thread-local scratch pad is
actually eliminating allocator round-trips under the benchmark harness.

---

### Pattern 3: O(N²·M) Pairwise Depth

**What:** Band Depth (`src/depth/band.rs:band_1d`) checks all C(N,2) pairs for
containment across all M time points. Modified Band Depth (`modified_band_1d`)
uses `SortedReferenceState` (sorted columns) to reduce per-query cost from
O(N·M) to O(M·log N). Fraiman-Muniz depth uses the same sorted-column structure.

**Complexity:**

| Measure | Build | Per-query | Total (N queries against N references) |
|---------|-------|-----------|----------------------------------------|
| FM / MBD (sorted) | O(N·M·log N) | O(M·log N) | O(N·M·log N) |
| BD (full reference) | O(N·M) | O(N·M) with early exit | O(N²·M) worst case |
| Random projection depth | O(nproj·N·M) | O(nproj·log N) | O(nproj·N·M) |
| Spatial depth | O(N·M) | O(N·M) | O(N²·M) |

**Parallelism:** `iter_maybe_parallel!` covers the outer (query) loop in BD
(`src/depth/band.rs`, line 66), and `maybe_par_chunks_mut_enumerate!` covers
random projection pre-computation. FM and MBD dispatch through
`StreamingDepth::depth_batch()`, which is not parallelised at the batch level —
queries are processed sequentially over the query matrix in
`streaming_depth/fraiman_muniz.rs` and `streaming_depth/mbd.rs`.

**Audit action:** Check whether `StreamingDepth::depth_batch()` implementations
use `iter_maybe_parallel!` for the outer query loop or are purely sequential.
The batch can be trivially parallelised since each query is independent.

---

### Pattern 4: Classification CV — FPCA Re-Run Per Fold

**What:** `fclassif_cv()` (`src/classification/cv.rs`) runs K folds sequentially.
Inside each fold `cv_fold_predict()` calls `fdata_to_pc_1d()` on the training
set, which re-does full SVD from scratch. For nfold=10, N=200, M=100 this means
10 separate SVD calls on (180 × 100) matrices.

**Complexity:** O(nfold · N · M · min(N,M)) total, dominated by the SVD inside
each fold.

**Gap:** The folds are executed sequentially (plain `for fold in 0..nfold` loop,
line 76). There is no parallelism over folds. Since folds are independent, they
could be parallelised with `iter_maybe_parallel!`.

**Audit action:** Confirm folds run sequentially. Quantify wall-time cost via
the classification benchmark. This is a straightforward `iter_maybe_parallel!`
insertion with a caveat about thread-safe RNG seeding (already handled by the
`seed + k` pattern used elsewhere in the codebase).

---

### Pattern 5: Elastic FPCA — Sequential N-Loops Without Rayon

**What:** `src/elastic_fpca.rs` contains seven SVD calls with copies (see Pattern
1 above) and multiple sequential `for i in 0..n` loops in helper functions such
as `shooting_vectors_from_psis()` (line 701), `build_augmented_srsfs()` (line
720), and `center_matrix()` (line 734). None of these inner O(N) loops use
`iter_maybe_parallel!`.

**Complexity driver:** Each of the seven SVD sites operates on a matrix whose
dimensions range from N×M (amplitude FPCA) to N×(M+1) (augmented SRSF FPCA) to
N×(2M) (joint FPCA). At N=500, M=200 a joint SVD is on a 500×400 matrix. Full
SVD of an n×m matrix is O(n·m·min(n,m)); for the joint case that is
O(500·400·400) ≈ 80 billion ops before the copy overhead.

**Audit action:** Count the sequential N-loops in elastic_fpca.rs that could be
replaced with `iter_maybe_parallel!` and estimate the speedup. For the SVD sites
specifically, evaluate whether a covariance-eigendecomposition path (O(M²·N +
M³) via a pre-computed covariance matrix, much cheaper when N < M) is
applicable.

---

### Pattern 6: P-Spline — Normal-Equation Solve Per Curve

**What:** `pspline_fit_1d()` (`src/basis/pspline.rs`, line 66) correctly computes
the Gram matrix B^T B and penalised inverse once for all N curves, then solves
a cheap O(B²) system per curve. The per-curve `DMatrix` allocations (`DVector`
and `DMatrix` temporaries) accumulate when N is large.

**Complexity:**

| Step | Cost |
|------|------|
| B-spline basis matrix (M × B) | O(M·B·order) |
| B^T B (B × B) | O(M·B²) |
| Pseudoinverse (SVD of B × B) | O(B³) |
| Per-curve solve (B vector multiply) | O(B²) |
| Total for N curves | O(M·B² + N·B²) |

B is typically 15–40 (number of B-spline bases), so the solve is fast. The
dominant cost for large N is the N·B² term, but at B=30 and N=1000 this is
only ~1 million ops — not a bottleneck. The per-curve `DVector::from_vec`
heap allocation is mildly wasteful but unlikely to be measurable.

**Audit action:** LOW priority. Note the `DMatrix` allocation pattern for
completeness; no change needed unless profiling reveals it.

---

### Pattern 7: GMM Clustering Sequential Inner Loops

**What:** `clustering.rs` uses `maybe_par_chunks_mut_enumerate!` for the
assignment step (line 161) and `slice_maybe_parallel!` for the centroid update
(line 251), but the outer iteration loop and convergence check are sequential.
Distance computation per observation is O(K·M) (K cluster centroids, M points),
making the full assignment pass O(N·K·M) per iteration, parallelised over N.

**Gap:** The fuzzy c-means variant (`fuzzy_cmeans_fd` at line 854 via
`iter_maybe_parallel!`) is parallelised. Verify the hard k-means assignment
parallel path reaches the assignment step correctly — the inner loop over K is
still sequential within each chunk.

**Audit action:** LOW–MED. Parallelism exists; confirm it is effective at
realistic K values. Profile under N=500, K=5, M=200.

---

## Data Flow: Performance Cost Drivers

```text
Input FdMatrix (N rows = curves, M cols = grid points)
         │
         ├──► fdata_to_pc_1d()            COST: O(N·M·min(N,M)) SVD
         │         │  weighted.clone()     COST: O(N·M) alloc + copy
         │         │  to_dmatrix()         COST: O(N·M) alloc + copy
         │         │  nalgebra::SVD::new() COST: O(N·M·min(N,M)) Bidiagonal+QR
         │         └► FpcaResult           BENEFIT: cached — reuse via .project()
         │
         ├──► karcher_mean() / elastic_self_distance_matrix()
         │         │  srsf_transform()     COST: O(N·M) — cheap
         │         │  iter_maybe_parallel! over N pairs or N×N pairs
         │         │  dp_alignment_core_banded() per pair  COST: O(M²) or O(M·r)
         │         └► KarcherMeanResult    COST TOTAL: O(K·N·M²) or O(N²·M²)
         │
         ├──► band_1d() / modified_band_1d()
         │         │  FullReferenceState::from_reference()   COST: O(N·M)
         │         │  bd_one_inner() for each query          COST: O(N·M) + early exit
         │         └► depths Vec<f64>      COST TOTAL: O(N²·M) worst case
         │
         ├──► elastic_fpca variants (vert/horiz/joint)
         │         │  karcher_mean()       COST: O(K·N·M²)
         │         │  7× SVD with copies   COST: 7 × O(N·M·min(N,M)) + 7 × O(N·M) copies
         │         └► result structs       NOTE: no parallelism on inner N-loops
         │
         └──► fclassif_cv()
                   │  for fold in 0..K    SEQUENTIAL over K folds
                   │  fdata_to_pc_1d()    COST: O(N·M·min(N,M)) per fold
                   └► error_rate          COST TOTAL: O(K_folds·N·M·min(N,M))
```

---

## Scaling Behaviour Summary

| Subsystem | Dominant Complexity | Scales Badly With | Parallelised |
|-----------|---------------------|-------------------|--------------|
| FPCA (regression.rs) | O(N·M·min(N,M)) | large M (M→N column space) | No — SVD is single-threaded in nalgebra |
| Elastic alignment pairwise | O(M²) per pair | M (grid resolution) | Outer N loop: yes via iter_maybe_parallel! |
| Karcher mean | O(K·N·M²) | M and iteration count K | N inner loop: yes; K loop: no |
| Elastic distance matrix | O(N²·M²) | Both N and M | Outer N loop: yes |
| Elastic FPCA | O(K·N·M²) + 7·O(N·M²) SVD | N, M | Inner N-loops: no |
| Band Depth | O(N²·M) | N | Outer query loop: yes |
| MBD / FM depth | O(N·M·log N) build + O(M·log N) query | N·M | Batch queries: no |
| Classification CV | O(nfold·N·M·min(N,M)) | nfold and M | Folds loop: no |
| k-means clustering | O(iters·N·K·M) | K and M | Assignment step: yes |
| P-spline fit | O(M·B² + N·B²) | N (for large N only) | No (but fast in practice) |

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Redundant FdMatrix Clones Before SVD

**What happens:** `fdata_to_pc_1d()` calls `centered.clone()` to create
`weighted`, then `weighted.to_dmatrix()` to create the nalgebra matrix. That is
two complete N×M copies before SVD starts. In `elastic_fpca.rs` the same pattern
appears 7 times.

**Why it's wrong:** At N=500, M=200, each copy is 800 KB; 7 copies in one
elastic-FPCA call = 5.6 MB of allocation and memcpy overhead before any linear
algebra begins.

**Do this instead:** Apply the weight scaling in-place or during the copy-to-DMatrix
step. Use `DMatrix::from_fn(n, m, |i, j| centered[(i,j)] * sqrt_weights[j])` to
merge the scale and copy into a single pass. For SVD, consider whether the
covariance approach (B = X^T X, eigendecompose B) halves the work when M < N.

### Anti-Pattern 2: Karcher Mean Without Band

**What happens:** `karcher_mean()` passes `band = None` to
`align_srsf_pair_banded()`, using the full O(M²) DP grid for every pair on every
iteration.

**Why it's wrong:** For M=200, N=100, K=30 iterations, this is 30 × 100 × 40 000
= 120 million elementary cell evaluations per iteration step. The banded variant
reduces this by ~7× at 15% band with negligible accuracy loss on nearly-diagonal
warps.

**Do this instead:** Select a default band (e.g. `band_frac = 0.2`) for the
high-level `karcher_mean()` API and document the accuracy trade-off. Expose the
current unbounded path only when `band_frac = 0.0` or `None` is explicitly
requested.

### Anti-Pattern 3: Sequential Folds in Classification CV

**What happens:** The fold loop in `fclassif_cv()` runs `for fold in 0..nfold`
with a full `fdata_to_pc_1d()` SVD inside each iteration. Folds are fully
independent.

**Why it's wrong:** On a machine with 8 cores, 10 folds takes 10× the single-fold
wall time instead of ~2×.

**Do this instead:** Wrap the fold iteration in `iter_maybe_parallel!(0..nfold)`.
The RNG is not used in the fold loop body (fold assignment is done before the
loop), so there is no seeding hazard. Each fold creates its own `fdata_to_pc_1d`
context with no shared mutable state.

### Anti-Pattern 4: Non-Parallel depth_batch in Streaming Depth

**What happens:** `StreamingDepth::depth_batch()` implementations in
`streaming_depth/fraiman_muniz.rs` and `streaming_depth/mbd.rs` process query
curves sequentially via a plain iterator over the query matrix rows.

**Why it's wrong:** FM and MBD are trivially parallel across queries — each row
is independent given the pre-built reference state.

**Do this instead:** Implement `depth_batch()` using `iter_maybe_parallel!` over
the query rows. The `SortedReferenceState` is immutable after construction and
safe to share across threads (`&self` borrow only).

---

## Priority Order for the Audit

The following sequence minimises effort while maximising coverage of the highest
expected gains.

1. **Elastic alignment / Karcher / distance matrices** (`src/alignment/`) —
   O(N²·M²) without banding is the most likely source of 10–100× slowdowns at
   real data sizes. Static analysis to confirm band is not auto-enabled; benchmark
   to quantify.

2. **FPCA SVD copies** (`src/regression.rs`, `src/elastic_fpca.rs`) — the
   FdMatrix→DMatrix round-trip pattern appears 8 times across the codebase. Static
   analysis counts copies per call; benchmark `fdata_to_pc_1d` at N=100/500/1000
   vs M=50/200 to separate SVD cost from copy cost.

3. **Elastic FPCA sequential N-loops** (`src/elastic_fpca.rs`) — 7 SVD calls
   plus unparallelised O(N) helpers. Audit for `iter_maybe_parallel!` insertion
   opportunities in the phase-FPCA shooting-vector and augmented-SRSF loops.

4. **Classification CV folds** (`src/classification/cv.rs`) — trivial
   parallelisation opportunity; benchmark nfold=10 with and without parallel
   iteration.

5. **Depth batch parallelism** (`src/streaming_depth/`) — `depth_batch` is
   sequential for FM and MBD; the fix is one macro insertion.

6. **Band Depth scaling** (`src/depth/band.rs`) — the O(N²·M) cost is
   structural; audit the outer-loop parallelism and consider whether random
   projection depth (`random_projection_1d`) should be the recommended default
   for large N.

7. **Clustering and GMM** (`src/clustering.rs`) — parallelism exists; audit
   whether the assignment-step chunk size is appropriate and whether per-iteration
   allocation pressure is visible.

8. **P-spline / B-spline / smoothing** — LOW priority; complexity is well-bounded
   and smoothing is already parallelised.

---

## Integration Points

### External Library Boundaries

| Boundary | Integration | Performance Note |
|----------|-------------|------------------|
| FdMatrix → nalgebra DMatrix | `to_dmatrix()` / `from_dmatrix()` in `src/matrix.rs` | Full O(N·M) copy; zero-copy path would require unsafe or owned storage hand-off |
| nalgebra SVD | `nalgebra::SVD::new(matrix, true, true)` | Full bidiagonal reduction + QR iteration; single-threaded; no truncation support |
| faer (linalg feature) | Used only in `src/linalg.rs:cholesky_d()` and `anofox-regression` for ridge; not used for FPCA SVD | faer supports truncated/randomised SVD — candidate for FPCA replacement |
| rayon thread pool | `iter_maybe_parallel!`, `slice_maybe_parallel!`, `maybe_par_chunks_mut_enumerate!` | 5 macros in `parallel.rs`, 185 call sites; disabled by default in tests |

### Internal Module Boundaries

| Boundary | Communication | Audit Note |
|----------|---------------|------------|
| alignment/ → regression.rs | `fdata_to_pc_1d()` called from elastic_fpca for phase FPCA | Both sites trigger the SVD-copy anti-pattern |
| depth/ → streaming_depth/ | All depth measure functions delegate to StreamingDepth trait impls | Parallelism gap is in the trait impls, not the depth module itself |
| classification/cv.rs → regression.rs | `fdata_to_pc_1d()` called once per fold | Sequential fold loop is the bottleneck, not the SVD itself |
| spm/mfpca.rs → matrix.rs | Horizontally stacks variables then calls `stacked.to_dmatrix()` for a single wide SVD | Matrix size is N × sum(M_p); at 5 variables × M=200 this is a 500×1000 SVD |

---

## Sources

- Direct source analysis of `fdars-core/src/` (2026-08-07)
- `.planning/codebase/ARCHITECTURE.md` — documented anti-patterns and module map
- `.planning/codebase/STACK.md` — nalgebra 0.33, faer 0.23, rayon 1.10 versions
- Golub & Van Loan, "Matrix Computations" (4th ed.) — SVD complexity O(n·m·min(n,m))
- Srivastava & Klassen, "Functional and Shape Data Analysis" — SRSF DP alignment O(M²)
- Sakoe & Chiba (1978) — DTW band constraint O(M·r)
- López-Pintado & Romo (2009) — Band Depth complexity analysis

---

*Architecture research for: fdars-core performance audit*
*Researched: 2026-08-07*
