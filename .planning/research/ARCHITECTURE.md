# Architecture Research

**Domain:** Rust functional-data-analysis library — k-Shape clustering & Shape-Based Distance (v0.34.0)
**Researched:** 2026-09-02
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Public API layer                                │
│   lib.rs (pub use) · prelude.rs                                      │
│   sbd · sbd_matrix_fd · SbdResult · sbd_kmedoids                    │
│   kshape_fd · KShapeConfig · KShapeResult                           │
├──────────────────────────────────────────────────────────────────────┤
│                      Domain modules (NEW)                            │
│  ┌───────────────────────┐    ┌────────────────────────────────────┐ │
│  │  metric/sbd.rs (NEW)  │    │  kshape.rs (NEW, top-level)        │ │
│  │  sbd() pairwise       │    │  kshape_fd() fit                   │ │
│  │  sbd_matrix_fd()      │    │  KShapeConfig (config struct)      │ │
│  │  SbdResult            │    │  KShapeResult (result struct)      │ │
│  └──────────┬────────────┘    │    .predict(&FdMatrix) method       │ │
│             │                 │  sbd_kmedoids() convenience         │ │
│             │ distance matrix └──────────────┬───────────────────── ┘ │
│             │                               │ uses sbd_matrix_fd    │
│             └───────────────────────────────┘                       │
├──────────────────────────────────────────────────────────────────────┤
│                      Shared infrastructure (REUSED)                  │
│  ┌──────────────┐  ┌────────────────────┐  ┌──────────────────────┐  │
│  │ matrix.rs    │  │ alignment/         │  │ shapelet/distance.rs │  │
│  │ FdMatrix     │  │ clustering.rs      │  │ z_normalize_window() │  │
│  │ row_to_buf() │  │ kmedoids_from_     │  │ z_normalize_into()   │  │
│  │ row()        │  │ distances()        │  │                      │  │
│  └──────────────┘  └────────────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌────────────────────┐  ┌──────────────────────┐  │
│  │ helpers.rs   │  │ parallel.rs        │  │ error.rs             │  │
│  │ seed_for_    │  │ iter_maybe_        │  │ FdarError            │  │
│  │ thread()     │  │ parallel! macro    │  │                      │  │
│  └──────────────┘  └────────────────────┘  └──────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ metric/mod.rs — self_distance_matrix() helper (upper-triangle)  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│                      External dependencies (REUSED)                  │
│  ┌───────────────────┐  ┌──────────────────────────────────────────┐  │
│  │ rustfft 6.2       │  │ nalgebra 0.33                            │  │
│  │ FftPlanner::new() │  │ SymmetricEigen (shape extraction)        │  │
│  │ plan_fft_forward  │  │ DMatrix::from_fn / column access         │  │
│  │ plan_fft_inverse  │  │                                          │  │
│  └───────────────────┘  └──────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `metric/sbd.rs` | SBD primitive: FFT NCCc between two z-normalized series → `dist = 1 - max(NCC)` + optimal shift; `sbd()` pairwise + `sbd_matrix_fd()` n×n symmetric | NEW |
| `kshape.rs` | k-Shape fit: iterative SBD assignment + shape-extraction centroid refinement, n_init restarts, empty-cluster recovery, deterministic seeding; `KShapeConfig`, `KShapeResult`, `KShapeResult::predict()` | NEW |
| `sbd_kmedoids()` | Convenience function: compute SBD distance matrix then delegate to existing `kmedoids_from_distances()` — no new algorithm | thin wrapper in `kshape.rs` |
| `metric/mod.rs` | Add `pub mod sbd` + `pub use sbd::{sbd, sbd_matrix_fd, SbdResult}` to the metric barrel | MODIFIED |
| `lib.rs` | Add `pub mod kshape`, re-export kshape + metric/sbd public symbols | MODIFIED |

## Recommended Project Structure

```
fdars-core/src/
├── metric/
│   ├── sbd.rs              # NEW — SBD core (FFT NCCc, z-norm, optimal shift)
│   ├── mod.rs              # MODIFIED — add `pub mod sbd`, re-export sbd functions
│   ├── gak.rs              # existing — GAK kernel (v0.32.0)
│   └── soft_dtw.rs         # existing — soft-DTW (unchanged)
├── kshape.rs               # NEW — KShapeConfig, KShapeResult, kshape_fd, sbd_kmedoids
├── kernel_kmeans.rs        # existing (closest analog — mirrors this structure)
├── alignment/
│   └── clustering.rs       # existing — kmedoids_from_distances (consumed by sbd_kmedoids)
├── shapelet/
│   └── distance.rs         # existing — z_normalize_window, z_normalize_into (reused by SBD)
└── lib.rs                  # MODIFIED — re-export new public symbols
```

### Structure Rationale

- **`metric/sbd.rs`:** SBD is a distance primitive, not a clustering algorithm. It belongs in the `metric/` submodule family alongside `dtw.rs`, `gak.rs`, `soft_dtw.rs`. The `metric/` module already has `self_distance_matrix()` and `cross_distance_matrix()` helpers that the SBD matrix builder can reuse directly. Placing SBD here keeps it usable independently of k-Shape (e.g. feeding directly into kmedoids or hierarchical clustering) and matches how `gak.rs` was placed before `kernel_kmeans.rs` consumed it.

- **`kshape.rs` (top-level, peer of `kernel_kmeans.rs`):** k-Shape is a full clustering algorithm — config struct, result struct with retained centroids, n_init loop, fit function, out-of-sample predict method. This mirrors `kernel_kmeans.rs` exactly. Both are flat top-level files (not submodules) because they are self-contained algorithms that consume a `metric/` primitive. A dedicated directory (e.g. `kshape/`) would be premature; `kernel_kmeans.rs` (v0.32.0) set the pattern.

- **`sbd_kmedoids()` in `kshape.rs`:** This is a convenience adapter (compute `sbd_matrix_fd`, then call `kmedoids_from_distances`). It lives in `kshape.rs` to keep k-Shape-family functions grouped, and is re-exported from `lib.rs` at the crate root.

## Architectural Patterns

### Pattern 1: Metric-First Layering

**What:** The SBD distance is implemented as a standalone primitive in `metric/sbd.rs`, independent of any clustering consumer. `kshape.rs` imports from `metric::sbd` but `metric::sbd` imports nothing from `kshape.rs`. The `alignment::clustering` kmedoids also imports only the distance matrix, not the metric implementation.

**When to use:** Every time a new distance measure is added (as `gak.rs` and `soft_dtw.rs` demonstrate). Keeps the metric reusable across clustering, depth, and SPM consumers without circular imports.

**Trade-offs:** Slight indirection (two files instead of one), but it matches the existing codebase structure and enables the `sbd_kmedoids` consumer without coupling.

**Example:**
```rust
// metric/sbd.rs — no dependency on kshape.rs
pub fn sbd(x: &[f64], y: &[f64]) -> SbdResult { ... }
pub fn sbd_matrix_fd(data: &FdMatrix) -> FdMatrix { ... }

// kshape.rs — imports the metric
use crate::metric::sbd::{sbd, sbd_matrix_fd};

// sbd_kmedoids convenience in kshape.rs
pub fn sbd_kmedoids(
    data: &FdMatrix,
    config: &KMedoidsConfig,
) -> Result<KMedoidsResult, FdarError> {
    let dist = sbd_matrix_fd(data);
    kmedoids_from_distances(&dist, config)
}
```

### Pattern 2: Mirror kernel_kmeans.rs for Config / Result / n_init / predict

**What:** `KShapeConfig` and `KShapeResult` follow the exact structural pattern of `KernelKmeansConfig` and `KernelKmeansResult`. The n_init restart loop, seed arithmetic (`StdRng::seed_from_u64(config.seed.wrapping_add(restart as u64))`), empty-cluster recovery by reseeding the farthest point, and `predict` as a method on the result struct are all replicated from `kernel_kmeans.rs`.

**Key difference from kernel_kmeans:** k-Shape has explicit centroid curves (unlike kernel k-means which has no centroids), so `KShapeResult` carries a `centers: FdMatrix` field (k rows × m cols, each row is a z-normalized shape prototype). The `predict` method assigns by minimum SBD to stored centroids rather than by Gram-matrix cross-kernel sums.

**When to use:** Whenever adding a new clustering algorithm — the established pattern ensures API consistency, serde-gated derives, and `#[non_exhaustive]` on result structs for forward compatibility.

**Proposed config/result shapes:**
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KShapeConfig {
    pub n_clusters: usize,  // default 2
    pub n_init: usize,      // default 10 — matches KernelKmeansConfig
    pub max_iter: usize,    // default 300
    pub tol: f64,           // default 1e-4
    pub seed: u64,          // default 0
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KShapeResult {
    pub cluster: Vec<usize>,  // length n — matches KmeansResult / KernelKmeansResult
    pub centers: FdMatrix,    // k × m centroid curves (z-normalized shape prototypes)
    pub inertia: f64,
    pub iter: usize,
    pub converged: bool,
    pub n_init_best: usize,   // which restart won — matches KernelKmeansResult
}
```

### Pattern 3: FFT NCCc via FftPlanner (mirrors fts/spectral.rs)

**What:** SBD uses cross-correlation via FFT: zero-pad both z-normalized series to length `>= 2m-1` (next power of two for efficiency), forward-FFT both, multiply element-wise by the complex conjugate of one, inverse-FFT the product, read the maximum of the real part (divided by `m` for normalization) as `max_NCC`, compute `dist = 1 - max_NCC`. The optimal shift is the argmax index (cyclically interpreted).

**Reuse idiom from `fts/spectral.rs`:**
```rust
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

// Build planner once per function call (not per pair).
let mut planner = FftPlanner::<f64>::new();
let fft  = planner.plan_fft_forward(padded_len);
let ifft = planner.plan_fft_inverse(padded_len);
// Fill complex buffers, call fft.process(&mut buf), etc.
```

In `sbd_matrix_fd`, the parallel path (`iter_maybe_parallel!`) must build one planner per rayon thread (not one globally) because `FftPlanner` is not `Sync`. This is the same constraint that led `fts/spectral.rs` to build the planner inside the function scope.

**When to use:** Any FFT-based pairwise distance computation on equal-length series.

### Pattern 4: Shape-Extraction Centroid via nalgebra SymmetricEigen

**What:** The k-Shape centroid update is the shape-extraction step (Paparrizos & Gravano 2015, Section 3.2). For a cluster's `n_c` series (each z-normalized and optimally shift-aligned to the current centroid), form `S = X^T X` (m × m) where `X` is the `n_c × m` aligned matrix, then take the top eigenvector of `M = (I - (1/m) * 1_m * 1_m^T) * S` via `nalgebra::SymmetricEigen`. The new centroid is that eigenvector, z-normalized.

**Reuse from existing codebase:** `fts/spectral.rs` already uses `nalgebra::SymmetricEigen` (its `eigen_at_frequency` helper). The shape-extraction eigenproblem is simpler — a single real symmetric `m × m` matrix, top-1 eigenvector needed — so `DMatrix` + `SymmetricEigen::new(mat, true)` + taking the last column of `.eigenvectors` (nalgebra sorts eigenvalues ascending) is correct.

```rust
use nalgebra::DMatrix;

// Build M (m x m) from the n_c aligned z-normalized curves in the cluster.
let mat: DMatrix<f64> = /* ... */;
let eig = nalgebra::SymmetricEigen::new(mat, true);
// Last column = top eigenvector (nalgebra sorts ascending by eigenvalue).
let centroid_raw: Vec<f64> = eig.eigenvectors
    .column(m - 1)
    .iter()
    .copied()
    .collect();
// z-normalize the centroid before storing.
let centroid = crate::shapelet::distance::z_normalize_window(&centroid_raw);
```

**When to use:** Shape-extraction centroid update only. The SBD assignment phase uses `sbd()` not nalgebra.

## Data Flow

### SBD Pairwise Distance

```
&[f64] x, &[f64] y (raw series slices)
    |
    v
z_normalize_window(x), z_normalize_window(y)    [shapelet::distance reuse]
    |
    v
zero-pad both to padded_len = next_pow2(2*m - 1)
    |
    v
FftPlanner -> forward FFT both
    -> pointwise multiply: buf_x[k] * conj(buf_y[k])
    -> inverse FFT
    |
    v
max(real part of IFFT output) / m = max_NCC
argmax index (cyclically) = optimal shift
    |
    v
SbdResult { dist: 1.0 - max_NCC, shift: i64 }
```

### sbd_matrix_fd (n×n symmetric)

```
FdMatrix (n x m, column-major)
    |
    v
metric::self_distance_matrix(n, |i, j| {
    data.row_to_buf(i, &mut buf_i);
    data.row_to_buf(j, &mut buf_j);
    sbd(&buf_i, &buf_j).dist
})
    |
    v
FdMatrix (n x n symmetric distance matrix)
```

### k-Shape Fit (one restart)

```
FdMatrix (n x m data) + KShapeConfig + restart seed
    |
    v
random-partition init (StdRng::seed_from_u64(seed + restart))
ensure_no_empty_random()
    |
    v
loop (max_iter):
  ┌─ Assignment step ──────────────────────────────────────────────────┐
  │ for each series i:                                                 │
  │   for each cluster c: sbd(series_i, centers.row(c)) -> dist_ic   │
  │   assign i to argmin cluster c                                     │
  │ recover_empty_clusters() — farthest-point reseeding               │
  └─────────────────────────────────────────────────────────────────── ┘
    |
  ┌─ Centroid update (shape extraction) ──────────────────────────────┐
  │ for each cluster c:                                               │
  │   z-normalize each member series                                  │
  │   optimally shift-align each to current centers.row(c) via SBD   │
  │   build M = (I - 1/m * 11^T) * X^T * X  (m x m)                │
  │   top eigenvector of M via nalgebra::SymmetricEigen              │
  │   z-normalize eigenvector -> new centroid for cluster c           │
  └─────────────────────────────────────────────────────────────────── ┘
    |
  convergence check (label stability OR rel. inertia change < tol)
    |
    v
RestartOutcome { cluster, centers, inertia, iter, converged, restart_idx }
```

### k-Shape n_init Restart Loop

```
for restart in 0..config.n_init:
    rng = StdRng::seed_from_u64(config.seed.wrapping_add(restart as u64))
    outcome = run_one_restart(data, config, rng, restart)
    if outcome.inertia < best.inertia: best = outcome

return KShapeResult from best outcome
```

### k-Shape predict (out-of-sample)

```
&KShapeResult (fitted centroids in centers: FdMatrix)
+ &FdMatrix new_data (n_test x m)
    |
    v
for each new series i in new_data:
    for each cluster c in 0..k:
        dist = sbd(new_data.row(i), centers.row(c)).dist
    assign i to argmin cluster c
    |
    v
Vec<usize> (length n_test)
```

### sbd_kmedoids (convenience adapter)

```
FdMatrix (n x m) + KMedoidsConfig
    |
    v
sbd_matrix_fd(&data)                           -> FdMatrix (n x n)
    |
    v
alignment::clustering::kmedoids_from_distances(&dist_mat, config)
    |
    v
KMedoidsResult { labels, medoid_indices, within_distances, ... }
```

## Dependency-Ordered Build Sequence

### Phase 61 — SBD Core (`metric/sbd.rs`)

New file only; nothing in the codebase changes except `metric/mod.rs` (add `pub mod sbd` + re-exports) and `lib.rs` (add SBD re-exports).

Dependencies satisfied before starting: `rustfft` (existing dep), `shapelet::z_normalize_window` + `z_normalize_into` (v0.33.0), `metric::self_distance_matrix` helper (existing in `metric/mod.rs`).

Deliverables:
- `sbd(x: &[f64], y: &[f64]) -> SbdResult` — pairwise, pure-slice, no FdMatrix dependency
- `SbdResult { dist: f64, shift: i64 }` — distance + the optimal cyclic lag
- `sbd_matrix_fd(data: &FdMatrix) -> FdMatrix` — n×n symmetric distance matrix using `metric::self_distance_matrix` (rayon-gated via `iter_maybe_parallel!`)
- Tests: unit (known NCC by hand), symmetry (`sbd(x,y).dist == sbd(y,x).dist`), `dist == 0.0` for identical series after z-norm, constant-series guard (z-norm produces zero vector → dist defined), optional shift sign check

### Phase 62 — k-Shape Fit + Predict (`kshape.rs`)

New top-level file; depends on Phase 61 (`metric::sbd`, `metric::sbd_matrix_fd`).

Dependencies satisfied before starting: Phase 61 complete; `nalgebra::SymmetricEigen` (existing dep); `shapelet::z_normalize_window` / `z_normalize_into`; `StdRng::seed_from_u64` (rand existing dep); `FdMatrix::row_to_buf`.

Deliverables:
- `KShapeConfig` (config struct — mirrors `KernelKmeansConfig`)
- `KShapeResult` with `cluster`, `centers` (k × m `FdMatrix`), `inertia`, `iter`, `converged`, `n_init_best`
- `kshape_fd(data: &FdMatrix, config: &KShapeConfig) -> Result<KShapeResult, FdarError>`
- `KShapeResult::predict(&self, new_data: &FdMatrix) -> Result<Vec<usize>, FdarError>`
- Tests: separation recovery (two well-separated curve groups), determinism (same seed = same result), n_init no-worse-than-single-init, empty-cluster recovery (k > natural clusters), predict round-trip (training data predicts its own label), validation errors (n_clusters=0, n_clusters>n, n_init=0, empty data)

### Phase 63 — SBD-k-medoids Convenience + Integration (`kshape.rs` + `lib.rs`)

Thin wrapper + re-exports + benchmark; depends on Phase 61 and `alignment::clustering::kmedoids_from_distances` (existing).

Deliverables:
- `sbd_kmedoids(data: &FdMatrix, config: &KMedoidsConfig) -> Result<KMedoidsResult, FdarError>` in `kshape.rs`
- Re-exports in `lib.rs`: `pub mod kshape`, `kshape_fd`, `KShapeConfig`, `KShapeResult`, `sbd`, `sbd_matrix_fd`, `SbdResult`, `sbd_kmedoids`
- `prelude.rs` update: add `KShapeConfig`, `KShapeResult`
- Module-level doctest in `kshape.rs` demonstrating the full pipeline (fit, predict, sbd_kmedoids)
- Integration test: confirm `sbd_kmedoids(data, cfg)` produces the same `labels` as calling `sbd_matrix_fd` + `kmedoids_from_distances` independently
- Criterion benchmark entry: `benches/kshape.rs` — pairwise SBD (m=100), `kshape_fd` (n=50, m=100, k=3), criterion group named `kshape`
- `[[bench]] name = "kshape"` in `Cargo.toml`

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `metric/sbd.rs` -> `shapelet/distance.rs` | `use crate::shapelet::distance::{z_normalize_into, z_normalize_window}` | `z_normalize_into` is the alloc-free hot-loop variant; `z_normalize_window` for allocating init. Constant-window guard (std <= 1e-12 -> zeros) already implemented. |
| `metric/sbd.rs` -> `rustfft` | `FftPlanner::<f64>::new()`, `plan_fft_forward(len)`, `plan_fft_inverse(len)` | Build planner once per `sbd()` call; in `sbd_matrix_fd` parallel path, each rayon thread builds its own planner (FftPlanner is not Sync). |
| `metric/sbd.rs` -> `metric/mod.rs` | `self_distance_matrix(n, closure)` helper for upper-triangle parallel loop | Already present; identical usage to `dtw_self_1d`, `lp_self_1d`. |
| `kshape.rs` -> `metric/sbd.rs` | `use crate::metric::sbd::{sbd, sbd_matrix_fd}` | Phase 61 must land before Phase 62 starts. |
| `kshape.rs` -> `nalgebra` | `nalgebra::DMatrix`, `nalgebra::SymmetricEigen` | Shape-extraction centroid step only; no new dep. |
| `kshape.rs` -> `alignment/clustering.rs` | `use crate::alignment::clustering::{kmedoids_from_distances, KMedoidsConfig, KMedoidsResult}` | `sbd_kmedoids` delegates entirely; no change to `alignment/clustering.rs`. |
| `kshape.rs` -> `shapelet/distance.rs` | `z_normalize_window` for centroid z-norm after shape extraction | Same import as SBD core. |
| `lib.rs` -> `kshape.rs` + `metric/sbd.rs` | `pub mod kshape; pub use kshape::{...}; pub use metric::{sbd, sbd_matrix_fd, SbdResult}` | Additive re-exports; zero existing symbol changes. |

### External Constraints (WASM / R binding safety)

All new symbols are additive — no existing public signatures change. `KShapeConfig` and `KShapeResult` carry the same trait derives as every other config/result in the codebase (`Debug + Clone + PartialEq`, conditionally `serde::Serialize + Deserialize` behind the `serde` feature). The `#[non_exhaustive]` attribute on `KShapeResult` ensures forward compatibility without breaking R or WASM callers when fields are added in later milestones. No new crate dependency is introduced: `rustfft` and `nalgebra` are existing deps; `rand` (for `StdRng`) is an existing dep.

## Reuse Map

| Reused item | Location | How reused in v0.34.0 |
|-------------|----------|-----------------------|
| `z_normalize_window()` | `shapelet/distance.rs` | z-normalize series before SBD FFT; z-normalize shape-extraction centroid after eigenvector extraction |
| `z_normalize_into()` | `shapelet/distance.rs` | In-place, alloc-free z-norm in hot inner loop of `sbd_matrix_fd` (no allocation per pair) |
| `FftPlanner::<f64>::new()` + `plan_fft_forward` + `plan_fft_inverse` | `rustfft` dep (idiom from `fts/spectral.rs`) | FFT-based NCCc in `sbd()` |
| `nalgebra::SymmetricEigen` | `nalgebra` dep (used in `fts/spectral.rs:eigen_at_frequency`) | Shape-extraction eigenproblem in k-Shape centroid update |
| `metric::self_distance_matrix()` helper | `metric/mod.rs` | Build the n×n SBD distance matrix using upper-triangle parallel loop |
| `kmedoids_from_distances()` | `alignment/clustering.rs` | Consumed unchanged by `sbd_kmedoids` convenience adapter |
| `KMedoidsConfig` / `KMedoidsResult` | `alignment/clustering.rs` | Parameter and return types for `sbd_kmedoids` — unchanged |
| n_init restart loop pattern | `kernel_kmeans.rs` | Direct structural mirror: seed arithmetic, best-inertia selection, `n_init_best` field in result |
| Empty-cluster recovery by farthest-point reseeding | `kernel_kmeans.rs:recover_empty_clusters` | Same logic adapted for SBD distances (not Gram entries) |
| `predict` as method on result struct | `kernel_kmeans.rs:KernelKmeansResult::predict` | `KShapeResult::predict()` mirrors the signature; uses SBD to stored centroids instead of cross-Gram kernel sums |
| `StdRng::seed_from_u64(seed.wrapping_add(restart as u64))` | `kernel_kmeans.rs` | Deterministic per-restart seeding — identical idiom |
| `ensure_no_empty_random()` pattern | `kernel_kmeans.rs` | Random-partition init that prevents empty clusters at start |
| `#[must_use]` on fit function | convention throughout codebase (74+ functions) | Applied to `kshape_fd` |
| `#[non_exhaustive]` on result structs | convention throughout codebase | Applied to `KShapeResult` |
| Serde-gated derives | convention throughout codebase | Applied to `KShapeConfig` and `KShapeResult` |
| `FdMatrix::row_to_buf()` | `matrix.rs` | Copy one series into a scratch buffer without allocating a new Vec in the inner loop of `sbd_matrix_fd` |

## New vs Modified

### New files

| File | What it contains |
|------|-----------------|
| `fdars-core/src/metric/sbd.rs` | `SbdResult`, `sbd()`, `sbd_matrix_fd()` |
| `fdars-core/src/kshape.rs` | `KShapeConfig`, `KShapeResult`, `kshape_fd()`, `KShapeResult::predict()`, `sbd_kmedoids()` |
| `benches/kshape.rs` | Criterion benchmark group (Phase 63) |

### Modified files

| File | Change |
|------|--------|
| `fdars-core/src/metric/mod.rs` | Add `pub mod sbd;` + `pub use sbd::{sbd, sbd_matrix_fd, SbdResult};` |
| `fdars-core/src/lib.rs` | Add `pub mod kshape;`, re-export kshape + metric/sbd symbols |
| `fdars-core/src/prelude.rs` | Add `KShapeConfig`, `KShapeResult` |
| `Cargo.toml` (benchmarks section) | Add `[[bench]] name = "kshape"` entry |
| `.planning/research/GAP-BACKLOG.md` | Mark GAP-03 as promoted (per milestone convention) |

### Unchanged files (confirmed reused, not modified)

- `fdars-core/src/alignment/clustering.rs` — `kmedoids_from_distances` consumed as-is
- `fdars-core/src/shapelet/distance.rs` — `z_normalize_window` / `z_normalize_into` consumed as-is
- `fdars-core/src/kernel_kmeans.rs` — structural pattern mirrored, not modified
- `fdars-core/src/metric/gak.rs`, `soft_dtw.rs`, etc. — unchanged

## Anti-Patterns

### Anti-Pattern 1: Building a new z-normalization implementation

**What people do:** Implement z-normalization inline inside `sbd()` or `kshape.rs`.
**Why it's wrong:** `shapelet::distance` already has `z_normalize_into` (alloc-free, constant-window-guarded, population std) and `z_normalize_window` (allocating variant). Duplicating produces divergent constant-window handling and contradicts the project's deduplication convention established in v0.30.0.
**Do this instead:** Import `crate::shapelet::distance::{z_normalize_into, z_normalize_window}` directly.

### Anti-Pattern 2: Building the FFT planner inside the pairwise loop

**What people do:** Call `FftPlanner::new()` + `plan_fft_forward()` inside the inner loop of `sbd_matrix_fd` (i.e., once per pair).
**Why it's wrong:** `FftPlanner` caches twiddle-factor tables and is expensive to construct. The `fts/spectral.rs` idiom builds the planner once per function call. Over O(n^2) pairs this would repeat plan construction n*(n-1)/2 times.
**Do this instead:** Build the planner once before the loop in the sequential path; in the parallel path, each rayon closure captures or builds its own planner (not Sync).

### Anti-Pattern 3: Materializing the shifted series as a new Vec per pair in the centroid update

**What people do:** Allocate a new `Vec<f64>` for the cyclic shift of one series when computing the optimal-shift alignment during each centroid update iteration.
**Why it's wrong:** The centroid update runs for every cluster member every iteration; heap allocation in this loop is a performance hotspot for large n and m.
**Do this instead:** The shift is known from `SbdResult.shift`; apply it with modular index arithmetic into a reusable scratch buffer (pre-allocated once before the loop, refilled with `row_to_buf` + index offset).

### Anti-Pattern 4: Placing SBD logic inside kshape.rs (not in metric/)

**What people do:** Implement `sbd()` as a private helper inside `kshape.rs` rather than as a public primitive in `metric/sbd.rs`.
**Why it's wrong:** SBD is a reusable distance metric. It is the direct input to `sbd_kmedoids`, to hierarchical clustering, and to any future caller that wants shape-based distance without clustering. Burying it inside `kshape.rs` prevents that reuse and violates the metric-first layering pattern the codebase has followed since v0.32.0 (gak -> kernel_kmeans, soft_dtw -> barycenter consumer).
**Do this instead:** `metric/sbd.rs` as a public module, re-exported from `metric/mod.rs` and `lib.rs`.

### Anti-Pattern 5: A dedicated `sbd_kmedoids.rs` module file

**What people do:** Add a dedicated `src/sbd_kmedoids.rs` for the convenience adapter.
**Why it's wrong:** `sbd_kmedoids` is effectively a two-line function (call `sbd_matrix_fd` then `kmedoids_from_distances`). A dedicated file adds module boilerplate with negligible content, inconsistent with how similar thin adapters are handled elsewhere.
**Do this instead:** Keep `sbd_kmedoids` as a `pub fn` at the bottom of `kshape.rs` and re-export it from `lib.rs`.

## Sources

- Paparrizos & Gravano (2015), "k-Shape: Efficient and Accurate Clustering of Time Series" (SIGMOD) — SBD definition, shape-extraction centroid algorithm
- tslearn v0.9.0 `KShape` source — reference implementation for NCC normalization convention and centroid update formula
- `fdars-core/src/kernel_kmeans.rs` — structural mirror (config/result/n_init/predict/empty-cluster/seeding patterns)
- `fdars-core/src/alignment/clustering.rs` — `kmedoids_from_distances` entry point signature (unchanged consumer)
- `fdars-core/src/fts/spectral.rs` — `FftPlanner` idiom and `nalgebra::SymmetricEigen` usage pattern
- `fdars-core/src/shapelet/distance.rs` — `z_normalize_window` / `z_normalize_into` reuse target
- `fdars-core/src/metric/mod.rs` — `self_distance_matrix` helper, metric barrel structure, existing `pub use` pattern

---
*Architecture research for: k-Shape clustering & Shape-Based Distance in fdars-core*
*Researched: 2026-09-02*
