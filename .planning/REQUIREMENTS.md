# Requirements: fdars v0.32.0 — Global Alignment Kernel & Kernel Clustering

**Defined:** 2026-09-02
**Core Value:** Add a PSD Global Alignment Kernel for curve sets and the kernel machinery it unlocks — clustering natively on curves and enabling external precomputed-kernel SVMs via an exported Gram matrix. Promotes GAP-01 (top-ranked, score 3.00) from the v0.31.0 `GAP-BACKLOG.md`.

## Milestone Requirements

Requirements for v0.32.0. Each maps to a roadmap phase. Implementation milestone — real `fdars-core/src/` changes, additive/non-breaking, no new crate dependency. Reference baseline: tslearn@0.9.0 (`gak`, `cdist_gak`, `sigma_gak`, `KernelKMeans`); Cuturi 2011 (Triangular GAK).

### GAK Kernel Core

- [x] **GAK-01**: User can compute the pairwise GAK similarity between two curves via a log-domain (log-sum-exp) forward DP over the alignment lattice, using a triangular local Gaussian kernel with bandwidth σ. Log-domain is mandatory — the raw-product recursion underflows to 0/NaN for series longer than ~50 points.
- [x] **GAK-02**: The GAK similarity is normalized by `sqrt(k(x,x)·k(y,y))` to yield a valid similarity in `[0,1]` with unit self-similarity (`k(x,x)=1`). Normalization is mandatory for positive-semi-definiteness — unnormalized GAK silently breaks kernel-SVM.
- [x] **GAK-03**: User can compute an n×n GAK Gram matrix over a curve set (`cdist_gak`-equivalent), guaranteed symmetric and PSD with unit diagonal, parallelized via the existing `iter_maybe_parallel!` machinery under the `parallel` feature.
- [x] **GAK-04**: User can auto-select the GAK bandwidth σ from a curve set via the median-distance heuristic (`sigma_gak`-equivalent), so a sensible kernel width is available without manual tuning.

### Gram-Matrix Export (external precomputed-kernel SVM)

- [x] **GAK-05**: User can export a training Gram matrix (n_train × n_train, symmetric PSD, unit diagonal) suitable for a precomputed-kernel SVM (`SVC(kernel='precomputed')` convention).
- [x] **GAK-06**: User can export a prediction Gram matrix (n_test × n_train) whose entries use the correct cross-normalization against the stored training self-kernels, so an external SVM trained on GAK-05 can score new curves in the same feature space. A split train/predict API prevents the silent self-kernel-normalization bug.

### Kernel-k-means Clustering

- [x] **GAK-07**: User can cluster a curve set with kernel-k-means through the GAK kernel — assignments computed from Gram-matrix kernel distances (no explicit centroid curve), with `n_init` random-partition restarts, empty-cluster recovery, and deterministic per-restart RNG seeding (`seed + restart_idx`).
- [x] **GAK-08**: User can assign new (out-of-sample) curves to a fitted kernel-k-means model via a `predict` path that reuses the same GAK kernel and normalization as the fit.

## Future Requirements

Deferred to a later milestone; tracked but not in this roadmap.

### Native kernel machines

- **SVM-01**: A native in-crate kernel-SVM classifier (SMO/QP solver) consuming the GAK kernel directly, removing the external-SVM round-trip. Deferred — much larger than "effort S"; the exported Gram matrix (GAK-05/06) covers the use case in the interim.

### Kernel-method breadth

- **KRN-01**: Additional curve kernels (e.g. RBF-on-features, other alignment kernels) and kernel-SVM/kernel-PCA consumers reusing the GAK Gram infrastructure.

## Out of Scope

Explicitly excluded for v0.32.0. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Native kernel-SVM classifier | Out of the promoted GAP-01 scope; deferred to SVM-01. The Gram-matrix export (GAK-05/06) is the realistic interpretation of "kernel-SVM glue" and delivers the capability without an in-crate SVM. |
| GPU / batched-broadcast GAK kernels | fdars targets a portable CPU/WASM numeric core (recorded OOS-01 in `GAP-BACKLOG.md`). |
| Other GAP-BACKLOG items (GAP-02/03/05/06/07/08) | Shapelets, k-Shape, FOptDes, PEER, wavelet regression, differentiable core — carry forward to future milestones, drawn top-first. |
| Triangular band-width truncation as a required parameter | The Cuturi triangular constraint may be exposed as an optional optimization, but the milestone ships the full (untruncated) log-domain DP first; a mandatory band parameter is not required for correctness. |

## Traceability

Mapped during roadmap creation (2026-09-02). Three-phase dependency spine: 54 (kernel core) → 55 (Gram export) → 56 (kernel-k-means).

| Requirement | Phase | Status |
|-------------|-------|--------|
| GAK-01 | Phase 54 | Complete |
| GAK-02 | Phase 54 | Complete |
| GAK-03 | Phase 54 | Complete |
| GAK-04 | Phase 54 | Complete |
| GAK-05 | Phase 55 | Complete |
| GAK-06 | Phase 55 | Complete |
| GAK-07 | Phase 56 | Complete |
| GAK-08 | Phase 56 | Complete |

**Coverage:**

- Milestone requirements: 8 total
- Mapped to phases: 8 ✓ (Phase 54: GAK-01/02/03/04 · Phase 55: GAK-05/06 · Phase 56: GAK-07/08)
- Unmapped: 0 ✓ (no orphans, no duplicates)

---
*Requirements defined: 2026-09-02*
*Last updated: 2026-09-02 after roadmap creation (traceability populated)*
