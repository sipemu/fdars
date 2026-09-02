# Research Summary: Global Alignment Kernel + Kernel-K-Means + Gram-Matrix Export (v0.32.0)

**Project:** fdars-core (Rust functional-data-analysis library)
**Milestone:** v0.32.0 — GAK kernel implementation + kernel-k-means clustering + Gram-matrix export for external SVM
**Researched:** 2026-09-02
**Confidence:** HIGH (mathematical spec from primary sources, API verified against tslearn@0.9.0 and Cuturi 2011, architecture patterns validated against codebase)

---

## Executive Summary

fdars v0.32.0 adds three interconnected deliverables: a Global Alignment Kernel (GAK) for time-series similarity in log-domain, kernel-k-means clustering operating purely on precomputed Gram matrices, and Gram-matrix export for external precomputed-kernel SVM workflows. The research converges on a critical finding: **all three features can be implemented entirely using the existing dependency stack** — no new crates required. The architecture is additive and non-breaking, placing GAK in `src/metric/gak.rs` (sibling to `soft_dtw.rs`) and kernel-k-means in `src/kernel_kmeans.rs` (top-level, like `clustering.rs`).

The research identifies two mandatory implementation decisions that must be nailed from the start: **log-domain accumulation is non-negotiable** (raw products underflow to zero for series longer than ~50 samples, a known deficiency in tslearn 0.9.0), and **triangular normalization is required for positive semi-definiteness** (raw unnormalized GAK is not PSD and breaks kernel-SVM silently). Beyond these core constraints, the implementation is straightforward — it reuses the 2-row rolling-buffer DP pattern from existing `soft_dtw.rs`, mirrors the pairwise-matrix parallelism from `metric/mod.rs`, and follows fdars conventions for config structs, result types, and deterministic seeding.

Roadmap risks are well-characterized: floating-point asymmetry can produce non-symmetric Gram matrices, wrong sigma selection degenerates the Gram to near-identity or near-constant, and kernel-k-means requires n_init restarts + empty-cluster recovery (no k-means++ initialization, since GAK values are similarities not distances). All pitfalls have low-cost recovery strategies embedded in the test suite.

---

## Key Findings

### Recommended Stack

**No new dependencies required.** All three v0.32.0 deliverables build entirely on fdars' existing stack: Rust 1.81+, nalgebra, rayon (optional), rand, and the column-major `FdMatrix` type. rustfft is irrelevant (GAK is pure O(n·m) DP, not FFT-based). faer and linalg features are not involved in the GAK/kernel-k-means path — both algorithms operate on scalar `f64` arithmetic and Gram matrix entries.

**Core technologies used by v0.32.0:**
- **Rust (1.81 MSRV)** — All implementation; MSRV stays unchanged
- **nalgebra 0.33** — Not used by GAK core
- **rayon 1.10** — Parallelizes Gram-matrix row-pair loops via `iter_maybe_parallel!`
- **rand 0.8** — Seeds reproducible restarts in kernel-k-means
- **FdMatrix** — Gram matrix output type
- **`softmin3` pattern from `soft_dtw.rs`** — Reused for log-sum-exp stabilization

### Expected Features

**Must have (table stakes, v0.32.0):**
1. Log-domain GAK recursion — unnormalized kernel via DP in log-space; avoids underflow for series > ~50
2. Normalized GAK kernel — triangular normalization guarantees PSD for kernel machines
3. Gram-matrix builders — `gak_gram_train()` (n×n) and `gak_gram_test()` (n_test×n_train)
4. Kernel-k-means — `kernel_kmeans()` with n_init restarts and empty-cluster recovery
5. Cluster prediction — `KernelKMeansResult::predict()` for new curves
6. Configuration structs — `GakConfig`, `KernelKMeansConfig` following fdars conventions
7. Sigma heuristic — `gak_sigma_median()` for automatic bandwidth selection
8. Series-length validation — guard against invalid series length ratios

**Should have (low-cost differentiators):**
- Rayon parallelism on Gram-matrix computation
- Optional triangular band constraint
- Deterministic seeding for reproducible clustering
- `#[must_use]` annotations
- Serde support via `#[cfg_attr]`

**Defer to v0.33+:**
- Native kernel SVM (requires QP solver)
- Multivariate curve support
- Kernel PCA via GAK
- Online/streaming kernel-k-means
- Wavelet/FFT backend

### Architecture Approach

GAK integrates cleanly with zero breaking changes. **Module placement:** GAK in `src/metric/gak.rs` (sibling to `soft_dtw.rs`), kernel-k-means in `src/kernel_kmeans.rs` (top-level, like `clustering.rs`).

**Reuse from existing code:**
- 2-row rolling-buffer from `soft_dtw_distance()` 
- Log-sum-exp stabilization from `softmin3` pattern
- Pairwise matrix parallelism from `self_distance_matrix()` / `cross_distance_matrix()`
- Random seeding pattern from elastic-FPCA: `StdRng::seed_from_u64(seed + restart_idx)`
- Config/result struct conventions (97 existing types)

**New vs modified files:** Only 2 new files (`metric/gak.rs`, `kernel_kmeans.rs`); 2 minor modifications (`metric/mod.rs`, `lib.rs`). No existing public API changes.

### Critical Pitfalls (Top 5)

1. **Log-domain accumulation mandatory** — Raw DP underflows for m > ~50; implement in log-space from day one
2. **Only normalized triangular GAK is PSD** — Unnormalized form has negative eigenvalues, breaks kernel-SVM silently
3. **Floating-point asymmetry breaks symmetry** — Different evaluation order produces `G[i,j] ≠ G[j,i]`; symmetrize by assignment
4. **Wrong sigma silently degenerates Gram** — Too small → near-identity; too large → rank-1; provide heuristic + sensitivity test
5. **Kernel-k-means needs n_init restarts** — GAK values are similarities (not distances), so k-means++ weighting inverted; use random uniform restarts

All pitfalls have detection tests and low-cost recovery documented in PITFALLS.md.

---

## Implications for Roadmap

**Suggested 3-phase decomposition:**

### Phase 54: GAK Kernel Core
**Rationale:** All downstream features depend on correct log-domain GAK. Resolve Pitfalls 1-4.
**Delivers:** `gak()`, `logsumexp3`, `gak_sigma_median()`, `GakConfig`, comprehensive tests
**Blocks:** Phase 55, 56

### Phase 55: Gram-Matrix Export (SVM Glue)
**Rationale:** Wraps proven kernel into SVM-export interface. Split train/predict API prevents normalization bugs.
**Delivers:** `gak_gram_train()`, `gak_gram_test()`, parallel construction, rustdoc example
**Depends on:** Phase 54

### Phase 56: Kernel-K-Means Clustering
**Rationale:** Final consumer. Multi-restart seeding + empty-cluster recovery. Resolves Pitfalls 5-7.
**Delivers:** `kernel_kmeans()`, `KernelKMeansResult`, `predict()`, `KernelKMeansConfig`
**Depends on:** Phase 54, 55

### Phase Ordering Rationale

Phase 54 first (core kernel must work), Phase 55 second (mechanical API wrapping), Phase 56 third (clustering consumer). No algorithmic risk in Phase 55/56; all risk front-loaded into Phase 54.

### Research Flags

**Phases needing research during planning:**
- **Phase 54:** Sigma heuristic sensitivity on real fdars curves — add phase-exit criterion
- **Phase 56:** Kernel-k-means++ vs random restarts — lightweight experiment to decide

**Phases with standard patterns (skip research):**
- **Phase 54:** DP, log-sum-exp, parallelism all reuse existing patterns
- **Phase 55:** Mechanical API wrapping, mirrors existing structure
- **Phase 56:** Config/result struct conventions well-established

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| **Stack** | HIGH | No new deps; all existing. Verified against Cargo.toml and soft-DTW patterns. |
| **Features** | HIGH | Cuturi 2011 primary, tslearn@0.9.0 + scikit-learn official docs cross-checked. All table-stakes well-defined. |
| **Architecture** | HIGH | Direct codebase reading. Zero conflicts. Module placement reasoning sound. |
| **Pitfalls** | HIGH | Derived from math analysis, tslearn known bugs, scikit-learn SVC contract, fdars codebase. All have detection tests and recovery. |

**Overall: HIGH**

### Gaps to Address

1. **Sigma heuristic on real fdars data** — Cuturi formula for normalized unit-variance series may not fit real FDA curves. Phase 54 planning includes sensitivity analysis on representative datasets.

2. **Kernel-k-means initialization** — Random uniform restarts (current plan) vs kernel-k-means++. Lightweight experiment during Phase 56 planning informs final choice.

3. **Series-length ratio guard (2:1)** — Cuturi enforces this for TGAK. Phase 54 planning confirms whether hard guard or relax.

4. **PSD eigenvalue test** — Propose nalgebra eigendecomposition; fdars doesn't currently use this. Phase 54 planning clarifies scope.

All gaps non-blocking for roadmap. Phase planning makes final decisions.

---

## Sources

### Primary (HIGH confidence)

- **Cuturi, M. (2011).** "Fast Global Alignment Kernels." ICML 2011. https://icml.cc/2011/papers/489_icmlpaper.pdf
- **Dhillon, I., Guan, Y., Kulis, B. (2004).** "Kernel k-Means, Spectral Clustering and Normalized Cuts." KDD 2004.
- **tslearn@0.9.0 official documentation.** https://tslearn.readthedocs.io/en/stable/
- **scikit-learn SVC precomputed-kernel documentation.** https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- **fdars-core source code:** `src/metric/soft_dtw.rs`, `src/metric/mod.rs`, `src/clustering.rs`, `Cargo.toml`

### Secondary (MEDIUM confidence)

- **tslearn@0.9.0 source code (GitHub).** Log-domain DP, sigma heuristic implementation.
- **R dtwclust package documentation.** GAK function, triangular band constraint.
- **GAP-BACKLOG.md (v0.31.0).** GAP-01 scope.

### Tertiary (LOW confidence)

- Pattern inference from fdars modules (elastic alignment, classification).

---

*Research completed: 2026-09-02*
*Status: Ready for roadmap creation*
