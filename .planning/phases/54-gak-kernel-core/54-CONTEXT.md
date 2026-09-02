# Phase 54: GAK Kernel Core - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning
**Mode:** Smart-discuss (autonomous) — grey areas resolved from `.planning/research/` (SUMMARY/FEATURES/ARCHITECTURE/PITFALLS). No open user decisions; scope was locked at milestone start.

<domain>
## Phase Boundary

Deliver a numerically-stable, positive-semi-definite Global Alignment Kernel (GAK) between curves — the correctness foundation for Phases 55 (Gram export) and 56 (kernel-k-means). New `src/metric/gak.rs` (sibling of `soft_dtw.rs`), re-exported at the crate root. Additive/non-breaking, no new crate dependency.

In scope (GAK-01/02/03/04):
- Pairwise GAK similarity via a **log-domain** (log-sum-exp) forward DP over the alignment lattice, triangular local Gaussian kernel with bandwidth σ.
- Triangular **normalization** `k(x,y)/sqrt(k(x,x)·k(y,y))` → similarity in `[0,1]`, unit self-similarity.
- n×n **Gram matrix** builder (`cdist_gak`-equivalent), symmetric + PSD, parallel via `iter_maybe_parallel!`.
- **σ median-distance heuristic** (`sigma_gak`-equivalent).

Out of scope this phase: Gram train/predict export API (Phase 55), kernel-k-means (Phase 56), native SVM (deferred SVM-01), triangular band truncation as a required parameter (ship full DP first).
</domain>

<decisions>
## Implementation Decisions

### Resolved (from research — treat as fixed unless planning surfaces a blocker)

1. **Module placement:** new `src/metric/gak.rs`, a sibling of `soft_dtw.rs`; add `pub mod gak;` to `src/metric/mod.rs` and re-export the public surface at the crate root (`src/lib.rs`), matching how `soft_dtw` items are surfaced. Only `metric/mod.rs` + `lib.rs` are modified; everything else is new.

2. **Log-domain DP is mandatory, not optional.** The forward recursion accumulates in log space:
   `L[i][j] = log_local(i,j) + logsumexp3(L[i-1][j], L[i-1][j-1], L[i][j-1])`
   where `log_local(i,j) = -(x_i - y_j)^2 / (2σ²) - log(2 - exp(-(x_i-y_j)²/(2σ²)))` (Cuturi's TGAK local kernel; the `2 - exp(...)` denominator is the "triangular"/infinitely-divisible correction). A **new** `logsumexp3(a,b,c)` helper is required — do NOT reuse `softmin3` (that is a soft-MIN; GAK needs a soft-MAX / log-sum-exp). Use the max-subtraction trick for stability, mirroring `softmin3`'s structure.

3. **Rolling-buffer DP.** Reuse the 2-row rolling-buffer pattern from `soft_dtw.rs` (O(m) memory) rather than a full O(m²) table where feasible; a full table is acceptable if it simplifies correctness (Phase 54 prioritizes correctness; a later perf pass can tighten memory).

4. **Normalization is mandatory for PSD.** The public similarity is always `exp(logGAK(x,y) - 0.5*(logGAK(x,x)+logGAK(y,y)))`, giving `k∈[0,1]`, `k(x,x)=1`. Unnormalized GAK is silently non-PSD and breaks downstream kernel-SVM — never expose it as the default.

5. **Public API shape (fdars conventions):**
   - `GakConfig { sigma: Option<f64> }` (config struct; `sigma=None` → use the median heuristic). Derive `Debug, Clone, PartialEq`; serde-gated; `Default`.
   - `pub fn gak(x: &[f64], y: &[f64], sigma: f64) -> f64` — normalized pairwise similarity (thin, panics-free scalar entry; validates σ>0 upstream in matrix fns).
   - `pub fn gak_gram_matrix(data: &FdMatrix, config: &GakConfig) -> Result<FdMatrix, FdarError>` — n×n symmetric PSD Gram (`cdist_gak` self-form). `#[must_use]`.
   - `pub fn sigma_gak(data: &FdMatrix) -> f64` — median-distance bandwidth heuristic.
   - Internal `pub(crate)`: `loggak(x,y,sigma)` (unnormalized log-kernel, reused by Phase 55 for the split-normalization export) and `logsumexp3`.
   - All fallible entry points return `Result<_, FdarError>`; dimension/σ validation at entry (σ>0, non-empty curves, matching layout).

6. **Symmetry by assignment.** Build the Gram by computing the upper triangle + diagonal once and mirroring (`G[j][i] = G[i][j]`), so symmetry is bit-exact (not merely within tolerance). Diagonal is exactly 1.0 after normalization.

7. **σ median heuristic:** σ = median pairwise Euclidean distance between curves (optionally scaled by a small constant, e.g. tslearn uses a factor tied to series length/κ — pick the tslearn@0.9.0 convention and document any divergence). Guard against σ=0 (degenerate/identical data) with a positive floor.

8. **Parallelism:** the Gram builder parallelizes the row/pair loop with `iter_maybe_parallel!` (gated by `parallel`); deterministic (no RNG in the kernel). Sequential build must be bit-identical to the parallel build.

### Claude's Discretion
Exact helper factoring, buffer-vs-table choice, and internal naming are at Claude's discretion within the above. Prefer reusing `soft_dtw.rs` structure to minimize review surface.
</decisions>

<code_context>
## Existing Code Insights

- `src/metric/soft_dtw.rs`: `pub(super) fn softmin3(a,b,c,gamma)` at L29 (the structural template for `logsumexp3`); `soft_dtw_distance` (L49) shows the DP + rolling buffer; `soft_dtw_self_1d`/`soft_dtw_cross_1d` (L88/L103) show the FdMatrix pairwise-matrix pattern to mirror for `gak_gram_matrix`.
- `src/metric/mod.rs`: has `self_distance_matrix` / `cross_distance_matrix` helpers and the 10-submodule `pub mod` + re-export pattern; add `gak` as the 11th.
- `src/parallel.rs`: `iter_maybe_parallel!` (and siblings) gate rayon by the `parallel` feature; deterministic per-thread seeding pattern exists but GAK needs no RNG.
- `src/matrix.rs`: column-major `FdMatrix` (rows=curves, cols=eval points); `row_to_buf`/`row_dot`/`row_l2_sq` for allocation-free row access.
- `src/error.rs`: `FdarError::{InvalidDimension, InvalidParameter, ComputationFailed}` for entry validation.
- Convention: `#[must_use]` on expensive computations; `Debug, Clone, PartialEq` on public types; serde-gated derives.
</code_context>

<specifics>
## Specific Ideas (verification hooks — from PITFALLS.md)

Tests the plan must include:
- `test_gak_no_underflow`: pairwise GAK on long series (m ≥ 100–400) returns off-diagonal > 1e-10 (proves log-domain, not raw-product).
- `test_gak_normalized_range`: all Gram entries in `[0,1]`; diagonal == 1.0 (±1e-12); NaN/Inf-free even for wholly dissimilar curves.
- `test_gak_gram_symmetric`: `G[i][j] == G[j][i]` bit-exact.
- `test_gak_gram_psd`: minimum eigenvalue of the Gram ≥ −1e-8 (symmetric eigendecomposition via nalgebra).
- `test_sigma_gak_healthy`: with the median-heuristic σ, off-diagonal Gram entries span a non-degenerate range (≈0.05–0.95), not near-identity or near-constant.
- `test_gak_vs_tslearn_reference`: matches tslearn@0.9.0 within 1e-6 on a small hand-checked dataset (hard-code the reference values).
- `test_gak_parallel_matches_sequential`: Gram is bit-identical under `parallel` on/off.
- Doctest on the primary public fn.
</specifics>

<deferred>
## Deferred Ideas

- Triangular band-width truncation (`triangular: Option<usize>`) as an optional performance optimization — ship the full untruncated DP first (correctness over speed this phase).
- Rolling-buffer memory tightening / SIMD — a later perf pass (out of scope for correctness phase).
- The train/predict split-normalization export (Phase 55) reuses `loggak` + the stored diagonal — leave the `pub(crate) loggak` seam in place for it.
</deferred>
