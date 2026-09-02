# Phase 63: SBD-based k-medoids & Wrap-up - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning
**Mode:** Smart-discuss (autonomous) — resolved from `.planning/research/` + Phase 61/62 API + the k-medoids API. No open user decisions. FINAL phase of milestone v0.34.0.

<domain>
## Phase Boundary

Deliver the SBD-based k-medoids convenience (a REAL public function, the user's 4th chosen deliverable), finalize the crate-root public re-exports for SBD + k-Shape, add them to the prelude, and add a criterion benchmark. Additive/non-breaking, no new dependency.

In scope (KSH-05):
- **`sbd_kmedoids`** — build the SBD distance matrix (Phase 61) then run the existing `kmedoids_from_distances`.
- **Crate-root re-exports** — finalize the flat public surface for `metric::sbd::{sbd, sbd_distance_matrix, SbdResult}` and `kshape::{kshape_fd, KShapeConfig, KShapeResult, sbd_kmedoids}`; add to `prelude.rs`.
- **Criterion benchmark** — `benches/kshape.rs`.

Out of scope: nothing further — this completes v0.34.0.
</domain>

<decisions>
## Implementation Decisions

### Resolved (from research + the k-medoids API)

1. **`sbd_kmedoids` (KSH-05):** `pub fn sbd_kmedoids(data: &FdMatrix, config: &KMedoidsConfig) -> Result<KMedoidsResult, FdarError>` (`#[must_use]`), living in `src/kshape.rs`. Implementation: `let dist = metric::sbd::sbd_distance_matrix(data)?; alignment::clustering::kmedoids_from_distances(&dist, config)`. This bridges the SBD distance (Phase 61) into the existing k-medoids solver — a shape-based clustering alternative to k-Shape. Reuses `KMedoidsConfig`/`KMedoidsResult` unchanged (both already crate-root re-exported at `lib.rs:278/290`).

2. **Crate-root re-exports (finalize):** in `src/lib.rs`, add:
   - `pub use metric::sbd::{sbd, sbd_distance_matrix, SbdResult};` (or extend an existing `pub use metric::{…}` block if one aggregates metric items — check; otherwise add a dedicated line).
   - `pub use kshape::{kshape_fd, sbd_kmedoids, KShapeConfig, KShapeResult};` (keep `pub mod kshape;`).
   Verify all names resolve from the crate root. Mirror how the v0.33.0 shapelet re-exports / v0.32.0 gak+kernel_kmeans re-exports were finalized in their last phase.

3. **Prelude:** add the same public SBD + k-Shape items to `src/prelude.rs` alongside the other clustering exports (kernel_kmeans, clustering).

4. **Benchmark:** add `fdars-core/benches/kshape.rs` (criterion, `harness = false`) benchmarking `sbd_distance_matrix` and `kshape_fd` on a SMALL synthetic set (e.g. n=30, m=64, k=3, n_init=2 to keep it quick); add a `[[bench]] name = "kshape"` entry to `fdars-core/Cargo.toml` (mirror the existing `[[bench]]` entries, e.g. the `shapelet` one added in v0.33.0).

5. **No behavior change to existing code** — purely additive. Do NOT modify existing signatures. Crate version stays 0.33.0 (the milestone-end bump to 0.34.0 is the orchestrator's ship step, NOT this phase).
</decisions>

<code_context>
## Existing Code Insights
- `src/alignment/clustering.rs:169`: `kmedoids_from_distances(dist_mat: &FdMatrix, config: &KMedoidsConfig) -> Result<KMedoidsResult, FdarError>` (validates square matrix, k≥1, k≤n). `KMedoidsConfig { k, seed, … }`, `KMedoidsResult` — both re-exported at `lib.rs:278/290`.
- Phase 61 `src/metric/sbd.rs`: `sbd_distance_matrix(data) -> Result<FdMatrix,FdarError>` (n×n symmetric, zero diagonal — exactly the k-medoids input).
- Phase 62 `src/kshape.rs`: `kshape_fd`, `KShapeConfig`, `KShapeResult` (add `sbd_kmedoids` here).
- `src/lib.rs`: crate-root `pub use` blocks (see the v0.33.0 shapelet block + `alignment::clustering` block L278/290 for the pattern); `src/prelude.rs`.
- `fdars-core/benches/*.rs` + their `[[bench]]` Cargo.toml entries (mirror one, e.g. `shapelet`).
- Conventions: `#[must_use]`, `Result<_,FdarError>`, additive/non-breaking.
</code_context>

<specifics>
## Specific Ideas (verification hooks)
Tests the plan must include:
- `test_sbd_kmedoids_recovers_groups`: two shifted-shape groups → `sbd_kmedoids` recovers them at high purity (proves it uses the SBD matrix, not L2/DTW).
- `test_sbd_kmedoids_uses_sbd_matrix`: the medoid assignment is consistent with `sbd_distance_matrix` + `kmedoids_from_distances` composed manually (i.e. `sbd_kmedoids` == the two-step composition).
- `test_sbd_kmedoids_validation`: k=0 / k>n → errors (propagated from `kmedoids_from_distances`).
- `test_kshape_reexports` (or a compile-level `use`): crate-root `use fdars_core::{sbd, sbd_distance_matrix, SbdResult, kshape_fd, KShapeConfig, KShapeResult, sbd_kmedoids};` resolves.
- Doctest on `sbd_kmedoids`.
- The new `benches/kshape.rs` must COMPILE cleanly under `cargo clippy --all-targets` (do not run it in the gate).
</specifics>

<deferred>
## Deferred Ideas
- Multivariate SBD / variable-length series / SBD in hierarchical clustering → future (KSH-BREADTH).
- Parallel-restart determinism tuning / SBD FFT-plan caching → later perf pass.
</deferred>
