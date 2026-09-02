---
phase: 63
title: SBD-based k-medoids & Wrap-up
requirement: KSH-05
status: passed
milestone: v0.34.0
impl_commit: 53e8ab12
---

# Phase 63 Verification — SBD-based k-medoids & Wrap-up (KSH-05)

Each of the 4 ROADMAP success criteria → PASS/FAIL with evidence.

## Criterion 1 — `sbd_kmedoids` is a real public function (SBD matrix → k-medoids)
**PASS.** `pub fn sbd_kmedoids(data: &FdMatrix, config: &KMedoidsConfig) -> Result<KMedoidsResult, FdarError>` added in `src/kshape.rs` (`#[must_use]`), body builds the SBD distance matrix via `sbd_distance_matrix(data)?` and feeds `kmedoids_from_distances(&dist, config)`. Returns the standard `KMedoidsResult` (medoid indices + labels). Not a doc example — a compiled public fn, re-exported at the crate root. `test_sbd_kmedoids_recovers_groups` exercises it end-to-end (medoid_indices.len()==2, labels.len()==16).

## Criterion 2 — Provably uses SBD (not L2/DTW); integration test + doctest
**PASS.** `test_sbd_kmedoids_uses_sbd_matrix` asserts `sbd_kmedoids(data,cfg)` produces bit-identical `labels`, `medoid_indices`, and `total_within_distance` to the manual composition `kmedoids_from_distances(&sbd_distance_matrix(data)?, cfg)`. `test_sbd_kmedoids_recovers_groups` recovers two **circularly-shifted** shape groups at purity ≥ 0.9 — a shift-sensitive L2/DTW backend could not. The `sbd_kmedoids` doctest shows the explicit `sbd_distance_matrix` → `kmedoids_from_distances` flow and asserts equivalence, making the SBD backend unambiguous.

## Criterion 3 — Full v0.34.0 surface re-exported additively; prelude updated; non-breaking
**PASS.** Crate root (`src/lib.rs`): `pub mod kshape;` retained; `pub use kshape::{kshape_fd, sbd_kmedoids, KShapeConfig, KShapeResult};` added; `pub use metric::{…, sbd, sbd_distance_matrix, SbdResult, …};` extended. Prelude (`src/prelude.rs`) gains the same SBD + k-Shape items (incl. `KShapeConfig`/`KShapeResult`). `test_kshape_reexports` confirms all items resolve; the doctest resolves them via external `use fdars_core::{…}`. Purely additive: no existing signature changed, no new dependency, crate stays 0.33.0. Whole-crate `clippy --all-targets` compiles cleanly (examples/bench/tests) → 28 examples + WASM/R bindings unaffected.

## Criterion 4 — Criterion benchmark added; whole-crate gates pass
**PASS.** `benches/kshape.rs` (criterion, `harness = false`) benchmarks `sbd_distance_matrix` (n=30, m=64) and `kshape_fd` (k=3, n_init=2) on a small synthetic set; `[[bench]] name = "kshape"` added to `Cargo.toml`. Gates:
- `cargo fmt --check` → clean.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → `Finished`, no warnings (bench compiled).
- `cargo test -p fdars-core --features linalg kshape` → 11 lib + 2 doc pass.
- `cargo test -p fdars-core kshape` (default) → 11 lib + 2 doc pass.

## Verdict
**status: passed** — all 4 criteria PASS. Milestone v0.34.0 code deliverables complete; version bump/tag/publish deferred to the orchestrator's ship step.
