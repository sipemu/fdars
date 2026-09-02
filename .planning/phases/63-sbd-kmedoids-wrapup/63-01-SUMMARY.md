# Phase 63 — Plan 63-01 SUMMARY (KSH-05: SBD-based k-medoids & Wrap-up)

**Milestone:** v0.34.0 (FINAL phase) · **Requirement:** KSH-05 · **Status:** Complete
**Impl commit:** `53e8ab12`

## Files created / modified
- **M** `fdars-core/src/kshape.rs` — added `sbd_kmedoids` public fn + 4 new inline tests; imports now pull `kmedoids_from_distances`, `KMedoidsConfig`, `KMedoidsResult` from `crate::alignment` and `sbd_distance_matrix` from `crate::metric::sbd`.
- **M** `fdars-core/src/lib.rs` — extended the `pub use metric::{…}` block with `sbd, sbd_distance_matrix, SbdResult`; added `pub use kshape::{kshape_fd, sbd_kmedoids, KShapeConfig, KShapeResult};`.
- **M** `fdars-core/src/prelude.rs` — extended `pub use crate::metric::{…}` with `sbd, sbd_distance_matrix, SbdResult`; added `pub use crate::kshape::{kshape_fd, sbd_kmedoids, KShapeConfig, KShapeResult};`.
- **C** `fdars-core/benches/kshape.rs` — criterion bench (`harness = false`) of `sbd_distance_matrix` (n=30, m=64) and `kshape_fd` (k=3, n_init=2).
- **M** `fdars-core/Cargo.toml` — added `[[bench]] name = "kshape"`.
- **C** `.planning/phases/63-sbd-kmedoids-wrapup/63-01-PLAN.md`.

## Public API added
```rust
// src/kshape.rs
#[must_use]
pub fn sbd_kmedoids(data: &FdMatrix, config: &KMedoidsConfig)
    -> Result<KMedoidsResult, FdarError>;
// = { let dist = sbd_distance_matrix(data)?; kmedoids_from_distances(&dist, config) }
```
Reuses `KMedoidsConfig` / `KMedoidsResult` unchanged. No new dependency, no signature change, crate version stays 0.33.0.

### Crate-root re-exports finalized (v0.34.0 surface)
- `pub use metric::{…, sbd, sbd_distance_matrix, …, SbdResult, …};`
- `pub use kshape::{kshape_fd, sbd_kmedoids, KShapeConfig, KShapeResult};`
- Prelude gains the same SBD + k-Shape items.

## Tests + results
New inline tests in `kshape.rs`:
- `test_sbd_kmedoids_recovers_groups` — two shifted-shape groups, purity ≥ 0.9 (proves SBD backend). PASS
- `test_sbd_kmedoids_uses_sbd_matrix` — output == manual `sbd_distance_matrix` + `kmedoids_from_distances` (labels, medoids, total distance bit-identical). PASS
- `test_sbd_kmedoids_validation` — k=0 and k>n → `InvalidParameter`. PASS
- `test_kshape_reexports` — all new crate-root items resolve (compile-level). PASS
- Doctest on `sbd_kmedoids` — shows explicit SBD-matrix → k-medoids flow + equivalence; uses external `use fdars_core::{…}` (covers out-of-crate resolution). PASS

Counts: `linalg` feature → 11 kshape lib tests + 2 kshape doctests green; default feature → same 11 + 2 green.

## Gate tails
- `cargo fmt --check` → clean (`FMT_OK`).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → `Finished` (bench compiled, no warnings).
- `cargo test -p fdars-core --features linalg kshape` → `ok. 11 passed; 0 failed`.
- `cargo test -p fdars-core kshape` (default) → `ok. 11 passed; 0 failed`.
- Doctests (both feature sets) → `ok. 2 passed; 0 failed`.

## Divergences
- 63-CONTEXT.md specified the import path `alignment::clustering::kmedoids_from_distances`, but `alignment::clustering` is a **private** module. Used the public re-export `crate::alignment::{kmedoids_from_distances, KMedoidsConfig, KMedoidsResult}` instead (same items, identical behavior).
- `test_kshape_reexports` uses `crate::` paths (inline lib tests cannot self-reference `fdars_core::`); the external `use fdars_core::{…}` resolution is covered by the `sbd_kmedoids` doctest (compiled as an out-of-crate binary).
- Bench: `KShapeConfig` is `#[non_exhaustive]`, so the bench (a separate crate) builds it via `KShapeConfig::new(3)` then sets public fields rather than a struct literal.

## Open concerns
None. Milestone v0.34.0 code work complete; version bump to 0.34.0 + tag/publish is the orchestrator's ship step.
