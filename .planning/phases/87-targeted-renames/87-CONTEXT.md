# Phase 87: Targeted Renames - Context

**Gathered:** 2026-09-08
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure/rename phase — smart discuss skipped)

<domain>
## Phase Boundary

The small, low-blast-radius naming inconsistencies are resolved. API-shape-only, no numeric/behavioral change; every consolidated signature routes to a **byte-identical private `_impl`** body (the v0.41.0 pattern; code-review gate confirms no numeric drift).

- **NAME-01**: `geometric_median_1d` / `geometric_median_2d` (`src/fdata.rs`) → one `Dim`-dispatched signature.
- **NAME-02**: `hausdorff_*` family (`src/metric/hausdorff.rs`: `hausdorff_self_1d`/`_2d`, `hausdorff_cross_1d`/`_2d`) → `Dim`-dispatched signature(s). NOTE: `hausdorff_3d` operates on `&[(f64,f64,f64)]` point clouds — a different signature; planning must decide whether it stays separate (likely yes — it is not a `_1d`/`_2d` grid pair).
- **NAME-03**: `functional_spatial_1d`/`_2d` and `kernel_functional_spatial_1d`/`_2d` (`src/depth/spatial.rs`) → `Dim`-dispatched signatures.
- **NAME-05**: `LpeerResult` → `LocalPeerResult` everywhere (definition, references, `lib.rs`/`prelude.rs` re-exports, examples, doctests). ~15 references in src alone.

`fdars-core` only, no new crate dependency. Depends on Phase 86 (done).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
Pure infrastructure/refactor phase — all choices at Claude's discretion, guided by:
- **The v0.41.0 `Dim`-dispatch pattern** (`src/dim.rs` defines `enum Dim`). Replicate whatever an already-consolidated v0.41.0 function does: a public `fn foo(dim: Dim, …)` (or `Dim`-carrying arg) that matches on `Dim` and delegates to byte-identical private `foo_1d_impl` / `foo_2d_impl` bodies. Research must surface a concrete already-consolidated example to copy exactly.
- The 1d and 2d forms have **different argument lists** (2d needs a second-dimension argvals). The consolidated signature must accommodate both — follow the established v0.41.0 convention (likely `Dim` enum carries the grid/second-argvals, or the public fn takes a superset). Do not invent a new convention.
- Byte-identical `_impl` bodies: move the existing 1d/2d bodies verbatim into private `_impl` fns; the public fn only dispatches. A code-review gate confirms zero numeric drift.

Key constraints (from REQUIREMENTS / STATE):
- Old lone `_1d`/`_2d` public forms are REMOVED (breaking — intended for v0.42.0).
- All external construction/call sites (28 examples + doctests + `tests/`) updated to the new signatures (compile-time proof).
- Sequenced BEFORE Phase 88 (the large high-blast-radius suffix batch). This phase is the small, low-blast-radius subset.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/dim.rs:21` — `enum Dim` (the dispatch vehicle).
- Prior v0.41.0 consolidations already exist in the tree — find one and copy its exact shape (public dispatch fn + private `_impl` bodies).

### Target sites (from recon)
- `src/fdata.rs:1076` `geometric_median_1d`, `:1101` `geometric_median_2d`.
- `src/metric/hausdorff.rs`: `hausdorff_self_1d`(46), `hausdorff_cross_1d`(63), `hausdorff_3d`(81 — separate), `hausdorff_self_2d`(137), `hausdorff_cross_2d`(148).
- `src/depth/spatial.rs`: `functional_spatial_1d`(18), `functional_spatial_2d`(77), `kernel_functional_spatial_1d`(199), `kernel_functional_spatial_2d`(219).
- `LpeerResult` — ~15 refs in `src/` + refs in examples/tests; check `lib.rs`, `prelude.rs`, `peer.rs`.

### Integration Points
- `src/lib.rs` + `src/prelude.rs` re-exports (both the renamed type and any re-exported `_1d`/`_2d` fns that collapse).
- 28 examples + doctests + `tests/` — external call sites of the old names must be updated.

</code_context>

<specifics>
## Specific Ideas

Research MUST produce: (1) the exact v0.41.0 `Dim`-dispatch shape from an already-consolidated function (verbatim signature + dispatch body), (2) the full enumeration of every external caller (examples/doctests/tests) of each old name, and (3) a decision on `hausdorff_3d` (keep separate vs fold).

</specifics>

<deferred>
## Deferred Ideas

- NAME-04 (the large lone-`_1d`/`_2d` suffix batch across the rest of the crate) — Phase 88, deliberately isolated as the high-blast-radius edit.

</deferred>
