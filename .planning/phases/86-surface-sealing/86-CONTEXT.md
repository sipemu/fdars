# Phase 86: Surface Sealing - Context

**Gathered:** 2026-09-08
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — smart discuss skipped)

<domain>
## Phase Boundary

The public interchange (`wire`) surface is removed from the public API and every public config struct is future-proofed against field additions without breaking external construction. Specifically:

- **SEAL-01**: `pub mod wire` → `pub(crate) mod wire`; ~24 public interchange types in `src/wire.rs` no longer externally reachable; any crate-root/prelude re-exports of wire types removed. Crate + all 28 examples + all doctests still compile.
- **SEAL-02**: Every public config struct not already carrying `#[non_exhaustive]` is marked `#[non_exhaustive]` (exact target set — public, not `pub(crate)`, not yet sealed — enumerated during planning; ~66 `*Config` defs exist, 15 already sealed).
- **SEAL-03**: Every struct sealed under SEAL-02 gets a non-literal construction escape hatch — `Default` + documented `..Default::default()` path (and/or builder) — so external code that can no longer use a `Config { .. }` literal has a supported construction path.

API-shape-only: no numeric or behavioral change. `fdars-core` only, no new crate dependency.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — this is a pure infrastructure/refactor phase. Guided by:
- The v0.41.0 sealing/dispatch pattern already established in the crate.
- ROADMAP success criteria and REQUIREMENTS (SEAL-01/02/03).
- Existing crate conventions (`.planning/codebase/CONVENTIONS.md`): all public types derive `Debug, Clone, PartialEq`; `#[non_exhaustive]` already used on result structs; column-major `FdMatrix`; `Result<T, FdarError>` returns.

Key constraints (from REQUIREMENTS / Out of Scope):
- Wire is sealed `pub(crate)`, NOT wired up as a supported public interchange API (no current JS/R consumer).
- Config escape hatches use `Default`/hand-written builders, NOT a derive-builder crate (no new dependency).
- Byte-identical behavior; SEAL-02 and SEAL-03 ship together (sealing a config without a construction path breaks external construction).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/wire.rs` — the ~24 interchange types to seal; check `src/lib.rs` and `src/prelude.rs` for re-exports to remove.
- `#[non_exhaustive]` is already an established pattern on public result structs (`FregreLmResult`, etc.) and 15 config structs.
- `Default` derive / manual `Default` impls already present on several config structs.

### Established Patterns
- Config structs are the builder-pattern types: `GmmClusterConfig`, `StlConfig`, `ConformalConfig`, `ClassifCvConfig`, `ElasticConfig`, `ElasticPcrConfig`, etc.
- `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` conditional serde on public types.

### Integration Points
- `src/lib.rs` root re-exports and `src/prelude.rs` — remove wire re-exports (SEAL-01).
- 28 `[[example]]` entries in `fdars-core/Cargo.toml` + doctests — must still compile (compile-time proof).
- `cargo build --features serde` must still compile.

</code_context>

<specifics>
## Specific Ideas

The exact set of config structs to seal (public, not `pub(crate)`, not already `#[non_exhaustive]`) must be enumerated during planning — do not assume the Phase 81 audit's ~22 estimate; the current tree has ~66 `*Config` definitions with 15 already sealed.

</specifics>

<deferred>
## Deferred Ideas

- Wiring `wire` up as a supported public interchange API — deliberately deferred; re-expose only when JS/R bindings need it.

</deferred>
