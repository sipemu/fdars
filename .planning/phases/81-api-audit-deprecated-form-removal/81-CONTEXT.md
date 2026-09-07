# Phase 81: API Audit & Deprecated-Form Removal - Context

**Gathered:** 2026-09-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Audit the full public surface of `fdars-core` into a ranked, user-approved breaking-change inventory covering all four scopes (deprecated-form removal, accidental `pub` exposure, `#[non_exhaustive]` gaps, naming inconsistencies), and remove the 6 already-specified deprecated forms. The inventory is the source of truth for the concrete change sets consumed by Phases 82 (API-02/03) and 83 (API-04). No behavior/numeric change — API shape only.

</domain>

<decisions>
## Implementation Decisions

### Audit Aggressiveness & Deprecation Strategy
- **Naming-unification inventory (feeds Phase 83): Balanced** — recommend high-value renames (collapse the worst `_1d`/`_2d`/`_nd` sprawl + inconsistent config/result naming into consistent dispatchers), and explicitly flag low-value/marginal renames as optional/deferrable so the user can approve a reduced scope. Phase 83 is the largest, highest-risk item; the inventory must let the user right-size it.
- **Accidental-`pub` sealing (feeds Phase 82): Aggressive** — recommend sealing everything not deliberately part of the public API (`pub` → `pub(crate)`, drop unintended re-exports, hide leaked helper types). Rationale: a 1.0 should commit to the smallest defensible surface; easier to widen later than to narrow.
- **`#[non_exhaustive]` coverage (feeds Phase 82): Full** — audit every public enum and result struct, recommending `#[non_exhaustive]` on all that could plausibly grow fields/variants, so post-1.0 additions stay non-breaking.
- **Deprecated-form removal (API-01): Hard-remove now** — remove the 6 forms (`mean_2d`, `fanova`, `random_tukey_2d`, `random_projection_2d`, `fraiman_muniz_2d`, `modal_2d`) and their crate-root/prelude re-exports outright, no re-deprecation shim. Still 0.x, so breaking is permitted; migrate all internal callers, unit tests, doctests, and example 21 to the `Dim`/`_seeded` replacements.

### Approval Gate
- The ranked inventory is presented to the user for approval before Phases 82/83 draw their concrete change sets. The user may approve a reduced scope (especially for naming). Each entry lists location, proposed change, blast radius (internal callers, examples, doctests), and a value/risk rating.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- The 6 deprecated forms and their `Dim`/`_seeded` replacements already exist in the crate (deprecation was staged in a prior milestone) — API-01 is mechanical migration + removal.
- Column-major `FdMatrix` layout and the feature-gated parallelism model are stable and out of scope.

### Established Patterns
- All public types derive `Debug, Clone, PartialEq`; result structs already use `#[non_exhaustive]` in many places — the audit closes the gaps.
- Barrel `mod.rs` files use explicit `pub use` (no wildcards) — accidental-pub leaks are traceable via these re-export lists plus direct `pub` scans.
- Gates: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, `--features serde` build.

### Integration Points
- Crate-root (`src/lib.rs`) and `src/prelude.rs` re-exports are where deprecated-form and accidental-pub exposure surface.
- All 28 examples + doctests are the compile-time proof; example 21 specifically uses a deprecated form.

</code_context>

<specifics>
## Specific Ideas

- The audit inventory should be written as a durable artifact (markdown) so Phases 82/83 can reference the approved entries by ID.
- Historical build/CI hazards apply (per MEMORY.md): run clippy with `--all-targets --features linalg,parallel`; run `cargo fmt` per commit; keep `--features serde` green; watch `/tmp` and `target/` disk pressure on full builds.

</specifics>

<deferred>
## Deferred Ideas

- Low-value naming renames the user declines at the approval gate → fold into the STAB-03 1.0 gap checklist (Phase 84) rather than forcing them into Phase 83.
- The `fdars-r` FdMatrix migration (`fdars-j75`) is explicitly out of scope — external package.

</deferred>
