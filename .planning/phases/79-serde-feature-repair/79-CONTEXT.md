# Phase 79: Serde Feature Repair - Context

**Gathered:** 2026-09-07
**Status:** Ready for planning
**Mode:** Mechanical build fix (infrastructure — discuss auto-scoped; approach fully determined by crate convention)

<domain>
## Phase Boundary

`cargo build --features serde` (and the CI feature set `--features linalg,parallel,serde`) compiles cleanly again, and a test + CI guard stop it silently re-breaking. Add the crate's conditional serde derives to the types that lack them (broken since Phase 60), plus a serde round-trip test. Additive/non-breaking — no behavior change, no new dependency.

</domain>

<decisions>
## Implementation Decisions

### Scope (confirmed by reproduction, not just the roadmap's "ClassifFit")
- The build fails on THREE types missing conditional serde derives, not one: `ClassifFit` (`fdars-core/src/classification/fit.rs:49`), `NonConformityScore` (conformal module), and `JointFpcaResult` (`fdars-core/src/elastic_fpca.rs:73`). Fix ALL of them, plus recursively any type they embed that also lacks the derive until `cargo build --features serde` AND `cargo build --features linalg,parallel,serde` both compile. Do not stop at the first three — chase the derive cascade to a clean build.

### Derive convention (Claude's Discretion — follow existing crate pattern)
- Use the crate's established form verbatim: `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` (as in `pda.rs`, `multi_fdata.rs`, and ~most public result types). Place it alongside the existing `#[derive(Debug, Clone, ...)]` on each type.
- If a type embeds a field whose foreign type cannot derive serde, prefer `#[serde(skip)]` / a documented shim ONLY as a last resort — first try adding the derive to the crate-owned embedded type. Note any skip in the SUMMARY.

### Re-breakage guard
- **CI guard already exists**: `rust-ci.yml:55` runs `cargo test --features linalg,parallel,serde` and line 77/110/154 also exercise serde — so CI has been RED on serde since Phase 60 and turns green once this compiles. Confirm this; optionally add an explicit serde-only `cargo build --features serde` step (the exact minimal broken config) as a fast guard. Claude's discretion on adding the extra step vs relying on the existing serde matrix.
- **Round-trip test**: add a `#[cfg(feature = "serde")]`-gated test that serializes → deserializes at least `ClassifFit` (the headline type) and asserts round-trip equality (`PartialEq`), exercising real serde codepaths so a future missing-derive regression fails a test, not just a build.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Conditional-serde pattern already used throughout: `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` (`pda.rs:64`, `multi_fdata.rs:41`, and most result structs). All target types already derive `Debug, Clone, PartialEq`, so round-trip equality assertions are available.
- `serde` + `serde_json` are already optional deps behind the `serde` feature.

### Established Patterns
- Conditional derive alongside standard derives; `#[non_exhaustive]` on public result structs (compatible with serde).

### Integration Points
- CI: `.github/workflows/rust-ci.yml` already has serde in its test/clippy/doc/coverage feature sets.
- Confirmed failing types: `ClassifFit`, `NonConformityScore`, `JointFpcaResult` (12 trait-bound errors) — plus any transitively-embedded crate types.

</code_context>

<specifics>
## Specific Ideas

- Root cause (MEMORY.md): Phase 60 (commit ea39c623) added `ShapeletTransformClassifier` embedding non-serde `ClassifFit`; the derive gap propagated. Additive fix — just add the missing derives.
- Build hazards carry over: clippy `--all-targets --features linalg,parallel,serde -- -D warnings` (note: CI adds a few `-A` allows for serde/clippy — see rust-ci.yml:77); `cargo fmt`; commit `--no-verify` (30s hook timeout on full suite).

</specifics>

<deferred>
## Deferred Ideas

None — mechanical build fix, stays within phase scope.

</deferred>
