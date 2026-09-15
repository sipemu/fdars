# Phase 111: Release Preparation & Verification - Context

**Gathered:** 2026-09-14
**Status:** Ready for planning
**Mode:** Auto-generated (terminal release phase — discuss skipped)

<domain>
## Phase Boundary

Terminal phase. Bump the crate 0.45.0 → 0.46.0, record the milestone in both
CHANGELOGs (root + crate), and verify every whole-crate gate is green — leaving
`fdars-core` release-ready. The `git tag v0.46.0` → crates.io publish is the
DEFERRED operator step (GSD `git.create_tag` is off because `release.yml`
couples tag-push to publish) — never performed inside a phase.

</domain>

<decisions>
## Implementation Decisions

### Fixed constraints
- Bump `fdars-core/Cargo.toml` `version` 0.45.0 → 0.46.0; refresh `Cargo.lock`.
- `[0.46.0]` entries in BOTH `CHANGELOG.md` (root) and `fdars-core/CHANGELOG.md`,
  matching the existing Keep-a-Changelog style; describe the quality-hardening
  work (complexity/entropy/SATD/unwrap + CI guardrail), strictly non-breaking +
  numeric-identical, no new crate dependency.
- **Release gate battery (all must be green):** `cargo fmt --check`,
  `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full
  `cargo test`, a `--features serde` build guard, all 28 examples + doctests,
  `cargo package`, **and** the curated `pmat quality-gate --checks
  complexity,satd,dead-code,duplicates` (0 violations).
- **Do NOT** `git tag` or publish — deferred operator step.

</decisions>

<code_context>
## Existing Code Insights

- Version lives only in `fdars-core/Cargo.toml` (workspace root has no package
  version). Two CHANGELOGs: `CHANGELOG.md` (root) + `fdars-core/CHANGELOG.md`.
- The serde build had a historical break (Phase 60) but was repaired by v0.40.0
  and builds clean now (re-verified this phase).
- 28 `[[example]]` entries in `fdars-core/Cargo.toml`.
- `release.yml` publishes to crates.io on a `v*` tag push (why tagging is deferred).

</code_context>

<specifics>
## Specific Ideas

Release-ready = crate at 0.46.0, CHANGELOGs updated, full gate battery green
including the curated pmat gate. Publish is manual/deferred.

</specifics>

<deferred>
## Deferred Ideas

- `git tag v0.46.0` → crates.io publish (operator step, out of phase scope).

</deferred>
