# Phase 105: Release Preparation & Verification - Context

**Gathered:** 2026-09-13
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Terminal release-preparation phase for milestone v0.45.0 (REL-01). Bump `fdars-core`
0.44.0 → 0.45.0, add a `[0.45.0]` entry to BOTH CHANGELOGs (workspace root + crate), check
off `SDTW-O1` on `documentation/ROADMAP-TO-1.0.md`, and pass all whole-crate gates green:
`cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`,
full `cargo test`, a `--features serde` build guard, all 28 examples + doctests, and
`cargo package`. The full-suite gate is the end-to-end proof of the milestone.

The `git tag v0.45.0` → crates.io publish is the DEFERRED operator step (GSD `git.create_tag`
stays OFF — the tag auto-publishes via `release.yml`). This phase does NOT tag or publish.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — this is a mechanical release-prep
phase. Use the milestone's shipped work (SDTW-01/02/03/04) to write the CHANGELOG entry and
follow the existing CHANGELOG format/version-bump conventions in the repo.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/Cargo.toml` — `version = "0.44.0"` → bump to `0.45.0`.
- Root CHANGELOG + crate CHANGELOG (`fdars-core/CHANGELOG.md`) — follow the existing
  `[X.Y.Z]` entry format (backfilled 0.31.0–0.44.0 in prior milestones).
- `documentation/ROADMAP-TO-1.0.md` — the 1.0 gap checklist; `SDTW-O1` line to check off.
  With SDTW-O1 cleared, only `fdars-j75` (R ecosystem) remains before the terminal 1.0-CUT.

### Established Patterns
- Release = bump `fdars-core/Cargo.toml` version → CHANGELOG entries → checklist checkoff →
  gates green → (operator) `git tag vX.Y.Z` → `release.yml` runs `cargo publish`.
- crates.io registry currently at 0.40.0/0.42.0/0.44.0 (0.41.0/0.43.0 never published,
  superseded); 0.45.0 will be the next published version.

### Integration Points
- Milestone shipped: SDTW-01 (in-crate L-BFGS), SDTW-02 (deterministic multi-restart),
  SDTW-03 (SoftDtwBarycenterConfig + non-breaking legacy wrapper), SDTW-04 (escape-local-
  minimum + f64-parity evidence). Strictly additive/non-breaking; no new crate dependency.

</code_context>

<specifics>
## Specific Ideas

- CHANGELOG [0.45.0] should summarize: replaced the fixed-step MM/soft-DBA barycenter descent
  with an in-crate L-BFGS optimizer + deterministic multi-restart wrapper, added
  `SoftDtwBarycenterConfig` + `soft_dtw_barycenter_with_config` (legacy 4-arg fn preserved as
  a thin non-breaking wrapper), added quality-evidence tests. Non-breaking, no new dependency.
- Gate battery MUST all pass; the full `cargo test` run is the milestone's end-to-end proof.
- Known repo hazards: pre-commit full-cargo hook times out (commit `--no-verify`); /tmp tmpfs
  small for serde doctest linking (route serde build target if needed); free
  `target/debug/{incremental,examples}` if /home is near full.
- Do NOT create a git tag or publish — deferred operator step.

</specifics>

<deferred>
## Deferred Ideas

- `git tag v0.45.0` → crates.io publish — DEFERRED operator step (not this phase).
- `fdars-j75` (R ecosystem FdMatrix migration) and the terminal 1.0-CUT — future milestones.

</deferred>
