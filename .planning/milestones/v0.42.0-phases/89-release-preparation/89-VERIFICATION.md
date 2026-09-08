---
phase: 89-release-preparation
verified: 2026-09-08T21:40:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 89: Release Preparation & Verification Report

**Phase Goal:** The crate is version-bumped, the breaking API changes are documented, every gate is green, and the 1.0 checklist reflects the cleared API items — release-ready for the operator to tag and publish.
**Verified:** 2026-09-08T21:40:00Z
**Status:** passed

Verification performed by the orchestrator directly (docs/version-only phase; every gate was run inline). Code review is N/A — no source files changed (Cargo.toml version + CHANGELOG + ROADMAP-TO-1.0 only).

## Success Criteria

### SC-1 — version bump + breaking CHANGELOG (root + crate) — PASSED
- `fdars-core/Cargo.toml`: `version = "0.42.0"` [verified: grep].
- `CHANGELOG.md` and `fdars-core/CHANGELOG.md` both carry a `## [0.42.0] - 2026-09-08` breaking-framed entry covering: sealed `wire` (SEAL-01), `#[non_exhaustive]` on 38 configs + the construction-idiom change (SEAL-02/03), `Dim`-consolidations + `LpeerResult`→`LocalPeerResult` (NAME-01/02/03/05), and the 41 consolidations + 5 plain-renames (NAME-04), with a Migration section [verified: grep for `[0.42.0]`, `LocalPeerResult`, `construction idiom`].

### SC-2 — whole-crate gates green incl. cargo package — PASSED
Run at the release commit:
- `cargo fmt --check` → clean.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → clean.
- `cargo test --features linalg,parallel` → 2857 lib + all integration (equivalence goldens 7/5/8, validate_against_r 172, validate_spm_math 34, …) + 208 doctests; **0 failed**.
- `cargo build --features serde` → Finished.
- `cargo build --examples` → Finished (all 28 examples).
- `cargo package -p fdars-core --features linalg,parallel` → **Packaged 395 files (6.4 MiB / 1.3 MiB compressed) + verified build clean**.

### SC-3 — ROADMAP-TO-1.0 API section cleared — PASSED
- `documentation/ROADMAP-TO-1.0.md`: AUD-09/AUD-13, AUD-12, AUD-19, AUD-20, AUD-21, AUD-22, AUD-23 are all `- [x]` (7 bullet lines) with an "all API-section items cleared in v0.42.0" note [verified: grep count = 7]. Non-API items (golden flake, SDTW-O1, diff-core, fdars-r, 1.0-CUT) remain `- [ ]`.

### SC-4 — tag/publish deferred — PASSED
- No `v0.42.0` git tag exists [verified: `git tag --list v0.42.0` empty]. The tag→crates.io publish is documented in the SUMMARY as the operator's step.

## Verdict

All 4 success criteria met. v0.42.0 is release-ready; the milestone's entire API section of the 1.0 checklist is cleared. **PASSED.**
