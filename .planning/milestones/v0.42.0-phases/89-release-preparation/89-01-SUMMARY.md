---
phase: 89-release-preparation
plan: "01"
subsystem: infra
tags: [release, changelog, versioning, cargo-package, 1.0-checklist]

requires:
  - phase: 86-surface-sealing
    provides: "SEAL-01/02/03 breaking changes to document"
  - phase: 87-targeted-renames
    provides: "NAME-01/02/03/05 changes to document"
  - phase: 88-large-suffix-batch
    provides: "NAME-04 changes to document"
provides:
  - "fdars-core version 0.42.0"
  - "breaking-framed [0.42.0] CHANGELOG (root + crate)"
  - "documentation/ROADMAP-TO-1.0.md API section cleared (AUD-09/12/13/19-23)"
  - "verified release-readiness (full gate set + cargo package green); tag/publish left to operator"
affects: []

actuals:
  tokens: 18000
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Release-readiness proven with cargo package (verified build of the packaged tarball), not just cargo build"

key-files:
  created: []
  modified:
    - "fdars-core/Cargo.toml (0.41.0 → 0.42.0)"
    - "CHANGELOG.md + fdars-core/CHANGELOG.md ([0.42.0] breaking entry)"
    - "documentation/ROADMAP-TO-1.0.md (AUD-09/12/13/19-23 checked off)"

key-decisions:
  - "CHANGELOG explicitly documents the SEAL-02/03 construction-idiom change (external code uses Config::default()+field-assign, NOT struct literals / ..Default::default()) — the subtle breaking change beyond 'sealed configs'."
  - "No git tag / publish — the release.yml publishes to crates.io on any v* tag push, so tagging is deliberately left as the deferred operator step (per REL-01 SC4)."
  - "Cargo.lock is gitignored for this library crate — no lock commit needed; the local build synced it to 0.42.0 for cargo package."

patterns-established: []

requirements-completed: [REL-01]

coverage:
  - id: D1
    description: "version 0.42.0 + breaking-framed [0.42.0] CHANGELOG (root + crate) covering SEAL + all NAME items + construction-idiom change"
    requirement: "REL-01"
    verification:
      - kind: integration
        ref: "grep version 0.42.0 in Cargo.toml + [0.42.0] in both CHANGELOGs + LocalPeerResult callout"
        status: pass
    human_judgment: false
  - id: D2
    description: "whole-crate gates green incl. cargo package"
    requirement: "REL-01"
    verification:
      - kind: integration
        ref: "cargo fmt --check; clippy --all-targets -D warnings; cargo test (2857+integration+208 doctests, 0 failed); cargo build --features serde; cargo build --examples; cargo package -p fdars-core (verified build)"
        status: pass
    human_judgment: false
  - id: D3
    description: "ROADMAP-TO-1.0 API section (AUD-09/12/13/19-23) checked off; tag/publish deferred"
    requirement: "REL-01"
    verification:
      - kind: integration
        ref: "7 AUD bullets are [x]; no v0.42.0 git tag exists"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-09-08
status: complete
---

# Phase 89: Release Preparation & Verification Summary

**fdars-core v0.42.0 is release-ready: version bumped, all breaking API-shape changes from phases 86–88 documented in a breaking-framed CHANGELOG (root + crate), the 1.0-checklist API section cleared, and every gate — including `cargo package` — green. The `git tag`/publish is left as the deferred operator step.**

## Performance
- **Duration:** ~30 min
- **Tasks:** 3 (version+CHANGELOG, ROADMAP-TO-1.0 checkoff, gate suite + cargo package)
- **Files modified:** 4 code/doc files (+ Cargo.lock synced, gitignored)

## Accomplishments
- **Version:** `fdars-core/Cargo.toml` 0.41.0 → 0.42.0 (Cargo.lock synced).
- **CHANGELOG:** breaking-framed `[0.42.0]` entry in BOTH `CHANGELOG.md` and `fdars-core/CHANGELOG.md` — sealed `wire` (SEAL-01), `#[non_exhaustive]` on 38 configs + the construction-idiom change (SEAL-02/03), the Phase-87 `Dim`-consolidations + `LpeerResult`→`LocalPeerResult` (NAME-01/02/03/05), and the Phase-88 41 consolidations + 5 plain-renames (NAME-04), with a Migration section. API-shape-only, no numeric change stated explicitly.
- **1.0 checklist:** `documentation/ROADMAP-TO-1.0.md` — AUD-09/12/13/19-23 checked off with an "all API-section items cleared in v0.42.0" note. Non-API items (golden flake, SDTW-O1, diff-core, fdars-r, 1.0-CUT) left open.
- **Gates (all green at the release commit):** `cargo fmt --check`; `cargo clippy --all-targets --features linalg,parallel -- -D warnings`; `cargo test --features linalg,parallel` (2857 lib + all integration + 208 doctests, 0 failed); `cargo build --features serde`; `cargo build --examples`; `cargo package -p fdars-core` (Packaged 395 files + verified build clean).
- **No tag/publish** — confirmed no `v0.42.0` git tag exists.

## Task Commits
1. **Tasks A+B (version + CHANGELOG + ROADMAP-TO-1.0)** — `docs(89): bump 0.42.0, breaking CHANGELOG, clear ROADMAP-TO-1.0 API items`
2. **Cargo.lock sync** — attempted; lock is gitignored (library crate) so no commit landed — synced locally for cargo package.
3. **Task C** — verification-only (no source edits).

## Decisions Made
- Documented the construction-idiom change explicitly (the real breaking impact of `#[non_exhaustive]` on configs).
- Did NOT tag/publish — deferred operator step; release.yml auto-publishes on any `v*` tag.

## Deviations from Plan
None material. Cargo.lock turned out to be gitignored (expected for a library), so the "sync lock" step needed no commit.

## Issues Encountered
- `/home` disk pressure (~3.5G free) — freed `target/debug/{incremental,examples}` + `target/package` and routed examples/serde builds to `/tmp` tmpfs; `cargo package` verify build fit on the main tree.

## User Setup Required
**Operator step to publish (deliberately deferred):** `git tag v0.42.0 && git push origin v0.42.0` triggers `release.yml` → `cargo publish` to crates.io. This milestone prepared and verified release-readiness only.

## Next Phase Readiness
- Milestone v0.42.0 is complete: the entire **API section** of the 1.0 checklist is cleared. Remaining 1.0-blockers (golden-test flake, SDTW-O1, differentiable core, fdars-r migration) stay on `documentation/ROADMAP-TO-1.0.md` for future milestones before the 1.0-CUT.

---
*Phase: 89-release-preparation*
*Completed: 2026-09-08*
