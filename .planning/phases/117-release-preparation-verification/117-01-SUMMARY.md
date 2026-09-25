---
phase: 117-release-preparation-verification
plan: 01
subsystem: infra
tags: [release, changelog, cargo-package, semver, pmat, gitignore]

# Dependency graph
requires:
  - phase: 116-cross-language-re-run-published-report
    provides: "documentation/PERFORMANCE.md (pinned 1.19x face_covariance figure, six-op head-to-head at N=500,M=200) — the source of truth this plan's CHANGELOG entries cite"
provides:
  - "fdars-core at version 0.47.0 (manifest + refreshed, uncommitted Cargo.lock)"
  - "Root and crate CHANGELOG.md each carry an honest [0.47.0] entry (pinned 1.19x only, no unpinned 3.45x)"
  - "README.md + documentation/ guides reference 0.47 instead of stale 0.41/0.43"
  - ".gitignore excludes fdars-core/.pmat/ and fdars-core/.pmat-metrics/ so pmat local caches never enter the crate tarball"
affects: [117-02-pmat-decision, 117-03-cargo-gates, 117-04-package-pmat-harness-gates]

actuals:
  tokens: 2800
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns: ["Keep-a-Changelog entry pairs (root + crate) with shared headline/Changed, crate entry trims external-harness prose to one line", "cargo tree --depth 0 --offline to refresh a gitignored Cargo.lock without re-resolving third-party deps"]

key-files:
  created: []
  modified:
    - fdars-core/Cargo.toml
    - CHANGELOG.md
    - fdars-core/CHANGELOG.md
    - README.md
    - documentation/GETTING-STARTED.md
    - documentation/ARCHITECTURE.md
    - documentation/DEVELOPMENT.md
    - .gitignore

key-decisions:
  - "CHANGELOG cites only the pinned, noise-gate-clean 1.19x face_covariance figure; the unpinned multi-threaded Phase 115 3.45x figure is never mentioned (D-02)."
  - "Crate-level fdars-core/CHANGELOG.md links documentation/PERFORMANCE.md via the absolute GitHub URL (https://github.com/sipemu/fdars/blob/main/documentation/PERFORMANCE.md) since relative `../` links do not resolve once the file ships inside the packaged crate tarball."
  - "Cargo.lock refreshed via `cargo tree -p fdars-core --depth 0 --offline` (a resolve, not `cargo update`) so only the path-package version entry changes; file stays gitignored, never committed."

requirements-completed: [REL-01]

coverage:
  - id: D1
    description: "fdars-core bumped 0.46.0 -> 0.47.0 with no other manifest line touched"
    requirement: "REL-01"
    verification:
      - kind: other
        ref: "git diff -U0 0c7dc99a56849ba36502427e4168748583b045e4 -- fdars-core/Cargo.toml (exactly one -/+ pair on the version line)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Root and crate CHANGELOG.md each carry an honest [0.47.0] entry matching documentation/PERFORMANCE.md (1.19x pinned figure, scikit-fda 0.10.1/R head-to-head at N=500 M=200, no unpinned 3.45x)"
    requirement: "REL-01"
    verification:
      - kind: other
        ref: "plan 117-01 Task 1 <automated> METADATA_OK grep gate (Changed/Added headings, sym_cov_irreg, 1.19x, scikit-fda 0.10.1, N=500, PERFORMANCE.md link, benchmarks/ one-liner, absence of 3.45)"
        status: pass
    human_judgment: false
  - id: D3
    description: "cargo package emits fdars-core-0.47.0.crate with a normalized 0.47.0 manifest; package list ships no benchmarks/, .pmat/, .pmat-metrics/, .planning/, target/ or .claude/ entries"
    requirement: "REL-01"
    verification:
      - kind: other
        ref: "plan 117-01 Task 1 PACKAGE_META_OK + Task 2 PACKAGE_LIST_OK automated checks"
        status: pass
    human_judgment: false
  - id: D4
    description: "README.md and documentation/ guides reference 0.47 in install snippets and current-version mentions instead of stale 0.41/0.43"
    requirement: "REL-01"
    verification:
      - kind: other
        ref: "plan 117-01 Task 2 DOCS_OK automated grep gate"
        status: pass
    human_judgment: false
  - id: D5
    description: "No fdars-core/src file modified, no git tag created, nothing published"
    requirement: "REL-01"
    verification:
      - kind: other
        ref: "plan 117-01 Task 2 SCOPE_OK automated check (cargo fmt --check clean, no src diff, no v0.47* tag)"
        status: pass
    human_judgment: false

duration: 4min
completed: 2026-09-25
status: complete
---

# Phase 117 Plan 01: Release Metadata (0.47.0 Bump + CHANGELOG + Docs + pmat Gitignore) Summary

**fdars-core bumped 0.46.0 -> 0.47.0 with honest paired CHANGELOG [0.47.0] entries (pinned 1.19x face_covariance figure only), README/docs install snippets refreshed to 0.47, and pmat's local caches gitignored out of the crate package.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-09-25T20:47:54Z
- **Completed:** 2026-09-25T20:51:16Z
- **Tasks:** 2/2
- **Files modified:** 8

## Accomplishments
- `fdars-core/Cargo.toml` `[package] version` bumped 0.46.0 -> 0.47.0, the only changed line in the manifest; `Cargo.lock` refreshed via `cargo tree --depth 0 --offline` (gitignored, not committed).
- Root `CHANGELOG.md` and `fdars-core/CHANGELOG.md` each gained a `## [0.47.0] - 2026-09-25` entry in house Keep-a-Changelog style: shared bold headline + `### Changed (internal only)` (face_covariance `sym_cov_irreg` upper-triangle optimization, 1.19x pinned; elastic alignment/SVD near-optimal, no speedup claimed) + `### Added (tooling & docs)` (crosslang_grid bench, PERFORMANCE.md report, external `benchmarks/` harness — reduced to one line in the crate file per D-03).
- Confirmed `cargo package -p fdars-core --no-verify --allow-dirty` emits `target/package/fdars-core-0.47.0.crate` with a normalized 0.47.0 manifest, and `Cargo.lock` records fdars-core at 0.47.0.
- README.md, `documentation/GETTING-STARTED.md` (5 snippets), `documentation/ARCHITECTURE.md`, and `documentation/DEVELOPMENT.md` bumped from stale 0.41/0.43 version mentions to 0.47; historical references (STABILITY.md, ROADMAP-TO-1.0.md, older CHANGELOG entries) left untouched.
- `.gitignore` gained `fdars-core/.pmat/` and `fdars-core/.pmat-metrics/` so pmat's local caches (previously untracked-but-not-ignored, 408-entry package list included `.pmat/context.db` etc.) can never leak into the published crate tarball; root `.pmat/dead-code-cache.json` stays tracked.
- Verified `cargo package -p fdars-core --list --allow-dirty` is now clean of `benchmarks/`, `.pmat/`, `.pmat-metrics/`, `.planning/`, `target/`, `.claude/` entries while still including `benches/crosslang_grid.rs` and `CHANGELOG.md`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Release-metadata tracer — 0.47.0 bump -> both CHANGELOG [0.47.0] entries -> packaged-manifest sanity** - `fee423f6` (chore)
2. **Task 2: Bump stale version mentions in README + guides and keep pmat caches out of the crate package** - `59cdaa1b` (docs)

**Plan metadata:** (this commit, docs: complete plan)

_Note: no TDD tasks in this plan; single commit per task, no test/feat/refactor split._

## Files Created/Modified
- `fdars-core/Cargo.toml` - version 0.46.0 -> 0.47.0, only line changed
- `CHANGELOG.md` - new `## [0.47.0] - 2026-09-25` root entry
- `fdars-core/CHANGELOG.md` - new `## [0.47.0] - 2026-09-25` crate entry (ships inside the crate package; absolute GitHub link to PERFORMANCE.md)
- `README.md` - install snippet versions 0.41/0.43 -> 0.47
- `documentation/GETTING-STARTED.md` - five install snippet versions 0.41 -> 0.47
- `documentation/ARCHITECTURE.md` - current-version mention v0.41.0 -> v0.47.0
- `documentation/DEVELOPMENT.md` - current-version mention v0.41.0 -> v0.47.0
- `.gitignore` - added `fdars-core/.pmat/` and `fdars-core/.pmat-metrics/`

## Decisions Made
- Cargo.lock refresh chosen as `cargo tree -p fdars-core --depth 0 --offline` rather than `cargo update`/`cargo generate-lockfile`, so only the in-workspace path-package version entry changes and no third-party dependency gets re-resolved (D-10 discretion, exercised conservatively).
- Crate CHANGELOG's `documentation/PERFORMANCE.md` link uses the absolute GitHub URL rather than a relative `../` path, since the crate file ships standalone inside the packaged tarball where relative repo-root links do not resolve.
- Elastic/SVD prose in both CHANGELOG entries mirrors the report's honest framing exactly ("already near-optimal, no speedup claimed") rather than inventing a marginal number for the within-noise Phase 114 result.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

One transient verification failure, fixed inline before commit: the head-to-head sentence in both CHANGELOG entries initially wrapped `scikit-fda` and `0.10.1` across a markdown line break, which the plan's exact-string grep gate (`scikit-fda 0\.10\.1`) correctly rejected since grep matches within a single line only. Rewrapped both paragraphs so `scikit-fda 0.10.1` stays on one line; re-ran the gate, which then passed (METADATA_OK). No rule invoked — this was a wording fix caught by the plan's own verification loop before the task was considered done, not a deviation from scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Plan 02 (pmat decision — user already chose `refactor`) can proceed against this committed 0.47.0 metadata.
- Plans 03/04 (cargo gate battery, package/pmat/harness gates) run against the now-committed version bump and CHANGELOG entries; no additional metadata prerequisites remain.
- No blockers.

---
*Phase: 117-release-preparation-verification*
*Completed: 2026-09-25*

## Self-Check: PASSED

All 8 modified files found on disk; both task commits (`fee423f6`, `59cdaa1b`) found in `git log --oneline --all`.
