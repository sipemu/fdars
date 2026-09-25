---
phase: 117-release-preparation-verification
plan: 04
subsystem: release
tags: [package, pmat, harness, release, rel-01]
requirements: [REL-01]
key-files:
  modified:
    - .planning/phases/117-release-preparation-verification/117-GATE-LOG.md
completed: 2026-09-25
---

# Plan 117-04 Summary: packaging, pmat, harness, verdict

Executed INLINE by the orchestrator.

- **package-verify:** `cargo package -p fdars-core` without `--allow-dirty` → 401 files, 6.7 MiB (1.4 MiB compressed), verify build green; tarball manifest `version = "0.47.0"`, `.cargo_vcs_info.json` sha1 `976a1c84`, not dirty.
- **package-list:** 0 entries under benchmarks/ .pmat/ .pmat-metrics/ .planning/ target/ .claude/; `benches/crosslang_grid.rs`, `CHANGELOG.md`, `README.md` present.
- **package-bench-fallback:** extracted tarball in `~/.cache/fdars-pkg-verify` (no `../benchmarks`), hermetic `--offline --locked` build of the `crosslang_grid` bench, criterion `--test n100_m50` exit 0 with all six `Testing crosslang_<op>/n100_m50` lines (synthetic fallback). Scratch dirs removed.
- **pmat:** `Quality Gate: PASSED`, `Total violations: 0`.
- **harness:** pytest 209 passed; `render_report.py --check` + `--check-prose` exit 0.
- **changelog-claims:** both `[0.47.0]` sections cite 1.19×, neither cites the unpinned Phase 115 figure.
- **no-tag-no-publish:** no `v0.47*` tag; only `fdars-core/src/alignment/mod.rs` changed (refactor decision); nothing published.
- Deferred operator step documented in the gate log.

**REL-01: SATISFIED — all 16 gates PASS.**

Commit: `d1c60c71`.

## Deviations
None.
