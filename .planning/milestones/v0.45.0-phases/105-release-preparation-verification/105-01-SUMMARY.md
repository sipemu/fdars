---
phase: 105-release-preparation-verification
plan: 01
subsystem: release
tags: [release, changelog, version-bump, gate-battery, sdtw-o1]

# Dependency graph
requires:
  - phase: 104-quality-evidence-escapes-local-minimum-f64-parity
    provides: SDTW-04 quality-evidence tests; all SDTW-01/02/03/04 requirements satisfied
provides:
  - fdars-core version 0.45.0 in Cargo.toml
  - "[0.45.0] CHANGELOG entries in root and crate CHANGELOGs"
  - SDTW-O1 checked off on documentation/ROADMAP-TO-1.0.md
  - Full whole-crate gate battery green (fmt/clippy-all-targets/full-test/serde/doctests+28examples/package)
affects: [crates.io-publish-operator-step]

actuals:
  tokens: 3500          # ~14000 chars / 4 over the realized diff (CHANGELOG + metadata)
  tasks: 2
  commits: 1
  plan_head_before: d8c04bb9427f6c92e9306a86c472a5f68753ce66

tech-stack:
  added: []
  patterns:
    - "Version-bump-only release prep: Cargo.toml [package] version only, no dependency changes"
    - "Dual-CHANGELOG pattern: root entry carries fuller prose; crate entry stays terse per house style"
    - "--no-verify commits after out-of-band gate battery (project hazard: pre-commit hook is watchdog-killed)"

key-files:
  created: []
  modified:
    - fdars-core/Cargo.toml
    - CHANGELOG.md
    - fdars-core/CHANGELOG.md
    - documentation/ROADMAP-TO-1.0.md

key-decisions:
  - "Root CHANGELOG carries fuller prose [0.45.0] entry summarizing all four SDTW requirements (SDTW-01/02/03/04)"
  - "Crate CHANGELOG keeps entry terse per its established house style"
  - "SDTW-O1 clearing note appended: 'Done v0.45.0 — only fdars-j75 and 1.0-CUT remain'"
  - "Gate battery run out-of-band foreground with 600s timeout per project hazard notes; committed --no-verify"
  - "No git tag, no cargo publish — deferred operator step"

metrics:
  duration: "~7 min"
  completed_date: "2026-09-13"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 4

requirements: [REL-01]

status: complete
---

# Phase 105 Plan 01: Release Preparation & Verification Summary

**Version bump 0.44.0 → 0.45.0 with dual CHANGELOG entries (SDTW-01/02/03/04 summary), SDTW-O1 checkoff on ROADMAP-TO-1.0.md, and full whole-crate gate battery green — REL-01 milestone proof for v0.45.0**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-09-13T12:26:32Z
- **Completed:** 2026-09-13T12:33:01Z
- **Tasks:** 2 (Task 1: metadata edits; Task 2: gate battery — verification only)
- **Files modified:** 4

## Accomplishments

- Bumped `fdars-core` version 0.44.0 → 0.45.0 in `fdars-core/Cargo.toml` (`[package]` line only; no dependency versions changed)
- Prepended `## [0.45.0] - 2026-09-13` to root `CHANGELOG.md` with prose summary of SDTW-01/02/03/04 (in-crate L-BFGS, multi-restart, config API, quality-evidence tests; strictly additive/non-breaking, no new crate dependency)
- Prepended `## [0.45.0] - 2026-09-13` to `fdars-core/CHANGELOG.md` (terse crate style, same SDTW-01/02/03/04 bullet coverage)
- Flipped `SDTW-O1` from `- [ ]` to `- [x]` on `documentation/ROADMAP-TO-1.0.md` with clearing note: "Done v0.45.0 — only fdars-j75 and 1.0-CUT remain"
- Full whole-crate gate battery passed green (details below)

## Task Commits

1. **Task 1: Release-metadata slice** — `18dfedeb` (feat)
2. **Task 2: Gate battery** — verification only, no source changes, no commit

## Files Created/Modified

- `fdars-core/Cargo.toml` — version 0.44.0 → 0.45.0 (1 line)
- `CHANGELOG.md` — [0.45.0] entry prepended (root; fuller prose style, 39 lines added)
- `fdars-core/CHANGELOG.md` — [0.45.0] entry prepended (crate; terse style, 20 lines added)
- `documentation/ROADMAP-TO-1.0.md` — SDTW-O1 checked off + clearing note (6 lines changed)

## Gate Battery Results (Task 2)

| Gate | Command | Result |
|------|---------|--------|
| 1. fmt | `cargo fmt -p fdars-core --check` | PASS (exit 0, no diffs) |
| 2. clippy | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | PASS (exit 0, clean) |
| 3. full test | `cargo test -p fdars-core --features linalg,parallel` | PASS (2942 lib+integration + 0 failed; 0 ignored) |
| 4. serde build | `cargo build -p fdars-core --features serde` | PASS (exit 0) |
| 5a. doctests | `cargo test -p fdars-core --doc` | PASS (215 passed, 5 ignored, 0 failed) |
| 5b. 28 examples | `cargo build -p fdars-core --examples --features linalg,parallel` | PASS (exit 0, all 28 compile) |
| 6. package | `cargo package -p fdars-core --no-verify --allow-dirty` | PASS (398 files, 6.6MiB) |

**All gates green.** No pre-existing serde/ShapeletTransformClassifier issue surfaced (the build was clean).

## Decisions Made

- The root `CHANGELOG.md` [0.45.0] entry uses prose consistent with neighboring entries (multi-paragraph narrative under `### Added` bullets, no fenced code blocks)
- The crate `fdars-core/CHANGELOG.md` [0.45.0] entry stays terse (bullet list per its established pattern)
- SDTW-O1 clearing note appended inline on the same line in `ROADMAP-TO-1.0.md` per the pattern established for DIF-F1/F2/F3

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None. No new network endpoints, auth paths, file access, or schema changes. The release.yml publish boundary was explicitly NOT crossed (no `git tag v0.45.0`, no `cargo publish`).

## Self-Check

### Files

- [x] `fdars-core/Cargo.toml` contains `version = "0.45.0"` — VERIFIED
- [x] `CHANGELOG.md` contains `## [0.45.0]` — VERIFIED
- [x] `fdars-core/CHANGELOG.md` contains `## [0.45.0]` — VERIFIED
- [x] `documentation/ROADMAP-TO-1.0.md` contains `- [x] **\`SDTW-O1\`` — VERIFIED

### Commits

- [x] `18dfedeb` — feat(105-01): release metadata — FOUND

### Gate Assertions

- [x] `grep '^version = "0.45.0"' fdars-core/Cargo.toml` — matches; 0.44.0 line gone
- [x] `grep '## \[0.45.0\]' CHANGELOG.md fdars-core/CHANGELOG.md` — matches in both
- [x] `grep -E '^- \[x\] \*\*\`SDTW-O1\`' documentation/ROADMAP-TO-1.0.md` — matches
- [x] `cargo package -p fdars-core --no-verify --allow-dirty` — exit 0, v0.45.0 packaged
- [x] No `git tag` executed, no `cargo publish` executed

## Self-Check: PASSED

All metadata changes verified, gate battery green, commit present.

---
*Phase: 105-release-preparation-verification*
*Completed: 2026-09-13*
