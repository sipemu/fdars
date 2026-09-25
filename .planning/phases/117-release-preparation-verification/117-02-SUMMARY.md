---
phase: 117-release-preparation-verification
plan: 02
pmat_decision: refactor
subsystem: alignment
tags: [pmat, refactor, numeric-identical, release]
requirements: [REL-01]
key-files:
  modified:
    - fdars-core/src/alignment/mod.rs
    - CHANGELOG.md
    - fdars-core/CHANGELOG.md
key-decisions:
  - "User chose `refactor` (2026-09-25): clear the dp_fill_grid pmat violation with a numeric-identical decomposition rather than ship a recorded deviation"
completed: 2026-09-25
---

# Plan 117-02 Summary: dp_fill_grid decomposition (pmat guardrail)

Executed INLINE by the orchestrator.

## Task 1 — decision
Blocking checkpoint resolved by the user: **refactor** (recorded as `pmat_decision: refactor` before any src edit).

## Task 2 — refactor
- Characterization baseline on untouched code: equivalence_phase114 `ok. 2 passed`, equivalence_phase47 `ok. 7 passed` (debug).
- Added private `#[inline] fn dp_band_cols(tr, ncols, band) -> Range<usize>` (None → `0..ncols`; Some(r) → `tr.saturating_sub(r)..(tr + r).min(ncols - 1) + 1`, the exclusive-end form of the old inclusive corridor; empty when lower > upper exactly as before). `dp_fill_grid` is now one loop nest over that range with the origin skip; `match band` removed. Signature, visibility, generic bound and caller unchanged.
- After: equivalence_phase114 `ok. 2 passed` + equivalence_phase47 `ok. 7 passed` (debug); equivalence_phase114 `ok. 2 passed` (release); `--lib alignment` `ok. 274 passed`; lib clippy `-D warnings` clean; `cargo fmt --check` clean. Only `fdars-core/src/alignment/mod.rs` changed; no `pub` or lint-allow lines added/removed.

## Task 3 — pmat + CHANGELOG
- `cd fdars-core && pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation` → exit 0, `Quality Gate: PASSED`, `Total violations: 0`; no dp_fill_grid line.
- Added the "dp_fill_grid decomposed (pmat guardrail)" bullet to `### Changed (internal only)` in both `[0.47.0]` entries (same wording; no perf claim).
- Committed `47f7bd44` (alignment/mod.rs + both CHANGELOGs). No tag, no publish, no CI/pmat config change.

Logs: `~/.cache/fdars-bench-tmp/117-gates/117-02-*.log`.

## Deviations
None.
