---
phase: 80-release-hardening-ship-v0-40-0
verified: 2026-09-07T00:00:00Z
status: passed
score: 6/6 must-haves verified
behavior_unverified: 0
verified_by: orchestrator-inline (release-prep phase; all gates + artifacts re-checked directly)
---

# Phase 80: Release Hardening & Ship v0.40.0 Verification Report

**Phase Goal:** The crate is validated and release-ready — 75/76/77 signed off, version/CHANGELOG/docs at v0.40.0, all whole-crate gates green — so the operator's `git tag v0.40.0` → crates.io publish is the only remaining step. The phase prepares + verifies; it does not tag/publish.
**Verified:** 2026-09-07
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Phases 75/76/77 VALIDATION.md are `status: validated` + `nyquist_compliant: true` (or recorded gap) | ✓ VERIFIED | All three flipped to validated/true; no gaps; existing tests confirmed passing (autodiff 30, differentiable/soft_dtw/fpca, grad+doctest) |
| 2 | `fdars-core/Cargo.toml` version is exactly `0.40.0` | ✓ VERIFIED | `grep '^version' → 0.40.0`; no stray 0.38 |
| 3 | CHANGELOG carries both `[0.39.0]` (AD core) and `[0.40.0]` (fixes) entries | ✓ VERIFIED | Both headings present above `[0.38.0]`; [0.40.0] notes the soft_dtw barycenter behavior change |
| 4 | Tracked version strings bumped 0.38 → 0.40 across README + documentation/ | ✓ VERIFIED | `git grep '0\.38'` in README/documentation → no version-related hits; README/GETTING-STARTED/ARCHITECTURE/DEVELOPMENT updated + AD highlight added |
| 5 | Whole-crate gates green under the serde feature set | ✓ VERIFIED | fmt --check clean; clippy --all-targets serde (CI allows) clean; full serde suite 2862 lib + 107 + 174 + 209 doc, 0 failed |
| 6 | SUMMARY documents operator ship steps; phase did NOT tag/publish | ✓ VERIFIED | 80-02-SUMMARY has "Operator Ship Steps" (tag → push → release.yml); `git tag -l v0.40.0` empty (no tag created) |

**Score:** 6/6 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.../v0.39.0-phases/{75,76,77}-*/*-VALIDATION.md` | draft → validated | ✓ | all three validated + nyquist_compliant:true |
| `fdars-core/Cargo.toml` | version 0.40.0 | ✓ | bumped from 0.38.0 |
| `CHANGELOG.md` | [0.39.0] + [0.40.0] entries | ✓ | keepachangelog format, soft_dtw behavior-change noted |
| `README.md` + `documentation/*.md` | version refresh + AD highlight | ✓ | all 0.38→0.40, Autodiff feature row added |
| `80-01-SUMMARY.md`, `80-02-SUMMARY.md` | phase summaries + operator steps | ✓ | 80-02 carries the operator ship-steps section |

**Artifacts:** 5/5 verified

## Gates (re-run by orchestrator on final tree)

- `cargo fmt --check` — clean
- `cargo clippy --all-targets --features linalg,parallel,serde -- -D warnings <CI allows>` — clean
- `cargo test -p fdars-core --features linalg,parallel,serde` — 2862 lib + 107 + 174 + 209 doc, 0 failed
- version/CHANGELOG/docs edits confirmed by grep; no v0.40.0 git tag exists

## Notes

- REL-01 + REL-02 both fully covered (Plans 01, 02).
- Ship boundary respected: NO `git tag` / `cargo publish` performed. The operator's manual step (`git tag v0.40.0` → `git push origin v0.40.0` → `release.yml` runs `cargo publish`) is documented in 80-02-SUMMARY.
- This milestone folds in the never-published v0.39.0 forward-mode AD core plus the v0.40.0 correctness/build fixes (CORR-01 soft_dtw + barycenter stabilization, CORR-02 audit, BUILD-01 serde).
