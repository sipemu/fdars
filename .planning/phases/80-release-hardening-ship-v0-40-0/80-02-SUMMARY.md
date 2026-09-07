---
phase: 80-release-hardening-ship-v0-40-0
plan: "02"
status: complete
provides: [REL-02]
key-files:
  - fdars-core/Cargo.toml
  - CHANGELOG.md
  - README.md
  - documentation/ARCHITECTURE.md
  - documentation/DEVELOPMENT.md
  - documentation/GETTING-STARTED.md
completed: "2026-09-07"
---

# Plan 80-02 — REL-02 Version + CHANGELOG + Docs + Gates Summary

## Accomplishments

Release-readiness verification and documentation for fdars-core v0.40.0. The crate version
was bumped from 0.38.0 to 0.40.0 (skipping the never-published 0.39.0), CHANGELOG entries
were added for both [0.39.0] and [0.40.0], all tracked version strings in README and
documentation/ were refreshed, a forward-mode AD feature highlight was added to the README,
and all whole-crate gates pass green. The operator's final manual ship steps are documented
below. This phase did NOT run `git tag` or `cargo publish`.

## Task Commits

| Commit | Hash | Description |
|--------|------|-------------|
| chore(80-02): bump version + CHANGELOG + docs | d9b4dd25 | Cargo.toml 0.38.0→0.40.0; CHANGELOG [0.39.0]+[0.40.0]; README ×2 + AD row; docs version strings |

## Files Modified

- `fdars-core/Cargo.toml` — version bumped from 0.38.0 to 0.40.0
- `CHANGELOG.md` — added [0.40.0] and [0.39.0] entries above [0.38.0]; amended preamble
- `README.md` — version pins 0.38→0.40 (×2); added Autodiff row to Features table
- `documentation/GETTING-STARTED.md` — all 0.38 version strings → 0.40 (×6)
- `documentation/ARCHITECTURE.md` — v0.38.0 → v0.40.0
- `documentation/DEVELOPMENT.md` — v0.38.0 → v0.40.0
- `.planning/phases/80-release-hardening-ship-v0-40-0/80-02-SUMMARY.md` — this file

## Verification

### Gate Results

All three whole-crate gates run against `--features linalg,parallel,serde`:

```
cargo fmt --manifest-path fdars-core/Cargo.toml -- --check
→ EXIT 0 (clean, no drift)

cargo clippy --all-targets --features linalg,parallel,serde -- \
    -D warnings \
    -A clippy::too_many_arguments \
    -A clippy::useless_vec \
    -A clippy::type_complexity \
    -A clippy::manual_memcpy \
    -A clippy::wildcard_in_or_patterns
→ EXIT 0 (fdars-core v0.40.0 — Finished dev profile, no warnings)

cargo test -p fdars-core --features linalg,parallel,serde
→ EXIT 0
   lib tests:         2862 passed, 0 failed, 0 ignored
   integration tests: 174 + 56 + 16 + 34 + 209 + (more) passed, 0 failed
   doctests:          209 passed (4 ignored), 0 failed
```

### Version Verification

```
grep -q '^version = "0.40.0"' fdars-core/Cargo.toml  → match
git grep -n '0\.38' -- README.md 'documentation/*.md'  → (empty, no stray 0.38 remaining)
grep -qiE 'differentiable|forward-mode|autodiff|Dual' README.md  → match (Autodiff row added)
```

### CHANGELOG Verification

```
grep -qE '^## \[0\.40\.0\]' CHANGELOG.md  → match
grep -qE '^## \[0\.39\.0\]' CHANGELOG.md  → match
grep -qE '^## \[0\.38\.0\]' CHANGELOG.md  → match
awk order check (0.40.0 < 0.39.0 < 0.38.0 line numbers)  → EXIT 0
grep -iqE 'soft.?dtw' CHANGELOG.md  → match (behavior change documented)
```

## Operator Ship Steps

> **LOCKED ship boundary**: This phase prepared and verified release-readiness only.
> The phase did **NOT** run `git tag`, did **NOT** push any tag, and did **NOT** run
> `cargo publish`. Those are manual operator steps documented here.

Once you have confirmed:
1. Phase 78 (CORR-01/CORR-02 soft-DTW fix) — green
2. Phase 79 (BUILD-01 serde fix) — green
3. Phase 80 (this phase: gates green, version 0.40.0, CHANGELOG complete) — green

Then run the following in order:

```bash
# 1. Create the annotated release tag
git tag v0.40.0

# 2. Push the tag to trigger the release workflow
git push origin v0.40.0
```

The `release.yml` GitHub Actions workflow fires on `v*` tag pushes and runs
`cargo publish` to upload fdars-core v0.40.0 to crates.io.

**Do NOT push a tag for audit-only milestones** (no crate version change).
This milestone has real code changes (Phases 78/79 fixes + 75/76/77 AD substrate),
so a real crates.io publish is appropriate.

## Backlog Notes from Plan 80-01

No coverage gaps were found in the REL-01 Nyquist sign-off (Plan 80-01). Nothing
to promote to backlog from the sign-off process.

Pre-existing backlog items (not introduced by this phase):
- SDTW-O1: Proper soft-DTW barycenter optimizer (L-BFGS) — logged in Phase 78
- fdars-r R-wrapper FdMatrix migration (issue fdars-j75) — deferred
