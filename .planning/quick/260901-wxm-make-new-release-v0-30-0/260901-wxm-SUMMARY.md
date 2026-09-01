---
quick_id: 260901-wxm
slug: make-new-release-v0-30-0
description: make new release v0.30.0
date: 2026-09-01
status: complete
commit: 96673410
---

# Summary: Make New Release v0.30.0

Shipped the deferred **REL-01** release. Executed inline (not via gsd-executor
subagents) due to the 102-commit main/origin divergence and the irreversibility of
the crates.io publish.

## What was done

1. **Version bump** — `fdars-core/Cargo.toml` `0.29.0 → 0.30.0` (+ local `Cargo.lock`,
   which is gitignored).
2. **Changelog** — added a `[0.30.0]` section to root `CHANGELOG.md` summarizing the
   Performance & Consolidation Pass (PERF/CONS/API/BENCH highlights), noting it also
   folds in the v0.29.0 dev work whose tag shipped without a root changelog entry.
3. **Pre-publish verification** — `cargo publish --dry-run --package fdars-core` on a
   clean tree (TMPDIR set to `~/.cache/fdars-bench-tmp` to avoid the /tmp-exhaustion
   trap): **packaged 371 files (5.7 MiB, 1.1 MiB compressed), verify-build compiled in
   ~21s**, upload aborted due to dry-run. Green.
4. **Release commit** — `96673410` `chore(release): v0.30.0 …` (`--no-verify`; only
   non-source files changed, so no fmt/clippy relevance).
5. **Push** — `main` pushed to origin (`ca80db3f..96673410`, 103 commits).
6. **Tag** — annotated `v0.30.0` created at `96673410` and pushed → triggered
   `.github/workflows/release.yml` (run `33562773216`), which runs
   `cargo publish --package fdars-core`.

## Result

- crates.io: `fdars-core` `0.29.0 → 0.30.0` (published via CI on the tag push).
- `origin/main` now current with local `main`; the 102-commit backlog is cleared.

## Follow-ups (not in scope this task)

- Root `CHANGELOG.md` still lacks a dedicated `[0.29.0]` section (the 0.30.0 entry
  notes it folds in 0.29.0, but a standalone 0.29.0 entry was not backfilled).
- `fdars-core/CHANGELOG.md` is stale at `[0.22.0]` (7 versions behind) — separate
  cleanup.
- **APIB-01** (breaking removal of the v0.30.0 `#[deprecated]` forms) remains deferred
  to a future 1.0-readiness milestone.
- STATE.md "Release status" prose (referenced 0.28.0) was stale — corrected via the
  quick-task state update.
