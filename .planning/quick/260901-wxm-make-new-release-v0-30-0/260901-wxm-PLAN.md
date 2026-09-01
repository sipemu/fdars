---
quick_id: 260901-wxm
slug: make-new-release-v0-30-0
description: make new release v0.30.0
created: 2026-09-01
---

# Quick Task 260901-wxm: Make New Release v0.30.0

Ship the deferred REL-01 release. crates.io is at `0.29.0` (tag `v0.29.0` pushed);
the v0.30.0 milestone (Phases 46–51, Performance & Consolidation Pass) shipped code
but never bumped the version or tagged. Local `main` is 102 commits ahead of origin.

Release mechanism: pushing a `v*` tag triggers `.github/workflows/release.yml`, which
runs `cargo publish --package fdars-core` (auth via `CARGO_REGISTRY_TOKEN`).

## Decisions (confirmed with operator)

- **Version:** 0.30.0 (minor bump, matches the milestone; additive/behavior-preserving).
- **Scope:** Full publish — bump + changelog + commit, push `main`, tag `v0.30.0`, push
  tag → CI auto-publishes to crates.io.
- **Verify:** run `cargo publish --dry-run` first (TMPDIR set to avoid /tmp exhaustion).
- **Changelog:** add a `[0.30.0]` section to root `CHANGELOG.md` only (pre-existing
  0.29.0 gap in root + stale `fdars-core/CHANGELOG.md` at 0.22.0 left as follow-up).

## Tasks

1. Bump `fdars-core/Cargo.toml` `0.29.0 → 0.30.0` + matching `Cargo.lock` entry.
2. Add `[0.30.0]` section to root `CHANGELOG.md`.
3. `cargo publish --dry-run --package fdars-core` — must succeed before tagging.
4. Commit the release (Cargo.toml, Cargo.lock, CHANGELOG.md).
5. Push `main` to origin (102 commits).
6. Tag `v0.30.0`, push tag → CI publishes to crates.io.

## Execution note

Executed **inline** (not via gsd-executor subagents): local main diverges 102 commits
from origin (worktree base divergence → sequential per project memory), and the
irreversible crates.io publish needs tight operator control. Executors also stall on
long fdars cargo builds / drop at wrap-up (project memory).
