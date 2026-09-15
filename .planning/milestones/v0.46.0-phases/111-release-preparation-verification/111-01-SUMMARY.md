---
plan: "111-01"
phase: "111"
title: "Release preparation & verification — fdars-core 0.46.0"
status: complete
completed: "2026-09-14"
requirements: [REL-01]
key_files:
  created: []
  modified:
    - fdars-core/Cargo.toml
    - CHANGELOG.md
    - fdars-core/CHANGELOG.md
---

# Plan 111-01 Summary — Release Preparation & Verification

## What was done

- **Version bump:** `fdars-core/Cargo.toml` `0.45.0 → 0.46.0`; `Cargo.lock` refreshed
  to 0.46.0 (Cargo.lock is gitignored in this project, so not committed).
- **CHANGELOGs:** `## [0.46.0] - 2026-09-14` entries added to BOTH the root
  `CHANGELOG.md` and `fdars-core/CHANGELOG.md`, in the existing Keep-a-Changelog
  style — describing the quality-hardening milestone (complexity decomposition to
  the pmat gate bar, entropy reduction, the `.unwrap()`/SATD fixes, and the pmat CI
  guardrail), emphasizing strictly non-breaking + numeric-identical, no new crate
  dependency.

## Release gate battery — all green on 0.46.0

| Gate | Result |
|------|--------|
| `cargo fmt -p fdars-core --check` | clean |
| `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | clean |
| `cargo test -p fdars-core --features linalg,parallel` (full suite) | 2942 lib + 215 doctests pass |
| `cargo build -p fdars-core --features serde` (serde guard) | builds (historical break repaired long ago) |
| `cargo build -p fdars-core --examples --features linalg,parallel` | all 28 examples build |
| `cargo package -p fdars-core` | 405 files, verify-build clean |
| `pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation` | **PASSED — 0 violations** |

## Deferred (operator step — NOT done in-phase)

- `git tag v0.46.0` → crates.io publish. GSD `git.create_tag` is off because
  `release.yml` couples a `v*` tag push to `cargo publish`; tagging/publishing is
  a deliberate manual operator action. Confirmed no `v0.46.0` tag was created.

## Requirement disposition

- **REL-01:** SATISFIED. Crate at 0.46.0, both CHANGELOGs updated, entire gate
  battery green including the curated `pmat quality-gate`. Release-ready; publish
  deferred.
