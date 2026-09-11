---
phase: 100-release-preparation-verification
plan: "01"
subsystem: release
tags: [release, changelog, version-bump, roadmap-1.0, rel-01, gates]

requires:
  - phase: 94-reverse-mode-autodiff-core
  - phase: 95-generic-scalar-hot-path-signatures
  - phase: 96-differentiable-basis-evaluation-inner-products
  - phase: 97-differentiable-regression-prediction-smoothing-penalties
  - phase: 98-differentiable-depth-curve-distances
  - phase: 99-end-to-end-autodiff-flow-gradient-api

provides:
  - "fdars-core 0.44.0 release-ready: version bumped, [0.44.0] in both CHANGELOGs, DIF-F1/F2/F3 cleared on ROADMAP-TO-1.0.md, all whole-crate gates green"

affects: []

actuals: { tokens: 0, tasks: 2, commits: 1 }

tech-stack: { added: [], patterns: [] }

key-files:
  created: []
  modified:
    - fdars-core/Cargo.toml
    - CHANGELOG.md
    - fdars-core/CHANGELOG.md
    - documentation/ROADMAP-TO-1.0.md

key-decisions:
  - "git tag v0.44.0 → crates.io publish is the DEFERRED operator step (git.create_tag off; release.yml couples tag-push to publish). Not performed in-phase."
  - "Executed inline by the orchestrator (mechanical release prep; context-constrained)."

commits:
  - "cd524b1f: release(100): bump fdars-core 0.43.0 -> 0.44.0; CHANGELOG + 1.0-checklist (REL-01)"

gates:
  - "cargo fmt --check: clean"
  - "cargo clippy --all-targets --features linalg,parallel -- -D warnings: clean"
  - "full cargo test --features linalg,parallel: 0 failed (lib + integration) + 213 doctests"
  - "cargo build --features serde: clean"
  - "cargo build --examples --features linalg,parallel: 28 examples build"
  - "cargo package -p fdars-core: v0.44.0, 398 files, 1.3MiB compressed"
---

# Phase 100 — Release Preparation & Verification (REL-01)

fdars-core is release-ready at **0.44.0**. Version bumped in `fdars-core/Cargo.toml`; `[0.44.0]` entries added to the root and crate CHANGELOGs (Differentiable Core — DIF-F1/F2/F3 summary); `DIF-F1`/`DIF-F2`/`DIF-F3` checked off on `documentation/ROADMAP-TO-1.0.md`.

**All whole-crate gates green:** `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test` (0 failed + 213 doctests — the milestone's end-to-end proof), `--features serde` build, 28 examples, and `cargo package` (v0.44.0, 398 files). No new dependency.

**Deferred operator step:** `git tag v0.44.0` → crates.io publish is intentionally NOT performed in-phase (GSD `git.create_tag` is off; `release.yml` couples any `v*` tag push to `cargo publish`). The registry catch-up (0.40.0 → 0.44.0, superseding the unpublished 0.41.0–0.43.0) remains an operator action.
