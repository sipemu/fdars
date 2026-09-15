---
plan: "110-01"
phase: "110"
title: "pmat quality-gate CI guardrail"
status: complete
completed: "2026-09-14"
requirements: [CI-01]
key_files:
  created: []
  modified:
    - .github/workflows/rust-ci.yml
---

# Plan 110-01 Summary — pmat CI Guardrail

## What was done

Added a `pmat-quality-gate` job to `.github/workflows/rust-ci.yml`, mirroring the
existing `determinism-guardrail` job structure:

- `runs-on: ubuntu-latest`, `working-directory: fdars-core`.
- checkout → `dtolnay/rust-toolchain@stable` → `actions/cache@v4` (cargo dirs
  incl. `~/.cargo/bin/` so the pmat binary is cached, keyed on pmat version +
  Cargo.lock) → install pmat → run gate.
- **Install:** `cargo install pmat --version 3.18.0 --locked` (guarded by a
  version check so a cache hit skips reinstall). pmat is a dev/CI tool — it is
  NOT added to `Cargo.toml` (standing project convention). Version pinned so the
  gate's thresholds match what the milestone validated.
- **Gate command (CURATED):**
  `pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation`.
  The stochastic `entropy` analyzer is intentionally EXCLUDED (per the operator
  decision — it is non-deterministic and would make CI flaky). The deterministic
  checks (complexity, satd, dead-code, duplicates) are enforced; `--fail-on-violation`
  makes the job exit non-zero on any violation.
- **Release gating:** added `pmat-quality-gate` to `publish.needs` (alongside
  `determinism-guardrail`) so a quality regression blocks a crates.io release.
- **Local reproduction** documented in an inline comment:
  `pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation`
  (from `fdars-core/`).

## Verification

- **YAML valid** (parsed with a YAML loader); `pmat-quality-gate` present in
  `jobs` and in `publish.needs`.
- **The exact CI gate command passes locally:** run from `fdars-core/`, exit code
  0 (Quality Gate: PASSED, 0 violations) — the CI job will pass on the current
  (Phase-109-clean) tree and will fail if any violation is reintroduced.
- No source/`Cargo.toml` change; pmat stays external.

## Requirement disposition

- **CI-01:** SATISFIED. A pmat quality-gate CI job enforces the curated gate on
  every push/PR to main and gates release; documented for local reproduction;
  mirrors the v0.43.0 determinism-guardrail pattern; no new crate dependency.
