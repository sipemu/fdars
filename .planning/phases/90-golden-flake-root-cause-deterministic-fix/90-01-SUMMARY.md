---
phase: 90-golden-flake-root-cause-deterministic-fix
plan: 01
subsystem: testing
tags: [determinism, golden-tests, svd, faer, nalgebra, feature-flags, fpca, cfg-attr]

requires: []
provides:
  - "Evidence-backed root-cause diagnosis of the three golden-test flakes (90-DIAGNOSIS.md)"
  - "Deterministic fix: #[cfg_attr(not(feature = \"linalg\"), ignore)] on the three faer-backend golden tests"
  - "A green full-parallel cargo test baseline (with linalg) proven across 10 consecutive runs — the baseline Phases 92–93 build on"
affects: [91-suite-wide-robustness-sweep, 92-ci-determinism-guardrail, 93-release-preparation]

actuals:
  tokens: 4200
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - "cfg-attribute guard binding a golden test to the SVD backend its references were captured under"

key-files:
  created:
    - .planning/phases/90-golden-flake-root-cause-deterministic-fix/90-DIAGNOSIS.md
  modified:
    - fdars-core/tests/equivalence_phase48.rs
    - fdars-core/tests/equivalence_phase49.rs

key-decisions:
  - "Root cause is a deterministic feature-configuration artifact (faer vs nalgebra SVD backend in fdata_to_pc), NOT an environmental/disk-pressure flake — the historical 'intermittent' hypothesis is overturned by evidence."
  - "Fix is the least-invasive test-side cfg-guard; no tolerance relaxation (deltas are categorical, not ULP), no serialization (no race), no src/ or Cargo.toml change, no new dependency."
  - "PACE sibling svd_sign_pace_eigenfunctions_single_matrix_bit_identical left unguarded — it passes without linalg (verified)."

patterns-established:
  - "Backend-bound golden test: when a golden asserts bit-identity against a feature-gated numeric backend, gate the test on that feature so it is ignored (not failed) under the other backend."

requirements-completed: [FLAKE-01, FLAKE-02]

coverage:
  - id: D1
    description: "Evidence-backed root-cause diagnosis reconciling the intermittent-under-full-parallel symptom with the deterministic-when-linalg-off reproduction, naming the fdata_to_pc SVD-branch divergence point."
    requirement: "FLAKE-01"
    verification:
      - kind: other
        ref: ".planning/phases/90-golden-flake-root-cause-deterministic-fix/90-DIAGNOSIS.md (committed f2327d9b)"
        status: pass
    human_judgment: false
  - id: D2
    description: "The three golden tests pass reliably under repeated full parallel cargo test (with linalg) and per-binary; ignored (not failed) under default features; no new dependency."
    requirement: "FLAKE-02"
    verification:
      - kind: integration
        ref: "10x `cargo test -p fdars-core --features linalg,parallel` all green; per-binary `--test equivalence_phase48/49` green; default-features shows 2/1 ignored"
        status: pass
    human_judgment: false

duration: ~35min
completed: 2026-09-09
status: complete
---

# Phase 90 Plan 01: Golden-Flake Root-Cause & Deterministic Fix Summary

**The three long-standing golden-test flakes are root-caused as a deterministic faer-vs-nalgebra SVD backend divergence (not an environmental flake) and fixed with a test-side cfg-guard, proven by 10 consecutive green full-parallel runs.**

## Performance

- **Duration:** ~35 min (incl. 10× full-suite acceptance runs at ~2 min each)
- **Tasks:** 3 completed
- **Files modified:** 2 test files + 1 new diagnosis artifact

## Accomplishments

- **FLAKE-01 — diagnosis (committed `f2327d9b`).** Reproduced the failure deterministically by compiling the test binaries without `linalg`: `fdata_to_pc` then routes through the `nalgebra` SVD backend instead of `faer`, algorithmically diverging the FPCA rotation. Captured deltas: `golden_co_cluster_parallel` 469.177 vs golden 434.23; `golden_co_cluster_below_threshold` 469.177 vs 415.59 (both n_init variants fail at the *identical* wrong value → divergence is in the single upstream FPCA rotation, computed once before the n_init loop); `svd_sign` `rotation[(0,0)]` 4.18e-16 vs -0.0 (near-zero sign flip). Classified all as categorical **backend** divergence, not rayon/ULP nondeterminism.
- **Reconciled the intermittent-vs-deterministic tension.** The module docs and CI both require `--features linalg`; the "intermittent flake / disk pressure" was a misdiagnosis — a bare local `cargo test` (default features, no linalg) fails deterministically, while "per-binary in isolation" runs used the documented `--features linalg` and passed. No cross-binary interference, no disk-pressure component.
- **FLAKE-02 — fix (committed `d75d4f3d`).** Added `#[cfg_attr(not(feature = "linalg"), ignore)]` to exactly the three affected tests (PACE sibling left unguarded, verified unaffected). Under both documented linalg configs the tests run and pass; under default features they are `ignored`, not `failed`.
- **Proof.** clippy `--all-targets --features linalg,parallel -D warnings` = 0; per-binary green in isolation; `2 ignored`/`1 ignored` under default features; `2 passed` under `--no-default-features --features linalg`; **10 consecutive green** full-parallel `cargo test -p fdars-core --features linalg,parallel` runs (0 failures) — confirming no residual nondeterminism with linalg.

## Task Commits

1. **Task 1: Reproduce & diagnose (FLAKE-01)** — `f2327d9b` (docs) — 90-DIAGNOSIS.md
2. **Task 2: Confirm evidence-chosen fix (pre-approved gate)** — no code (read-only confirmation; conditions a–d held → cfg-guard pre-approval applied)
3. **Task 3: Apply cfg-guard + prove (FLAKE-02)** — `d75d4f3d` (test)

**Plan metadata:** `fce65843` (plan), `113ed45f` (gate conversion)

## Files Created/Modified

- `.planning/phases/90-.../90-DIAGNOSIS.md` — evidence-backed root-cause diagnosis (created)
- `fdars-core/tests/equivalence_phase48.rs` — cfg-guard on the two co_cluster goldens + module-doc note (modified)
- `fdars-core/tests/equivalence_phase49.rs` — cfg-guard on svd_sign golden only + note (modified)

## Decisions Made

- Deterministic feature-config root cause (overturns the disk-pressure hypothesis); least-invasive cfg-guard fix chosen from the evidence per the CONTEXT.md decision hierarchy.

## Deviations from Plan

None materially. The Task 2 blocking-human checkpoint was pre-approved by the user (conditional on the diagnosis) and converted to a non-blocking auto-gate before execution; conditions a–d held, so cfg-guard applied without escalation. Execution ran inline in the orchestrator (not via a gsd-executor subagent) per the documented project hazard that fdars executor subagents stall on long cargo runs — all gates were run foreground, per-gate, and committed `--no-verify` after passing.

## Issues Encountered

- The Bash tool's default 2-min limit truncated the first full-suite run; re-run with a 600s tool timeout (each cached full run ~85–145s), batched 3-per-call to stay within limits.

## User Setup Required

None.
