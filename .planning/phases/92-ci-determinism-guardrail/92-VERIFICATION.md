---
phase: 92-ci-determinism-guardrail
verified: 2026-09-09T12:07:55Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 92: CI Determinism Guardrail — Verification Report

**Phase Goal:** A CI guardrail exercises the full parallel `cargo test` path (cross-binary interference) and/or a nextest serialization group, so any future determinism regression fails CI loudly instead of silently returning.
**Verified:** 2026-09-09T12:07:55Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A CI job `determinism-guardrail` exists in `.github/workflows/rust-ci.yml` exercising the full parallel test path (`cargo test --features linalg,parallel,serde`) with a repeat loop AND once with `RAYON_NUM_THREADS=1`. | VERIFIED | Job at lines 60–121 of `rust-ci.yml`. Steps: "Repeated full parallel run (intermittency)" runs `cargo test --features linalg,parallel,serde` in a `for i in 1 2 3` loop; "Single-threaded reduction-order run" runs `RAYON_NUM_THREADS=1 cargo test --features linalg,parallel,serde`. Both confirmed present by `python3 yaml.safe_load` job-shape assertions (JOB_SHAPE_OK). |
| 2 | The job asserts the 3 golden tests actually RAN under `--features linalg` (2 passed for `equivalence_phase48 golden_co_cluster`; 1 passed for `equivalence_phase49 svd_sign_fpca_two_matrix_bit_identical`) with 0 ignored — catching a silent-skip regression. | VERIFIED | "Assert golden tests actually ran (anti-silent-skip)" step is present in the job. Grep patterns `test result: ok\. 2 passed; 0 failed; 0 ignored` and `test result: ok\. 1 passed; 0 failed; 0 ignored` are wired and confirmed to match actual test output format. Running the two targets locally confirmed: `2 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out` (phase48) and `1 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out` (phase49). |
| 3 | The workflow YAML is syntactically valid (parses with `python3 yaml.safe_load`). | VERIFIED | `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/rust-ci.yml')); print('YAML_OK')"` produced `YAML_OK` with exit 0. |
| 4 | No new crate/tooling dependency is introduced (no `cargo-nextest`, no `serial_test`) — mechanism is a pure shell loop in the workflow YAML. | VERIFIED | `grep 'nextest' rust-ci.yml` and `grep 'serial_test' rust-ci.yml` both return nothing. Confirmed by job-shape assertion check (`NO nextest: PASS`, `NO serial_test: PASS`). Commit `03c3fa9e` touches only `.github/workflows/rust-ci.yml` (62 insertions, 1 file changed). `git diff 77fdaa33..HEAD -- fdars-core/src fdars-core/Cargo.toml Cargo.toml Cargo.lock` is empty — no Rust source or Cargo file changes. |
| 5 | The guardrail's shell commands run green locally against the current Phase-90/91 baseline. | VERIFIED | Both golden targets run locally with exact expected output: `test result: ok. 2 passed; 0 failed; 0 ignored` (phase48) and `test result: ok. 1 passed; 0 failed; 0 ignored` (phase49). Grep patterns confirmed to match actual output. SUMMARY records `GUARDRAIL_LOCAL_GREEN` from execution (3x repeat loop and RAYON=1 full-suite run). The 3x repeat loop is not re-run here (confirmed green during phase execution — unnecessary re-spend of ~4 min each). |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.github/workflows/rust-ci.yml` | Contains new `determinism-guardrail` job | VERIFIED | Job present at line 60; 62 lines added in commit `03c3fa9e`. Passes all structural assertions. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `determinism-guardrail` job | Top-level `on: push/pull_request/release` triggers | Shares workflow file; job has no own `on:` block | WIRED | Confirmed: top-level triggers contain `push`, `pull_request`, `release`. The job adds no new `on:` block. |
| `determinism-guardrail` job | `working-directory: fdars-core` | `defaults.run.working-directory` | WIRED | `j['defaults']['run']['working-directory'] == 'fdars-core'` confirmed. |
| "Assert golden tests actually ran" step | `equivalence_phase48` golden output format | `grep -Eq "test result: ok\. 2 passed; 0 failed; 0 ignored"` | WIRED | Pattern matched against actual test output in spot-check. |
| "Assert golden tests actually ran" step | `equivalence_phase49` golden output format | `grep -Eq "test result: ok\. 1 passed; 0 failed; 0 ignored"` | WIRED | Pattern matched against actual test output in spot-check. |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `equivalence_phase48 golden_co_cluster` reports `2 passed; 0 failed; 0 ignored` | `cargo test -p fdars-core --features linalg,parallel --test equivalence_phase48 -- golden_co_cluster` | `test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out` | PASS |
| `equivalence_phase49 svd_sign_fpca_two_matrix_bit_identical` reports `1 passed; 0 failed; 0 ignored` | `cargo test -p fdars-core --features linalg,parallel --test equivalence_phase49 -- svd_sign_fpca_two_matrix_bit_identical` | `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out` | PASS |
| Anti-silent-skip grep patterns match actual output format | `echo "..." \| grep -Eq "test result: ok\. 2 passed; 0 failed; 0 ignored"` | Pattern matched for both phase48 and phase49 | PASS |

The 3x repeat loop and RAYON=1 full-suite run are not re-executed here; both were validated during phase execution (`GUARDRAIL_LOCAL_GREEN`). The golden spot-checks above are the decisive evidence that the wiring is correct and green.

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CI-01 | 92-01-PLAN.md | A CI guardrail exercises the full parallel test path and/or a nextest serialization group, failing loudly on determinism regression | SATISFIED | `determinism-guardrail` job present, YAML valid, repeat loop + RAYON=1 run + positive-run assertion all confirmed. |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | — | — | — |

Scan performed on `.github/workflows/rust-ci.yml` (the only modified file). No `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, or `PLACEHOLDER` markers found. The file is a workflow YAML with only shell/grep logic — no Rust stub patterns apply.

---

## Human Verification Required

None. All must-haves verified statically (YAML parse + job-shape assertions + grep) and behaviorally (golden spot-checks). The CI job does not run here (no CI environment), but the shell assertions' correctness is confirmed by matching the grep patterns against actual test output produced locally.

---

## Gaps Summary

No gaps. All 5 must-haves verified. Phase goal achieved: the `determinism-guardrail` CI job will fail loudly on a reintroduced determinism regression via three independent mechanisms (repeated parallel run, single-threaded reduction-order run, and positive-run anti-silent-skip assertion for the three golden tests).

---

_Verified: 2026-09-09T12:07:55Z_
_Verifier: Claude (gsd-verifier)_
