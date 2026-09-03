---
phase: "65"
slug: "greedy-selection-integration"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-03"
---

# Phase 65 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (inline `mod tests` in `optimal_design.rs`) |
| **Config file** | none — Cargo `cargo test` runner |
| **Quick run command** | `cargo test -p fdars-core --lib optimal_design --features linalg,parallel` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Bench compile check** | `cargo build --benches -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | quick module run ~seconds; full suite ~minutes |

---

## Sampling Rate

- **After every task commit:** `cargo test -p fdars-core --lib optimal_design --features linalg,parallel`
- **After every plan wave:** full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt -p fdars-core --check`
- **Before `/gsd-verify-work`:** full lib + doctest suite green; `cargo build --benches -p fdars-core --features linalg,parallel` (bench compiles). Do NOT run `cargo bench` in CI (too slow).
- **Max feedback latency:** module-test seconds

---

## Per-Task Verification Map

| Task ID | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|-------------|-----------------|-----------|-------------------|-------------|--------|
| basic | FOD-04 | returns `budget` selected indices | unit | `cargo test ... test_optimal_design_basic` | ❌ W0 | ⬜ pending |
| determinism | FOD-04 | two identical calls → byte-identical indices | unit | `cargo test ... test_determinism_two_calls` | ❌ W0 | ⬜ pending |
| seq==parallel | FOD-04 | identical result with/without `--features parallel` | unit | `cargo test ... test_determinism_two_calls` (both feature sets) | ❌ W0 | ⬜ pending |
| duplicate-free | FOD-04 | no index appears twice | unit | `cargo test ... test_duplicate_free` | ❌ W0 | ⬜ pending |
| monotone trace | FOD-04 | `trace[i+1] <= trace[i] + 1e-12` | unit | `cargo test ... test_monotone_trace` | ❌ W0 | ⬜ pending |
| budget==0 | FOD-04 | → InvalidParameter | unit | `cargo test ... test_validation_budget_zero` | ❌ W0 | ⬜ pending |
| budget>grid | FOD-04 | → InvalidParameter | unit | `cargo test ... test_validation_budget_exceeds_grid` | ❌ W0 | ⬜ pending |
| off-grid | FOD-04 | candidate ∉ argvals → InvalidParameter | unit | `cargo test ... test_validation_off_grid_candidate` | ❌ W0 | ⬜ pending |
| ncomp==0 | FOD-04 | → InvalidParameter | unit | `cargo test ... test_validation_ncomp_zero` | ❌ W0 | ⬜ pending |
| sigma2<=0 | FOD-04 | → InvalidParameter | unit | `cargo test ... test_validation_sigma2_nonpositive` | ❌ W0 | ⬜ pending |
| trajectory first-pt | FOD-05 | selects the numerically-computed argmin first point | unit | `cargo test ... test_trajectory_selects_informative_point` | ❌ W0 | ⬜ pending |
| score(A) | FOD-05 | valid result structure under Score(A) | unit | `cargo test ... test_score_a_selects` | ❌ W0 | ⬜ pending |
| config default | FOD-05 | `OptDesConfig::default()` valid; empty grid caught at call | unit | `cargo test ... test_config_default` | ❌ W0 | ⬜ pending |
| prelude re-export | FOD-05 | `use fdars_core::prelude::*` reaches OptDesConfig | unit/doctest | `cargo test ... test_prelude_reexport` | ❌ W0 | ⬜ pending |
| crate-root re-export | FOD-05 | `fdars_core::optimal_design` reachable | compile | `cargo build -p fdars-core --features linalg,parallel` | ❌ W0 | ⬜ pending |
| module doctest | FOD-05 | end-to-end doctest compiles+passes | doctest | `cargo test -p fdars-core --doc --features linalg,parallel` | ❌ W0 | ⬜ pending |
| bench compiles | FOD-05 | criterion bench builds | compile | `cargo build --benches -p fdars-core --features linalg,parallel` | ❌ W0 | ⬜ pending |
| suite unbroken | Both | 2672+ existing tests still pass | regression | `cargo test -p fdars-core --features linalg,parallel` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Test stubs for the 13 module test functions above (inline `#[cfg(test)] mod tests` in `optimal_design.rs`)
- [ ] `fdars-core/benches/optimal_design.rs` — new bench file
- [ ] `[[bench]] name = "optimal_design" harness = false` stanza in `fdars-core/Cargo.toml`

*Existing `#[cfg(test)] mod tests` block + `synthetic_model` fixture in `optimal_design.rs` cover the test location.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | All phase behaviors have automated verification (determinism, known-answer, compile gates). |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags; no `cargo bench` in CI
- [ ] Feedback latency < module-test seconds
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
