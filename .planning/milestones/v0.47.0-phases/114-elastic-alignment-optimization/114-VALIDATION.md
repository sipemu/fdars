---
phase: "114"
slug: "elastic-alignment-optimization"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-23"
---

# Phase 114 — Validation Strategy

> Optimization phase (first real `fdars-core/src` change this milestone). The optimization
> (`dp_fill_grid` band-iteration) MUST be numeric-identical — the validation is dominated by the
> golden-value assertion + the full test suite (release build) + the determinism guardrail, plus
> committed before/after criterion proving the speedup.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]`) + criterion 0.5 (before/after) |
| **Config file** | none — cargo |
| **Quick run command** | `cargo test -p fdars-core --features linalg alignment::` |
| **Full suite command** | `cargo test -p fdars-core --features linalg` then `cargo test -p fdars-core --release --features linalg` (numeric-identity proof) |
| **Estimated runtime** | test suite ~minutes; criterion before/after on the target cell ~minutes |

---

## Sampling Rate

- **After every task commit:** `cargo test -p fdars-core --features linalg alignment::` + the golden-value elastic test.
- **After the optimization commit:** full `cargo test` + `cargo test --release --features linalg` + determinism guardrail + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check`.
- **Before `/gsd-verify-work`:** golden-value ≤1e-10 green; before/after criterion committed showing a statistically significant speedup on `audit_p3_elastic_cross_banded`; all five build variants compile.
- **Max feedback latency:** ~2 min for the quick alignment tests + golden value.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 114-01-01 | 01 | 1 | OPT-01 | unit | `cargo bench -p fdars-core --features linalg,parallel --bench alignment_benchmarks -- audit_p3_elastic_cross_banded/n500_m50` (committed BEFORE-baseline) | ⬜ pending |
| 114-01-02 | 01 | 1 | OPT-01 | unit | golden-value elastic self-distance-matrix test (fixed seed n=20,m=50) asserts ≤1e-10 before vs after | ⬜ pending |
| 114-02-01 | 02 | 2 | OPT-01 | unit | `cargo test -p fdars-core --release --features linalg alignment::` (numeric-identity under release) + determinism guardrail | ⬜ pending |
| 114-02-02 | 02 | 2 | OPT-01 | unit | AFTER criterion on the same cell + committed before/after comparison showing speedup | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Golden-value fixture: a fixed-seed elastic self-distance matrix (n=20, m=50) whose exact values are asserted bit-exact / ≤1e-10 — captured BEFORE the optimization so the after-optimization run can prove identity.
- [ ] Committed BEFORE-baseline criterion for `audit_p3_elastic_cross_banded/n500_m50` (never skip the before-baseline).

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The measured speedup is statistically significant (not noise) | OPT-01 | Criterion CI overlap is a judgment on the before/after distributions | Inspect the committed before/after criterion: after-CI upper bound < before-CI lower bound (non-overlapping), on the noise-stable banded cell |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Golden-value ≤1e-10 assertion present and green (numeric identity)
- [ ] `cargo test --release --features linalg` green (release-build numeric identity)
- [ ] Determinism guardrail green
- [ ] Before/after criterion committed showing a non-overlapping-CI speedup
- [ ] No watch-mode flags
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
