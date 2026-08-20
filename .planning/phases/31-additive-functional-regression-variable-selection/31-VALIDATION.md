---
phase: 31
slug: additive-functional-regression-variable-selection
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-20
---

# Phase 31 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — Cargo workspace |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel additive` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~90 seconds full suite; ~5s for additive-only |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg,parallel additive`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 31-01-01 | 01 | 1 | REG-04 | — / — | FAM recovers known additive signal; invalid input → FdarError | unit | `cargo test -p fdars-core --features linalg,parallel additive::tests::fam` | ❌ W0 | ⬜ pending |
| 31-01-02 | 01 | 1 | REG-04 | — / — | GKAM/GSAM fit + predict, invariants hold | unit | `cargo test -p fdars-core --features linalg,parallel additive::tests::gkam` | ❌ W0 | ⬜ pending |
| 31-02-01 | 02 | 2 | REG-04 | — / — | variable_selection recovers active subset | unit | `cargo test -p fdars-core --features linalg,parallel additive::tests::var_select` | ❌ W0 | ⬜ pending |
| 31-02-02 | 02 | 2 | REG-04 | — / — | permutation test: small p under effect, ns under null (seeded) | unit | `cargo test -p fdars-core --features linalg,parallel additive::tests::perm` | ❌ W0 | ⬜ pending |
| 31-02-03 | 02 | 2 | REG-04 | — / — | history-index recovers lagged effect; invalid lag → FdarError | unit | `cargo test -p fdars-core --features linalg,parallel additive::tests::history` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

*Task IDs are indicative — the planner sets the authoritative task breakdown; this map is refined during validate-phase.*

---

## Wave 0 Requirements

- [ ] `fdars-core/src/scalar_on_function/additive.rs` — new module with inline `#[cfg(test)] mod tests`
- [ ] No new test framework needed — Rust built-in harness covers all requirements.

*Existing infrastructure covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification (synthetic-recovery + invariant + seeded-permutation + invalid-input tests).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
