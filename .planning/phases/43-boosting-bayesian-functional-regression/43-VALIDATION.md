---
phase: 43
slug: boosting-bayesian-functional-regression
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-23
---

# Phase 43 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — inline unit tests per source file (project convention) |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~90–180 seconds (full suite, 1650+ tests) |

---

## Sampling Rate

- **After every task commit:** Run the quick command (module-scoped `boosting_regression` tests)
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite must be green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

*Seeded pre-planning — the planner/validate-phase fills exact task IDs after PLAN.md exists. Each REG-06 requirement maps to inline `#[cfg(test)]` recovery + error-path tests in the new `src/boosting_regression/` module.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 1 | REG-06-01 | — | N/A (pure numeric) | unit | `cargo test -p fdars-core --features linalg boost_fosr` | ❌ W0 | ⬜ pending |
| TBD | — | — | REG-06-02 | — | N/A | unit | `cargo test -p fdars-core --features linalg boost_fofr` | ❌ W0 | ⬜ pending |
| TBD | — | — | REG-06-03 | — | N/A | unit | `cargo test -p fdars-core --features linalg gamlss` | ❌ W0 | ⬜ pending |
| TBD | — | — | REG-06-04 | — | N/A | unit | `cargo test -p fdars-core --features linalg bayesian` | ❌ W0 | ⬜ pending |
| TBD | — | — | REG-06-05 | — | N/A | unit | `cargo test -p fdars-core --features linalg stability` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing Rust test harness covers all phase requirements — no framework install needed.
- New inline `#[cfg(test)] mod tests` blocks will be authored alongside each submodule (boost_fosr, boost_fofr, gamlss, bayesian, stability).

**Test oracles (from RESEARCH.md Validation Architecture §):**
- Boosted FOSR/FoFR: recover a known functional coefficient on simulated data; training loss decreases monotonically along the boosting path; `r_squared ∈ [0,1]`.
- GAMLSS: recover known μ(t) and σ(t) on heteroscedastic simulated data; σ(t) > 0 everywhere (log-link guard).
- Bayesian FOSR: posterior mean β(t) ≈ penalized/ridge point estimate within tolerance; credible bands cover the truth at nominal rate; determinism under fixed seed.
- Stability selection: selection frequencies ∈ [0,1]; a strong signal predictor exceeds threshold π; determinism under fixed seed.
- Error paths: dimension-mismatch and invalid-parameter inputs return `FdarError` (not panic).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | All phase behaviors have automated verification (numeric library) | — |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
