---
phase: 40
slug: fr-chet-object-data-regression
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-22
---

# Phase 40 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — Cargo built-in |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel frechet` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~90 seconds (full), ~10s (module) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg,parallel frechet`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 40-XX | TBD | TBD | FRE-01-01..08 | — / — | N/A (pure numeric lib) | unit | `cargo test -p fdars-core --features linalg,parallel frechet` | ❌ W0 | ⬜ pending |

*Populated per-plan by the planner; each FRE-01 requirement maps to inline `#[cfg(test)]` tests in `frechet/`.*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fdars-core/src/frechet/` new module dir with `mod.rs` + submodules, inline `#[cfg(test)] mod tests` covering FRE-01-01..08
- [ ] Wire `mod frechet; pub use frechet::{...}` into `src/lib.rs` (crate-root re-export)

*Rust inline-test convention: no separate test files or framework install needed — existing harness covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification via inline `#[cfg(test)]` tests (identical-object zero-variance, dispersion monotonicity, barycenter agreement, W₂ hand-computed reference, regression tracking a known predictor→density relationship, Fréchet ANOVA flags/ignores group differences with seeded permutation reproducibility).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
