---
phase: 35
slug: basis-system-completions
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-21
---

# Phase 35 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — `fdars-core/Cargo.toml` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel basis:: multi_fdata:: pda` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~90–180s (full); ~20–40s (module-scoped) |

---

## Sampling Rate

- **After every task commit:** Run the module-scoped quick command for the touched module.
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel` (relevant modules).
- **Before `/gsd-verify-work`:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.
- **Max feedback latency:** ~180 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 35-*-* | * | — | REP-01 | — / — | N/A (pure numeric lib, no untrusted I/O) | unit | `cargo test -p fdars-core --features linalg,parallel basis:: multi_fdata:: pda` | ❌ W0 (new bases + multi_fdata.rs + pda this phase) | ⬜ pending |

*Populated by the planner/executor. Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] New basis factories (`monomial_basis`/`exponential_basis`/`power_basis`/`polygonal_basis` + `BasisSystem`) with closed-form reference tests
- [ ] `fdars-core/src/multi_fdata.rs` (`MultiFunData`) with invariant tests
- [ ] `Lfd` + `principal_differential_analysis` + `PdaResult` with a harmonic-ODE recovery test
- [ ] Crate-root re-exports + `pub mod multi_fdata;` in `src/lib.rs`

*Existing cargo test harness covers all phase requirements — no framework install needed. Existing `bspline`/`fourier`/`constant` factories and `smooth_basis`/`pspline` penalties are UNTOUCHED.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | All phase behaviors have automated `#[cfg(test)]` verification (closed-form basis values, penalty symmetry/PSD, MultiFunData invariants, Lfd on known function, PDA harmonic-ODE recovery). |

---

## Validation Sign-Off

- [ ] All tasks have automated `#[cfg(test)]` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (new bases + multi_fdata + pda)
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
