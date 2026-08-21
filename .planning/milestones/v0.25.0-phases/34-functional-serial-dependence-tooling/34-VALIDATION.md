---
phase: 34
slug: functional-serial-dependence-tooling
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-21
---

# Phase 34 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — `fdars-core/Cargo.toml` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel fts::` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~90–180 seconds (full); ~20s (module-scoped) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg,parallel fts::`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 34-01-* | 01 | — | FTS-02 | — / — | N/A (pure numeric lib, no untrusted I/O) | unit | `cargo test -p fdars-core --features linalg,parallel fts::` | ❌ W0 (fts/ created this phase) | ⬜ pending |

*Populated by the planner/executor as plans are created. Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fdars-core/src/fts/mod.rs` + `fdars-core/src/fts/acf.rs` — new module with inline `#[cfg(test)]` tests for each entry point (fACF/fPACF, stationarity, long-run cov, differencing)
- [ ] `pub mod fts;` registered in `src/lib.rs` + crate-root re-exports

*Existing infrastructure (cargo test harness) covers all phase requirements — no test framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | All phase behaviors have automated `#[cfg(test)]` verification (white-noise-band containment, injected-dependence detection, differencing round-trip, stationarity reject/accept, bw=0 → lag-0 covariance). |

---

## Validation Sign-Off

- [ ] All tasks have automated `#[cfg(test)]` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (new `fts/` module)
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
