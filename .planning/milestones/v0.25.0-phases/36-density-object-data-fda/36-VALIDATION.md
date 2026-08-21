---
phase: 36
slug: density-object-data-fda
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-21
---

# Phase 36 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — `fdars-core/Cargo.toml` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel density_fda::` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~90–180s (full); ~20s (module-scoped) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg,parallel density_fda::`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 36-*-* | * | — | DENS-01 | — / — | N/A (pure numeric lib, no untrusted I/O) | unit | `cargo test -p fdars-core --features linalg,parallel density_fda::` | ❌ W0 (density_fda.rs created this phase) | ⬜ pending |

*Populated by the planner/executor. Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fdars-core/src/density_fda.rs` — new module with inline `#[cfg(test)]` tests for `lqd_transform`/`inverse_lqd` (round-trip), `lqd_fpca` (FVE monotone), `wasserstein_barycenter` (reduction properties), `normalize_density` (integral-to-1)
- [ ] `pub mod density_fda;` in `src/lib.rs` + crate-root re-exports

*Existing cargo test harness + `fdata_to_pc_1d`/`helpers` cover all phase needs — no framework install, no new dependency.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | All phase behaviors have automated `#[cfg(test)]` verification (LQD round-trip within tolerance, inverse normalized+nonneg, FVE monotone→1, single-mode PC capture, barycenter reduction, normalization integral-to-1). |

---

## Validation Sign-Off

- [ ] All tasks have automated `#[cfg(test)]` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (new density_fda module)
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
