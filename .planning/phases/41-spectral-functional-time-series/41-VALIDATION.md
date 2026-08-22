---
phase: 41
slug: spectral-functional-time-series
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-22
---

# Phase 41 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — `cargo test` on the existing workspace |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel fts::spectral simulation` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–120 seconds (full suite; module-scoped quick run ~10s) |

---

## Sampling Rate

- **After every task commit:** Run the module-scoped quick command for the touched module.
- **After every plan wave:** Run `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + full suite.
- **Before `/gsd-verify-work`:** Full suite + clippy must be green.
- **Max feedback latency:** 120 seconds.

---

## Per-Task Verification Map

> Seeded from RESEARCH.md test oracles; task IDs finalized by the planner. Each requirement maps to at least one automated inline `#[cfg(test)]` test.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 41-01-xx | 01 | 1 | FTS-03-01 | — | White-noise spectral density flat = C_0/(2π) at all Fourier freqs | unit | `cargo test -p fdars-core --features linalg,parallel fts::spectral::tests` | ❌ W0 | ⬜ pending |
| 41-01-xx | 01 | 1 | FTS-03-02 | — | DPCA eigen-filters + scores are finite, correct shapes, sign-aligned | unit | `cargo test -p fdars-core --features linalg,parallel fts::spectral::tests` | ❌ W0 | ⬜ pending |
| 41-01-xx | 01 | 1 | FTS-03-03 | — | Reconstruction error monotone-decreasing in # retained components; rank-1 exact with K=1 | unit | `cargo test -p fdars-core --features linalg,parallel fts::spectral::tests` | ❌ W0 | ⬜ pending |
| 41-02-xx | 02 | 1 | FTS-03-04 | — | VAR(1) with zero operator = pure i.i.d. innovations; bit-identical for same seed | unit | `cargo test -p fdars-core --features linalg,parallel simulation::tests` | ❌ W0 | ⬜ pending |
| 41-02-xx | 02 | 1 | FTS-03-05 | — | FARMA combines AR+MA; rank-1 AR operator → nonzero lag-1 ACF; deterministic under seed | unit | `cargo test -p fdars-core --features linalg,parallel simulation::tests` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Inline `#[cfg(test)] mod tests` added alongside the new `fts/spectral.rs` and the `simulation.rs` additions (no separate test files — crate convention).
- Reuses existing test helpers (`test_helpers::uniform_grid`) and `simulation.rs` KL generators for constructing fixtures.

*Existing Rust test infrastructure covers all phase requirements — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Divergence from `freqdom`/`ftsa` (Re-only Hermitian eigendecomposition; 1/2π pre-factor omission; score trimming) documented in rustdoc | FTS-03-01/02/03 | Prose/rustdoc claim, not a runtime assertion | `cargo doc --no-deps` renders; reviewer confirms divergence notes present in `fts/spectral.rs` module/fn docs |

*All numeric phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
