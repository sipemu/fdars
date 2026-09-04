---
phase: "67"
slug: "automatic-lambda-selection-gcv-reml"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-04"
---

# Phase 67 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)] mod tests` in `peer.rs`) |
| **Config file** | none — existing crate test infrastructure |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel peer::` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~5 s (module tests) / minutes (full suite) |

---

## Sampling Rate

- **After every task commit:** `cargo test -p fdars-core --features linalg,parallel peer::`
- **After every plan wave:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + module tests
- **Before `/gsd-verify-work`:** Full suite green
- **Max feedback latency:** ~10 s (module tests)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 67-01-01 | 01 | 1 | PER-03 | — | `LambdaChoice::Fixed(λ)` used verbatim, no search; λ unchanged in result | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ✅ (Phase 66) | ⬜ pending |
| 67-01-02 | 01 | 1 | PER-03 | — | GCV picks grid-argmin λ; deterministic (two runs → same λ) | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ✅ | ⬜ pending |
| 67-01-03 | 01 | 1 | PER-03 | — | REML picks a sensible positive λ; variance components σ²≥0; deterministic | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ✅ | ⬜ pending |
| 67-01-04 | 01 | 1 | PER-03 | — | REML-vs-GCV β(t) agree within documented tolerance; both recover known β on SNR data | unit | `cargo test -p fdars-core --features linalg,parallel peer::` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Extend `peer.rs` inline tests with a synthetic **SNR fixture** (known β(t), controlled noise) so both selectors land a non-degenerate λ.
- [ ] Mechanically update the 8 Phase 66 `PeerConfig { lambda: <f64> }` test constructions + the module-header doctest to `LambdaChoice::Fixed(<f64>)`.

*Existing Rust test infrastructure covers the framework; no new deps.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated known-answer verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
