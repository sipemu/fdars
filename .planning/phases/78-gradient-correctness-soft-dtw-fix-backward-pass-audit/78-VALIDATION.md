---
phase: "78"
slug: "gradient-correctness-soft-dtw-fix-backward-pass-audit"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-06"
---

# Phase 78 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` / `#[test]`) |
| **Config file** | none — `fdars-core/Cargo.toml` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel soft_dtw` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~30–90 seconds (2859 tests; commit hook times out at 30s → use `--no-verify`) |

---

## Sampling Rate

- **After every task commit:** Run the soft_dtw-scoped quick command
- **After every plan wave:** Run the full suite
- **Before `/gsd-verify-work`:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check` must be green
- **Max feedback latency:** ~90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 78-01-01 | 01 | 1 | CORR-01 | — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel soft_dtw` | ✅ | ⬜ pending |
| 78-01-02 | 01 | 1 | CORR-01 | — | non-zero gradient + barycenter moves from pointwise mean, cross-checked vs Dual/oracle | unit | `cargo test -p fdars-core --features linalg,parallel soft_dtw` | ❌ W0 | ⬜ pending |
| 78-02-01 | 02 | 1 | CORR-02 | — | audit disposition traceable in 78-AUDIT.md | doc+unit | `cargo test -p fdars-core --features linalg,parallel` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] New/tightened `test_soft_dtw_barycenter_*` + non-zero-gradient regression test — for CORR-01
- [ ] `78-AUDIT.md` disposition table — for CORR-02 traceability

*Existing infrastructure (Rust test harness) covers all phase requirements; no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification (unit tests + gates).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
