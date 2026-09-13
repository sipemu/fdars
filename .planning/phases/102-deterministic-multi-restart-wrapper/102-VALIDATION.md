---
phase: "102"
slug: "deterministic-multi-restart-wrapper"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-13"
---

# Phase 102 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]`) |
| **Config file** | none — inline `mod tests` in `fdars-core/src/metric/soft_dtw.rs` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel soft_dtw` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–120 seconds (full suite) |

---

## Sampling Rate

- **After every task commit:** `cargo test -p fdars-core --features linalg,parallel soft_dtw`
- **After every plan wave:** `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 102-01-01 | 01 | 0 | SDTW-02 | — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel soft_dtw` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Unit tests covering SDTW-02:
  - multi-restart from mean + seeded perturbations returns lowest-total-cost barycenter,
  - kept barycenter total cost ≤ every individual restart's cost,
  - same seed + inputs → bit-identical barycenter (determinism),
  - restart 0 (unperturbed mean) result ≤ multi-restart never worse than single-init,
  - wasm32 compile guard (`StdRng::seed_from_u64` — no OS entropy).

*Existing regression guard: `test_soft_dtw_barycenter_vs_tslearn` continues to pass.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
