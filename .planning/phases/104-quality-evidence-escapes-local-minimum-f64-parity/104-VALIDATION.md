---
phase: "104"
slug: "quality-evidence-escapes-local-minimum-f64-parity"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-13"
---

# Phase 104 — Validation Strategy

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
- **Before `/gsd-verify-work`:** Full suite green (incl. RAYON_NUM_THREADS=1 determinism)
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 104-01-01 | 01 | 0 | SDTW-04 | — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel soft_dtw` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Tests covering SDTW-04:
  - escape-from-local-minimum: on a crafted MM-stall dataset, multi-restart total soft-DTW cost
    is strictly lower than the single-init (MM baseline) cost by a clear margin,
  - legacy-wrapper f64 parity / no-divergence: `soft_dtw_barycenter` returns a finite barycenter
    with every coordinate within the data's convex hull (no NaN/inf),
  - deterministic (fixed seed) — passes under full `cargo test` + `RAYON_NUM_THREADS=1`.

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
