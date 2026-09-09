---
phase: "90"
slug: "golden-flake-root-cause-deterministic-fix"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-09"
---

# Phase 90 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[test]`), criterion for benches |
| **Config file** | none — workspace `Cargo.toml` / `fdars-core/Cargo.toml` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel --test equivalence_phase48 --test equivalence_phase49` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` (the determinism proof) |
| **Estimated runtime** | ~60–180 seconds per full run (heavier under repeated runs) |

---

## Sampling Rate

- **After every task commit:** Run the quick command (the two affected test binaries) under BOTH feature configs (`--features linalg,parallel` AND default features) to prove the cfg-guard behaves.
- **After every plan wave:** Run the full suite command.
- **Before `/gsd-verify-work`:** Full suite must be green; the flake-proof gate (10 consecutive full parallel runs + per-binary green) must pass.
- **Max feedback latency:** ~180 seconds per iteration.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 90-01-01 | 01 | 1 | FLAKE-01 | — | N/A | diagnosis | `cargo test -p fdars-core --features linalg,parallel --test equivalence_phase48 --test equivalence_phase49` (baseline green) vs. default-features run (reproduces divergence) | ✅ | ⬜ pending |
| 90-01-02 | 01 | 2 | FLAKE-02 | — | N/A | integration | `for i in $(seq 1 10); do cargo test -p fdars-core --features linalg,parallel || break; done` (10 consecutive green) | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* The three flaky tests already exist in `fdars-core/tests/equivalence_phase48.rs` and `equivalence_phase49.rs`; no new test framework or fixtures are needed. The fix modifies existing test attributes and adds a diagnosis artifact.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Flake is genuinely gone under repeated full parallel runs | FLAKE-02 | Intermittent flake — a single green run is insufficient proof | Run the full suite 10 times consecutively (acceptance bar); confirm zero failures, and run each affected binary in isolation. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
