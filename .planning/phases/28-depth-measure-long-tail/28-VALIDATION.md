---
phase: 28
slug: depth-measure-long-tail
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-19
---

# Phase 28 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[cfg(test)]` (inline module tests) |
| **Config file** | none — Cargo built-in test harness |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel depth::` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–120 seconds (full suite; 2100+ tests) |

---

## Sampling Rate

- **After every task commit:** Run the quick command (`depth::` filter — new-measure tests only)
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must be green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

> One measure family per plan file; each new measure carries inline `#[cfg(test)]` tests asserting
> the known depth ordering (central curve deepest, magnitude/shape outliers shallowest) plus error paths.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 28-01-01 | 01 | 1 | DEPTH-01 | — / — | N/A (numeric library) | unit | `cargo test -p fdars-core --features linalg,parallel depth::hypo_epi` | ❌ W0 | ⬜ pending |
| 28-01-02 | 01 | 1 | DEPTH-01 | — / — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel depth::half_region` | ❌ W0 | ⬜ pending |
| 28-02-01 | 02 | 2 | DEPTH-01 | — / — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel depth::extremal` | ❌ W0 | ⬜ pending |
| 28-02-02 | 02 | 2 | DEPTH-01 | — / — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel depth::erl` | ❌ W0 | ⬜ pending |
| 28-02-03 | 02 | 2 | DEPTH-01 | — / — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel depth::linf` | ❌ W0 | ⬜ pending |
| 28-03-01 | 03 | 3 | DEPTH-01 | — / — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel depth::tvd` | ❌ W0 | ⬜ pending |
| 28-03-02 | 03 | 3 | DEPTH-01 | — / — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel depth::dispatch` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky. Plan/wave breakdown is provisional — the planner sets the authoritative task IDs.*

---

## Wave 0 Requirements

- [ ] `src/depth/hypo_epi.rs`, `half_region.rs`, `extremal.rs`, `erl.rs`, `linf.rs`, `tvd.rs` — new files with inline `#[cfg(test)] mod tests`
- [ ] No framework install — Rust built-in test harness already present

*Existing infrastructure (Cargo test harness + `depth::tests`) covers all phase requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification — numeric per-curve depth outputs are asserted against known orderings and hand-computed small-case values inline.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
