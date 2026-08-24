---
phase: 44
slug: fem-pde-smoothing-irregular-2d-domains
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-24
---

# Phase 44 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — inline unit tests per source file (project convention) |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~30–180 seconds |

---

## Sampling Rate

- **After every task commit:** module-scoped quick command (`fem_smoothing` / `smooth_basis`)
- **After every plan wave:** full suite command
- **Before `/gsd-verify-work`:** full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

*Seeded pre-planning — the planner/validate-phase fills exact task IDs after PLAN.md exists.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 1 | REP-02-01 (P1 FE basis + assembly) | — | N/A (pure numeric) | unit | `cargo test -p fdars-core --features linalg fem_smoothing` | ❌ W0 | ⬜ pending |
| TBD | — | — | REP-02-02 (SR-PDE surface smoothing) | — | N/A | unit | `cargo test -p fdars-core --features linalg fem_smoothing::smooth` | ❌ W0 | ⬜ pending |
| TBD | — | — | REP-02-03 (positive smoother) | — | N/A | unit | `cargo test -p fdars-core --features linalg smooth_positive` | ❌ W0 | ⬜ pending |
| TBD | — | — | REP-02-04 (monotone smoother) | — | N/A | unit | `cargo test -p fdars-core --features linalg smooth_monotone` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing Rust test harness covers all phase requirements — no framework install needed.
- New inline `#[cfg(test)] mod tests` blocks authored alongside `fem_smoothing.rs` and the additive `smooth_basis.rs` smoothers.

**Test oracles (from RESEARCH.md Validation Architecture §):**
- P1 basis: partition-of-unity (Σ_k φ_k(x)=1 inside domain); linear-field interpolation exactness.
- Assembly: stiffness symmetric + PSD + row-sums≈0 (constant null space); mass symmetric PD.
- SR-PDE: recovers a known smooth surface within tolerance; → interpolation as λ→0; GCV finite.
- Positive smoother: fit ≥ 0 everywhere.
- Monotone smoother: fit nondecreasing (structural guarantee even if NLS underconverges).
- Error paths: bad connectivity index, degenerate (zero-area) triangle, point outside mesh, dimension mismatch → `FdarError` (no panic).

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | All phase behaviors have automated verification (numeric library) | — |

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
