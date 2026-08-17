---
phase: 24
slug: concurrent-varying-coefficient-regression
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-17
---

# Phase 24 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — Cargo workspace at `/home/simonm/projects/rust/fdars/Cargo.toml` |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib concurrent_regression` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~25 seconds (full lib suite) |

---

## Sampling Rate

- **After every task commit:** Run the quick command (module-scoped tests)
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 24-01-* | 01 | 1 | REG-01 | — | N/A (pure numeric library, no untrusted input surface) | unit | `cargo test -p fdars-core --features linalg,parallel --lib concurrent_regression` | ❌ W0 (new module) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Behaviors requiring automated coverage (SC → test):**
- SC1 — public entry point returns `Result<ConcurrentRegrResult, FdarError>` with `{ beta_curve, intercept, fitted, residuals, argvals }` (smoke/shape test).
- SC2 — recovery: known β(t) + low noise → recovered `beta_curve` within tolerance.
- SC3 — monotone smoothness: larger bandwidth → smaller Σ second-difference² of `beta_curve`.
- SC4 — consistency: `residuals == response − fitted` pointwise; invalid inputs (mismatched grids/dims, empty data) → appropriate `FdarError`, no panic.
- SC5 — additive/non-breaking: full suite + clippy `--all-targets` green; no existing signature changed.

---

## Wave 0 Requirements

- [ ] `concurrent_regression.rs` inline `#[cfg(test)] mod tests` — new module carries its own tests (shape, recovery, smoothness, consistency, error guards).

*Existing test harness (built-in `cargo test`) covers execution; no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification (numeric assertions in inline tests).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
