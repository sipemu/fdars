---
phase: 25
slug: functional-glm-exponential-family
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-17
---

# Phase 25 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — Cargo workspace at `/home/simonm/projects/rust/fdars/Cargo.toml` |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib glm` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~30 seconds (full lib suite) |

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
| 25-01-* | 01 | 1 | REG-02 | — | N/A (pure numeric library, no untrusted input surface) | unit | `cargo test -p fdars-core --features linalg,parallel --lib glm` | ❌ W0 (new module) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Behaviors requiring automated coverage (SC → test):**
- SC1 — `functional_glm(data, y, family, …)` returns `Result<FunctionalGlmResult, FdarError>` with the generalized fields; `GlmFamily{Binomial,Poisson,Gamma,Gaussian}` each with canonical link + variance (shape/smoke test per family).
- SC2 — Binomial parity: `functional_glm(…, Binomial)` coefficients + fitted_values agree with `functional_logistic` within tolerance; `functional_logistic` signature unchanged.
- SC3 — per-family recovery: Poisson (log link) counts + Gamma (inverse link) responses recover the known generative signal within stated tolerance.
- SC4 — IRLS over `fdata_to_pc_1d` FPC scores; invalid inputs (dimension mismatch, out-of-domain response e.g. negative Poisson counts, non-positive Gamma) → appropriate `FdarError`, no panic.
- SC5 — additive/non-breaking: full suite + clippy `--all-targets` green; no new dependency (`statrs` already present).

---

## Wave 0 Requirements

- [ ] `scalar_on_function/glm.rs` inline `#[cfg(test)] mod tests` — new module carries its own tests (per-family shape, Binomial parity, Poisson/Gamma recovery, error guards).

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
