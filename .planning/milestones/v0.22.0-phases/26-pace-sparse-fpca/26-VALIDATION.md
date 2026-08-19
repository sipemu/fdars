---
phase: 26
slug: pace-sparse-fpca
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-18
---

# Phase 26 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)]` inline modules) |
| **Config file** | none — Cargo workspace at `/home/simonm/projects/rust/fdars/Cargo.toml` |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib pace_fpca` |
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
| 26-01-* | 01 | 1 | FPCA-01 | — | N/A (pure numeric library, no untrusted input surface) | unit | `cargo test -p fdars-core --features linalg,parallel --lib pace_fpca` | ❌ W0 (new module) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Behaviors requiring automated coverage (SC → test):**
- SC1 — `pace_fpca(&IrregFdata, &PaceFpcaConfig)` returns `Result<PaceFpcaResult, FdarError>` with mean, eigenvalues, eigenfunctions, conditional-expectation scores, fitted trajectories + `fitted_lower`/`fitted_upper` bands (shape/smoke test).
- SC2 — recovery: synthetic sparse data from a known generative model (√2 sin/cos eigenfunctions, known λ, Gaussian scores, per-curve 3–8 random points, σ² noise) → recovered eigenfunctions (sign-aligned, corr > 0.95) + scores (corr > 0.8) match ground truth.
- SC3 — reuse-only construction (`cov_irreg` + nalgebra symmetric eig + `mean_irreg` + `helpers::linear_interp` + `linalg::cholesky_solve`); no new dependency (Cargo.toml unchanged).
- SC4 — invalid inputs (empty IrregFdata, too-few-point curve, ncomp too large, non-positive bandwidth/sigma2, mismatched work_grid) → appropriate `FdarError`, no panic; `fitted` lies within `[fitted_lower, fitted_upper]`.
- SC5 — additive/non-breaking: existing FPCA APIs unchanged; full suite + clippy `--all-targets` green.

---

## Wave 0 Requirements

- [ ] `pace_fpca.rs` inline `#[cfg(test)] mod tests` — new module carries its own tests (shape, recovery, band coverage, error guards).

*Existing `cargo test` harness covers execution; no framework install needed.*

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
