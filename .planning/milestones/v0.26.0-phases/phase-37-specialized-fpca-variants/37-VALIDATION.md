---
phase: 37
slug: specialized-fpca-variants
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-21
---

# Phase 37 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `#[cfg(test)]` (inline modules) |
| **Config file** | none — uses cargo test |
| **Quick run command** | `cargo test -p fdars-core fpca_variants -- --test-threads=4` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~90 seconds (full suite); quick run ~5s |

**Note on TMPDIR:** Use `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` for build/doctest linking to avoid /tmp tmpfs exhaustion (MEMORY.md).

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core fpca_variants -- --test-threads=4`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- **Max feedback latency:** ~90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 37-fpca_der-01 | TBD | 1 | FPCA-02-01 | — | N/A | unit | `cargo test -p fdars-core test_fpca_der` | ❌ W0 | ⬜ pending |
| 37-fpca_der-02 | TBD | 1 | FPCA-02-01 | — | N/A | unit | `cargo test -p fdars-core test_fpca_der_nderiv0` | ❌ W0 | ⬜ pending |
| 37-fpca_der-03 | TBD | 1 | FPCA-02-01 | — | invalid inputs → FdarError | unit | `cargo test -p fdars-core test_fpca_der_errors` | ❌ W0 | ⬜ pending |
| 37-fsvd-01 | TBD | 1 | FPCA-02-02 | — | N/A | unit | `cargo test -p fdars-core test_fsvd_unit_norm` | ❌ W0 | ⬜ pending |
| 37-fsvd-02 | TBD | 1 | FPCA-02-02 | — | N/A | unit | `cargo test -p fdars-core test_fsvd_rank1` | ❌ W0 | ⬜ pending |
| 37-fsvd-03 | TBD | 1 | FPCA-02-02 | — | mismatched n → FdarError | unit | `cargo test -p fdars-core test_fsvd_errors` | ❌ W0 | ⬜ pending |
| 37-crosscov-01 | TBD | 1 | FPCA-02-03 | — | N/A | unit | `cargo test -p fdars-core test_cross_cov_self` | ❌ W0 | ⬜ pending |
| 37-crosscov-02 | TBD | 1 | FPCA-02-03 | — | N/A | unit | `cargo test -p fdars-core test_cross_cov_shape` | ❌ W0 | ⬜ pending |
| 37-crosscov-03 | TBD | 1 | FPCA-02-03 | — | mismatched n → FdarError | unit | `cargo test -p fdars-core test_cross_cov_errors` | ❌ W0 | ⬜ pending |
| 37-dyncorr-01 | TBD | 1 | FPCA-02-04 | — | N/A | unit | `cargo test -p fdars-core test_dyncorr_identical` | ❌ W0 | ⬜ pending |
| 37-dyncorr-02 | TBD | 1 | FPCA-02-04 | — | N/A | unit | `cargo test -p fdars-core test_dyncorr_negated` | ❌ W0 | ⬜ pending |
| 37-dyncorr-03 | TBD | 1 | FPCA-02-04 | — | N/A | unit | `cargo test -p fdars-core test_dyncorr_range` | ❌ W0 | ⬜ pending |
| 37-dyncorr-04 | TBD | 1 | FPCA-02-04 | — | mismatched grids → FdarError | unit | `cargo test -p fdars-core test_dyncorr_errors` | ❌ W0 | ⬜ pending |
| 37-ssvd-01 | TBD | 1 | FPCA-02-05 | — | N/A | unit | `cargo test -p fdars-core test_ssvd_dense_limit` | ❌ W0 | ⬜ pending |
| 37-ssvd-02 | TBD | 1 | FPCA-02-05 | — | N/A | unit | `cargo test -p fdars-core test_ssvd_orthonormality` | ❌ W0 | ⬜ pending |
| 37-ssvd-03 | TBD | 1 | FPCA-02-05 | — | invalid inputs → FdarError | unit | `cargo test -p fdars-core test_ssvd_errors` | ❌ W0 | ⬜ pending |
| 37-reexport-01 | TBD | 1 | FPCA-02-05 | — | N/A | integration | `cargo test -p fdars-core smoke_reexports` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fdars-core/src/fpca_variants.rs` — new file created in Wave 0
- [ ] `FsvdResult` struct defined in `fpca_variants.rs` before `fsvd` is implemented
- [ ] `lib.rs` module declaration (`pub mod fpca_variants;`) + crate-root re-export line

*Existing test infrastructure covers all other phase requirements — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | — |

*All phase behaviors have automated verification (inline `#[cfg(test)]`).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
