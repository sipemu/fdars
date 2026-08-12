---
phase: 14
slug: shift-registration
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-12
---

# Phase 14 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` harness (inline `#[cfg(test)] mod tests`) |
| **Config file** | `fdars-core/Cargo.toml` (no separate test config) |
| **Quick run command** | `cargo test -p fdars-core --features linalg -- alignment::shift alignment::quality` |
| **Full suite command** | `cargo test -p fdars-core --features linalg` |
| **Estimated runtime** | ~90 seconds (full suite) |

**Note:** `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` may be required for doctest/bench linking — `/tmp` tmpfs exhaustion causes bogus "No space left". Use `--no-verify` for docs commits.

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p fdars-core --features linalg -- alignment`
- **After every plan wave:** Run `cargo test -p fdars-core --features linalg`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~90 seconds

---

## Per-Task Verification Map

| Req ID | Behavior | Test Type | Automated Command | File Exists | Status |
|--------|----------|-----------|-------------------|-------------|--------|
| FEAT-06-A | Already-aligned (constant) set → δᵢ ≈ 0.0 ∀i | unit | `cargo test -p fdars-core --features linalg -- test_shift_already_aligned` | ❌ W0 | ⬜ pending |
| FEAT-06-B | Injected offset δ recovered within tolerance (δ=0.1 → δᵢ ≈ 0.1 ± 1e-4) | unit | `cargo test -p fdars-core --features linalg -- test_shift_recovers_injected_offset` | ❌ W0 | ⬜ pending |
| FEAT-06-C | Registered curves re-evaluated at correct shifted argvals (spot-check values) | unit | `cargo test -p fdars-core --features linalg -- test_shift_registration_curve_values` | ❌ W0 | ⬜ pending |
| FEAT-06-D | Empty data → `Err(FdarError::InvalidDimension)` | unit | `cargo test -p fdars-core --features linalg -- test_shift_registration_empty_data` | ❌ W0 | ⬜ pending |
| FEAT-06-E | Argvals length mismatch → `Err(FdarError::InvalidDimension)` | unit | `cargo test -p fdars-core --features linalg -- test_shift_registration_argvals_mismatch` | ❌ W0 | ⬜ pending |
| FEAT-07-A | `least_squares_score` on identical constant curves = 0.0 | unit | `cargo test -p fdars-core --features linalg -- test_ls_score_identical_curves` | ❌ W0 | ⬜ pending |
| FEAT-07-B | `least_squares_score` drops after registration on shifted bumps | unit | `cargo test -p fdars-core --features linalg -- test_ls_score_drops_after_registration` | ❌ W0 | ⬜ pending |
| FEAT-07-C | `pairwise_correlation_score` rises after registration on shifted bumps | unit | `cargo test -p fdars-core --features linalg -- test_pairwise_corr_rises_after_registration` | ❌ W0 | ⬜ pending |
| FEAT-07-D | `pairwise_correlation_score` with n=1 → `Err(FdarError::InvalidParameter)` | unit | `cargo test -p fdars-core --features linalg -- test_pairwise_corr_n1_error` | ❌ W0 | ⬜ pending |
| FEAT-07-E | `sobolev_least_squares_score` with λ=0 equals `least_squares_score` | unit | `cargo test -p fdars-core --features linalg -- test_sobolev_score_lambda_zero` | ❌ W0 | ⬜ pending |
| FEAT-07-F | `sobolev_least_squares_score` with λ>0 ≥ λ=0 score (penalty adds non-negative term) | unit | `cargo test -p fdars-core --features linalg -- test_sobolev_score_lambda_positive` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fdars-core/src/alignment/shift.rs` — new file with inline `#[cfg(test)] mod tests` covering FEAT-06-A…E, using a shared `make_shifted_bumps(n, m, delta)` / `gaussian_bump(argvals, mu, sigma)` synthetic fixture (defined inline).
- [ ] `fdars-core/src/alignment/quality.rs` — added tests covering FEAT-07-A…F (reuse the same shifted-bumps fixture).

*Built-in `#[test]` harness already configured — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| — | — | — | All phase behaviors have automated verification. |

---

## Security Domain (ASVS L1)

Pure numerical library addition — no I/O, auth, network, or file access. Applicable ASVS categories:
- **V5 Input Validation:** dimension checks at every public-fn entry (`data.ncols() == argvals.len()`, `nrows > 0`, `argvals.len() >= 2`, `max_shift > 0`, `lambda >= 0`, pairwise needs `n >= 2`).
- **V7 Error Handling:** all failures via `Result<T, FdarError>`; no panic paths (entry checks prevent index-out-of-bounds in loops).

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter (by `/gsd-validate-phase`)

**Approval:** pending
