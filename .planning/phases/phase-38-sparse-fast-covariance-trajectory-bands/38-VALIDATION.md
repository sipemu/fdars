---
phase: 38
slug: sparse-fast-covariance-trajectory-bands
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-08-21
---

# Phase 38 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` + `#[cfg(test)]` (inline modules) |
| **Config file** | none — uses cargo test |
| **Quick run command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel irreg_fdata::face -- --test-threads=4` |
| **Full suite command** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~90s (full); quick ~5s |

**Note on TMPDIR:** Use `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` for build/doctest linking to avoid /tmp tmpfs exhaustion (MEMORY.md).

---

## Sampling Rate

- **After every task commit:** `cargo test -p fdars-core --features linalg,parallel irreg_fdata::face -- --test-threads=4`
- **After every plan wave:** `cargo test -p fdars-core --features linalg,parallel`
- **Before `/gsd-verify-work`:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- **Max feedback latency:** ~90s

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 38-face-01 | TBD | 1 | SPARSE-01-01 | unit | `cargo test ... face::tests::test_face_covariance_shape` | ❌ W0 | ⬜ pending |
| 38-face-02 | TBD | 1 | SPARSE-01-01 | unit | `cargo test ... face::tests::test_face_covariance_dense_limit` | ❌ W0 | ⬜ pending |
| 38-face-03 | TBD | 1 | SPARSE-01-01 | unit | `cargo test ... face::tests::test_face_covariance_errors` | ❌ W0 | ⬜ pending |
| 38-mface-01 | TBD | 2 | SPARSE-01-02 | unit | `cargo test ... face::tests::test_mface_shape` | ❌ W0 | ⬜ pending |
| 38-mface-02 | TBD | 2 | SPARSE-01-02 | unit | `cargo test ... face::tests::test_mface_known_structure` | ❌ W0 | ⬜ pending |
| 38-mface-03 | TBD | 2 | SPARSE-01-02 | unit | `cargo test ... face::tests::test_mface_errors` | ❌ W0 | ⬜ pending |
| 38-traj-01 | TBD | 2 | SPARSE-01-03 | unit | `cargo test ... face::tests::test_face_trajectory_bands` | ❌ W0 | ⬜ pending |
| 38-traj-02 | TBD | 2 | SPARSE-01-03 | unit | `cargo test ... face::tests::test_face_trajectory_delegation` | ❌ W0 | ⬜ pending |
| 38-reexport | TBD | 2 | SPARSE-01-03 | unit | `cargo test ... face::tests::test_reexports` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `fdars-core/src/irreg_fdata/face.rs` — entire new file
- [ ] `gaussian_smooth_cov` in `fpca_variants.rs` → `pub(crate) fn` (single-line visibility change; no public API impact)
- [ ] `irreg_fdata/mod.rs` — `pub mod face;` + `pub use face::{...}`
- [ ] `lib.rs` — crate-root `pub use` for `face_covariance`, `mface_covariance`, result struct, `face_trajectory`

*Existing test infrastructure covers all phase requirements — no framework install needed.*

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
