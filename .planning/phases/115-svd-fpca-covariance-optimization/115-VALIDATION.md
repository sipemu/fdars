---
phase: "115"
slug: "svd-fpca-covariance-optimization"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-23"
---

# Phase 115 — Validation Strategy

> Optimization phase (OPT-02) + the consolidated cross-cutting proof (OPT-03). The covariance
> optimization (`sym_cov_irreg` upper-triangle + mirror) MUST be numeric-identical (bit-exact on the
> upper triangle; the mirrored half is a bit-copy). Validation = golden-value ≤1e-10 + full/release
> test suite + determinism guardrail + five builds + pmat quality-gate + committed before/after
> criterion proving the ~2× speedup.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness + criterion 0.5 (before/after) + pmat quality-gate |
| **Config file** | none — cargo |
| **Quick run command** | `cargo test -p fdars-core --features linalg irreg_fdata::face` + the golden fixture |
| **Full suite command** | `cargo test -p fdars-core --features linalg` then `cargo test -p fdars-core --release --features linalg` |
| **Estimated runtime** | test suite ~minutes; criterion before/after ~minutes; pmat gate ~1–2 min |

---

## Sampling Rate

- **After every task commit:** `cargo test -p fdars-core --features linalg irreg_fdata::face` + golden-value covariance test.
- **After the optimization commit:** full `cargo test` + `cargo test --release --features linalg` + determinism guardrail + clippy `--all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check`.
- **OPT-03 (end of phase):** all five build variants compile + pmat quality-gate green + committed before/after criterion showing the covariance speedup.
- **Max feedback latency:** ~2 min for the quick face tests + golden value.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 115-01-01 | 01 | 1 | OPT-02 | unit | committed BEFORE-baseline criterion for the `face_covariance` group (largest clean cell) | ⬜ pending |
| 115-01-02 | 01 | 1 | OPT-02 | unit | golden-value `face_covariance` test (fixed seed) asserts ≤1e-10 before vs after | ⬜ pending |
| 115-02-01 | 02 | 2 | OPT-02 | unit | `sym_cov_irreg` refactor → golden green (debug + `--release --features linalg`) + AFTER criterion showing non-overlapping-CI ~2× speedup | ⬜ pending |
| 115-02-02 | 02 | 2 | OPT-02 | unit | SVD re-benchmark at higher sample counts → documented near-optimal (or clean target if one emerges) | ⬜ pending |
| 115-03-01 | 03 | 3 | OPT-03 | integration | consolidated proof: release test + determinism guardrail + 5 builds compile + pmat quality-gate green + fmt/clippy | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Golden-value `face_covariance` fixture (fixed seed) capturing the current output bit-exact / ≤1e-10 BEFORE the optimization.
- [ ] Committed BEFORE-baseline criterion for the `face_covariance` group (never skip the before-baseline).
- [ ] Confirm the `--bench` binary for the `face_covariance` criterion group (`face_benchmarks` or `perf_hotpaths` — verify via `--list`).

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The measured covariance speedup is statistically significant | OPT-02 | Criterion CI overlap is a judgment on the before/after distributions | Inspect the committed before/after criterion: after-CI upper < before-CI lower (non-overlapping) — expect ~2× on the noise-stable face_covariance cell |
| pmat quality-gate is green | OPT-03 | Requires running `pmat quality-gate` on the host | `pmat quality-gate --checks complexity,satd,dead-code,duplicates` exits 0 (any new helper ≤23 cognitive complexity) |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Golden-value ≤1e-10 present and green (numeric identity of the covariance optimization)
- [ ] `cargo test --release --features linalg` green (OPT-03)
- [ ] Determinism guardrail green (OPT-03)
- [ ] All five build variants compile (OPT-03)
- [ ] pmat quality-gate green (OPT-03)
- [ ] Before/after criterion committed showing a non-overlapping-CI covariance speedup
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
