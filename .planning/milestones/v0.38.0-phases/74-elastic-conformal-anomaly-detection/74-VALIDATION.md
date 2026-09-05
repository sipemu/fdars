---
phase: "74"
slug: "elastic-conformal-anomaly-detection"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-05"
---

# Phase 74 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)] mod tests`) + doctests |
| **Config file** | none — inline module tests in `fdars-core/src/` |
| **Quick run command** | `cargo test -p fdars-core --lib --features linalg,parallel conformal_anomaly` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–120 seconds (full suite ~2809+ tests) |

---

## Sampling Rate

- **After every task commit:** Run the quick command scoped to the new conformal_anomaly tests
- **After every plan wave:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check`
- **Before `/gsd-verify-work`:** Full suite green, clippy clean, fmt clean, `cargo test --doc` green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 74-01-* | 01 | 1 | ECA-01 | — | N/A | unit | `cargo test -p fdars-core --lib --features linalg,parallel conformal_anomaly` | ❌ W0 | ⬜ pending |
| 74-01-* | 01 | 1 | ECA-02 | — | N/A | unit | `cargo test -p fdars-core --lib --features linalg,parallel conformal_anomaly` | ❌ W0 | ⬜ pending |
| 74-01-* | 01 | 1 | ECA-03 | — | N/A | doctest | `cargo test -p fdars-core --features linalg,parallel --doc conformal_anomaly` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Validation Architecture (make-or-break behavioral gates)

1. **Nonconformity zero-for-identical + non-negativity (ECA-01)** — `elastic_nonconformity(curve, curve, argvals, lambda, variant)` (curve scored against itself as template) is < 1e-4 for every variant; all scores ≥ 0 on arbitrary inputs. (Use a **tolerance** ~1e-4 — the underlying elastic distances are only "near zero", not exactly zero, for identical curves per RESEARCH.)
2. **Marginal validity (ECA-02)** — on exchangeable clean data (calibration + test drawn from the same generator), the empirical flag rate ≈ α within a tolerance band (e.g. |rate − α| ≤ small margin over a moderate test set). Deterministic fixture (fixed seed).
3. **Magnitude outlier flagged (ECA-02)** — an injected scaled/shifted (amplitude) outlier is flagged by AmplitudeElastic/CombinedElastic.
4. **Shape outlier flagged (ECA-02)** — an injected warped/phase-distorted (shape) outlier is flagged by PhaseElastic/CombinedElastic.
5. **p-value + threshold correctness (ECA-02/03)** — p = (1 + #{calib ≥ test})/(n_calib+1); flags == (p ≤ α); `threshold` == the (1−α) empirical quantile of calibration scores; `ConformalAnomalyResult` fields populated with correct lengths.
6. **Band path unchanged (ECA-03 regression)** — `conformal_prediction_band` still works for SupNorm/L2 and returns None/error for the new template-based elastic variants (no panic; non-exhaustive match handled).
7. **Module doctest (ECA-03)** — a running calibrate→flag doctest under `cargo test --doc`.
8. **Additive/non-breaking** — existing signatures unchanged; whole-crate test/clippy `--all-targets`/fmt green.

---

## Wave 0 Requirements

- [ ] New inline `#[cfg(test)] mod tests` in `tolerance/conformal_anomaly.rs` covering gates 1–7.
- [ ] Deterministic fixtures: a spanning clean generator (fixed seed) for marginal validity; explicit magnitude + shape outlier constructors.

*Existing test harness covers all phase requirements — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| (none) | — | — | — |

*All phase behaviors have automated verification (known-answer + statistical-band tests + doctest).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
