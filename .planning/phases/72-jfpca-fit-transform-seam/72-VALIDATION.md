---
phase: "72"
slug: "jfpca-fit-transform-seam"
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-05"
---

# Phase 72 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)] mod tests`) + doctests |
| **Config file** | none — inline module tests in `fdars-core/src/` |
| **Quick run command** | `cargo test -p fdars-core --features linalg,parallel elastic_fpca` |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–120 seconds (full suite ~2795 tests) |

---

## Sampling Rate

- **After every task commit:** Run the quick command scoped to the new jfPCA fit/transform tests
- **After every plan wave:** Run the full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check`
- **Before `/gsd-verify-work`:** Full suite green, clippy clean, fmt clean, `cargo test --doc` green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 72-01-* | 01 | 1 | VEE-01 | — | N/A (numerical library) | unit | `cargo test -p fdars-core --features linalg,parallel jfpca_fit` | ❌ W0 | ⬜ pending |
| 72-01-* | 01 | 1 | VEE-02 | — | N/A | unit | `cargo test -p fdars-core --features linalg,parallel jfpca_transform` | ❌ W0 | ⬜ pending |
| 72-01-* | 01 | 1 | VEE-01/02 | — | N/A | doctest | `cargo test -p fdars-core --features linalg,parallel --doc elastic_fpca` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Validation Architecture (make-or-break known-answer gates)

1. **Fit reproduces `joint_fpca` scores within 1e-8** — `jfpca_fit(curves, argvals, ncomp, balance_c)` training scores equal `joint_fpca(...)` scores element-wise (assert max abs diff < 1e-8). Guards against the fit path diverging from the reference algorithm.
2. **fit→transform round-trip within tolerance** — `model.transform(&training_curves).scores` reproduces the training scores (tolerance driven by alignment determinism; document the achieved tolerance). This is the top correctness risk (out-of-sample alignment must match training alignment against the *trained* Karcher mean).
3. **Grid-mismatch error path** — `model.transform` on curves whose `argvals` length differs from the trained grid returns `FdarError::InvalidDimension` (no panic, no silent resample).
4. **Zero/degenerate guards** — dimension checks at entry mirror `joint_fpca` (n≥2, m≥2, ncomp≥1, argvals length == m).
5. **Module doctest** — a running `cargo test --doc` example exercising fit → transform end-to-end.

---

## Wave 0 Requirements

- [ ] New inline `#[cfg(test)] mod tests` cases in `fdars-core/src/elastic_fpca.rs` (or a new `jfpca_model` submodule) covering gates 1–5 above.
- [ ] Known-answer fixtures: a small spanning set of pseudo-random / multi-frequency curves (n≫m, full-rank design — per project memory, avoid low-rank phase-shifted single-freq sinusoids that silently pass).

*Existing test harness (built-in) covers all phase requirements — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| (none) | — | — | — |

*All phase behaviors have automated verification (numerical known-answer tests + doctest).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
