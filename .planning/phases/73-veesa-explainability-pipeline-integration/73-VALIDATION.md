---
phase: "73"
slug: "veesa-explainability-pipeline-integration"
status: draft
nyquist_compliant: false
wave_0_complete: false
created: "2026-09-05"
---

# Phase 73 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test harness (`#[cfg(test)] mod tests`) + doctests |
| **Config file** | none — inline module tests in `fdars-core/src/` |
| **Quick run command** | `cargo test -p fdars-core --lib --features linalg,parallel elastic_pfi` (+ `jfpca_model` for VEE-04) |
| **Full suite command** | `cargo test -p fdars-core --features linalg,parallel` |
| **Estimated runtime** | ~60–120 seconds (full suite ~2801+ tests) |

---

## Sampling Rate

- **After every task commit:** Run the quick command scoped to the new PFI / reconstruction tests
- **After every plan wave:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check`
- **Before `/gsd-verify-work`:** Full suite green, clippy clean, fmt clean, `cargo test --doc` green
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 73-01-* | 01 | 1 | VEE-03 | — | N/A | unit | `cargo test -p fdars-core --lib --features linalg,parallel elastic_pfi` | ❌ W0 | ⬜ pending |
| 73-01-* | 01 | 1 | VEE-04 | — | N/A | unit | `cargo test -p fdars-core --lib --features linalg,parallel principal_direction` | ❌ W0 | ⬜ pending |
| 73-01-* | 01 | 1 | VEE-05 | — | N/A | doctest | `cargo test -p fdars-core --features linalg,parallel --doc` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Validation Architecture (make-or-break behavioral gates)

1. **PFI seed-determinism** — `elastic_pfi(...)` with a fixed `seed` produces identical importance
   values across runs (byte-equal). Uses a single advancing `StdRng::seed_from_u64(seed)` (per
   RESEARCH finding — NOT per-component reseed, which is behavior-changing).
2. **PFI informative-PC ranking** — on a known-signal design where one PC carries the response,
   that PC's importance ranks strictly above the noise PCs. (Use a spanning, full-rank fixture,
   n≫m — never low-rank single-freq sinusoids that silently pass.)
3. **Principal-direction `c=0` reproduces the jfPCA mean** — `principal_directions(..., c_values=[0.0], ...)`
   amplitude curve == `karcher_mean` and phase == identity warping, within ~1e-8. (Guaranteed by
   `exp_map_sphere` returning `mean_psi` for tiny perturbation + `srsf_inverse` of the mean SRSF.)
4. **σⱼ correctness** — reconstruction uses `sqrt(eigenvalues[j])` (std-dev), NOT the raw eigenvalue
   (variance). Add a test with a nonzero c that asserts the amplitude perturbation magnitude scales
   with `sqrt(eigenvalue)` (the c=0 gate does NOT catch a raw-vs-sqrt bug — RESEARCH warning).
5. **Amplitude/phase split shapes** — `PrincipalDirections` returns amplitude_curves + phase_curves
   with correct dimensions (per c-value, per selected PC index).
6. **End-to-end pipeline doctest** — `veesa_pipeline` (fit → transform → PFI) runs under `cargo test --doc`.
7. **Additive/non-breaking** — existing signatures unchanged; new items re-exported at `lib.rs` +
   `prelude.rs`; whole-crate test/clippy `--all-targets`/fmt green.

---

## Wave 0 Requirements

- [ ] New inline `#[cfg(test)] mod tests` in `elastic_pfi.rs` (VEE-03/05) and in `jfpca_model.rs`
      (or the reconstruction module) for VEE-04, covering gates 1–6.
- [ ] Known-signal PFI fixture (spanning, full-rank, n≫m) with a designed informative PC.

*Existing test harness covers all phase requirements — no framework install needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| (none) | — | — | — |

*All phase behaviors have automated verification (determinism, ranking, known-answer c=0, σ-scaling, doctest).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
