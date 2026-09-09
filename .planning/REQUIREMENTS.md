# Requirements: fdars — v0.43.0 Test Determinism & Release Hardening

**Defined:** 2026-09-09
**Core Value:** A comprehensive, fast Rust functional-data-analysis library — this milestone clears the **Quality** blocker on `documentation/ROADMAP-TO-1.0.md` by making `cargo test` reliably deterministic under full parallel runs, hardening CI, and shipping the crate to crates.io.

## v0.43.0 Requirements

Requirements for this milestone. Each maps to a roadmap phase.

### Flake (golden-test determinism)

- [x] **FLAKE-01**: Root-cause *why* `golden_co_cluster_parallel` / `golden_co_cluster_below_threshold` (`equivalence_phase48`) and `svd_sign_fpca_two_matrix_bit_identical` (`equivalence_phase49`) pass per-binary but flake under full parallel `cargo test` — produce an evidence-backed diagnosis (BLAS threading / SVD sign / test ordering / disk pressure) recorded as an artifact.
- [x] **FLAKE-02**: Deterministically fix the three affected tests (tolerance-comparison vs serialization, chosen from FLAKE-01's diagnosis) so they pass reliably under repeated full parallel `cargo test` runs.

### Robustness (suite-wide sweep)

- [x] **ROBUST-01**: Audit the whole test suite for other fragile bit-identity, nondeterministic, or environment/BLAS/disk-dependent assertions; produce a findings list.
- [x] **ROBUST-02**: Fix each additional fragile test found (or document why it is safe to leave) so the suite is reliably green under full parallel runs.

### CI (regression guardrail)

- [x] **CI-01**: Add a CI guardrail that exercises the full parallel `cargo test` path (cross-binary interference) and/or a nextest serialization group, so a determinism regression fails CI rather than silently returning.

### Release

- [ ] **REL-01**: Bump crate 0.42.0 → 0.43.0, write CHANGELOG `[0.43.0]`, refresh docs, and check off the **Quality** item in `documentation/ROADMAP-TO-1.0.md`.
- [ ] **REL-02**: Verify release-readiness — whole-crate gates green (`cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, `--features serde` build, all 28 examples + doctests, `cargo package`); prepare the operator-driven `git tag v0.43.0` → crates.io publish (0.43.0 supersets the unpublished 0.41.0/0.42.0 — registry is still at 0.40.0).

## Future Requirements

Deferred to a later milestone (remaining `ROADMAP-TO-1.0.md` gaps). Tracked, not in this roadmap.

### Algorithm

- **SDTW-O1**: Replace the `soft_dtw_barycenter` MM-step descent with a proper global optimizer (L-BFGS / multi-restart).

### Differentiable core

- **DIF-F1**: Reverse-mode / VJP differentiation.
- **DIF-F2**: Broaden the differentiable subset beyond elastic + FPCA.
- **DIF-F3**: Generic `f64` hot-path signatures so autodiff types flow through.

### Ecosystem

- **fdars-j75**: Migrate the external `fdars-r` wrapper to the `FdMatrix` API (separate package).

### Release (terminal)

- **1.0-CUT**: Bump to `1.0.0` and declare the public API stable once every `ROADMAP-TO-1.0.md` item clears.

## Out of Scope

Explicitly excluded this milestone. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New FDA algorithms / capabilities | This is a quality + release-hardening milestone, not a feature milestone |
| SDTW-O1 optimizer replacement | Algorithm-quality item; separate `ROADMAP-TO-1.0.md` gap, not a test-determinism concern |
| Differentiable-core expansion (DIF-F1/F2/F3) | Differentiable-core milestone; unrelated to test determinism |
| `fdars-r` FdMatrix migration (fdars-j75) | Separate external package, out of `fdars-core` scope |
| The terminal 1.0-CUT | Blocked until algorithm + diff-core + ecosystem items also clear; deliberately deferred |
| Breaking API changes | Milestone is additive/non-breaking — protects R + WASM bindings + 28 examples |

## Traceability

Which phases cover which requirements. Filled during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FLAKE-01 | Phase 90 | Complete |
| FLAKE-02 | Phase 90 | Complete |
| ROBUST-01 | Phase 91 | Complete |
| ROBUST-02 | Phase 91 | Complete |
| CI-01 | Phase 92 | Complete |
| REL-01 | Phase 93 | Pending |
| REL-02 | Phase 93 | Pending |

**Coverage:**

- v0.43.0 requirements: 7 total
- Mapped to phases: 7 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-09*
*Last updated: 2026-09-09 after roadmap creation (traceability mapped)*
