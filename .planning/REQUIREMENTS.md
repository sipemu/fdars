# Requirements: fdars — v0.40.0 Correctness & Release Hardening

**Defined:** 2026-09-06
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against reference ecosystems — this milestone is a correctness & release-hardening pass: fix the bugs and build breakage found during recent milestones, formally validate the outstanding v0.39.0 phases, then bump/tag/publish fdars' first crates.io release since v0.38.0 (folding in the unpublished v0.39.0 AD core).

## v1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Correctness

- [x] **CORR-01**: `soft_dtw_backward` no longer overwrites the `E[n][m]=1.0` endpoint seed, so it returns a non-zero soft-alignment matrix and `soft_dtw_barycenter` genuinely refines the barycenter. A regression test asserts (a) `soft_dtw_backward`/`soft_dtw_accumulate_gradient` produce a non-zero gradient on non-identical input, and (b) `soft_dtw_barycenter` on non-identical curves converges to a barycenter measurably different from the pointwise mean — cross-checked against the v0.39.0 `Dual` gradient path / `corrected_oracle_gradient` reference (all within tolerance). The existing `test_soft_dtw_barycenter_*` tests are tightened so they can no longer pass on an all-zero gradient.
- [x] **CORR-02**: Every hand-written backward/gradient pass in the crate is audited for boundary-seed and analogous correctness bugs — at minimum `alignment/differentiable`, `autodiff`, `boosting_regression/gamlss`, `elastic_regression/logistic`, `explain_generic/counterfactual`, `regression`, `seasonal/mod`, `smooth_basis`, and `metric/soft_dtw`. Each pass is either confirmed correct (with a one-line rationale recorded in the phase artifact) or fixed with an accompanying regression test. The audit result is written up so the "clean vs fixed" disposition of every pass is traceable.

### Build & Features

- [x] **BUILD-01**: `cargo build --features serde` compiles cleanly again (broken since Phase 60). `ClassifFit` and any types it embeds that currently lack serde support gain conditional `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` derives, consistent with the crate's existing serde convention. A guard — a `--features serde` build/round-trip check runnable in CI — prevents silent re-breakage.

### Release & Validation

- [x] **REL-01**: Phases 75/76/77 `VALIDATION.md` are moved from `status: draft` to `validated` (Nyquist sign-off) via the validate-phase flow, reflecting the green test suite; any genuine coverage gaps surfaced during sign-off are filled or explicitly recorded.
- [x] **REL-02**: The crate is release-ready — `fdars-core` version bumped 0.38.0 → 0.40.0, `CHANGELOG.md` updated with the v0.39.0 (AD core) and v0.40.0 (this milestone) entries, README/`documentation/` refreshed where they reference version or the fixed behavior, and the whole-crate gates (`cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`) pass. Tag `v0.40.0` → crates.io publish via `release.yml` is the final operator-driven step (documented in the phase SUMMARY).

## Future Requirements

Deferred — tracked but not in this milestone's roadmap.

### Differentiable core (from v0.39.0 deferrals)

- **DIF-F1**: Reverse-mode / VJP autodiff.
- **DIF-F2**: Broaden the differentiable subset beyond elastic distance + FPCA scores (basis eval, inner products, …).
- **DIF-F3**: Make existing f64 signatures generic over `Scalar`.

## Out of Scope

Explicitly excluded from v0.40.0.

| Feature | Reason |
|---------|--------|
| New FDA algorithms / capability parity work | All parity/gap backlogs (scikit-fda, R core, multi-ecosystem) are exhausted; this milestone is correctness/hardening only |
| Reverse-mode AD (DIF-F1) and broadening the differentiable subset (DIF-F2/F3) | Deferred v0.39.0 items; not correctness/release work |
| Fixing the underlying soft-DTW barycenter *algorithm* beyond the endpoint-seed bug | Scope is the acknowledged correctness defect + a validating test, not a redesign of the optimizer |
| Breaking API changes / removing deprecated forms | Additive/non-breaking convention preserved (protects R + WASM bindings + 28 examples) |
| New crate dependency | Carried convention — fixes reuse existing machinery |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORR-01 | Phase 78 | Complete |
| CORR-02 | Phase 78 | Complete |
| BUILD-01 | Phase 79 | Complete |
| REL-01 | Phase 80 | Complete |
| REL-02 | Phase 80 | Complete |

**Coverage:**

- v1 requirements: 5 total
- Mapped to phases: 5 ✓
- Unmapped: 0

---
*Requirements defined: 2026-09-06*
*Last updated: 2026-09-06 after roadmap creation (traceability mapped)*
