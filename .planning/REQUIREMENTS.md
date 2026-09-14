# Requirements: fdars — v0.46.0 Quality Hardening (pmat Complexity Pass & CI Guardrail)

**Defined:** 2026-09-13
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps on the road to a deliberate 1.0 cut.

## Milestone Context

An extra **quality cycle** inserted before the remaining 1.0-checklist items (`fdars-j75`, `1.0-CUT`). The codebase grew fast and reuse-first across v0.14.0–v0.45.0; a **pmat 3.18.0** audit (2026-09-13) surfaced concentrated technical debt:

- **Complexity: 120 quality-gate violations** — median cognitive **39**, max cyclomatic **59** (`funfem_cluster`), est. 449h refactoring. The dominant debt category.
- **Entropy: 11 violations** — repeated `DataTransformation` patterns (test helpers + a few src sites); low pattern diversity (77%).
- **SATD: 2 violations** — both doc-comment false positives (keywords "Defect:"/"Performance:" in `///` text): `depth/dispatch.rs:281`, `metric/lp.rs:175`.
- **Defects: 1 critical** — a single `.unwrap()` at `depth/rpd.rs:229`.
- **Clean:** dead code 0, duplicates 0.0%.

This milestone drives `pmat quality-gate` to **PASS** (0 violations), fixes the unwrap, and wires a pmat CI guardrail so debt can't silently regress.

**Governing contract:** strictly **non-breaking + numeric-preserving**. All 2942 lib tests + 215 doctests + 28 examples stay green; refactoring is pure internal decomposition (extract helpers), never a behavior or numeric change — the established crate norm (private impl bodies stay behavior-identical). `fdars-core` only (protects R + WASM bindings). **No new *crate* dependency** — pmat is a dev/CI tool, not a `Cargo.toml` dependency.

## v0.46.0 Requirements

### Complexity Refactoring

- [x] **QUAL-01**: The worst top-level complexity hotspots — `funfem_cluster` (cyc 59), `elastic_net_cd` (cog 54), `wnet_cv_lambda` (cog 45), `wnet` (cog 31), `boost_fofr` (cyc 39), `pace_fpca` (cyc 37), `bayesian_fosr` (cyc 36), `kcfc_cluster` (cyc 28) — are refactored into smaller helpers, each under pmat's cognitive/cyclomatic thresholds, with behavior and numeric output identical (verified by the existing test suite).
- [x] **QUAL-02**: The SPM complexity cluster — `mfpca`, `spm_phase1_iterative`, `spm_mewma_monitor`, `select_ncomp`, `conditional_expectation` (partial), `spm_monitor_from_fields` (phase), `profile_phase1`, `spe_multivariate` (stats) — is refactored under pmat thresholds, behavior/numeric-identical.
- [ ] **QUAL-03**: All remaining complexity violations across `fdars-core` are resolved so `pmat analyze complexity` / `pmat quality-gate` reports **0 complexity findings**.

### Entropy & Technical Debt

- [ ] **QUAL-04**: The repeated `DataTransformation` patterns flagged by `pmat analyze entropy` (11 violations across `metric/tests.rs`, `fem_smoothing.rs`, `explain_generic/friedman.rs`, `seasonal/tests.rs`, `regression.rs`, `covariance.rs`, `detrend/tests.rs`, `scalar_on_function/tests.rs`, and the project-level pattern-diversity finding) are extracted into shared helpers so pmat entropy reports **0 actionable violations**.
- [ ] **QUAL-05**: The 2 SATD doc-comment false-positives (`depth/dispatch.rs:281`, `metric/lp.rs:175`) are reworded so keyword matchers no longer flag them, AND the 1 critical `.unwrap()` at `depth/rpd.rs:229` is replaced with proper error handling (`?` or `.expect()` with a descriptive message) — `pmat analyze satd` and `pmat analyze defects` both report **0 violations**.

### CI Guardrail

- [ ] **CI-01**: A pmat quality-gate CI job is added (GitHub Actions) that installs pmat and runs `pmat quality-gate` (and/or a complexity-threshold check) against `fdars-core`, failing the build if pmat violations return — mirroring the v0.43.0 `determinism-guardrail` pattern. Documented so contributors can reproduce it locally.

### Release

- [ ] **REL-01**: Crate bumped 0.45.0 → 0.46.0 with `[0.46.0]` in both CHANGELOGs (root + crate). All whole-crate gates green — `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, `--features serde` build guard, all 28 examples + doctests, `cargo package` — **plus** `pmat quality-gate` passing. The `git tag v0.46.0` → crates.io publish is the DEFERRED operator step.

## Future Requirements

Deferred to future milestones — tracked but not in this roadmap.

### 1.0 Checklist (unchanged by this milestone)

- **fdars-j75**: Migrate the external `fdars-r` wrapper to the current `FdMatrix` API (separate package, ecosystem gap).
- **1.0-CUT**: Bump to `1.0.0` and declare the API stable (terminal item, done last).

### pmat Composite Score

- **PVLINT-01**: Address the pmat `PV Lint` (provability-lint) sub-score (0.0) and chase a specific composite pmat grade. Deferred — this milestone targets the falsifiable `quality-gate` PASS, not the opinionated composite score.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Any behavioral or numeric change | Milestone contract is strictly behavior-preserving; refactoring is internal decomposition only |
| New algorithms / capabilities | This is a quality/refactor cycle, not a feature milestone |
| `fdars-r` / WASM binding changes | `fdars-core` only; refactors must be internal and not alter the public surface bindings depend on |
| New `Cargo.toml` crate dependency | pmat is a dev/CI tool invoked externally, never added as a crate dependency (standing project convention) |
| Chasing the pmat composite score / PV Lint to a target grade | The verifiable target is `pmat quality-gate` PASS; composite-score tuning deferred (PVLINT-01) |
| The 1.0-checklist items (`fdars-j75`, `1.0-CUT`) | This is an extra quality cycle inserted *before* them; they remain future work |
| Public API changes to reduce complexity | Complexity must be reduced by internal helper extraction, not by changing signatures (protects callers/bindings/examples) |

## Traceability

Which phases cover which requirements.

| Requirement | Phase | Status |
|-------------|-------|--------|
| QUAL-01 | Phase 106 | Complete |
| QUAL-02 | Phase 107 | Complete |
| QUAL-04 | Phase 108 | Pending |
| QUAL-05 | Phase 108 | Pending |
| QUAL-03 | Phase 109 | Pending |
| CI-01 | Phase 110 | Pending |
| REL-01 | Phase 111 | Pending |

**Coverage:**

- v0.46.0 requirements: 7 total
- Mapped to phases: 7 (100%) ✓
- Unmapped: 0 ✓
- No orphans, no duplicates (QUAL-04 and QUAL-05 share Phase 108 — independent cleanup work grouped by fine-granularity boundary)

---
*Requirements defined: 2026-09-13*
*Last updated: 2026-09-13 after roadmap creation (traceability filled)*
