# Phase 110: pmat CI Guardrail - Context

**Gathered:** 2026-09-14
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure/CI phase — discuss skipped)

<domain>
## Phase Boundary

Add a GitHub Actions job that installs pmat 3.18.0 and runs the **curated**
quality-gate against `fdars-core`, failing the build if violations return —
mirroring the v0.43.0 `determinism-guardrail` pattern — so quality can't
silently regress. Depends on Phase 109 (the gate is green — verified 0
violations).

Out of scope: release/version bump (Phase 111).

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion (with these fixed constraints)
- **The gate command is the CURATED one** (per the operator decision in STATE.md
  Decisions / [[v0460-gate-definition-decision]]):
  `pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation`
  — NOT bare `pmat quality-gate` (which includes the stochastic `entropy`
  analyzer and would make CI flaky).
- **pmat is a dev/CI tool, NOT a crate dependency** — install it in CI via
  `cargo install pmat --version 3.18.0 --locked` (pin the version so the gate's
  thresholds match what the milestone validated: cognitive >23 / >25, cyc, etc.).
  Never add pmat to `Cargo.toml`.
- **Mirror the `determinism-guardrail` job** in `.github/workflows/rust-ci.yml`:
  a dedicated job, `runs-on: ubuntu-latest`, `working-directory: fdars-core`,
  checkout → install Rust stable → cache cargo (incl. `~/.cargo/bin/` so pmat is
  not recompiled every run) → install pmat → run the curated gate.
- **Gate the `publish` job on it** — add the new job to `publish.needs` (the
  determinism-guardrail is already there), so a quality regression blocks release.
- **Document local reproduction** — a comment in the job (and/or a note) giving
  the exact local command so a developer can reproduce a CI failure.

</decisions>

<code_context>
## Existing Code Insights

- CI lives at `.github/workflows/rust-ci.yml`. Jobs: `test` (matrix
  stable/beta/nightly), `determinism-guardrail`, `clippy`, `fmt`, `docs`,
  `wasm`, `coverage`, `publish` (needs: test, clippy, fmt, docs, wasm,
  determinism-guardrail; runs on release published).
- The `determinism-guardrail` job (lines ~60-120) is the exact structural
  template: checkout → `dtolnay/rust-toolchain@stable` → `actions/cache@v4`
  (cargo dirs incl. `~/.cargo/bin/`) → run steps.
- Triggers already cover `fdars-core/**` + `.github/workflows/rust-ci.yml` on
  push/PR to main + release published — the new job needs no trigger changes.
- pmat is installed locally as crate `pmat` 3.18.0 (crates.io), binary `pmat`.

</code_context>

<specifics>
## Specific Ideas

The job must FAIL (non-zero exit) when the curated gate finds any violation —
`--fail-on-violation` provides that. Cache the pmat binary so CI stays fast.
After this phase, a reintroduced complexity/SATD/defect/dead-code/duplicate
violation fails CI on the offending push/PR.

</specifics>

<deferred>
## Deferred Ideas

None — CI-config phase.

</deferred>
