# Phase 92: CI Determinism Guardrail - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Add a CI guardrail that exercises the full parallel `cargo test` path (cross-binary interference) so a future determinism regression fails CI loudly instead of silently returning. Requirement CI-01. Depends on Phase 91 (lands after the fixes so the gate reflects a green baseline). Scope is a `.github/workflows/` change only.

Out of scope: release prep / version bump / CHANGELOG / 1.0-checklist tick (Phase 93, REL-01/02).
</domain>

<decisions>
## Implementation Decisions

### Guardrail Mechanism
- **Std-only: repeat + thread-count variation.** A dedicated CI job runs the full parallel `cargo test --features linalg,parallel,serde` a few times consecutively (catch intermittency) AND once with `RAYON_NUM_THREADS=1` (catch thread-order / parallel-reduction sensitivity — if a golden ever depends on parallel reduction order, forcing single-threaded reduction would drift it). No nextest, no serialization group.

### New Tooling / Dependency
- **None.** Pure shell loop in the workflow YAML — consistent with the Phase 90 decision (no `serial_test`, no nextest) and the milestone's no-new-dependency constraint. cargo-nextest is explicitly NOT added.

### Defend the Specific Phase-90 Failure Mode (anti-silent-skip)
- **Assert the three golden tests actually RAN** (not silently ignored) under `--features linalg`. The Phase 90 fix makes them `#[cfg_attr(not(feature = "linalg"), ignore)]`; a regression where the cfg-guard/feature wiring makes them NEVER run (always ignored) would otherwise pass CI silently. The guardrail must assert a positive run: e.g. `cargo test -p fdars-core --features linalg,parallel --test equivalence_phase48 -- golden_co_cluster` reports `2 passed` (not `0 passed … 2 ignored`), and `--test equivalence_phase49 -- svd_sign_fpca_two_matrix_bit_identical` reports `1 passed`.

### Placement
- **Dedicated CI job** named `determinism-guardrail` in `.github/workflows/rust-ci.yml` — isolated, clearly named, obvious pass/fail. Mirror the existing `test` job's scaffolding (`working-directory: fdars-core`, `actions/checkout@v4`, `dtolnay/rust-toolchain@stable`, the `actions/cache@v4` cargo cache block). Trigger on the same push/PR events as the other jobs.

### Claude's Discretion
- Exact repeat count in CI (a "few" — e.g. 3, balancing signal vs CI minutes), the precise grep/exit-code assertions, and whether the thread-count-1 run is a separate step or folded into the loop are at the executor's discretion, guided by the success criteria.
- Whether to exclude `dhat-heap` (yes — same reason as the existing test job: `#[global_allocator]` panics under parallel dhat tests) and `js` (WASM-only).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets / Patterns (from `.github/workflows/rust-ci.yml`, 180 lines)
- Jobs: `test` (matrix stable/beta/nightly), `clippy`, `fmt`, `docs`, `wasm`, `coverage`, `publish`. All use `defaults.run.working-directory: fdars-core`.
- The `test` job already runs `cargo test --features linalg,parallel,serde` (main) and `cargo test --no-default-features --features linalg` (sequential). The guardrail ADDS a repeated/thread-varied run — it does not replace these.
- Job scaffolding to copy for the new job: `runs-on: ubuntu-latest`; `actions/checkout@v4`; `dtolnay/rust-toolchain@stable`; the `actions/cache@v4` block caching `~/.cargo/...` + `fdars-core/target/` keyed on `${{ runner.os }}-cargo-<rust>-${{ hashFiles('**/Cargo.lock') }}`.
- `on:` triggers: push (some branches), pull_request, release.
- dhat-heap is deliberately excluded from parallel test runs (a `#[global_allocator]` in `tests/alloc_audit_fpca.rs` panics under parallel dhat tests); `js` is WASM-only.

### Integration Points
- The guardrail passes against the now-green baseline from Phases 90–91 (a guardrail added before the fix would start red — hence this is the third phase).
- Phase 93 (release gate REL-02) will treat all CI gates green as part of release readiness.

### Build/CI Hazards (MEMORY.md)
- Clippy in CI uses `--all-targets --features linalg,parallel,serde -- -D warnings` with several explicit `-A` allows (do not regress). This is a workflow-only change — validate the YAML is well-formed. There is no local `act` runner assumed; the executor should validate YAML syntax and (optionally) dry-reason the shell logic, and confirm the guardrail commands run green locally as a proxy for CI.

</code_context>

<specifics>
## Specific Ideas

- The guardrail's regression target is precise (Success Criterion 3): "a reintroduced determinism regression would fail this CI gate." Two concrete regression modes to catch: (a) real nondeterminism under repeated/thread-varied runs → the loop + RAYON_NUM_THREADS=1 catches it; (b) the golden tests silently never running (guard/feature regression) → the positive-run assertion catches it.
- Locally validate the exact shell the CI job will run (the repeat loop, the RAYON_NUM_THREADS=1 run, and the "assert goldens ran" greps) so CI won't start red. Use the Phase 90/91 proven invocations.
- Keep the job fast enough to be practical: a small repeat count (e.g. 3) + one single-threaded run is sufficient signal; document the count choice.

</specifics>

<deferred>
## Deferred Ideas

- Version bump 0.42.0 → 0.43.0, CHANGELOG `[0.43.0]`, docs, 1.0-checklist Quality tick, full release-gate verification → Phase 93 (REL-01/02).
- The `git tag v0.43.0` push → crates.io publish is the deferred OPERATOR step (not done in any phase).

</deferred>
