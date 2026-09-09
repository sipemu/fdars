# Phase 90: Golden-Flake Root-Cause & Deterministic Fix - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Root-cause and deterministically fix the three known flaky golden tests, backed by evidence:
- `golden_co_cluster_parallel` (`fdars-core/tests/equivalence_phase48.rs`) — n_init=4, PARALLEL branch, bit-identical `log_likelihood` + `row_labels` + `col_labels`.
- `golden_co_cluster_below_threshold` (`equivalence_phase48.rs`) — n_init=2, SEQUENTIAL branch, same assertions. **Key clue: this uses the sequential path yet still flakes**, so the cause is not purely the parallel n_init reduce ordering.
- `svd_sign_fpca_two_matrix_bit_identical` (`equivalence_phase49.rs`) — `fdata_to_pc` rotation + scores bit-identical (`assert_eq!` on exact `f64`).

Symptom (from MEMORY.md / STATE.md): all three pass reliably per-binary in isolation but fail intermittently ONLY under full parallel `cargo test`. Treated as an environment / cross-binary-interference flake (disk pressure, thread-count contention), NOT a numeric regression from any single commit.

Deliverable: an evidence-based diagnosis of WHY they flake under full parallel runs (FLAKE-01) plus a deterministic fix (FLAKE-02) that makes them pass reliably under repeated full parallel `cargo test`. FLAKE-01 gates FLAKE-02 — the fix design is chosen from the evidence.

Out of scope: the broader suite-wide fragility sweep (Phase 91), the CI guardrail (Phase 92), and release prep (Phase 93).

</domain>

<decisions>
## Implementation Decisions

### Diagnosis & Fix Strategy
- **Fix-approach bias = least-invasive, diagnosis-gated.** Investigate first; prefer test-side robustness. Relax bit-identity to a tight relative tolerance ONLY where FP nondeterminism is proven; serialize the tests ONLY if a genuine shared-state race is found. Force-determinism-in-`src/` is the last resort (highest blast radius — protects R + WASM bindings + 28 examples). Minimal `src/` determinism changes are permitted only if the root cause is proven to live in library code.
- **No new crate dependency.** If serialization turns out to be necessary, use a std-only test guard (e.g. a `static` Mutex / lock acquired at the top of the affected tests). Do NOT add `serial_test`; do NOT introduce a nextest-groups config in this phase.
- **Bit-identity may relax to tight tolerance where nondeterminism is proven.** If FP nondeterminism is demonstrated as the root cause for a specific assertion, that assertion may move from `assert_eq!` (bit-identical) to a tight relative tolerance (target ~1e-12); keep bit-identity for every assertion where determinism still holds. Document the evidence justifying each relaxation.
- **Acceptance bar = 10 consecutive green full parallel `cargo test` runs AND per-binary green.** This is the proof-of-fix gate for the phase. Verify BOTH the full-parallel path (where the flake appears) and per-binary isolation (which already passes).

### Claude's Discretion
- Exact reproduction harness (iteration count during diagnosis, `RAYON_NUM_THREADS` sweep, disk-pressure simulation, capturing actual-vs-expected on failure) is at the executor's discretion, guided by the acceptance bar above.
- Whether the root cause is one shared mechanism across all three tests or distinct per test is an open question for the diagnosis to answer.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Flaky tests live in `fdars-core/tests/equivalence_phase48.rs` (lines ~50–80: co_cluster goldens) and `equivalence_phase49.rs` (line ~381: svd_sign golden).
- Determinism contract helper `helpers::seed_for_thread(seed, k)` = `StdRng::seed_from_u64(seed.wrapping_add(k as u64))` — the documented per-thread RNG seeding pattern (equivalence_phase49.rs notes it).
- `co_cluster` (src/coclustering.rs) uses "parallel n_init map + SEQUENTIAL strict-`>` reduce (lowest-init-index tie-break)" — the goldens were captured pre-parallel and asserted bit-identical after the rayon swap.
- FPCA sign core: `fdata_to_pc` (src/regression.rs), sign-decision fixes column signs; SVD routes through `nalgebra::SVD`.

### Established Patterns
- Parallelism is feature-gated via the 5 `*_maybe_parallel!` macros in `src/parallel.rs`; the `parallel` feature is on by default. Golden refs must hold under BOTH feature configs.
- Column-major `FdMatrix`; integration weights via `helpers::simpsons_weights`.
- Tests use the built-in harness (`#[test]`), run multi-binary under `cargo test`.

### Integration Points
- Full-suite `cargo test` is the determinism proof for the whole milestone (REL-02, Phase 93). Any fix here must keep the full-parallel path green — that is the signal Phase 92's CI guardrail and Phase 93's release gate depend on.

### Build/Disk Hazards (MEMORY.md)
- `target/` fills `/home`; free `target/debug/{incremental,examples}` before long full runs. Doctests link in a small `/tmp` tmpfs. Pre-commit hook runs the full cargo gate and times out → prefer inline execution + `commit --no-verify` after out-of-band gates. Run clippy `--all-targets --features linalg,parallel -- -D warnings`. Run `cargo fmt` per commit to avoid CI fmt drift.

</code_context>

<specifics>
## Specific Ideas

- The sequential-branch test (`golden_co_cluster_below_threshold`) flaking is the strongest diagnostic lead: it rules out the parallel n_init reduce as the sole cause and points at something shared under multi-binary load — candidate hypotheses to test: rayon global-pool thread-count sensitivity in a shared code path (integration/FPCA via `iter_maybe_parallel!`), BLAS/faer runtime thread or SIMD detection under CPU contention, or an environment/disk-pressure-driven timing effect.
- Capture actual-vs-expected values on failure (which assertion, how far off) to distinguish a genuine FP-ordering drift (→ tolerance relaxation candidate) from a categorical label flip (→ likely a real determinism bug to fix at source).

</specifics>

<deferred>
## Deferred Ideas

- Suite-wide audit for other analogous fragile assertions → Phase 91 (ROBUST-01/02).
- CI guardrail exercising the full parallel path / nextest serialization group → Phase 92 (CI-01).
- Version bump, CHANGELOG, 1.0-checklist Quality tick, release gates → Phase 93 (REL-01/02).

</deferred>
