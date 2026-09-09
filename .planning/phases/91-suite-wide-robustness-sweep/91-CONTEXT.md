# Phase 91: Suite-Wide Robustness Sweep - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Beyond the three golden tests fixed in Phase 90, audit the **rest of the test suite** for analogous fragility and either fix or document-as-safe every additional fragile test, so the whole suite is reliably green under full parallel runs across the CI-supported feature configurations.

Reuses the Phase 90 diagnosis technique. Depends on Phase 90 (the golden-flake fix must be in place first). Requirements: ROBUST-01 (audit the whole suite for other fragile bit-identity / nondeterministic / env-BLAS-disk-dependent assertions) and ROBUST-02 (fix or justify each).

Out of scope: the CI guardrail (Phase 92, CI-01) and release prep (Phase 93, REL-01/02).
</domain>

<decisions>
## Implementation Decisions

### Audit Scope (test layers)
- Sweep **integration tests + the ~2857 lib unit tests + doctests**, across all CI-supported feature configs. This is the whole-suite determinism goal, not just the `tests/` integration files.
- Concretely, the two CI-supported configs to exercise are: `--features linalg,parallel,serde` and `--no-default-features --features linalg` (mirrors `.github/workflows/rust-ci.yml`). Bare `cargo test` (default features, no linalg) is NOT a CI-supported config for the backend-dependent golden tests — Phase 90 established these require linalg and guarded them accordingly; the sweep may still run it to catch any OTHER unguarded backend-dependent test (a fast way to surface the class).

### Fragility Classes to Hunt (all three)
1. **Feature-backend-dependent goldens** — `assert_eq!` bit-identity on outputs that route through a feature-gated numeric backend (the confirmed Phase 90 class: faer vs nalgebra SVD in `fdata_to_pc`; also anything using `#[cfg(feature = "linalg")]`-gated linear algebra). Known partial guards already exist in `svd_equivalence.rs` and `validate_against_r.rs` — audit whether coverage is complete.
2. **RNG / thread-order nondeterminism** — tests asserting exact values on outputs of parallel/seeded code; confirm the `seed_for_thread(seed, k)` per-thread reseeding actually makes them thread-count-independent (bit-identity should hold, but verify under repeated runs and thread-count variation).
3. **Environment / BLAS-thread / disk dependence** — assertions that could vary with `RAYON_NUM_THREADS`, BLAS/faer runtime SIMD/thread detection, or disk/timing. Note Phase 90 OVERTURNED the disk-pressure hypothesis for the golden flake — so treat env/disk as a hypothesis to *disprove with evidence*, not assume.

### Fix-vs-Document Policy (mirror Phase 90)
- **cfg-guard** backend-dependent tests (`#[cfg_attr(not(feature = "linalg"), ignore)]` or the appropriate feature gate) so they run only under the backend their goldens were captured with.
- **Relax `assert_eq!` to a tight relative tolerance ONLY where FP nondeterminism is PROVEN** for that assertion; keep bit-identity everywhere it still holds; document the evidence per relaxation.
- **Document genuinely-robust tests as safe-to-leave** with rationale in the audit artifact (no code change) rather than over-guarding sound tests.
- **No new crate dependency.** If serialization were ever needed, a std-only guard — but Phase 90 found no race, so this is unlikely.

### Acceptance Bar
- Full suite **green under BOTH CI-supported configs** (`--features linalg,parallel,serde` AND `--no-default-features --features linalg`), **plus a repeated-run check** (a handful of consecutive full runs, not a single run, to catch intermittency).
- Every fragile test found is either fixed (guard/tolerance) or documented as safe-to-leave with rationale, captured in a written audit artifact (e.g. `91-AUDIT.md`).

### Claude's Discretion
- Exact number of repeated full runs in the acceptance check (a "handful" — e.g. 3–5 per config), thread-count sweep specifics, and the audit-artifact format are at the executor's discretion, guided by the acceptance bar.
- Whether to mechanically grep for the structural pattern (`assert_eq!` + backend/RNG-dependent call) first, then confirm by running the configs, or vice versa.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase 90 technique & artifacts** — `90-DIAGNOSIS.md` establishes the backend-divergence class and the reproduction method (run under a config that omits `linalg` → nalgebra path diverges). Reuse this lens.
- **Integration test inventory** (`fdars-core/tests/`): `equivalence_phase47.rs`, `equivalence_phase48.rs` (guarded in P90), `equivalence_phase49.rs` (guarded in P90), `equivalence_phase50.rs`, `svd_equivalence.rs` (already has linalg guards), `validate_against_r.rs` (already has linalg guards), `validate_phase_bands.rs`, `validate_spm_math.rs`, `validate_new_modules.rs`, `alloc_audit_dpca.rs`, `alloc_audit_fpca.rs`, `serde_feature_roundtrip.rs`, and the 5 `integration_explain_*.rs` files.
- Tests calling SVD-backed fns (`fdata_to_pc`, `co_cluster`, `pace_fpca`, `fpca`) are the prime backend-dependent candidates: `validate_phase_bands.rs`, `validate_new_modules.rs`, `alloc_audit_fpca.rs`, `alloc_audit_dpca.rs`, `equivalence_phase47.rs`, `equivalence_phase50.rs`, and the `integration_explain_*` set.

### Established Patterns
- The Phase 90 guard idiom: `#[cfg_attr(not(feature = "linalg"), ignore)]` above `#[test]`, with a doc note explaining the backend dependence.
- Feature-gated parallelism via the 5 `*_maybe_parallel!` macros; per-thread RNG reseeding `seed_for_thread(seed, k)` = `StdRng::seed_from_u64(seed.wrapping_add(k as u64))` (documented thread-count-independence contract to verify).
- CI configs of record (`.github/workflows/rust-ci.yml`): `cargo test --features linalg,parallel,serde` and `cargo test --no-default-features --features linalg`.

### Integration Points
- The whole-suite green result under both configs is what Phase 92 (CI guardrail) will lock and Phase 93 (release gate REL-02) will treat as the determinism proof. Any guard/tolerance added here must keep BOTH CI configs green.

### Build/Disk Hazards (MEMORY.md)
- Bash tool default timeout is 2 min — run long full-suite gates with an explicit 600s tool timeout, batched. `target/debug/{incremental,examples}` fills `/home` (free before long runs). Doctests link in a small `/tmp` tmpfs. Keep `--features serde` build green (do not regress). Pre-commit hook runs the full gate and times out → run gates foreground/out-of-band, then `git commit --no-verify`. Run `cargo fmt` per commit. Execute inline (fdars executor subagents stall on long cargo).

</code_context>

<specifics>
## Specific Ideas

- Fast structural discovery: `grep` for `assert_eq!` on floats combined with calls into `fdata_to_pc` / `co_cluster` / `pace_fpca` / `fpca` / SVD, and for any `#[cfg(feature = "linalg")]`-gated numeric assertion lacking a run-guard, then confirm by running the full suite under the no-linalg config to see what actually fails.
- Verify the already-guarded files (`svd_equivalence.rs`, `validate_against_r.rs`) guard the RIGHT tests and none slipped through.
- Treat the disk-pressure/env hypothesis as something to disprove with evidence (Phase 90 overturned it for the goldens) — do not add serialization or env-pinning without a proven race.

</specifics>

<deferred>
## Deferred Ideas

- CI guardrail exercising the full parallel path / nextest serialization group → Phase 92 (CI-01).
- Version bump, CHANGELOG, 1.0-checklist Quality tick, release gates → Phase 93 (REL-01/02).
</deferred>
