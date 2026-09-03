---
phase: 65-greedy-selection-integration
plan: 02
type: execute
wave: 2
depends_on: ["65-01"]
files_modified:
  - fdars-core/src/optimal_design.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
  - fdars-core/Cargo.toml
  - fdars-core/benches/optimal_design.rs
autonomous: true
requirements: [FOD-05]
estimate:
  tokens: 55000
  raw_tokens: 37000
  tasks: 3
  confidence: high
must_haves:
  truths:
    - "fdars_core::optimal_design function is reachable at the crate root (FOD-05)"
    - "use fdars_core::prelude::*; OptDesConfig::default() compiles — prelude re-export reachable (FOD-05)"
    - "Module-level doctest (fit PACE -> optimal_design -> read selected_argvals) compiles and passes (FOD-05)"
    - "benches/optimal_design.rs compiles under cargo build --benches (FOD-05)"
    - "[[bench]] name = \"optimal_design\" registered in Cargo.toml (FOD-05)"
    - "Full lib + doctest suite passes; 28 examples + WASM + R unaffected; no existing signature changed (FOD-05)"
  artifacts:
    - "fdars-core/src/lib.rs (extended re-export line: optimal_design, OptDesConfig, OptDesResult added)"
    - "fdars-core/src/prelude.rs (FOptDes full-surface re-export block added)"
    - "fdars-core/src/optimal_design.rs (module-level //! doctest added)"
    - "fdars-core/benches/optimal_design.rs (new criterion 0.5 benchmark)"
    - "fdars-core/Cargo.toml ([[bench]] name = optimal_design stanza)"
  key_links:
    - "lib.rs re-export line extends the existing Phase-64 pub use optimal_design::{...} additively"
    - "doctest exercises the real pace_fpca fit path (PaceFpcaResult is #[non_exhaustive] — no struct-literal from an external doctest)"
    - "[[bench]] stanza registration is required or the bench file is never compiled by cargo build --benches"
---

<objective>
Finalize the FOptDes public surface: extend the additive crate-root re-export in `lib.rs`
and `prelude.rs` to the full six-symbol surface, add the module-level end-to-end doctest
to `optimal_design.rs`, create the criterion 0.5 benchmark `benches/optimal_design.rs`,
and register its `[[bench]]` stanza in `Cargo.toml`. Then run the whole-crate gates.

Purpose: Complete FOD-05 — the two-stage workflow is reachable from the crate root and
prelude, documented by a runnable doctest, and benchmark-covered — additively and
non-breakingly (28 examples + WASM + R bindings unaffected).
Output: Extended `lib.rs`/`prelude.rs`/`optimal_design.rs`/`Cargo.toml` + one new bench file,
all whole-crate gates green (fmt, clippy `--all-targets`, full test + doctest suite, bench compile).
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/65-greedy-selection-integration/65-CONTEXT.md
@.planning/phases/65-greedy-selection-integration/65-RESEARCH.md

# Re-export sites and existing Phase-64 line to extend
@fdars-core/src/lib.rs
@fdars-core/src/prelude.rs
# Benchmark template (criterion 0.5) + [[bench]] stanza precedent
@fdars-core/benches/kshape.rs
@fdars-core/Cargo.toml
# pace_fpca fit path for the doctest end-to-end example (IrregFdata -> pace_fpca -> PaceFpcaResult)
@fdars-core/src/pace_fpca.rs
</context>

<artifacts_this_phase_produces>
- `fdars-core/src/lib.rs`: the existing Phase-64 line
  `pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};`
  becomes `pub use optimal_design::{design_criterion, optimal_design, DesignCriterion, OptDesConfig, OptDesResult, OptimalityKind};`
- `fdars-core/src/prelude.rs`: NEW block re-exporting the same six symbols via `crate::optimal_design::{...}`.
- `fdars-core/src/optimal_design.rs`: NEW module-level `//!` doctest (end-to-end).
- `fdars-core/benches/optimal_design.rs`: NEW criterion 0.5 bench (design_criterion + optimal_design, Trajectory + Score(A)).
- `fdars-core/Cargo.toml`: NEW `[[bench]] name = "optimal_design"  harness = false` stanza after the `kshape` bench.
</artifacts_this_phase_produces>

<tasks>

<task type="auto">
  <name>Task 1: Additive crate-root + prelude re-exports (full FOptDes surface)</name>
  <files>fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - `fdars-core/src/lib.rs:592-593` — existing `pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};` (Phase 64) to EXTEND, and `pub mod optimal_design;` already declared (~line 109) — do NOT add a new `pub mod`.
    - `fdars-core/src/prelude.rs:51` — the k-Shape prelude block (`pub use crate::kshape::{...}`) is the peer-pattern insertion point.
  </read_first>
  <action>
    In `lib.rs`, replace the existing Phase-64 re-export line with the full six-symbol additive
    line (add `optimal_design`, `OptDesConfig`, `OptDesResult`; keep `design_criterion`,
    `DesignCriterion`, `OptimalityKind`). Update the trailing comment to note the Phase-65
    finalized surface. This is a scoped `Edit`, NOT a rewrite — change only that one `pub use` line.

    In `prelude.rs`, add a new block after the k-Shape block:
    `pub use crate::optimal_design::{design_criterion, optimal_design, DesignCriterion, OptDesConfig, OptDesResult, OptimalityKind};`
    with a `// Optimal experimental design (FOptDes, v0.35.0)` comment (mirror the kshape/kernel_kmeans peer pattern).

    Change NO existing signature or existing re-export entry — purely additive. Do not touch WASM
    or R binding files.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo build -p fdars-core --features linalg,parallel 2>&1 | tail -5</automated>
    <fails_when>output contains "error[" (unresolved/duplicate import or missing symbol)</fails_when>
  </verify>
  <acceptance_criteria>
    - `fdars_core::optimal_design`, `OptDesConfig`, `OptDesResult` reachable at the crate root additively (FOD-05).
    - `fdars_core::prelude` re-exports the full six-symbol FOptDes surface (FOD-05).
    - No existing re-export entry removed or renamed; no new crate dependency (FOD-05).
  </acceptance_criteria>
  <done>Crate root and prelude expose the full FOptDes surface; crate builds under `--features linalg,parallel`.</done>
  <reversibility rating="reversible">Additive re-export lines only; revert by restoring the single original Phase-64 line and deleting the prelude block.</reversibility>
</task>

<task type="auto">
  <name>Task 2: Module-level end-to-end doctest (fit PACE -> optimal_design -> selected_argvals)</name>
  <files>fdars-core/src/optimal_design.rs</files>
  <read_first>
    - `fdars-core/src/optimal_design.rs:1-24` — existing `//!` module doc block (append the runnable example after it).
    - `fdars-core/src/pace_fpca.rs:52-90` — `PaceFpcaConfig` fields + `Default` (`work_grid`); `pace_fpca(data: &IrregFdata, config: &PaceFpcaConfig) -> Result<PaceFpcaResult, FdarError>` at line 266.
    - `fdars-core/src/irreg_fdata/mod.rs:52-90` — `IrregFdata::from_lists(argvals_list, values_list)` constructor.
    - MEMORY.md `tmp-exhaustion-blocks-precommit` — `/tmp` tmpfs exhaustion blocks doctest linking; run `df /tmp` first, free `/tmp` / use `--no-verify` on the docs commit if needed.
  </read_first>
  <action>
    Add a `//! # End-to-end example` section with a runnable ```rust doctest to the module-level
    `//!` block. The doctest MUST go through the real fit path — `PaceFpcaResult` is
    `#[non_exhaustive]`, so an external-crate doctest CANNOT struct-literal it; construct data with
    `IrregFdata::from_lists`, fit via `pace_fpca(&data, &PaceFpcaConfig::default())` (or a small
    explicit config with a set `work_grid`), then call `optimal_design`.

    Flow: build a tiny synthetic `IrregFdata` (a handful of curves sampled on a shared small grid,
    enough for `pace_fpca` to extract >=1 component — reuse the sampling shape from the pace_fpca
    module tests if a minimal one exists), set `candidate_grid = model.argvals.clone()`, `budget = 2`,
    `criterion: DesignCriterion::Trajectory`, call `optimal_design(&model, &config).unwrap()`, and
    `assert_eq!(result.selected_indices.len(), 2); assert_eq!(result.criterion_trace.len(), 2);`
    plus read `result.selected_argvals`. Import via the crate-root surface
    (`use fdars_core::{optimal_design, DesignCriterion, OptDesConfig};` + `pace_fpca` + `irreg_fdata`).
    Keep the example minimal so doctest link time and `/tmp` pressure stay low.

    Before running the doctest, check `df /tmp`; if the tmpfs is near-full, free it (or defer the
    docs commit with `--no-verify` per the MEMORY hazard) — a doctest link OOM is NOT a code defect.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && df /tmp | tail -1 && cargo test -p fdars-core --doc --features linalg,parallel optimal_design 2>&1 | tail -15</automated>
    <fails_when>output contains "test result: FAILED" or "error[" (doctest compile/assert failure) — a "No space left on device" link error is a /tmp-tmpfs hazard, not a code defect (free /tmp and rerun)</fails_when>
  </verify>
  <acceptance_criteria>
    - Module-level doctest compiles and passes via the real `pace_fpca` fit path (FOD-05).
    - Doctest demonstrates fit PACE -> `optimal_design` -> read `selected_argvals` (FOD-05).
    - Doctest imports through the crate-root re-export surface (validates Task 1 reachability).
  </acceptance_criteria>
  <done>`cargo test -p fdars-core --doc --features linalg,parallel` passes the new optimal_design doctest.</done>
</task>

<task type="auto">
  <name>Task 3: Criterion benchmark + [[bench]] stanza + whole-crate gates</name>
  <files>fdars-core/benches/optimal_design.rs, fdars-core/Cargo.toml</files>
  <read_first>
    - `fdars-core/benches/kshape.rs:1-58` — criterion 0.5 structure: `use criterion::{black_box, criterion_group, criterion_main, Criterion};`, inline synthetic fixture, `criterion_group!`/`criterion_main!`.
    - `fdars-core/Cargo.toml:143-148` — existing `[[bench]] name = "kshape"  harness = false` stanza to append after.
    - `fdars-core/src/optimal_design.rs` `synthetic_model_params` fixture — NOT reachable from `benches/` (it is `#[cfg(test)]`), so DUPLICATE the ~20-line synthetic `PaceFpcaResult` builder inline in the bench (as kshape's bench duplicates its fixture). NOTE: `PaceFpcaResult` is `#[non_exhaustive]` — but `benches/` compiles as an EXTERNAL crate, so it CANNOT struct-literal `PaceFpcaResult` either. Build the bench model via the real `pace_fpca` fit path (IrregFdata -> pace_fpca) OR, if a public constructor exists, use it. Confirm the chosen path compiles from an external crate before finalizing.
    - MEMORY.md `ci-clippy-all-targets-gate` and `target-dir-fills-home-partition` (adding a bench grows `target/debug/` — `rm -rf target/debug/{incremental,examples}` if `/home` fills).
  </read_first>
  <action>
    Create `fdars-core/benches/optimal_design.rs` (criterion 0.5). Because `benches/` is an external
    crate and `PaceFpcaResult` is `#[non_exhaustive]`, build the benchmark model through the public
    fit path (`IrregFdata::from_lists` -> `pace_fpca(&data, &PaceFpcaConfig{ work_grid: <m-point grid>, ..default })`)
    on a small representative dataset that yields a grid of ~51 points and >=2 components. Benchmark
    four functions in one group `"optimal_design"`:
    - `design_criterion(&model, &[<5 fixed indices>], DesignCriterion::Trajectory)`
    - `design_criterion(&model, &[<same 5>], DesignCriterion::Score(OptimalityKind::A))`
    - `optimal_design(&model, &{grid=model.argvals.clone(), budget=5, Trajectory})`
    - `optimal_design(&model, &{grid=model.argvals.clone(), budget=5, Score(OptimalityKind::A)})`
    Wrap inputs in `black_box`. Import the six symbols via the crate root
    (`use fdars_core::{design_criterion, optimal_design, DesignCriterion, OptDesConfig, OptimalityKind};`).
    Keep the dataset small so `cargo build --benches` and the eventual `cargo bench` stay quick.

    In `Cargo.toml`, append after the `kshape` bench stanza:
    `[[bench]]` / `name = "optimal_design"   # Phase 65 FOD-05 — design_criterion + optimal_design pipeline coverage` / `harness = false`.

    Then run the whole-crate gates: `cargo fmt -p fdars-core` (fix drift), the `--all-targets` clippy
    gate (covers the new bench code), `cargo build --benches` (bench compiles), and the FULL test +
    doctest suite. Do NOT run `--features serde` (pre-existing ClassifFit break, unrelated). Do NOT
    run `cargo bench` in CI (too slow) — compile-check only.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo build --benches -p fdars-core --features linalg,parallel 2>&1 | tail -8</automated>
    <fails_when>output contains "error[" or "no bench target named" (bench does not compile or [[bench]] stanza missing)</fails_when>
    <automated>cd /home/simonm/projects/rust/fdars && cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -8</automated>
    <fails_when>output contains "error:" or "warning:" (clippy denies bench/test/lib code under -D warnings)</fails_when>
    <automated>cd /home/simonm/projects/rust/fdars && cargo fmt -p fdars-core --check 2>&1 | tail -3</automated>
    <fails_when>command exits non-zero / prints a diff (fmt drift present)</fails_when>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -15</automated>
    <fails_when>output contains "test result: FAILED" (full lib/integration/doctest suite regression)</fails_when>
  </verify>
  <acceptance_criteria>
    - `fdars-core/benches/optimal_design.rs` compiles under `cargo build --benches --features linalg,parallel` (FOD-05).
    - `[[bench]] name = "optimal_design"` registered in Cargo.toml (FOD-05).
    - Bench covers `design_criterion` + `optimal_design` for both Trajectory and Score(A) (FOD-05).
    - Whole-crate clippy `--all-targets`, fmt, and the full test+doctest suite are clean — 28 examples + WASM + R unaffected (FOD-05).
  </acceptance_criteria>
  <done>Bench compiles and is registered; whole-crate fmt/clippy/full-suite gates green; the full FOptDes surface is shipped additively.</done>
  <reversibility rating="reversible">New bench file + additive Cargo stanza; revert by deleting both.</reversibility>
</task>

</tasks>

<verification>
- `cargo build -p fdars-core --features linalg,parallel` — crate root + prelude re-exports resolve.
- `cargo test -p fdars-core --doc --features linalg,parallel` — module doctest passes.
- `cargo build --benches -p fdars-core --features linalg,parallel` — bench compiles.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
- `cargo fmt -p fdars-core --check` clean.
- `cargo test -p fdars-core --features linalg,parallel` — full suite green, no regression.
- Numerical-robustness note (no threat_model required — additive re-exports + doctest + benchmark, no external API/schema/identity/network surface): the only runtime surface is the already-validated `optimal_design`; this plan adds no new execution path, only public reachability + coverage.
- Do NOT run `--features serde` (pre-existing unrelated ClassifFit break) or `cargo bench` in CI (too slow).
</verification>

<success_criteria>
- FOD-05 complete: full crate-root + prelude re-export surface (`optimal_design`, `design_criterion`, `OptDesConfig`, `OptDesResult`, `DesignCriterion`, `OptimalityKind`), a passing end-to-end module doctest, and a registered criterion benchmark.
- Additive/non-breaking: no existing public signature changed, 28 examples + WASM + R bindings unaffected, no new crate dependency, MSRV 1.81 preserved.
</success_criteria>

<output>
Create `.planning/phases/65-greedy-selection-integration/65-02-SUMMARY.md` when done.
</output>
