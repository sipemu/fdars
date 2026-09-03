---
phase: 65-greedy-selection-integration
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/optimal_design.rs
autonomous: true
requirements: [FOD-04, FOD-05]
estimate:
  tokens: 70000
  raw_tokens: 47000
  tasks: 3
  confidence: high
must_haves:
  truths:
    - "optimal_design(model, config) returns config.budget selected indices (FOD-04)"
    - "Two identical optimal_design calls produce byte-identical selected_indices and criterion_trace (FOD-04)"
    - "selected_indices identical with and without --features parallel (FOD-04)"
    - "No index appears twice in selected_indices — duplicate-free (FOD-04)"
    - "criterion_trace is monotone non-increasing (trace[i+1] <= trace[i] + 1e-12) (FOD-04)"
    - "budget==0 -> Err(InvalidParameter) (FOD-04)"
    - "budget>candidate_grid.len() -> Err(InvalidParameter) (FOD-04)"
    - "off-grid candidate (not in model.argvals within 1e-9) -> Err(InvalidParameter) (FOD-04)"
    - "model.ncomp==0 -> Err(InvalidParameter) (FOD-04)"
    - "model.sigma2<=0 -> Err(InvalidParameter) (FOD-04)"
    - "Trajectory first-step selection equals the numerically-computed argmin over all candidates (FOD-05)"
    - "Score(A) criterion produces a valid OptDesResult structure (FOD-05)"
    - "OptDesConfig::default() constructs; empty grid is caught at call time not construction (FOD-05)"
    - "PaceFpcaResult consumed read-only — no re-estimation (FOD-05)"
  artifacts:
    - "fdars-core/src/optimal_design.rs (extended with OptDesConfig, OptDesResult, optimal_design fn, greedy helpers, 13 inline tests)"
  key_links:
    - "optimal_design greedy loop delegates every candidate evaluation to Phase 64's design_criterion"
    - "candidate_grid values mapped to model.argvals indices via 1e-9 FP-tolerant position search"
    - "parallel candidate evaluation via iter_maybe_parallel! then SEQUENTIAL fold-based argmin (never rayon min_by)"
---

<objective>
Extend `fdars-core/src/optimal_design.rs` with the deterministic greedy sequential
forward-selection loop `optimal_design(model, config)` plus its `OptDesConfig` /
`OptDesResult` types and the 13-test inline verification block. This is the tracer:
the whole greedy path is wired end-to-end (config in → validation → candidate→index
mapping → greedy loop delegating to Phase 64's `design_criterion` → result out) and
proven green with determinism, monotonicity, duplicate-free, known-answer, and all
five validation gates. Adds NO new math — pure orchestration over `design_criterion`.

Purpose: Deliver FOD-04 (greedy selection) and the algorithmic half of FOD-05
(two-stage read-only entry point) as a fully-tested slice before the additive
re-export / doctest / benchmark integration in plan 65-02.
Output: An extended `optimal_design.rs` whose greedy loop + config/result types
compile and pass all 13 module tests under `--features linalg,parallel`.
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
@.planning/phases/65-greedy-selection-integration/65-VALIDATION.md

# Phase 64 deliverable being extended — design_criterion signature, enums, synthetic_model fixture
@fdars-core/src/optimal_design.rs
# iter_maybe_parallel! contract + !Send constraints
@fdars-core/src/parallel.rs
# Config/Result derive + non_exhaustive + determinism-test precedent
@fdars-core/src/pace_fpca.rs
</context>

<artifacts_this_phase_produces>
NEW public symbols introduced by this plan (in `fdars-core/src/optimal_design.rs`):
- `pub struct OptDesConfig` — fields `candidate_grid: Vec<f64>`, `budget: usize`, `criterion: DesignCriterion`; `Default` impl (`{ candidate_grid: vec![], budget: 1, criterion: DesignCriterion::Trajectory }`); derives `Debug, Clone, PartialEq` + serde-gated. NOT `#[non_exhaustive]`.
- `pub struct OptDesResult` — fields `selected_indices: Vec<usize>`, `selected_argvals: Vec<f64>`, `criterion_trace: Vec<f64>`; `#[non_exhaustive]`; derives `Debug, Clone, PartialEq` + serde-gated.
- `pub fn optimal_design(model: &PaceFpcaResult, config: &OptDesConfig) -> Result<OptDesResult, FdarError>` — `#[must_use = "..."]`.
- Private helper(s) for candidate→index mapping (Claude's discretion on factoring).

These are NOT re-exported from `lib.rs`/`prelude.rs` in this plan — that is plan 65-02.
Reachable within the crate as `crate::optimal_design::{OptDesConfig, OptDesResult, optimal_design}`.
</artifacts_this_phase_produces>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: OptDesConfig/OptDesResult types + optimal_design greedy loop — one path end-to-end</name>
  <files>fdars-core/src/optimal_design.rs</files>
  <read_first>
    - `fdars-core/src/optimal_design.rs:40-64` — `DesignCriterion` / `OptimalityKind` enums (derive Clone; `DesignCriterion::Score(OptimalityKind)`).
    - `fdars-core/src/optimal_design.rs:86-90` — `design_criterion(model: &PaceFpcaResult, selected: &[usize], criterion: DesignCriterion) -> Result<f64, FdarError>` signature (takes `criterion` BY VALUE; needs `.clone()` per step).
    - `fdars-core/src/pace_fpca.rs:51-97` — `PaceFpcaConfig` (NOT non_exhaustive, `Default` impl) and `PaceFpcaResult` (IS `#[non_exhaustive]`) derive/serde-gate pattern to mirror.
    - `fdars-core/src/parallel.rs:42-55` — `iter_maybe_parallel!` expands to `into_par_iter()` (parallel) / `into_iter()` (sequential); collect BEFORE the sequential argmin.
    - `fdars-core/src/error.rs` — `FdarError::InvalidParameter { parameter: &'static str, message: String }` shape.
  </read_first>
  <behavior>
    - Test (basic, FOD-04): `optimal_design(&synthetic_model(51), &{grid=argvals, budget=3, Trajectory})` returns `selected_indices.len()==3`, `selected_argvals.len()==3`, `criterion_trace.len()==3`.
    - Test (determinism two-call, FOD-04): two identical calls -> `r1.selected_indices==r2.selected_indices` AND `r1.criterion_trace==r2.criterion_trace`.
    - Test (duplicate-free, FOD-04): budget-5 result has all-unique `selected_indices`.
    - Test (monotone trace, FOD-04): for every consecutive pair, `trace[i+1] <= trace[i] + 1e-12`.
    - Test (trajectory first-point, FOD-05): compute the argmin index by scanning `design_criterion(model, &[idx], Trajectory)` over all candidates in-test; assert `result.selected_indices[0]` equals it (numerically computed, NOT hardcoded).
    - Test (score(A), FOD-05): budget-2 Score(A) call returns a well-formed result (`selected_indices.len()==2`, trace non-increasing).
    - Test (config default, FOD-05): `OptDesConfig::default()` constructs; a call with the default (empty grid, budget 1) returns `Err(InvalidParameter)` — the empty grid is caught at call time, not at `default()`.
  </behavior>
  <action>
    Add the three symbols to `optimal_design.rs` (place after the `design_criterion` public
    fn, before the `#[cfg(test)]` block). Mirror `pace_fpca.rs` derive conventions exactly.

    `OptDesConfig`: derive `Debug, Clone, PartialEq`, `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`, NOT `#[non_exhaustive]`. Fields `candidate_grid: Vec<f64>`, `budget: usize`, `criterion: DesignCriterion`. Hand-write `impl Default` returning `{ candidate_grid: vec![], budget: 1, criterion: DesignCriterion::Trajectory }` per D — the CONTEXT locked API (single `criterion` field, no separate `OptimalityKind` field; `Score(OptimalityKind)` already carries it).

    `OptDesResult`: same derives PLUS `#[non_exhaustive]`. Fields `selected_indices: Vec<usize>`, `selected_argvals: Vec<f64>`, `criterion_trace: Vec<f64>`. Doc each field (selection order; trace length == budget, monotone non-increasing).

    `optimal_design`: `#[must_use = "expensive computation whose result should not be discarded"]` (NOT bare `#[must_use]` — that trips clippy `double_must_use` on a `Result` return under `-D warnings`). Signature `pub fn optimal_design(model: &PaceFpcaResult, config: &OptDesConfig) -> Result<OptDesResult, FdarError>`.

    Body order:
    1. Validation (all return `FdarError::InvalidParameter`, never panic): `config.budget == 0`; `config.budget > config.candidate_grid.len()`; `model.ncomp == 0`; `model.sigma2 <= 0.0`. (Delegate deeper model checks to `design_criterion`, but front these so a bad budget/model fails fast before any candidate work.)
    2. Map `candidate_grid` -> `model.argvals` indices ONCE via an FP-tolerant sequential position search: for each candidate, `model.argvals.iter().position(|&t| (t - cand).abs() < 1e-9)`, `.ok_or_else(|| FdarError::InvalidParameter { parameter: "config.candidate_grid", message: format!("candidate {cand:.6} not found in model.argvals within tolerance 1e-9") })`. Collect into `Vec<usize>` `candidate_indices` (may factor into a private helper — Claude's discretion). Preserve candidate_grid order.
    3. Greedy loop `for _step in 0..config.budget`: build `remaining` = `candidate_indices` filtered to exclude already-`selected` (preserves order); parallel-evaluate each remaining candidate via `iter_maybe_parallel!(remaining.iter().copied())` mapping `idx` -> `(idx, design_criterion(model, &trial, config.criterion.clone())?)` where `trial = selected.clone(); trial.push(idx);`, collecting `Result<Vec<(usize,f64)>, FdarError>` and `?`-propagating; then take the SEQUENTIAL argmin over the collected `scores` via `.fold(None, ...)` keeping the FIRST minimum (strict `val < bv`) so the smallest-index candidate wins ties (rayon `min_by` is NOT stable — do not use it). Push winner to `selected` and its value to `trace`.
    4. Build `OptDesResult`: `selected_argvals` = `selected.iter().map(|&i| model.argvals[i]).collect()`; `criterion_trace = trace`. Because `OptDesResult` is `#[non_exhaustive]`, construct it in-crate via struct literal (allowed within the defining crate).

    The parallel closure captures only `&model` and `&selected` (immutable) and allocates its own `trial` — `PaceFpcaResult` is `Send + Sync` (all fields `Vec<f64>`/`FdMatrix`/`usize`/`f64`), no `!Send` FftPlanner here, so `--features parallel` compiles. Under `#[cfg(feature = "parallel")]` add `use rayon::iter::ParallelIterator;` in scope of the map/collect as the codebase does.

    Do NOT re-estimate the model, do NOT mutate `model`, do NOT add caching inside `design_criterion`. Read-only two-stage contract (FOD-05).
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo build -p fdars-core --features linalg,parallel 2>&1 | tail -5</automated>
    <fails_when>output contains "error[" or "error:" (compile failure — missing symbol, type mismatch, or !Send closure)</fails_when>
  </verify>
  <acceptance_criteria>
    - `optimal_design`, `OptDesConfig`, `OptDesResult` exist and the crate builds under `--features linalg,parallel` (FOD-04, FOD-05).
    - `OptDesConfig::default()` returns `{ vec![], 1, Trajectory }` (FOD-05).
    - Greedy argmin is a sequential fold over collected scores, smallest-index tie-break; no rayon `min_by` (FOD-04).
  </acceptance_criteria>
  <done>Crate compiles with the three new symbols under `--features linalg,parallel`; the greedy path is wired end-to-end with sequential tie-break argmin and read-only model access.</done>
  <reversibility rating="reversible">Additive new symbols in one file; no existing signature changed — revert by deleting the added block.</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Validation-guard + determinism + known-answer tests (13 module tests)</name>
  <files>fdars-core/src/optimal_design.rs</files>
  <read_first>
    - `fdars-core/src/optimal_design.rs:368-415` — existing `#[cfg(test)] mod tests` block and `synthetic_model(m)` / `synthetic_model_params(m, eigenvalues, sigma2)` fixtures (reuse both; do NOT duplicate).
    - `fdars-core/src/pace_fpca.rs` determinism test pattern (two identical calls, assert_eq on the deterministic output vectors).
    - Task 1 body above for the exact validation error conditions.
  </read_first>
  <behavior>
    - budget==0 -> `matches!(err, Err(FdarError::InvalidParameter{..}))`.
    - budget>grid.len() -> InvalidParameter.
    - off-grid candidate (push a value like `argvals[0] + 0.5/(m as f64)` between grid points, or `999.0`) -> InvalidParameter.
    - ncomp==0 -> InvalidParameter (build a model via `synthetic_model_params` then set `ncomp=0`, or a fixture with ncomp 0; delegate is acceptable but assert the Err).
    - sigma2<=0 -> InvalidParameter (fixture with `sigma2=0.0` or negative).
    - seq==parallel: the two-call determinism test doubles as the seq==parallel gate under the CI feature matrix; assert stable `selected_indices` + `criterion_trace`.
  </behavior>
  <action>
    Add the following `#[test]` functions to the existing inline `mod tests` block (append; keep
    the existing Phase 64 tests intact). Reuse `synthetic_model` / `synthetic_model_params`.

    Names (exact — the VALIDATION.md per-task map references these):
    `test_optimal_design_basic`, `test_determinism_two_calls`, `test_duplicate_free`,
    `test_monotone_trace`, `test_validation_budget_zero`, `test_validation_budget_exceeds_grid`,
    `test_validation_off_grid_candidate`, `test_validation_ncomp_zero`,
    `test_validation_sigma2_nonpositive`, `test_trajectory_selects_informative_point`,
    `test_score_a_selects`, `test_config_default`, and one small `test_prelude_reexport`
    placeholder that constructs `OptDesConfig::default()` via the in-crate path (the true
    prelude-path assertion lands as a doctest/crate-root check in plan 65-02 — here just assert
    `OptDesConfig::default().budget == 1` so the name exists for the validation map; note in a
    code comment that the external prelude reachability is verified in 65-02).

    For `test_trajectory_selects_informative_point`: compute the expected first index IN THE TEST
    by scanning `design_criterion(&model, &[idx], DesignCriterion::Trajectory).unwrap()` for
    `idx in 0..m`, taking the sequential smallest-index argmin, then assert
    `result.selected_indices[0] == expected`. Do NOT hardcode the index.

    For the validation guards that are enforced inside `design_criterion` (ncomp==0, sigma2<=0):
    it is acceptable that the error surfaces via delegation — assert the returned `Err` is
    `FdarError::InvalidParameter` regardless of which layer raised it.

    Comment-text discipline: do not put the literal token an assertion negative-greps for into an
    action/comment. (No negative-grep gates are used in this plan; unit tests assert positively.)
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --lib optimal_design --features linalg,parallel 2>&1 | tail -25</automated>
    <fails_when>output contains "test result: FAILED" or "error[" or any "FAILED" line</fails_when>
  </verify>
  <acceptance_criteria>
    - All 13 named tests present and passing (FOD-04, FOD-05).
    - `test_determinism_two_calls` asserts byte-identical `selected_indices` AND `criterion_trace` (FOD-04).
    - `test_duplicate_free` asserts all-unique indices; `test_monotone_trace` asserts `trace[i+1] <= trace[i] + 1e-12` (FOD-04).
    - `test_trajectory_selects_informative_point` computes the expected argmin in-test, not hardcoded (FOD-05).
    - All five validation guards each assert `Err(FdarError::InvalidParameter)` (FOD-04).
  </acceptance_criteria>
  <done>`cargo test -p fdars-core --lib optimal_design --features linalg,parallel` passes with all Phase-64 and the 13 new tests green.</done>
</task>

<task type="auto">
  <name>Task 3: seq==parallel confirmation + module-scope clippy/fmt gate</name>
  <files>fdars-core/src/optimal_design.rs</files>
  <read_first>
    - STATE.md Blockers/Concerns — CI clippy gate is `cargo clippy --all-targets --features linalg,parallel -- -D warnings`; `--no-verify` commits leave fmt drift, so run `cargo fmt`.
    - MEMORY.md pointer `ci-clippy-all-targets-gate` and `noverify-commits-leave-fmt-drift`.
  </read_first>
  <action>
    No new code unless a gate fails. Prove the seq==parallel determinism gate by running the
    module tests WITHOUT the parallel feature and confirming identical selection, then run the
    default-feature build so the sequential `iter_maybe_parallel!` arm is exercised. Run
    `cargo fmt -p fdars-core` (fix drift) and the whole-crate clippy gate. If clippy flags the new
    code (e.g. `needless_range_loop`, `double_must_use`, `redundant_clone`), fix minimally without
    changing behavior. Do NOT run `--features serde` (pre-existing unrelated ClassifFit break —
    NOT a Phase 65 regression).
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo test -p fdars-core --lib optimal_design 2>&1 | tail -10</automated>
    <fails_when>output contains "test result: FAILED" (sequential/default-feature build selects differently — determinism/seq==parallel violation)</fails_when>
    <automated>cd /home/simonm/projects/rust/fdars && cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -8</automated>
    <fails_when>output contains "error:" or "warning:" (clippy denies under -D warnings)</fails_when>
    <automated>cd /home/simonm/projects/rust/fdars && cargo fmt -p fdars-core --check 2>&1 | tail -3</automated>
    <fails_when>command exits non-zero / prints a diff (fmt drift present)</fails_when>
  </verify>
  <acceptance_criteria>
    - Module tests pass in BOTH default (sequential) and `linalg,parallel` feature sets — seq==parallel confirmed (FOD-04).
    - `cargo clippy --all-targets --features linalg,parallel -- -D warnings` is clean including the new code (Both).
    - `cargo fmt -p fdars-core --check` clean (Both).
  </acceptance_criteria>
  <done>Determinism holds across feature sets; whole-crate clippy `--all-targets` and fmt are clean.</done>
</task>

</tasks>

<verification>
- `cargo build -p fdars-core --features linalg,parallel` compiles the three new symbols.
- `cargo test -p fdars-core --lib optimal_design --features linalg,parallel` — all 13 new tests + Phase-64 tests green.
- `cargo test -p fdars-core --lib optimal_design` (default/sequential) — identical selection (seq==parallel).
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean.
- `cargo fmt -p fdars-core --check` clean.
- Numerical-robustness note (no threat_model required — pure in-crate numerical orchestration, only surface is input validation which is a first-class acceptance criterion): all five validation guards return `FdarError::InvalidParameter` and never panic; `?` propagates any `design_criterion` numerical failure into the caller — no silent NaN enters `criterion_trace`.
</verification>

<success_criteria>
- FOD-04: greedy loop returns `budget` unique indices, deterministic, monotone non-increasing trace, seq==parallel, all five validation guards enforced.
- FOD-05 (algorithmic half): `optimal_design(&PaceFpcaResult, &OptDesConfig)` two-stage read-only entry point implemented; `OptDesConfig`/`OptDesResult` follow the pace_fpca derive + non_exhaustive precedent.
- No existing public signature changed; no new crate dependency; MSRV 1.81 preserved.
</success_criteria>

<output>
Create `.planning/phases/65-greedy-selection-integration/65-01-SUMMARY.md` when done.
</output>
