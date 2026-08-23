---
phase: 43-boosting-bayesian-functional-regression
plan: 05
type: execute
wave: 2
depends_on: [43-01]
files_modified:
  - fdars-core/src/boosting_regression/stability.rs
autonomous: true
requirements: [REG-06-05]
estimate:
  tokens: 50000
  raw_tokens: 25000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "stability_selection runs FDboost-style stability selection over the boosting base-learners (REG-06-05): B seeded subsamples of ⌊n/2⌋ without replacement, each fitting boost_fosr, aggregated into per-learner selection frequencies and a stable predictor set at threshold π (default 0.9)"
    - "Selection frequencies lie in [0,1]; the stable set is a subset of all base-learner indices; the reported PFER bound is >= 0"
    - "The wrapper is deterministic under a fixed seed (per-replicate seed = seed.wrapping_add(b)); a strong-signal predictor exceeds the threshold"
  artifacts:
    - fdars-core/src/boosting_regression/stability.rs
  key_links:
    - "stability_selection wraps boost_fosr from Plan 01, running it on each subsample and aggregating selected_learners into selection frequencies"
    - "Per-replicate RNG seeding StdRng::seed_from_u64(config.seed.wrapping_add(b as u64)) inside iter_maybe_parallel!(0..B) — each closure owns its RNG (deterministic + parallel-safe)"
---

<objective>
Implement FDboost-style stability selection (REG-06-05): a subsampling wrapper around `boost_fosr` (Plan 01) that draws B seeded subsamples of ⌊n/2⌋ observations without replacement, fits the boosted model on each, records which base-learners were ever selected, and aggregates per-learner selection frequencies plus a stable predictor set at threshold π and an informational PFER bound. This fills the `stability.rs` skeleton from Plan 01.

Purpose: Stability selection turns the boosting path into a principled variable-selection tool (Meinshausen–Bühlmann), a capability fdars lacks. It reuses `boost_fosr` and the established seeded-parallel-resampling pattern — the only new logic is subsampling-without-replacement and frequency aggregation.
Output: A working `stability_selection()` returning `StabilityResult`, with inline range + subset + determinism + strong-signal + error-path tests.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md
@.planning/phases/43-boosting-bayesian-functional-regression/43-PATTERNS.md
@.planning/phases/43-boosting-bayesian-functional-regression/43-01-boosting-core-fosr-SUMMARY.md

Depends on Plan 01: `BoostingConfig`, `StabilityConfig`, `StabilityResult`, and `boost_fosr`. `stability.rs` already exists as a compiling skeleton — this plan replaces its body. Reuses `iter_maybe_parallel!` (`src/parallel.rs`) and the per-replicate seeding pattern from `src/scalar_on_function/bootstrap.rs:89`. Prefix cargo with `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`.
</context>

<artifacts_produced>
New public symbol implemented (signature declared in Plan 01; drift excludes it):
- `pub fn stability_selection(data: &FdMatrix, predictors: &FdMatrix, argvals: &[f64], boost_config: &BoostingConfig, stab_config: &StabilityConfig) -> Result<StabilityResult, FdarError>`

`StabilityResult` (Plan 01 mod.rs) fields populated: `selection_freq`, `stable_set`, `pi_thr`, `pfer_bound`, `n_resamples`.

Private helper (this file, new logic — no analog):
- `fn sample_without_replacement(rng: &mut StdRng, n: usize, k: usize) -> Vec<usize>` — Fisher–Yates partial shuffle returning ⌊n/2⌋ distinct indices
</artifacts_produced>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end stability_selection — seeded subsampling wrapper + frequency aggregation</name>
  <files>fdars-core/src/boosting_regression/stability.rs</files>
  <read_first>
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Algorithm 5: subsampling scheme, selection-frequency aggregation, stable-set threshold, PFER bound formula) + Common Pitfall 6 (RNG state per parallel resample)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-PATTERNS.md (stability.rs section: bootstrap.rs analog, per-replicate seeding, sample_without_replacement helper, result struct, function signature)
    - fdars-core/src/scalar_on_function/bootstrap.rs lines 80-105 (iter_maybe_parallel! resample loop + StdRng::seed_from_u64(seed.wrapping_add(b)) + subsample_rows for row subsetting)
    - fdars-core/src/boosting_regression/boost_fosr.rs (Plan 01 — boost_fosr signature + BoostFosrResult.selected_learners which drives selection detection)
    - fdars-core/src/parallel.rs lines 42-70 (iter_maybe_parallel! macro)
  </read_first>
  <action>
Replace the `stability.rs` skeleton body with the stability-selection implementation per RESEARCH §Algorithm 5.

Validate inputs: n>=4 (need ⌊n/2⌋>=2), m>0, predictors.nrows()==n → `FdarError::InvalidDimension`; stab_config: n_resamples>=1, pi_thr in (0.5, 1.0] → `FdarError::InvalidParameter`; also validate boost_config the way boost_fosr does (or rely on boost_fosr's own validation surfacing an early error on the first subsample).

Implement `sample_without_replacement` (Fisher–Yates partial shuffle over 0..n, take first ⌊n/2⌋). 

Resample loop: `let per_replicate: Vec<Vec<bool>> = iter_maybe_parallel!(0..stab_config.n_resamples).map(|b| { let mut rng = StdRng::seed_from_u64(stab_config.seed.wrapping_add(b as u64)); let idx = sample_without_replacement(&mut rng, n, n/2); subset data + predictors rows by idx (mirror bootstrap.rs subsample_rows); run boost_fosr(&sub_data, &sub_pred, argvals, boost_config); collect a length-p bool vector marking which predictors appear in result.selected_learners (treat a failed fit as all-false). }).collect();` Each closure owns its RNG per Pitfall 6.

Aggregate: `select_count[j]` = number of replicates where predictor j was selected; `selection_freq[j] = select_count[j] / B`; `stable_set` = indices with selection_freq[j] >= pi_thr; PFER bound q = mean per-subsample selection count = (Σ select_count) / B, `pfer_bound = q*q / ((2.0*pi_thr - 1.0) * p as f64)`. Assemble `StabilityResult { selection_freq, stable_set, pi_thr, pfer_bound, n_resamples }`. Mark `#[must_use]`. Rustdoc must document: Meinshausen–Bühlmann ⌊n/2⌋ subsampling, π=0.9 default, PFER is informational, and any divergence from `stabs` cited.

Do NOT inline fenced code beyond what read_first names — follow RESEARCH §Algorithm 5 and the bootstrap.rs resample pattern.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo build -p fdars-core --features linalg,parallel 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - `cargo build -p fdars-core --features linalg,parallel` exits 0
    - `stability.rs` no longer contains the string `not yet implemented`
    - `grep -q "fn sample_without_replacement" fdars-core/src/boosting_regression/stability.rs` succeeds
    - `grep -q "wrapping_add" fdars-core/src/boosting_regression/stability.rs` succeeds (per-replicate seeding)
    - `grep -q "iter_maybe_parallel" fdars-core/src/boosting_regression/stability.rs` succeeds
  </acceptance_criteria>
  <done>stability_selection is implemented end-to-end: seeded ⌊n/2⌋ subsampling wrapping boost_fosr, frequency aggregation, stable set, PFER bound; the crate compiles.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: stability_selection range, subset, determinism, strong-signal, and error-path tests + gate</name>
  <files>fdars-core/src/boosting_regression/stability.rs</files>
  <read_first>
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Validation Architecture → REG-06-05 rows: frequencies in [0,1]; stable set subset of learners; PFER bound >= 0)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-VALIDATION.md (stability oracles: strong-signal predictor exceeds threshold; determinism under seed)
    - fdars-core/src/scalar_on_function/bootstrap.rs inline tests (seeded-determinism test style)
    - fdars-core/src/test_helpers.rs (`uniform_grid`)
  </read_first>
  <behavior>
    - Every selection_freq[j] lies in [0.0, 1.0]
    - stable_set is a subset of 0..p and contains no duplicates
    - pfer_bound >= 0.0
    - Determinism: two `stability_selection` calls with the same stab_config.seed produce identical selection_freq and stable_set
    - Strong signal: on synthetic data where exactly one predictor carries a strong signal (others noise), that predictor's selection_freq is the highest and (with a low enough pi_thr for the test, e.g. 0.6) is in the stable set
    - Error path: pi_thr <= 0.5 (or > 1.0) returns FdarError::InvalidParameter
    - Error path: predictors.nrows() != data.nrows() returns FdarError::InvalidDimension
  </behavior>
  <action>
Add an inline `#[cfg(test)] mod tests` block. Build a synthetic dataset (n>=30, m>=15) with p>=3 predictors where exactly one drives a strong smooth signal in Y and the rest are noise; use a small B (e.g. n_resamples=10-20) and modest mstop to keep the test fast. Write tests for every `<behavior>` bullet: `stability_freq_in_range`, `stability_stable_set_is_subset`, `stability_pfer_non_negative`, `stability_deterministic_under_seed`, `stability_strong_signal_selected`, `stability_errors_on_invalid_pi_thr`, `stability_errors_on_dimension_mismatch`. Then run the full clippy gate + full test suite (with the `parallel` feature so the parallel resample path is exercised); fix any findings in this file.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression::stability 2>&1 | tail -12 && TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -6</automated>
  </verify>
  <acceptance_criteria>
    - `cargo test -p fdars-core --features linalg,parallel boosting_regression::stability` exits 0
    - `cargo clippy --all-targets --features linalg,parallel -- -D warnings` exits 0
    - `grep -c "#\[test\]" fdars-core/src/boosting_regression/stability.rs` returns >= 6
    - A determinism test asserting two same-seed runs are equal is present
  </acceptance_criteria>
  <done>REG-06-05 has inline range + subset + determinism + strong-signal + error-path tests, all green; clippy gate clean.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Pure numeric in-process Rust library function. No I/O, network, deserialization, auth, or external attack surface — inputs are in-memory `FdMatrix` + config from calling Rust code. RNG is for statistical reproducibility, not security. |

## STRIDE Threat Register

Attack surface: NONE — pure numeric computation on in-memory `FdMatrix`. Only failure modes are numerical, handled as `FdarError` returns.

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-43-05a | Tampering | integer overflow in `seed.wrapping_add(b as u64)` | low | mitigate | `wrapping_add` is the correct Rust idiom — no panic on overflow (RESEARCH Security Domain) |
| T-43-05b | Tampering (non-determinism) | shared RNG across parallel resamples | low | mitigate | Each `iter_maybe_parallel!` closure creates its own `StdRng` locally — no cross-thread RNG sharing (Pitfall 6); determinism test enforces this |
| T-43-05c | DoS (numerical) | `n_resamples` / `pi_thr` params | low | mitigate | Validate n_resamples>=1, pi_thr in (0.5,1.0], n>=4 at entry → `FdarError`; PFER denominator `(2·pi_thr−1)` is positive for pi_thr>0.5 |

No package installs. No supply-chain checkpoint required.
</threat_model>

<verification>
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression::stability` green (REG-06-05), including same-seed determinism.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.
- Only `stability.rs` modified (no mod.rs edit — collision-free with sibling wave-2 plans).
</verification>

<success_criteria>
- REG-06-05 satisfied: `stability_selection` runs seeded ⌊n/2⌋ subsampling over the boosting base-learners, producing per-learner selection frequencies in [0,1], a stable set (subset of learners), and a non-negative PFER bound, deterministic under seed, with error-path handling.
- Inline tests green; full clippy gate clean.
</success_criteria>

<output>
Create `.planning/phases/43-boosting-bayesian-functional-regression/43-05-stability-selection-SUMMARY.md` when done.
</output>
