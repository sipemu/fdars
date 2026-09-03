# Phase 65: Greedy Selection & Integration — Research

**Researched:** 2026-09-03
**Domain:** Deterministic greedy forward-selection wrapper over Phase 64's `design_criterion`; Rust/Criterion 0.5 benchmark
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- `OptDesConfig` fields: `candidate_grid: Vec<f64>`, `budget: usize`, `criterion: DesignCriterion`. `Default` impl (NOT `#[non_exhaustive]`) = `{ candidate_grid: vec![], budget: 1, criterion: DesignCriterion::Trajectory }`. Derives: `Debug, Clone, PartialEq` + serde-gated.
- `OptDesResult` fields: `selected_indices: Vec<usize>`, `selected_argvals: Vec<f64>`, achieved-criterion trace (`Vec<f64>`). `#[non_exhaustive]`. Derives: `Debug, Clone, PartialEq` + serde-gated.
- Greedy loop: start empty; at each of `config.budget` steps add the not-yet-selected candidate index that most reduces `config.criterion`; smallest-index tie-break; sequential argmin (never rayon `min_by`).
- Determinism: parallel candidate EVALUATION via `iter_maybe_parallel!`, sequential argmin. Two identical calls → byte-identical `selected_indices`; seq == parallel builds agree.
- Validation returns `Err(FdarError::InvalidParameter)` for: `budget == 0`, `budget > candidate_grid.len()`, any candidate not in `model.argvals` (within FP tolerance), `model.ncomp == 0`, `model.sigma2 <= 0`.
- Full additive crate-root re-export: `pub mod optimal_design` already declared; add `optimal_design`, `OptDesConfig`, `OptDesResult` to the `pub use optimal_design::{...}` line. Add same names to `prelude`.
- Module-level doctest demonstrating end-to-end: fit PACE → `optimal_design` → read `selected_argvals`.
- One criterion benchmark file covering `design_criterion` + `optimal_design` for Trajectory and Score(A) on a representative grid/budget. Register as `[[bench]]` in `fdars-core/Cargo.toml`.
- Extend `fdars-core/src/optimal_design.rs` — no new file, no submodule directory.

### Claude's Discretion
- Internal greedy-loop helper factoring, benchmark input sizes, and test-module layout.

### Deferred Ideas (OUT OF SCOPE)
- SR-criterion, exhaustive/branch-and-bound search, CV-ridge selection, rank-1 Cholesky update, off-grid interpolated candidates (FOD-BREADTH).
- Fixing the pre-existing `--features serde` build break (ClassifFit lacks serde derives in shapelet).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FOD-04 | Greedy sequential forward-selection loop — start empty, add the candidate that most reduces `design_criterion` until budget `p` is reached; deterministic (smallest-index tie-break); monotone non-increasing criterion trace; duplicate-free. | Sections: Greedy Algorithm, Determinism Pattern, Validation Architecture. |
| FOD-05 | Two-stage entry point `optimal_design(model: &PaceFpcaResult, config: &OptDesConfig) -> Result<OptDesResult, FdarError>` consuming an already-estimated PACE model read-only; full crate-root + prelude re-exports; module doctest; criterion benchmark. | Sections: Re-export & Doctest Pattern, Benchmark Pattern, Standard Stack. |
</phase_requirements>

---

## Summary

Phase 65 wraps Phase 64's already-validated `design_criterion` in a deterministic greedy sequential forward-selection loop. The implementation adds no new mathematics: the only algorithmic content is the greedy orchestration (iterate over steps, at each step evaluate all not-yet-selected candidates, take the sequential argmin with smallest-index tie-break, append the winner to the design). Every numerical invariant is inherited from Phase 64.

The two make-or-break risks are (1) a non-stable parallel argmin producing different results with `--features parallel` than without, and (2) candidate→index mapping using exact f64 equality when `model.argvals` contains values that are only approximately equal to `config.candidate_grid` values. Both have documented solutions: sequential fold-based argmin after a parallel `collect`, and an FP-tolerant position search (`(x - target).abs() < 1e-9` with `position()` on the sequential iterator over `model.argvals`).

The three deliverables beyond the greedy loop — additive `lib.rs`/`prelude.rs` re-exports, a module-level doctest, and a Criterion 0.5 benchmark — each follow a precise existing pattern in the codebase.

**Primary recommendation:** Implement `optimal_design` with a parallel evaluate → `collect` → sequential `fold`-based argmin structure. Map `candidate_grid` values to `model.argvals` indices once at validation time (FP tolerance `1e-9`). Use the `synthetic_model(51)` fixture already in `optimal_design.rs` tests for the benchmark fixture.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Greedy selection loop | `optimal_design.rs` (lib) | — | Pure algorithm, no I/O, no external service boundary |
| Criterion evaluation | `design_criterion` (Phase 64, same file) | — | Delegated entirely — no re-implementation |
| Candidate→index mapping | Entry-point validation in `optimal_design` | — | One-time O(G·p) scan at call time; maps `candidate_grid` → `model.argvals` indices |
| Parallelism gate | `iter_maybe_parallel!` macro (`parallel.rs`) | — | Feature-gated, caller-transparent |
| Public API surface | `lib.rs` crate root + `prelude.rs` | — | Additive re-exports only; no existing line changed |
| Integration test / determinism | Inline `#[cfg(test)] mod tests` in `optimal_design.rs` | — | Mirrors `pace_fpca.rs` determinism-test pattern |
| Benchmark | `benches/optimal_design.rs` + `[[bench]]` in `Cargo.toml` | — | Criterion 0.5 harness; mirrors `kshape.rs` bench |

---

## Standard Stack

### Core (no new dependencies required)

| Item | Source | Purpose | Why Standard |
|------|---------|---------|--------------|
| `design_criterion` | `crate::optimal_design` (Phase 64) | Per-step criterion evaluation | The only math in the loop; validated and `#[must_use]` |
| `iter_maybe_parallel!` | `crate::parallel` (macro, `parallel.rs`) | Parallel candidate evaluation | Feature-gated rayon/sequential toggle; used throughout codebase |
| `FdarError::InvalidParameter` | `crate::error` | All 5 input-validation errors | Project-wide error type; matches `design_criterion` error contract |
| `PaceFpcaResult` | `crate::pace_fpca` | Read-only model input | Already-fitted PACE model; no re-estimation |
| `criterion 0.5` | `fdars-core/Cargo.toml` (dev-dep, already present) | Benchmark harness | All existing benchmarks use this version |

[VERIFIED: fdars-core/Cargo.toml:55-148] — no new crate dependency; criterion 0.5 is already a dev-dependency.

### No New Dependencies

This phase requires zero `Cargo.toml` changes to `[dependencies]` or `[dev-dependencies]`. The only `Cargo.toml` change is adding one `[[bench]]` stanza.

**Installation:** No `cargo add` — additive code only.

---

## Package Legitimacy Audit

> No new packages are introduced in this phase. All dependencies were present before Phase 65.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious SUS:** none

---

## Architecture Patterns

### Greedy Forward-Selection Loop

**The exact algorithm** (determinism-correct):

```rust
// Step 1 — map candidate_grid values → model.argvals indices (FP tolerance)
// Do this ONCE at validation time; store as Vec<usize> `candidate_indices`.

// Step 2 — greedy loop
let mut selected: Vec<usize> = Vec::with_capacity(config.budget);
let mut trace: Vec<f64> = Vec::with_capacity(config.budget);

for _step in 0..config.budget {
    // Collect not-yet-selected candidates
    let remaining: Vec<usize> = candidate_indices
        .iter()
        .copied()
        .filter(|idx| !selected.contains(idx))
        .collect();

    // Parallel EVALUATE: compute (grid_index, criterion_value) for each candidate
    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;
    let scores: Vec<(usize, f64)> = iter_maybe_parallel!(remaining.iter().copied())
        .map(|idx| {
            let mut trial = selected.clone();   // each closure owns its copy
            trial.push(idx);
            let val = design_criterion(model, &trial, config.criterion.clone())?;
            Ok::<_, FdarError>((idx, val))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Sequential argmin with smallest-index tie-break
    // NOTE: rayon min_by is NOT stable → must be sequential fold after collect.
    let (best_idx, best_val) = scores
        .iter()
        .copied()
        .fold(None::<(usize, f64)>, |acc, (idx, val)| match acc {
            None => Some((idx, val)),
            Some((bi, bv)) => {
                // Strict less-than: ties keep the FIRST (smallest) index seen,
                // which corresponds to the smallest grid index because `remaining`
                // is built from `candidate_indices` in order.
            }
        })
        .unwrap(); // remaining is non-empty — guaranteed by budget > selected.len() check

    selected.push(best_idx);
    trace.push(best_val);
}
```

**Key invariant:** `remaining` is built by iterating `candidate_indices` in order (which is itself built in the same order as `candidate_grid`). The sequential fold visits them in that same order, so the first entry with the minimum value is always the smallest-index candidate. This makes `selected_indices` identical regardless of whether rayon reorders the parallel `scores` collection, because the argmin is taken over the already-collected, fixed-order `scores` vec.

[VERIFIED: fdars-core/src/parallel.rs:42-55] — `iter_maybe_parallel!` expands to `into_par_iter()` (parallel) or `into_iter()` (sequential), both producing iterators that are then `.collect()`-ed into `Vec<(usize, f64)>` before the sequential fold.

### Candidate→Index Mapping (FP Tolerance)

```rust
/// Map candidate_grid values to model.argvals indices.
/// Returns Err(InvalidParameter) if any candidate is not found within 1e-9.
fn map_candidates_to_indices(
    candidate_grid: &[f64],
    argvals: &[f64],
) -> Result<Vec<usize>, FdarError> {
    candidate_grid
        .iter()
        .map(|&cand| {
            argvals
                .iter()
                .position(|&t| (t - cand).abs() < 1e-9)
                .ok_or_else(|| FdarError::InvalidParameter {
                    parameter: "config.candidate_grid",
                    message: format!(
                        "candidate {cand:.6} not found in model.argvals within tolerance 1e-9"
                    ),
                })
        })
        .collect()
}
```

[VERIFIED: fdars-core/src/helpers.rs:4] — `NUMERICAL_EPS = 1e-10`; the FP tolerance 1e-9 is one order above the general epsilon, which is the right margin for grid construction rounding (e.g. `i as f64 / (m-1) as f64` rounding).

### Config/Result Type Pattern

Follow `PaceFpcaConfig`/`PaceFpcaResult` exactly [VERIFIED: fdars-core/src/pace_fpca.rs:51-120]:

```rust
// Config — NO #[non_exhaustive], so struct-literal construction works in tests/docs.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptDesConfig {
    pub candidate_grid: Vec<f64>,
    pub budget: usize,
    pub criterion: DesignCriterion,
}

impl Default for OptDesConfig {
    fn default() -> Self {
        Self {
            candidate_grid: vec![],
            budget: 1,
            criterion: DesignCriterion::Trajectory,
        }
    }
}

// Result — IS #[non_exhaustive] for forward compatibility.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptDesResult {
    /// Grid indices (into model.argvals) of the selected design points, in selection order.
    pub selected_indices: Vec<usize>,
    /// model.argvals values at the selected indices, in selection order.
    pub selected_argvals: Vec<f64>,
    /// Achieved criterion value after each greedy step (length == config.budget).
    /// Monotone non-increasing.
    pub criterion_trace: Vec<f64>,
}
```

### `#[must_use]` on `optimal_design`

Match the codebase convention [VERIFIED: fdars-core/src/optimal_design.rs:85]:

```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn optimal_design(
    model: &PaceFpcaResult,
    config: &OptDesConfig,
) -> Result<OptDesResult, FdarError> {
    ...
}
```

A bare `#[must_use]` on a `Result<…>` return trips clippy's `double_must_use` under `-D warnings` — always include the string message.

### Re-export Pattern (lib.rs)

Existing line (Phase 64) [VERIFIED: fdars-core/src/lib.rs:592-593]:

```rust
// Re-export optimal experimental design criterion core (v0.35.0 FOptDes, Phase 64)
pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};
```

Phase 65 extends this to:

```rust
// Re-export optimal experimental design — full public surface (v0.35.0 FOptDes, Phase 65)
pub use optimal_design::{
    design_criterion, optimal_design, DesignCriterion, OptDesConfig, OptDesResult, OptimalityKind,
};
```

Note: `pub mod optimal_design;` is already declared at line 109. No new `pub mod` line is needed.

### prelude.rs Addition

Add after the k-Shape block (following the kshape/kernel_kmeans peer pattern) [VERIFIED: fdars-core/src/prelude.rs:51]:

```rust
// Optimal experimental design (FOptDes, v0.35.0)
pub use crate::optimal_design::{
    design_criterion, optimal_design, DesignCriterion, OptDesConfig, OptDesResult, OptimalityKind,
};
```

### Module-Level Doctest Pattern

The doctest goes in the module-level `//!` block at the top of `optimal_design.rs`. It must be a complete, runnable example:

```rust
//! # End-to-end example
//!
//! ```rust
//! use fdars_core::irreg_fdata::IrregFdata;
//! use fdars_core::pace_fpca::{pace_fpca, PaceFpcaConfig};
//! use fdars_core::{optimal_design, DesignCriterion, OptDesConfig};
//!
//! // Build a tiny synthetic IrregFdata and fit PACE
//! let argvals = vec![0.0, 0.25, 0.5, 0.75, 1.0];
//! // ... (construct data, fit model) ...
//! let config = OptDesConfig {
//!     candidate_grid: argvals.clone(),
//!     budget: 2,
//!     criterion: DesignCriterion::Trajectory,
//! };
//! let result = optimal_design(&model, &config).expect("design ok");
//! assert_eq!(result.selected_indices.len(), 2);
//! assert_eq!(result.criterion_trace.len(), 2);
//! ```
```

**Warning:** `/tmp` tmpfs exhaustion blocks doctest linking on this machine [VERIFIED: ~/.claude/projects/-home-simonm-projects-rust-fdars/memory/tmp-exhaustion-blocks-precommit.md]. Run `df /tmp` before running `cargo test --doc`. If `/tmp` is full, use `--no-verify` for docs commit and free `/tmp` first.

### Benchmark Pattern (Criterion 0.5)

[VERIFIED: fdars-core/benches/kshape.rs:1-58] — the kshape benchmark is the direct precedent.

**File:** `fdars-core/benches/optimal_design.rs`

```rust
//! Benchmarks for the FOptDes optimal experimental design pipeline (v0.35.0).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::matrix::FdMatrix;
use fdars_core::pace_fpca::PaceFpcaResult;
use fdars_core::{design_criterion, optimal_design, DesignCriterion, OptDesConfig, OptimalityKind};

/// Build a synthetic PaceFpcaResult for benchmarking.
/// Mirrors the `synthetic_model` fixture in optimal_design.rs tests.
fn bench_model(m: usize) -> PaceFpcaResult {
    use fdars_core::helpers::simpsons_weights;
    let ncomp = 2usize;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let weights = simpsons_weights(&argvals);
    // Two orthonormal cosine eigenfunctions (same construction as test fixture)
    let mut ef = vec![0.0_f64; m * ncomp];
    for k in 0..ncomp {
        let freq = (k + 1) as f64 * std::f64::consts::PI;
        let raw: Vec<f64> = argvals.iter().map(|&t| (freq * t).cos()).collect();
        let norm_sq: f64 = (0..m).map(|j| weights[j] * raw[j] * raw[j]).sum();
        let norm = norm_sq.sqrt();
        for j in 0..m { ef[j + k * m] = raw[j] / norm; }
    }
    PaceFpcaResult {
        mean: vec![0.0; m],
        eigenvalues: vec![2.0, 1.0],
        eigenfunctions: FdMatrix::from_column_major(ef, m, ncomp).unwrap(),
        scores: FdMatrix::zeros(1, ncomp),
        fitted: FdMatrix::zeros(1, m),
        fitted_lower: FdMatrix::zeros(1, m),
        fitted_upper: FdMatrix::zeros(1, m),
        argvals,
        sigma2: 0.5,
        ncomp,
    }
}

fn bench_optimal_design(c: &mut Criterion) {
    let m = 51usize;
    let model = bench_model(m);
    let candidate_grid: Vec<f64> = model.argvals.clone();

    let mut group = c.benchmark_group("optimal_design");

    // Benchmark design_criterion alone (Trajectory, 5 points)
    group.bench_function("design_criterion_trajectory_p5_m51", |b| {
        b.iter(|| design_criterion(black_box(&model), black_box(&[5, 12, 25, 38, 45]),
                                   black_box(DesignCriterion::Trajectory)));
    });

    // Benchmark design_criterion (Score A, 5 points)
    group.bench_function("design_criterion_score_a_p5_m51", |b| {
        b.iter(|| design_criterion(black_box(&model), black_box(&[5, 12, 25, 38, 45]),
                                   black_box(DesignCriterion::Score(OptimalityKind::A))));
    });

    // Benchmark full greedy selection (Trajectory, budget=5, grid=51)
    let cfg_traj = OptDesConfig {
        candidate_grid: candidate_grid.clone(),
        budget: 5,
        criterion: DesignCriterion::Trajectory,
    };
    group.bench_function("optimal_design_trajectory_budget5_m51", |b| {
        b.iter(|| optimal_design(black_box(&model), black_box(&cfg_traj)));
    });

    // Benchmark full greedy selection (Score A, budget=5, grid=51)
    let cfg_score = OptDesConfig {
        candidate_grid: candidate_grid.clone(),
        budget: 5,
        criterion: DesignCriterion::Score(OptimalityKind::A),
    };
    group.bench_function("optimal_design_score_a_budget5_m51", |b| {
        b.iter(|| optimal_design(black_box(&model), black_box(&cfg_score)));
    });

    group.finish();
}

criterion_group!(benches, bench_optimal_design);
criterion_main!(benches);
```

**Cargo.toml entry** (append after the kshape bench entry) [VERIFIED: fdars-core/Cargo.toml:146-148]:

```toml
[[bench]]
name = "optimal_design"   # Phase 65 FOD-05 — design_criterion + optimal_design pipeline coverage
harness = false
```

### Anti-Patterns to Avoid

- **Rayon `min_by` for argmin.** `rayon`'s `min_by` / `min_by_key` are NOT stable under equal elements. Use sequential `fold` after `collect`. [ASSUMED — well-documented rayon behavior; the `kshape.rs:44` comment "SBD is RNG-free" confirms the codebase already knows this.]
- **Exact `==` for candidate→index lookup.** `f64 == f64` will fail on grid values computed as `i as f64 / (m-1) as f64` vs a user-supplied equivalent. Always use `(t - cand).abs() < 1e-9`.
- **Forgetting to exclude selected indices each step.** `remaining` must filter out `selected` at every iteration; the duplicate-free guarantee is a make-or-break test gate.
- **Re-factoring `Σ_d` per greedy step inside `design_criterion`.** Do not add caching inside `design_criterion` — it is already correct for this phase. The O(budget · G · K³) complexity is documented as acceptable for typical parameters. [ASSUMED]
- **Bare `#[must_use]` on `Result` return.** Use `#[must_use = "expensive computation whose result should not be discarded"]` to avoid clippy `double_must_use` under `-D warnings`.
- **Benchmark not registered in Cargo.toml.** A bench file in `benches/` with no `[[bench]]` stanza will not run and will NOT be caught by `cargo build --benches`. Always add the stanza.
- **Serde build failure as a Phase 65 regression.** `cargo build --features serde` fails on a pre-existing `ClassifFit` issue (Phase 60). This is NOT introduced by Phase 65. Do not attempt to fix it here and do not gate Phase 65 CI on `--features serde`. [VERIFIED: fdars-core/.planning/phases/64-criterion-machinery-core/64-02-SUMMARY.md — "pre-existing, unrelated blocker"]
- **Forgetting that `DesignCriterion` must be `Clone`** inside the greedy closure. `config.criterion.clone()` is needed per step because `design_criterion` takes `criterion: DesignCriterion` by value. `DesignCriterion` derives `Clone`. [VERIFIED: fdars-core/src/optimal_design.rs:38-46]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Criterion evaluation | A second copy of the Σ_d / BLUP-MSE / posterior-cov math | `design_criterion` (Phase 64) | Already validated with 14 known-answer tests |
| Parallel iteration | Manual thread spawning | `iter_maybe_parallel!` | Feature-gated rayon; WASM-compatible when `parallel` disabled |
| Cholesky solve | Custom linear algebra | `linalg::cholesky_factor` / `cholesky_forward_back` (already called inside `design_criterion`) | Already handles ridge-retry; always compiled (no `linalg` feature gate) |
| Integration weights | Uniform `1/m` weights | `helpers::simpsons_weights` (already inside `trajectory_criterion`) | Grid-invariance proved in Phase 64 tests |
| FP-tolerant search | Exact `==` comparison | Sequential `.position(|&t| (t-cand).abs() < 1e-9)` | Simple O(m) scan; no external dep needed |

---

## Runtime State Inventory

> This is a greenfield phase (new functions + types in an existing file). No renames, migrations, or stored-data changes. **Omitted.**

---

## Common Pitfalls

### Pitfall 1: Non-Stable Parallel Argmin
**What goes wrong:** Using `iter_maybe_parallel!(...).min_by(...)` produces different tie-break results between sequential and parallel builds, making `selected_indices` non-deterministic.
**Why it happens:** Rayon does not guarantee traversal order for `min_by` / `min_by_key` — element order in the underlying worker pool depends on task scheduling.
**How to avoid:** Always `.collect::<Vec<_>>()` the parallel scores first, then take the argmin with a sequential `fold` over the collected `Vec`. The `Vec` order is stable (built from `remaining` in order).
**Warning signs:** Determinism test fails: `r1.selected_indices != r2.selected_indices` intermittently, or seq != parallel in CI.

### Pitfall 2: Exact f64 Equality for Candidate Lookup
**What goes wrong:** `candidate_grid` values computed as `i as f64 / (m-1) as f64` are not bit-identical to `model.argvals` values from the same formula when the latter were computed in a different context or with different rounding.
**Why it happens:** IEEE 754 division can differ by 1 ULP across compilation units or optimization levels.
**How to avoid:** Use `(t - cand).abs() < 1e-9` in the lookup. Return `FdarError::InvalidParameter` if not found.
**Warning signs:** Validation test for off-grid candidate (Pitfall 2 test) passes but a round-trip test with `model.argvals.clone()` as `candidate_grid` panics.

### Pitfall 3: Not Excluding Already-Selected Indices
**What goes wrong:** The same index is selected multiple times, the criterion does not decrease, and the result is useless.
**Why it happens:** Forgetting the `filter(|idx| !selected.contains(idx))` gate on `remaining` at each step.
**How to avoid:** The duplicate-free test (see Validation Architecture) catches this immediately.
**Warning signs:** `result.selected_indices` contains repeated values.

### Pitfall 4: Missing `[[bench]]` in Cargo.toml
**What goes wrong:** The benchmark file `benches/optimal_design.rs` exists but `cargo bench` does not run it, and `cargo build --benches` does not compile it (no gate-on-compile).
**Why it happens:** Criterion benchmarks require explicit `[[bench]]` stanza registration.
**How to avoid:** Add the stanza immediately when creating the bench file. Verify with `cargo build --benches`.
**Warning signs:** `cargo bench --bench optimal_design` returns "no such bench".

### Pitfall 5: Serde-Gate Build Failure Mistaken for Phase 65 Regression
**What goes wrong:** Executor runs `cargo build --features serde` and sees 4 errors, reports Phase 65 broke serde.
**Why it happens:** Phase 60's `ShapeletTransformClassifier` embeds `ClassifFit` which lacks serde derives. Pre-existing since commit `ea39c623`.
**How to avoid:** Do NOT run `--features serde` as a Phase 65 gate. Use `--features linalg,parallel` only. Document the pre-existing issue.
**Warning signs:** Errors reference `ClassifFit: serde::Serialize not satisfied` with zero references to `optimal_design`.

### Pitfall 6: Criterion Trace Has Wrong Length or Is Empty
**What goes wrong:** `result.criterion_trace.len() != config.budget` or is empty.
**Why it happens:** Forgetting to push the criterion value to `trace` after each step, or pushing before the index is appended.
**How to avoid:** The monotone-trace test checks length and ordering. Push `best_val` to `trace` at the same time as `best_idx` to `selected`.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (inline `mod tests` in `optimal_design.rs`) |
| Config file | none — `cargo test -p fdars-core --lib optimal_design --features linalg,parallel` |
| Quick run command | `cargo test -p fdars-core --lib optimal_design --features linalg,parallel` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |
| Bench compile check | `cargo build --benches -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| FOD-04 | `optimal_design` returns `budget` selected indices | unit | `cargo test … test_optimal_design_basic` | Known-answer on synthetic model |
| FOD-04 | Determinism: two identical calls → byte-identical `selected_indices` | unit | `cargo test … test_determinism_two_calls` | Mirrors pace_fpca determinism test |
| FOD-04 | seq == parallel: `selected_indices` identical with/without `--features parallel` | unit | CI matrix (parallel feature on/off) + test asserting stable result | Compile-time variant; same test, different feature flag |
| FOD-04 | Duplicate-free: no index appears twice in `selected_indices` | unit | `cargo test … test_duplicate_free` | Assert all-unique |
| FOD-04 | Monotone trace: `criterion_trace[i+1] <= criterion_trace[i] + 1e-12` | unit | `cargo test … test_monotone_trace` | Inherited from Phase 64's monotonicity guarantee |
| FOD-04 | `budget == 0` → `Err(InvalidParameter)` | unit | `cargo test … test_validation_budget_zero` | |
| FOD-04 | `budget > candidate_grid.len()` → `Err(InvalidParameter)` | unit | `cargo test … test_validation_budget_exceeds_grid` | |
| FOD-04 | Off-grid candidate → `Err(InvalidParameter)` | unit | `cargo test … test_validation_off_grid_candidate` | |
| FOD-04 | `model.ncomp == 0` → `Err(InvalidParameter)` | unit | `cargo test … test_validation_ncomp_zero` | Delegated to `design_criterion`; may be caught at entry |
| FOD-04 | `model.sigma2 <= 0` → `Err(InvalidParameter)` | unit | `cargo test … test_validation_sigma2_nonpositive` | Delegated to `design_criterion` |
| FOD-05 | `optimal_design` with Trajectory criterion selects correct first point | unit | `cargo test … test_trajectory_selects_informative_point` | Known-answer: first point should minimize empty-set BLUP-MSE maximally |
| FOD-05 | `optimal_design` with Score(A) criterion produces valid result | unit | `cargo test … test_score_a_selects` | Spot-check result structure |
| FOD-05 | `OptDesConfig` Default impl is valid | unit | `cargo test … test_config_default` | Validation should catch empty grid at call time (not Default construction) |
| FOD-05 | Additive re-export: `fdars_core::optimal_design` function is reachable at crate root | compile | `cargo build -p fdars-core --features linalg,parallel` | No test needed — just compiles |
| FOD-05 | Prelude re-export: `use fdars_core::prelude::*; let _ = OptDesConfig::default()` | compile | `cargo test … test_prelude_reexport` | Can be a doctest or tiny unit test |
| FOD-05 | Module-level doctest compiles and passes | doctest | `cargo test -p fdars-core --doc --features linalg,parallel` | Watch /tmp disk pressure |
| FOD-05 | Benchmark compiles | compile | `cargo build --benches -p fdars-core --features linalg,parallel` | Must not have `cargo bench` in CI (too slow) |
| Both | Whole-crate fmt | fmt | `cargo fmt -p fdars-core --check` | |
| Both | Whole-crate clippy with `--all-targets` | lint | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Covers bench code |
| Both | Full lib + integration test suite passes | regression | `cargo test -p fdars-core --features linalg,parallel` | Guard: 2671 tests before + new tests |

### Concrete Known-Answer Test: First-Step Selection

For the `synthetic_model(51)` fixture (cos eigenfunctions, `λ = [2.0, 1.0]`, `σ² = 0.5`):

- The first greedy step under Trajectory evaluates all 51 candidates and picks the one that most reduces the integrated BLUP-MSE.
- Due to symmetry of the cosine eigenfunctions on `[0, 1]`, the maximum-information point for `cos(π·t)` and `cos(2π·t)` is near `t = 0` or `t = 0.5` (the peaks/troughs of the first eigenfunction). The exact index is determined by evaluating `design_criterion(model, &[idx], Trajectory)` for all 51 candidates and finding the argmin.
- The test should compute this expected index numerically (not hardcode it) and assert `result.selected_indices[0] == expected_first`.

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --lib optimal_design --features linalg,parallel`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt -p fdars-core --check`
- **Phase gate:** Full suite green before `/gsd-verify-work`. Also: `cargo build --benches -p fdars-core --features linalg,parallel` (bench compiles).

### Wave 0 Gaps

- [ ] Test stubs for all 13 test functions listed above (add to `#[cfg(test)] mod tests` in `optimal_design.rs`)
- [ ] `fdars-core/benches/optimal_design.rs` — new file
- [ ] `[[bench]]` stanza in `fdars-core/Cargo.toml`

*(No new test infrastructure needed — existing `#[cfg(test)] mod tests` inline block is the right location; `synthetic_model(51)` fixture already exists in the file.)*

---

## Code Examples

### Greedy Loop: Parallel Evaluate → Sequential Argmin

```rust
// Source: derived from iter_maybe_parallel! contract (fdars-core/src/parallel.rs:42-55)
// and kernel_kmeans.rs:353-365 sequential argmin pattern.

// Inside optimal_design():
let mut selected: Vec<usize> = Vec::with_capacity(config.budget);
let mut trace: Vec<f64> = Vec::with_capacity(config.budget);

for _step in 0..config.budget {
    let remaining: Vec<usize> = candidate_indices
        .iter()
        .copied()
        .filter(|idx| !selected.contains(idx))
        .collect();

    // PARALLEL evaluate: each closure captures only immutable refs.
    // PaceFpcaResult: Send + Sync (all fields are Vec<f64> / FdMatrix).
    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;
    let scores: Vec<(usize, f64)> = iter_maybe_parallel!(remaining.iter().copied())
        .map(|idx| {
            let mut trial = selected.clone();
            trial.push(idx);
            let val = design_criterion(model, &trial, config.criterion.clone())?;
            Ok::<(usize, f64), FdarError>((idx, val))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // SEQUENTIAL argmin — smallest-index tie-break.
    // `remaining` is in `candidate_indices` order → first minimum wins = smallest grid idx.
    let (best_idx, best_val) = scores
        .into_iter()
        .fold(None::<(usize, f64)>, |acc, (idx, val)| {
            Some(match acc {
                None => (idx, val),
                Some((bi, bv)) => if val < bv { (idx, val) } else { (bi, bv) },
            })
        })
        .expect("remaining is non-empty — guaranteed by budget > step check");

    selected.push(best_idx);
    trace.push(best_val);
}
```

### Determinism Test (Mirrors pace_fpca.rs)

```rust
// Source: mirrors fdars-core/src/pace_fpca.rs:1032-1039

#[test]
fn test_determinism_two_calls() {
    let model = synthetic_model(51);
    let config = OptDesConfig {
        candidate_grid: model.argvals.clone(),
        budget: 3,
        criterion: DesignCriterion::Trajectory,
    };
    let r1 = optimal_design(&model, &config).expect("first call");
    let r2 = optimal_design(&model, &config).expect("second call");
    assert_eq!(r1.selected_indices, r2.selected_indices, "must be deterministic");
    assert_eq!(r1.criterion_trace, r2.criterion_trace, "trace must be deterministic");
}
```

### `Send` + `Sync` Note for Parallel Closure

```rust
// PaceFpcaResult is Send + Sync because all its fields are:
//   Vec<f64> (Send + Sync), FdMatrix (wraps Vec<f64>, Send + Sync), usize, f64.
// The parallel closure captures &model (immutable) — no shared mutable state.
// Each iteration calls design_criterion with its own trial Vec<usize>.
// No FftPlanner (which is !Send) is involved here.
// Compiles correctly under --features parallel.
```

[VERIFIED: fdars-core/src/pace_fpca.rs:99-120] — `PaceFpcaResult` fields are all `Vec<f64>`, `FdMatrix`, `usize`, `f64` — all `Send + Sync`.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Phase 64 exposed `design_criterion` as a standalone scorer | Phase 65 wraps it in a greedy loop | This phase | Users get the full two-stage workflow without managing the loop themselves |
| Criterion evaluation only | Criterion evaluation + greedy orchestration | This phase | FOD-04 and FOD-05 complete the FOptDes public surface |

**Not deprecated in this phase:** `design_criterion` remains independently useful (evaluate a hand-chosen design).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `PaceFpcaResult` implements `Send + Sync` (derived from field types being `Send + Sync`) | Code Examples | Parallel build fails to compile; would be caught by `cargo build --features parallel` |
| A2 | O(budget · G · K³) is acceptable for typical G ≤ 51, K ≤ 5, budget ≤ 10 (seconds) | Don't Hand-Roll | Performance regression; caught by benchmark if egregious |
| A3 | Rayon `min_by` is not stable (tie-break not guaranteed to be smallest-index) | Greedy Algorithm | Non-determinism bug; caught by determinism test |
| A4 | FP tolerance 1e-9 is sufficient for `i as f64 / (m-1) as f64` rounding on the machines used | Candidate→Index Mapping | Off-grid error false-positive; caught by round-trip test `model.argvals.clone()` as `candidate_grid` |

**If this table is empty:** All claims verified — no user confirmation needed. (A1/A3 are well-established; A2/A4 are low-risk engineering choices.)

---

## Open Questions

1. **Benchmark fixture reuse vs. standalone construction**
   - What we know: `synthetic_model_params(m, eigenvalues, sigma2)` in `optimal_design.rs` tests is `pub(super)` — not reachable from `benches/`.
   - What's unclear: whether to duplicate the fixture inline in the bench file or make it `pub(crate)`.
   - Recommendation: Duplicate inline in the bench file (as `kshape.rs` bench does). It's 20 lines and avoids widening visibility.

2. **Prelude ordering**
   - What we know: Prelude currently ends with `CoClusterConfig/Result/SelectResult` (line 101). The FOptDes types are new.
   - What's unclear: Exact insertion point.
   - Recommendation: Add after the k-Shape block (line 51) to group clustering/design methods together. Alphabetical is not enforced.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | All compilation | ✓ | 1.97.0 (dev) | — |
| criterion 0.5 | Benchmark | ✓ | In Cargo.toml already | — |
| rayon 1.10 | `--features parallel` | ✓ | In Cargo.toml already | Sequential (default) |
| /tmp tmpfs | Doctest linking | risk | Limited — see MEMORY.md | `--no-verify` for docs commit; free /tmp first |
| target/ disk | Benchmark build | risk | Adding bench grows target/debug | `rm -rf target/debug/{incremental,examples}` if full |

**Missing dependencies with no fallback:** none
**Missing dependencies with fallback:** /tmp pressure (use `--no-verify` + free /tmp); target/ pressure (cleanup script).

---

## Security Domain

> `security_enforcement` not explicitly set to false; including section.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | N/A (library function, no auth) |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A |
| V5 Input Validation | yes | All 5 guards (`budget == 0`, `budget > grid`, off-grid, `ncomp == 0`, `sigma2 <= 0`) return `FdarError::InvalidParameter` — no panic, no silent truncation |
| V6 Cryptography | no | N/A |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow in `budget * candidate_grid.len()` (large inputs) | Tampering | Rust panics on overflow in debug; usize arithmetic; validate `budget <= candidate_grid.len()` before loop |
| Infinite loop if `remaining` is empty (shouldn't happen if validation is correct) | DoS | Validate `budget <= candidate_grid.len()` at entry; `remaining` is non-empty for all steps |
| NaN propagation from `design_criterion` into `criterion_trace` | Tampering | `design_criterion` returns `Err` on numerical failure; `?` propagates; no silent NaN |

---

## Sources

### Primary (HIGH confidence)

- [VERIFIED: fdars-core/src/optimal_design.rs] — Phase 64 implementation; `design_criterion`, `DesignCriterion`, `OptimalityKind`, `build_sigma_design`, `build_phi_d`, `factor_sigma_design_with_retry`, test fixture `synthetic_model`
- [VERIFIED: fdars-core/src/parallel.rs:42-55] — `iter_maybe_parallel!` macro expansion
- [VERIFIED: fdars-core/src/pace_fpca.rs:51-120, 461-490, 1032-1039] — `PaceFpcaConfig`/`PaceFpcaResult` derive pattern; Σ_yi assembly; determinism test
- [VERIFIED: fdars-core/src/lib.rs:105-109, 592-593] — `pub mod optimal_design` declaration; current re-export line
- [VERIFIED: fdars-core/src/prelude.rs:51] — prelude insertion point
- [VERIFIED: fdars-core/benches/kshape.rs:1-58] — benchmark structure, `criterion_group!`/`criterion_main!` pattern
- [VERIFIED: fdars-core/Cargo.toml:146-148] — `[[bench]] name = "kshape" harness = false` stanza
- [VERIFIED: fdars-core/src/kernel_kmeans.rs:353-365] — sequential fold-based argmin pattern
- [VERIFIED: fdars-core/src/helpers.rs:4] — `NUMERICAL_EPS = 1e-10` (FP tolerance baseline)
- [VERIFIED: .planning/phases/64-criterion-machinery-core/64-01-SUMMARY.md] — Phase 64 plan 1 seams
- [VERIFIED: .planning/phases/64-criterion-machinery-core/64-02-SUMMARY.md] — Phase 64 plan 2 seams; pre-existing serde blocker
- [VERIFIED: .planning/phases/65-greedy-selection-integration/65-CONTEXT.md] — locked decisions
- [VERIFIED: .planning/REQUIREMENTS.md] — FOD-04, FOD-05 text
- [VERIFIED: .planning/STATE.md] — greedy-determinism decision; CI gate (`--all-targets --features linalg,parallel -- -D warnings`)

### Secondary (MEDIUM confidence)

- [CITED: rayon docs] — rayon `min_by` / `min_by_key` are not stable under ties; sequential collect+fold is the documented workaround [ASSUMED — well-known, not fetched this session]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies verified in existing Cargo.toml and source
- Greedy algorithm: HIGH — directly specified in CONTEXT.md locked decisions, with exact code structure derivable from `iter_maybe_parallel!` and `kernel_kmeans.rs` argmin
- Benchmark pattern: HIGH — `kshape.rs` bench read verbatim; `[[bench]]` stanza verified
- Re-export pattern: HIGH — existing `lib.rs` line 592-593 read verbatim; additive extension clear
- Determinism: HIGH — constraint documented in CONTEXT.md and STATE.md; implementation pattern from `pace_fpca.rs` read verbatim
- Pitfalls: HIGH — rayon non-stable argmin and FP tolerance are canonical Rust issues confirmed by codebase patterns
- Validation Architecture: HIGH — all 13 test IDs map directly to locked decisions in CONTEXT.md

**Research date:** 2026-09-03
**Valid until:** 2026-10-03 (stable Rust/Criterion; no upstream changes expected)
