---
phase: "66"
plan: "01"
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/peer.rs
  - fdars-core/src/lib.rs
autonomous: true
requirements: [PER-01, PER-02]
estimate:
  tokens: 68000
  raw_tokens: 34000
  tasks: 4
  confidence: low
must_haves:
  truths:
    - "peer() recovers a known β(t) within tolerance on synthetic data and returns a PeerResult carrying beta, intercept, fitted_values, effective_df, lambda, penalty_type (PER-01)"
    - "Penalty family is selectable via PeerPenalty; Ridge, Difference{order:2} (reusing penalty_matrix), and caller-supplied Decree(Q) each fit without error (PER-02)"
    - "Decree(Q) with a partition-boundary structure yields a β(t) demonstrably different from the plain Difference{2} roughness fit on the same data (PER-02)"
    - "Wrong-dimension Q (or invalid penalty input) returns a descriptive FdarError, never panics; no NaN β(t) across all three families (PER-01, PER-02)"
  artifacts:
    - "fdars-core/src/peer.rs (new file: peer, PeerConfig, PeerResult, PeerPenalty + inline #[cfg(test)] mod tests)"
    - "fdars-core/src/lib.rs (adds `pub mod peer;`)"
  key_links:
    - "peer.rs imports simpsons_weights (helpers), cholesky_factor/cholesky_forward_back/cholesky_solve (linalg), penalty_matrix (function_on_scalar) — all pub(crate)/pub, no visibility change"
    - "`pub mod peer;` in lib.rs makes the module reachable for tests (crate-root/prelude re-exports DEFERRED to Phase 68)"
---

<objective>
Deliver a public `peer()` estimator for structured-penalty scalar-on-function
regression in a single new file `fdars-core/src/peer.rs`, registered via
`pub mod peer;` in `lib.rs`. The estimator fits β(t) pointwise on the argvals
grid via penalized normal equations `(W_c'W_c + λQ)β = W_c'y_c` solved by
Cholesky, where the penalty Q is chosen from three families: `Ridge` (identity),
`Difference { order: 2 }` (2nd-difference roughness, reusing `penalty_matrix`),
and `Decree(Q)` (caller-supplied raw matrix). Returns `PeerResult` carrying
β(t), intercept (= ȳ), fitted values, effective df (trace of hat matrix), λ, and
the penalty type.

Purpose: PEER is the estimator that distinguishes structured-penalty regression
from plain FPCR — it lets a caller inject a-priori signal structure through Q.
This phase ships the core fixed-λ estimator; auto λ-selection (Phase 67) and
lpeer/predict/exports (Phase 68) build on this contract.

Output: `fdars-core/src/peer.rs` (new symbols `peer`, `PeerConfig`, `PeerResult`,
`PeerPenalty` with variants `Ridge`, `Difference { order }`, `Decree`) plus the
`pub mod peer;` registration line in `lib.rs`.
</objective>

<artifacts_this_phase_produces>
New symbols (all in `fdars-core/src/peer.rs`):
- `peer(data: &FdMatrix, y: &[f64], argvals: &[f64], config: &PeerConfig) -> Result<PeerResult, FdarError>` — public entry point.
- `PeerConfig` — builder struct: `{ penalty: PeerPenalty, lambda: f64 }`. `Default` = `{ penalty: Difference { order: 2 }, lambda: 1.0 }`. Serde behind `serde` feature.
- `PeerResult` — result struct: `{ beta: Vec<f64>, intercept: f64, fitted_values: Vec<f64>, effective_df: f64, lambda: f64, penalty_type: PeerPenalty }`. Derives `Debug, Clone, PartialEq`; `#[non_exhaustive]`; `#[must_use]`; serde behind `serde` feature.
- `PeerPenalty` — enum: `Ridge`, `Difference { order: usize }`, `Decree(Vec<f64>, usize)` (flat row-major Q + dimension p). Default `Difference { order: 2 }`. Derives `Debug, Clone, PartialEq`; serde behind `serde` feature.

New file: `fdars-core/src/peer.rs`.
Registration line: `pub mod peer;` added to `fdars-core/src/lib.rs` (between `pub mod outliers;` and `pub mod regression;`).

DEFERRED (do NOT add this phase): crate-root `pub use` re-exports, prelude entry, module doctest, `lpeer`, `predict` — all Phase 68.
</artifacts_this_phase_produces>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/66-core-peer-estimator-penalty-families/66-CONTEXT.md
@.planning/phases/66-core-peer-estimator-penalty-families/66-RESEARCH.md
@.planning/phases/66-core-peer-estimator-penalty-families/66-VALIDATION.md

# Reuse map — read these signatures before implementing (do NOT re-derive):
@fdars-core/src/function_on_scalar.rs   # penalty_matrix (line 99, pub(crate)); penalized_solve (121); compute_trace_hat (401)
@fdars-core/src/linalg.rs               # cholesky_factor (85), cholesky_forward_back (113), cholesky_solve (131) — all pub(crate)
@fdars-core/src/helpers.rs              # simpsons_weights (76, pub)
@fdars-core/src/matrix.rs               # FdMatrix: zeros (84), shape (106), Index/IndexMut mat[(i,j)] column-major
@fdars-core/src/error.rs                # FdarError variants (6-25)
@fdars-core/src/scalar_on_function/mod.rs  # FregreLmResult (65-98) — result-struct shape template
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: End-to-end peer() for a single penalty family — β(t) recovery, one path only</name>
  <files>fdars-core/src/peer.rs, fdars-core/src/lib.rs</files>
  <read_first>
    - `fdars-core/src/helpers.rs:76-105` — `simpsons_weights(argvals: &[f64]) -> Vec<f64>` (pub). Weights for ∫X_i(t)ψ(t)dt.
    - `fdars-core/src/function_on_scalar.rs:99-116` — `pub(crate) fn penalty_matrix(m) -> Vec<f64>` (row-major m×m D'D, order-2 only; returns zeros if m<3).
    - `fdars-core/src/linalg.rs:85-134` — `cholesky_factor`, `cholesky_forward_back`, `cholesky_solve` (all pub(crate), row-major inputs; ComputationFailed if diagonal ≤ 1e-12).
    - `fdars-core/src/matrix.rs:50-127` — `FdMatrix::zeros(nrows,ncols)`, `shape()`, `mat[(i,j)]` column-major indexing.
    - `fdars-core/src/scalar_on_function/mod.rs:65-98` — `FregreLmResult` field layout, as the shape template for `PeerResult`.
    - `fdars-core/src/lib.rs:105-113` — module registration region; `pub mod peer;` goes between `pub mod outliers;` and `pub mod regression;`.
    - `fdars-core/src/test_helpers.rs:6` — `uniform_grid(n) -> Vec<f64>` (available inside `#[cfg(test)]` only).
  </read_first>
  <behavior>
    - Test `test_peer_difference_beta_recovery`: build n=50 synthetic curves on an m=40 uniform grid, choose true β(t)=sin(π·t), generate y_i = Σ_j X_i(t_j)·β(t_j)·w_j + small noise (w = simpsons_weights). Fit with `PeerConfig { penalty: Difference { order: 2 }, lambda: 1e-4 }`. Assert max_j |beta[j] − true_beta[j]| < 0.1 AND all fitted_values finite. (Resolves RESEARCH assumptions A1/A2 — the design matrix formulation W[i,j]=X_i(t_j)·w_j.)
    - Test `test_peer_result_shape`: on the same fixture, assert `result.beta.len() == m`, `result.fitted_values.len() == n`, `result.effective_df.is_finite() && result.effective_df > 0.0`, `result.lambda == 1e-4`, `result.penalty_type == PeerPenalty::Difference { order: 2 }`, `result.intercept` equals the mean of `y` within 1e-9.
  </behavior>
  <action>
    Create `fdars-core/src/peer.rs` and add `pub mod peer;` to `lib.rs` (alphabetical slot: after `pub mod outliers;`, before `pub mod regression;`).

    Define the three public types with crate-convention derives:
    - `PeerPenalty` enum — variants `Ridge`, `Difference { order: usize }`, `Decree(Vec<f64>, usize)`. Derive `Debug, Clone, PartialEq`; serde behind `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`. Default via `impl Default for PeerPenalty` returning `Difference { order: 2 }`.
    - `PeerConfig` struct — public fields `penalty: PeerPenalty`, `lambda: f64`. Same derives + serde. `impl Default` returning `{ penalty: PeerPenalty::default(), lambda: 1.0 }`.
    - `PeerResult` struct — public fields `beta: Vec<f64>`, `intercept: f64`, `fitted_values: Vec<f64>`, `effective_df: f64`, `lambda: f64`, `penalty_type: PeerPenalty`. Derive `Debug, Clone, PartialEq`; `#[non_exhaustive]`; `#[must_use = "..."]`; serde behind feature. Document each field with a `///` doc comment (β(t) length m, intercept = centered-response mean, fitted length n, effective_df = trace of hat matrix, lambda used, penalty type used).

    Implement `pub fn peer(data, y, argvals, config) -> Result<PeerResult, FdarError>` wiring the ONE Difference{2} path end-to-end (this is the tracer — the other families are Task 2). Entry validation (full validation is Task 4; here assert the minimum the happy path needs): `let (n, m) = data.shape();` and check `argvals.len() == m`, `y.len() == n`, `n > 0`, `m >= 3` — return `FdarError::InvalidDimension { parameter, expected, actual }` on mismatch. Then:
    1. `let w = simpsons_weights(argvals);` (import from `crate::helpers`).
    2. Weighted design: build `wmat` as `FdMatrix::zeros(n, m)` with `wmat[(i,j)] = data[(i,j)] * w[j]`.
    3. Center: `y_bar = mean(y)`, `yc[i] = y[i] - y_bar`; column means `w_bar[j]`; `wc[(i,j)] = wmat[(i,j)] - w_bar[j]`.
    4. Penalty Q via a private `build_q(m, &config.penalty) -> Result<Vec<f64>, FdarError>` helper — for the tracer, implement ONLY the `Difference { order: 2 } => Ok(penalty_matrix(m))` arm; leave the other arms as `todo!()`/`unimplemented!()` stubs to be filled in Task 2 (functionality gap, not architecture gap). Import `penalty_matrix` from `crate::function_on_scalar`.
    5. `wtw` (row-major m×m, symmetric): `wtw[j*m+k] = Σ_i wc[(i,j)]·wc[(i,k)]`. `wty[j] = Σ_i wc[(i,j)]·yc[i]`.
    6. `a[i] = wtw[i] + lambda*q[i]`; `let beta = cholesky_solve(&a, &wty, m)?;` (import from `crate::linalg`).
    7. Effective df via a private `compute_peer_trace_hat(&wtw, &q, lambda, m, n) -> f64` replicating the `compute_trace_hat` column-solve pattern from `function_on_scalar.rs:401-419` (factor `A=wtw+λQ`, for each j solve `A z = wtw[:,j]`, accumulate `z[j]`; on Cholesky failure fall back to `p as f64`; clamp to `n as f64`).
    8. `fitted_values[i] = y_bar + Σ_j wc[(i,j)]·beta[j]`.
    9. Return `PeerResult { beta, intercept: y_bar, fitted_values, effective_df, lambda: config.lambda, penalty_type: config.penalty.clone() }`.

    Add the inline `#[cfg(test)] mod tests` with the two tests from `<behavior>`, importing `crate::test_helpers::uniform_grid`. Reuse `simpsons_weights` in the fixture to build y so the design formulation is self-consistent. Use a small deterministic noise term (no RNG dependency) so the test is reproducible.

    Do NOT wire the Ridge or Decree arms here, do NOT add crate-root re-exports, do NOT add a module doctest — those are later tasks/phases.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_difference_beta_recovery peer::tests::test_peer_result_shape -- --nocapture</automated>
    <fails_when>non-zero exit, or "test result: FAILED" in output, or the assertion message "Difference beta recovery error" prints a max error ≥ 0.1 (β(t) not recovered → design formulation A1/A2 wrong)</fails_when>
  </verify>
  <acceptance_criteria>
    - `fdars-core/src/peer.rs` exists with public `peer`, `PeerConfig`, `PeerResult`, `PeerPenalty` (all three variants declared).
    - `pub mod peer;` present in `lib.rs` between `outliers` and `regression`.
    - The Difference{2} path fits synthetic data end-to-end and recovers β(t)=sin(π·t) with max abs error < 0.1; all fitted_values finite.
    - `PeerResult` field shape/derives match `<artifacts_this_phase_produces>`; `intercept` == mean(y) within 1e-9.
    - Both tracer tests pass; the two-test command exits zero.
  </acceptance_criteria>
  <done>The single Difference{2} happy path fits, recovers a known β(t) within tolerance, returns a well-formed PeerResult, and is committed. Ridge/Decree arms remain stubbed for Task 2.</done>
  <reversibility rating="one-way">Public `peer`/`PeerConfig`/`PeerResult`/`PeerPenalty` API shape publishes on the v0.36.0 crate tag. Additive and extended (not broken) by Phases 67/68, so no blocking checkpoint — rated for the record only.</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Expand build_q to all three penalty families — Ridge + Decree fit without error</name>
  <files>fdars-core/src/peer.rs</files>
  <read_first>
    - `fdars-core/src/peer.rs` — the `build_q` helper and `peer()` from Task 1 (the arms to fill).
    - `fdars-core/src/error.rs:6-25` — `FdarError::InvalidParameter { parameter, message }` for unsupported Difference order; `InvalidDimension` for Decree size mismatch.
    - `66-RESEARCH.md:730-761` — the `build_q` dispatch reference (Ridge = identity; Difference{2} = penalty_matrix; Difference{other} = InvalidParameter; Decree validates len == m*m and p == m).
  </read_first>
  <behavior>
    - Test `test_peer_ridge_fits`: fit the recovery fixture with `PeerPenalty::Ridge`, `lambda: 1e-4`; assert `Ok`, `beta.len() == m`, all `beta` finite, all `fitted_values` finite.
    - Test `test_peer_decree_fits`: build a symmetric PSD m×m Q (e.g. the same `penalty_matrix(m)` output reused as a caller-supplied Decree matrix), fit with `PeerPenalty::Decree(q, m)`, `lambda: 1e-4`; assert `Ok`, `beta.len() == m`, all finite.
    - Test `test_peer_difference_order_rejected`: fit with `PeerPenalty::Difference { order: 3 }`; assert `Err(FdarError::InvalidParameter { .. })` (matched by variant, not by message text).
  </behavior>
  <action>
    Fill the remaining `build_q` arms (replace the Task-1 stubs):
    - `PeerPenalty::Ridge => ` build a row-major m×m identity: zero vec of len m*m, set `q[i*m + i] = 1.0`.
    - `PeerPenalty::Difference { order: 2 } => Ok(penalty_matrix(m))` (already from Task 1 — keep).
    - `PeerPenalty::Difference { order } => Err(FdarError::InvalidParameter { parameter: "penalty", message: format!("Difference order {order} unsupported; only order 2 is available in this release") })`. (Restrict to order 2 per RESEARCH open-question 2 — `penalty_matrix` only builds order-2.)
    - `PeerPenalty::Decree(q_raw, p_q) => ` validate `*p_q == m && q_raw.len() == m*m`; on mismatch return `FdarError::InvalidDimension { parameter: "penalty", expected: format!("{m}x{m}"), actual: format!("{p_q}x{p_q} ({} elems)", q_raw.len()) }`; else `Ok(q_raw.clone())`. Q is assumed symmetric (row-major == column-major for symmetric matrices); do not transpose.

    No changes to the `peer()` body are needed — it already calls `build_q(m, &config.penalty)?` and propagates the error. Add the three tests from `<behavior>` to the inline test module.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_ridge_fits peer::tests::test_peer_decree_fits peer::tests::test_peer_difference_order_rejected -- --nocapture</automated>
    <fails_when>non-zero exit, or "test result: FAILED" in output (a family fails to fit, or an unsupported Difference order is not rejected with InvalidParameter)</fails_when>
  </verify>
  <acceptance_criteria>
    - All three `PeerPenalty` families construct a valid Q; Ridge and Decree(valid) each produce a finite β(t) with no error (PER-02).
    - `Difference { order }` with order ≠ 2 returns `FdarError::InvalidParameter` (no panic, no zeros-fallback silently used).
    - Decree with `p_q == m` and `q_raw.len() == m*m` fits; the `build_q` stubs from Task 1 are gone.
    - The three-test command exits zero.
  </acceptance_criteria>
  <done>Ridge, Difference{2}, and Decree(valid) all fit without error; unsupported Difference orders are rejected with a descriptive FdarError. Committed.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Decree(Q) partition structure yields β(t) distinct from plain roughness</name>
  <files>fdars-core/src/peer.rs</files>
  <read_first>
    - `fdars-core/src/peer.rs` — the completed `peer()` and `build_q` (Tasks 1–2).
    - `66-CONTEXT.md:57-66` — Decree Q is a caller-supplied raw structured matrix representing a partitioned domain; the partition-awareness gate is a first-class success criterion.
  </read_first>
  <behavior>
    - Test `test_peer_decree_distinct_from_roughness`: on the shared synthetic fixture, fit twice with the same `lambda`: (a) `PeerPenalty::Difference { order: 2 }` and (b) a `PeerPenalty::Decree(q_partition, m)` where `q_partition` encodes a partition boundary (a block-structured / boundary-decoupled penalty — e.g. an identity-like Q that penalizes only the second half of the grid, or a difference operator that is zeroed across a mid-grid boundary so the two halves are penalized independently). Assert the two β(t) vectors differ meaningfully: `max_j |beta_decree[j] - beta_diff[j]| > 1e-3` (a numeric threshold well above float noise), AND both fits are `Ok` with all-finite β(t).
  </behavior>
  <action>
    Construct a partition-aware Decree Q inside the test: start from a mid-grid boundary index `b = m/2`, and build a row-major m×m penalty that treats the two halves independently — for example a second-difference operator whose stencil is NOT applied across the boundary (drop the difference rows that straddle index `b`), or a diagonal Q that penalizes only indices `>= b`. The exact structure is the test author's choice provided it (1) is symmetric, (2) yields a numerically stable Cholesky at the test lambda, and (3) is genuinely different in structure from the full-grid `penalty_matrix(m)`.

    Fit both penalties on the SAME `data`, `y`, `argvals`, and `lambda` (use a moderate lambda such as `1.0` so the penalty visibly shapes β(t), not `1e-4`). Compare the resulting `beta` vectors. Assert the max pointwise difference exceeds `1e-3` and both β(t) are all-finite. This is the ROADMAP success-criterion-3 gate: the structured Q produces a partition-aware β(t) demonstrably different from the plain-roughness fit.

    If the two fits come out numerically identical (difference ≤ 1e-3), the Decree Q is not actually entering the normal equations distinctly — that is a real failure to surface, not a test to weaken.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_decree_distinct_from_roughness -- --nocapture</automated>
    <fails_when>non-zero exit, or "test result: FAILED" in output, or the printed max pointwise |β_decree − β_diff| is ≤ 1e-3 (structured Q did not distinctly shape β(t))</fails_when>
  </verify>
  <acceptance_criteria>
    - A partition-structured Decree Q and the plain Difference{2} penalty, fit on identical data at identical lambda, produce β(t) vectors differing by > 1e-3 max pointwise (PER-02, ROADMAP criterion 3).
    - Both β(t) are all-finite (no NaN from the structured penalty).
    - The test exits zero.
  </acceptance_criteria>
  <done>The Decree(Q) partition penalty yields a β(t) demonstrably distinct from the roughness fit on the same data; test passes and is committed.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 4: Error/NaN surface — wrong-dim Q descriptive error, no panic, no NaN across families</name>
  <files>fdars-core/src/peer.rs</files>
  <read_first>
    - `fdars-core/src/peer.rs` — `peer()` entry validation and `build_q` (Tasks 1–3).
    - `fdars-core/src/error.rs:6-25` — `FdarError::InvalidDimension { parameter, expected, actual }` field types (`parameter: &'static str`, `expected: String`, `actual: String`).
    - `66-RESEARCH.md:614-639` — Pitfalls 2–5 (Decree dim mismatch, non-PSD Q, NaN when lambda=0 rank-deficient, storage convention).
  </read_first>
  <behavior>
    - Test `test_peer_decree_wrong_dim`: fit with `PeerPenalty::Decree(vec![0.0; (m-1)*(m-1)], m-1)` (Q sized m-1 ≠ m); assert `Err(FdarError::InvalidDimension { .. })`, matched by variant — and the call returns (does not panic).
    - Test `test_peer_argvals_mismatch`: call `peer` with `argvals` of length `m+1`; assert `Err(FdarError::InvalidDimension { .. })`.
    - Test `test_peer_no_nan_all_families`: fit the recovery fixture with each of `Ridge`, `Difference { order: 2 }`, and a valid `Decree(penalty_matrix(m), m)` at `lambda: 1.0`; assert for every family `result.beta.iter().all(|v| v.is_finite())` and `result.fitted_values.iter().all(|v| v.is_finite())` and `result.effective_df.is_finite()`.
  </behavior>
  <action>
    Harden the entry validation and guard the solve so no input path panics or yields NaN:
    - Confirm the Task-1 entry checks (`argvals.len() == m`, `y.len() == n`, `n > 0`, `m >= 3`) all return `FdarError::InvalidDimension` with descriptive `expected`/`actual` strings; add any missing check.
    - Decree size mismatch is already caught in `build_q` (Task 2) — the `test_peer_decree_wrong_dim` test exercises that path end-to-end through `peer()` to prove it never panics.
    - NaN guard: after `cholesky_solve`, if any `beta[j]` is non-finite, return `FdarError::ComputationFailed { operation: "peer", detail: "non-finite coefficient (singular penalized system)".into() }` rather than returning NaN β(t). (Cholesky already errors on non-PD; this guards residual numerical NaN per RESEARCH Pitfall 4.)
    - Add the three tests from `<behavior>`. The `test_peer_no_nan_all_families` test is the cross-family finiteness gate (ROADMAP criterion 4).

    Do not weaken any prior assertion. All three families must fit cleanly at lambda=1.0 with finite outputs.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel peer:: -- --nocapture</automated>
    <fails_when>non-zero exit, or "test result: FAILED" in output (a wrong-dim input panicked instead of returning FdarError, or any family produced a non-finite β(t)/fitted value/df)</fails_when>
  </verify>
  <acceptance_criteria>
    - Wrong-dimension Decree Q and mismatched argvals both return `FdarError::InvalidDimension` (never panic) — PER-01/PER-02 criterion 4.
    - Non-finite β(t) is converted to `FdarError::ComputationFailed` rather than returned as NaN.
    - All three penalty families produce finite β(t), fitted_values, and effective_df at lambda=1.0.
    - The full `peer::` module test set exits zero (all tasks' tests green together).
  </acceptance_criteria>
  <done>Invalid penalty/dimension inputs return descriptive FdarError without panicking; no family yields NaN β(t); the whole peer:: test module is green. Committed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller → `peer()` args | Untrusted numeric inputs (data, y, argvals, Q) cross into a pure in-process numerical routine. No network, no filesystem, no untrusted deserialization. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-66-01 | Tampering | `peer()` entry (wrong-dim Q, mismatched argvals/y/n) | low | mitigate | Entry-point dimension validation returns `FdarError::InvalidDimension`; never panics (Task 4). |
| T-66-02 | Denial of Service | Cholesky solve on rank-deficient / non-PSD Q | low | mitigate | `cholesky_factor` errors on non-PD → `FdarError::ComputationFailed`; post-solve NaN guard converts non-finite β(t) to `ComputationFailed` (Task 4). |
| T-66-03 | Information Disclosure | error messages | low | accept | Error strings carry only dimension metadata (no secrets); pure library, no sensitive state. |

No external API/SDK/service integrated (declared in `${PHASE_DIR}/COVERAGE.md`). No high-severity threats; ASVS L1 input-validation mitigations in place (dimension checks at entry, NaN-guarded β(t), never-panic contract).
</threat_model>

<verification>
- Per task: `cargo test -p fdars-core --features linalg,parallel peer:: -- --nocapture`
- Wave/phase gate (before /gsd-verify-work):
  - `cargo test -p fdars-core --features linalg,parallel` (full suite green)
  - `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (CI gate — must use --all-targets, per MEMORY ci-clippy-all-targets-gate)
  - `cargo fmt -- --check` (per MEMORY noverify-commits-leave-fmt-drift — run `cargo fmt` before committing)
- Commit with `git commit --no-verify` if the pre-commit hook stalls on a long build / full /tmp (per MEMORY tmp-exhaustion / executor-subagents-stall), then run fmt+clippy out of band.
</verification>

<success_criteria>
1. `peer()` recovers β(t)=sin(π·t) within max abs error < 0.1 on synthetic data and returns a well-formed `PeerResult` (beta, intercept, fitted_values, effective_df, lambda, penalty_type). [PER-01]
2. Ridge, Difference{2} (via `penalty_matrix`), and Decree(valid) each fit without error; unsupported Difference orders return `FdarError::InvalidParameter`. [PER-02]
3. A partition-structured Decree Q yields a β(t) differing by > 1e-3 max pointwise from the plain Difference{2} fit on the same data. [PER-02 / criterion 3]
4. Wrong-dimension Q and mismatched argvals return descriptive `FdarError::InvalidDimension` (never panic); no NaN β(t)/fitted/df across all three families. [criterion 4]
5. `fdars-core/src/peer.rs` exists; `pub mod peer;` is registered in `lib.rs`; crate-root/prelude re-exports and doctest remain DEFERRED to Phase 68.
6. Full suite green; `cargo clippy --all-targets --features linalg,parallel -- -D warnings` and `cargo fmt -- --check` pass.
</success_criteria>

<output>
Create `.planning/phases/66-core-peer-estimator-penalty-families/66-01-SUMMARY.md` when done.
</output>
