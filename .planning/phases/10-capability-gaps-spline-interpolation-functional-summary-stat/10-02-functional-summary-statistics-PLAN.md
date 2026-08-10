---
phase: 10-capability-gaps-spline-interpolation-functional-summary-statistics
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/fdata.rs
  - fdars-core/src/lib.rs
autonomous: true
requirements: [FEAT-02]
estimate:
  tokens: 60000
  raw_tokens: 32000
  tasks: 3
  confidence: high
must_haves:
  truths:
    - "functional_variance[j] equals functional_std[j] squared at every evaluation point, within 1e-10 (FEAT-02, Success Criterion 2)"
    - "the diagonal of functional_covariance equals functional_variance pointwise, within 1e-10 (FEAT-02, Success Criterion 2)"
    - "functional_variance and functional_covariance use the Bessel correction (ddof = n-1) so that hand-computed references match (RESEARCH Assumption A1)"
    - "depth_based_median returns the index of the maximum Fraiman-Muniz depth curve (argmax over fraiman_muniz_1d(data, data, true)) (FEAT-02, Success Criterion 2)"
    - "trim_mean with alpha = 0 equals mean_1d pointwise, within 1e-10 (FEAT-02, Success Criterion 2)"
    - "every one of the five functions returns FdarError (never panics) on bad input: n<2 for variance/covariance/std, alpha outside [0,1) for trim_mean, empty/dimension-mismatched data (Success Criterion 3)"
    - "cargo test -p fdars-core --features linalg and cargo clippy -p fdars-core --features linalg pass with all five functions covered (Success Criterion 4)"
  artifacts:
    - "fdars-core/src/fdata.rs — pub fn functional_variance"
    - "fdars-core/src/fdata.rs — pub fn functional_std"
    - "fdars-core/src/fdata.rs — pub fn functional_covariance"
    - "fdars-core/src/fdata.rs — pub fn depth_based_median"
    - "fdars-core/src/fdata.rs — pub fn trim_mean"
    - "fdars-core/src/lib.rs — five new functions added to the fdata re-export block"
  key_links:
    - "functional_variance/std/covariance -> FdMatrix::column(j) + mean_1d (pointwise sample statistics, no integration weights)"
    - "functional_covariance -> center_1d (mean-centered data for the M×M sample covariance)"
    - "depth_based_median / trim_mean -> depth::fraiman_muniz_1d(data, data, true) (per-curve self-depth scores)"
    - "lib.rs re-export -> crate-root visibility of all five functions (mirrors mean_1d/center_1d)"
---

<objective>
Add five public functional descriptive-statistics functions to `fdata.rs`, each accepting an `FdMatrix` and returning `Result<_, FdarError>`: `functional_variance` (pointwise), `functional_std` (pointwise), `functional_covariance` (M×M sample covariance), `depth_based_median` (index of deepest curve), and `trim_mean` (depth-trimmed mean). Re-export all five at the crate root.

Purpose: Close capability gap FEAT-02 (audit EXPL-02) — fdars currently has no functional summary statistics vs scikit-fda's `FDataGrid.var()`, `.std()`, `.cov()`, depth median, and trimmed mean.

Output: Five `pub fn` in `fdata.rs` with inline `#[cfg(test)]` tests verifying cross-consistency (var = std², cov diagonal = var, depth_based_median = argmax depth, trim_mean(alpha=0) = mean_1d) and all input-validation error paths, plus crate-root re-exports in `lib.rs`.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/10-capability-gaps-spline-interpolation-functional-summary-stat/10-RESEARCH.md

@fdars-core/src/fdata.rs
@fdars-core/src/depth/fraiman_muniz.rs
@fdars-core/src/depth/mod.rs
@fdars-core/src/matrix.rs
@fdars-core/src/error.rs
</context>

## Artifacts this phase produces (plan 10-02)

New public symbols introduced by this plan:

| Symbol | File | Kind |
|--------|------|------|
| `functional_variance` | `fdars-core/src/fdata.rs` | `pub fn` → `Result<Vec<f64>, FdarError>` (length m) |
| `functional_std` | `fdars-core/src/fdata.rs` | `pub fn` → `Result<Vec<f64>, FdarError>` (length m) |
| `functional_covariance` | `fdars-core/src/fdata.rs` | `pub fn` → `Result<FdMatrix, FdarError>` (M×M) |
| `depth_based_median` | `fdars-core/src/fdata.rs` | `pub fn` → `Result<usize, FdarError>` (curve index) |
| `trim_mean` | `fdars-core/src/fdata.rs` | `pub fn` → `Result<Vec<f64>, FdarError>` (length m) |
| five re-exports | `fdars-core/src/lib.rs` | added to the `pub use fdata::{...}` block |

Nothing is removed. Existing `mean_1d`, `center_1d` (fdata.rs:166-236) remain.

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: End-to-end pointwise trio — functional_variance + functional_std + functional_covariance, cross-consistency tests green</name>
  <files>fdars-core/src/fdata.rs</files>
  <read_first>
    - `fdars-core/src/fdata.rs:166-236` — the file being modified: `mean_1d` (fdata.rs:166-178) for the pointwise mean pattern, `center_1d` (fdata.rs:211-236) for mean-centered data; study the inline `#[cfg(test)] mod tests` block at the end of the file for placement and style.
    - `fdars-core/src/matrix.rs:50-130` — `FdMatrix::zeros`, `FdMatrix::from_column_major`, `FdMatrix::column(j) -> &[f64]` (zero-copy, length n), `shape()`, `nrows()`, `ncols()`, indexed `[(i,j)]`.
    - `fdars-core/src/error.rs:1-25` — `FdarError::InvalidDimension { parameter, expected, actual }` field types.
    - `10-RESEARCH.md` Pattern 2 (RESEARCH.md:140-169) — pointwise (NOT integration-weighted) sample statistics with ddof = n-1 (Bessel); and Pitfall 4 (RESEARCH.md:262-265) — n<2 guard for Bessel correction; and the anti-pattern "Using integration weights in functional_variance/std" (RESEARCH.md:211).
  </read_first>
  <behavior>
    - Test `functional_variance_equals_std_squared`: for a small hand-built FdMatrix, `functional_std(&d)?[j].powi(2)` equals `functional_variance(&d)?[j]` within 1e-10 at every j.
    - Test `functional_covariance_diagonal_matches_variance`: the diagonal element `cov[(j,j)]` equals `functional_variance(&d)?[j]` within 1e-10 at every j.
    - Test `functional_variance_hand_computed`: for a 2×2-ish fixture with a known mean, the Bessel-corrected variance (divide by n-1) matches the hand-computed reference within 1e-10.
  </behavior>
  <action>
    Implement, each `#[must_use]` and returning `FdarError` on bad input (no panics), the pointwise trio in `fdata.rs`:

    - `pub fn functional_variance(data: &FdMatrix) -> Result<Vec<f64>, FdarError>`: read `(n, m) = data.shape()`; require `n >= 2` else `FdarError::InvalidDimension { parameter: "data", expected: ">= 2 rows", actual: n.to_string() }`. Compute `means = mean_1d(data)`; for each column j, `var[j] = sum_i (data.column(j)[i] - means[j])^2 / (n - 1) as f64` (Bessel correction, ddof = n-1). Return the length-m Vec.
    - `pub fn functional_std(data: &FdMatrix) -> Result<Vec<f64>, FdarError>`: `functional_variance(data)?.iter().map(|v| v.sqrt()).collect()` (delegates so the var = std² identity holds by construction).
    - `pub fn functional_covariance(data: &FdMatrix) -> Result<FdMatrix, FdarError>`: require `n >= 2`; guard `m.checked_mul(m)` against usize overflow (threat T-10-02-04) → `FdarError::InvalidParameter { parameter: "data", .. }` if it overflows. Build mean-centered data via `center_1d(data)`; allocate `FdMatrix::zeros(m, m)`; for each `(j1, j2)`, `cov[(j1,j2)] = sum_i centered[(i,j1)] * centered[(i,j2)] / (n - 1) as f64`. Use `centered.column(j1)` / `column(j2)` slices, NOT `row(i)` allocation (anti-pattern RESEARCH.md:210). Return the M×M FdMatrix.

    Wire these three end-to-end and add the three `<behavior>` tests before expanding — the var/std/cov cross-consistency is the tracer that proves the pointwise architecture. Do NOT apply Simpson's / integration weights anywhere (these are plain sample statistics).
  </action>
  <acceptance_criteria>
    - `fdars-core/src/fdata.rs` contains the exact strings `pub fn functional_variance(`, `pub fn functional_std(`, `pub fn functional_covariance(`.
    - `functional_variance` and `functional_covariance` divide by `(n - 1)` (grep `fdata.rs` shows `(n - 1)` in both; no `simpsons_weights` call inside these functions).
    - Inline tests `functional_variance_equals_std_squared`, `functional_covariance_diagonal_matches_variance`, `functional_variance_hand_computed` exist in the `#[cfg(test)] mod tests` block.
    - `cargo test -p fdars-core --features linalg functional_variance 2>&1` and `... functional_covariance 2>&1` show the tests passing.
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p fdars-core --features linalg functional_var 2>&1 | tail -6 && cargo test -p fdars-core --features linalg functional_covariance 2>&1 | tail -6</automated>
  </verify>
  <done>The pointwise trio is implemented in fdata.rs, compiles, and the var=std² + cov-diagonal=var + hand-computed cross-consistency tests pass.</done>
  <reversibility rating="costly" reason="Public API signatures + ddof=n-1 (Bessel) are published-contract decisions, but ROADMAP Success Criterion 2 already fixes them — the one-way decision is pre-made and locked, so no checkpoint is needed; flagged only so the executor implements the fixed signatures/convention verbatim without redesign."/>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Depth-based duo — depth_based_median + trim_mean via fraiman_muniz_1d, with validation</name>
  <files>fdars-core/src/fdata.rs</files>
  <read_first>
    - `fdars-core/src/depth/fraiman_muniz.rs:32-39` — `fraiman_muniz_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64>`; returns one depth per curve (length `data_obj.nrows()`); larger = more central. Self-depth call is `fraiman_muniz_1d(data, data, true)`.
    - `fdars-core/src/depth/mod.rs:24` — `fraiman_muniz_1d` re-export; import via `use crate::depth::fraiman_muniz_1d;`.
    - `fdars-core/src/fdata.rs:166-178` — `mean_1d` (for the `trim_mean(alpha=0) == mean_1d` identity and for averaging retained curves).
    - `10-RESEARCH.md` Pattern 3 (RESEARCH.md:173-201) — depth_based_median = argmax depth index; trim_mean excludes the `floor(alpha * n)` least-deep curves and averages the rest; Pitfall 5 (RESEARCH.md:267-271) — alpha must be in [0,1).
    - `fdars-core/src/error.rs:1-25` — `InvalidParameter`, `InvalidDimension`, `ComputationFailed` fields.
  </read_first>
  <behavior>
    - Test `depth_based_median_argmax`: for a fixture where one curve is clearly most central, `depth_based_median(&d)?` returns that curve's index (== argmax of `fraiman_muniz_1d(&d, &d, true)`).
    - Test `trim_mean_alpha_zero_equals_mean`: `trim_mean(&d, 0.0)?` equals `mean_1d(&d)` pointwise within 1e-10 (excludes zero curves).
    - Test `trim_mean_rejects_bad_alpha`: `trim_mean(&d, 1.0)` and `trim_mean(&d, -0.1)` each return `Err(FdarError::InvalidParameter { parameter: "alpha", .. })`.
  </behavior>
  <action>
    Implement in `fdata.rs`, each `#[must_use]` and returning `FdarError` (no panics):

    - `pub fn depth_based_median(data: &FdMatrix) -> Result<usize, FdarError>`: read `(n, _) = data.shape()`; require `n >= 1` else `FdarError::InvalidDimension { parameter: "data", expected: ">= 1 row", actual: "0" }`. Compute `depths = crate::depth::fraiman_muniz_1d(data, data, true)`; return the argmax index via `depths.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i)`, mapping the None case to `FdarError::ComputationFailed { operation: "depth_based_median", .. }`. Return `Ok(idx)`.
    - `pub fn trim_mean(data: &FdMatrix, alpha: f64) -> Result<Vec<f64>, FdarError>`: require `alpha >= 0.0 && alpha < 1.0` else `FdarError::InvalidParameter { parameter: "alpha", message: format!("must be in [0, 1), got {alpha}") }`; require `n >= 1` else `InvalidDimension`. Compute self-depths; determine `k = (alpha * n as f64).floor() as usize` curves to drop; select the `n - k` deepest curve indices (sort indices by descending depth); average those curves pointwise into a length-m Vec (reuse the `mean_1d` averaging pattern over the retained subset). alpha=0 drops zero curves so the result equals `mean_1d` exactly.

    Add the three `<behavior>` tests. Assert error variants by matching the `Err(FdarError::…)` shape.
  </action>
  <acceptance_criteria>
    - `fdars-core/src/fdata.rs` contains the exact strings `pub fn depth_based_median(` and `pub fn trim_mean(`.
    - Both delegate to `fraiman_muniz_1d(` (grep `fdata.rs` shows the call in both functions).
    - Inline tests `depth_based_median_argmax`, `trim_mean_alpha_zero_equals_mean`, `trim_mean_rejects_bad_alpha` exist.
    - No `panic!`/`unwrap()`/`expect(` on user-derived values inside either function body (the `partial_cmp(...).unwrap_or(...)` NaN-safe comparator is the only permitted unwrap-family call).
    - `cargo test -p fdars-core --features linalg depth_based_median 2>&1` and `... trim_mean 2>&1` show tests passing.
  </acceptance_criteria>
  <verify>
    <automated>cargo test -p fdars-core --features linalg depth_based_median 2>&1 | tail -6 && cargo test -p fdars-core --features linalg trim_mean 2>&1 | tail -6</automated>
  </verify>
  <done>depth_based_median returns the argmax-depth index and trim_mean(alpha=0)==mean_1d; bad alpha returns FdarError; all three tests pass.</done>
</task>

<task type="auto">
  <name>Task 3: Consolidated validation tests, crate-root re-exports, clippy + full suite</name>
  <files>fdars-core/src/fdata.rs, fdars-core/src/lib.rs</files>
  <read_first>
    - `fdars-core/src/lib.rs` — locate the `pub use fdata::{...}` re-export block (search for `fdata::` and `mean_1d`); add the five new functions there, mirroring how `mean_1d`/`center_1d` are exported (RESEARCH Open Question 3, RESEARCH.md:488-491).
    - `fdars-core/src/fdata.rs` — the five new function bodies and the inline test block from Tasks 1-2.
    - `fdars-core/src/error.rs:1-25` — variant fields for the consolidated validation test.
  </read_first>
  <action>
    1. Add a consolidated inline test `functional_stats_input_validation` in `fdata.rs` covering the remaining never-panic paths not already asserted: `functional_variance` / `functional_std` / `functional_covariance` each return `Err(FdarError::InvalidDimension { .. })` when `n < 2`; `depth_based_median` / `trim_mean` return `Err(FdarError::InvalidDimension { .. })` on an empty (0-row) matrix. Assert by matching the `Err(FdarError::…)` variant shape.
    2. Add all five functions — `functional_variance`, `functional_std`, `functional_covariance`, `depth_based_median`, `trim_mean` — to the `pub use fdata::{...}` block in `lib.rs` (alphabetical among existing identifiers), following the existing `mean_1d`/`center_1d` re-export pattern. Do not remove any existing re-export.
    3. Run the full `linalg` suite and clippy to confirm the additive change is clean and existing `fdata` exports are untouched (Success Criterion 4).
  </action>
  <acceptance_criteria>
    - `fdars-core/src/lib.rs` contains `functional_variance`, `functional_std`, `functional_covariance`, `depth_based_median`, `trim_mean` within the `pub use fdata::{` block; `fdars_core::functional_variance` (and the other four) resolve at the crate root.
    - `grep -n 'mean_1d\|center_1d' fdars-core/src/lib.rs` still shows the pre-existing fdata re-exports present (nothing removed).
    - Inline test `functional_stats_input_validation` exists in `fdata.rs` and covers n<2 and empty-matrix error paths for all five functions.
    - `cargo build -p fdars-core --features linalg` exits 0.
    - `cargo test -p fdars-core --features linalg` exits 0 (full suite green).
    - `cargo clippy -p fdars-core --features linalg` exits 0 with no new warnings on the added code.
  </acceptance_criteria>
  <verify>
    <automated>cargo clippy -p fdars-core --features linalg 2>&1 | tail -5 && cargo test -p fdars-core --features linalg 2>&1 | tail -8</automated>
  </verify>
  <done>All five functions are re-exported at the crate root, the consolidated validation test passes, the full linalg suite is green, and clippy is clean.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller → summary-statistics functions | Untrusted numeric input (matrix dims, alpha) crosses into pure-Rust numeric routines. No network/FS/auth surface. |

## STRIDE Threat Register (ASVS L1)

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-10-02-01 | Denial of Service | functional_variance/std/covariance with n<2 (Bessel divide-by-zero / usize underflow) | low | mitigate | Validate `n >= 2` before dividing by `(n-1)` (Pitfall 4). Task 1 + Task 3 tests. |
| T-10-02-02 | Tampering (data integrity) | trim_mean alpha outside [0,1) → empty average / out-of-range retained count | low | mitigate | Validate `alpha ∈ [0,1)` returning FdarError (Pitfall 5). Task 2. |
| T-10-02-03 | Tampering (data integrity) | depth_based_median / trim_mean on 0-row matrix → empty argmax | low | mitigate | Validate `n >= 1`; None-argmax maps to FdarError::ComputationFailed, never panics. Task 2 + Task 3. |
| T-10-02-04 | Denial of Service | integer overflow in M×M covariance allocation (`m * m`) | low | mitigate | Guard `m.checked_mul(m)` before allocating the covariance matrix (RESEARCH Security Domain, RESEARCH.md:449). Task 1. |
| T-10-02-05 | Tampering | NaN in depth comparison producing nondeterministic argmax | low | accept | Use `partial_cmp(...).unwrap_or(Ordering::Equal)` for a total, deterministic order; residual risk is a well-defined tie-break. |

No new external packages are introduced (Package Legitimacy Audit: none — RESEARCH.md:389-394). No high-severity threats → non-blocking.
</threat_model>

<verification>
- Success Criterion 2: `functional_variance_equals_std_squared`, `functional_covariance_diagonal_matches_variance`, `functional_variance_hand_computed`, `depth_based_median_argmax`, `trim_mean_alpha_zero_equals_mean` all green.
- Success Criterion 3: `trim_mean_rejects_bad_alpha` and `functional_stats_input_validation` green; no panic path in any of the five function bodies.
- Success Criterion 4: `cargo test -p fdars-core --features linalg` and `cargo clippy -p fdars-core --features linalg` both exit 0; existing `mean_1d`/`center_1d` re-exports still present.
</verification>

<success_criteria>
- Five `pub fn` (`functional_variance`, `functional_std`, `functional_covariance`, `depth_based_median`, `trim_mean`) exist in `fdata.rs`, each `#[must_use]`, each returning `Result<_, FdarError>`.
- All five re-exported at the crate root in `lib.rs`.
- Cross-consistency + validation tests pass; full `linalg` suite green; clippy clean.
</success_criteria>

<output>
Create `.planning/phases/10-capability-gaps-spline-interpolation-functional-summary-stat/10-02-SUMMARY.md` when done.
</output>
