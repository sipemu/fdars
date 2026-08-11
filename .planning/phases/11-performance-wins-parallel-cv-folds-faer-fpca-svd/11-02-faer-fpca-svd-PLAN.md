---
phase: 11-performance-wins-parallel-cv-folds-faer-fpca-svd
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/regression.rs
autonomous: true
requirements: [PERF-02]

estimate:
  tokens: 58000
  raw_tokens: 38000
  tasks: 3
  confidence: med

must_haves:
  truths:
    - "PERF-02: `fdata_to_pc_1d` (regression.rs) computes its SVD via faer `Svd::new_thin(MatRef::from_column_major_slice(weighted.as_slice(), n, m))` on a zero-copy view under `#[cfg(feature = \"linalg\")]`."
    - "PERF-02: faer-path `FpcaResult` matches the nalgebra-path `FpcaResult` within tolerance — significant components (singular_values[k] >= 1e-8 * singular_values[0]) agree within ~1e-8·σ₁ on singular_values, rotation, and scores; near-zero/noise components are excluded."
    - "PERF-02: singular-vector signs are reconciled via a `fix_svd_signs` helper (flip each component so the largest-magnitude element of rotation[:, k] is positive), applied to BOTH cfg branches so the equivalence test is reproducible."
    - "PERF-02: the nalgebra SVD path is retained under `#[cfg(not(feature = \"linalg\"))]` so `\"\"` and `parallel` (non-linalg) builds are unchanged."
    - "Both / build gate: `cargo test -p fdars-core --features linalg` and `cargo clippy -p fdars-core --features linalg -- -D warnings` pass; existing non-linalg FPCA tests still pass under default features."
  artifacts:
    - "fdars-core/src/regression.rs — feature-gated SVD in `fdata_to_pc_1d`; new `fix_svd_signs` helper; new `#[cfg(all(test, feature = \"linalg\"))]` equivalence test."
  key_links:
    - "FdMatrix::as_slice() (already public in matrix.rs) → faer MatRef::from_column_major_slice(weighted.as_slice(), n, m): zero-copy column-major view."
    - "fix_svd_signs(&mut rotation, &mut scores, ncomp) called from BOTH cfg branches BEFORE the sqrt_weights unscaling loop → reproducible sign convention across backends."
    - "faer V is m×k (loadings in columns): rotation[(j,k)] = V[(j,k)] directly (no transpose, unlike nalgebra's v_t[(k,j)])."
  prohibitions:
    - "MUST NOT add any new external dependency to Cargo.toml (faer and nalgebra are already present)."
    - "MUST NOT alter the non-linalg code path's observable output beyond the shared `fix_svd_signs` sign convention — default-feature FPCA tests must still pass."
    - "MUST NOT add a `pub(crate) fn as_slice` to matrix.rs — a public `FdMatrix::as_slice(&self) -> &[f64]` already exists (matrix.rs:291); use it directly."
    - "MUST NOT populate rotation with faer `V[(k,j)]` (transposed) — faer returns V un-transposed; use `V[(j,k)]`."
    - "MUST NOT apply the sign-fix AFTER the sqrt_weights unscaling loop — apply it BEFORE, in both branches."
---

<objective>
Swap the FPCA SVD backend in `fdata_to_pc_1d` to faer `thin_svd` on a zero-copy `MatRef` view under the `linalg` feature (PERF-02), reconcile singular-vector sign conventions with a shared helper, retain the nalgebra path for non-`linalg` builds, and prove numerical equivalence with an inline test.

Purpose: `fdata_to_pc_1d` currently calls `nalgebra::SVD::new(weighted.to_dmatrix(), ...)`, which makes a dense O(n·m) copy before decomposing. faer's `Svd::new_thin` operates on a zero-copy `MatRef::from_column_major_slice` view of the same column-major buffer, measured 1.8–4.1× faster at fdars' real FPCA sizes, eliminating the dominant allocation. SVD sign ambiguity between backends is resolved deterministically so the two paths produce equivalent `FpcaResult`s.

Output: `fdars-core/src/regression.rs` with a feature-gated SVD in `fdata_to_pc_1d`, a `fix_svd_signs` helper applied to both branches, and a `#[cfg(all(test, feature = "linalg"))]` equivalence test.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/11-performance-wins-parallel-cv-folds-faer-fpca-svd/11-RESEARCH.md
</context>

<artifacts_produced>
New symbols this plan introduces (source-grounding drift verification must EXCLUDE these as newly-created):
- Helper fn `fix_svd_signs(rotation: &mut FdMatrix, scores: &mut FdMatrix, ncomp: usize)` in `fdars-core/src/regression.rs`.
- Test fn `test_faer_svd_matches_nalgebra` under `#[cfg(all(test, feature = "linalg"))]` in the existing `#[cfg(test)] mod tests` block of regression.rs.
- New feature-gated use-imports inside the `#[cfg(feature = "linalg")]` SVD block: `faer::linalg::solvers::Svd` (aliased, e.g. `as FaerSvd`) and `faer::MatRef`.

Reused (NOT new — do not re-create): `FdMatrix::as_slice()` (already public at matrix.rs:291), `fdata_to_pc_1d`, `FpcaResult` and its fields, `extract_pc_components`, `center_columns`, `simpsons_weights`, `nalgebra::SVD`, `generate_test_fdata(n, m)` test helper.
</artifacts_produced>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: Feature-gate fdata_to_pc_1d SVD — faer thin_svd (linalg) vs retained nalgebra path</name>
  <files>fdars-core/src/regression.rs</files>
  <read_first>
    - fdars-core/src/regression.rs — the file being modified: top `use` block (lines 1–12, note the existing `#[cfg(feature = "linalg")] use anofox_regression::...` pattern and `use nalgebra::SVD;`); `fdata_to_pc_1d` body (~lines 249–322) including the `weighted` matrix construction, the `SVD::new(weighted.to_dmatrix(), true, true)` call (~line 298), `extract_pc_components`, and the sqrt_weights unscaling loop; the `FpcaResult { singular_values, rotation, scores, mean, centered, weights }` return.
    - fdars-core/src/regression.rs — `extract_pc_components` (~lines 184–210): shows nalgebra maps `v_t[(k,j)] → rotation[(j,k)]` and `u[(i,k)] * sv_k → scores[(i,k)]`.
    - fdars-core/src/matrix.rs — confirm `pub fn as_slice(&self) -> &[f64]` exists at ~line 291 (it does — use it; do NOT add a new accessor); `FdMatrix::zeros(nrows, ncols)`.
    - fdars-core/src/linalg.rs — established faer conversion idiom in this crate for the correct import style and MatRef/Mat usage.
    - .planning/phases/11-performance-wins-parallel-cv-folds-faer-fpca-svd/11-RESEARCH.md — "PERF-02" section: faer 0.23 API (`Svd::new_thin`, `.U()` m×k, `.V()` m×k, `.S().column_vector()`), the "Target faer SVD Path" and "Feature-Gate Structure" excerpts, and Pitfalls 2, 6, 7.
  </read_first>
  <behavior>
    - Under `#[cfg(feature = "linalg")]`: `(singular_values, rotation, scores)` come from `faer::linalg::solvers::Svd::new_thin` on a zero-copy `MatRef` of `weighted`; faer errors map to `FdarError::ComputationFailed`.
    - Under `#[cfg(not(feature = "linalg"))]`: the existing `nalgebra::SVD::new` + `extract_pc_components` path is retained verbatim.
    - Both branches produce the same `(singular_values, rotation, scores)` shape and, after sign reconciliation (Task 2), equivalent values within tolerance.
    - The sqrt_weights unscaling loop and `FpcaResult` construction remain shared/common after the cfg branches.
  </behavior>
  <action>
    Restructure `fdata_to_pc_1d` so the SVD extraction is a cfg-branched expression producing `(singular_values, mut rotation, mut scores)`, keeping everything before it (dimension checks, `center_columns`, `simpsons_weights`, `weighted` scaling) and everything after it (the sqrt_weights unscaling loop and the `FpcaResult { ... }` return) as shared common code. In the `#[cfg(feature = "linalg")]` branch: bring `faer::linalg::solvers::Svd as FaerSvd` and `faer::MatRef` into scope (feature-gated), build `let mat_ref = MatRef::<f64>::from_column_major_slice(weighted.as_slice(), n, m);` (zero-copy — `weighted` is a named local so the borrow is valid for the call), call `FaerSvd::new_thin(mat_ref).map_err(|_| FdarError::ComputationFailed { operation: "SVD (faer)", detail: "faer thin_svd failed; try reducing ncomp or check for zero-variance columns in the data".to_string() })?`, extract `singular_values` from `svd.S().column_vector().iter().take(ncomp).copied().collect()`, populate `rotation: FdMatrix::zeros(m, ncomp)` with `rotation[(j,k)] = svd.V()[(j,k)]` (faer V is m×k un-transposed — do NOT use `V[(k,j)]`, Pitfall 2), and populate `scores: FdMatrix::zeros(n, ncomp)` with `scores[(i,k)] = svd.U()[(i,k)] * singular_values[k]`. In the `#[cfg(not(feature = "linalg"))]` branch: retain the current `let svd = SVD::new(weighted.to_dmatrix(), true, true);` + `extract_pc_components(&svd, n, m, ncomp).ok_or_else(...)?` exactly as-is. Bind both branches to `let (singular_values, mut rotation, mut scores) = { ... };`. Do NOT add any accessor to matrix.rs — `as_slice` already exists. Do NOT touch the sqrt_weights unscaling loop yet (sign-fix insertion is Task 2). Keep `#[must_use]` on the function.
  </action>
  <verify>
    <automated>cargo build -p fdars-core && cargo build -p fdars-core --features linalg && grep -q 'from_column_major_slice(weighted.as_slice()' fdars-core/src/regression.rs && grep -q 'new_thin' fdars-core/src/regression.rs && grep -q 'cfg(not(feature = "linalg"))' fdars-core/src/regression.rs</automated>
  </verify>
  <acceptance_criteria>
    - `cargo build -p fdars-core` exits 0 (nalgebra path compiles under default features).
    - `cargo build -p fdars-core --features linalg` exits 0 (faer path compiles).
    - regression.rs contains `Svd::new_thin` (via the `FaerSvd` alias) and `MatRef::from_column_major_slice(weighted.as_slice(), n, m)`.
    - regression.rs contains `#[cfg(not(feature = "linalg"))]` guarding the retained `SVD::new(weighted.to_dmatrix()` call.
    - matrix.rs is NOT modified (no new `as_slice` added — the existing public one is used).
    - `grep 'V()\[(j' fdars-core/src/regression.rs` present (un-transposed faer V access) and no `V()\[(k,` transposed access.
  </acceptance_criteria>
  <reversibility rating="reversible">Feature-gated backend swap; the nalgebra path is retained under `cfg(not(feature = "linalg"))`, so non-linalg builds are unaffected and the change is a two-way door.</reversibility>
  <done>`fdata_to_pc_1d` selects faer `new_thin` on a zero-copy `MatRef` under `linalg` and the retained nalgebra path otherwise; both feature configs compile; matrix.rs is untouched.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add fix_svd_signs helper and apply to both cfg branches before unscaling</name>
  <files>fdars-core/src/regression.rs</files>
  <read_first>
    - fdars-core/src/regression.rs — the Task-1-modified `fdata_to_pc_1d`: the `let (singular_values, mut rotation, mut scores) = { ... };` binding and the sqrt_weights unscaling loop that follows it.
    - fdars-core/src/matrix.rs — `FdMatrix::nrows()`, `FdMatrix::ncols()`, and `[(row, col)]` indexing for the helper's iteration.
    - .planning/phases/11-performance-wins-parallel-cv-folds-faer-fpca-svd/11-RESEARCH.md — "Sign-Convention Reconciliation" section (the exact `fix_svd_signs` algorithm and the largest-magnitude-element rule) and Pitfall 3 (apply BEFORE unscaling, in both branches).
  </read_first>
  <behavior>
    - `fix_svd_signs(rotation, scores, ncomp)`: for each component k, find `j_max` = index of the largest-absolute-value element in column k of `rotation`; if `rotation[(j_max, k)] < 0.0`, negate every element of `rotation[:, k]` and `scores[:, k]`.
    - After calling it on both backends' output, a given matrix yields the same signs regardless of which backend produced the raw decomposition.
    - Existing default-feature FPCA tests continue to pass (nalgebra already applies a compatible convention; the helper only enforces it deterministically).
  </behavior>
  <action>
    Add a module-level helper `fn fix_svd_signs(rotation: &mut FdMatrix, scores: &mut FdMatrix, ncomp: usize)` implementing the largest-magnitude-element sign convention: for each `k in 0..ncomp`, compute `j_max` via `(0..rotation.nrows()).max_by(|&a, &b| rotation[(a,k)].abs().partial_cmp(&rotation[(b,k)].abs()).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0)`; if `rotation[(j_max,k)] < 0.0`, negate all of `rotation[(j,k)]` for `j` and all of `scores[(i,k)]` for `i`. Call `fix_svd_signs(&mut rotation, &mut scores, ncomp);` in `fdata_to_pc_1d` immediately AFTER the cfg-branched `(singular_values, mut rotation, mut scores)` binding and BEFORE the sqrt_weights unscaling loop — a single call site covers both branches since the binding is shared (Pitfall 3). Derive the row counts inside the helper from `rotation.nrows()` (= m) and `scores.nrows()` (= n) so no extra params are needed.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg -q 2>&1 | tail -5 && cargo test -p fdars-core -q 2>&1 | tail -5 && grep -q 'fn fix_svd_signs' fdars-core/src/regression.rs</automated>
  </verify>
  <acceptance_criteria>
    - `grep -c 'fn fix_svd_signs' fdars-core/src/regression.rs` returns 1.
    - `grep -c 'fix_svd_signs(&mut rotation, &mut scores, ncomp)' fdars-core/src/regression.rs` returns 1 (single shared call site, covers both cfg branches).
    - The call site appears BEFORE the sqrt_weights unscaling loop (sign-fix precedes unscaling).
    - `cargo test -p fdars-core` exits 0 (existing default-feature FPCA tests, e.g. `test_fdata_to_pc_1d_basic`, still pass — nalgebra path + sign convention intact).
    - `cargo test -p fdars-core --features linalg` exits 0.
  </acceptance_criteria>
  <reversibility rating="reversible">Pure deterministic post-processing of SVD output shared by both paths; removable without affecting the backend selection.</reversibility>
  <done>`fix_svd_signs` exists, is called once between the SVD binding and the unscaling loop (covering both cfg branches), and all existing FPCA tests pass under both default and `linalg` features.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Add faer-vs-nalgebra FPCA equivalence test under cfg(linalg)</name>
  <files>fdars-core/src/regression.rs</files>
  <read_first>
    - fdars-core/src/regression.rs — the existing `#[cfg(test)] mod tests { ... }` block; the `generate_test_fdata(n, m)` helper (~line 754) and the existing FPCA tests (`test_fdata_to_pc_1d_basic` etc.) for the data-construction and assertion style.
    - fdars-core/src/regression.rs — `fdata_to_pc_1d` (which under `linalg` now runs faer) and `FpcaResult` fields `singular_values: Vec<f64>`, `rotation: FdMatrix`, `scores: FdMatrix`.
    - .planning/phases/11-performance-wins-parallel-cv-folds-faer-fpca-svd/11-RESEARCH.md — "Numerical Equivalence Test Protocol" (significant-component filter `singular_values[k] >= 1e-8 * singular_values[0]`, tolerance `1e-8 * singular_values[0]`) and "Wave 0 Gaps".
  </read_first>
  <behavior>
    - Under `#[cfg(all(test, feature = "linalg"))]`: build test data with `generate_test_fdata`, call `fdata_to_pc_1d` (faer path active), and compute a reference nalgebra decomposition inline (reproduce the `center_columns` + sqrt_weights scaling + `SVD::new(weighted.to_dmatrix(), true, true)` + `extract_pc_components` + `fix_svd_signs` + unscaling steps) so both go through the identical sign convention.
    - Compare per significant component (where `sv[k] >= 1e-8 * sv[0]`): `|faer_sv[k] - nalgebra_sv[k]| < 1e-8 * sv[0]`, and each `rotation[(j,k)]` and `scores[(i,k)]` within `1e-8 * sv[0]`.
    - Near-zero/noise components (below the threshold) are excluded from comparison.
  </behavior>
  <action>
    Add `#[cfg(all(test, feature = "linalg"))] #[test] fn test_faer_svd_matches_nalgebra()` inside the existing `#[cfg(test)] mod tests` block. In the test: (1) build data via `let (data, t) = generate_test_fdata(n, m);` with moderate n, m (e.g. n=30, m=40) and `ncomp` around 5; (2) call `let faer = fdata_to_pc_1d(&data, ncomp, &t).unwrap();` (faer path under `linalg`); (3) compute the reference the same way the non-linalg branch does — replicate the center/scale/`SVD::new`/`extract_pc_components`/`fix_svd_signs`/unscale sequence inline in the test to produce a reference `(singular_values, rotation, scores)` (call the crate `fix_svd_signs` so both use the identical convention); (4) let `s1 = faer.singular_values[0]`; for each `k` with `faer.singular_values[k] >= 1e-8 * s1`, assert `(faer.singular_values[k] - ref_sv[k]).abs() < 1e-8 * s1`, and for every `j`/`i` assert `(faer.rotation[(j,k)] - ref_rotation[(j,k)]).abs() < 1e-8 * s1` and `(faer.scores[(i,k)] - ref_scores[(i,k)]).abs() < 1e-8 * s1`. Name the test exactly `test_faer_svd_matches_nalgebra`. Keep the dataset modest so the test is fast.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg test_faer_svd_matches_nalgebra</automated>
  </verify>
  <acceptance_criteria>
    - `cargo test -p fdars-core --features linalg test_faer_svd_matches_nalgebra` exits 0.
    - `grep -c 'fn test_faer_svd_matches_nalgebra' fdars-core/src/regression.rs` returns 1.
    - The test is gated `#[cfg(all(test, feature = "linalg"))]` (does not compile/run under default features).
    - The test filters to significant components (`>= 1e-8 * singular_values[0]`) and compares singular_values, rotation, AND scores within `1e-8 * singular_values[0]`.
    - `cargo test -p fdars-core --features linalg` (full suite) exits 0 and `cargo clippy -p fdars-core --features linalg -- -D warnings` exits 0.
  </acceptance_criteria>
  <reversibility rating="reversible">Test-only addition; no production behavior change.</reversibility>
  <done>`test_faer_svd_matches_nalgebra` exists under `#[cfg(all(test, feature = "linalg"))]`, passes, and asserts significant-component equivalence of singular_values, rotation, and scores between the faer and nalgebra paths within `1e-8·σ₁`.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none new) | This change swaps an internal SVD backend behind a feature gate. No new external input, no new public API, no new dependency — faer and nalgebra are already present. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-11-02-01 | Tampering | `fdata_to_pc_1d` FPCA result correctness under the faer backend | low | mitigate | `test_faer_svd_matches_nalgebra` proves the faer path's `FpcaResult` matches the retained nalgebra path within `1e-8·σ₁`; `fix_svd_signs` removes cross-backend sign ambiguity. |
| T-11-02-02 | Denial of Service | faer `new_thin` on degenerate/zero-variance input | low | mitigate | faer's `SvdError` is mapped to `FdarError::ComputationFailed` (never panics), preserving the crate's Result-based error contract. |

Honest assessment: PERF-02 is an internal numerical refactor swapping one SVD library for another behind `#[cfg(feature = "linalg")]`. There is NO new external input, NO new attack surface, and NO new dependency. No other threats apply.
</threat_model>

<verification>
- `cargo test -p fdars-core --features linalg` passes (faer path + equivalence test).
- `cargo test -p fdars-core` passes (retained nalgebra path under default features unchanged).
- `cargo clippy -p fdars-core --features linalg -- -D warnings` passes.
- `grep 'from_column_major_slice(weighted.as_slice()' fdars-core/src/regression.rs` matches; `grep 'new_thin'` matches; `grep 'cfg(not(feature = "linalg"))'` matches.
- matrix.rs is unmodified; no Cargo.toml dependency change.
</verification>

<success_criteria>
`fdata_to_pc_1d` computes its SVD via faer `thin_svd` on a zero-copy `MatRef` under `linalg`, with signs reconciled by a shared `fix_svd_signs` helper applied to both backends; the nalgebra path is retained under `cfg(not(feature = "linalg"))` leaving non-linalg builds unchanged; an inline equivalence test confirms the faer `FpcaResult` matches the nalgebra path within `1e-8·σ₁` on significant components; all linalg tests and clippy pass with no new dependency.
</success_criteria>

<output>
Create `.planning/phases/11-performance-wins-parallel-cv-folds-faer-fpca-svd/11-02-faer-fpca-svd-SUMMARY.md` when done.
</output>
