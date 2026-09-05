---
phase: 73-veesa-explainability-pipeline-integration
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/elastic_pfi.rs
  - fdars-core/src/jfpca_model.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [VEE-03, VEE-04, VEE-05]

estimate:
  tokens: 78000
  raw_tokens: 39000
  tasks: 5
  confidence: med

must_haves:
  truths:
    - "elastic_pfi is deterministic under seed: two runs with the same seed return byte-identical importance (VEE-03)"
    - "On a known-signal design where one PC carries the response, that PC's importance ranks strictly above every noise PC (VEE-03)"
    - "principal_directions at c=0 reproduces the jfPCA mean: amplitude curve == karcher_mean within 1e-10 and phase is the identity warp (VEE-04)"
    - "Reconstruction scales amplitude perturbation by sigma_j = eigenvalues[j].sqrt() (std-dev), NOT the raw eigenvalue (VEE-04)"
    - "PrincipalDirections returns amplitude_curves and phase_curves each shaped (n_c, m) (VEE-04)"
    - "veesa_pipeline ties fit -> score_training -> elastic_pfi end-to-end and runs as a module doctest under cargo test --doc (VEE-05)"
    - "All changes are additive and non-breaking: existing public signatures unchanged; new items re-exported from lib.rs and prelude.rs (VEE-05)"
  artifacts:
    - "fdars-core/src/elastic_pfi.rs — elastic_pfi, PfiMetric, ElasticPfiResult, veesa_pipeline, VeesaPipelineResult + inline tests + module doctest"
    - "fdars-core/src/jfpca_model.rs — JfpcaModel::principal_directions + PrincipalDirections + inline tests"
    - "fdars-core/src/lib.rs — pub mod elastic_pfi; + re-export block"
    - "fdars-core/src/prelude.rs — prelude re-export line"
  key_links:
    - "elastic_pfi consumes model.score_training().scores (Phase 72 seam) as the PFI input matrix"
    - "principal_directions reads model.eigenvalues / mean_q / vert_component / mean_psi / horiz_component and calls srsf_inverse + exp_map_sphere + psi_to_gam"
    - "elastic_pfi reuses shuffle_global + clone_scores_matrix from explain::helpers WITHOUT modifying them"
    - "veesa_pipeline delegates to jfpca_fit (Phase 72) + elastic_pfi"
---

<objective>
Build the VEESA explainability layer on top of Phase 72's `JfpcaModel` fit/transform seam:
model-agnostic permutation feature importance (PFI) over jfPCA PC scores (VEE-03),
principal-direction reconstruction split into amplitude and phase parts (VEE-04), and a
cohesive end-to-end `veesa_pipeline` with full crate-root + prelude re-exports and a running
module doctest (VEE-05).

Purpose: Completes the VEESA pipeline started in Phase 72 — matches the R `compute_pfi` +
principal-direction visualization design, generic over any predictor so external models
(random forest trained outside the crate, etc.) can be explained.

Output: A new additive module `elastic_pfi.rs`, an additive `impl JfpcaModel` method
`principal_directions`, and additive re-exports. Zero changes to existing public signatures
(protects R/WASM bindings + 28 examples). No new crate dependency.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md

## Build / Test / Gate hazards (READ — from MEMORY.md + Phase 72 SUMMARY)
- Pre-commit hook runs the FULL suite and TIMES OUT on long fdars builds. Run all gates
  OUT-OF-BAND (manually), then `git commit --no-verify`. Run `cargo fmt` per commit
  (`--no-verify` commits otherwise leave fmt drift that fails CI fmt-check).
- CI clippy uses `--all-targets` (lints test/bench code). A plain `-p ... -D warnings` gives a
  FALSE GREEN. Always gate with `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- `clippy::int_plus_one` fires on `assert!(x <= n - 1)` in tests — write `assert!(x < n)`.
- Disk pressure: if a full build dies at example LINK ("linking with cc failed") or "No space
  left", free space with `rm -rf target/debug/{incremental,examples}` (frees ~100GB) — not a code bug.
- serde feature build is PRE-EXISTING broken (shapelet/ClassifFit, Phase 60). New structs must
  only embed serde-clean types; `PfiMetric` must NOT derive serde (its `Custom` variant holds a `Box<dyn Fn>`).
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/73-veesa-explainability-pipeline-integration/73-RESEARCH.md
@.planning/phases/72-jfpca-fit-transform-seam/72-01-SUMMARY.md
@fdars-core/src/jfpca_model.rs
</context>

<artifacts_this_phase_produces>
Every NEW public symbol introduced by this phase (all re-exported from `lib.rs` + `prelude.rs`):

- `elastic_pfi` — fn (VEE-03): `elastic_pfi(scores: &FdMatrix, y: &[f64], predict: impl Fn(&FdMatrix) -> Vec<f64>, metric: &PfiMetric, n_repeats: usize, seed: u64) -> Result<ElasticPfiResult, FdarError>`
- `PfiMetric` — enum (VEE-03): `{ Mse, Mae, Accuracy, Custom(Box<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>) }`. `#[non_exhaustive]`. NO serde derive (Custom is non-serializable).
- `ElasticPfiResult` — struct (VEE-03): `{ importance: Vec<f64>, baseline_metric: f64, permuted_metric: Vec<f64> }`. `#[derive(Debug, Clone, PartialEq)]`, `#[non_exhaustive]`, conditional serde.
- `JfpcaModel::principal_directions` — method (VEE-04): `fn principal_directions(&self, pc_index: usize, c_values: &[f64]) -> Result<PrincipalDirections, FdarError>`
- `PrincipalDirections` — struct (VEE-04): `{ pc_index: usize, c_values: Vec<f64>, amplitude_curves: FdMatrix, phase_curves: FdMatrix }`. `#[derive(Debug, Clone, PartialEq)]`, `#[non_exhaustive]`, conditional serde.
- `veesa_pipeline` — fn (VEE-05): fit -> score_training -> elastic_pfi convenience wrapper.
- `VeesaPipelineResult` — struct (VEE-05): `{ model: JfpcaModel, training_scores: JfpcaTransform, pfi: ElasticPfiResult }`. `#[derive(Debug, Clone, PartialEq)]`, `#[non_exhaustive]`, conditional serde (all embedded types are serde-clean).
- Re-exports: `lib.rs` — `pub mod elastic_pfi;` (after line 141) + `pub use elastic_pfi::{elastic_pfi, veesa_pipeline, ElasticPfiResult, PfiMetric, VeesaPipelineResult};` (after line 509); `PrincipalDirections` re-exported from `jfpca_model` alongside the existing jfPCA block. `prelude.rs` — a matching `pub use crate::{...}` line after line 75.
</artifacts_this_phase_produces>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end veesa_pipeline tracer — fit -> score_training -> PFI, one happy path, wired + verified</name>
  <files>fdars-core/src/elastic_pfi.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - fdars-core/src/jfpca_model.rs:29-56 (module doctest fixture pattern to mirror), :58-64 (imports), :75-127 (JfpcaModel fields), :336 (score_training signature/return JfpcaTransform)
    - fdars-core/src/explain/helpers/permutation.rs:9-21 (shuffle_global signature — pub(crate), takes &mut FdMatrix, &FdMatrix, k, n, &mut StdRng)
    - fdars-core/src/explain/helpers/projection.rs:40-48 (clone_scores_matrix signature — pub(crate))
    - fdars-core/src/explain/helpers/mod.rs:19,35 (confirm both helpers are re-exported at explain::helpers)
    - fdars-core/src/explain/importance.rs:126-146 (existing FPC PFI advancing-RNG convention to mirror)
    - fdars-core/src/lib.rs:141 (mod decl insertion point), :509 (re-export insertion point)
    - fdars-core/src/prelude.rs:75 (prelude re-export insertion point)
    - 73-RESEARCH.md §"Full elastic_pfi Skeleton" and §"Pattern 3: Pipeline Wrapper"
  </read_first>
  <action>
    Create `fdars-core/src/elastic_pfi.rs`. Bring the whole VEESA PFI + pipeline happy path online end to end in this one task — expansion tasks harden it afterward.

    Define `PfiMetric` (enum: Mse, Mae, Accuracy, Custom(Box<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>)) with `#[non_exhaustive]` and NO serde derive (the Custom variant is non-serializable — per D from CONTEXT: "PfiMetric does NOT get serde"). Define `ElasticPfiResult { importance, baseline_metric, permuted_metric }` and `VeesaPipelineResult { model, training_scores, pfi }`, both with `#[derive(Debug, Clone, PartialEq)]`, `#[non_exhaustive]`, and the conditional serde attr `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`.

    Implement a private `compute_metric(y, pred, metric) -> f64`: Mse = mean squared error, Mae = mean absolute error, Accuracy = fraction where predicted rounds to the true label, Custom = call the caller closure. Higher-is-better convention: importance = baseline_metric - mean_permuted_metric.

    Implement `elastic_pfi(scores, y, predict, metric, n_repeats, seed)`. Import `use crate::explain::helpers::{clone_scores_matrix, shuffle_global};` — call them directly, do NOT modify their bodies. Compute baseline_metric ONCE via `predict(scores)`. Seed a SINGLE `StdRng::seed_from_u64(seed)` and ADVANCE it across the whole component loop (this supersedes the CONTEXT.md `seed + k` note — the RESEARCH finding requires a single advancing RNG matching importance.rs; do NOT reseed per component). For each PC column k: for each of n_repeats, `clone_scores_matrix` then `shuffle_global(&mut perm, scores, k, n, &mut rng)`, `predict(&perm)`, accumulate metric; importance[k] = baseline - mean.

    Implement `veesa_pipeline(data, argvals, ncomp, balance_c, lambda, max_iter, y, predict, metric, n_repeats, seed)`: call `jfpca_fit(...)?`, then `model.score_training()?`, then `elastic_pfi(&training_scores.scores, y, predict, metric, n_repeats, seed)?`; return `VeesaPipelineResult`. Mark `elastic_pfi` and `veesa_pipeline` `#[must_use = "..."]`.

    Wire re-exports: `lib.rs` add `pub mod elastic_pfi;` after line 141 and `pub use elastic_pfi::{elastic_pfi, veesa_pipeline, ElasticPfiResult, PfiMetric, VeesaPipelineResult};` after the jfPCA block at line 509. `prelude.rs` add a matching `pub use crate::{elastic_pfi, veesa_pipeline, ElasticPfiResult, PfiMetric, VeesaPipelineResult};` after line 75.

    Add ONE inline `#[cfg(test)] mod tests` with a single smoke test that builds a small spanning fixture (mirror jfpca_model.rs:29-49 multi-harmonic pattern, n>=8, m>=12), runs `veesa_pipeline` with a trivial closure `|s| (0..n).map(|i| s[(i,0)]).collect()`, `PfiMetric::Mse`, n_repeats=5, seed=42, and asserts `result.pfi.importance.len() == result.model.ncomp` and the pipeline returns Ok. This is the tracer's runnable end-to-end proof.

    Do NOT touch importance.rs, permutation.rs, projection.rs, or any existing signature. Additive only.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --lib --features linalg,parallel elastic_pfi::tests 2>&1 | tail -5</automated>
    <fails_when>output contains "error[" (compile failure) or "test result: FAILED" or the smoke test line is absent from "running N tests"</fails_when>
    <automated>cargo build -p fdars-core --features linalg,parallel 2>&1 | tail -3</automated>
    <fails_when>output contains "error[" or "cannot find" (broken re-export) instead of "Finished"</fails_when>
  </verify>
  <done>elastic_pfi.rs exists with PfiMetric/ElasticPfiResult/VeesaPipelineResult/elastic_pfi/veesa_pipeline; re-exports wired in lib.rs + prelude.rs; the end-to-end smoke test passes and the crate builds with linalg,parallel.</done>
  <reversibility rating="reversible">Public API names are locked in CONTEXT.md; module is additive and can be deleted without touching existing code.</reversibility>
</task>

<task type="auto">
  <name>Task 2: PFI make-or-break gates — seed determinism, known-signal ranking, n_repeats=0 rejection (VEE-03)</name>
  <files>fdars-core/src/elastic_pfi.rs</files>
  <read_first>
    - fdars-core/src/elastic_pfi.rs (the module written in Task 1)
    - fdars-core/src/jfpca_model.rs:416-433 (spanning multi-harmonic fixture pattern — 4 harmonics, distinct per-curve amplitudes; use n>=8, m>=12, n>m for full rank)
    - 73-RESEARCH.md §"Known-Signal Test Design", §"Seed Determinism Test"
    - 73-VALIDATION.md gates 1 and 2
    - .planning memory note: recovery tests need full-rank design — use spanning pseudo-random / multi-frequency curves, NEVER low-rank single-freq sinusoids, and NEVER loosen tolerance to make a low-rank fixture pass
  </read_first>
  <action>
    Extend the inline `#[cfg(test)] mod tests` in elastic_pfi.rs with the VEE-03 behavioral gates. Add entry-validation to `elastic_pfi` first: if `scores.nrows() != y.len()` return `FdarError::InvalidDimension` naming the parameter; if `n_repeats == 0` return `FdarError::InvalidParameter` with a message; if `scores.nrows() < 1` or `scores.ncols() < 1` return the appropriate error. (These are the V5 input-validation controls.)

    Test `pfi_seed_determinism`: fit jfpca on the spanning fixture, run `elastic_pfi` twice with the SAME seed (e.g. 42) and n_repeats=10, assert `result1.importance == result2.importance` (exact Vec equality — both are deterministic from the same seed).

    Test `pfi_known_signal_ranking`: build a spanning full-rank fixture (n>=8, m>=12, n>m; follow jfpca_model.rs:416-433). Fit with ncomp=3. Define the response tied to PC 0 only: `y[i] = scores[(i,0)] * 2.0`. Supply the matching closure `|s| (0..n).map(|i| s[(i,0)] * 2.0).collect()`. Run `elastic_pfi(..., n_repeats=20, seed=42, &PfiMetric::Mse)`. Assert `importance[0] > importance[1]` AND `importance[0] > importance[2]` (informative PC ranks strictly above both noise PCs). Do NOT weaken these strict inequalities — if they fail, the fixture is too low-rank; fix the fixture, never the assertion.

    Test `pfi_rejects_zero_repeats`: assert `elastic_pfi(..., n_repeats=0, ...)` returns `Err(FdarError::InvalidParameter { .. })`.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --lib --features linalg,parallel elastic_pfi::tests 2>&1 | tail -8</automated>
    <fails_when>output contains "test result: FAILED" or any of pfi_seed_determinism / pfi_known_signal_ranking / pfi_rejects_zero_repeats is missing from the run count, or "assertion `left == right` failed" / "importance" assertion panic appears</fails_when>
  </verify>
  <done>elastic_pfi validates its inputs; all three VEE-03 gate tests (seed determinism, known-signal ranking with strict inequalities, zero-repeats rejection) pass.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: principal_directions reconstruction — c=0 mean gate + sigma_j sqrt + shapes + bad-pc rejection (VEE-04)</name>
  <files>fdars-core/src/jfpca_model.rs</files>
  <behavior>
    - c=0 amplitude curve reproduces model.karcher_mean within 1e-10 (make-or-break known-answer gate)
    - c=0 phase curve is the identity warp on the argvals domain (exp_map_sphere returns mean_psi for a zero tangent vector)
    - amplitude perturbation magnitude at c=1 scales with sqrt(eigenvalue), NOT the raw eigenvalue (the c=0 gate does NOT catch a raw-vs-sqrt bug — dedicated test required)
    - amplitude_curves and phase_curves are each shaped (n_c, m)
    - pc_index >= ncomp returns FdarError::InvalidParameter; empty c_values returns an error
  </behavior>
  <read_first>
    - fdars-core/src/jfpca_model.rs:58-64 (existing imports), :75-127 (fields: mean_q length m+1, vert_component ncomp×(m+1), mean_psi length m, horiz_component ncomp×m, eigenvalues length ncomp, argvals length m, karcher_mean length m)
    - fdars-core/src/alignment/srsf.rs:65-76 (srsf_inverse(q, argvals, f0) — pub, integrates q(s)|q(s)| from f0)
    - fdars-core/src/warping.rs:73 (psi_to_gam(psi, time) — pub), :112-124 (exp_map_sphere(psi, v, time) — pub; returns psi unchanged when v_norm < 1e-10)
    - fdars-core/src/alignment/mod.rs:104 (srsf_inverse re-export path: crate::alignment::srsf_inverse)
    - fdars-core/src/elastic_fpca.rs:789-792 (eigenvalue = sv^2/(n-1), i.e. variance sigma^2 -> sigma_j = eigenvalues[j].sqrt())
    - 73-RESEARCH.md §"principal_directions Skeleton", §"VEE-04: Principal-Direction Reconstruction Notes" (exact formulas), Pitfalls 1/2/3
    - 73-VALIDATION.md gates 3, 4, 5
  </read_first>
  <action>
    RED first: add the c=0 known-answer test and the sigma-scaling test to the existing `#[cfg(test)] mod tests` in jfpca_model.rs BEFORE writing the method, so they fail to compile / fail the assertion; then implement to green.

    Add imports at module top: `use crate::alignment::srsf_inverse;` and `use crate::warping::{exp_map_sphere, psi_to_gam};`. Define `PrincipalDirections { pc_index: usize, c_values: Vec<f64>, amplitude_curves: FdMatrix, phase_curves: FdMatrix }` with `#[derive(Debug, Clone, PartialEq)]`, `#[non_exhaustive]`, and conditional serde attr.

    Implement `impl JfpcaModel { pub fn principal_directions(&self, pc_index: usize, c_values: &[f64]) -> Result<PrincipalDirections, FdarError> }`. Validate: `pc_index < self.ncomp` else `InvalidParameter`; `!c_values.is_empty()` else `InvalidParameter`. Let `m = self.argvals.len()`, `n_c = c_values.len()`, `sigma_j = self.eigenvalues[pc_index].sqrt()` (CRITICAL: sqrt — the raw eigenvalue is variance; use the std-dev). Build the normalized `time: Vec<f64> = (0..m).map(|i| i as f64 / (m-1) as f64).collect()` and `domain = self.argvals[m-1] - self.argvals[0]`.

    For each (ci, c): AMPLITUDE — `q_perturbed[l] = self.mean_q[l] + c*sigma_j*self.vert_component[(pc_index, l)]` for l in 0..m; `aug_val = self.mean_q[m] + c*sigma_j*self.vert_component[(pc_index, m)]` (index the m-th augmented column — Pitfall 2); `f0 = aug_val.signum() * aug_val * aug_val`; `amp = srsf_inverse(&q_perturbed, &self.argvals, f0)`; write into amplitude_curves row ci. PHASE — `v_perturbed[l] = c*sigma_j*self.horiz_component[(pc_index, l)]` for l in 0..m; `psi_p = exp_map_sphere(&self.mean_psi, &v_perturbed, &time)`; `gam = psi_to_gam(&psi_p, &time)`; `phase_curves[(ci, l)] = self.argvals[0] + gam[l] * domain` (Pitfall 3: normalized time in, scale back to argvals domain). Return `PrincipalDirections`.

    Re-export `PrincipalDirections` from `lib.rs` (add to the jfPCA re-export block near line 509: `pub use jfpca_model::PrincipalDirections;`) and from `prelude.rs`.

    Tests (inline in jfpca_model.rs mod tests): `principal_directions_c0_mean` — c_values = [-2,-1,0,1,2], assert amplitude_curves row for c=0 matches karcher_mean elementwise within 1e-10 (per RESEARCH §"VEE-04a"). `principal_directions_sigma_sqrt_scaling` — with a nonzero c, assert the amplitude deviation from the mean scales with sqrt(eigenvalue): e.g. verify the perturbation direction magnitude is consistent with sigma_j = eigenvalues[pc].sqrt() and would be wrong (too large) if the raw eigenvalue were used — assert the max abs deviation at c=1 is within a plausible band derived from sqrt(eigenvalue)*max|vert_component row|, and fails the raw-eigenvalue alternative. `principal_directions_shapes` — assert both matrices are (n_c, m). `principal_directions_rejects_bad_pc` — `principal_directions(self.ncomp, &[0.0])` returns Err(InvalidParameter). Use `assert!(x < n)` form to avoid clippy::int_plus_one.

    Do not re-implement SRSF inversion or sphere math — call srsf_inverse / exp_map_sphere / psi_to_gam verbatim.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --lib --features linalg,parallel jfpca_model::tests::principal_directions 2>&1 | tail -10</automated>
    <fails_when>output contains "test result: FAILED", or "c=0 amplitude curve deviates from karcher_mean" panic, or fewer than 4 principal_directions_* tests run, or "error[" compile failure</fails_when>
  </verify>
  <done>JfpcaModel::principal_directions implemented + PrincipalDirections re-exported; c=0 reproduces karcher_mean within 1e-10, phase c=0 is identity, sigma_j=sqrt(eigenvalue) scaling asserted, shapes (n_c,m) verified, bad pc_index rejected — all 4 gate tests pass.</done>
  <reversibility rating="reversible">Additive impl block + struct; deletable without touching existing JfpcaModel behavior.</reversibility>
</task>

<task type="auto">
  <name>Task 4: End-to-end module doctest for veesa_pipeline / elastic_pfi (VEE-05)</name>
  <files>fdars-core/src/elastic_pfi.rs</files>
  <read_first>
    - fdars-core/src/elastic_pfi.rs (module written in Tasks 1-2)
    - fdars-core/src/jfpca_model.rs:29-56 (module doctest pattern — the exact fixture + `# Ok::<(), fdars_core::FdarError>(())` closer to mirror)
    - 73-RESEARCH.md §"Doctest Pattern"
    - 73-VALIDATION.md gate 6
  </read_first>
  <action>
    Add a module-level `//!` doctest at the top of `elastic_pfi.rs` (mirror the jfpca_model.rs:29-56 module doctest style). It must import from the crate root (`use fdars_core::{jfpca_fit, elastic_pfi, PfiMetric};` + `use fdars_core::matrix::FdMatrix;`), build a small spanning multi-frequency fixture (n=8, m=12), fit `jfpca_fit(&data, &argvals, 3, None, 0.0, 10)?`, call `model.score_training()?`, then call `elastic_pfi(&tr.scores, &y, |s: &FdMatrix| (0..n).map(|i| s[(i,0)]).collect(), &PfiMetric::Mse, 5, 42)?`, assert `pfi.importance.len() == model.ncomp`, and close with `# Ok::<(), fdars_core::FdarError>(())`. Also add a compact runnable doctest on `veesa_pipeline`'s doc comment showing the one-call convenience path returning `VeesaPipelineResult` and asserting `result.pfi.importance.len() == result.model.ncomp`.

    Keep the doctests small (n=8, m=12) — doctests link in a tiny /tmp tmpfs; large fixtures risk the /tmp-exhaustion link failure noted in MEMORY.md.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel --doc elastic_pfi 2>&1 | tail -6</automated>
    <fails_when>output shows "test result: FAILED", "0 passed" for the doctest, or a compile error inside the doctest block; passing shows the elastic_pfi doctests in "test result: ok"</fails_when>
  </verify>
  <done>elastic_pfi module doctest and veesa_pipeline doctest both run and pass under `cargo test --doc` for the linalg,parallel features.</done>
</task>

<task type="auto">
  <name>Task 5: Full-crate gates — non-regression suite, clippy --all-targets, fmt (VEE-05)</name>
  <files>fdars-core/src/elastic_pfi.rs, fdars-core/src/jfpca_model.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - .planning/phases/72-jfpca-fit-transform-seam/72-01-SUMMARY.md (Gate Results table — baseline was 2801 tests, 0 failed; clippy + fmt clean)
    - 73-VALIDATION.md gates 7 (additive/non-breaking) and the sampling contract
    - execution_context Build/Test/Gate hazards above (run gates out-of-band, commit --no-verify, fmt per commit)
  </read_first>
  <action>
    Run the full-crate gates OUT-OF-BAND (the pre-commit hook times out — commit with `--no-verify` after gates pass, and run `cargo fmt` before each commit). Confirm additive/non-breaking: the whole existing suite still passes (count grows from 2801 by the new tests, none removed/changed), clippy is clean under `--all-targets` (this lints the new inline tests + doctests too), and fmt is clean.

    If clippy flags `int_plus_one` in any new test assertion, rewrite `assert!(x <= n - 1)` as `assert!(x < n)`. If a full `cargo test` dies at example LINK with "linking with cc failed" / "No space left", free disk (`rm -rf target/debug/{incremental,examples}`) and re-run — that is disk pressure, not a code regression. Do NOT modify any existing public signature to satisfy a gate; the phase is additive-only.
  </action>
  <verify>
    <automated>cargo fmt --check 2>&1 | tail -3</automated>
    <fails_when>output lists any file with a "Diff in" fmt suggestion instead of producing no output / exit 0</fails_when>
    <automated>cargo clippy --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -6</automated>
    <fails_when>output contains "warning:" or "error:" (denied warnings) rather than ending in "Finished"</fails_when>
    <automated>cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -8</automated>
    <fails_when>output contains "test result: FAILED" or any "FAILED" line, or the total passed count is below the Phase 72 baseline of 2801</fails_when>
  </verify>
  <done>fmt clean, clippy `--all-targets` clean under -D warnings, full suite green (>= 2801 tests, 0 failed) — additive changes introduce no regression.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller -> elastic_pfi / principal_directions / veesa_pipeline | Caller-supplied `FdMatrix`, `&[f64]`, scalar params, and a prediction closure cross into pure in-process numerical computation. No network, no filesystem, no untrusted deserialization, no identity/authz. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-73-01 | Denial of Service (bad input) | elastic_pfi / principal_directions entry | low | mitigate | V5 input validation at entry — dimension + parameter checks return `FdarError::InvalidDimension`/`InvalidParameter` (never panic): `scores.nrows()==y.len()`, `n_repeats>=1`, `pc_index<ncomp`, non-empty `c_values` (Tasks 2 & 3). |
| T-73-NA | (all other STRIDE categories) | library crate | n/a | accept | N/A — pure in-process numerical explainability over caller-supplied matrices + a caller closure. No external API/SDK, no identity transition, no ORM/schema, no I/O, no untrusted input, no crypto (StdRng is a determinism tool, not a security primitive). No package installs (no Cargo.toml change). Capability checkpoints do not fire. |
</threat_model>

<verification>
- Per task: the scoped `<automated>` command(s) above (quick, <120s each per VALIDATION sampling rate).
- After the wave (all 5 tasks): full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check` (Task 5) + `cargo test --doc` (Task 4).
- Make-or-break behavioral gates (73-VALIDATION.md): PFI seed-determinism (Task 2), informative-PC ranking (Task 2), c=0 reproduces karcher_mean (Task 3), sigma_j=sqrt(eigenvalue) scaling (Task 3), amplitude/phase split shapes (Task 3), end-to-end doctest (Task 4), additive/non-breaking full suite (Task 5).
- All gates run OUT-OF-BAND; commit `--no-verify` with `cargo fmt` per commit (pre-commit hook times out).
</verification>

<success_criteria>
- VEE-03: `elastic_pfi` is model-agnostic (any `Fn(&FdMatrix) -> Vec<f64>`), deterministic under seed, and ranks the informative PC above noise PCs on a full-rank known-signal design.
- VEE-04: `JfpcaModel::principal_directions` returns amplitude + phase curves shaped (n_c, m); c=0 reproduces `karcher_mean` within 1e-10; amplitude scales with `sqrt(eigenvalue)`.
- VEE-05: `veesa_pipeline` ties fit -> transform -> PFI; all new items re-exported from `lib.rs` + `prelude.rs`; a module doctest runs under `cargo test --doc`.
- Additive/non-breaking: no existing public signature changed; full suite green (>= 2801), clippy `--all-targets` clean, fmt clean.
</success_criteria>

<output>
Create `.planning/phases/73-veesa-explainability-pipeline-integration/73-01-veesa-explainability-pipeline-SUMMARY.md` when done.
</output>
