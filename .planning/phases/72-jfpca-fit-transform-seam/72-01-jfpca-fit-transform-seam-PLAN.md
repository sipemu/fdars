---
phase: "72"
plan: "01"
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/jfpca_model.rs
  - fdars-core/src/elastic_fpca.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [VEE-01, VEE-02]

estimate:
  tokens: 95000
  raw_tokens: 47500
  tasks: 4
  confidence: med

must_haves:
  truths:
    - "jfpca_fit(curves, argvals, ncomp, balance_c, lambda, max_iter) returns a JfpcaModel whose training scores equal joint_fpca(...) scores element-wise within 1e-8 (VEE-01)."
    - "JfpcaModel stores karcher_mean, mean_q (len m+1), mean_psi (len m), vert_component, horiz_component, balance_c, argvals, eigenvalues, ncomp (clamped), lambda, and an embedded joint_result (VEE-01)."
    - "model.transform(&training_curves).scores reproduces the training scores within 1e-8 (fit->transform round-trip, VEE-02)."
    - "model.transform on curves whose argvals length != trained grid returns FdarError::InvalidDimension (no panic, no silent resample) (VEE-02)."
    - "A running module doctest exercises fit -> transform under cargo test --doc."
    - "All new surface is additive: existing elastic_fpca / joint_fpca signatures unchanged; whole-crate test/clippy --all-targets/fmt pass."
  artifacts:
    - fdars-core/src/jfpca_model.rs
  key_links:
    - "jfpca_fit must run horiz_fpca explicitly to capture mean_psi (joint_fpca discards it) and capture mean_q from center_matrix (joint_fpca discards it)."
    - "transform aligns new curves to model.karcher_mean via align_to_target (NOT a fresh Karcher mean), centers with the trained mean_q, and scores via dot(q_aug_centered, vert_component[k]) + balance_c * dot(shooting, horiz_component[k])."
    - "crate-root (lib.rs) + prelude re-exports of jfpca_fit, JfpcaModel, JfpcaTransform."
---

<objective>
Deliver a public, reusable jfPCA fit -> transform seam over the shipped `elastic_fpca.rs` machinery (VEE-01, VEE-02). `jfpca_fit` trains a `JfpcaModel` on curves whose training scores reproduce `joint_fpca` within 1e-8; `model.transform(&new_curves)` aligns out-of-sample curves to the trained Karcher-mean template and projects them onto the trained joint-FPCA basis in the trained coordinate system, returning `JfpcaTransform { scores, aligned, warping }`. Foundational for Phase 73 (VEESA explainability).

Purpose: expose a fit/transform contract that Phase 73 consumes; additive/non-breaking (protects R + WASM bindings + 28 examples), reuse-first, no new crate dependency.
Output: new `fdars-core/src/jfpca_model.rs`, one `pub(crate)` visibility promotion in `elastic_fpca.rs`, and additive `lib.rs`/`prelude.rs` re-exports.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md

## Build/Gate hazards (from STATE.md / MEMORY.md / VALIDATION.md) — READ BEFORE RUNNING GATES
- The pre-commit hook runs the FULL suite and TIMES OUT on long fdars builds. Run gates OUT-OF-BAND and commit with `git commit --no-verify`.
- After a `--no-verify` commit, `--no-verify` also skips `cargo fmt`; run `cargo fmt` per commit or CI fmt-check fails despite green clippy.
- CI clippy uses `--all-targets` (lints test/bench code): always run `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, never a bare `-p ... -D warnings`.
- Disk pressure: `target/` can fill `/home`; `/tmp` tmpfs can fill and break doctest linking with a bogus "No space left". If a build dies at LINK, `rm -rf target/debug/{incremental,examples}` frees space — it is not a code bug.
- If executor subagents stall on long cargo builds, execute inline and commit `--no-verify` after out-of-band gates pass.
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/72-jfpca-fit-transform-seam/72-CONTEXT.md
@.planning/phases/72-jfpca-fit-transform-seam/72-RESEARCH.md
@.planning/phases/72-jfpca-fit-transform-seam/72-VALIDATION.md

# Primary source of truth for the seam (read before writing any code):
@fdars-core/src/elastic_fpca.rs
@fdars-core/src/alignment/mod.rs
@fdars-core/src/alignment/set.rs
@fdars-core/src/alignment/karcher.rs
</context>

## Artifacts this phase produces

New public symbols (all re-exported at crate root + prelude):
- `struct JfpcaModel` — trained transformer. Fields: `karcher_mean: Vec<f64>`, `mean_q: Vec<f64>`, `mean_psi: Vec<f64>`, `vert_component: FdMatrix`, `horiz_component: FdMatrix`, `balance_c: f64`, `argvals: Vec<f64>`, `eigenvalues: Vec<f64>`, `ncomp: usize`, `joint_result: JointFpcaResult`, `lambda: f64`.
- `struct JfpcaTransform` — transform output. Fields: `scores: FdMatrix`, `aligned: FdMatrix`, `warping: FdMatrix`.
- `fn jfpca_fit(data, argvals, ncomp, balance_c, lambda, max_iter) -> Result<JfpcaModel, FdarError>`.
- method `JfpcaModel::transform(&self, new_curves: &FdMatrix) -> Result<JfpcaTransform, FdarError>`.

Visibility promotion in `elastic_fpca.rs`:
- `build_combined_representation` (line ~914): private `fn` -> `pub(crate) fn`. (All other needed helpers — `build_augmented_srsfs`, `center_matrix`, `warps_to_normalized_psi`, `shooting_vectors_from_psis`, `sphere_karcher_mean` — are already `pub(crate)`. `project_onto_eigenvectors` stays private: the transform uses a direct dot-product formula, NOT that helper.)

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: End-to-end fit -> transform tracer — one path only</name>
  <files>fdars-core/src/jfpca_model.rs, fdars-core/src/elastic_fpca.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - fdars-core/src/elastic_fpca.rs:286-358 (joint_fpca body — the exact training path to reproduce: vert_fpca -> horiz_fpca -> build_augmented_srsfs -> center_matrix -> build_combined_representation -> SVD -> svd_scores_and_eigenvalues -> split_joint_eigenvectors)
    - fdars-core/src/elastic_fpca.rs:53-86 (HorizFpcaResult with mean_psi + shooting_vectors; JointFpcaResult fields)
    - fdars-core/src/elastic_fpca.rs:645-670, 705-725, 728-755, 759-774, 914-... (warps_to_normalized_psi, shooting_vectors_from_psis, build_augmented_srsfs, center_matrix, build_combined_representation)
    - fdars-core/src/alignment/karcher.rs:293-301 (karcher_mean signature: data, argvals, max_iter, tol, lambda)
    - fdars-core/src/alignment/set.rs:51-83 (align_to_target: data, target, argvals, lambda -> AlignmentSetResult{gammas, aligned_data, distances})
    - fdars-core/src/alignment/mod.rs:139-167 (AlignmentSetResult, KarcherMeanResult fields incl. mean, gammas, aligned_data, aligned_srsfs)
    - fdars-core/src/lib.rs:134-138 (pub mod ordering: insert `pub mod jfpca_model;` alphabetically among the elastic_* modules), lib.rs:502 (existing `pub use elastic_fpca::{...}` re-export block to mirror)
    - fdars-core/src/prelude.rs:71 (existing `pub use crate::elastic_fpca::{...}` line to mirror)
    - RESEARCH.md sections 2, 9 (the exact out-of-sample score formula) and the JfpcaModel/JfpcaTransform/jfpca_fit skeletons
  </read_first>
  <action>
    Create `fdars-core/src/jfpca_model.rs` and wire the MINIMAL end-to-end happy path proving the architecture: fit a model, transform curves, get scores. Production-quality on the one path (real error handling, real formula) — stubs allowed only for the round-trip/error-path tests added in later tasks, never for the core computation.

    Promote `build_combined_representation` in `elastic_fpca.rs` from `fn` to `pub(crate) fn` (single visibility edit; do not change its body or any other signature).

    Define `JfpcaModel` and `JfpcaTransform` with the exact fields listed in "Artifacts this phase produces". Derive `Debug, Clone, PartialEq`; add `#[non_exhaustive]`; add `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` per crate convention. Add a `//!` module doc comment and `///` docs on every public item.

    Implement `jfpca_fit(data: &FdMatrix, argvals: &[f64], ncomp: usize, balance_c: Option<f64>, lambda: f64, max_iter: usize) -> Result<JfpcaModel, FdarError>` (exposes lambda default 0.0 and max_iter default 20 per D — researcher recommendation resolving the open questions; caller passes them explicitly). Steps:
      1. Validate at entry mirroring joint_fpca: n>=2, m>=2, ncomp>=1, argvals.len()==m; else FdarError::InvalidDimension.
      2. `let karcher = karcher_mean(data, argvals, max_iter, 1e-4, lambda);`
      3. `let joint_result = joint_fpca(&karcher, argvals, ncomp, balance_c)?;`  (this is what training scores must reproduce — reuse it directly so scores are IDENTICAL, not merely close).
      4. `let horiz = horiz_fpca(&karcher, argvals, ncomp)?;` and keep `horiz.mean_psi` (joint_fpca discards it — this is the make-or-break capture).
      5. Recompute mean_q the same way joint_fpca does: `(n,m)=karcher.aligned_data.shape()`; `qn = srsf_transform(&karcher.aligned_data, argvals)` (or reuse karcher.aligned_srsfs if Some, matching joint_fpca's choice); `q_aug = build_augmented_srsfs(&qn, &karcher.aligned_data, n, m)`; `(_, mean_q) = center_matrix(&q_aug, n, m+1)`. Store `mean_q`.
      6. Store clamped ncomp as `joint_result.eigenvalues.len()` (NOT the user-supplied ncomp — joint_fpca clamps to n-1).
      7. Assemble JfpcaModel (clone vert_component/horiz_component/eigenvalues/karcher.mean out of the results, move joint_result in last).
    Mark `jfpca_fit` `#[must_use = "..."]`.

    Implement `JfpcaModel::transform(&self, new_curves: &FdMatrix) -> Result<JfpcaTransform, FdarError>`. On the tracer path implement the full real formula (do not stub scoring):
      1. `(n_new, m_new) = new_curves.shape()`; if `m_new != self.argvals.len()` return FdarError::InvalidDimension (grid mismatch). Also validate n_new>=1.
      2. `let aln = align_to_target(new_curves, &self.karcher_mean, &self.argvals, self.lambda);` — align to the FIXED trained template; do NOT run a fresh karcher_mean.
      3. psi-space shooting: `let time: Vec<f64> = (0..m).map(|i| i as f64 / (m-1) as f64).collect();` then `let psis = warps_to_normalized_psi(&aln.gammas, &self.argvals);` then `let shooting = shooting_vectors_from_psis(&psis, &self.mean_psi, &time);`.
      4. augmented SRSF for new curves: `qn_new = srsf_transform(&aln.aligned_data, &self.argvals)`; `q_aug = build_augmented_srsfs(&qn_new, &aln.aligned_data, n_new, m)`.
      5. center with the TRAINED mean (do NOT call center_matrix on new data): subtract `self.mean_q[j]` from `q_aug[(i,j)]` for j in 0..(m+1).
      6. scores via the direct dot-product formula (RESEARCH §9): for each component k in 0..self.ncomp and each i: `s = sum_{j=0..m+1} q_aug_centered[(i,j)]*self.vert_component[(k,j)] + self.balance_c * sum_{j=0..m} shooting[(i,j)]*self.horiz_component[(k,j)]`. Do NOT call project_onto_eigenvectors (that helper is for vert_fpca covariance-SVD U and gives numerically different results).
      7. Return JfpcaTransform{ scores, aligned: aln.aligned_data, warping: aln.gammas }.

    Wire re-exports: in `lib.rs` add `pub mod jfpca_model;` (alphabetically among elastic_* modules) and a `pub use jfpca_model::{jfpca_fit, JfpcaModel, JfpcaTransform};` line mirroring the existing elastic_fpca re-export block; in `prelude.rs` add `pub use crate::{JfpcaModel, JfpcaTransform, jfpca_fit};` mirroring line 71. Additive only — touch no existing signature or re-export.

    Add ONE tracer test in `#[cfg(test)] mod tests` proving the path end-to-end: build a full-rank spanning fixture (see below), fit, transform the SAME training curves, assert `transform.scores` has shape (n, model.ncomp) and is finite. Full numerical gates come in Task 2.

    TEST FIXTURE (project memory hazard — recovery/round-trip tests silently pass on low-rank data): do NOT reuse the single-frequency phase-shifted-sinusoid `generate_test_data` from elastic_fpca.rs — phase-shifted single-freq sinusoids span only a 2-D subspace and mask errors. Build a spanning multi-frequency fixture with n >> m in rank terms: e.g. n=12, m=15, each curve a distinct combination of >=3 sine/cosine harmonics with per-curve random-but-deterministic amplitudes/phase (seed a StdRng), plus a small per-curve warp, so the augmented representation is full-rank up to ncomp. Use ncomp=4.
  </action>
  <verify>
    <automated>cargo build -p fdars-core --features linalg,parallel</automated>
    <fails_when>non-zero exit, or "error[E" / "cannot find" / "is private" in output</fails_when>
    <automated>cargo test -p fdars-core --features linalg,parallel jfpca_model::tests::tracer</automated>
    <fails_when>non-zero exit, or "test result: FAILED" in output, or "0 passed" in the summary</fails_when>
  </verify>
  <done>New module compiles; `jfpca_fit` and `.transform()` exist and are re-exported at crate root + prelude; the single tracer test fits + transforms training curves end-to-end and asserts finite scores of shape (n, ncomp); committed.</done>
  <reversibility rating="costly">Public API names (JfpcaModel / JfpcaTransform / jfpca_fit / .transform()) are a published-contract decision — costly to rename once downstream (Phase 73) consumes them, but already locked in 72-CONTEXT.md, so no new checkpoint is needed (decision pre-confirmed by the user).</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Make-or-break numerical gates — fit reproduces joint_fpca (1e-8) + round-trip (1e-8)</name>
  <files>fdars-core/src/jfpca_model.rs</files>
  <read_first>
    - fdars-core/src/jfpca_model.rs (the module from Task 1 — jfpca_fit, transform, and the spanning fixture helper)
    - fdars-core/src/elastic_fpca.rs:286-358 (joint_fpca — the reference whose scores must be reproduced exactly)
    - .planning/phases/72-jfpca-fit-transform-seam/72-VALIDATION.md (gates 1-2; achieved-tolerance documentation requirement)
    - 72-RESEARCH.md Pitfalls 1-6 (project_onto_eigenvectors misuse, mean_q recompute, fresh karcher mean, lambda mismatch, mean_psi capture, ncomp clamp)
  </read_first>
  <behavior>
    - Test test_fit_scores_match_joint_fpca: fit a model, independently run karcher_mean + joint_fpca with the SAME args, assert max element-wise abs diff between model.joint_result.scores and joint_fpca(...).scores < 1e-8 (VEE-01a). Since fit reuses joint_fpca directly the diff should be ~0.
    - Test test_model_fields_populated: assert mean_psi.len()==m, mean_q.len()==m+1, vert_component shape (ncomp, m+1), horiz_component shape (ncomp, m), eigenvalues.len()==model.ncomp, argvals==input argvals, and model.ncomp == joint_result.eigenvalues.len() (clamp respected) (VEE-01b).
    - Test test_roundtrip_training_curves: transform the ORIGINAL training curves through model.transform and assert max abs diff vs model.joint_result.scores < 1e-8 (VEE-02a). This is the top correctness risk — if it fails at >1e-4 the transform is likely calling the wrong projection primitive or recomputing mean_q.
  </behavior>
  <action>
    Add the three tests above to `#[cfg(test)] mod tests` in jfpca_model.rs, all using the full-rank spanning fixture from Task 1 (never the low-rank single-freq generator). Use a small helper `max_abs_diff(a: &FdMatrix, b: &FdMatrix) -> f64` local to the test module.

    For the round-trip test, transform the exact training FdMatrix (the raw curves passed to jfpca_fit, not the aligned ones). The trained scores live at model.joint_result.scores. Document the achieved round-trip tolerance in a test comment (per VALIDATION.md gate 2) but keep the assertion threshold at 1e-8 — a spanning fixture with deterministic alignment should meet it; if it genuinely cannot, STOP and surface the achieved value rather than loosening silently (per project memory: never loosen tolerance to paper over a low-rank/design fault).

    If test_roundtrip_training_curves fails, the cause is almost certainly one of: (a) calling project_onto_eigenvectors instead of the dot-product formula, (b) centering with a freshly-computed mean instead of self.mean_q, (c) a fresh karcher_mean instead of align_to_target against self.karcher_mean, or (d) lambda mismatch. Fix the transform, do not adjust the tolerance.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel jfpca_model::tests::test_fit_scores_match_joint_fpca jfpca_model::tests::test_model_fields_populated jfpca_model::tests::test_roundtrip_training_curves</automated>
    <fails_when>non-zero exit, or "test result: FAILED" in output, or fewer than "3 passed" in the summary</fails_when>
  </verify>
  <done>All three numerical gates pass: training scores reproduce joint_fpca within 1e-8, model fields populated with correct shapes and clamped ncomp, and fit->transform round-trip on training curves within 1e-8; committed.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Error paths + entry validation (grid mismatch, degenerate input)</name>
  <files>fdars-core/src/jfpca_model.rs</files>
  <read_first>
    - fdars-core/src/jfpca_model.rs (transform + jfpca_fit validation from Tasks 1-2)
    - fdars-core/src/elastic_fpca.rs:290-300 (joint_fpca's entry validation pattern to mirror: n>=2, m>=2, ncomp>=1, argvals.len()==m -> FdarError::InvalidDimension)
    - fdars-core/src/error.rs (FdarError::InvalidDimension { parameter, expected, actual } shape)
    - .planning/phases/72-jfpca-fit-transform-seam/72-VALIDATION.md (gates 3-4: grid-mismatch error path, degenerate guards)
  </read_first>
  <behavior>
    - Test test_transform_grid_mismatch_error: fit on m-column curves, call model.transform on curves with a DIFFERENT column count (e.g. m+3); assert the result is Err(FdarError::InvalidDimension { .. }) — matches!(res, Err(FdarError::InvalidDimension{..})) — no panic, no silent resample (VEE-02b).
    - Test test_fit_rejects_degenerate: jfpca_fit with argvals.len() != data ncols, and with ncomp==0, each returns Err(FdarError::InvalidDimension). n<2 also rejected.
  </behavior>
  <action>
    Ensure `transform` checks `new_curves.shape().1 != self.argvals.len()` FIRST and returns FdarError::InvalidDimension with a descriptive parameter/expected/actual (mirror joint_fpca's message style), before any alignment work. Ensure jfpca_fit's entry validation from Task 1 covers argvals length mismatch, ncomp<1, n<2, m<2.

    Add the two tests above to the test module. Use matches! or an explicit match on the returned FdarError variant — do NOT assert on the formatted message string (brittle).
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel jfpca_model::tests::test_transform_grid_mismatch_error jfpca_model::tests::test_fit_rejects_degenerate</automated>
    <fails_when>non-zero exit, or "test result: FAILED" in output, or fewer than "2 passed" in the summary</fails_when>
  </verify>
  <done>Grid-mismatch and degenerate-input paths return FdarError::InvalidDimension (no panic); both tests pass; committed.</done>
</task>

<task type="auto">
  <name>Task 4: Module doctest + full-crate gates (test/clippy --all-targets/fmt/doc)</name>
  <files>fdars-core/src/jfpca_model.rs</files>
  <read_first>
    - fdars-core/src/jfpca_model.rs (public items needing a doctest)
    - fdars-core/src/lib.rs:502 and prelude.rs:71 (confirm re-exports so the doctest can `use fdars_core::{jfpca_fit, ...}`)
    - fdars-core/src/elastic_fpca.rs (any existing module-level doctest as a style reference)
    - .planning/phases/72-jfpca-fit-transform-seam/72-VALIDATION.md (gate 5: running module doctest under cargo test --doc)
  </read_first>
  <action>
    Add a running module-level doctest (a ```rust fenced block in the `//!` doc of jfpca_model.rs, or on `jfpca_fit`) that: constructs a small deterministic multi-harmonic (full-rank, NOT single-freq) FdMatrix + argvals, calls `jfpca_fit(&data, &argvals, ncomp, None, 0.0, 20)?`, then `model.transform(&data)?`, and asserts the returned `scores` shape equals (n, model.ncomp). Import via the crate-root re-exports (`use fdars_core::{jfpca_fit, JfpcaModel, JfpcaTransform};`). Make the doctest self-contained and fast (small n, m).

    Then run the full phase gate battery OUT-OF-BAND (the pre-commit hook times out — see execution_context). Fix any clippy warning in the new module or its test code (clippy --all-targets lints tests). Run `cargo fmt` before committing and commit with `--no-verify`.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel --doc jfpca_model</automated>
    <fails_when>non-zero exit, or "test result: FAILED" in output, or "0 passed" in the doc-test summary</fails_when>
    <automated>cargo test -p fdars-core --features linalg,parallel</automated>
    <fails_when>non-zero exit, or "test result: FAILED" in output</fails_when>
    <automated>cargo clippy --all-targets --features linalg,parallel -- -D warnings</automated>
    <fails_when>non-zero exit, or "warning:" / "error:" emitted by clippy</fails_when>
    <automated>cargo fmt --check</automated>
    <fails_when>non-zero exit, or any "Diff in" line in output</fails_when>
  </verify>
  <done>Module doctest runs green under cargo test --doc; whole-crate test suite green; clippy --all-targets clean; cargo fmt --check clean; committed with --no-verify after fmt.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

N/A — No new trust boundary. Pure in-process numerical transform over caller-supplied `FdMatrix` values; no external API/SDK, no I/O, no untrusted-input surface, no crate dependency. Input validation is via the crate's existing `FdarError` dimension checks at function entry (ASVS V5, standard library pattern).

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-72-01 | Tampering/DoS | jfpca_fit / transform entry | low | mitigate | Dimension checks at entry (n>=2, m>=2, ncomp>=1, argvals.len()==m; transform grid-length match) return FdarError::InvalidDimension — Tasks 1 & 3 |
| T-72-02 | DoS | NaN/Inf from degenerate input | low | accept | Existing karcher / SVD paths surface ComputationFailed; no new handling — matches crate convention |
</threat_model>

<verification>
Phase-level checks (run after Task 4, out-of-band, commit with --no-verify):
- `cargo test -p fdars-core --features linalg,parallel` — whole crate green (new jfpca_model tests + all existing elastic_fpca regression tests still pass → additive/non-breaking confirmed).
- `cargo test -p fdars-core --features linalg,parallel --doc` — module doctest green.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean (CI parity; lints test code).
- `cargo fmt --check` — clean.
- Grep confirms no existing public signature in elastic_fpca.rs / lib.rs / prelude.rs was changed (only additive re-exports + one `pub(crate)` promotion).
</verification>

<success_criteria>
- VEE-01: `jfpca_fit` returns a `JfpcaModel` storing karcher_mean, mean_psi, mean_q, vert_component/horiz_component, balance_c, argvals, eigenvalues, clamped ncomp, lambda, embedded joint_result; training scores reproduce `joint_fpca` within 1e-8.
- VEE-02: `model.transform(&new_curves)` aligns new curves to the trained template and projects onto the trained basis, returning `JfpcaTransform { scores, aligned, warping }`; fit->transform round-trip on training curves within 1e-8; grid mismatch returns `FdarError::InvalidDimension`.
- Running module doctest under `cargo test --doc`.
- Additive/non-breaking; whole-crate test + clippy --all-targets + fmt gates pass.
</success_criteria>

<output>
Create `.planning/phases/72-jfpca-fit-transform-seam/72-01-SUMMARY.md` when done.
</output>
