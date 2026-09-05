---
phase: 74-elastic-conformal-anomaly-detection
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/tolerance/types.rs
  - fdars-core/src/tolerance/conformal.rs
  - fdars-core/src/tolerance/conformal_anomaly.rs
  - fdars-core/src/tolerance/mod.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [ECA-01, ECA-02, ECA-03]

estimate:
  tokens: 78000
  raw_tokens: 39000
  tasks: 5
  confidence: med

must_haves:
  truths:
    - "elastic_nonconformity(curve, curve, argvals, lambda, variant) < 1e-4 for every elastic variant (zero-for-identical, tolerance not exact) [ECA-01]"
    - "elastic_nonconformity returns a non-negative score for all valid inputs and Err(InvalidParameter) for SupNorm/L2 [ECA-01]"
    - "On exchangeable clean data the empirical flag rate is approximately alpha (marginal validity) [ECA-02]"
    - "An injected magnitude outlier is flagged by AmplitudeElastic/CombinedElastic [ECA-02]"
    - "An injected shape (phase-distorted) outlier is flagged by PhaseElastic/CombinedElastic [ECA-02]"
    - "p_value == (1 + #{calib_score >= test_score}) / (n_calib + 1); flags == (p_value <= alpha); threshold == (1-alpha) order-statistic quantile of calibration scores [ECA-02, ECA-03]"
    - "conformal_prediction_band still works for SupNorm/L2 and returns None for the three elastic variants (band regression, no panic) [ECA-03]"
    - "A calibrate->flag module doctest runs green under cargo test --doc [ECA-03]"
    - "Additive/non-breaking: existing public signatures unchanged; whole-crate test + clippy --all-targets + fmt green"
  artifacts:
    - "fdars-core/src/tolerance/conformal_anomaly.rs (new module)"
    - "NonConformityScore::AmplitudeElastic / PhaseElastic / CombinedElastic variants in tolerance/types.rs"
    - "elastic_nonconformity fn, elastic_conformal_anomaly fn, ConformalAnomalyConfig, ConformalAnomalyResult"
  key_links:
    - "elastic_nonconformity dispatch -> crate::alignment::{amplitude_distance, phase_distance_pair, elastic_distance} (verbatim, argument order curve=f1, template=f2)"
    - "template resolution -> crate::alignment::karcher_mean(calibration, argvals, max_iter, tol, lambda).mean"
    - "threshold -> crate::helpers::sort_nan_safe + local order-statistic calibrated_threshold"
    - "re-exports wired through tolerance/mod.rs -> lib.rs -> prelude.rs"
---

<objective>
Add an inductive conformal anomaly detector for functional data that flags both
magnitude and shape outliers by scoring curves against a reference template using
elastic distances. Additive and non-breaking: the existing
`conformal_prediction_band` path stays behaviorally unchanged.

Purpose: Deliver ECA-01 (elastic nonconformity scores), ECA-02 (inductive
conformal detector with marginal validity + outlier flagging), and ECA-03
(result type + re-exports + doctest) as one cohesive additive module, per the
locked CONTEXT.md decisions.

Output: new `tolerance/conformal_anomaly.rs`, three new `NonConformityScore`
variants, `elastic_nonconformity` + `elastic_conformal_anomaly` public API,
`ConformalAnomalyConfig` + `ConformalAnomalyResult` types, and full
crate-root + prelude re-exports with a running module doctest.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/74-elastic-conformal-anomaly-detection/74-CONTEXT.md
@.planning/phases/74-elastic-conformal-anomaly-detection/74-RESEARCH.md
@.planning/phases/74-elastic-conformal-anomaly-detection/74-VALIDATION.md
@fdars-core/src/tolerance/types.rs
@fdars-core/src/tolerance/conformal.rs
@fdars-core/src/tolerance/mod.rs
@fdars-core/src/alignment/pairwise.rs
@fdars-core/src/alignment/karcher.rs
@fdars-core/src/helpers.rs
</context>

<artifacts_this_phase_produces>
Every new public symbol this phase introduces (all additive, all re-exported from
`tolerance/mod.rs`, `lib.rs`, and `prelude.rs` per D-ECA-03):

- `NonConformityScore::AmplitudeElastic` — variant (tolerance/types.rs)
- `NonConformityScore::PhaseElastic` — variant (tolerance/types.rs)
- `NonConformityScore::CombinedElastic` — variant (tolerance/types.rs)
- `elastic_nonconformity(curve, template, argvals, lambda, variant) -> Result<f64, FdarError>` (tolerance/conformal_anomaly.rs)
- `elastic_conformal_anomaly(calibration, test, argvals, config) -> Result<ConformalAnomalyResult, FdarError>` (tolerance/conformal_anomaly.rs)
- `ConformalAnomalyConfig` struct (variant, alpha, lambda, max_iter, tol, template) with `Default` (tolerance/conformal_anomaly.rs)
- `ConformalAnomalyResult` struct { p_values, scores, flags, threshold } (tolerance/conformal_anomaly.rs)

The existing `NonConformityScore` re-exports auto-cover the three new variants —
no separate variant re-export line is needed.
</artifacts_this_phase_produces>

<execution_hazards>
HAZARD (from project memory): the pre-commit hook runs the full test suite and
TIMES OUT on long fdars builds; backgrounded full-hook commits get killed.
- Run all verify gates OUT OF BAND (inline), then commit with `--no-verify`.
- Run `cargo fmt` before EACH commit (--no-verify skips the fmt hook -> CI fmt-check
  would otherwise fail despite green clippy).
- Do a whole-crate `cargo fmt` sweep at the end of the plan.
CI parity: clippy MUST be `--all-targets` (CI lints test/bench code; a plain
`-p ... -D warnings` misses test-code warnings and gives a false green).
</execution_hazards>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1: End-to-end elastic conformal anomaly tracer — calibrate -> score test -> flag, one happy path wired + verified</name>
  <files>fdars-core/src/tolerance/types.rs, fdars-core/src/tolerance/conformal.rs, fdars-core/src/tolerance/conformal_anomaly.rs, fdars-core/src/tolerance/mod.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - fdars-core/src/tolerance/types.rs:25-33 (NonConformityScore enum: #[non_exhaustive], derives Debug/Clone/Copy/PartialEq/Eq — the three new variants MUST keep the enum Copy, so carry no data)
    - fdars-core/src/tolerance/conformal.rs:10-26 (private nonconformity_score match — the ONLY match site) and :70-86 (conformal_prediction_band signature returns Option<ToleranceBand>; note the existing None-on-invalid-input guard block to mirror placement)
    - fdars-core/src/alignment/pairwise.rs:103-105 (elastic_distance), :384-386 (amplitude_distance delegates to elastic_distance), :388-392 (phase_distance_pair -> warping::phase_distance on optimal gamma) — all signatures (f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64
    - fdars-core/src/alignment/karcher.rs:293-301 (karcher_mean signature: (data, argvals, max_iter, tol, lambda) -> KarcherMeanResult) and alignment/mod.rs:148-167 (KarcherMeanResult.mean is Vec<f64> length m, and :67 karcher_mean is re-exported)
    - fdars-core/src/helpers.rs:302-317 (quantile_sorted) and helpers.rs:10 (sort_nan_safe) — both pub in crate::helpers
    - fdars-core/src/tolerance/mod.rs:17-43 (submodule decls + re-export block; NonConformityScore already re-exported via types)
    - fdars-core/src/lib.rs:346-353 (tolerance re-export block) and prelude.rs:89-92 (prelude tolerance block)
    - fdars-core/src/simulation.rs:374-385 (sim_fundata signature for the doctest fixture: (n, t, big_m, EFunType, EValType, Option<seed>) -> FdMatrix)
  </read_first>
  <action>
    Wire ONE happy path end-to-end through every layer the phase touches, production-quality
    (real error handling on the single path; stubs only where later-fillable without architecture change).

    1. tolerance/types.rs: add three variants to NonConformityScore — AmplitudeElastic,
       PhaseElastic, CombinedElastic — each a unit variant (the enum derives Copy; keep it Copy).
       Doc-comment each (amplitude = Fisher-Rao elastic distance to template; phase = geodesic
       distance of optimal warp from identity; combined = a genuine combination of amplitude and
       phase, see task 2). Per D-ECA-01.

    2. tolerance/conformal.rs: add an EARLY GUARD at the very top of conformal_prediction_band
       (before the existing input-validation block) that returns None when score_type is any of the
       three elastic variants (use matches! with the three variants). Then add a defensive
       `_ => unreachable!("elastic variants are rejected by the early guard in conformal_prediction_band")`
       arm to the private nonconformity_score match so the crate compiles under #[non_exhaustive].
       Do NOT change nonconformity_score's signature (stays -> f64). Do NOT touch the band's
       scoring math. Per D-ECA-01 (band returns error/None for template-based variants) and the
       RESEARCH-recommended approach B.

    3. tolerance/conformal_anomaly.rs (NEW): implement the full happy path.
       - ConformalAnomalyConfig struct: fields variant: NonConformityScore, alpha: f64, lambda: f64,
         max_iter: usize, tol: f64, template: Option<Vec<f64>>. Derive Debug, Clone, PartialEq;
         #[non_exhaustive]; conditional serde via #[cfg_attr(feature = "serde", derive(...))].
         impl Default: variant = CombinedElastic, alpha = 0.1, lambda = 0.0, max_iter = 20,
         tol = 1e-4, template = None. Per D-ECA-02.
       - ConformalAnomalyResult struct: fields p_values: Vec<f64>, scores: Vec<f64>, flags: Vec<bool>,
         threshold: f64. Same derives + #[non_exhaustive] + conditional serde. Per D-ECA-03.
       - elastic_nonconformity(curve, template, argvals, lambda, variant) -> Result<f64, FdarError>:
         dispatch AmplitudeElastic -> crate::alignment::amplitude_distance(curve, template, argvals, lambda);
         PhaseElastic -> crate::alignment::phase_distance_pair(curve, template, argvals, lambda);
         CombinedElastic -> a GENUINE combination (task 2 pins the formula — for the tracer use
         the combination, not an alias of amplitude); SupNorm/L2/_ -> Err(FdarError::InvalidParameter
         { parameter: "variant", message: "..." }). ARGUMENT ORDER IS LOAD-BEARING: curve = f1,
         template = f2 (the alignment warps f2 onto f1); the zero-for-identical test in task 3 catches a flip.
         Document the chosen order in the fn doc.
       - a private calibrated_threshold(sorted_scores: &[f64], alpha: f64) -> f64 using the
         order-statistic k = ceil((n+1)*(1-alpha)) as usize; return f64::INFINITY when k > n else
         sorted_scores[k.saturating_sub(1)]. Do NOT use conformal/mod.rs::conformal_quantile (it is
         pub(super), inaccessible). Sort with crate::helpers::sort_nan_safe first. Per RESEARCH.
       - elastic_conformal_anomaly(calibration: &FdMatrix, test: &FdMatrix, argvals: &[f64],
         config: &ConformalAnomalyConfig) -> Result<ConformalAnomalyResult, FdarError>,
         marked #[must_use = "expensive computation whose result should not be discarded"]:
         (a) validate at entry (dimension checks -> FdarError::InvalidDimension / InvalidParameter):
         calibration and test column count == argvals.len(); n_calib >= 1; alpha in (0.0, 1.0)
         exclusive; config.variant is an elastic variant (else InvalidParameter). (b) resolve template:
         config.template.clone() if Some, else karcher_mean(calibration, argvals, config.max_iter,
         config.tol, config.lambda).mean (Vec<f64> length m). (c) score every calibration row via
         elastic_nonconformity into calib_scores. (d) sort a copy, compute threshold via
         calibrated_threshold. (e) for each test row: score a_star; p_value =
         (1 + count(calib_scores where a >= a_star)) as f64 / (n_calib + 1) as f64;
         flag = p_value <= config.alpha. (f) return ConformalAnomalyResult. Per D-ECA-02.
       - Module docstring (//!): a running calibrate->flag doctest — build calibration + test with
         sim_fundata (fixed seeds), CombinedElastic, alpha 0.1, assert result vector lengths ==
         n_test, threshold >= 0.0, and a loose flag-rate upper bound. Per D-ECA-03. (Full gates in task 4.)

    4. Wire re-exports (additive only, existing items unchanged):
       - tolerance/mod.rs: add `mod conformal_anomaly;` and
         `pub use conformal_anomaly::{elastic_conformal_anomaly, elastic_nonconformity, ConformalAnomalyConfig, ConformalAnomalyResult};`
       - lib.rs:346-353 tolerance block: add the same four items.
       - prelude.rs:89-92 tolerance block: add ConformalAnomalyConfig, ConformalAnomalyResult.
       Per D-ECA-03.

    This is a tracer: functionality gaps (extra variant edge coverage, exhaustive outlier gates)
    are filled in tasks 2-4; there must be NO architectural gap left. NO fenced code in this action.
  </action>
  <verify>
    <automated>cargo build -p fdars-core --features linalg,parallel 2>&1 | tail -3</automated>
    <fails_when>compiler reports non-exhaustive match on NonConformityScore in nonconformity_score, an unresolved import for a re-exported symbol, or a type error in elastic_conformal_anomaly</fails_when>
    <automated>cargo test -p fdars-core --features linalg,parallel --doc conformal_anomaly 2>&1 | tail -6</automated>
    <fails_when>the module doctest panics on a length assertion or threshold >= 0.0, or fails to compile the doctest imports</fails_when>
  </verify>
  <acceptance_criteria>
    - crate builds with linalg,parallel; three new variants exist and are Copy
    - conformal_prediction_band returns None for the elastic variants; nonconformity_score compiles with the defensive arm
    - elastic_conformal_anomaly runs end-to-end on sim_fundata calibration+test and returns a populated ConformalAnomalyResult
    - module doctest passes under cargo test --doc
    - all four new symbols resolve through tolerance/mod.rs, lib.rs, and prelude.rs
  </acceptance_criteria>
  <done>The single calibrate -> score -> flag path works end-to-end, is re-exported, has a green module doctest, and is committed.</done>
  <reversibility rating="reversible">Public API names locked in CONTEXT.md; variant additions are additive on a #[non_exhaustive] enum (non-breaking).</reversibility>
</task>

<task type="auto" tdd="true">
  <name>Task 2: elastic_nonconformity make-or-break gates — zero-for-identical (tol 1e-4), non-negativity, invalid-variant error, CombinedElastic formula (ECA-01)</name>
  <files>fdars-core/src/tolerance/conformal_anomaly.rs</files>
  <read_first>
    - fdars-core/src/tolerance/conformal_anomaly.rs (the module written in task 1 — the elastic_nonconformity dispatch and the CombinedElastic arm)
    - fdars-core/src/alignment/pairwise.rs:103-105, :384-392 (confirm amplitude_distance == elastic_distance exactly; phase_distance_pair is a distinct geodesic-of-warp value — so CombinedElastic must NOT be a bare alias of amplitude)
    - fdars-core/src/alignment/tests.rs:612-624 (existing self-distance test uses d < 0.1 tolerance — the gate here uses ~1e-4, NOT exact equality)
  </read_first>
  <behavior>
    - Test: elastic_nonconformity(curve, curve, argvals, 0.0, variant) < 1e-4 for AmplitudeElastic, PhaseElastic, and CombinedElastic (zero-for-identical, tolerance).
    - Test: elastic_nonconformity(curve, other, argvals, 0.0, variant) >= 0.0 for all three elastic variants on an arbitrary distinct curve (non-negativity).
    - Test: elastic_nonconformity(.., NonConformityScore::SupNorm) and (.., NonConformityScore::L2) each return Err(FdarError::InvalidParameter { parameter: "variant", .. }).
    - Test: for a curve that differs from the template in AMPLITUDE only (scaled), AmplitudeElastic score > 0 and CombinedElastic score >= AmplitudeElastic contribution (Combined is not degenerate).
  </behavior>
  <action>
    Pin the CombinedElastic formula and lock it with tests. Because amplitude_distance delegates to
    elastic_distance EXACTLY, CombinedElastic must be a GENUINE combination of amplitude and phase
    (NOT an alias of amplitude): compute amp = amplitude_distance(curve, template, argvals, lambda)
    and ph = phase_distance_pair(curve, template, argvals, lambda), then combine as the Euclidean
    combination sqrt(amp.powi(2) + ph.powi(2)). Document the exact formula in the CombinedElastic
    dispatch arm's doc/comment so downstream readers know the chosen combination. Update the
    CombinedElastic doc on the enum variant (task 1) if the wording drifted.

    Add an inline #[cfg(test)] mod tests to conformal_anomaly.rs implementing the four behavior tests
    above. Build fixtures inline (a smooth sinusoid curve on a >=50-point uniform grid; a distinct
    curve for non-negativity; a scaled copy for the amplitude case). Use the ~1e-4 tolerance for the
    zero-for-identical gate per RESEARCH (elastic self-distance is only near-zero, not exactly zero).
    Name the test fns so `conformal_anomaly` filters catch them
    (e.g. test_elastic_nonconformity_self_near_zero, _nonneg, _invalid_variant, _combined_not_alias).
    Per D-ECA-01. NO fenced code in this action.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --lib --features linalg,parallel conformal_anomaly 2>&1 | tail -10</automated>
    <fails_when>the self-score exceeds 1e-4 (argument-order flip), a score is negative, SupNorm/L2 do not return Err, or CombinedElastic equals AmplitudeElastic on the scaled curve (degenerate combination)</fails_when>
  </verify>
  <acceptance_criteria>
    - all four elastic_nonconformity tests pass
    - CombinedElastic = sqrt(amp^2 + phase^2), documented, and provably not an alias of amplitude
    - SupNorm/L2 dispatch returns FdarError::InvalidParameter
  </acceptance_criteria>
  <done>elastic_nonconformity is proven non-negative, zero-for-identical within tolerance, rejects non-elastic variants, and CombinedElastic is a genuine combination — committed.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: detector behavioral gates — marginal validity, magnitude + shape outliers, p-value/threshold correctness, result shape (ECA-02, ECA-03)</name>
  <files>fdars-core/src/tolerance/conformal_anomaly.rs</files>
  <read_first>
    - fdars-core/src/tolerance/conformal_anomaly.rs (elastic_conformal_anomaly + calibrated_threshold from task 1; the tests mod from task 2)
    - fdars-core/src/simulation.rs:374-385 (sim_fundata) and grep EFunType/EValType variants in simulation.rs for a genuinely different-shape generator (e.g. Fourier vs Wiener) to build a shape outlier
    - fdars-core/src/helpers.rs:302-317 (quantile_sorted) to cross-check the threshold order-statistic against an independent computation
    - .planning/phases/74-elastic-conformal-anomaly-detection/74-RESEARCH.md:663-669 (test design: n_calib=100, n_test=100, flag-rate tolerance [0.02, 0.20]; magnitude = scaled mean; shape = different EFunType or phase-shifted argvals)
  </read_first>
  <behavior>
    - Test (marginal validity): calibration + test both from sim_fundata with the SAME generator (fixed seeds, n_calib=100, n_test=100), CombinedElastic, alpha=0.1 -> empirical flag rate within a tolerance band (e.g. <= 0.25; assert deterministically with the fixed seed).
    - Test (magnitude outlier): a test set of clean curves plus one injected scaled/shifted (amplitude) curve, AmplitudeElastic -> the injected curve's flag == true (and its score > threshold).
    - Test (shape outlier): a test set of clean curves plus one injected phase-distorted / different-shape curve, PhaseElastic -> the injected curve's flag == true.
    - Test (combined catches both): CombinedElastic flags both a magnitude and a shape outlier in one test set.
    - Test (p-value/threshold correctness): for a small hand-checkable calibration set + one test curve, p_value == (1 + count(calib >= a_star))/(n_calib+1); flags[i] == (p_values[i] <= alpha) for all i; threshold == calibrated_threshold(sorted_calib, alpha) computed independently.
    - Test (result shape): p_values.len() == scores.len() == flags.len() == n_test; threshold is finite and >= 0.0.
  </behavior>
  <action>
    Add the six behavioral tests to the inline #[cfg(test)] mod tests in conformal_anomaly.rs.
    Build deterministic fixtures with sim_fundata and fixed seeds. For the magnitude outlier, inject
    a row that is the clean mean scaled by a large factor (e.g. 5x). For the shape outlier, inject a
    row generated with a genuinely different EFunType (or an argvals phase shift) so the phase/elastic
    distance is large. For the p-value/threshold correctness test, use a tiny calibration set whose
    order statistic is hand-verifiable, and recompute the expected p-value and threshold independently
    in the test (do not call the private helper via the same code path you are testing — recompute the
    formula inline). Widen the marginal-validity tolerance band per RESEARCH if the fixed-seed rate is
    near the edge; keep it deterministic (no unseeded RNG). Name fns so `conformal_anomaly` catches
    them. Per D-ECA-02 and D-ECA-03. NO fenced code in this action.
  </action>
  <verify>
    <automated>cargo test -p fdars-core --lib --features linalg,parallel conformal_anomaly 2>&1 | tail -12</automated>
    <fails_when>the injected magnitude or shape outlier is NOT flagged, the clean-data flag rate falls outside the tolerance band, flags disagree with (p_values <= alpha), threshold mismatches the independent order-statistic, or a result vector length != n_test</fails_when>
  </verify>
  <acceptance_criteria>
    - marginal-validity, magnitude-outlier, shape-outlier, combined, p-value/threshold, and result-shape tests all pass deterministically
    - flags are exactly (p_values <= alpha); threshold matches the independent (1-alpha) order statistic
  </acceptance_criteria>
  <done>The inductive detector is proven valid on clean data and flags both magnitude and shape outliers with correct p-values/threshold — committed.</done>
</task>

<task type="auto">
  <name>Task 4: band regression + non-exhaustive-match safety + module doctest hardening (ECA-03)</name>
  <files>fdars-core/src/tolerance/conformal_anomaly.rs, fdars-core/src/tolerance/conformal.rs</files>
  <read_first>
    - fdars-core/src/tolerance/conformal.rs:1-117 (early guard + nonconformity_score defensive arm from task 1; conformal_prediction_band doctest at :56-onward stays green)
    - fdars-core/src/tolerance/conformal_anomaly.rs (module docstring doctest from task 1)
    - fdars-core/src/tolerance/tests.rs (existing tolerance test module — where a band-regression test naturally sits) OR add the regression test inline in conformal_anomaly tests
    - .planning/phases/74-elastic-conformal-anomaly-detection/74-RESEARCH.md:596-622 (doctest example) and :669 (band regression: is_none() for all three elastic variants)
  </read_first>
  <action>
    Lock the regression + doctest gates:
    1. Add a band-regression test asserting conformal_prediction_band(&data, 0.2, 0.95, variant, 42).is_none()
       for each of AmplitudeElastic, PhaseElastic, CombinedElastic, AND that it still returns Some(..) for
       SupNorm and L2 on the same valid data (proving the band path is behaviorally unchanged, no panic).
    2. Harden the module docstring doctest in conformal_anomaly.rs into the canonical calibrate->flag
       example (imports via the crate-root or tolerance re-exports, sim_fundata calibration + test with
       fixed seeds, CombinedElastic, alpha 0.1): assert result.p_values.len()/flags.len() == n_test,
       result.threshold >= 0.0, and a loose flag-rate upper bound. Ensure it runs under cargo test --doc.
    Per D-ECA-03. NO fenced code in this action (the doctest itself lives in the module docstring, not here).
  </action>
  <verify>
    <automated>cargo test -p fdars-core --lib --features linalg,parallel conformal_anomaly 2>&1 | tail -12</automated>
    <fails_when>the band returns Some for an elastic variant, returns None for SupNorm/L2 on valid data, or panics on any variant</fails_when>
    <automated>cargo test -p fdars-core --features linalg,parallel --doc conformal_anomaly 2>&1 | tail -6</automated>
    <fails_when>the module doctest fails to compile or panics on a length/threshold/flag-rate assertion</fails_when>
  </verify>
  <acceptance_criteria>
    - band returns None for all three elastic variants and Some for SupNorm/L2 on valid data (no panic)
    - module calibrate->flag doctest passes under cargo test --doc
  </acceptance_criteria>
  <done>Band path proven unchanged for existing variants and safely rejecting elastic ones; module doctest green — committed.</done>
</task>

<task type="auto">
  <name>Task 5: full-crate gates — non-regression suite, clippy --all-targets, fmt (ECA-01/02/03)</name>
  <files>fdars-core/src/tolerance/conformal_anomaly.rs, fdars-core/src/tolerance/types.rs, fdars-core/src/tolerance/conformal.rs, fdars-core/src/tolerance/mod.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - all files touched in tasks 1-4 (final review pass for #[must_use], #[non_exhaustive], conditional serde derives, and doc comments on every new public item per project conventions)
  </read_first>
  <action>
    Final hardening + whole-crate gates. Confirm: elastic_conformal_anomaly is #[must_use];
    ConformalAnomalyConfig and ConformalAnomalyResult are #[non_exhaustive] with Debug/Clone/PartialEq
    and conditional serde derives; every new public fn/struct/field/variant has a doc comment;
    re-exports present in tolerance/mod.rs, lib.rs, prelude.rs. Run the full crate test suite (proves
    no regression across the whole crate), clippy --all-targets (CI parity — catches test/bench-code
    warnings), and fmt --check. Fix any warning/format drift. Run `cargo fmt` before committing.
    Per project conventions + the additive/non-breaking must-have. NO fenced code in this action.
    Commit with --no-verify (pre-commit hook times out on the full suite — gates are run here inline).
  </action>
  <verify>
    <automated>cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -8</automated>
    <fails_when>any pre-existing test regresses or any new conformal_anomaly test fails in the full-suite run</fails_when>
    <automated>cargo clippy --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -6</automated>
    <fails_when>clippy emits any warning in library OR test/bench code (denied via -D warnings)</fails_when>
    <automated>cargo fmt --check 2>&1 | tail -3</automated>
    <fails_when>rustfmt reports a diff (unformatted new code)</fails_when>
  </verify>
  <acceptance_criteria>
    - full crate test suite green (no regression)
    - clippy --all-targets clean under -D warnings
    - cargo fmt --check clean
    - every new public item documented, #[must_use]/#[non_exhaustive]/conditional-serde per convention
  </acceptance_criteria>
  <done>Whole crate is green (test + clippy --all-targets + fmt), additive and non-breaking — committed.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller -> library API | In-process numerical FDA computation over caller-owned matrices; no I/O, no network, no untrusted external input, no external API/SDK. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-74-01 | Tampering | elastic_conformal_anomaly input dimensions | low | mitigate | Dimension checks at entry via FdarError::InvalidDimension (calibration/test columns == argvals.len()); n_calib >= 1; never panic on shape mismatch. |
| T-74-02 | Tampering | p-value / threshold arithmetic | low | mitigate | Guard alpha in (0,1) and n_calib > 0 via FdarError::InvalidParameter before the (n_calib+1) divisor; sort with sort_nan_safe (NaN-safe). |

N/A — pure-numerical additive FDA computation: no authentication, session, access-control,
cryptography, external API, identity transition, ORM/schema, or untrusted-input surface. Input
validation (V5) via FdarError dimension/parameter checks is the only applicable control.
</threat_model>

<verification>
- Per task: `cargo test -p fdars-core --lib --features linalg,parallel conformal_anomaly`
- Doctest: `cargo test -p fdars-core --features linalg,parallel --doc conformal_anomaly`
- Whole-crate gate (task 5): full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check`
- HAZARD: pre-commit hook times out on the full suite — run gates inline out-of-band, commit `--no-verify`, run `cargo fmt` per commit.
</verification>

<success_criteria>
- Three elastic NonConformityScore variants added (additive, Copy, #[non_exhaustive]).
- elastic_nonconformity: non-negative, < 1e-4 for identical curve, Err on SupNorm/L2, CombinedElastic a genuine sqrt(amp^2+phase^2).
- elastic_conformal_anomaly: marginal validity on clean data, flags magnitude AND shape outliers, correct p-values/threshold, correct result shapes.
- conformal_prediction_band unchanged for SupNorm/L2, returns None for elastic variants (no panic).
- Module doctest green under cargo test --doc; all new symbols re-exported from tolerance/mod.rs, lib.rs, prelude.rs.
- Whole crate green: full suite + clippy --all-targets + fmt.
</success_criteria>

<output>
Create `.planning/phases/74-elastic-conformal-anomaly-detection/74-01-SUMMARY.md` when done
</output>
