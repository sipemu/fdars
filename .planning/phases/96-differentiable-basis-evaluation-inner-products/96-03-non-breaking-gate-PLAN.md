---
phase: 96-differentiable-basis-evaluation-inner-products
plan: 03
type: execute
wave: 3
depends_on: ["96-01", "96-02"]
files_modified: []
autonomous: true
requirements: [DOP-01]

estimate:
  tokens: 45000
  raw_tokens: 22000
  tasks: 3
  confidence: high

must_haves:
  truths:
    - "The whole crate compiles and all gates pass with the generalized basis eval: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, doctests, all 28 examples, `--features serde` build, and `wasm32-unknown-unknown --features js` build (DOP-01 non-breaking guarantee)."
    - "The git diff for Phase 96 is confined to `fdars-core/src/basis/bspline.rs`, `fdars-core/src/basis/fourier.rs`, and `fdars-core/src/basis/tests.rs` — no caller churn anywhere else in `fdars-core/src/`."
  artifacts:
    - ".planning/phases/96-differentiable-basis-evaluation-inner-products/96-03-SUMMARY.md — the gate evidence record"
  key_links:
    - "The full-suite + churn-diff gate is the end-to-end proof that the additive generalization is strictly non-breaking (no example, binding, or caller changed)."
  prohibitions:
    - statement: "git diff confined to basis/bspline.rs, basis/fourier.rs, basis/tests.rs — no caller churn."
    - statement: "No new crate dependency; --features serde build stays green."
    - statement: "Public signatures of fourier_basis / fourier_basis_with_period unchanged."
---

<objective>
NON-BREAKING GATE: prove the Phase 96 basis-eval generalization is strictly additive and non-breaking by running every whole-crate gate out-of-band (per-gate, foreground, 600s each) and confirming the git diff is confined to the two basis source files plus their test file. This is the terminal wave — its gates stand as end-to-end proof of DOP-01's non-breaking guarantee.

Purpose: Catch any hidden caller churn, doctest breakage, example breakage, serde/wasm regression, or clippy warning introduced by the generalization. Implements DOP-01.
Output: Gate evidence recorded in the plan SUMMARY; no source changes (this plan verifies, it does not modify src).
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/96-differentiable-basis-evaluation-inner-products/96-VALIDATION.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Clippy (--all-targets) + full test + doctest gates (per-gate, foreground, 600s each)</name>
  <files>fdars-core/src/basis/bspline.rs, fdars-core/src/basis/fourier.rs, fdars-core/src/basis/tests.rs</files>
  <read_first>
    - .planning/phases/96-differentiable-basis-evaluation-inner-products/96-VALIDATION.md (lines 57-62: the exact non-breaking gate command set)
  </read_first>
  <action>
    Run each gate SEPARATELY in the FOREGROUND with a 600s timeout — NEVER as one combined backgrounded command (a combined backgrounded gate gets killed mid-run per MEMORY.md). If a build fails with "No space left"/"linking with cc failed", run `rm -rf target/debug/{incremental,examples}` and retry that gate. Run in order: (1) `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code — `--all-targets` is required; a plain build misses test-code warnings); (2) full `cargo test -p fdars-core --features linalg,parallel`; (3) doctests `cargo test -p fdars-core --doc --features linalg,parallel`. This task does not modify source — it verifies the Plan 01/02 output. Implements DOP-01.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -15 && cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -15 && cargo test -p fdars-core --doc --features linalg,parallel 2>&1 | tail -10</automated>
    <fails_when>Clippy emits any warning (`-D warnings` → error), OR any lib/integration test fails (a numeric regression the parity guards should have caught, or a broken caller), OR any doctest fails. Note the co_cluster/svd_sign golden flake (MEMORY.md) fails ONLY under full non-linalg runs — with `--features linalg` present it must pass; if it flakes, re-run that binary in isolation to confirm it is the known env flake, not a Phase 96 regression.</fails_when>
  </verify>
  <done>Clippy `--all-targets` clean, full test suite green, doctests green.</done>
</task>

<task type="auto">
  <name>Task 2: 28 examples + serde build + wasm32 build gates (per-gate, foreground, 600s each)</name>
  <files>fdars-core/src/basis/bspline.rs, fdars-core/src/basis/fourier.rs</files>
  <read_first>
    - .planning/phases/96-differentiable-basis-evaluation-inner-products/96-VALIDATION.md (lines 57-62: examples + serde + wasm gate)
  </read_first>
  <action>
    Run each gate SEPARATELY in the FOREGROUND (600s each), never combined/backgrounded. If disk pressure hits ("No space left"/"linking with cc failed"), `rm -rf target/debug/{incremental,examples}` and retry. Run: (1) all 28 examples build `cargo build -p fdars-core --examples --features linalg,parallel`; (2) serde build guard `cargo build -p fdars-core --features serde` (must stay green — no regression, no new dependency); (3) wasm target `cargo build -p fdars-core --target wasm32-unknown-unknown --features js`. This task does not modify source. Implements DOP-01.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo build -p fdars-core --examples --features linalg,parallel 2>&1 | tail -15 && cargo build -p fdars-core --features serde 2>&1 | tail -10 && cargo build -p fdars-core --target wasm32-unknown-unknown --features js 2>&1 | tail -10</automated>
    <fails_when>Any of the 28 examples fails to compile (a caller of the generalized evaluators broke — should not happen given T=f64 inference), OR the `--features serde` build fails (regression), OR the wasm32 build fails. Distinguish a genuine failure from the pre-existing serde-feature breakage (ShapeletTransformClassifier/ClassifFit, MEMORY.md) — if the serde build fails with that exact pre-existing error and NOT in basis/, it is not a Phase 96 regression; record it, do not attribute to this phase.</fails_when>
  </verify>
  <done>All 28 examples build; `--features serde` build green (modulo the pre-existing unrelated ClassifFit issue if it surfaces); wasm32 build green.</done>
</task>

<task type="auto">
  <name>Task 3: git-diff-churn confinement check + fdars-r source grep + SUMMARY commit</name>
  <files>fdars-core/src/basis/bspline.rs, fdars-core/src/basis/fourier.rs, fdars-core/src/basis/tests.rs</files>
  <read_first>
    - .planning/phases/96-differentiable-basis-evaluation-inner-products/96-VALIDATION.md (line 62: the churn-confinement requirement)
  </read_first>
  <action>
    Confirm the Phase 96 changed source files are confined to `fdars-core/src/basis/bspline.rs`, `fdars-core/src/basis/fourier.rs`, `fdars-core/src/basis/tests.rs`. Use a git-diff-name-only over the Phase 96 commits (the HEAD range covering Plans 01 and 02, typically `HEAD~2..HEAD` on `fdars-core/src/*` before this SUMMARY commit) filtered to exclude the three allowed basis files — the result MUST be empty. To avoid the `grep -c`/comment self-invalidation pitfall, use `git diff --name-only <range> -- 'fdars-core/src'` then a fixed-string exclusion, not a bare count. Also grep confirm no fdars-r binding source references any basis symbol that changed shape (the R bindings live in the external `fdars-r` package; confirm no in-repo binding file references changed). Then write the plan SUMMARY and commit it with `git commit --no-verify` after `cargo fmt -p fdars-core` (message: `docs(96): non-breaking gate evidence for DOP-01 basis eval`). Implements DOP-01.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && CHANGED=$(git diff --name-only HEAD~2..HEAD -- 'fdars-core/src' | grep -vE 'basis/bspline\.rs|basis/fourier\.rs|basis/tests\.rs' || true); test -z "$CHANGED" && echo CHURN_CONFINED || { echo "UNEXPECTED_CHURN: $CHANGED"; false; }</automated>
    <fails_when>The diff over the Phase 96 commit range touches any `fdars-core/src` file other than basis/bspline.rs, basis/fourier.rs, basis/tests.rs — printing `UNEXPECTED_CHURN: <files>` and exiting non-zero. This signals a caller was edited (breaking the additive/non-breaking invariant) or a public signature was threaded through. If the commit range differs from HEAD~2..HEAD (e.g. extra fmt commits), adjust the range to span exactly the Plan 01+02 source commits.</fails_when>
  </verify>
  <done>git diff over the Phase 96 range is confined to the three basis files (`CHURN_CONFINED`); SUMMARY written and committed with `--no-verify` after `cargo fmt`.</done>
</task>

</tasks>

<threat_model>
No attack surface — this plan runs build/test gates and inspects git diffs over pure in-crate numeric basis-evaluation code. No external input, network, IO, auth, or untrusted data. No new crate dependency.

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-96-03 | Tampering | build/test gate scope | low | accept | Gate is read-only verification; churn-confinement diff is the guard against unintended source changes. |
</threat_model>

<verification>
- Clippy `--all-targets` clean; full test + doctests green; 28 examples + serde + wasm builds green.
- git diff over the Phase 96 range confined to basis/bspline.rs, basis/fourier.rs, basis/tests.rs.
</verification>

<success_criteria>
- Every whole-crate gate passes (clippy, test, doctest, examples, serde, wasm).
- Change confined to the two basis source files + their test file; public Fourier signatures unchanged.
- SUMMARY committed with `--no-verify` after `cargo fmt`.
</success_criteria>

<output>
Create `.planning/phases/96-differentiable-basis-evaluation-inner-products/96-03-SUMMARY.md` when done.
</output>
