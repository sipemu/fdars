---
phase: 97-differentiable-regression-prediction-smoothing-penalties
plan: 03
type: execute
wave: 3
depends_on: [97-01, 97-02]
files_modified: []
autonomous: true
requirements: [DOP-02, DOP-03]
estimate:
  tokens: 45000
  raw_tokens: 24000
  tasks: 3
  confidence: high
must_haves:
  truths:
    - "clippy --all-targets --features linalg,parallel is clean (no warnings) — new generics compile with inferred/turbofish types"
    - "Full cargo test + doctests green — the differentiability + non-regression proof for DOP-02 and DOP-03"
    - "28 examples + --features serde build + wasm32 build all compile unchanged (additive non-breaking)"
    - "git diff for the phase is confined to fregre_lm.rs, scalar_on_function/mod.rs, scalar_on_function/tests.rs, smooth_basis.rs, lib.rs, prelude.rs; no new crate dependency"
  artifacts:
    - ".planning/phases/97-differentiable-regression-prediction-smoothing-penalties/97-03-SUMMARY.md"
  key_links:
    - "gate is the end-to-end non-breaking proof that predict_curve_generic + penalty_value_generic are strictly additive"
---

<objective>
Prove the phase is strictly additive and non-breaking: run every whole-crate gate — clippy `--all-targets`, full test suite, doctests, all 28 examples, the `--features serde` build guard, the wasm32 build — and confirm the git diff for the phase is confined to the six touched files with no new crate dependency. This gate is the DOP-02 + DOP-03 end-to-end non-regression proof.

Purpose: guarantee no existing f64 call site, R/WASM binding, or example changed.
Output: 97-03-SUMMARY.md recording every gate result.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/97-differentiable-regression-prediction-smoothing-penalties/97-RESEARCH.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Clippy (--all-targets) + full test + doctest gates (per-gate, foreground, 600s each)</name>
  <files>fdars-core/src/lib.rs</files>
  <read_first>
    - 97-RESEARCH.md lines 591–605 (Phase gate commands + sampling rate)
    - CLAUDE.md build/test hazards: run gates per-gate FOREGROUND (never one combined backgrounded gate); on disk-pressure failure `rm -rf target/debug/{incremental,examples}` and retry
  </read_first>
  <action>Run the three whole-crate correctness gates in sequence, each FOREGROUND with a 600s timeout, NEVER as one combined backgrounded job (per CLAUDE.md — a combined backgrounded gate gets killed mid-run). Gate 1: clippy `--all-targets` (CI lints test/bench code — a plain -p ... -D warnings would miss test-code warnings and false-green). Gate 2: full test suite (the differentiability + non-regression proof — the three predict_curve_generic + three penalty_value_generic tests plus every pre-existing test). Gate 3: doctests. If any gate fails on a disk / "No space left" / linking error, run `rm -rf target/debug/{incremental,examples}` and retry that gate once. No commit in this task (read-only gate).</action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -15 && cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -15 && cargo test -p fdars-core --doc --features linalg,parallel 2>&1 | tail -10</automated>
    <fails_when>clippy emits any warning (treated as error by -D warnings), OR the test tail shows `test result: FAILED`, OR any doctest fails.</fails_when>
  </verify>
  <done>clippy --all-targets is clean; full test suite passes (0 failed) including the six new phase tests; all doctests pass.</done>
</task>

<task type="auto">
  <name>Task 2: 28 examples + serde build + wasm32 build gates (per-gate, foreground, 600s each)</name>
  <files>fdars-core/Cargo.toml</files>
  <read_first>
    - 97-RESEARCH.md lines 591–605 (serde/wasm build guards)
    - CLAUDE.md: keep `--features serde` green; NO new crate dependency; wasm target `wasm32-unknown-unknown --features js`
  </read_first>
  <action>Run the three build gates in sequence, each FOREGROUND with a 600s timeout, never combined/backgrounded. Gate 1: build all 28 examples (`cargo build -p fdars-core --examples --features linalg,parallel`) — proves no example call site changed. Gate 2: `--features serde` build guard. Gate 3: wasm32 build (`--target wasm32-unknown-unknown --features js`) — proves the WASM binding surface compiles unchanged. On any disk/link failure, `rm -rf target/debug/{incremental,examples}` and retry that gate once. No commit.</action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo build -p fdars-core --examples --features linalg,parallel 2>&1 | tail -15 && cargo build -p fdars-core --features serde 2>&1 | tail -10 && cargo build -p fdars-core --target wasm32-unknown-unknown --features js 2>&1 | tail -10</automated>
    <fails_when>Any of the three builds emits `error[` / `error:` in its tail (an example, the serde build, or the wasm build failed to compile).</fails_when>
  </verify>
  <done>All 28 examples build; `--features serde` builds; wasm32 target builds — all unchanged (additive non-breaking).</done>
</task>

<task type="auto">
  <name>Task 3: git-diff churn confinement + no-new-dependency check + SUMMARY commit</name>
  <files>fdars-core/Cargo.toml</files>
  <read_first>
    - 97-RESEARCH.md lines 620–632 (Non-Breaking Proof — churn confined to fregre_lm.rs / smooth_basis.rs + tests + additive re-exports)
    - CLAUDE.md: NO new crate dependency; commit `git commit --no-verify`; `cargo fmt` before commit
  </read_first>
  <action>Assert the phase's source churn is confined to the six files this phase owns: fregre_lm.rs, scalar_on_function/mod.rs, scalar_on_function/tests.rs, smooth_basis.rs, lib.rs, prelude.rs (tests are inline in fregre_lm's tests.rs and smooth_basis.rs). Compute the changed set across the phase's commits (the commits produced by Plan 01 + Plan 02) under fdars-core/src and confirm every changed path is in the allowed set. Confirm Cargo.toml / Cargo.lock have no new dependency (git diff of fdars-core/Cargo.toml shows no added dependency line). Write 97-03-SUMMARY.md recording every gate result from Tasks 1–2 and this confinement check, then `cargo fmt` and `git commit --no-verify` the SUMMARY.</action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && CHANGED=$(git diff --name-only HEAD~5..HEAD -- 'fdars-core/src' | grep -vE 'scalar_on_function/fregre_lm\.rs|scalar_on_function/mod\.rs|scalar_on_function/tests\.rs|smooth_basis\.rs|lib\.rs|prelude\.rs' || true); test -z "$CHANGED" && echo CHURN_CONFINED || { echo "UNEXPECTED_CHURN: $CHANGED"; false; }</automated>
    <fails_when>Prints `UNEXPECTED_CHURN:` followed by any path (a source file outside the six owned files was modified). Note: widen the HEAD~N range if the phase produced more than 5 commits.</fails_when>
  </verify>
  <done>All phase source churn is confined to the six owned files; no new crate dependency in Cargo.toml/Cargo.lock; 97-03-SUMMARY.md written and committed with every gate result.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Verification-only gate over in-crate numeric code; no external input, network, IO, or auth. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-97-03 | (n/a) | non-breaking gate | low | accept | No attack surface — verification gate over pure in-crate numeric code; no external input/network/IO/auth. No packages installed (in-crate only). |
</threat_model>

<verification>
- clippy --all-targets clean; full test + doctests green; 28 examples + serde + wasm32 build; churn confined to the six owned files; no new dependency.
</verification>

<success_criteria>
- Every whole-crate gate passes, proving DOP-02 + DOP-03 are strictly additive and non-breaking.
- git diff confined; no new crate dependency; SUMMARY records all results.
</success_criteria>

<output>
Create `.planning/phases/97-differentiable-regression-prediction-smoothing-penalties/97-03-SUMMARY.md` when done.
</output>