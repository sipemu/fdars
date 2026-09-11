---
phase: "95"
plan: "03"
type: execute
wave: 3
depends_on: ["95-02"]
files_modified: []
autonomous: true
requirements: [GEN-01]
estimate:
  tokens: 45000
  raw_tokens: 30000
  tasks: 3
  confidence: med
must_haves:
  truths:
    - "All 28 examples + the R/WASM binding surfaces compile unchanged against the generalized signatures (GEN-01)."
    - "The `--features serde` build stays green (GEN-01)."
    - "The full test suite + doctests pass with the generalized kernels (GEN-01)."
    - "`git diff --stat` against the phase base shows ONLY helpers.rs/utility.rs/warping.rs changed — no call-site edits (GEN-01)."
  artifacts:
    - "Compile-gate evidence recorded in 95-03-SUMMARY.md (per-gate pass/fail + churn diff)"
  key_links:
    - "Generalized kernels ↔ 28 examples (all pass &[f64], infer T=f64)"
    - "Generalized kernels ↔ wasm32-unknown-unknown target"
    - "fdars-core public signatures ↔ external fdars-r package (source-level f64-only usage grep)"
  prohibitions:
    - statement: "No source edit in this plan — it is the read-only GEN-01 compile-time proof"
    - statement: "No new crate dependency added to Cargo.toml"
---

<objective>
The NON-BREAKING COMPILE-GATE — the first-class GEN-01 deliverable. With all four kernels generalized (plans 01–02), prove non-breakingness at compile time across every surface: the crate, all 28 examples, the `--features serde` build, the `wasm32-unknown-unknown` target, clippy `--all-targets`, the full test suite, doctests, a `git diff --stat` churn check, and a source-level grep of the external `fdars-r` package for f64-only usage of the four kernels.

Purpose: This gate IS the deliverable as much as the signatures are — it is the compile-time proof that the generalization broke nothing. No source changes; if a gate fails, surface the exact failing surface (do not silently patch a call site).
Output: Evidence in 95-03-SUMMARY.md that every gate is green and churn is confined to the three kernel files.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/95-generic-scalar-hot-path-signatures/95-CONTEXT.md
@.planning/phases/95-generic-scalar-hot-path-signatures/95-RESEARCH.md
@.planning/phases/95-generic-scalar-hot-path-signatures/95-01-SUMMARY.md
@.planning/phases/95-generic-scalar-hot-path-signatures/95-02-SUMMARY.md
</context>

<build_and_commit_constraints>
- Run each cargo gate OUT-OF-BAND, per-gate, FOREGROUND, 600s timeout. NEVER one combined backgrounded gate (the combined run gets killed — MEMORY.md).
- On "No space left" / "linking with cc failed": `rm -rf target/debug/{incremental,examples}` then retry (not a code bug; doctests link in a small /tmp tmpfs).
- serde feature build is known-fragile (ShapeletTransformClassifier/ClassifFit, MEMORY.md): if it fails, confirm via `git stash`-free reasoning that the failure PRE-DATES this phase (compare against the phase base commit) before treating it as a regression.
- This plan makes NO source edits. If a gate exposes a genuine break caused by the generalization, STOP and surface it.
</build_and_commit_constraints>

<tasks>

<task type="auto">
  <name>Task 1: Churn check + pre-edit safety greps + clippy/test/doctest gates</name>
  <files></files>
  <read_first>
    - 95-RESEARCH.md §Compile-Gate Strategy (exact gate commands + ordering) and §Non-Churn Verification.
    - MEMORY.md hazards: CI clippy uses --all-targets; disk pressure; serde build fragility.
  </read_first>
  <action>
    First establish the phase base: `BASE=$(git merge-base HEAD HEAD)` is not meaningful here — instead record the commit BEFORE plan 01 (the parent of the first 95 commit) as the churn baseline, e.g. `git log --oneline | grep -m1 "milestone v0.44.0"` region; practically, run `git diff --stat <phase-base>..HEAD` where `<phase-base>` is the commit immediately before the plan-01 tracer commit. Run the two pre-edit safety greps from 95-RESEARCH.md §Pre-Edit Audit for ALL four kernels (fn-pointer coercions; `: f64 = <kernel>(` annotations) and confirm both return nothing. Free disk if needed (`rm -rf target/debug/{incremental,examples}`). Then run, each FOREGROUND with 600s timeout, in order: Gate 1 clippy `--all-targets`; Gate 2 full test suite; Gate 6 doctests. Record each gate's pass/fail in the SUMMARY.
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -15 && cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -15 && cargo test -p fdars-core --doc --features linalg,parallel 2>&1 | tail -10</automated>
    <fails_when>clippy emits a warning/error, any test or doctest fails, or a pre-edit grep finds a fn-pointer coercion / `: f64 =` annotation on a kernel (which would mean a real turbofish break).</fails_when>
  </verify>
  <acceptance_criteria>
    - clippy --all-targets clean with -D warnings.
    - Full test suite + doctests pass (co_cluster/svd_sign golden flake, if it appears, is a known env flake — verify the specific test in isolation per MEMORY.md before treating as a regression).
    - Both safety greps return nothing.
  </acceptance_criteria>
  <done>The crate lints clean on all targets and the full test + doctest suites pass against the generalized kernels; no fn-pointer/annotation break exists.</done>
</task>

<task type="auto">
  <name>Task 2: Examples + serde + WASM build gates</name>
  <files></files>
  <read_first>
    - 95-RESEARCH.md §Compile-Gate Strategy Gates 3/4/5 (serde, examples, wasm commands) and §Environment Availability (wasm target installed).
    - MEMORY.md: serde feature build broken pre-existing (ShapeletTransformClassifier/ClassifFit) — is a phase-independent baseline, not a GEN-01 regression.
  </read_first>
  <action>
    Run, each FOREGROUND with 600s timeout: Gate 4 all 28 examples `cargo build -p fdars-core --examples --features linalg,parallel`; Gate 3 serde build `cargo build -p fdars-core --features serde`; Gate 5 WASM `cargo build -p fdars-core --target wasm32-unknown-unknown --features js`. Free disk between gates if a link error appears. If the serde build fails, diff the failure against the phase-base commit to confirm it is the pre-existing ShapeletTransformClassifier/ClassifFit breakage (MEMORY.md) and NOT introduced by the kernel generalization — record that determination explicitly in the SUMMARY. If wasm target is missing, `rustup target add wasm32-unknown-unknown` first (no code change).
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && cargo build -p fdars-core --examples --features linalg,parallel 2>&1 | tail -15 && cargo build -p fdars-core --target wasm32-unknown-unknown --features js 2>&1 | tail -10</automated>
    <fails_when>Any of the 28 examples fails to compile, the WASM target build fails, or (separately) the serde build fails for a NEW reason attributable to the generalization rather than the documented pre-existing ClassifFit issue.</fails_when>
  </verify>
  <acceptance_criteria>
    - All 28 examples build.
    - WASM `wasm32-unknown-unknown` target builds.
    - serde build is either green or fails ONLY for the pre-existing documented reason (recorded in SUMMARY).
  </acceptance_criteria>
  <done>Examples and WASM surfaces compile unchanged against the generalized signatures; serde status characterized against the phase baseline.</done>
</task>

<task type="auto">
  <name>Task 3: git-diff-stat churn confirmation + fdars-r source grep + SUMMARY commit</name>
  <files></files>
  <read_first>
    - 95-RESEARCH.md §Non-Churn Verification and §Compile-Gate Gate 7 (fdars-r source grep — external package, may be absent locally).
    - 95-RESEARCH.md Assumptions A2 (fdars-r wraps high-level functions, not these low-level helpers).
  </read_first>
  <action>
    Confirm the churn diff: `git diff --stat <phase-base>..HEAD` must list ONLY fdars-core/src/helpers.rs, fdars-core/src/utility.rs, fdars-core/src/warping.rs (plus the .planning docs/state). Any src file outside those three appearing in the diff is a churn failure — surface it. Locate the external fdars-r package source if present locally (check common sibling paths, e.g. `../fdars-r`, `~/projects/rust/fdars-r`); if found, grep `l2_distance|inner_product|trapz|inner_product_l2` in its Rust source and confirm every usage passes f64 data (they wrap high-level functions, not these helpers). If fdars-r is not local, record "fdars-r not local — signature-kind check only: the four kernels remain pub and the f64 call form is proven by Gates 2+4" in the SUMMARY. Write 95-03-SUMMARY.md with the per-gate pass/fail table, the churn diff, and the fdars-r determination; `cargo fmt -p fdars-core` (no-op expected), remove `.git/index.lock` if present, and `git commit --no-verify -m "docs(95): non-breaking compile-gate evidence (GEN-01)"` (commits only the SUMMARY + planning docs).
  </action>
  <verify>
    <automated>cd /home/simonm/projects/rust/fdars && CHANGED=$(git diff --name-only HEAD~2..HEAD -- 'fdars-core/src/*.rs' | grep -vE 'helpers\.rs|utility\.rs|warping\.rs' || true); test -z "$CHANGED" && echo CHURN_CONFINED || { echo "UNEXPECTED: $CHANGED"; false; }</automated>
    <fails_when>Any src file other than helpers.rs/utility.rs/warping.rs appears in the phase diff (CHURN_CONFINED not printed), or the SUMMARY commit touches a source file.</fails_when>
  </verify>
  <acceptance_criteria>
    - The phase src diff is confined to helpers.rs, utility.rs, warping.rs (CHURN_CONFINED printed).
    - fdars-r usage determination is recorded in the SUMMARY (either f64-only grep result or the not-local signature-kind note).
    - 95-03-SUMMARY.md exists with the per-gate pass/fail table.
  </acceptance_criteria>
  <done>Churn is proven confined to the three kernel files; the fdars-r surface is characterized; the compile-gate evidence is recorded and committed. GEN-01 is proven non-breaking at compile time.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

No attack surface — this plan runs read-only compile/lint/test gates over in-place-generalized pure numeric kernels; no I/O, network, deserialization, or privilege boundary.

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-95-03 | Tampering | non-breaking guarantee (call-site churn) | low | mitigate | `git diff --stat` churn gate proves only the three kernel files changed; the 28-example + serde + wasm builds prove zero call-site churn. |
</threat_model>

<verification>
- Gates 1–6 (clippy --all-targets, full test, doctests, 28 examples, serde, wasm) all green (serde characterized against baseline).
- `git diff --stat <phase-base>..HEAD` src diff confined to helpers.rs/utility.rs/warping.rs.
- fdars-r usage determination recorded.
</verification>

<success_criteria>
Every compile-gate surface is green (or serde failure characterized as pre-existing); the source churn is confined to the three kernel files; GEN-01 is proven non-breaking at compile time.
</success_criteria>

<output>
Create `.planning/phases/95-generic-scalar-hot-path-signatures/95-03-SUMMARY.md` when done.
</output>