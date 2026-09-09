# Phase 90 Plan Verification

**Phase:** 90-golden-flake-root-cause-deterministic-fix  
**Plan:** 01  
**Checked:** 2026-09-09  
**Verdict:** PASS (with proportionality note on Task 2 gate)

---

## Executive Summary

The plan **will achieve the phase goal** — produce an evidence-backed diagnosis (FLAKE-01) and deterministically fix the three flaky golden tests (FLAKE-02) via a test-side cfg-guard, proven reliable across 10 consecutive full-parallel runs.

All three tasks are complete, specific, verifiable, and respect the locked context decisions. The plan enforces the FLAKE-01 → FLAKE-02 gate correctly. The blocking-human checkpoint (Task 2) is intentional — it validates that the fix is evidence-chosen, not guessed — though proportionately conservative for a one-line attribute change.

---

## Dimension 1: Requirement Coverage

| Requirement | Task | Coverage | Status |
|---|---|---|---|
| **FLAKE-01** | Task 1 (tracer) | Produce committed 90-DIAGNOSIS.md with evidence-backed root cause, reproduce instructions, actual-vs-expected deltas, reconciliation of intermittent-vs-deterministic symptom, and named fdata_to_pc SVD-branch divergence point | ✅ COVERED |
| **FLAKE-02** | Task 2 (gate) + Task 3 (auto) | Confirm evidence-chosen fix, then apply cfg-guard to three named tests; prove reliability across 10 consecutive full-parallel runs and per-binary isolation | ✅ COVERED |

**Finding:** Both requirements explicitly mapped to tasks. Task 1 produces the gate; Task 2 validates the gate; Task 3 implements and proves. No requirement orphaned.

---

## Dimension 2: Task Completeness

| Task | Type | Files | Action | Verify | Done | Reversibility | Status |
|---|---|---|---|---|---|---|---|
| **Task 1** | tracer | ✅ `.planning/…/90-DIAGNOSIS.md` | ✅ Detailed 6-step reproduction, capture deltas, classify, reconcile, locate divergence, write artifact | ✅ `test -s` + `grep` chain for artifact existence, linalg mention, reconciliation language, fdata_to_pc reference | ✅ States FLAKE-01 satisfied | ✅ reversible | ✓ |
| **Task 2** | checkpoint:decision | N/A | ✅ Explicit options (cfg-guard, tolerance, serialize) with pros/cons; cfg-guard expected; aligned with RESEARCH diagnosis | N/A | ✅ States decision approval needed; escalates if evidence differs | N/A | ✓ |
| **Task 3** | auto (tdd) | ✅ Two test files (phase48, phase49) | ✅ Specific: insert `#[cfg_attr(not(feature = "linalg"), ignore)]` above the THREE named tests only; PACE sibling untouched; update module docs; run gates foreground; 10-run acceptance bar | ✅ `cargo test` with feature flags (×3 runs: linalg+parallel, default features, no-default-features+linalg); 10-run loop; grep count check | ✅ Tests pass under linalg, ignored under default, clippy green, no src/Cargo.toml change, 10 runs green, no tolerance relaxed | ✅ reversible | ✓ |

**Finding:** All three tasks fully specified. Task 1's action is concrete (6 numbered steps). Task 3's action names the exact attribute string and line numbers. Verify blocks use automation with explicit `<fails_when>` signals.

---

## Dimension 3: Dependency Correctness

- **Wave assignment:** 1 (Task 1) → 2 (Task 2, blocked on Task 1) → 3 (Task 3, blocked on Task 2)
- **depends_on:** Plan declares `depends_on: []` (Wave 1 only) but contains blocking gates within Wave 1 tasks
- **Clarity:** The internal task sequencing is explicit:
  - Task 1 → must commit FLAKE-01 artifact
  - Task 2 → is a blocking-human gate; executor must confirm before proceeding
  - Task 3 → executes only after Task 2 approval

**Finding:** No circular dependencies, no forward references. Sequential ordering is correct. The `depends_on: []` is at the plan level (this plan doesn't depend on other plans in the phase), but internal task sequencing via blocking gates is properly declared.

---

## Dimension 4: Key Links Planned

**must_haves key_links:**
```
fdata_to_pc SVD feature-branch (src/regression.rs) 
  → co_cluster global FPCA rotation 
  → CEM basin + sign decision
  → divergence point (the cfg-guard target)
```

**Task 1 specifically:** "Locate the single divergence point in source: name the exact `#[cfg(feature = "linalg")]` vs `#[cfg(not(feature = "linalg"))]` SVD branch in `fdata_to_pc` (src/regression.rs)"

**Task 3 specifically:** "Insert `#[cfg_attr(not(feature = "linalg"), ignore)]` on the line immediately ABOVE the `#[test]` attribute for each of these THREE functions"

**Finding:** The key link (SVD backend branch → test behavior) is explicitly named in the must_haves and each task references it. No wiring gap.

---

## Dimension 5: Scope Sanity

| Metric | Value | Target | Status |
|---|---|---|---|
| Tasks/plan | 3 | 2–3 | ✅ Tight |
| Files modified | 3 (two test files + one diagnosis artifact) | 5–8 | ✅ Minimal |
| Lines changed per file | ~5 per test file (one attribute line per function) | — | ✅ Surgical |
| Estimated context | 55K tokens (plan estimate) | ~50% = 50K | ⚠️ At budget |
| Confidence | Low (3 completed phases tracked) | — | ⚠️ Calibration note |

**Finding:** Scope is **tight and minimal**. Three simple tasks with well-defined artifacts. The estimate of 55K is advisory (confidence: low — fewer than 3 completed milestone phases with actuals); the per-file changes are surgical (one attribute per test). No scope creep — Phase 91 sweep and Phase 92/93 CI/release work explicitly deferred.

---

## Dimension 6: Verification Derivation (must_haves)

**must_haves.truths:**

1. **"A committed diagnosis artifact records the evidence-backed root cause (FLAKE-01)"** 
   - User-observable: Yes. The artifact is a tangible deliverable.
   - Task 1 produces it; Task 1 verify checks for its existence.
   - ✅ Proper.

2. **"The three named golden tests pass reliably under repeated full parallel cargo test with --features linalg,parallel (10 consecutive green runs) AND each affected binary passes in isolation"**
   - User-observable: Yes. Test pass/fail is the proof of fix.
   - Task 3 action includes the 10-run loop; Task 3 verify has a `for i in $(seq 1 10)...` command.
   - ✅ Proper.

3. **"Under default features (no linalg) the three tests are reported ignored, not failed"**
   - User-observable: Yes. Test output clearly shows "ignored" vs "failed".
   - Task 3 action includes a gate: `cargo test --features parallel --test equivalence_phase48 -- golden_co_cluster` (expect `2 ignored`).
   - Task 3 acceptance_criteria: "the three tests are reported ignored (`2 ignored` for phase48 co_cluster; `1 ignored` for phase49 svd_sign), not failed."
   - ✅ Proper.

4. **"No new non-dev crate dependency is introduced"**
   - User-observable: Yes. `git diff --name-only` must not include `Cargo.toml` or new dependency files.
   - Task 3 acceptance_criteria: "No `src/` file, `Cargo.toml`, or dependency changed".
   - Task 3 verify checks: `cargo clippy --all-targets` (would fail if Cargo.toml added a dep).
   - ✅ Proper.

**must_haves.artifacts:**
- `.planning/…/90-DIAGNOSIS.md` — task 1 produces
- `fdars-core/tests/equivalence_phase48.rs` — task 3 modifies (adds cfg_attr)
- `fdars-core/tests/equivalence_phase49.rs` — task 3 modifies (adds cfg_attr)

All listed artifacts have covering tasks. ✅

**Findings:** Truths are user-observable, testable, and specific. Artifacts are mapped to tasks. Key links name the SVD backend divergence. No implementation-focused platitudes ("Rust 1.81 compiled") — all truths encode observable outcomes.

---

## Dimension 7: Context Compliance

**From 90-CONTEXT.md Locked Decisions:**

1. **"Fix-approach bias = least-invasive, diagnosis-gated. Prefer test-side robustness."**
   - Plan enforces this: Task 1 produces diagnosis; Task 2 gates the fix choice; Task 3 implements cfg_attr (test-side).
   - ✅ Honored.

2. **"No new crate dependency."**
   - Plan ensures this: Task 3 action says "do not modify any `src/` file, `Cargo.toml`, or add any dependency."
   - Task 3 verification includes `cargo clippy` (would fail if Cargo.toml broken).
   - ✅ Honored.

3. **"Bit-identity may relax to tight tolerance where nondeterminism is proven."**
   - Plan rejects tolerance relaxation for this fix: Task 3 action says "Do NOT relax any `assert_eq!` to a tolerance — bit-identity holds under linalg per the diagnosis."
   - Task 3 acceptance_criteria: "No `assert_eq!` in either test file was converted to a tolerance comparison."
   - ✅ Honored (tolerance explicitly rejected, matching the diagnosis that shows categorical backend divergence, not FP nondeterminism).

4. **"Acceptance bar = 10 consecutive green full parallel `cargo test` runs AND per-binary green."**
   - Plan enforces both: Task 3 action includes `for i in $(seq 1 10); do cargo test -p fdars-core --features linalg,parallel || { echo "RUN $i FAILED"; break; }; done` (10-run bar) AND per-binary isolation gates.
   - Task 3 verify includes the 10-run loop with `<fails_when>` signal.
   - ✅ Honored.

5. **Deferred ideas (Phase 91/92/93 sweeps) must not appear in scope.**
   - Plan objective explicitly states: "Out of scope (do NOT pull in): suite-wide sweep (Phase 91), CI guardrail (Phase 92), release prep / version bump / CHANGELOG (Phase 93)."
   - ✅ Deferred ideas properly excluded.

**Finding:** Plan respects all locked decisions. Diagnosis gate is enforced. Fix choice is evidence-based (Task 2 is the gate). Tolerance relaxation is rejected per the diagnosis. Acceptance bar is explicit and verifiable.

---

## Dimension 8: Nyquist Compliance (Verify Command Format & Fails_When)

**Task 1 verify:**
```
<automated>test -s .planning/…/90-DIAGNOSIS.md && grep -q "linalg" … && grep -qi "intermittent\|reconcil\|per-binary" … && grep -q "fdata_to_pc" …</automated>
<fails_when>non-zero exit (artifact missing/empty, or missing the linalg cause, the intermittent-vs-deterministic reconciliation, or the fdata_to_pc divergence point)</fails_when>
```
- ✅ `test -s` is POSIX-standard (checks file exists and non-empty)
- ✅ `grep -q` suppresses output, exits 0/1 (testable)
- ✅ `fails_when` is explicit: "non-zero exit" + prose listing what causes the failure
- ✅ No pipe ambiguity (all greps are chained with `&&`)

**Task 3 verify (three commands):**
```
1. cargo test -p fdars-core --features linalg,parallel --test equivalence_phase48 --test equivalence_phase49 -- golden_co_cluster svd_sign_fpca_two_matrix_bit_identical
   <fails_when>non-zero exit, or "FAILED" / "test result: FAILED" in output</fails_when>
   ✅ Clear: test output will include one of those strings on failure

2. grep -c 'cfg_attr(not(feature = "linalg"), ignore)' fdars-core/tests/equivalence_phase49.rs
   <fails_when>output is not exactly 1 (PACE test wrongly guarded, or svd_sign test missing the guard)</fails_when>
   ✅ Clear: count must be exactly 1

3. for i in $(seq 1 10); do cargo test -p fdars-core --features linalg,parallel >/dev/null 2>&1 || { echo "RUN $i FAILED"; exit 1; }; done; echo ALL_10_GREEN
   <fails_when>output contains "RUN" and "FAILED", or does not end with "ALL_10_GREEN"</fails_when>
   ✅ Clear: final output must be "ALL_10_GREEN"; any "RUN X FAILED" is a failure signal
```

**Findings:** All verify commands use observable failure signals (exit codes, grep output, string matching). No ambiguous "might fail silently" patterns. `<fails_when>` is explicit for all commands.

---

## Dimension 8b: Feature Flag Compliance

**Critical requirement:** Acceptance gate MUST use `--features linalg,parallel`.

Plan action (Task 3):
- ✅ "Run gates FOREGROUND, per-gate, `timeout: 600000`"
  1. `cargo fmt -p fdars-core` (formatting gate)
  2. `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (lint WITH linalg)
  3. Per-binary: `cargo test -p fdars-core --features linalg,parallel --test equivalence_phase48 -- golden_co_cluster` (WITH linalg)
  4. Ignored-not-failed: `cargo test -p fdars-core --features parallel --test equivalence_phase48 -- golden_co_cluster` (default features, WITHOUT linalg)
  5. No-default-features path: `cargo test -p fdars-core --no-default-features --features linalg --test equivalence_phase48 -- golden_co_cluster` (explicit linalg)
  6. 10-run bar: `for i in $(seq 1 10); do cargo test -p fdars-core --features linalg,parallel || ...` (WITH linalg)

**Finding:** All gates use `--features linalg,parallel` for the acceptance bar. Gate 4 intentionally omits linalg to verify the tests are ignored. This is correct per the diagnosis and RESEARCH.md Pitfall 1.

---

## Dimension 9: Blocking-Human Gate Proportionality

**Task 2 Structure:**
- Type: `checkpoint:decision` with `gate="blocking-human"`
- Purpose: "Confirm the evidence-chosen fix before touching test source"
- Rationale: "This checkpoint is blocking-human because auto-selecting the wrong fix would defeat the diagnosis-gate."

**Analysis:**

The gate is **intentionally conservative** but **appropriate** for this phase's design:

1. **Why it exists:** The phase explicitly enforces a "diagnosis gates fix" pattern (FLAKE-01 → FLAKE-02). A wrong fix choice (e.g., auto-selecting tolerance relaxation instead of cfg_attr despite the diagnosis proving categorical backend divergence) would silently undermine the diagnosis and leave the flake half-fixed.

2. **Proportionality in context:**
   - The fix is **one-line per test** (a cfg_attr), which is reversible and trivial.
   - But the **fix choice is not trivial** — it depends on interpreting the 90-DIAGNOSIS.md evidence correctly. If an autonomous executor misunderstood the diagnosis and relaxed tolerances instead, the tests would appear to pass locally but flake on CI (different context).
   - The gate ensures a human verifies: "Does this diagnosis really point to cfg_attr, or did I misread it and should we serialize instead?"

3. **Decision options provided:**
   - Option 1 (cfg-guard): Least invasive, matches evidence, expected outcome.
   - Option 2 (tolerance): Explicitly marked "WRONG unless FP nondeterminism is proven WITH linalg — the co_cluster delta is 35 log-units (categorical), far beyond any tolerance."
   - Option 3 (serialize): "No race exists (both functions are purely functional); highest blast radius; rejected by the diagnosis."
   - Each option has pros/cons listed; the executor is guided but not forced.

4. **Cost-benefit:**
   - Cost of human gate: ~5 minutes of human time (read diagnosis, confirm cfg_attr is correct, approve).
   - Cost of wrong choice: Phase flakes later, CI gate fails, 1.0 release milestone blocked.
   - Proportional. ✅

**Finding:** The blocking-human gate is **not an over-cautious stall**. It is a proportionate safety valve for a phase that explicitly gates fix choice on evidence. An autonomous executor might read the diagnosis incorrectly (e.g., confuse "4.178e-16 drifts" with "FP nondeterminism" and relax tolerances). The gate prevents that silent failure mode.

**Note:** An autonomous GSD run would halt here, present the diagnosis and the three options, and wait for human approval before proceeding to Task 3. This is intentional and correct.

---

## Dimension 10: CLAUDE.md Compliance

Relevant project constraints from `.planning/PROJECT.md` and `.claude/CLAUDE.md`:

1. **No code changes outside GSD workflow** — Plan uses `/gsd-execute-phase`
   - ✅ Plan is for execution within GSD; no ad-hoc edits recommended.

2. **Commit gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`**
   - ✅ Task 3 action includes: "2. `cargo clippy --all-targets --features linalg,parallel -- -D warnings`"

3. **Format gate: `cargo fmt` per commit**
   - ✅ Task 3 action includes: "1. `cargo fmt -p fdars-core`"

4. **Run gates FOREGROUND with `timeout: 600000`**
   - ✅ Task 3 action says: "Run gates FOREGROUND, per-gate, `timeout: 600000`"

5. **No --no-verify bypass unless after out-of-band gate confirmation**
   - ✅ Task 3 action: "Commit with `git commit --no-verify` after gates pass (pre-commit hook times out)." — justified by the MEMORY.md hazard note.

6. **Preserve bit-identity for assertions (no tolerance relaxation)**
   - ✅ Task 3 action: "Do NOT relax any `assert_eq!` to a tolerance — bit-identity holds under linalg per the diagnosis."
   - ✅ Task 3 acceptance_criteria: "No `assert_eq!` in either test file was converted to a tolerance comparison."

**Finding:** Plan respects all project-specific conventions and hazards. Build/disk mitigation is acknowledged.

---

## Dimension 11: Research Resolution

**RESEARCH.md status:** "Open Questions (RESOLVED)" check

From 90-RESEARCH.md:
```
## Open Questions

1. **Is there a nalgebra-path golden for co_cluster that could serve as a second target?**
   - Recommendation: Defer to Phase 91 (robustness sweep). For Phase 90, the cfg_attr guard is sufficient.

2. **What happens to per-binary `cargo test --test equivalence_phase48` (no features) after the fix?**
   - Answer: The 2 co_cluster golden tests become "ignored" (2 ignored, 3 passed). This is the correct behavior...
```

The RESEARCH document **addresses all open questions** (questions 1 and 2 are answered; the phase explicitly defers nalgebra-golden double-maintenance to Phase 91). The questions are answered in the document, though the section heading is not explicitly marked "(RESOLVED)".

**Finding:** Open questions are answered in prose. For stricter compliance with Dimension 11 (Research Resolution), the RESEARCH.md section heading should read `## Open Questions (RESOLVED)` to match the expected marker. However, the **content** is fully resolved. This is a **minor documentation polish** (WARNING, not blocker).

---

## Dimension 12: Pattern Compliance

**PATTERNS.md status:** No PATTERNS.md file found in the phase directory or parent.

The phase modifies test attributes only (no new application code that would need a pattern analog). Test fixtures (`co_cluster_data`, `fpca_sign_fixture`, `pace_sign_fixture`) are pre-existing helpers used by existing tests; no new patterns are introduced.

**Finding:** Dimension 12 is not applicable (no PATTERNS.md exists for this phase, and the changes are test-only attributes, not new application code). SKIPPED (expected).

---

## Summary of Issues

### Blockers
**None.** All tasks are complete, specific, verifiable, and respect context decisions.

### Warnings

**W-01 [research_resolution]:** RESEARCH.md section "## Open Questions" should be marked "(RESOLVED)" for strict compliance.
- **File:** 90-RESEARCH.md
- **Current:** `## Open Questions` (heading)
- **Expected:** `## Open Questions (RESOLVED)` (per Dimension 11 compliance spec)
- **Impact:** Minor — the content is fully resolved; the marker is missing.
- **Fix:** One-line edit to RESEARCH.md heading.
- **Severity:** WARNING (does not block execution; pure documentation polish)

### Info Notes

**I-01 [scope_estimate]:** Plan's `estimate.confidence: low` is correct.
- Only 3 milestones with completed phases tracked (per project memory). Calibration will improve as more phases complete.
- The estimate of 55K tokens is advisory; actual usage may vary.

**I-02 [gate_proportionality]:** Task 2's blocking-human gate is proportionate.
- Prevents silent wrong-fix-choice (tolerance vs cfg_attr) that could cause later flakes.
- Autonomous runs will halt here; human approval is required.
- This is intentional per the phase's "diagnosis gates fix" design.

---

## Verification Findings Summary

| Dimension | Result | Notes |
|---|---|---|
| 1. Requirement Coverage | ✅ PASS | FLAKE-01 and FLAKE-02 both mapped to tasks |
| 2. Task Completeness | ✅ PASS | All tasks have files, action (concrete steps), verify (automated), done (acceptance criteria) |
| 3. Dependency Correctness | ✅ PASS | No circular deps; internal sequencing via blocking gates is correct |
| 4. Key Links Planned | ✅ PASS | SVD backend divergence point named; Task 1 → Task 2 → Task 3 wiring is explicit |
| 5. Scope Sanity | ✅ PASS | 3 tasks, 3 files modified, ~5 lines per file; surgical changes; estimate at budget |
| 6. must_haves Derivation | ✅ PASS | Truths are user-observable (artifact, test pass, ignored behavior, no deps); artifacts mapped |
| 7. Context Compliance | ✅ PASS | All locked decisions honored; fix choice gated by evidence; deferred ideas excluded |
| 8. Nyquist Compliance | ✅ PASS | All verify commands have explicit `<fails_when>` signals; feature flags correct |
| 9. Blocking-Human Gate | ✅ PROPORTIONATE | Gate prevents silent wrong-fix-choice; proportionate for this phase's design |
| 10. CLAUDE.md Compliance | ✅ PASS | All project conventions respected (clippy, fmt, timeout, no-verify justification, bit-identity) |
| 11. Research Resolution | ⚠️ WARNING | Content resolved; section heading should be marked "(RESOLVED)" (one-line fix) |
| 12. Pattern Compliance | ✅ SKIPPED | No PATTERNS.md; changes are test attributes only (no new patterns introduced) |

---

## FINAL VERDICT: **PASS**

The Phase 90 plan **will achieve the phase goal** under execution.

- ✅ FLAKE-01 (evidence-backed diagnosis) will be produced and committed by Task 1.
- ✅ FLAKE-02 (deterministic fix) will be applied by Task 3 after evidence review (Task 2 gate).
- ✅ Acceptance bar (10 consecutive full-parallel runs + per-binary green) is explicitly verified.
- ✅ All locked context decisions honored.
- ✅ No blockers.

**Single warning (non-blocking):** RESEARCH.md section heading `## Open Questions` should be marked "(RESOLVED)" for strict compliance — one-line fix.

### Recommendation

Execute the plan as written. Before execution, optionally fix the RESEARCH.md section heading `## Open Questions` → `## Open Questions (RESOLVED)` for documentation consistency (one-line edit).

---

## Checklist for Executor

Before running `/gsd-execute-phase`:

- [ ] Read 90-CONTEXT.md decisions (locked, do not invent alternatives)
- [ ] Read 90-RESEARCH.md diagnosis (the root cause will be reproduced in Task 1)
- [ ] Ensure `/home` has >10GB free disk (full test runs can be heavy)
- [ ] Prepare to run Task 1 gates FOREGROUND with `timeout: 600000`
- [ ] **Task 2 is blocking-human:** After Task 1 commits 90-DIAGNOSIS.md, review it carefully and confirm "cfg-guard is the right fix per the evidence" before approving Task 3
- [ ] Task 3 includes 10-run acceptance bar — ensure machine can sustain that without hitting resource limits

