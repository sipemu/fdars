# Phase 67 Plan Verification — Automatic λ Selection (GCV + REML)

**Verification Date:** 2026-09-04  
**Verified by:** Claude  
**Phase Goal:** Add automatic smoothing-parameter (λ) selection to Phase 66 `peer()` via GCV grid search or REML/mixed-model estimation, selectable via `LambdaChoice` enum, with explicit `Fixed(λ)` honored verbatim.  
**Plan:** `67-01-lambda-selection-tracer-PLAN.md`  
**Status:** ⚠️ **NEEDS REVISION** (see blockers below)

---

## Summary

The plan is **well-structured** and addresses the four ROADMAP success criteria with concrete task decomposition. **However**, three blockers must be fixed before execution:

1. **BLOCKER: Task 1 wiring incomplete** — The dispatch placeholder logic for `Gcv` and `Reml` arms is documented but the actual mechanism for routing to "Task 2/Task 3 wire selectors here" is vague. The tracer discipline requires explicit confirmation that the placeholder path *itself* is syntactically valid and compiles.

2. **BLOCKER: REML EM implementation underspecified** — The action prose describes the EM algorithm in high-level terms, but critical numerical details (GLS update via Woodbury, exact trace computations for M-step σ²_e, handling of singular Z_null'Z_null) need explicit Rust pseudocode to verify feasibility within task scope.

3. **BLOCKER: Task 1 acceptance criteria insufficient** — Criterion "All 12 Phase 66 tests compile and pass" should specify *exactly* which tests exist and how the `PeerConfig::default()` sites behave post-migration. The RESEARCH confirms 8 explicit `PeerConfig{...}` literals, but the current plan text lists "12 Phase 66 tests" without reconciling this count.

---

## Verification by Dimension

### 1. Requirement Coverage ✅ PASS

| Requirement | Plan | Tasks | Coverage |
|-------------|------|-------|----------|
| PER-03 (all 4 success criteria) | 67-01 | 1–4 | Full — each criterion maps to a test + acceptance criteria |

**Finding:** The plan's `must_haves.truths` **directly instantiate all four ROADMAP success criteria** as testable truths:
- Truth 1: GCV determinism + grid-argmin + β recovery
- Truth 2: REML positivity + agreement with GCV + β recovery  
- Truth 3: Fixed(λ) verbatim + gcv=None + lambda_method=Fixed
- Truth 4: Non-degenerate λ on SNR data, recovery within tolerance

**No requirement gap detected.**

---

### 2. Task Completeness ⚠️ NEEDS REVISION

#### Task 1: Tracer — API evolution
- ✅ `<files>` specified: `fdars-core/src/peer.rs`
- ✅ `<read_first>` includes Phase 66 code locations (PeerPenalty template, peer() body, tests)
- ✅ `<action>` describes enum definitions, field changes, test migration
- ❌ **ISSUE: Placeholder logic vague** — Action says:
  > "For the `Gcv` and `Reml` arms, temporarily route them through the Fixed path using a placeholder lambda equal to the old default (`1.0`) so the code compiles"
  
  **Problem:** This is ambiguous about whether the dispatch is a match-on-LambdaChoice (syntactically complex) or a fallthrough. The exact code pattern for the placeholder is not given. The acceptance criteria require "compiles and passes" but do NOT verify that the placeholder is actually syntactically valid as written.

  **Recommendation:** Add a code snippet to the action showing the exact placeholder dispatch form:
  ```rust
  let (lambda, gcv, lambda_method) = match &config.lambda {
      LambdaChoice::Fixed(lam) => (*lam, None, LambdaMethod::Fixed),
      LambdaChoice::Gcv => (1.0, None, LambdaMethod::Fixed),  // <-- PLACEHOLDER
      LambdaChoice::Reml => (1.0, None, LambdaMethod::Fixed), // <-- PLACEHOLDER
  };
  ```

- ✅ `<verify>` commands: two `cargo test` calls, one for module tests, one for doctests
- ⚠️ `<fails_when>` acceptable but could be stricter on "compile error naming `lambda`..." — should also name `LambdaChoice` enum not found

#### Task 2: GCV selector
- ✅ `<files>`, `<read_first>` solid (cites function_on_scalar.rs and linalg.rs)
- ⚠️ `<action>` is prose-level design; no Rust pseudocode provided
  - Describes the grid loop, RSS computation, tr(H) guard conceptually
  - Inline test descriptions (determinism, known-answer) are clear
  
- ✅ `<verify>` commands target the specific GCV tests; `<fails_when>` captures determinism failure + regression
- ✅ `<done>` statement is concrete: "GCV runs an automatic deterministic grid search..."

**No critical issue**, but the action lacks concrete pseudocode.

#### Task 3: REML EM selector
- ✅ `<files>`, `<read_first>` well-chosen (fpca_variants eigendecomposition pattern, linalg Cholesky, famm structure confirmation)
- ❌ **BLOCKER: Action is high-level design, not implementation pseudocode**
  
  The action describes:
  1. Eigendecompose Q via `DMatrix::from_row_slice(...).symmetric_eigen()` — ✅ clear
  2. Partition null/range by eigenvalue tolerance — ✅ clear
  3. Build Z_null and Z_range — ✅ clear
  4. Init σ²_e, σ²_u, α — ✅ clear
  5. **EM loop (items 5–6):** The text says:
     > "E-step: form `ZtZ_range` (r*r), `M = ZtZ_range/sigma2_e + (1/sigma2_u)*I_r`, invert via Cholesky to get `Sigma_b`; `b_hat = Sigma_b * Z_range' * r_alpha / sigma2_e` where `r_alpha = yc - Z_null*alpha`. M-step: `sigma2_u = (b_hat'b_hat + tr(Sigma_b)) / r`; `sigma2_e = (norm2(r_alpha - Z_range*b_hat) + tr(Z_range Sigma_b Z_range')) / n` ... When `s > 0`, re-estimate `alpha` via a GLS/Woodbury update..."

     **Problems:**
     - The Woodbury identity for GLS is cited but NOT written out. A single error in the Woodbury expansion kills the fit.
     - The trace computation `tr(Z_range Sigma_b Z_range')` is terse; it maps to `tr(Sigma_b ZtZ_range)` per RESEARCH, but verifying this chain in Rust requires explicit computation (column-wise sum? row-wise sum?).
     - "re-estimate `alpha` via GLS/Woodbury" — no pseudocode for which linear system is solved or how Woodbury enters the solve.

  **Impact:** A developer reading this must reverse-engineer the math from the RESEARCH section plus interpret "GLS/Woodbury" on their own. The EM loop is the most complex part; a **high-risk task for correctness**.

- ✅ `<behavior>` tests are clear: determinism, positivity, agreement with GCV, edge cases (Ridge, zero-Q)
- ✅ `<verify>` commands match the test names; `<fails_when>` captures the key failure modes
- ✅ `<done>` statement is concrete

**Verdict:** Task 3 is high-risk due to EM pseudocode being under-specified.

#### Task 4: Phase gate
- ✅ Straightforward: run full suite, clippy, fmt
- ✅ `<verify>` command is comprehensive
- ✅ `<done>` is concrete
- **No issues**

---

### 3. Dependency Correctness ✅ PASS

- Plan `depends_on: ["66-01"]` — Phase 66 core estimator must exist. ✅ Phase 66 is complete (per ROADMAP).
- All tasks in a single plan (wave 1), so no inter-task dependency complexity.
- No forward references or circular dependencies.

**Finding:** Dependency is clean and necessary.

---

### 3b. Undeclared Coupling ✅ PASS

Single plan, single wave — no coupling risk between plans.

---

### 4. Key Links Planned ✅ PASS

`must_haves.key_links`:
1. "peer() dispatch on config.lambda → selected λ flows into the existing cholesky_solve of (WtW + λQ)."
2. "GCV inner loop reuses compute_peer_trace_hat (Phase 66) for tr(H)."
3. "REML eigendecomposition of Q via nalgebra symmetric_eigen partitions null (fixed) vs range (random) space."

**Verification:**
- Link 1: Task 1 wires the dispatch; Tasks 2–3 plug in the selectors. Flows into the existing solve at line 256 of peer.rs. ✅
- Link 2: Task 2 action cites `compute_peer_trace_hat(..., m, n)` call inside the GCV loop. ✅
- Link 3: Task 3 reads fpca_variants.rs pattern and deploys `DMatrix::symmetric_eigen()`. ✅

**All key links are planned and connected.**

---

### 5. Scope Sanity ⚠️ WARNING

| Metric | Value | Assessment |
|--------|-------|------------|
| Tasks | 4 | Within target (2–3 good; 4 acceptable; 5+ warning) |
| Files modified | 1 (peer.rs) | Well-scoped (no new files, no new dependencies) |
| Estimated tokens | 92,000 (raw: 61,000) | ~50% of budget; confidence=low |

**Confidence note:** RESEARCH marked confidence=low on the REML EM assumptions (A1 — standard smoothing mixed-model math not cross-verified against a primary source). The raw token estimate (61k) suggests the plan is feasible, but the "low confidence" reflects uncertainty in the EM complexity.

**Recommendation:** The scope is acceptable, but the low-confidence estimate plus the under-specified EM action (see §2 above) create execution risk. The tracer task (Task 1) is low-risk and should complete quickly; Task 2 (GCV) is moderate-risk (grid loop + existing helpers); Task 3 (REML) is high-risk (novel EM implementation).

---

### 6. Verification Derivation ✅ PASS

`must_haves.truths` are all **user-observable**:
- "GCV runs an automatic grid search and returns the GCV-minimizing λ recorded in `PeerResult.lambda` + `PeerResult.gcv`"
- "REML fits λ via self-contained mixed-model EM, returns a positive finite λ"
- "Explicit `LambdaChoice::Fixed(λ)` is used verbatim"
- "Both selectors pick a non-degenerate λ and recover known β(t)"

**Not implementation-focused** (e.g., no "nalgebra symmetric_eigen installed"), **all testable**, **directly trace back to ROADMAP success criteria**.

**Artifacts support the truths:**
- `LambdaChoice`, `LambdaMethod` enums — used to select and report which path ran
- `PeerResult.gcv`, `lambda_method` — carry the GCV score and method marker
- Private helpers (`select_lambda_gcv_peer`, `select_lambda_reml_peer`) — implement the selectors

**Findings:** must_haves derivation is sound.

---

### 7. Context Compliance ✅ PASS

**Locked decisions from CONTEXT.md:**
1. `PeerConfig.lambda: f64` → `lambda: LambdaChoice` enum — **Plan complies.** Task 1 implements this exactly.
2. `LambdaChoice::Fixed(λ)` verbatim, no search — **Plan complies.** Task 1 wires the Fixed arm; Task acceptance criteria require `gcv == None`, `lambda_method == Fixed`, and `lambda` bit-exact.
3. GCV: fixed 40-point log-spaced grid [1e-6, 1e4], deterministic tie-break by smaller grid index — **Plan complies.** Task 2 action specifies the grid formula and ties → "first occurrence wins".
4. REML: self-contained EM via eigendecomposition of Q, NOT `famm::fit_scalar_mixed_model` — **Plan complies.** Task 3 action explicitly does NOT call famm; builds Z_null/Z_range and runs EM locally.
5. `PeerResult.gcv: Option<f64>`, `lambda_method: LambdaMethod` — **Plan complies.** Task 1 defines these fields.
6. Tests: mechanically migrate Phase 66 `PeerConfig { lambda: <f64> }` → `LambdaChoice::Fixed(<f64>)` — **Plan complies.** Task 1 action describes the migration.

**Deferred ideas (out of scope):**
- Longitudinal `lpeer` — not mentioned in the plan. ✅
- Out-of-sample `predict` — not mentioned. ✅
- Crate-root/prelude exports — not mentioned. ✅
- User-configurable GCV grid — not mentioned. ✅

**Finding:** Plan honors all locked decisions and excludes all deferred ideas. **Context-compliant.**

---

### 7b. Scope Reduction Detection ✅ PASS

Scanning the plan for scope-reduction language:
- No "v1", "simplified", "static for now", "hardcoded" in the action prose.
- No "future enhancement", "placeholder", "basic version", "minimal" claims.
- Task 3 mentions "For simplicity in Phase 67, we do NOT call `famm::fit_scalar_mixed_model`" — this is **not** a scope reduction; it's a design decision locked in CONTEXT.md and necessary for the tracer architecture. The full REML EM is **included**, not deferred.

**Finding:** No scope reduction detected. Plans deliver the full decision scope.

---

### 7c. Architectural Tier Compliance ⚠️ SKIPPED

**Condition:** RESEARCH.md exists and contains a responsibility map.

**Check:** Read `67-RESEARCH.md` — **YES**, it includes §Architectural Responsibility Map:

| Capability | Primary Tier | Rationale |
|------------|-------------|-----------|
| GCV grid search | peer.rs (new fn) | Reuses Phase 66 helpers |
| REML EM loop | peer.rs (new fn) | Self-contained; no new file/module |
| Variance update math | peer.rs inline | Port equations, not function calls |
| LambdaChoice/Method enums | peer.rs | Live next to PeerConfig/PeerResult |

**Verification:** All three selector capabilities (`select_lambda_gcv_peer`, `select_lambda_reml_peer`, lambda dispatch in `peer()`) **route through the same tier** — `peer.rs`, the module that owns the estimator. This aligns with the responsibility map.

**Finding:** Architectural tier compliance is sound.

---

### 8. Nyquist Compliance ⚠️ NEEDS VERIFICATION AT EXECUTION

Per `67-VALIDATION.md`:
- Framework: Rust built-in test harness (inline `#[cfg(test)]` modules)
- Sampling: After every task commit, run module tests (fast); full suite before phase verification
- Wave 0: Extend inline tests with SNR fixture, mechanically migrate Phase 66 tests

**Plan does reference the SNR fixture** (`make_fixture()`, reused from Phase 66) in Task 2 and 3 behavior blocks.

**Check: Does the plan ensure continuous automated feedback?**
- Task 1: Two `cargo test` calls (module + doc tests) — ✅
- Task 2: GCV-specific tests (`test_peer_gcv_deterministic`, `test_peer_gcv_recovers_beta`) plus regression (`peer::` module) — ✅
- Task 3: REML-specific tests plus regression — ✅
- Task 4: Full suite — ✅

**Check: Are verify commands using patterns that can fail?**
- Task 1–3 verify blocks use `<automated>cargo test ... </automated>` with `<fails_when>non-zero exit or "test result: FAILED"` — ✅ Standard Rust test framework format, reliable.
- Task 4 uses a single chained command `cargo test && cargo clippy && cargo fmt -- --check` — ✅ Each gate fails fast on error.

**Sampling continuity:** No three consecutive tasks without automated verify. Tasks 1–4 all have verify blocks.

**Finding:** Nyquist compliance cannot be fully verified until execution, but the plan structure (4 tasks with automated verify on every task) is sound. The `<fails_when>` patterns are standard and reliable.

---

### 9. Cross-Plan Data Contracts ✅ PASS

Single plan, single modified file — no cross-plan data sharing. NA.

---

### 10. CLAUDE.md Compliance ✅ PASS

**Project-specific conventions from `/home/simonm/projects/rust/fdars/.claude/CLAUDE.md`:**

| Directive | Status |
|-----------|--------|
| Rust 2021, MSRV 1.81 | Unchanged; Phase 67 uses nalgebra 0.33 (already in Cargo.toml) |
| No new dependencies | ✅ No new `Cargo.toml` entry; nalgebra reused from fpca_variants.rs |
| All public fns return `Result<T, FdarError>` | ✅ peer() already does; selectors are private helpers |
| `#[derive(Debug, Clone, PartialEq)]` on public types | ✅ LambdaChoice, LambdaMethod have these; Plan 1 action specifies the derives |
| Conditional serde via `#[cfg_attr(...)]` | ✅ Plan 1 action copies PeerPenalty's serde derive pattern |
| Deterministic numerics; no RNG | ✅ Both selectors explicitly deterministic; no `rand` imported |
| Inline `#[cfg(test)]` tests | ✅ Tests extend the existing module tests block |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` for CI | ✅ Task 4 runs this exact command |
| `cargo fmt` for format compliance | ✅ Task 4 runs this |

**Finding:** Plan respects all project conventions.

---

### 11. Research Resolution ✅ PASS

**RESEARCH.md has §Assumptions Log:**

| # | Claim | Risk |
|----|-------|------|
| A1 | REML EM update equations follow standard smoothing-spline mixed model | Tested by known-answer tests; loose REML/GCV tolerance (0.2) accommodates potential differences |
| A2 | nalgebra `symmetric_eigen()` numerically sound for ≤40×40 PSD matrices | Low risk; nalgebra is production-quality |
| A3 | refund `peer()` defaults to REML; GCV/REML tolerance achievable on fixture | If refund differs, tolerance can be loosened; tests will reveal |
| A4 | GLS α-update via Woodbury is stable for s ≤ 5 | Mitigated by 1e-10 ridge regularization mentioned in RESEARCH |
| A5 | 40-pt grid covers optimal λ | Detected by β-recovery tests; boundary values cause > 0.15 error |

**Section heading:** "## Open Questions" → None in the RESEARCH. All questions are resolved or explicitly listed as assumptions.

**Finding:** Research is complete; no unresolved open questions blocking the plan.

---

### 12. Pattern Compliance ⚠️ SKIPPED

**Condition:** No `PATTERNS.md` exists for Phase 67.

**Output:** Dimension 12 is SKIPPED.

---

### Verify Command Format Sanity ⚠️ MINOR

Task 4 verify block chains three commands:
```bash
cargo test -p fdars-core --features linalg,parallel && cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo fmt -- --check
```

**Pattern check:**
- ✅ `cargo test` — standard, no grep pipes with anchors
- ✅ `cargo clippy --all-targets` — correct per MEMORY.md CI gate
- ✅ `cargo fmt -- --check` — dry-run, no `2>/dev/null` suppression

**No issues.**

---

## Blockers (Must Fix Before Execution)

### BLOCKER 1: Task 1 placeholder dispatch ambiguous

**Location:** Task 1 action, paragraph 3: "For the `Gcv` and `Reml` arms, temporarily route them through the Fixed path..."

**Problem:** The exact Rust code for the placeholder is not provided. Acceptance criterion says "compiles and passes" but does not verify that the dispatch itself is syntactically correct.

**Fix Required:** Add to Task 1 action:
```rust
// EXACT DISPATCH PATTERN FOR TASK 1 (Task 2–3 will replace the placeholders):
let (lambda, gcv, lambda_method) = match &config.lambda {
    LambdaChoice::Fixed(lam) => (*lam, None, LambdaMethod::Fixed),
    LambdaChoice::Gcv => (1.0, None, LambdaMethod::Fixed),      // Placeholder — Task 2 replaces
    LambdaChoice::Reml => (1.0, None, LambdaMethod::Fixed),     // Placeholder — Task 3 replaces
};
```
Add acceptance criterion: "The dispatch compiles without type errors; `Gcv` and `Reml` arms are in place syntactically."

---

### BLOCKER 2: Task 3 EM implementation underspecified

**Location:** Task 3 action, items 5–6 (GLS/Woodbury + EM loop trace computations).

**Problem:** Critical numerical details are prose-level:
- "GLS/Woodbury update" — Woodbury identity not written out; exact linear system not specified
- "tr(Z_range Sigma_b Z_range')" — the computation order (column-wise? row-wise trace?) not explicit
- Trace accumulation in M-step σ²_e — "norm2(r_alpha - Z_range*b_hat)" given, but the second term trace is tersely described

**Impact:** A developer must reverse-engineer the math from RESEARCH §REML Path plus interpret prose. High risk of algebraic error.

**Fix Required:** Add to Task 3 action a concrete Rust pseudocode block:
```rust
// E-STEP PSEUDOCODE:
// ZtZ_range = Z_range' * Z_range  (r×r, symmetric, built via double loop)
// M = ZtZ_range / sigma2_e + (1/sigma2_u)*I_r
// Cholesky factor M, then:
//   Sigma_b = M^{-1}  (computed by solving r copies of M*x = e_i)
//   b_hat = Sigma_b * (Z_range' * r_alpha / sigma2_e)
//   
// M-STEP PSEUDOCODE:
// trace_Sigma_b = sum of diagonal of M^{-1}  (accumulated during Cholesky solves)
// sigma2_u_new = (dot(b_hat, b_hat) + trace_Sigma_b) / r
// r_alpha = yc - Z_null*alpha  (or yc if s == 0)
// ZSbZt = sum_i,j  Sigma_b[i,j] * ZtZ_range[j,i]  (trace of matrix product)
// sigma2_e_new = (sum((r_alpha - Z_range*b_hat)^2) + ZSbZt) / n
//
// GLS PSEUDOCODE (if s > 0):
// Form Σ_inv via Woodbury: Σ^{-1} = (1/σ²_e) * (I - (σ²_u/(σ²_e + σ²_u tr(ZZ')))  * Z_range * Z_range')
// Compute α_new = (Z_null' * Σ^{-1} * Z_null)^{-1} * Z_null' * Σ^{-1} * y_c
// Add 1e-10 * I to diagonal of (Z_null' * Σ^{-1} * Z_null) for ridge stability
```
Update acceptance criterion: "EM loop correctly computes trace terms and Woodbury GLS; test confirms σ²_u > 0, σ²_e > 0 at convergence."

---

### BLOCKER 3: Task 1 acceptance criterion count/reconciliation

**Location:** Task 1 acceptance criteria: "All 12 Phase 66 tests compile and pass..."

**Problem:** RESEARCH §API Evolution lists exactly 8 explicit `PeerConfig{...}` literals plus 1 module-header doctest = 9 items requiring migration. Four `PeerConfig::default()` sites are mentioned as "no code change needed" because the validation error fires before selection. So the total "tests that must pass" is ambiguous: Is it 8 + 4 = 12? Or 8 + 1 (doctest) = 9? Are the `::default()` sites also "tests" that must pass?

**Impact:** Unclear acceptance boundary. A developer might miss some test sites or assume they're not included.

**Fix Required:** Clarify Task 1 acceptance criteria to:
```
- The 8 explicit PeerConfig{...} literals (lines ~402, ~430, ~474, ~494, ~515, ~533, ~563, ~640, ~654, ~672, ~705) 
  are updated to PeerConfig{lambda: LambdaChoice::Fixed(<f64>), ...}.
- The module-header doctest (line ~23) is updated to lambda: LambdaChoice::Fixed(1.0).
- The 4 PeerConfig::default() sites (in validation-error tests ~615, ~739, ~757, plus one regression test) 
  compile but their default-to-Gcv behavior is NOT exercised (validation errors fire first); 
  no code change needed.
- All tests in the #[cfg(test)] mod tests block pass without error.
```
OR:
```
- Regression: all 12 existing tests in #[cfg(test)] mod tests pass.
  (The count 12 includes the 8 explicit PeerConfig{...}, the 4 default() sites, some overlap.)
```

---

## Warnings (Recommended Fixes, Not Blockers)

### WARNING 1: Task 2 and 3 actions lack Rust pseudocode

**Location:** Task 2 action (GCV loop), Task 3 action (REML EM).

**Issue:** Both tasks describe the algorithm in prose without pseudocode. A developer must read RESEARCH carefully to get the exact Rust form.

**Severity:** Warning. The RESEARCH section is detailed and precise; a careful developer can implement correctly. But the risk of off-by-one errors, missing guards, or misinterpreted variable names is elevated.

**Recommendation:** Add pseudocode sketches to Task 2 and 3 actions (2–10 lines each) showing the nested loop structure, guard conditions, and return values. This reduces interpretation risk without bloating the plan.

---

### WARNING 2: Confidence=low on REML assumptions

**Location:** Plan frontmatter `estimate: {confidence: low}`.

**Issue:** The REML EM math (Assumption A1 in RESEARCH) is not cross-verified against a peer-reviewed source. The EM update equations follow "standard smoothing-spline mixed model" (Wahba 1985, Ruppert et al. 2003) but were not directly checked.

**Severity:** Warning. The known-answer tests (Task 3 behavior: "REML and GCV β agree within 0.2"; "recover true β within 0.15") will catch gross errors. The loose tolerance (0.2 for β agreement) hedges this uncertainty.

**Recommendation:** None (risk is bounded by test tolerances). This is acceptable for Phase 67; if REML/GCV diverge more than 0.2 on the SNR fixture, the EM can be debugged during execution.

---

### WARNING 3: No explicit test for Cholesky failures in GCV loop

**Location:** Task 2 action: "solve `beta = cholesky_solve(&a, wty, m)` (on `Err`, `continue` that grid point)"

**Issue:** The GCV action mentions skipping degenerate grid points on Cholesky failure, but the acceptance criteria and verify block do not test this edge case explicitly.

**Severity:** Warning. The regular test fixtures (high-SNR data, well-conditioned design) will not trigger Cholesky failures, so the skip path remains untested. On pathological data (singular design, very low SNR), a Cholesky failure could occur and the `continue` would silently move to the next grid point.

**Recommendation:** Add a brief test case (or a docstring note) that confirms: "If every grid point fails Cholesky, return the smallest grid lambda with GCV = Infinity (or documented fallback)." The current verify block does not detect this.

---

### WARNING 4: VALIDATION.md Wave 0 checklist not ticked

**Location:** `67-VALIDATION.md` Wave 0 Requirements section.

**Issue:** Checklists for "Extend tests with SNR fixture" and "mechanically update Phase 66 tests" are listed as items to do, not items the plan guarantees.

**Severity:** Warning (administrative). The plan does describe these in Task 1 action, but VALIDATION.md is presented as a gate-sign-off document with unchecked boxes.

**Recommendation:** After plan approval, update VALIDATION.md checkboxes:
- [x] Extend `peer.rs` inline tests with SNR fixture (Phase 66 `make_fixture()` reused)
- [x] Mechanically update 8 Phase 66 `PeerConfig{...}` test constructions + module doctest

---

## Non-Issues (Potential Concerns Ruled Out)

### ✅ "What if fnpeel doesn't export `DMatrix`?"
**Answer:** Task 3 reads fpca_variants.rs and cites the precedent (`use nalgebra::DMatrix;` already in scope). nalgebra is in Cargo.toml. No blocker.

### ✅ "What if compute_peer_trace_hat doesn't work for GCV inside the loop?"
**Answer:** RESEARCH verified the function already exists (Phase 66, lines 324–340) and the guard `clamps to n as f64` is in place. Task 2 adds a secondary `denom <= 0.0` guard. No blocker.

### ✅ "Will the REML α-update singular matrix in low-rank s kill the solve?"
**Answer:** Task 3 action mentions adding 1e-10 ridge to diagonal for regularization. Also, typical penalty families (Difference{2}, Ridge) have well-behaved null spaces (s ≤ 2 for roughness). Mitigated. Acceptable risk for a tracer.

### ✅ "Does Phase 68 depend on Phase 67 completing exactly as specified?"
**Answer:** Yes. But Phase 68 is out-of-scope for this verification. The plan itself does not reference Phase 68 work, so no cross-phase blocker.

---

## Summary Table

| Dimension | Result | Notes |
|-----------|--------|-------|
| Requirement Coverage | ✅ PASS | All four ROADMAP criteria mapped to tests |
| Task Completeness | ⚠️ NEEDS REVISION | Blockers 1–3 must be fixed (see §Blockers) |
| Dependency Correctness | ✅ PASS | Depends on Phase 66 (complete); no cycles |
| Key Links | ✅ PASS | Dispatch→solve, GCV→trace-hat, REML→eigen all wired |
| Scope Sanity | ✅ PASS (with caveats) | 4 tasks, 1 file, low-confidence estimate; acceptable for tracer |
| Verification Derivation | ✅ PASS | must_haves are user-observable and testable |
| Context Compliance | ✅ PASS | Locked decisions honored; deferred ideas excluded |
| Scope Reduction | ✅ PASS | No silent simplification detected |
| Architectural Tiers | ✅ PASS | All work in peer.rs (correct tier) |
| Nyquist Compliance | ⚠️ PENDING | Structure sound; execution will verify |
| Cross-Plan Contracts | ✅ PASS | Single plan; N/A |
| CLAUDE.md Compliance | ✅ PASS | Respects all project conventions |
| Research Resolution | ✅ PASS | No unresolved open questions |
| Patterns | ⚠️ SKIPPED | No PATTERNS.md for Phase 67 |
| Verify Format | ✅ PASS | Commands use reliable Rust test framework |

---

## Verdict

**NEEDS REVISION**

The plan is **structurally sound** and addresses all four ROADMAP success criteria with concrete task decomposition and thorough acceptance criteria. However, **three blockers** prevent execution:

1. **Task 1 placeholder dispatch is ambiguous** — needs explicit pseudocode for the match-on-LambdaChoice dispatch.
2. **Task 3 EM implementation is underspecified** — needs Rust pseudocode for the GLS Woodbury update and trace computations (the hardest part).
3. **Task 1 acceptance criteria are imprecise** — reconciliation of the test count (8 vs 12 vs 9) and behavior of default() sites needs clarification.

**Recommended action:** Planner should revise the plan to:
- Add pseudocode blocks to Tasks 1, 2, 3 (especially Task 1 dispatch and Task 3 EM)
- Clarify Task 1 acceptance criteria with exact test locations and behavior of `PeerConfig::default()` post-migration
- (Optional) Add edge-case tests for GCV Cholesky failures (Warning 3)

Once these revisions are incorporated, the plan will be **execution-ready**. The underlying logic is sound; it needs surface clarification only.

---

## Detailed Revision Requests

### For Planner: Specific Text to Add/Revise

**Task 1 Action — After "...so the code compiles and the tracer proves the API evolution end-to-end;"**

Add:
```
The dispatch must be a Rust match expression on &config.lambda with three arms:
  LambdaChoice::Fixed(lam) → (lambda, gcv, lambda_method) = (*lam, None, LambdaMethod::Fixed)
  LambdaChoice::Gcv → (1.0, None, LambdaMethod::Fixed)  [PLACEHOLDER — Task 2 replaces with GCV selector]
  LambdaChoice::Reml → (1.0, None, LambdaMethod::Fixed) [PLACEHOLDER — Task 3 replaces with REML selector]

The placeholder 1.0 lambda must feed into the existing solve (A = WtW + lambda*Q; cholesky_solve)
so the tracer proves the data flow end-to-end before selectors are added.
```

**Task 1 Acceptance Criteria — Replace the "12 Phase 66 tests" line with:**

```
- Mechanical migration: The 8 PeerConfig{...} literals listed in RESEARCH.md 
  §API Evolution §Tests that must be mechanically updated (lines ~402, ~430, ~474, ~494, ~515, ~533, ~563, ~640, ~654, ~672, ~705) 
  have been changed from `lambda: <f64>` to `lambda: LambdaChoice::Fixed(<f64>)`.
- Module doctest: The module-header doctest (peer.rs:16–25) has been updated to 
  `lambda: LambdaChoice::Fixed(1.0)`.
- PeerConfig::default() sites: The 4 tests using PeerConfig::default() 
  (test_peer_argvals_mismatch, test_peer_rejects_single_observation, 
   test_peer_rejects_non_monotonic_argvals, test_peer_rejects_non_finite_y) 
  compile and still pass (their behavior is unchanged: validation error fires before selection).
- All tests in #[cfg(test)] mod tests pass: `cargo test -p fdars-core --features linalg,parallel peer::` 
  returns green with no failures or panics.
```

**Task 3 Action — After item 5 (EM loop cap 100 iterations), add a section:**

```
### GLS Alpha Update (when s > 0)

Solve for α ∈ ℝ^s using generalized least squares:
  Σ = σ²_e I_n + σ²_u Z_range Z_range'  (n×n covariance of observations)
  α_new = (Z_null' Σ^{-1} Z_null)^{-1} Z_null' Σ^{-1} y_c

Use the Woodbury identity to avoid forming n×n Σ:
  Σ^{-1} = (1/σ²_e) * (I_n - (σ²_u / (σ²_e + σ²_u ||Z_range row||²)) * Z_range Z_range')

Compute locally:
  (a) Q_mat = Z_range' Z_range  (r×r)
  (b) C_factor = σ²_e + σ²_u * tr(Q_mat)  [denominator]
  (c) Z_null' Z_range (s×r)
  (d) Solve the s×s system:  (Z_null' Z_null - (σ²_u/C_factor) * Z_null' Z_range Z_range' Z_null) α = Z_null' y_c
      Add 1e-10 * I_s to diagonal for ridge stability before Cholesky solve.

### Trace Computation (M-step σ²_e)

tr(Z_range Σ_b Z_range') = tr(Σ_b (Z_range' Z_range)) = sum_{i,j}  Σ_b[i,j] * ZtZ_range[j,i]

Compute after inverting M in the E-step: accumulate diagonal of M^{-1} as trace_Sigma_b 
(one Cholesky solve per column); then form ZtZ_range (r×r double loop); 
then sum the pointwise products.
```

---

## Conclusion

The Phase 67 plan is **well-designed and comprehensive** but requires **surface-level revisions** to clarify implementation details, especially the Task 1 dispatch and Task 3 EM pseudocode. Once revised, it is **execution-ready** and will deliver all four ROADMAP success criteria.

**Estimated revision time:** 30–60 minutes (adding pseudocode and clarifying acceptance criteria).

