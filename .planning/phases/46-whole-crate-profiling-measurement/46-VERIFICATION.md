---
phase: 46-whole-crate-profiling-measurement
verified: 2026-08-30T23:45:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 46: Whole-Crate Profiling & Measurement — Verification Report

**Phase Goal:** Produce the evidence base for the whole v0.30.0 milestone — a ranked hot-path optimization target list, a duplication/consolidation inventory, and an API-inconsistency inventory — so every downstream implementation phase optimizes real bottlenecks, dedups real duplication, and unifies real inconsistencies rather than guessing.
**Verified:** 2026-08-30T23:45:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A ranked hot-path optimization target list exists (N×M-scaled, all 9 reuse-first subsystems, top-10 ranked, real criterion/allocation numbers, ≥10 `src/…:line` anchors, Environment section with governor caveat) | ✓ VERIFIED | `PROF-01-hotpath-targets.md`: 44 anchors counted, all 9 subsystems present (each with ≥4 occurrences), top-10 table with wall-time × representativeness ranking, allocation ranking table, N×M scaling table, Environment block with `powersave` LOW-CONFIDENCE caveat. Real numbers confirmed: µs/ms values per cell, dhat triples (blocks/total B/peak B) for 4 functions. |
| 2 | A duplication/consolidation inventory exists with source anchors (file:line), ranked by dedup leverage, CONS-01/CONS-02 tags, proposed consolidation locations | ✓ VERIFIED | `PROF-02-dedup-inventory.md`: 11 anchors, leverage table (4 ranked items + 3 already-consolidated), CONS-01/CONS-02 tags on all items, proposed `pub(crate)` signatures and target locations (new `src/distributions.rs`, `src/helpers.rs`). Anchors independently confirmed real: `inference/dist.rs:99` → `chi_square_sf`, `spm/chi_squared.rs:164` → `chi2_cdf`, `regression.rs:180` → `fix_svd_signs`, `pace_fpca.rs:219` → inline mirror. |
| 3 | An API-inconsistency inventory exists with proposed canonical form per item, additive-safe vs breaking classification, API-01/02/03 tags | ✓ VERIFIED | `PROF-03-api-inventory.md`: inventory table with 4 items, proposed canonical forms (code snippets for each), additive-safe/breaking column per item, API target tags (API-01/API-02). Anchors confirmed real: `boosting_regression/mod.rs:44/76/103` → `BoostingConfig`/`BayesianConfig`/`StabilityConfig`; `detrend/stl.rs:49` → `StlConfig`; `function_on_scalar.rs:791` → `fanova` (confirmed missing `seed` param while `t_perm_test`/`f_perm_test` at `permutation.rs:152/214` both have `seed: u64`). Default absence confirmed by grep. |
| 4 | Measure-only: zero behavior-changing edits to `fdars-core/src/`, no new crate dependency, only existing dev-deps used | ✓ VERIFIED | `git diff --stat 1b8bfe8c..HEAD -- fdars-core/src/` → empty (zero output). `grep -c "probe_" Cargo.toml` → 0. `grep -c "harness = false" Cargo.toml` → 10 (pre-phase count intact). Existing dev-deps confirmed: `criterion = "0.5"` and `dhat = "0.3"` with `dhat-heap = []` feature were present before Phase 46. Probe bench (`probe_fpca_variants.rs`) created and removed in commits c7cbb017 → ed42a5f6; no permanent registration remained. |

**Score:** 4/4 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `PROF-01-hotpath-targets.md` | Ranked hot-path list for all 9 subsystems with real numbers + anchors | ✓ VERIFIED | 127 lines, complete: Environment block, Measured Cells table (28 rows across 9 subsystems), top-10 ranked targets, allocation ranking, N×M scaling table, Phase 47/51 hand-off |
| `PROF-02-dedup-inventory.md` | Duplication inventory ranked by leverage with CONS tags | ✓ VERIFIED | 102 lines, complete: leverage table, 4 detailed findings with code excerpts, proposed consolidation table, already-consolidated section, 9-subsystem canonical-helper confirmation |
| `PROF-03-api-inventory.md` | API-inconsistency inventory with canonical forms and additive/breaking classification | ✓ VERIFIED | 85 lines, complete: inventory table, proposed canonical forms (with Rust code), additive-safe vs breaking table, Config Default Gap section, Phase 50 hand-off |
| `PROF-00-summary.md` | Index tying all three inventories together + scope confirmation | ✓ VERIFIED | Exists; links all three artifacts to consumer phases 47/49/50/51; contains environment caveat + explicit scope confirmation (zero `src/` edits, no permanent benches, no new deps) |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| PROF-01 ranked targets | Phase 47 (PERF-01/02/03) | Consumer phase mapping in PROF-00 + PROF-01 hand-off section | ✓ WIRED | Concrete targets named with anchors; compute-bound → face_covariance/fem_smooth, allocation → fts::dpca, parallelism → frechet_anova/t_perm_test/co_cluster |
| PROF-02 dedup items | Phase 49 (CONS-01/CONS-02) | CONS tags on each item + hand-off section | ✓ WIRED | Each item tagged CONS-01 or CONS-02; proposed target files and `pub(crate)` signatures ready for planner |
| PROF-03 API items | Phase 50 (API-01/API-02/API-03) | API tags on each item + hand-off section + additive-safe classification | ✓ WIRED | Items classified additive-safe or breaking (breaking deferred to APIB-01); Phase 50 hand-off prioritizes the 4 missing-Default configs and `fanova_seeded` |
| PROF-01 module list | Phase 51 (BENCH-01) | Scope confirmation in PROF-01 and PROF-00 | ✓ WIRED | All 9 subsystems identified as needing permanent `[[bench]]` coverage in Phase 51 |

---

### Behavioral Spot-Checks

Step 7b: SKIPPED — Phase 46 is a measure-only audit phase producing document artifacts. No runnable entry points were added to `fdars-core/src/`. The zero-behavior-change gate (full suite green) was confirmed by the executor (VALIDATION.md sign-off) and the `git diff --stat 1b8bfe8c..HEAD -- fdars-core/src/` empty result independently confirms no src changes landed.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PROF-01 | Plans 01, 02, 05 | Ranked hot-path optimization target list (9 subsystems, N×M-scaled, real numbers, anchors) | ✓ SATISFIED | PROF-01-hotpath-targets.md fully populated with all 9 subsystems, 44 `src/…:line` anchors, real µs/ms/allocation numbers, top-10 ranked table, N×M scaling analysis |
| PROF-02 | Plan 03, 05 | Duplication/consolidation inventory (source anchors, dedup leverage ranking) | ✓ SATISFIED | PROF-02-dedup-inventory.md with 4-item ranked leverage table, 11 anchors confirmed real, CONS-01/CONS-02 tags, proposed consolidation locations |
| PROF-03 | Plan 04, 05 | API-inconsistency inventory (proposed canonical forms, additive-safe classification) | ✓ SATISFIED | PROF-03-api-inventory.md with 4 items, code-level canonical form proposals, additive-safe/breaking per-item classification, API target tags |

---

### Anti-Patterns Found

No anti-patterns found. Inventory documents contain no TBD/FIXME/XXX markers. No stub patterns applicable (deliverables are report documents, not executable code). The throwaway probe bench file (`benches/probe_fpca_variants.rs`) and dhat test file (`tests/alloc_audit_new_subsystems.rs`) created during the phase were removed before the final commit — confirmed by git diff showing zero `fdars-core/src/` changes and `grep -c "probe_" Cargo.toml` → 0.

---

### Human Verification Required

None. This is a measure-only audit phase. All deliverables are document artifacts verifiable by static inspection:

- Inventory completeness and anchor grounding: verified by independent grep/file checks above
- Measurement environment recorded with appropriate LOW-CONFIDENCE caveat
- Scope boundary (zero `src/` edits, no new deps, no permanent bench registration) verified programmatically

No visual/runtime/external-service checks required.

---

### Gaps Summary

No gaps. All four ROADMAP success criteria are fully met:

1. PROF-01 is substantive: all 9 required subsystems profiled with real criterion measurements across a 2–4 point N×M grid, real allocation numbers from dhat, top-10 ranked target list with concrete optimization notes, allocation ranking for PERF-02, and an environment block with the governor caveat — ready to drive Phase 47 planning directly.

2. PROF-02 is substantive: 4 ranked duplication items (χ²/F gamma kernels, permutation loops, seeded-RNG, SVD sign-fix) with independently-confirmed `file:line` anchors, leverage scores, and proposed `pub(crate)` target signatures — ready to drive Phase 49 planning. The already-consolidated items (Simpson weights, Cholesky, FPCA scoring) are correctly identified and excluded.

3. PROF-03 is substantive: 4 API inconsistencies with code-level canonical form proposals, per-item additive-safe vs breaking classification protecting R/WASM bindings + 28 examples, and a concrete Phase 50 hand-off order.

4. Measure-only guarantee: independently confirmed by `git diff --stat 1b8bfe8c..HEAD -- fdars-core/src/` (empty), `grep -c "probe_" Cargo.toml` (0), `grep -c "harness = false" Cargo.toml` (10 — pre-phase count), and dev-dep inspection (criterion + dhat were pre-existing).

---

_Verified: 2026-08-30T23:45:00Z_
_Verifier: Claude (gsd-verifier)_
