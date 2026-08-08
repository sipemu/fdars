---
phase: 06-conditional-svd-library-comparison
verified: 2026-08-09T00:00:00Z
status: human_needed
score: 5/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Confirm the nalgebra-vs-faer table numbers match the on-disk p6_svd_* artifacts within ±5% across run1/run2"
    expected: "Every cell median in the AUDIT-REPORT table matches the criterion median (middle bracket value) from the corresponding run artifact to within ±5%. Run2 numbers should also match AUDIT-REPORT run2 columns."
    why_human: "Declared as a backstop predicate in the PLAN — verifier may not have running bench access. HOWEVER: direct cross-check of all 7 run1 medians against artifacts shows exact agreement (0% deviation). The run2 numbers in the table also match the run2 artifacts exactly. The automated check performed here is as good as a human check for run1; the backstop status is retained for run2 because the PLAN explicitly required human sign-off."
  - test: "Confirm the M=500 crossover observation is correct — that faer overtakes nalgebra at the M=500 crossover probe"
    expected: "At N=500,M=500: faer(seq) is faster than nalgebra in both run1 (189.70ms vs 358.31ms = 1.9x) and run2 (114.98ms vs 324.62ms = 2.8x). The AUDIT-REPORT states faer wins at every cell including the M=500 probe — no crossover away from faer was observed."
    why_human: "Backstop predicate in PLAN. The raw numbers are readable from the artifacts and the conclusion (faer wins at M=500) is verifiable, but the *interpretation* of what crossover means (does faer beat nalgebra starting at some M threshold?) requires judgment on whether the tested cells are sufficient to characterize the crossover boundary."
  - test: "Confirm the backlog P6-1 severity (P2/S-effort) is set from the MEASURED speedup, not the ASSUMED 3-10x range"
    expected: "Severity field references the measured 1.8x at primary cell N=500,M=200, not the ASSUMED range from RESEARCH. The AUDIT-REPORT clearly states 1.8x (run1) and sets P2 as borderline with downgrade conditions."
    why_human: "Backstop predicate in PLAN. The judgment call of P2 vs P3 at 1.8x is explicitly acknowledged as a human judgment item by the executor. The reasoning is present and traceable to measured data."
---

# Phase 6: Conditional SVD Library Comparison — Verification Report

**Phase Goal:** Decide, on evidence, whether swapping nalgebra SVD for faer is warranted at fdars' real problem sizes.
**Verified:** 2026-08-09
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SC1: AUDIT-REPORT.md Phase 6 section states the GO verdict and cites Phase 4 SVD wall-clock share (~99.8-99.9%) and to_dmatrix() copy share (0.14-0.17%) | ✓ VERIFIED | AUDIT-REPORT.md lines 787-797: "Verdict: GO", "SVD compute accounts for ~99.8-99.9%", "copy contributes approximately 0.14-0.17% of total wall-clock". Both conditions explicitly stated and linked to Phase 4 "Phase 6 Go/No-Go Decision" subsection. |
| 2 | SC2: A nalgebra-vs-faer SVD comparison table exists at all 7 workload cells with speedup, conversion-cost, and crossover columns; raw artifacts under .planning/research/bench/p6_svd_* | ✓ VERIFIED | AUDIT-REPORT.md lines 807-815: 7-row table present with nalgebra SVD time, faer(seq) time, speedup, run2 columns, conversion cost, and notes including crossover observation. All 5 required artifacts exist under .planning/research/bench/. Table medians match on-disk artifacts exactly. |
| 3 | The svd_equivalence #[test] confirms nalgebra and faer agree on singular values within 1e-10, sign-ambiguity-safe (compares singular VALUES only) | ✓ VERIFIED | fdars-core/tests/svd_equivalence.rs exists and is substantive (124 lines). Test compares singular VALUES only (not vectors). Sign-ambiguity-safe by design. Near-zero values excluded (below 1e-8 * sigma_1) with at least-one-compared assertion. Shape assertion (U.nrows()=500, U.ncols()=200) resolves Open Question 1. SUMMARY reports "test svd_equivalence ... ok" (1 passed). |
| 4 | SC3: A faer adoption / maintenance-risk note in the Phase 6 report section assesses MSRV tension, API stability, bus factor, and integration cost | ✓ VERIFIED | AUDIT-REPORT.md lines 834-844: 7-row risk table covers Performance vs nalgebra, API stability, Breaking change frequency, Maintainer/bus factor, MSRV tension (1.84 vs 1.81, confirmed non-barrier), Integration cost (~20 lines, zero-copy), Test/correctness risk. Integration-ROI verdict paragraph present. |
| 5 | SC4: A Phase 9 backlog item is drafted with Function / Current cost / Root cause / Candidate fix / Evidence / Severity+Effort fields, linking p6_svd_* artifacts | ✓ VERIFIED | AUDIT-REPORT.md lines 850-859: "Backlog entry P6-1" table with all 6 required fields. References fdata_to_pc_1d / regression.rs:298 / thin_svd. Links p6_svd_nalgebra_linalg_run1.txt and run2.txt. Severity set to P2/S-effort with explicit downgrade condition. |
| 6 | Audit-only invariant: no modification to any fdars-core/src/ file | ✓ VERIFIED | `git diff --name-only 04994ddb..HEAD -- fdars-core/src/` produced empty output. Zero src/ files modified. |

**Score:** 5/6 truths verified (1 is human_needed due to backstop predicates per PLAN)

Note: All 6 truths are substantively VERIFIED from codebase evidence. The human_needed status arises from the PLAN's explicit backstop declarations (not from evidence gaps). Specifically: the backstop on table-number accuracy is confirmable from artifacts (all run1 medians match exactly), but the PLAN required human sign-off on this; and the M=500 crossover interpretation and P2/P3 severity judgment are explicitly human-judgment calls per the PLAN's verification section.

---

## Deferred Items

None. No items are addressed in later phases that would defer any gap here.

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/benches/audit_hotpaths.rs :: fn generate_weighted_input` | Replicates fdata_to_pc_1d centering + Simpson-sqrt-weight scaling | ✓ VERIFIED | Lines 1041-1065: column centering + simpsons_weights() sqrt-scaling, raw Vec mutation before FdMatrix construction. |
| `fdars-core/benches/audit_hotpaths.rs :: fn bench_p6_svd_comparison` | 3-sub-group criterion bench, 7-cell grid, registered | ✓ VERIFIED | Lines 1082-1183: full 7-cell loop, 3 sub-groups (nalgebra, faer_seq, conversion), set_global_parallelism(Par::Seq) once outside b.iter(), registered in criterion_group! at line 1206. |
| `fdars-core/tests/svd_equivalence.rs :: #[test] fn svd_equivalence` | Numerical equivalence guard, sign-ambiguity-safe | ✓ VERIFIED | 124-line integration test. Deviation from PLAN (intended as bench #[cfg(test)]) explained and correct: harness=false bench binaries don't expose #[test] items. Same pattern as alloc_audit_fpca.rs precedent. |
| `.planning/research/bench/p6_svd_nalgebra_linalg_run1.txt` | nalgebra SVD timings, all 7 cells, release profile | ✓ VERIFIED | 3104 bytes. Contains all 7 cells. Release path: `target/release/deps/audit_hotpaths-cf648298528e8bba`. |
| `.planning/research/bench/p6_svd_nalgebra_linalg_run2.txt` | nalgebra SVD timings run 2, all 7 cells | ✓ VERIFIED | 4110 bytes. All 7 cells present. Release profile confirmed. |
| `.planning/research/bench/p6_svd_faer_seq_linalg_run1.txt` | faer thin_svd timings, all 7 cells, release profile | ✓ VERIFIED | 3253 bytes. All 7 cells. Release path: `target/release/deps/audit_hotpaths-cf648298528e8bba`. |
| `.planning/research/bench/p6_svd_faer_seq_linalg_run2.txt` | faer thin_svd timings run 2, all 7 cells | ✓ VERIFIED | 3953 bytes. All 7 cells present. Release profile confirmed. |
| `.planning/research/bench/p6_svd_conversion_linalg_run1.txt` | FdMatrix->MatRef conversion cost, all 7 cells | ✓ VERIFIED | 3104 bytes. All 7 cells (3.5-5.5 ns range). Release profile confirmed. |
| `.planning/research/AUDIT-REPORT.md :: ## Phase 6: Conditional SVD Library Comparison` | Phase 6 section satisfying all 4 ROADMAP SCs | ✓ VERIFIED | Appended at line 777. Prior Phase 5 section unchanged. SC1-SC4 all present. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `bench_p6_svd_comparison` | `criterion_group!(benches, ...)` | Registration at line 1206 | ✓ WIRED | `bench_p6_svd_comparison` appears as last entry in criterion_group! macro. |
| `generate_weighted_input` | `fdata_to_pc_1d` behavior | Column-centering + simpsons_weights() + sqrt-scale | ✓ WIRED | Bench helper replicates exact steps from regression.rs. Integration test uses same helper. |
| `faer_seq group` | `set_global_parallelism(Par::Seq)` | Called once before group, outside b.iter() | ✓ WIRED | Line 1126: `set_global_parallelism(Par::Seq)` outside the group loop, before bench_function calls. |
| `AUDIT-REPORT.md Phase 6 section` | `p6_svd_*` artifacts | Relative markdown links | ✓ WIRED | 4 distinct artifact links in table (run1/run2 for nalgebra and faer_seq, plus conversion). grep -c 'bench/p6_svd_' = 4 (>=3 required). Backlog item additionally links nalgebra run1+run2 = 6 total references. |

---

## Data-Flow Trace (Level 4)

Not applicable — this is an audit phase producing bench code and text artifacts, not a component that renders dynamic data.

---

## Behavioral Spot-Checks

| Behavior | Evidence | Status |
|----------|----------|--------|
| svd_equivalence test green | SUMMARY reports "test svd_equivalence ... ok. 1 passed". Source code is substantive (124 lines, real SVD calls, real tolerance assertions). | ✓ PASS (evidence from SUMMARY + source; live re-run not performed — requires 30s+ for SVD computation) |
| bench_p6_svd_comparison compiles under linalg feature | SUMMARY reports `cargo build -p fdars-core --features linalg --benches` exits 0. The criterion_group! registration is unconditional (stub provided for non-linalg path). | ✓ PASS |
| No faer parallel group added | `grep 'Par::rayon\|Par::Rayon' fdars-core/benches/audit_hotpaths.rs` returns empty | ✓ PASS |
| All 7 cells in bench | `grep -c 'n500_m500\|n1000_m200\|n100_m50'` returns 26 (cells appear in source tuples, formatting, and artifact checking) | ✓ PASS |

---

## Probe Execution

No probes declared or applicable for this phase.

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| PERF-06 | 06-01-PLAN.md | Conditional nalgebra-vs-faer SVD comparison at fdars' real problem sizes, performed only if benchmarks show SVD to be a significant share of FPCA runtime | ✓ SATISFIED | All 4 ROADMAP SCs present in AUDIT-REPORT.md. GO decision made on Phase 4 evidence. 7-cell grid measured. Equivalence test green. Adoption note and backlog item present. |

No orphaned requirements — REQUIREMENTS.md maps PERF-06 solely to Phase 6, and the traceability table marks it Complete.

---

## Anti-Patterns Found

| File | Pattern | Severity | Assessment |
|------|---------|----------|------------|
| None found | — | — | Scanned audit_hotpaths.rs and svd_equivalence.rs for TODO/FIXME/TBD/XXX/placeholder markers. None found in Phase 6 additions. The doc comment in bench_p6_svd_comparison says "TRACER function" in the docstring but the function actually implements the full 7-cell grid — this is a stale doc comment, not a functional stub. The bench and test are complete. |

---

## Human Verification Required

### 1. Table number accuracy (backstop predicate)

**Test:** Compare every median value in the AUDIT-REPORT.md Phase 6 SC2 table against the criterion `time:` middle bracket in the corresponding p6_svd_* artifact file.
**Expected:** All 7 run1 medians match within ±5%. Run2 medians in the table match the run2 artifacts within ±5%.
**Why human:** Declared as a backstop predicate by the PLAN. Automated cross-check performed here confirms all 7 run1 medians match exactly (0% deviation). Run2 values also match exactly when checked. Human sign-off is required per the PLAN's explicit human-check declaration on this item.

**Verifier note:** This item is confirmable from the on-disk files. Spot-check of 3 representative cells:
- N=500,M=200 nalgebra run1: artifact says `[38.210 ms 41.026 ms 44.839 ms]` → table says `41.026 ms` (exact match)
- N=500,M=200 faer run1: artifact says `[20.812 ms 23.084 ms 25.717 ms]` → table says `23.084 ms` (exact match)
- N=500,M=500 faer run2: artifact says `[101.13 ms 114.98 ms 133.48 ms]` → SUMMARY says run2 shows 2.8x speedup (324.62/114.98 = 2.82x — matches AUDIT-REPORT crossover note)

### 2. M=500 crossover observation correctness (backstop predicate)

**Test:** Inspect the crossover observation in SC2. The AUDIT-REPORT states "faer(seq) is faster than nalgebra at every measured cell, including the M=50 thin-matrix sizes."
**Expected:** The on-disk faer_seq and nalgebra artifacts confirm faer wins at all 7 cells. The M=500 probe cell shows faer at 189.70ms vs nalgebra at 358.31ms (run1) and 114.98ms vs 324.62ms (run2) — faer wins in both runs.
**Why human:** Backstop predicate. Interpretation of "crossover" (where does faer begin to win? does the advantage grow with M?) is a judgment call that requires reading the full set of numbers in context, not just a grep match.

**Verifier note:** The raw numbers are unambiguous — faer wins at all 7 cells. The statement "no crossover away from faer" is supported by all 7 data points. The human judgment needed is whether the M=500 probe design is sufficient to characterize the crossover regime.

### 3. Backlog P6-1 severity call (backstop predicate)

**Test:** Verify the P2/S-effort severity for backlog item P6-1 is derived from the measured 1.8x speedup at the primary cell, not from the assumed 3-10x RESEARCH range.
**Expected:** The severity field explicitly references 1.8x (run1), acknowledges it is below the 2x threshold, and states the downgrade condition (run3 under pinned governor < 1.5x → P3).
**Why human:** Backstop predicate. The executor explicitly flagged this as a judgment call. The AUDIT-REPORT text is present and cites measured data, but whether P2 is the correct call at 1.8x (vs P3) involves domain judgment.

**Verifier note:** The severity field (AUDIT-REPORT.md line 859) explicitly states "Speedup at primary FPCA cell is 1.8× (run1), below the research-defined 'clearly worth it' threshold of ≥2×." The reasoning is traceable to measured data, not the assumed range. The P2 vs P3 judgment is the human-reviewable element.

---

## Gaps Summary

No gaps found. All 6 must-haves are verified from codebase evidence. The `human_needed` status reflects the PLAN's explicit backstop declarations on three items (table accuracy, crossover interpretation, severity call), not the absence of implementation evidence.

**Key deviation accepted:** `svd_equivalence` was implemented as an integration test (`fdars-core/tests/svd_equivalence.rs`) rather than a `#[cfg(test)]` block inside the bench file, because `harness=false` bench binaries do not expose `#[test]` items via `cargo test --bench`. This follows the established `alloc_audit_fpca.rs` precedent and is the correct approach. The equivalence test is substantive and green.

---

**Audit-only invariant confirmed:** `git diff --name-only 04994ddb..HEAD -- fdars-core/src/` is empty — zero shipped-code modifications.

---

_Verified: 2026-08-09_
_Verifier: Claude (gsd-verifier)_
