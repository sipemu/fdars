---
phase: 05-parallelism-gap-assessment
verified: 2026-08-08T19:09:04Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: # No previous VERIFICATION.md — initial verification
---

# Phase 5: Parallelism Gap Assessment Verification Report

**Phase Goal:** Measure rayon thread scaling and flag safe-to-parallelize sequential loops (audit-only — deliverable is evidence in AUDIT-REPORT.md + bench artifacts + backlog, ZERO edits to fdars-core/src/).
**Verified:** 2026-08-08T19:09:04Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Success Criteria)

| # | Truth (SC) | Status | Evidence |
|---|-----------|--------|----------|
| SC1a | rayon thread-scaling table (karcher heavy + streaming light sentinels, RAYON_NUM_THREADS ∈ {1,2,4,8}), median+spread+speedup, confidence flags | ✓ VERIFIED | AUDIT-REPORT.md:632-649 — 8-row table (karcher N=100/M=50 + streaming N=500/M=200 × 4 thread counts). Numbers cross-checked against raw artifacts: karcher T=1 central `1.5566 s` ≈ table 1553.8ms; T=8 `328.31ms` ≈ 328.3ms. LOW-CONFIDENCE flags present (governor unpinned; karcher T=4 spread 11.4%; streaming multi-thread 21–45%). |
| SC1b | payback-threshold N per target | ✓ VERIFIED | AUDIT-REPORT.md:651-682 — karcher payback N≤10 (below-grid), streaming payback N≈50. D-02 method note states baseline is `RAYON_NUM_THREADS=1` single-thread rayon, NOT `--no-default-features`. Raw numbers verified: streaming N=1 single `14.875µs`/default `20.590µs` = 0.72× loss (matches report); karcher N=10 single `243.50ms`/default `56.926ms` (matches). |
| SC2 | safe-to-parallelize sequential loops enumerated with evidence | ✓ VERIFIED | AUDIT-REPORT.md:686-702 — all 5 D-07 candidates (cv.rs:76, elastic_fpca.rs:701/720/764, regression.rs:167) each with (a) independence argument, (b) RNG-seeding note, (c) applicable macro, (d) projected speedup from SC1. All source line anchors verified accurate against src (cv.rs:76 = `for fold in 0..nfold`; three elastic_fpca lines = `for i in 0..n`; regression.rs:167 = `center_columns`; karcher.rs:185 = `iter_maybe_parallel!`). Static-argument-only, no src edit stated. |
| SC3 | cost of default unaccelerated path recorded | ✓ VERIFIED | AUDIT-REPORT.md:704-732 — (a) rayon-off ~10× (Phase-1 karcher `""`≈1555ms vs `parallel`≈162ms, cited artifacts) and (b) banding opt-in ~4–6× (Phase-3 karcher unbanded-vs-banded) with LOW-CONFIDENCE caveat on Phase-3 variance. Both framed as default-path costs. |
| SC4 | GSD-ready backlog entries drafted | ✓ VERIFIED | AUDIT-REPORT.md:734-773 — 4 backlog entries P5-1..P5-4 with Function / Current-cost / Root-cause / Candidate-direction / Projected-reduction / Evidence-link fields (Phase-3 D-07 format). |

**Score:** 4/4 success criteria verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/research/AUDIT-REPORT.md` §Phase 5 | Full SC1–SC4 section | ✓ VERIFIED | Section at line 612; methodology + thread table + payback-N + SC2/SC3/SC4 subsections all present and substantive. |
| `fdars-core/benches/audit_hotpaths.rs` | p5 bench cells + registration | ✓ VERIFIED | `bench_p5_karcher_threads` (grep 2 = def+reg), streaming + paybackN cells present (grep 4 + 10). |
| `p5_karcher_linalg,parallel_run{1,2,3}.txt` | karcher thread sweep, 4 thread blocks each | ✓ VERIFIED | Exist; each has all 4 `=== RAYON_NUM_THREADS=` blocks; release binary path confirmed. |
| `p5_streaming_linalg,parallel_run{1,2,3}.txt` | streaming thread sweep | ✓ VERIFIED | Exist; 4 thread blocks each. |
| `p5_karcher_paybackN_...run1.txt` | N∈{10,25,50,100} grid | ✓ VERIFIED | Grid present (n10/n25/n50/n100_m50). |
| `p5_streaming_paybackN_...run1.txt` | N_obj grid, single vs default | ✓ VERIFIED | Grid n1/n10/n50/n200/n500_m200 × {RAYON_NUM_THREADS=1, RAYON default}. |
| `p5_env_info.txt` | rustc/cargo + D-04 controls | ✓ VERIFIED | rustc/cargo 1.97.0; `=== D-04 CONTROLS ===` block records taskset -c 0-7 applied, governor performance FAILED (no root) → powersave, LOW-CONFIDENCE noted, 3-run protocol as backstop. |

### Key Link Verification

| From | To | Status | Details |
|------|----|--------|---------|
| RAYON_NUM_THREADS env → rayon pool → iter_maybe_parallel! (karcher.rs:185) → measured scaling | thread table rows | ✓ WIRED | karcher.rs:185 confirmed `iter_maybe_parallel!(0..n)`; artifacts show scaling 1.99×/3.84×/4.73×. |
| bench artifact median → thread-scaling table row → report | AUDIT-REPORT.md | ✓ WIRED | Spot-checked medians match artifacts exactly. |
| RAYON_NUM_THREADS=1 baseline vs default → payback N | payback tables | ✓ WIRED | Streaming/karcher payback numbers match raw artifacts. |
| SC1 measured scaling → SC2 projected speedups | SC2 table | ✓ WIRED | SC2 projections explicitly extrapolate from SC1 ~4.7× ceiling, bounded by payback-N. |

### Prohibition Verification (audit-only scope fence)

| Prohibition | Status | Evidence |
|-------------|--------|----------|
| No edits to `fdars-core/src/**` | ✓ VERIFIED (not violated) | Full phase-05 commit range (a8693876^..HEAD) touched only `fdars-core/benches/audit_hotpaths.rs` + `.planning/` files. `git log --name-only` grep for `fdars-core/src/` returned empty. Working tree src clean. |

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
|-------------|-------------|--------|----------|
| PERF-05 | 05-01/02/03 | ✓ SATISFIED | REQUIREMENTS.md:18 marked `[x]`, traceability row 65 `PERF-05 | Phase 5 | Complete`. All 4 SCs delivered in AUDIT-REPORT.md §Phase 5. |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| benches/audit_hotpaths.rs | `bench_p5_*paybackN` non_snake_case (2 fns) | ℹ️ Info | Per 05-REVIEW.md WR-01; cosmetic compiler warnings, does not affect audit deliverable or goal. Not a blocker (bench code, not src; not a debt marker). |
| benches/audit_hotpaths.rs | 5th karcher arg comment mislabeled band_frac vs lambda | ℹ️ Info | Per 05-REVIEW.md WR-02; value 0.0 correct for lambda, comment misleading only. No runtime bug. |

No TBD/FIXME/XXX debt markers introduced. No src stubs (audit-only phase).

### Human Verification Required

None. All four success criteria are evidenced by artifacts on disk with numbers cross-verified against raw criterion output; scope fence machine-checked via git history.

### Gaps Summary

No gaps. Every success criterion (SC1 thread-scaling table + payback-N, SC2 safe-to-parallelize list, SC3 unaccelerated-path cost, SC4 backlog) is present in AUDIT-REPORT.md §Phase 5 (line 612+) with substantive content. Bench artifacts exist and their reported medians/payback numbers match the raw criterion output on spot-check (karcher T=1/T=8, karcher payback N=10, streaming payback N=1). Cited SC2 source line anchors all verify against fdars-core/src. The defining audit-only scope fence (zero fdars-core/src/ edits) is upheld — the full phase-05 commit range modified only the bench harness cell and .planning/ artifacts. PERF-05 is marked complete in REQUIREMENTS.md. The two 05-REVIEW.md warnings are cosmetic bench-code issues (non_snake_case names, a misleading comment) that do not affect the deliverable.

---

_Verified: 2026-08-08T19:09:04Z_
_Verifier: Claude (gsd-verifier)_
