---
phase: 116-cross-language-re-run-published-report
plan: 05
subsystem: benchmarks
tags: [ab, criterion, face_covariance, headline, noise-gate]
requirements: [RPT-01]
provides: [phase116-face-covariance-ab, headline-eligibility]
key-files:
  created:
    - benchmarks/results/ab/phase116_face_covariance_n200_m30/before.json
    - benchmarks/results/ab/phase116_face_covariance_n200_m30/after.json
    - benchmarks/results/ab/phase116_face_covariance_n200_m30/before_r0.json
    - benchmarks/results/ab/phase116_face_covariance_n200_m30/after_r0.json
  modified:
    - benchmarks/results/ab/manifest.json
key-decisions:
  - "Round 0 (in-file settings sample_size 15 / 30 s / 3 s warm-up) already passes the noise gate on both sides with disjoint CIs — no settings bump rounds run"
  - "Single-threaded pinned speedup is 1.19x, NOT the ~3.45x Phase 115 measured on the default multi-threaded rayon pool; the report must headline 1.19x (pinned) and explain the discrepancy honestly"
duration: ~50 min (incl. ~40 min waiting for the machine to become quiet)
completed: 2026-09-24
---

# Plan 116-05 Summary: Controlled face_covariance A/B

Executed INLINE by the orchestrator (long cargo steps stall subagents).

## What was done
- Scratch detached worktrees: before = `7cde79b2` (parent of OPT-02 `c0e84bd8`), after = `e98b862d` (HEAD; `fdars-core/src` identical to `c0e84bd8`, verified). Main-checkout `Cargo.lock` copied into both, all builds `--locked`.
- Both sides built with separate `CARGO_TARGET_DIR`, then measured sequentially (before → after) with all six `*_NUM_THREADS=1`.
- Quiet-machine check: initially load 8–15 from unrelated workloads (duckdb, tsc, R tests); the user freed the machine; measurements taken at load1 = 1.78 (before) and 1.42 (after).
- Estimates copied to `benchmarks/results/ab/phase116_face_covariance_n200_m30/` (final + `_r0`), manifest entry appended (phase114/115 entries untouched). Worktrees removed + pruned, `target/ab116` deleted. Main checkout `fdars-core/src` + `perf_hotpaths.rs` untouched.

## Results (criterion, perf_face_covariance/n200_m30)
- BEFORE: [300.25 ms 301.90 ms 303.99 ms]
- AFTER:  [249.81 ms 251.93 ms 256.03 ms]

Final `render_report.py --ab-check` output (verbatim):
```
AB phase114_elastic_cross_banded_n500_m50 speedup=1.020 delta_pct=-2.00 significant=False noise=FAIL-NOISY/PASS headline_eligible=False
AB phase115_face_covariance_n200_m30 speedup=3.447 delta_pct=-70.99 significant=True noise=FAIL-NOISY/PASS headline_eligible=False
AB phase115_svd_rebench kind=rebench (no before/after pair)
AB phase116_face_covariance_n200_m30 speedup=1.192 delta_pct=-16.08 significant=True noise=PASS/PASS headline_eligible=True
AB-CHECK: PASS (phase116_face_covariance_n200_m30 eligible)
```

## Honesty note (must carry into the report)
The pinned single-thread speedup is **1.19×**, well below Phase 115's **~3.45×**. Phase 115's A/B used the default (multi-threaded, not pinned) rayon pool, and its BEFORE cell failed the noise gate. The controlled, noise-clean figure is 1.19×. The report headlines 1.19× (pinned) and notes the Phase 115 multi-threaded figure with its noise flag.

## Deviations
- Rounds 1–2 not needed (round 0 eligible), so no `before_r1/r2` files exist. This is permitted by the plan ("only the rounds actually run").
- The waiting time for a quiet machine was spent outside the plan's control (user-freed).

## Commits
- `3f8d0c78` feat(116-05): controlled thread-pinned face_covariance A/B (1.19x, headline-eligible)
