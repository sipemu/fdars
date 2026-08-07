# Phase 3: Elastic Alignment Hot Path - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-07
**Phase:** 3-Elastic Alignment Hot Path
**Areas discussed:** Band fraction & banded scope, karcher_mean fixed params, Cross-distance shape & targets, Backlog root-cause depth

---

## Band fraction & banded scope

| Option | Description | Selected |
|--------|-------------|----------|
| 0.1, all three targets | band_frac=0.1; banded vs unbanded for karcher_mean AND self + cross distance matrices | ✓ |
| 0.1, karcher_mean only | Meet SC2 minimum on karcher_mean; note distance-matrix _banded variants but don't sweep | |
| 0.05, all three | Tighter 5% corridor → larger speedup, higher alignment-quality risk | |

**User's choice:** 0.1, all three targets
**Notes:** band_frac is a domain fraction 0..1 (`band_radius(band_frac, m)`); at M=200 ≈20 pts ≈10× theoretical DP reduction, ~7× after overhead. All three targets have `_banded` twins, so no new library code is needed.

---

## karcher_mean fixed params

| Option | Description | Selected |
|--------|-------------|----------|
| max_iter=20, tol=1e-4, lambda=0.0 | Matches Phase 1 elastic baseline; lambda=0 standard elastic default; deterministic on seeded data | ✓ |
| max_iter=30, tol=1e-5 | More realistic convergence but slower; may exceed 60s budget at N=500×M=200 | |
| max_iter=10 (tractable cap) | Fastest, keeps expensive cell under budget, less representative | |

**User's choice:** max_iter=20, tol=1e-4, lambda=0.0
**Notes:** Chosen for cross-phase comparability with the Phase-1 baseline. Distance-matrix targets take the same lambda=0.0.

---

## Cross-distance shape & targets

| Option | Description | Selected |
|--------|-------------|----------|
| Square N×N, all three targets | data1=data2=N curves (N∈{100,500}); comparable to N×N self-distance; karcher+self+cross in table | ✓ |
| Fixed reference × N (e.g. 50×N) | Train/test kNN scenario with small reference set; less comparable to self-distance | |

**User's choice:** Square N×N, all three targets
**Notes:** Keeps cross-distance cost directly comparable to the self-distance matrix.

---

## Backlog root-cause depth

| Option | Description | Selected |
|--------|-------------|----------|
| Cite Phase 2 + one-line candidate fix | function / current-cost / root-cause (citing AUDIT-REPORT anti-pattern) + one-line candidate fix; GSD-ready | ✓ |
| Numbers + root-cause only | Exactly the SC4 fields, no candidate-fix speculation | |
| Fresh independent root-cause | Re-derive root cause per finding; more thorough but duplicates Phase 2 | |

**User's choice:** Cite Phase 2 + one-line candidate fix
**Notes:** Root-cause cites Phase 2's Anti-Pattern 2 (karcher_mean defaults band_frac=0.0 → unbanded). Candidate-fix framing kept to one line; full fix specs deferred to Phase 9 ranking.

---

## Claude's Discretion

- Criterion `sample_size` / `measurement_time` tuning per cell (workload matrix already prescribes 60s for the N=500×M=200 elastic cell).
- Seeded synthetic N×M input generator for the audit benches (extend the Phase-1 generator).
- Bench-group / artifact-file naming inside the audit bench, following the `p3_<target>_<features>_run<N>.txt` convention.
- Whether banded/unbanded share a criterion group per target or are split.

## Deferred Ideas

- Implementing the banding default / API change — future implementation milestone.
- Allocation profiling (dhat) of elastic SVD copies — Phase 4 (PERF-04).
- Parallelizing sequential elastic FPCA loops — Phase 5 (PERF-05).
- RAYON_NUM_THREADS thread-scaling sweep — Phase 5 (PERF-05).
- Final cross-module bottleneck ranking — Phase 9 (RPT-01).
