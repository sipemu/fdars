---
phase: 05-parallelism-gap-assessment
plan: 03
subsystem: audit-analysis
tags: [parallelism, safe-to-parallelize, unaccelerated-cost, backlog, audit-report, PERF-05]
status: complete
requires:
  - "Plan 01/02: completed SC1 (Thread-Scaling Table both sentinels + Payback-Threshold N) in the ## Phase 5 section"
  - "Phase 2 §Parallelism Gap List (5 SEQUENTIAL D-07 candidates with file:line anchors, independence + RNG notes)"
  - "Phase 1 §Karcher 4-combo baseline (rayon-off ~10×: 1555ms vs 162ms) + Phase 3 §Banded-vs-Unbanded Analysis (banding ~4–6×)"
provides:
  - "### SC2: Sequential Loops Safe to Parallelize — 5 D-07 candidates (independence + RNG note + macro + projected speedup), static-argument-only per D-06"
  - "### SC3: Cost of the Default Unaccelerated Path — rayon-off ~10× (cited p1_karcher_none/parallel) + banding ~4–6× (cited Phase-3 karcher, LOW-CONFIDENCE caveat)"
  - "### SC4: Parallelization Backlog (draft, GSD-ready) — P5-1..P5-4 with Function/Current-cost/Root-cause fields, Phase-9-rankable"
  - "PERF-05 complete: SC1+SC2+SC3+SC4 all delivered under the single ## Phase 5 heading"
affects:
  - "Phase 9 (ranking) consumes the SC4 backlog entries P5-1..P5-4 as candidate requirements"
tech-stack:
  added: []
  patterns:
    - "Static-argument-only parallelism analysis (D-06): independence + RNG note + macro + PROJECTED speedup extrapolated from SC1, no src edits, no new benchmark"
    - "Cite-not-remeasure (D-08): unaccelerated-path costs sourced from existing Phase-1/Phase-3 artifacts with LOW-CONFIDENCE caveat carried forward"
key-files:
  created: []
  modified:
    - ".planning/research/AUDIT-REPORT.md"
decisions:
  - "SC2 delivered as static safety argument + projected speedup with ZERO fdars-core/src edits (D-06 fence) — actual loop-wrapping + measurement deferred to a future implementation milestone."
  - "Projected speedups extrapolated from SC1 karcher thread-scaling (~4.7× ceiling at N=100), bounded by payback-N: heavy-body candidates (cv.rs:76 fold loop) pay back at any N; light-body candidates (elastic_fpca.rs:764, regression.rs:167) are streaming-regime and need a size guard."
  - "SC3 rayon-off ~10× and banding ~4–6× recorded by CITATION only (D-08), no new benchmark; Phase-3 karcher LOW-CONFIDENCE variance (34–204%) corroborated by stable elastic_cross cells (0–4% variance, 4.5–5.7×)."
  - "SC4 banded-default item (P5-4) cross-referenced (not duplicated) from the Phase-3 elastic-alignment backlog to avoid double-counting the same opt-in-default cost."
metrics:
  duration_min: 2
  completed: 2026-08-08
actuals:
  tokens: 4593
  tasks: 3
  commits: 3
---

# Phase 5 Plan 03: SC2/SC3/SC4 Gap Analysis & Backlog Summary

Completed PERF-05 by writing the final three Phase-5 analysis subsections into the `## Phase 5` section of AUDIT-REPORT.md: **SC2** enumerates the five D-07 SEQUENTIAL candidates as safe-to-parallelize with a static independence argument + RNG note + applicable macro + a speedup *projected* from the SC1 karcher scaling; **SC3** records both default-unaccelerated-path costs (rayon-off ~10× and banding ~4–6×) by citation of existing Phase-1/Phase-3 artifacts with the LOW-CONFIDENCE karcher-variance caveat; **SC4** drafts the GSD-ready parallelization backlog (P5-1..P5-4) with Function/Current-cost/Root-cause fields. Zero edits to `fdars-core/src/` — pure audit analysis (D-06 fence).

## What Was Built

**Task 1 — SC2 safe-to-parallelize list (commit 095815d0).**
Added `### SC2: Sequential Loops Safe to Parallelize` to the `## Phase 5` section. A five-row table (one per D-07 candidate: `classification/cv.rs:76`, `elastic_fpca.rs:701/720/764`, `regression.rs:167`) with columns Independence argument | RNG-seeding note | Applicable macro | Projected speedup. Each candidate cites the Phase-2 §Parallelism Gap List as source (not re-derived), states absence of shared mutable state, gives an RNG note (cv.rs:76 explicitly: fold RNG runs once before the loop, no RNG in loop body; the `StdRng::seed_from_u64(seed + k)` pattern referenced as the hypothetical), names the `iter_maybe_parallel!` macro, and projects a ~4–5× machine-default speedup from the SC1 karcher scaling — labeled a PROJECTION and bounded by the payback-N threshold. The subsection opens with the explicit D-06 evidence-standard statement (static argument + projection, no src edits).

**Task 2 — SC3 unaccelerated-path cost record (commit 4a5e972a).**
Added `### SC3: Cost of the Default Unaccelerated Path`, reporting BOTH opt-in dimensions by citation (D-08, no new benchmark): (a) rayon-off `--no-default-features` ~10× from the Phase-1 karcher 4-combo (`""`≈1555 ms vs `parallel`≈162 ms), linking p1_karcher_none/parallel artifacts and distinguishing this rayon-compiled-out cost from the SC1 `RAYON_NUM_THREADS=1` (feature-on) payback baseline; (b) banding opt-in ~7× nominal / measured ~4–6× from the Phase-3 karcher unbanded-vs-banded analysis, naming the `band_frac=0.0` default (Anti-Pattern 2, karcher.rs:300). Carried the LOW-CONFIDENCE caveat on the raw Phase-3 karcher 34–204% variance, corroborated by the stable elastic_cross cells (0–4% variance, 4.5–5.7×).

**Task 3 — SC4 GSD-ready parallelization backlog (commit 57652b87).**
Added `### SC4: Parallelization Backlog (draft, GSD-ready)` reusing the Phase-3 D-07 backlog field format. Four entries: **P5-1** parallelize the CV fold loop (`cv.rs:76`, high priority, heavy body → payback at any N); **P5-2** the three elastic-FPCA N-loops (`elastic_fpca.rs:701/720/764`, medium); **P5-3** `center_columns` (`regression.rs:167`, low — light body, SVD dominates); **P5-4** the elastic-alignment banded-default item cross-referenced (not duplicated) from the Phase-3 backlog. Each carries Function / Current-cost / Root-cause (+ candidate direction + projected/observed reduction + evidence link) and is phrased as a Phase-9-rankable candidate requirement. Exactly one `## Phase 5` heading remains.

## Deviations from Plan

None — plan executed exactly as written. All three tasks' `<verify>` gates passed on first attempt, all acceptance criteria met, and the audit-only scope fence (no `fdars-core/src/` edits) held across the entire plan diff.

### Commit-time infra exception (documented, not a failure)

**Pre-commit hook doctests fail under /tmp exhaustion → `--no-verify` used (all 3 commits).**
- **Issue:** The pre-commit hook runs `cargo test --doc`, linking ~129 doctest binaries in parallel into `/tmp` (a 32G tmpfs at ~95% capacity), producing the documented bus-error linker failure (MEMORY.md: "/tmp exhaustion blocks pre-commit"; matching 05-01/05-02).
- **Handling:** Checked for stale `/tmp/rustdoctest*` leftovers before each commit (none present to free — the pressure is the active 30G session scratchpad, which must not be deleted). Used `git commit --no-verify` per the documented /tmp-exhaustion doctest-link exception. All three commits are **docs-only** (`.planning/research/AUDIT-REPORT.md`) and entirely unrelated to doctests or any code — the exception plainly applies (no code was touched, so the doctest gate has nothing to validate against this plan).

## Scope Fence (threat T-05-03)

`git diff --name-only 095815d0^..HEAD` confirms the plan touched **only** `.planning/research/AUDIT-REPORT.md` — **no file under `fdars-core/src/`** was modified or added. The audit-only scope fence (SC2 is static-argument-only per D-06) was machine-checked in all three task `<verify>` gates and re-checked over the whole plan diff. No benchmark run, no network I/O, no package installs.

## Self-Check: PASSED
- `.planning/research/AUDIT-REPORT.md` — modified; contains `### SC2`, `### SC3`, `### SC4` subsections under exactly one `## Phase 5: Parallelism Gap Assessment` heading (count==1).
- All five D-07 anchors present in SC2 (classification/cv.rs:76, elastic_fpca.rs:701/720/764, regression.rs:167); `seed_from_u64` and `projected` markers present.
- SC3 cites `p1_karcher_none`, names `band_frac`, carries the `low.?confidence` caveat, and includes the `1555`/`10×` rayon-off figure.
- SC4 carries Function/Root-cause fields with the cv.rs:76 and regression.rs:167 anchors; P5-4 cross-references the Phase-3 backlog.
- Commits 095815d0, 4a5e972a, 57652b87 present in `git log`.
- No `fdars-core/src/` path in the plan's diff (095815d0^..HEAD).
