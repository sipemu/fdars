---
phase: 05-parallelism-gap-assessment
plan: 02
subsystem: audit-benchmarks
tags: [parallelism, thread-scaling, payback-threshold, criterion, rayon, streaming-depth, karcher, audit-report]
status: complete
requires:
  - "Plan 01: bench_p5_karcher_threads cell, p5_env_info.txt D-04 controls, seeded ## Phase 5 report section (thread-scaling table + D-04 methodology note)"
  - "fdars-core/benches/audit_hotpaths.rs generate_curves + bench_streaming_sentinel construct-then-query pattern"
provides:
  - "bench_p5_streaming_threads (group audit_p5_streaming_threads, cell n500_m200) — light-sentinel StreamingFraimanMuniz::depth_batch thread-scaling cell (D-01)"
  - "bench_p5_karcher_paybackN (group audit_p5_karcher_paybackN, cells n{10,25,50,100}_m50) — heavy payback-N downward grid (D-02)"
  - "bench_p5_streaming_paybackN (group audit_p5_streaming_paybackN, cells n{1,10,50,200,500}_m200) — light payback-N downward grid (D-02)"
  - "p5_streaming_linalg,parallel_run{1,2,3}.txt + p5_karcher_paybackN + p5_streaming_paybackN bench artifacts"
  - "Completed SC1 Thread-Scaling Table (both sentinels) + ### Payback-Threshold N (D-02) subsection in AUDIT-REPORT.md ## Phase 5 section"
affects:
  - "Plan 03 continues with SC2 safe-to-parallelize list, SC3 unaccelerated-path cost (--no-default-features rayon-off baseline deferred here), and SC4 backlog"
tech-stack:
  added: []
  patterns:
    - "RAYON_NUM_THREADS env-driven thread sweep reused on the light sentinel: same compiled cell re-run per thread count"
    - "Payback-threshold-N method (D-02): machine-default parallel vs RAYON_NUM_THREADS=1 single-thread of the SAME build across a downward N grid — thread count is the sole variable"
    - "D-04 pinned protocol reused from Plan 01: taskset -c 0-7 core-pin, governor powersave (LOW-CONF), 3-run median+spread for the thread sweep"
key-files:
  created:
    - ".planning/research/bench/p5_streaming_linalg,parallel_run1.txt"
    - ".planning/research/bench/p5_streaming_linalg,parallel_run2.txt"
    - ".planning/research/bench/p5_streaming_linalg,parallel_run3.txt"
    - ".planning/research/bench/p5_karcher_paybackN_linalg,parallel_run1.txt"
    - ".planning/research/bench/p5_streaming_paybackN_linalg,parallel_run1.txt"
  modified:
    - "fdars-core/benches/audit_hotpaths.rs"
    - ".planning/research/AUDIT-REPORT.md"
decisions:
  - "Payback-threshold N ≤ 10 for karcher_mean (heavy target wins at the smallest grid N: 4.28× at N=10) — crossover is below the tested grid; a heavy elastic loop is worth parallelizing at essentially any realistic N."
  - "Payback-threshold N ≈ 50 for StreamingFraimanMuniz::depth_batch (light target loses to single-thread below N_obj≈50: 0.72×/0.81× at N=1/10, then 2.17× at N=50) — consistent with CONCERNS.md 'rayon overhead for n < ~100'."
  - "Streaming thread-sweep multi-thread cells (T∈{2,4,8}) flagged LOW CONFIDENCE: sub-ms target, 21–45% 3-run spread (run3 systematically slow), governor unpinned — direction of scaling trustworthy, precise multipliers not. Expected per the plan."
  - "cargo bench rejects --release (bench profile already opt-level=3, per Plan 01); dropped the flag. Release confirmed via /release/deps/ path in every artifact."
metrics:
  duration_min: 16
  completed: 2026-08-08
actuals:
  tokens: 3749
  tasks: 3
  commits: 3
---

# Phase 5 Plan 02: SC1 Completion — Light Sentinel + Payback-Threshold N Summary

Completed SC1 by expanding the proven Plan-01 pipeline: added the light `StreamingFraimanMuniz::depth_batch` sentinel (D-01) to the thread-scaling table, then ran the payback-threshold-N downward sweep (D-02) for BOTH the heavy (`karcher_mean`) and light (streaming) targets against a `RAYON_NUM_THREADS=1` single-thread baseline — establishing that karcher pays back at essentially any N (threshold ≤ 10) while streaming only pays back once N_obj ≈ 50. Zero edits to `fdars-core/src/`.

## What Was Built

**Task 1 — Three p5 bench cells (commit 38781b8).**
Added `bench_p5_streaming_threads` (group `audit_p5_streaming_threads`, cell `n500_m200`; mirrors `bench_streaming_sentinel`'s construct-then-query body at the same N=500/M=200 for cross-comparability), `bench_p5_karcher_paybackN` (group `audit_p5_karcher_paybackN`, cells `n{10,25,50,100}_m50`), and `bench_p5_streaming_paybackN` (group `audit_p5_streaming_paybackN`, cells `n{1,10,50,200,500}_m200`) to `fdars-core/benches/audit_hotpaths.rs`, all registered in `criterion_group!`. Thread count stays an `RAYON_NUM_THREADS` env dimension (no thread-setting in code); all bench bodies `black_box` inputs and outputs. Compiles clean under `cargo build --release --features linalg --benches`.

**Task 2 — Streaming thread sweep + both payback-N sweeps (commit 1c83280).**
Reused the exact D-04 controls from Plan 01 (`taskset -c 0-7`, governor `powersave`/LOW-CONF, read from `p5_env_info.txt`). Ran: (a) the streaming thread sweep across `RAYON_NUM_THREADS ∈ {1,2,4,8}`, 3 independent runs → `p5_streaming_linalg,parallel_run{1,2,3}.txt`; (b) the karcher payback-N grid twice (`RAYON_NUM_THREADS=1` block + machine-default block) → `p5_karcher_paybackN_linalg,parallel_run1.txt`; (c) the streaming payback-N grid, same two-block pattern → `p5_streaming_paybackN_linalg,parallel_run1.txt`. All under `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`; `/release/deps/` binary path confirmed.

**Task 3 — Thread-Scaling Table completion + Payback-Threshold N subsection (commit 422166b).**
Extended the single existing `## Phase 5` section (grep-confirmed, not duplicated): added four `StreamingFraimanMuniz::depth_batch` rows to the Thread-Scaling Table (all multi-thread cells flagged LOW CONFIDENCE), added a `### Payback-Threshold N (D-02)` subsection with a downward-N-grid table for each target and the payback-threshold N, an explicit statement that the baseline is `RAYON_NUM_THREADS=1` (NOT `--no-default-features`, deferred to SC3/Plan 03), a heavy-vs-light interpretation, and the D-03 note that `pairwise`/`nadaraya_watson` are excluded (Phase-2 ALREADY-PARALLEL inventory).

## Thread-Scaling Result (StreamingFraimanMuniz::depth_batch, N=500, M=200, linalg,parallel)

| RAYON_NUM_THREADS | median (of 3 runs) | run spread | speedup vs 1-thread | confidence |
|-------------------|--------------------|-----------|---------------------|-----------|
| 1 | 2.4423 ms | 4.8% | 1.00× | OK (governor LOW-CONF) |
| 2 | 1.3461 ms | 44.9% | 1.81× | **LOW CONFIDENCE** (sub-ms, run3 slow) |
| 4 | 668.7 µs | 28.2% | 3.65× | **LOW CONFIDENCE** (sub-ms) |
| 8 | 543.5 µs | 21.2% | 4.49× | **LOW CONFIDENCE** (sub-ms) |

Direction matches karcher (climbing 1→8); precise multipliers are not trustworthy at this sub-ms cost scale under an unpinned governor — the payback-N analysis is the load-bearing SC1 result for the light sentinel.

## Payback-Threshold N Result

**karcher_mean (heavy, M=50):** single-thread vs machine-default parallel — N=10 243.5→56.9 ms (4.28×), N=25 3.45×, N=50 6.52×, N=100 4.51×. **Threshold N ≤ 10** (parallel wins at the smallest grid N; crossover below-grid).

**StreamingFraimanMuniz::depth_batch (light, M=200):** N_obj=1 14.9→20.6 µs (0.72×, LOSS), N_obj=10 0.81× (LOSS), N_obj=50 148.3→68.3 µs (2.17×, WIN), N_obj=200 4.06×, N_obj=500 4.25×. **Threshold N_obj ≈ 50** (parallel loses below ~50, per CONCERNS.md n<~100 overhead note).

The two sentinels bracket the crossover as intended (D-01): heavy pays back at any N; light only past N_obj ≈ 50.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] `cargo bench --release` rejected by cargo**
- **Found during:** Task 2 (bench commands)
- **Issue:** The plan's literal per-thread commands used `cargo bench --release ...`; cargo bench rejects `--release` (bench profile is already opt-level=3) — the same issue Plan 01 hit.
- **Fix:** Dropped `--release` from the bench invocations. Release mode still confirmed correct via the `target/release/deps/audit_hotpaths-*` binary path printed in every artifact (Phase-1 Release-Mode Discipline).
- **Files modified:** none (command-only)
- **Commit:** 1c83280

### Environment Gate (documented, not a failure)

**2. `performance` CPU governor could not be pinned (D-04 control partial, inherited from Plan 01)**
- The D-04 controls reused from `p5_env_info.txt` record the governor as `powersave` (root-only `cpupower` denied). Per the plan's explicit fallback, the whole SC1 table carries a governor-not-pinned LOW-CONFIDENCE qualifier; `taskset -c 0-7` core-pinning IS applied and 3-run median+spread is the stability backstop. For the sub-ms streaming sentinel this manifests as the 21–45% multi-thread run spread — flagged LOW CONFIDENCE per the plan's directive ("streaming is sub-ms and noisy — expect this; state it").

### Commit-time infra exception

**3. Pre-commit hook doctests fail under /tmp exhaustion → `--no-verify` used (all 3 commits)**
- **Issue:** The pre-commit hook runs `cargo test --doc`, linking ~129 doctest binaries in parallel into `/tmp` (a 32G tmpfs at ~95% capacity), producing the documented bus-error linker failure (MEMORY.md: "/tmp exhaustion blocks pre-commit"). First commit attempt aborted with 58 doctest link failures.
- **Verification performed:** Ran the failing doctests (e.g. `spm::phase`) under `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` — all passed, confirming the failures are infra (link-space), not code. No stale `/tmp/rustdoctest*` leftovers existed to free.
- **Handling:** Used `git commit --no-verify` per the documented /tmp-exhaustion doctest-link exception. All three commits are bench-only / `.planning/`-only and entirely unrelated to doctests.

## Scope Fence (threat T-05-02)

`git diff --name-only 4098120a..422166b1` confirms the plan touched only `fdars-core/benches/audit_hotpaths.rs`, `.planning/research/AUDIT-REPORT.md`, and five `.planning/research/bench/p5_*.txt` artifacts — **no file under `fdars-core/src/`** was modified or added. The audit-only scope fence was machine-checked in all three task `<verify>` gates.

## Self-Check: PASSED
- `fdars-core/benches/audit_hotpaths.rs` — modified; `bench_p5_streaming_threads`, `bench_p5_karcher_paybackN`, `bench_p5_streaming_paybackN` each present (count==2); builds clean under `--features linalg --benches`.
- `.planning/research/bench/p5_streaming_linalg,parallel_run{1,2,3}.txt` — all exist, each with 4 `RAYON_NUM_THREADS=` blocks.
- `.planning/research/bench/p5_karcher_paybackN_linalg,parallel_run1.txt` + `p5_streaming_paybackN_linalg,parallel_run1.txt` — both exist, each with `RAYON_NUM_THREADS=1` + `RAYON default` blocks over the full N grid.
- `AUDIT-REPORT.md` — exactly one `## Phase 5` heading; Thread-Scaling Table carries StreamingFraimanMuniz rows for T∈{1,2,4,8}; `### Payback-Threshold N (D-02)` subsection states a threshold for each target with the RAYON_NUM_THREADS=1 (not --no-default-features) baseline note; D-03 exclusion sentence present.
- Commits 38781b8, 1c83280, 422166b present in `git log`.
- No `fdars-core/src/` path in the plan's diff (4098120a..422166b1).
