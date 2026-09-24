---
phase: 116-cross-language-re-run-published-report
plan: "03"
subsystem: benchmarks
tags: [benchmark-harness, tdd, resumable-scheduler, criterion, rerun-log, disk-footprint]

# Dependency graph
requires:
  - phase: 116-cross-language-re-run-published-report
    plan: "01"
    provides: "run_all.py's THREAD_ENV/_build_env/_parse_cell/_run_with_timeout/run_rust_bench/_merge_cell_records/run_grid scaffold + --cell/--rust-bench/--rust-filter/--timeout-s CLI this plan extends"
  - phase: 116-cross-language-re-run-published-report
    plan: "02"
    provides: "the six crosslang_<op> criterion groups (crosslang_grid.rs) the P0 manifest tier targets"
provides:
  - "run_all.py RUST_RERUN_CHUNKS — the 67-entry re-run manifest (P0 six crosslang groups first, P1 optimization targets, P2 legacy comparators, P3 remainder) + ORPHAN_GROUPS (six unschedulable baseline groups)"
  - "run_all.py resumable budgeted scheduler: rerun_init, _load_rerun_log/_append_rerun_log (atomic, fail-closed), inferred_timeout_reason, run_rust_next, rust_status, rust_requeue, list_rust_chunks"
  - "run_all.py grid-cell hardening: CellTimeout/CellError, _Deadline, _run_checked, _reset_cell_files, _record_cell_status — a cell timeout/error is recorded and the grid continues; a gate failure records gate-fail then re-raises fail-closed"
  - "run_all.py generate_grid_data, prebuild, _dir_bytes, record_footprint, and run_full — the single --full reproduction command 116-06 will invoke"
  - "benchmarks/test_run_all.py — 15 TestCase classes / 56 tests covering the manifest, rerun log, scheduler, cell hardening, footprint, and --full sequencing"
affects: [116-06, 116-07]

# Actuals (#2632)
actuals:
  tokens: 24480
  tasks: 2
  commits: 4
  plan_head_before: 5182470cae5306fdb45ff0ec21fbcf1e01d6bd06

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Rerun log as an append-only JSON event log: _load_rerun_log/_append_rerun_log round-trip via a temp-file + os.replace atomic write (T-116-11), fail-closed on malformed JSON via SystemExit naming the file"
    - "Chunk pending/final-status derivation purely from the last matching log record for that chunk_idx (_chunk_is_pending/_final_status) rather than a separate status field — the log is the single source of truth, replayable from scratch"
    - "Shared per-cell _Deadline object threaded through generate + every Python/R runner subprocess call so one wall-clock budget covers the whole cell, converting timeout/non-zero-exit into CellTimeout/CellError instead of subprocess.CalledProcessError"
    - "Test-only results_dir/log_path injection points on run_grid so unit tests never touch the real committed benchmarks/results/*.json files or target/criterion"

key-files:
  created:
    - benchmarks/test_run_all.py
  modified:
    - benchmarks/run_all.py

key-decisions:
  - "The 67-chunk manifest's tiers/filters/baseline_mean_ns values were computed from the REAL committed benchmarks/results/comparison_baseline.csv (grouped by the `source` column's criterion group, not the `op` label) rather than re-deriving from the plan prose alone — every per-id filter's baseline_mean_ns was cross-checked against the exact baseline row it targets (e.g. audit_p3_elastic_cross/n500_m50 = 37964661211.7 ns = 37.965s, matching the plan's interfaces text) and group-level sums were computed by script, not by hand."
  - "A first-executed chunk's timeout is FINAL (status timeout) and the scheduler loop CONTINUES afterward (inferred-timeout skips cost no budget, and a further chunk still obeys the >=60s remaining-budget check); only a LATER chunk's timeout is recorded as non-final 'interrupted' and hard-stops the call. This is a deliberate reading of the plan's `<behavior>` spec: only the later-chunk case is described as stopping the call."
  - "run_grid gained `results_dir`/`log_path` parameters (defaulting to the prior hardcoded real paths) purely for test isolation — the real CLI path is unchanged, but unit tests can now redirect all per-cell file I/O to a tmp_path without ever touching the committed python_n100_m50.json/r_n100_m50.json files."
  - "TDD RED evidence was captured by reverting benchmarks/run_all.py to the pre-plan (116-02) commit for each task's RED phase (git checkout + selective re-application), since both tasks land in the same two files and a naive 'run pytest against the finished file' would not have produced a real failing-test artifact."

requirements-completed: [RPT-01]

coverage:
  - id: D1
    description: "67-entry RUST_RERUN_CHUNKS manifest: P0 six crosslang groups first, P1/P2/P3 covering every re-runnable baseline group (ORPHAN_GROUPS excludes the six that no longer exist), per-id chunks ascending by baseline_mean_ns"
    requirement: RPT-01
    verification:
      - kind: unit
        ref: "benchmarks/test_run_all.py::TestChunkManifest (8 tests, including test_coverage_every_rerunnable_baseline_id_matches_a_chunk against the real comparison_baseline.csv and fdars-core/benches/*.rs)"
        status: pass
      - kind: other
        ref: "benchmarks/run_all.py --list-rust-chunks -> exactly 67 lines, first six are the crosslang groups"
        status: pass
    human_judgment: false
  - id: D2
    description: "Resumable budgeted scheduler (rerun_init, run_rust_next, rust_status, rust_requeue) with fail-closed atomic rerun log persistence"
    requirement: RPT-01
    verification:
      - kind: unit
        ref: "benchmarks/test_run_all.py::TestRerunLog, TestRerunInit, TestScheduler, TestRustStatus, TestRustRequeue, TestRemediationRecord (28 tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Grid-cell hardening (CellTimeout/CellError, shared per-cell deadline, stale-file reset, gate-fail recording) and the single --full reproduction command"
    requirement: RPT-01
    verification:
      - kind: unit
        ref: "benchmarks/test_run_all.py::TestCellFiles, TestRunWithTimeout, TestFootprint, TestGridCellHardening, TestRunFull (18 tests)"
        status: pass
      - kind: integration
        ref: "benchmarks/.venv/bin/python -m pytest benchmarks/ -q (192 passed, full suite)"
        status: pass
      - kind: other
        ref: "run_all.py --grid --dry-run --cell 100x50 --cell 500x200 prints exactly 2 [GRID] Cell lines; --help lists every new flag"
        status: pass
    human_judgment: false

duration: 28min
completed: 2026-09-24
status: complete
---

# Phase 116 Plan 03: Resumable Budgeted fdars Criterion Re-Run + Grid Hardening Summary

**67-chunk resumable scheduler (rerun_init/run_rust_next/rust_status/rust_requeue) plus grid-cell timeout/error hardening and the single `run_all.py --full` reproduction command, all TDD RED→GREEN with 56 new tests.**

## Performance

- **Duration:** 28 min
- **Started:** 2026-09-24T19:58:00Z (approx.)
- **Completed:** 2026-09-24T20:26:00Z
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments

- `RUST_RERUN_CHUNKS`: a 67-entry manifest covering every re-runnable baseline criterion id — six `crosslang_<op>` groups first (Tier P0, no Phase 113 baseline), then Phase 114/115 optimization targets (Tier P1), legacy cross-language comparators (Tier P2), and the remainder of the baseline suite either per-id (ascending `baseline_mean_ns`, for the slow elastic/Karcher groups) or group-level (Tier P3). `ORPHAN_GROUPS` documents the six baseline groups (`face_covariance`, `coclustering_co_cluster`, `fem_smooth`, `fpca_variants_fsvd`, `frechet_anova`, `fts_long_run_covariance`) whose `benchmark_group(...)` literal no longer exists — they cannot be re-run and the report will list them as stale rows.
- A resumable, budgeted scheduler: `rerun_init` (archives stale `target/criterion` by rename, refuses if the log already exists), `run_rust_next` (bounded wall-clock budget — the first chunk executed this call gets a final `timeout` on expiry and the loop continues; a later chunk's timeout is non-final `interrupted` and stops the call; a costlier same-group sibling of an already-timed-out chunk is skipped as `skipped-inferred-timeout` without running; an `error` status exits 3), `rust_status`, and `rust_requeue` (only from a final `error` status — a genuine timeout, ok, or inferred-skip can never be erased).
- Grid-cell hardening: `CellTimeout`/`CellError` plus a shared `_Deadline` cover the generator and every Python/R runner subprocess call for one cell; a timeout or error is recorded (`grid_cell` status) and the grid run continues with the next cell (exiting 4 at the end if any cell failed this way); an unfiltered cell run resets stale per-cell files first (`_reset_cell_files`); a gate failure still aborts via the existing `sys.exit(1)` (fail-closed, D-11, `gate.py` untouched) but now records `gate-fail` first.
- `generate_grid_data`, `prebuild` (`cargo bench --no-run` under a timeout, resumable), `_dir_bytes` + `record_footprint` (target/, target/criterion/, target/release/ disk usage), and `run_full` — the single `--full` reproduction command 116-06 will invoke: rerun-init → generate-grid-data → prebuild → resumable rust-next loop → the 6-cell grid → capture-env → collect-rust → `assemble_comparison.py --gate-passed` → footprint.
- 56 new tests across 15 `TestCase` classes, each RED-then-GREEN per task (RED evidence captured by reverting `run_all.py` to the pre-plan commit for each task).

## Task Commits

Each task ran a full RED → GREEN TDD cycle, per the plan's `tdd="true"` tasks:

1. **Task 1 RED: add failing tests for chunk manifest + rerun log + scheduler** - `01eb9bc2` (test)
2. **Task 1 GREEN: implement chunk manifest + rerun log + budgeted resumable scheduler** - `12776609` (feat)
3. **Task 2 RED: add failing tests for grid-cell hardening + footprint + --full** - `4e85737e` (test)
4. **Task 2 GREEN: implement grid-cell hardening + prebuild + footprint + --full** - `89ecb6a8` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `benchmarks/run_all.py` - `RUST_RERUN_CHUNKS`/`ORPHAN_GROUPS`, rerun-log helpers, `run_rust_next`/`rust_status`/`rust_requeue`/`list_rust_chunks`, `CellTimeout`/`CellError`/`_Deadline`/`_run_checked`/`_reset_cell_files`, `generate_grid_data`/`prebuild`/`_dir_bytes`/`record_footprint`/`run_full`, `run_grid` hardening, 9 new CLI flags
- `benchmarks/test_run_all.py` - new file, 15 `TestCase` classes / 56 test methods

## Decisions Made

See `key-decisions` in the frontmatter above (manifest values cross-checked against the real committed baseline CSV, the first-timeout-continues-vs-later-timeout-stops scheduler semantics, test-only `results_dir`/`log_path` injection on `run_grid`, and the git-revert-based RED evidence methodology).

## Deviations from Plan

None — plan executed exactly as written. Both tasks carried `tdd="true"`; each ran the canonical RED → GREEN cycle. One clarification worth noting: since both tasks' RED/GREEN cycles land in the same two files, genuine RED evidence for each task required `git checkout` reverting `run_all.py` to the pre-plan commit, writing only that task's test additions, confirming failures, committing, then re-applying that task's implementation — rather than writing both files once and treating the finished state as sufficient. This is a testing-methodology clarification, not a change to what was built.

## Issues Encountered

None.

## Known Stubs

None introduced by this plan. `benchmarks/results/rerun_log.json` does not exist yet (by design — this plan only builds the scheduler/hardening plumbing and proves it with unit tests against temp paths; 116-06 performs the real re-run that creates the log for real).

## Threat Flags

None beyond the plan's own `<threat_model>` (T-116-08..12), all implemented as specified: `_reset_cell_files` only ever builds filenames from validated `GRID_N x GRID_M` integers; `rerun_init` renames (never deletes) a stale `target/criterion`; `_run_with_timeout`'s process-group SIGKILL is unit-tested with a real grandchild process; `rerun_log.json` parsing fails closed with `SystemExit` naming the file and writes are atomic (`os.replace`); every new subprocess call site remains list-form only (verified by the plan's `shell=True` grep acceptance criterion).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

`run_all.py` is ready for 116-06's real end-to-end re-run: `--full` (or the individually callable `--rerun-init`/`--generate-grid-data`/`--prebuild`/`--rust-next --budget-s <N>`/`--rust-status`/`--rust-requeue`/`--grid --cell-timeout-s <N>`/`--record-footprint` steps, for resuming across multiple bounded foreground calls) is the single reproduction command the report's Reproduction section will document. No `fdars-core/src` or `benchmarks/gate.py` change was made; no new dependency was introduced.

## Self-Check: PASSED

- Created/modified files verified present: `benchmarks/run_all.py` (67-entry `RUST_RERUN_CHUNKS`, `class CellTimeout` x1, `def _reset_cell_files` x1, `def _dir_bytes` x1, `gate-fail` x4, `interrupted` x6, `RUST_CHUNKS_REMAINING=` x2, `os.replace` x2 — all confirmed via grep), `benchmarks/test_run_all.py` (15 `TestCase` classes confirmed via grep) — both present on disk.
- Commits verified: `git log --oneline` shows `01eb9bc2`, `12776609`, `4e85737e`, `89ecb6a8` all present on `main`; `git rev-list --count 5182470c..HEAD` = 4 (matches `actuals.commits`).
- All plan-level `<verify>` and task `<verify>`/`<acceptance_criteria>` commands re-run and passing: `benchmarks/.venv/bin/python -m pytest benchmarks/test_run_all.py -q` (56 passed), `benchmarks/.venv/bin/python -m pytest benchmarks/ -q` (192 passed), `--list-rust-chunks` (67 lines, first six crosslang), `--grid --dry-run --cell 100x50 --cell 500x200` (2 `[GRID] Cell` lines), `--help` (all 9 new flags listed), `git diff --name-only ec57fa01 HEAD -- benchmarks/gate.py fdars-core/src` (empty), `grep -nE 'shell *= *True' benchmarks/run_all.py` (no unguarded matches), `test ! -e benchmarks/results/rerun_log.json` (confirmed absent).

---
*Phase: 116-cross-language-re-run-published-report*
*Completed: 2026-09-24*
