---
phase: 114-elastic-alignment-optimization
plan: "02"
subsystem: alignment
tags: [elastic-alignment, dp-fill-grid, numeric-identity, criterion, banded-optimization, OPT-01]

dependency_graph:
  requires:
    - phase: 114-01
      provides:
        - fdars-core/tests/equivalence_phase114.rs (golden fixture, 2 tests at tol 1e-10)
        - .planning/phases/114-elastic-alignment-optimization/114-baseline-criterion/ (BEFORE baseline: mean 7.310s, CI [7.073s, 7.552s])
  provides:
    - fdars-core/src/alignment/mod.rs — dp_fill_grid band-iteration refactor (OPT-01), dp_skip_cell removed
    - .planning/phases/114-elastic-alignment-optimization/114-after-criterion/audit_p3_elastic_cross_banded__n500_m50/estimates.json
    - .planning/phases/114-elastic-alignment-optimization/114-after-criterion/audit_p3_elastic_cross_banded__n500_m50/COMPARISON.md
  affects:
    - Plan 114-03 (consumes the committed after-criterion + honest speedup verdict)

actuals:
  tokens: 9500
  tasks: 2
  commits: 2
  plan_head_before: e7db7d2276f0a0b76e3fad0d153fea34076428af

tech_stack:
  added: []
  patterns:
    - Band-iteration by construction — banded dp_fill_grid arm iterates tc in tr.saturating_sub(r)..=(tr+r).min(ncols-1); the old skip-based loop has been replaced; the function is O(n*r) by inspection.
    - Dead code removal — dp_skip_cell (sole caller was dp_fill_grid) removed after its logic was folded inline.
    - Numeric-identity via golden fixture — debug + release assertion pattern (equivalence_phase114) provides the change-detection signal for future dp_fill_grid edits.

key_files:
  created:
    - .planning/phases/114-elastic-alignment-optimization/114-after-criterion/audit_p3_elastic_cross_banded__n500_m50/estimates.json
    - .planning/phases/114-elastic-alignment-optimization/114-after-criterion/audit_p3_elastic_cross_banded__n500_m50/COMPARISON.md
  modified:
    - fdars-core/src/alignment/mod.rs — dp_fill_grid refactored, dp_skip_cell removed
    - fdars-core/tests/equivalence_phase114.rs — doc indentation fix (clippy lint)

key-decisions:
  - "CIs overlap at n500_m50 (AFTER-upper 7.292s > BEFORE-lower 7.073s) — reported honestly as within-noise; tolerance never relaxed and no speedup fabricated (D-honesty-clause)."
  - "dp_skip_cell removed entirely (sole caller was dp_fill_grid; origin-check folded inline as early continue) to avoid clippy dead-code warning."
  - "Unbanded None arm kept as simple full-scan with inline origin skip — no change to algorithmic behaviour or performance; unbanded is already near-optimal per RESEARCH."
  - "Full e.resize(size, INFINITY) retained in dp_grid_solve_banded — Pitfall 5 (out-of-band predecessor reads relying on INFINITY guard) is not addressed in this plan; selective-scratch-init is deferred."

requirements-completed: [OPT-01]

coverage:
  - id: D1
    description: "dp_fill_grid banded arm iterates diagonal corridor only (O(n*r) by construction), output bit-identical — golden fixture passes at <=1e-10 in debug and release."
    requirement: OPT-01
    verification:
      - kind: integration
        ref: "fdars-core/tests/equivalence_phase114.rs#golden_elastic_self_distance_banded_n20_m50 (debug)"
        status: pass
      - kind: integration
        ref: "fdars-core/tests/equivalence_phase114.rs#golden_elastic_self_distance_unbanded_n20_m50 (debug)"
        status: pass
      - kind: integration
        ref: "fdars-core/tests/equivalence_phase114.rs#golden_elastic_self_distance_banded_n20_m50 (release)"
        status: pass
      - kind: integration
        ref: "fdars-core/tests/equivalence_phase114.rs#golden_elastic_self_distance_unbanded_n20_m50 (release)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Determinism guardrail (equivalence_phase47/48/49/50) green in debug and release."
    verification:
      - kind: integration
        ref: "cargo test --test equivalence_phase47 --test equivalence_phase48 --test equivalence_phase49 --test equivalence_phase50 — 8+7 tests, 0 failed"
        status: pass
    human_judgment: false
  - id: D3
    description: "All five build variants compile: default, linalg, serde, linalg+serde+parallel, WASM."
    verification:
      - kind: other
        ref: "cargo build -p fdars-core (default, linalg, serde, linalg+serde+parallel, --target wasm32-unknown-unknown) — all Finished dev"
        status: pass
    human_judgment: false
  - id: D4
    description: "AFTER criterion committed with honest before/after verdict — CIs overlap, speedup within noise."
    verification:
      - kind: other
        ref: ".planning/phases/114-elastic-alignment-optimization/114-after-criterion/audit_p3_elastic_cross_banded__n500_m50/COMPARISON.md"
        status: pass
    human_judgment: false

duration: 10min
completed: "2026-09-23"
status: complete
---

# Phase 114 Plan 02: Elastic Alignment Band-Iteration Optimization Summary

**dp_fill_grid banded arm refactored to O(n*r) diagonal-corridor iteration; numeric identity proven by golden fixture (debug + release at <=1e-10); AFTER criterion within noise at n500_m50 (-2.0%, CIs overlap, reported honestly per D-honesty-clause).**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-09-23T10:54:36Z
- **Completed:** 2026-09-23T11:04:36Z
- **Tasks:** 2
- **Files modified:** 2 (alignment/mod.rs, equivalence_phase114.rs)
- **Files created:** 2 (estimates.json, COMPARISON.md)

## Accomplishments

- Replaced the `dp_fill_grid` full-scan + per-cell `dp_skip_cell` pattern with a `match band` that iterates the corridor directly for the banded arm, eliminating ~79% of per-row skip-checks at n500_m50/r=5.
- Removed `dp_skip_cell` (now dead — its only caller was `dp_fill_grid`), keeping clippy clean under `--all-targets --features linalg,parallel`.
- Proved numeric identity: both golden tests (banded + unbanded) pass unchanged at <=1e-10 in debug AND release builds; full test suite (274 alignment tests, 1654+ total) green in both modes.
- Confirmed all five build variants compile (default / linalg / serde / linalg+serde+parallel / WASM).
- Captured honest AFTER criterion: mean 7.164s vs BEFORE 7.310s; CIs overlap (AFTER-upper 7.292s > BEFORE-lower 7.073s); criterion p=0.33 — speedup within measurement noise at this cell.

## dp_fill_grid Diff Summary

**BEFORE:** single nested loop `for tr in 0..nrows { for tc in 0..ncols { if dp_skip_cell(tr, tc, band) { continue; } ... } }` — scanned all m^2 cells and `continue`-skipped ~79% via `dp_skip_cell`.

**AFTER:**
```rust
match band {
    None => {
        for tr in 0..nrows {
            for tc in 0..ncols {
                if tr == 0 && tc == 0 { continue; }
                dp_relax_cell(e, parent, ncols, tr, tc, edge_cost);
            }
        }
    }
    Some(r) => {
        for tr in 0..nrows {
            let tc_lo = tr.saturating_sub(r);
            let tc_hi = (tr + r).min(ncols - 1);
            for tc in tc_lo..=tc_hi {
                if tr == 0 && tc == 0 { continue; }
                dp_relax_cell(e, parent, ncols, tr, tc, edge_cost);
            }
        }
    }
}
```

The `Some(r)` arm relaxes exactly the cells where `abs_diff(tr, tc) <= r`, in the same row-major order — bit-identical by construction. `dp_relax_cell`, `dp_edge_weight_core`, `COPRIME_NBHD_7`, and the full INFINITY scratch-init in `dp_grid_solve_banded` are **unchanged**.

## BEFORE vs AFTER Criterion

| Metric          | BEFORE (SHA 8c0b1d7d)    | AFTER (SHA 30d2c9a2)      | Delta           |
|-----------------|--------------------------|---------------------------|-----------------|
| mean            | 7,310 ms                 | 7,164 ms                  | -146 ms (-2.0%) |
| CI lower (95%)  | 7,073 ms                 | 7,025 ms                  |                 |
| CI upper (95%)  | 7,552 ms                 | 7,292 ms                  |                 |
| std_dev         | 406 ms                   | 227 ms                    |                 |

**Overlap verdict:** CIs overlap (AFTER-upper 7,292 ms > BEFORE-lower 7,073 ms). Criterion p=0.33 > 0.05. The measured -2.0% improvement is within noise at this cell (n=500, m=50, r=5).

**Why within noise at this cell:** with r=5 and m=50, each row already only needs 11 relaxations. The old skip-check cost (`abs_diff + comparison`, 1-2 ns each) is small relative to one `dp_relax_cell` call (7 neighbors, f64 compare+assign chain). The algorithmic saving (39 skipped checks/row eliminated) is subpercent of total wall time — masked by criterion's natural ±3% variation.

**Unbanded rank-#1 path:** intentionally not touched. RESEARCH confirmed it is already near-optimal (no per-cell branch overhead, `dp_relax_cell` dominates). The `match` dispatch adds one branch per `dp_fill_grid` call — negligible.

## Task Commits

1. **Task 1: Band-iteration refactor of dp_fill_grid** — `30d2c9a2` (refactor)
2. **Task 2: AFTER criterion + honest comparison** — `aed0251c` (chore)

## Files Created/Modified

- `fdars-core/src/alignment/mod.rs` — dp_fill_grid refactored (match-on-band, diagonal corridor); dp_skip_cell removed
- `fdars-core/tests/equivalence_phase114.rs` — doc overindent fix (clippy lint from Plan 01)
- `.planning/phases/114-elastic-alignment-optimization/114-after-criterion/audit_p3_elastic_cross_banded__n500_m50/estimates.json` — AFTER criterion raw data
- `.planning/phases/114-elastic-alignment-optimization/114-after-criterion/audit_p3_elastic_cross_banded__n500_m50/COMPARISON.md` — honest before/after verdict

## Decisions Made

- Reported AFTER CI overlap honestly; did not relax golden tolerance or fabricate a speedup.
- Removed `dp_skip_cell` entirely (dead code) rather than keeping it as an "alternative implementation" comment.
- Retained full INFINITY scratch-init in `dp_grid_solve_banded` — Pitfall 5 (selective-init) deferred per plan scope.
- Fixed Plan 01's clippy doc-indentation lint in equivalence_phase114.rs (Rule 3 auto-fix: blocking clippy clean).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed clippy doc-overindented-list-items in equivalence_phase114.rs**
- **Found during:** Task 1 (first clippy run)
- **Issue:** `//!       (band_frac=0.1)` continuation line had 6-space indent; clippy `--all-targets` reported `doc_overindented_list_items` with `-D warnings`
- **Fix:** Changed to 4-space indent (`//!     (band_frac=0.1)`)
- **Files modified:** fdars-core/tests/equivalence_phase114.rs
- **Verification:** `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` — clean
- **Committed in:** 30d2c9a2 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking clippy lint from Plan 01 test file)
**Impact on plan:** Pre-existing lint in Plan 01 test file, unblocked by `--all-targets` check. No scope creep.

## Threat Mitigations

- **T-114-03** (float-reassociation): mitigated — same cells, same order; `dp_relax_cell`/`dp_edge_weight_core`/`COPRIME_NBHD_7` untouched; proven by golden fixture in debug + release + equivalence_phase47/48/49/50.
- **T-114-04** (uninitialized scratch): mitigated — full `e.resize(size, INFINITY)` retained; INFINITY guard in `dp_relax_cell` still valid for out-of-band predecessor reads.
- **T-114-05** (new crate dep / API break): n/a — no new dependency, no public signature changed.

## Issues Encountered

None beyond the auto-fixed clippy lint.

## Next Phase Readiness

- OPT-01 refactor complete and proven numeric-identical. The AFTER criterion (within noise at n500_m50) is committed for the record.
- Plan 114-03 can consume the committed after-criterion + COMPARISON.md for the final report.
- Future optimization opportunity: selective scratch-init (Pitfall 5 deferred) or a larger-m bench cell could reveal the full speedup on production-scale inputs.

---
*Phase: 114-elastic-alignment-optimization*
*Completed: 2026-09-23*
