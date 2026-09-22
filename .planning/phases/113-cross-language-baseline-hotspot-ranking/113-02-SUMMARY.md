---
phase: 113-cross-language-baseline-hotspot-ranking
plan: "02"
subsystem: benchmarks
tags: [benchmark-harness, grid-baseline, comparison-csv, fpcr-disclosure, fpca-degeneracy, pearson-correlation]
status: halted

dependency_graph:
  requires:
    - phase: 113-01
      provides: recalibrated gate.py (FPCR threshold 0.95), fpca_native_r in r_runner.R, criterion output for fregre_lm + functional_boxplot
  provides:
    - benchmarks/run_all.py (--grid mode: N×M loop, per-cell gate-before-timing, per-cell result files)
    - benchmarks/gate.py (FPCR moved to disclosed/non-gating; five ops still hard-gated)
    - benchmarks/assemble_comparison.py (merge per-cell JSON to comparison.csv; FPCR disclosure note in rows)
    - benchmarks/test_assemble_comparison.py (12 unit tests, all green)
    - benchmarks/test_gate.py (+4 FPCR-disclosure non-gating tests, all green)
  affects: [113-03]

actuals:
  tokens: 40000
  tasks: 2
  commits: 3
  plan_head_before: 2016baab814c69bba43355df4135dfbce6cd3c19

tech-stack:
  added: []
  patterns:
    - Per-cell result file pattern: python_n{n}_m{m}.json + r_n{n}_m{m}.json written after gate passes
    - Gate-guarded CSV write: write_comparison_csv(gate_passed=True) raises on gate_passed=False (BENCH-05)
    - Compute-but-not-block disclosure: FPCR correlation computed + recorded, never aborts (five ops still hard-gated)
    - TDD (RED→GREEN) for assembler + gate disclosure tests

key-files:
  created:
    - benchmarks/assemble_comparison.py
    - benchmarks/test_assemble_comparison.py
  modified:
    - benchmarks/run_all.py
    - benchmarks/gate.py
    - benchmarks/test_gate.py

key-decisions:
  - "--grid mode runs cells strictly sequentially (Pitfall 2 — shared CSV filenames) with gate-before-timing per cell"
  - "fpca_native_r invoked explicitly per cell (not in ALL_OPS) — keeps smoke/default contract unchanged (Plan 01 decision)"
  - "assemble_comparison.py write_comparison_csv() raises ValueError if gate_passed=False (BENCH-05 invariant enforced at write time)"
  - "FPCR made compute-but-not-block (user-approved 2026-09-22): removed from _GATED_OPS, added to _DISCLOSED_OPS; correlation recorded in comparison.csv notes; five ops (fpca,basis_smoothing,depth,functional_boxplot,elastic) STILL hard-gated"
  - "NEW BLOCKER at cell 3 (n=300,m=50): FPCA hard-gate fdars-vs-python min|corr|=0.9874 < 0.99 — PC3/PC4 eigenvalue near-degeneracy (subspace agrees 0.999, individual axes 0.988). NOT the FPCR issue; a hard-gated op the coordinator said must stay hard-gated"

patterns-established:
  - "Grid cell gate-before-timing: run all ops (python + r + fpca_native_r) → collect rust → gate (FPCR disclosed) → persist per-cell files → assemble"
  - "Assembler schema: op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,rsd_pct,ci_width_pct,noise_gate,source,notes"

requirements-completed: []

coverage:
  - id: D1
    description: "--grid mode in run_all.py with per-cell gate-before-timing, sequential loop, fpca_native_r invocation"
    requirement: BASE-01
    verification:
      - kind: integration
        ref: "benchmarks/.venv/bin/python benchmarks/run_all.py --grid --dry-run → exits 0, iterates 6 cells"
        status: pass
      - kind: integration
        ref: "benchmarks/.venv/bin/python benchmarks/run_all.py --help | grep --grid → present"
        status: pass
    human_judgment: false
  - id: D2
    description: "assemble_comparison.py merges JSON to schema-correct comparison.csv with computed noise fields + FPCR disclosure note"
    requirement: BASE-01
    verification:
      - kind: unit
        ref: "benchmarks/test_assemble_comparison.py — 12 tests (rsd_pct, ci_width_pct, noise_gate labels, r_native_pca_fd notes, schema header, gate guard, FPCR disclosure note)"
        status: pass
    human_judgment: false
  - id: D3
    description: "FPCR made compute-but-not-block; five ops still hard-gated (gate.py + tests)"
    requirement: BASE-01
    verification:
      - kind: unit
        ref: "benchmarks/test_gate.py — 25 tests incl. 4 FPCR-disclosure non-gating tests (fpcr not in _GATED_OPS; below-threshold FPCR does not abort; hard-gated op still aborts)"
        status: pass
      - kind: integration
        ref: "grid run cells 1-2 passed with FPCR disclosed (corr 0.9495@n100m50, disclosed not blocking); cell 3 FPCR corr=0.8061 disclosed-below, NOT blocking"
        status: pass
    human_judgment: false
  - id: D4
    description: "Committed grid baseline comparison.csv (Task 3)"
    requirement: BASE-01
    verification: []
    human_judgment: true
    rationale: "Task 3 halted at cell 3: NEW blocker — FPCA HARD-gate fdars-vs-python min|corr|=0.9874 < 0.99 at n=300,m=50 (PC3/PC4 eigenvalue near-degeneracy). This is a hard-gated op (coordinator: must stay hard-gated); requires human decision. comparison.csv not written."

duration: ~2h (incl. gate recalibration + two grid runs)
completed: 2026-09-23
---

# Phase 113 Plan 02: Grid Baseline Run Summary

**--grid mode + assembler + FPCR compute-but-not-block disclosure implemented and committed (Tasks 1-2 + user-approved FPCR change); grid run cleared cells 1-2 (FPCR disclosed, not blocking) then halted at cell 3 (n=300,m=50) on a NEW FPCA HARD-gate failure — fdars-vs-python min|corr|=0.9874 < 0.99 from PC3/PC4 eigenvalue near-degeneracy. comparison.csv not written.**

## Performance

- **Duration:** ~2h across two grid runs + gate recalibration
- **Tasks:** 2 of 3 complete (Tasks 1-2 + FPCR disclosure); Task 3 halted at cell-3 gate
- **Files modified:** 5 (run_all.py, gate.py, test_gate.py, assemble_comparison.py [new], test_assemble_comparison.py [new])

## Accomplishments

- Added `--grid` mode to `run_all.py`: iterates N×{100,300,500} × M×{50,200} (6 cells) strictly sequentially, gate-before-timing per cell, writes `python_n{n}_m{m}.json` + `r_n{n}_m{m}.json` per cell for the assembler
- Added `fpca_native_r` invocation per grid cell (via explicit `--op fpca_native_r` in `_run_r_with_native_fpca`), separate from ALL_OPS loop
- Created `assemble_comparison.py`: merges per-cell JSON + rust.json into comparison.csv (pandas.to_csv, exact RESEARCH Section 5 schema); computes rsd_pct/ci_width_pct/noise_gate for Rust rows; FPCR rows carry cross-library disclosure note; gate_passed guard (BENCH-05)
- **FPCR compute-but-not-block (user-approved 2026-09-22):** removed fpcr from `_GATED_OPS`, added `_DISCLOSED_OPS={fpcr}`; `run_gate` computes the FPCR cross-library correlation and prints a `[DISCLOSURE]` line but NEVER aborts on it; the five other ops stay hard-gated. `compute_fpcr_disclosure_corr()` added. `_FPCR_CORR_THRESHOLD` is now a disclosure LABEL, not an abort threshold.
- Tests: `test_assemble_comparison.py` (12, incl. FPCR disclosure note); `test_gate.py` (25, incl. 4 FPCR non-gating tests). All green. Gate self-test green.
- **Grid run cleared cells 1-2** (n=100,m=50 and n=100,m=200): FPCR disclosed (corr=0.9495 @ n=100), NOT blocking — the user-approved change works exactly as intended
- **Task 3 halted at cell 3 (n=300,m=50):** a DIFFERENT, HARD-gated op failed — FPCA fdars-vs-python min|corr|=0.9874 < 0.99

## FPCR Disclosure — WORKING (user-approved change verified live)

| Cell | FPCR cross-library corr | Gate action |
|------|-------------------------|-------------|
| n=100, m=50 | 0.9495 | DISCLOSED-BELOW, NOT blocking → cell passed |
| n=100, m=200 | (disclosed) | NOT blocking → cell passed |
| n=300, m=50 | 0.8061 | DISCLOSED-BELOW, NOT blocking (cell aborted on FPCA instead) |

The `[DISCLOSURE] [r] fpcr: |corr|=X (disclosure bar 0.95; DISCLOSED-BELOW; NOT hard-gated)` line prints per cell; comparison.csv would carry the measured corr in the notes column. This is exactly the coordinator's requested behavior.

## NEW BLOCKER — FPCA hard-gate failure at cell 3 (n=300, m=50)

| Op | Cell | Check | min |corr| | Threshold | Result |
|----|------|-------|-------------|-----------|--------|
| fpca | n=300, m=50 | fdars-vs-python | 0.9874 | 0.99 | **FAIL** |
| fpca | n=300, m=50 | fdars-vs-r | 1.0000 | 0.99 | PASS |

**This is NOT the FPCR issue and NOT covered by the coordinator's FPCR decision.** FPCA is one of the five ops the coordinator explicitly said must stay hard-gated ("Do NOT lower any of the five hard-gated thresholds").

**Root cause — PC3/PC4 eigenvalue near-degeneracy (benign, but trips the per-component gate):**
- Per-component fdars-vs-python correlation at n=300,m=50: PC1=1.000, PC2=1.000, PC3=0.987, PC4=0.988, PC5=0.999
- The PC3+PC4 **subspace** agrees at principal-angle cosines 0.9998 / 0.9991 (i.e. the 2D plane is essentially identical)
- PC3 and PC4 have near-equal eigenvalues, so fdars and scikit-fda place the individual rotation axes slightly differently within that shared plane — the per-component Pearson gate (0.99) is stricter than the subspace reality
- Cells 1-2 (n=100) passed because the eigenvalue gap was larger there; at n=300 the PC3/PC4 gap narrows

**This is a well-known FPCA phenomenon (rotational ambiguity within near-degenerate eigenspaces), not a numerical bug in fdars.** But it is a genuine hard-gate failure and I am not authorized to relax the FPCA threshold (coordinator instruction + plan).

## Cells completed before halt

| Cell | Hard-gated ops | FPCR | Per-cell files written |
|------|----------------|------|------------------------|
| n=100, m=50 | PASS (fpca fdars-vs-py 0.9983) | disclosed 0.9495 | python_n100_m50.json, r_n100_m50.json |
| n=100, m=200 | PASS (fpca fdars-vs-py 0.9992) | disclosed | python_n100_m200.json, r_n100_m200.json |
| n=300, m=50 | **FPCA FAIL (0.9874)** | disclosed 0.8061 | NOT written (gate aborted) |

## Task Commits

1. **Task 1: --grid mode in run_all.py** — `9f3f7cb5`
2. **Task 2: assemble_comparison.py + TDD tests** — `f036bf8a`
3. **FPCR compute-but-not-block + disclosure (user-approved)** — `79146973`
4. **docs (this halt)** — pending
   - Task 3 grid run: HALTED at cell 3 (FPCA hard-gate); comparison.csv not written

## Files Created/Modified

- `benchmarks/run_all.py` — GRID_N/GRID_M, run_grid(), _run_r_with_native_fpca(), _write_cell_results(), --grid flag
- `benchmarks/gate.py` — FPCR removed from _GATED_OPS → _DISCLOSED_OPS; run_gate computes FPCR corr but never aborts; compute_fpcr_disclosure_corr()
- `benchmarks/test_gate.py` — +4 FPCR non-gating tests
- `benchmarks/assemble_comparison.py` — NEW: assembler + FPCR disclosure note
- `benchmarks/test_assemble_comparison.py` — NEW: 12 unit tests (TDD RED→GREEN)

## Decisions Made

- FPCR compute-but-not-block (user-approved): correlation recorded + disclosed, never aborts; five ops still hard-gated
- FPCA hard-gate failure at n=300 is a NEW blocker requiring a decision — NOT auto-resolved (would require relaxing a hard-gated threshold, which is prohibited)

## Phase 116 report finding

**FPCR cross-library divergence (corr ~0.95 at n=100, dropping to ~0.81 at n=300):** scikit-fda (FPCA→OLS) and R fda.usc (fregre.pc) implement FPCR with related but not identical procedures; the divergence grows with n. Disclosed, not gated. Recorded per-cell in comparison.csv notes when the baseline completes.

**FPCA rotational ambiguity in near-degenerate eigenspaces:** at n=300, PC3/PC4 have near-equal eigenvalues; the 2D subspace matches (cos 0.999) but individual axes differ (~0.988). This affects any per-component cross-library FPCA comparison at larger n.

## Deviations from Plan

### FPCR disclosure (coordinator-directed, user-approved) — implemented
Per the coordinator's user-approved decision, FPCR was moved from hard-gating to disclosed/non-gating. This is a directed change, not an autonomous deviation.

### FPCA hard-gate failure at n=300 (NEW blocker, halt)
A hard-gated op (FPCA) failed at cell 3 due to eigenvalue near-degeneracy. Per the coordinator instruction ("Do NOT lower any of the five hard-gated thresholds") and the plan, I did not relax the FPCA threshold. The grid aborted correctly (BENCH-05 preserved — no timing written). This requires a decision (see options below).

**Options for resolving the FPCA n=300 blocker:**
- **Option A:** Gate FPCA on the **subspace** agreement (principal angles) instead of per-component Pearson — robust to rotational ambiguity in near-degenerate eigenspaces. Cleanest fix; matches the numerical reality. Requires a gate.py change + tests (a hard-gate metric change, not a threshold relaxation).
- **Option B:** Reduce ncomp for the FPCA gate from 5 to 2 or 3 (PC1/PC2 always agree at 1.0) — gate only the well-separated leading components. Avoids the degeneracy entirely.
- **Option C:** Drop n=300 from the grid (use N∈{100,500}) — but n=500 will likely have the same or worse degeneracy.
- **Option D:** Accept fdars-vs-r (which passes at 1.0) as the FPCA gate and treat fdars-vs-python FPCA as disclosed (like FPCR) — but this weakens the FPCA hard gate.

Option A is recommended (correct metric for the phenomenon; keeps FPCA hard-gated on the honest subspace criterion).

## Known Stubs

- `benchmarks/results/comparison.csv` — NOT WRITTEN (Task 3 halted at cell-3 FPCA gate)
- `benchmarks/results/env.json` — stale (not re-captured from a gate-passing full grid run)
- Per-cell files exist only for cells 1-2 (n=100); cells 3-6 not run

## Threat Flags

None — no new network endpoints, auth paths, or schema changes. All changes are local script modifications handling trusted synthetic CSV data.

## Self-Check: PASSED (Tasks 1-2 + FPCR disclosure) / HALTED (Task 3) — VERIFIED

- benchmarks/run_all.py has --grid: EXISTS (--grid in --help, --grid --dry-run exits 0)
- benchmarks/gate.py FPCR disclosure: EXISTS (fpcr not in _GATED_OPS, in _DISCLOSED_OPS)
- benchmarks/assemble_comparison.py: EXISTS (importable, pandas to_csv, FPCR disclosure note)
- benchmarks/test_assemble_comparison.py: 12 tests pass
- benchmarks/test_gate.py: 25 tests pass (incl. 4 FPCR non-gating); gate self-test green
- Task 1 commit 9f3f7cb5: EXISTS
- Task 2 commit f036bf8a: EXISTS
- FPCR disclosure commit 79146973: EXISTS
- FPCR disclosure WORKING LIVE: cells 1-2 passed with FPCR disclosed (not blocking) — CONFIRMED
- comparison.csv: NOT WRITTEN (cell-3 FPCA hard-gate failure, correct BENCH-05 behavior)
- NEW blocker FPCA fdars-vs-python 0.9874 < 0.99 at n=300,m=50: CONFIRMED (PC3/PC4 degeneracy; subspace agrees 0.999)

---
*Phase: 113-cross-language-baseline-hotspot-ranking*
*Halted: 2026-09-23*
