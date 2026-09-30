---
phase: 113-cross-language-baseline-hotspot-ranking
fixed_at: 2026-09-23T00:00:00Z
review_path: .planning/phases/113-cross-language-baseline-hotspot-ranking/113-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 113: Code Review Fix Report

**Fixed at:** 2026-09-23
**Source review:** `.planning/phases/113-cross-language-baseline-hotspot-ranking/113-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 4 (CR-01, WR-01, WR-02, WR-03)
- Fixed: 4
- Skipped: 0

Note: IN-01 (stale test_gate.py docstring) was out of scope for `--fix` but was trivially fixed in the artifact-regeneration commit per the execution notes instruction. IN-02 (R median vs mean labeling) was explicitly descoped.

## Fixed Issues

### CR-01: `rsd_pct` computed from criterion `standard_error`, not `std_dev`

**Files modified:** `benchmarks/collect_rust.py`, `benchmarks/assemble_comparison.py`, `benchmarks/test_assemble_comparison.py`
**Commit:** `7dfd043c`
**Applied fix:**
`collect_rust.py` now reads `std_dev.point_estimate` from `estimates.json` and emits it as `std_dev_ns` in each record. The existing `std_err_ns` field (from `mean.standard_error`) is retained for reference but no longer used for the noise gate. `_build_rust_rows` in `assemble_comparison.py` already preferred `std_dev_ns` over `std_err_ns` via the fallback dict.get — with `collect_rust.py` now emitting `std_dev_ns`, the fallback to `std_err_ns` no longer fires. Updated `_noise_gate_rust` docstring to document the SE vs true-SD distinction explicitly. Updated module docstring to state `std_dev_ns = std_dev.point_estimate`. Removed `gate_passed` parameter from `assemble()` (WR-01, combined in same files) and updated 13 test call sites in `test_assemble_comparison.py`.

**Artifact impact (CR-01 fix + regeneration):**
The corrected true-RSD metrics changed the noise-gate classification for 3 cells:
- Before: 68 FAIL-NOISY, 69 PASS (SE ≈ SD/√n underestimated by ~10×)
- After: 71 FAIL-NOISY, 66 PASS
- Most significant: `functional_boxplot` group moved to FAIL-NOISY (true RSD 28–31%; SE-based RSD was ~3%). These were correctly passing the SE-based check despite genuine high variability.
- Elastic alignment remains confirmed rank #1 (anchor unchanged).
- Verification ran in the main checkout (no isolated worktree; `workflow.use_worktrees = false`).

---

### WR-01: `assemble()` silently ignores `gate_passed` parameter

**Files modified:** `benchmarks/assemble_comparison.py`, `benchmarks/test_assemble_comparison.py`
**Commit:** `7dfd043c` (combined with CR-01 — same files)
**Applied fix:**
Removed `gate_passed: bool = True` parameter from `assemble()` entirely. Updated docstring to state explicitly that `assemble()` does NOT enforce the BENCH-05 invariant and directs callers to `write_comparison_csv(gate_passed=True)` for enforcement. Updated 13 call sites in `test_assemble_comparison.py` to call `assembler.assemble(results_dir=tmp_path)` (no `gate_passed` kwarg). The internal `main()` call was updated from `assemble(results_dir=results_dir, gate_passed=True)` to `assemble(results_dir=results_dir)`.

---

### WR-02: NaN `mean_ns` on a cross-language row silently bypasses the leverage guard

**Files modified:** `benchmarks/rank_hotspots.py`
**Commit:** `09374314`
**Applied fix:**
Three explicit `math.isnan()` guards added:
1. `competitor_map` accumulation loop: `if math.isnan(mns) or mns <= 0: continue` — skips NaN competitor timings rather than injecting them into the minimum.
2. `compute_leverage`: explicit `math.isnan(fdars_ns)` and `math.isnan(min_other_ns)` checks before the `<= 0` guards — NaN comparisons in Python are always `False`, so the original `<= 0` guards silently passed NaN.
3. `apply_noise_gate`: explicit `math.isnan(rsd_f) or math.isnan(ci_f)` check before the threshold comparison — prevents a NaN `rsd_pct` from silently passing the gate (NaN < 5.0 is `False`, which would land in the `else` FAIL-NOISY branch, so the behavior was already correct, but the intent is now explicit and immune to future logic changes).

---

### WR-03: `write_timing_comparison` takes first Rust record by op without size matching

**Files modified:** `benchmarks/run_all.py`
**Commit:** `80f95628`
**Applied fix:**
Added `_match_rust_rec(rust_records, op, n, m)` helper. It first searches for records where `r.get("op") == op and r.get("n") == n and r.get("m") == m` (n/m are populated from the bench-field parser in `collect_rust.py`/`assemble_comparison.py`). If no exact n/m match is found, falls back to op-only match (preserving backward compatibility for single-size runs). `write_timing_comparison` now calls `_match_rust_rec` instead of the bare `next((r for r in rust_records if r.get("op") == op), {})`.

---

## Baseline Artifact Regeneration

All three artifact files were regenerated and committed at `597f857d`:
- `benchmarks/results/rust.json` — re-collected from criterion output with `std_dev_ns` populated
- `benchmarks/results/comparison.csv` — 215 rows; corrected `rsd_pct` from true SD; 71 FAIL-NOISY / 66 PASS
- `benchmarks/results/hotspot_ranking.md` — updated noise-gate labels; elastic alignment is still rank #1

The IN-01 stale docstring in `benchmarks/test_gate.py` was also fixed in this commit (changed "sign-aligned Pearson correlation > 0.99" to "SUBSPACE agreement: cos(largest principal angle) >= 0.99 [HARD-GATED] [Phase 113 Plan 02]").

**Test results (all 60 tests pass after fixes):**
```
60 passed, 2 warnings in 0.69s
```
Verification ran in the main checkout (`workflow.use_worktrees = false`).

---

_Fixed: 2026-09-23_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
