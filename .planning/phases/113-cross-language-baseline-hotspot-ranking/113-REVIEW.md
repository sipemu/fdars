---
phase: 113-cross-language-baseline-hotspot-ranking
reviewed: 2026-09-23T12:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - benchmarks/rank_hotspots.py
  - benchmarks/gate.py
  - benchmarks/run_all.py
  - benchmarks/assemble_comparison.py
  - benchmarks/collect_rust.py
  - benchmarks/python_runner.py
  - benchmarks/r_runner.R
  - benchmarks/test_rank_hotspots.py
  - benchmarks/test_gate.py
  - benchmarks/test_assemble_comparison.py
findings:
  critical: 0
  warning: 0
  info: 1
  total: 1
status: clean
---

# Phase 113: Code Review Report (Re-review after fix pass)

**Reviewed:** 2026-09-23
**Depth:** standard
**Files Reviewed:** 10
**Status:** clean

## Summary

All four fixes from the prior review have been applied correctly with no regressions introduced.
The data flow from criterion output through noise gating to hotspot ranking is now honest:
`rsd_pct` is computed from the true sample standard deviation (`std_dev.point_estimate`), NaN
values cannot silently pass the noise gate or contaminate the competitor denominator, `assemble()`
carries no dead parameter, and `write_timing_comparison` matches Rust records by op and n/m.

**Fix verification:**

- **CR-01 (`std_dev_ns` from true SD):** `collect_rust.py` line 133 reads
  `std_dev_section.get("point_estimate", 0.0)` into `std_dev_ns`, where `std_dev_section =
  data.get("std_dev", {})`. On disk criterion 0.5 always emits a `std_dev` section with a
  `point_estimate` field (verified on live criterion output). `assemble_comparison._build_rust_rows`
  line 99 fetches `rec.get("std_dev_ns", rec.get("std_err_ns", 0.0))`; because `std_dev_ns` is
  always present in every record emitted by `collect_rust.py`, the `std_err_ns` fallback is dead
  code in production. The `_noise_gate_rust` docstring (lines 64–68) correctly documents the
  invariant. `rsd_pct = std_dev_ns / mean_ns * 100` now reflects true RSD. **VERIFIED.**

- **WR-02 (NaN guards):** `apply_noise_gate` now guards `math.isnan(rsd_f) or math.isnan(ci_f)`
  at line 105 after the `float()` cast, so float-NaN rsd/ci-width rows from
  `_noise_gate_rust(mean_ns=0)` are excluded rather than passing. `compute_leverage` guards both
  inputs at lines 138 and 140. The competitor accumulation loop in `rank_rows` guards `math.isnan(mns)`
  at line 188. All three NaN paths are closed. **VERIFIED.**

- **WR-01 (`assemble()` dead parameter):** `assemble()` at line 305 has no `gate_passed`
  parameter. The BENCH-05 invariant is enforced solely in `write_comparison_csv` (raises
  `ValueError` when `gate_passed=False`). The CLI validates `--gate-passed` before calling
  `write_comparison_csv`. **VERIFIED.**

- **WR-03 (`write_timing_comparison` matches by op AND n/m):** `_match_rust_rec` (line 384)
  first filters for `r.get("op") == op and r.get("n") == n and r.get("m") == m`; the op-only
  fallback applies only when no exact n/m match exists. Per-cell Python/R timings are not
  cross-contaminated with Rust records from different grid cells. **VERIFIED.**

**Noise gate behaviour:** With true-SD-based RSD, functional_boxplot criterion cells register
RSD ~23–33%, correctly triggering `FAIL-NOISY`. The gate now honestly drives target selection.
Elastic alignment at rank #1 is confirmed by noise-stable `elastic_align_pair` and banded
cross-alignment cells (PASS, RSD < 5%). The Karcher-mean cells (RSD 29–46%) remain correctly
noise-excluded.

**No regressions found.** Gate logic, FPCA subspace check, FPCR disclosure policy, r_native_pca_fd
exclusion, deterministic sort, and grid orchestration all function correctly post-fix.

---

## Narrative Findings (AI reviewer)

No Critical or Warning findings remain.

---

## Info

### IN-01: `import re` placed inside for-loop body

**File:** `benchmarks/assemble_comparison.py:110`
**Issue:** `import re` is inside `_build_rust_rows`, within the `for rec in rust_records:` loop.
Python caches modules after the first import (via `sys.modules`), so there is no correctness
impact. It is a quality smell: module-level imports should appear at the top of the file for
readability and to make the dependency explicit at a glance. This was present before the fix
pass and is not a regression.
**Fix:** Move `import re` to the top of `assemble_comparison.py` alongside the other stdlib
imports (`argparse`, `json`, `pathlib`, `sys`).

---

*Note: IN-02 (R runner stores `bench::mark()$median` as `mean_ns`) was carried forward from
the prior review as a pre-existing design choice — it remains out of scope for this phase per
user decision and is not re-flagged.*

---

_Reviewed: 2026-09-23_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
