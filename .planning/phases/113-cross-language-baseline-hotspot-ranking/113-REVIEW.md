---
phase: 113-cross-language-baseline-hotspot-ranking
reviewed: 2026-09-23T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - benchmarks/rank_hotspots.py
  - benchmarks/gate.py
  - benchmarks/run_all.py
  - benchmarks/assemble_comparison.py
  - benchmarks/collect_rust.py
  - benchmarks/python_runner.py
  - benchmarks/r_runner.R
findings:
  critical: 1
  warning: 3
  info: 2
  total: 6
status: issues_found
---

# Phase 113: Code Review Report

**Reviewed:** 2026-09-23
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Phase 113 delivers the cross-language baseline harness (Python/R/Rust runners, numeric agreement gate, assembler, and hotspot ranker). The architecture is sound: the gate-before-timing BENCH-05 invariant is correctly enforced, the QR-based subspace gate for FPCA is mathematically correct, NaN `noise_gate` rows are conservatively excluded, and the `r_native_pca_fd` exclusion from both targets and denominator is correctly wired.

One critical finding was identified: `rsd_pct` in the assembled CSV is computed from criterion's `standard_error` (SE of the mean estimate) rather than `std_dev`. SE = SD / sqrt(n_samples), so `rsd_pct` systematically underestimates true RSD by a factor of sqrt(criterion sample count). This can allow genuinely noisy benchmarks to pass the RSD half of the noise gate. The CI-width check (which IS computed from the correct CI bounds) provides a partial backstop, but the two conditions are independent and a low-sample-count benchmark can have narrow SE and narrow CI simultaneously even with high variability. This directly affects Phase 114-115 target selection.

Three warnings cover: (1) a misleading API in `assemble_comparison.assemble()` where `gate_passed` is silently ignored; (2) a NaN `mean_ns` on a cross-language row can silently be injected into the competitor denominator (bypasses the `<= 0` guard) and reduces leverage to absolute-time-only; (3) `write_timing_comparison` (non-grid mode) picks the first matching Rust record by op when multiple sizes exist on disk.

Two info items cover stale documentation: (1) `test_gate.py` module docstring still describes FPCA as "sign-aligned Pearson correlation" though the gate was changed to subspace principal-angle cosine; (2) the `mean_ns` field in R records actually stores the **median** (`bench::mark()$median`), creating a systematic labeling inconsistency with Python (mean via timeit) and Rust (mean.point_estimate via criterion).

No `.rs` source files or `Cargo.toml` were modified (confirmed by `git log` — phase is benchmarks-only). All Python files parse cleanly.

---

## Critical Issues

### CR-01: `rsd_pct` computed from criterion `standard_error`, not `std_dev` — noise gate systematically underestimates true RSD

**File:** `benchmarks/assemble_comparison.py:93` and `benchmarks/collect_rust.py:131`

**Issue:**
`collect_rust.py` maps `mean.standard_error` from `estimates.json` to the field `std_err_ns` (line 131). `assemble_comparison._build_rust_rows` then retrieves it via:

```python
std_dev = float(rec.get("std_dev_ns", rec.get("std_err_ns", 0.0)))
```

Because `rust.json` records carry `std_err_ns` and not `std_dev_ns`, the fallback always fires and `std_dev` is populated with the **standard error of the mean**, not the standard deviation. Criterion's `standard_error` is SE = SD / sqrt(n_samples). The resulting `rsd_pct = std_err_ns / mean_ns * 100` is therefore SD / (sqrt(n) × mean) × 100, which is smaller than the true RSD by a factor of sqrt(n_samples).

For a benchmark with 100 criterion samples and true RSD = 4.8%:
- True rsd_pct = 4.8% → should FAIL-NOISY (below 5% threshold only marginally)
- Computed rsd_pct ≈ 0.48% → PASSES the noise gate

The CI-width check provides a partial backstop, but CI bounds are also derived from the same SE-scaled distribution. A benchmark with few iterations and a tightly clustered run can exhibit narrow CI while having high per-run variability. The two checks are correlated rather than independent guards.

This directly affects hotspot ranking: a genuinely noisy benchmark could be admitted as PASS and ranked as a Phase 114-115 optimization target when it should be noise-excluded.

Criterion's `estimates.json` does not expose `std_dev` directly (it reports only SE under `mean.standard_error`). The correct fix is to compute `std_dev` from the CI bounds using the t-distribution or normal approximation: `std_dev ≈ (ci_upper - ci_lower) / (2 × z_0.975)` where `z_0.975 ≈ 1.96`. Alternatively, use `ci_width_pct` as the sole noise metric and drop `rsd_pct` as a gating criterion, documenting that it stores SE-relative dispersion (informational only).

**Fix:**
In `assemble_comparison.py`, rename the column and update the gate formula, or derive an approximate SD from the CI:

```python
# Option A: Use ci_width_pct as sole gate; demote rsd_pct to informational
# In _noise_gate_rust, gate only on ci_width_pct:
def _noise_gate_rust(mean_ns, ci_lower_ns, ci_upper_ns, std_err_ns):
    if mean_ns <= 0:
        return float("nan"), float("nan"), "FAIL-NOISY"
    # std_err_ns is SE, not SD — store as-is but do not gate on it
    rsd_pct = std_err_ns / mean_ns * 100.0  # NOTE: SE-relative, not SD-relative
    ci_width_pct = (ci_upper_ns - ci_lower_ns) / mean_ns * 100.0
    # Gate solely on CI-width (correct metric from criterion bootstrap CI)
    gate = "PASS" if ci_width_pct < _CI_WIDTH_THRESHOLD else "FAIL-NOISY"
    return rsd_pct, ci_width_pct, gate

# Option B: Rename field in collect_rust.py and schema to avoid confusion:
# Change std_dev_ns → std_err_ns in SCHEMA_COLUMNS and all references.
# Update rank_hotspots.py, test files, and RESEARCH docs accordingly.
```

Also rename `std_err_ns` to make the contract unambiguous in `collect_rust.py` line 131 and `assemble_comparison.py` line 93.

---

## Warnings

### WR-01: `assemble()` silently ignores `gate_passed` parameter — misleading BENCH-05 contract

**File:** `benchmarks/assemble_comparison.py:301-366` and line 420

**Issue:**
`assemble()` accepts a `gate_passed: bool = True` parameter and documents it as "must be True (caller asserts the gate passed before assembling — BENCH-05)". However, the function body never checks this parameter. The comment on line 420 (`# assemble always OK`) reveals this is intentional, but the signature and docstring mislead callers into believing the BENCH-05 invariant is enforced here. Any caller that passes `gate_passed=False` silently gets a fully assembled DataFrame without any error or warning, violating the stated contract.

The only real enforcement is in `write_comparison_csv`, which does check `gate_passed`. A caller who separately assembles and writes (two-step) and forgets to pass `gate_passed=True` to `write_comparison_csv` gets the correct error. But a caller who only calls `assemble(gate_passed=False)` gets no protection.

**Fix:**
Either enforce the parameter, or remove it from `assemble()` entirely to avoid false confidence:

```python
def assemble(results_dir=None) -> pd.DataFrame:
    """
    Merge per-cell JSON result files. Does NOT enforce the BENCH-05 gate
    invariant — pass the resulting DataFrame to write_comparison_csv(gate_passed=True)
    to enforce it at write time.
    """
    ...
```

Remove the `gate_passed` parameter from `assemble()`'s signature. Update the test `assembler.assemble(results_dir=tmp_path, gate_passed=True)` calls in `test_assemble_comparison.py` accordingly.

---

### WR-02: NaN `mean_ns` on a cross-language row silently bypasses the `min_other_ns <= 0` leverage guard

**File:** `benchmarks/rank_hotspots.py:170-177` and `compute_leverage:129`

**Issue:**
In `rank_rows`, cross-language rows with `noise_gate='NA'` pass the noise gate unconditionally. Their `mean_ns` is parsed via `float(row["mean_ns"])` (line 174-175). If a Python or R runner emits a record with `mean_ns = NaN` (e.g., a failed benchmark that wrote `null` to JSON, which pandas reads as NaN and `_df_to_rows` converts to `None`, which `float(None)` raises a `TypeError` — but a literal `NaN` in the JSON would survive as `float('nan')`), that NaN enters `competitor_map`.

When `min_other_ns = float('nan')`, the guard `if fdars_ns <= 0 or min_other_ns <= 0` evaluates to `False` (NaN comparisons are always False in Python). Execution proceeds to `ratio = fdars_ns / min_other_ns` = NaN, then `max(1.0, NaN)` = 1.0 (Python's `max()` returns the non-NaN argument when one operand is NaN). The result is `leverage_score = fdars_ns × 1.0`, which is the absolute-time fallback — the same as "no competitor exists".

The net effect is a **silent misclassification**: a cell where fdars is genuinely slower than a competitor appears as "no cross-language comparison available", potentially lowering its rank below cells that correctly reflect their leverage ratio. The row appears ranked as if Rust is already optimal (factor = 1.0) when in fact a competitor's data was corrupted.

This is not triggered by the current test suite (which uses well-formed inputs), but it can occur silently if a runner produces a JSON file with `NaN` mean timings.

**Fix:**
Add an explicit NaN guard in the competitor accumulation:

```python
# In rank_rows, where competitor_map is built (line 171-177):
for row in cross_rows:
    k = _cell_key(row)
    try:
        mns = float(row["mean_ns"])
    except (TypeError, ValueError):
        continue
    if math.isnan(mns) or mns <= 0:  # explicit NaN + non-positive guard
        continue
    competitor_map.setdefault(k, []).append(mns)
```

Also add `compute_leverage` guard for NaN:

```python
def compute_leverage(fdars_ns: float, min_other_ns: float) -> float:
    if math.isnan(fdars_ns) or fdars_ns <= 0:
        return fdars_ns
    if math.isnan(min_other_ns) or min_other_ns <= 0:
        return fdars_ns
    ...
```

---

### WR-03: `write_timing_comparison` takes first Rust record by op without size matching

**File:** `benchmarks/run_all.py:416-420`

**Issue:**
In the non-grid single-run path, `write_timing_comparison` selects the Rust timing record for each op with:

```python
rust_rec = next(
    (r for r in rust_records if r.get("op") == op),
    {}
)
```

`rust.json` contains all criterion records collected from disk (all sizes and configurations). When multiple records exist for the same op (e.g., `elastic` at n=100,m=50 and n=500,m=200), the first match is used regardless of whether it corresponds to the `n`/`m` passed to `--n`/`--m`. The resulting `timing_comparison.json` silently associates Python/R timings for one grid cell with Rust timings from a different cell.

This function is not used in grid mode (which is the primary comparison path), but `timing_comparison.json` is a named output artifact that could be referenced by downstream analysis or scripts.

**Fix:**
Filter by both op and matching n/m from the bench ID:

```python
def _match_rust_rec(rust_records, op, n, m):
    """Return the best-matching Rust record for (op, n, m), or {} if none."""
    # Prefer exact n/m match (parsed from bench field)
    exact = [r for r in rust_records
             if r.get("op") == op and r.get("n") == n and r.get("m") == m]
    if exact:
        return exact[0]
    # Fall back to op-only match (single-size runs)
    return next((r for r in rust_records if r.get("op") == op), {})
```

Note: `assemble_comparison.py` (the main comparison path) does not have this issue — it joins Rust records against per-cell Python/R files correctly via the bench ID parser.

---

## Info

### IN-01: `test_gate.py` module docstring describes FPCA gate as "sign-aligned Pearson correlation" — stale after Phase 113 Plan 02 change

**File:** `benchmarks/test_gate.py:6-9`

**Issue:**
The module-level docstring lists the FPCA tolerance as `"fpca  sign-aligned Pearson correlation > 0.99"`. The gate was changed in Phase 113 Plan 02 to SUBSPACE agreement via principal angles (cos of largest principal angle ≥ 0.99). The test implementations themselves are correct — `TestFpcaAgreement.test_degenerate_rotation_same_subspace_passes` specifically proves the subspace metric is working. Only the module-level table is stale.

**Fix:**
Update the docstring table at the top of `test_gate.py`:

```python
#   fpca            SUBSPACE agreement: cos(largest principal angle) >= 0.99  [HARD-GATED] [Phase 113 Plan 02]
```

---

### IN-02: R runner stores median as `mean_ns` — systematic labeling inconsistency with Python and Rust

**File:** `benchmarks/r_runner.R:83-87`

**Issue:**
The `time_fn` helper uses `bench::mark()$median` and stores the result as `mean_ns`:

```r
mean_ns <- as.numeric(timing$median) * 1e9  # convert seconds -> nanoseconds
```

Python runner uses `timeit.timeit` / `n_repeats` (arithmetic mean). Rust/criterion records carry `mean.point_estimate` (arithmetic mean). The cross-language comparison in `comparison.csv` therefore mixes median (R) and mean (Python, Rust) under the same `mean_ns` column heading. For symmetric distributions the difference is minor, but for right-skewed distributions (common in benchmarks due to OS jitter) the median is lower than the mean, making R appear faster than it is relative to Python and Rust.

This is a pre-existing design choice that predates Phase 113. It is not a code error but a methodological inconsistency that affects the cross-language comparison interpretation, particularly for elastic alignment and FPCA at large N where right-skew is more pronounced.

**Fix (low priority):** Rename the R output field to `median_ns` and update `assemble_comparison._build_r_rows` to read it, while surfacing it separately or applying a correction factor. Alternatively, switch to `bench::mark()$mean` for consistency. This is a Phase 116 report concern (disclosed in the findings section of the report).

---

_Reviewed: 2026-09-23_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
