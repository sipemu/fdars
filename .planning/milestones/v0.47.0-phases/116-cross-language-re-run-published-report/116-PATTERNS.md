# Phase 116: Cross-Language Re-Run & Published Report - Pattern Map

**Mapped:** 2026-09-24
**Files analyzed:** 7
**Analogs found:** 6 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|--------------------|------|-----------|-----------------|----------------|
| `benchmarks/render_report.py` | utility (report generator) | batch/transform (CSV → markdown) | `benchmarks/rank_hotspots.py` | exact |
| `benchmarks/test_render_report.py` | test | transform (unit tests, pure functions) | `benchmarks/test_rank_hotspots.py` | exact |
| `benchmarks/results/comparison_baseline.csv` | config/data snapshot | file-I/O (committed CSV) | `benchmarks/results/comparison.csv` (currently gitignored, must become tracked) | role-match (needs `.gitignore` carve-out) |
| `documentation/PERFORMANCE.md` | doc | N/A | `documentation/TESTING.md` / `documentation/ARCHITECTURE.md` | role-match |
| `README.md` "## Performance" section | doc (edit) | N/A | existing `## Performance` section (README.md line ~153) + `### Guides` list (line ~163) | exact (in-place edit) |
| `fdars-core/benches/depth_benchmarks.rs` (`functional_boxplot` group) | config (criterion tuning) | batch | `fdars-core/benches/audit_hotpaths.rs` (`audit_p3_elastic_cross`, already has explicit `sample_size`/`measurement_time` per cell) | exact |
| `benchmarks/README.md` | doc (edit) | N/A | itself (existing file, extend "Running"/"Result Schema" sections) | exact |

## Pattern Assignments

### `benchmarks/render_report.py` (utility, batch/transform)

**Analog:** `benchmarks/rank_hotspots.py` (full file read, 558 lines)

**Module docstring + constants pattern** (lines 1-47):
```python
"""
rank_hotspots.py — Noise-gated leverage ranking for fdars hotspot identification.

Reads benchmarks/results/comparison.csv (produced by Plan 113-02) and emits:
  - benchmarks/results/hotspot_ranking.md  (RESEARCH Section-5 schema)
...
"""
from __future__ import annotations
import json, math, os, sys
from datetime import date
from typing import Any
import pandas as pd

_RSD_THRESHOLD = 5.0
_CI_THRESHOLD = 5.0
_EXCLUDED_LIBRARIES = {"r_native_pca_fd"}
```
`render_report.py` should mirror this: module docstring stating inputs (`comparison.csv` + `comparison_baseline.csv` + `env.json`) and output (`documentation/PERFORMANCE.md` tables + README headline table), plus module-level constants for the noise gate thresholds (reuse `_RSD_THRESHOLD = 5.0`, `_CI_THRESHOLD = 5.0` — import or duplicate exactly, do not drift) and a `_EXCLUDED_LIBRARIES` style constant for `r_native_pca_fd` (ecosystem-realistic, never headlined per CONTEXT decisions).

**Pure-function core, testable in isolation** (lines 55-114, `apply_noise_gate`):
```python
def apply_noise_gate(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    passed, excluded = [], []
    for row in rows:
        ng = row.get("noise_gate")
        if ng == "NA":
            passed.append(row); continue
        if ng is None or (isinstance(ng, float) and math.isnan(ng)):
            excluded.append(row); continue
        ...
    return passed, excluded
```
Copy this shape for `render_report.py`'s new "significant?" column: a pure function `is_significant(ci_lower_a, ci_upper_a, ci_lower_b, ci_upper_b) -> bool` (non-overlapping CI test) and a pure `apply_noise_gate`-style filter reused from this module (import `rank_hotspots.apply_noise_gate` directly rather than re-implementing — avoids drift between the two reports).

**NaN-safe dict/DataFrame conversion** (lines 230-259, `_nan_key`, `_nan_sort_key`, `_df_to_rows`):
```python
def _df_to_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to a list of row dicts."""
    return df.where(pd.notnull(df), None).to_dict(orient="records")
```
Reuse verbatim for loading `comparison.csv` and `comparison_baseline.csv` — this is the established idiom for turning the CSV schema into row dicts fdars scripts operate on.

**Markdown emission pattern** (lines 279-510, `emit_hotspot_ranking`):
```python
def emit_hotspot_ranking(ranked, excluded, env_path, output_path, today=None) -> None:
    if today is None:
        today = str(date.today())
    lines: list[str] = []
    lines.append("# fdars Hotspot Ranking — Phase 113 Baseline")
    lines.append("")
    lines.append(f"**Date:** {today}")
    ...
    lines.append("| Op | N | M | fdars (ms) | Leverage Score | Rank | Source |")
    lines.append("|----|---|---|-----------|----------------|------|--------|")
    for i, r in enumerate(all_elastic_pass[:10]):
        ...
        lines.append(f"| elastic | {n_str} | {m_str} | {fdars_ms} | {lev_str} | {rank_str} | {src} |")
    ...
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
```
`render_report.py` should follow this exact "build a `lines: list[str]`, append markdown, join+write" idiom for both `documentation/PERFORMANCE.md` (full tables: before/after, scaling exponents, methodology, reproduction) and a smaller README headline-table fragment (either emitted as a separate function `emit_readme_snippet()` writing to a temp file for manual splice, or printed to stdout — CONTEXT says README prose is hand-written, so render_report.py only needs to emit the generated *table*, not rewrite the whole README).

**Formatting helpers to copy** (lines 267-276):
```python
def _ms(ns: float) -> str:
    """Format nanoseconds as milliseconds with 3 decimal places."""
    return f"{ns / 1e6:.3f}"

def _format_source_label(source: str) -> str:
    if source and "criterion:" in source:
        return source.replace("criterion:", "")
    return source or "—"
```
Reuse both directly (or import) for consistent ms formatting and source labels across the new report tables.

**`__main__` entry-point pattern** (lines 518-557):
```python
def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "results", "comparison.csv")
    ...
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run Plan 113-02 first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    ...
if __name__ == "__main__":
    main()
```
Copy this "resolve `script_dir`, check inputs exist with a clear stderr message + `sys.exit(1)`, read via pandas, process, print a short console summary, write output" shape. `render_report.py`'s `main()` needs both CSVs (`comparison.csv` AND `comparison_baseline.csv`) plus `env.json` — fail closed with the same clear-message style if either baseline or after-run CSV is missing.

---

### `benchmarks/test_render_report.py` (test, transform)

**Analog:** `benchmarks/test_rank_hotspots.py` (full file read, 297 lines)

**Row-builder test fixtures** (lines 19-65):
```python
def _make_row(op="fpca", library="rust", n=100.0, m=50.0, mean_ns=1_000_000.0,
              rsd_pct=1.0, ci_width_pct=1.0, noise_gate="PASS",
              source="criterion:test", notes=""):
    """Return a dict with the comparison.csv schema."""
    return dict(op=op, library=library, n=n, m=m, mean_ns=mean_ns,
                ci_lower_ns=None, ci_upper_ns=None, std_dev_ns=None,
                rsd_pct=rsd_pct, ci_width_pct=ci_width_pct,
                noise_gate=noise_gate, source=source, notes=notes)

def _make_cross_row(op, library, n, m, mean_ns, notes=""):
    """Return a cross-language (python/r) row with noise_gate='NA'."""
    ...
```
Copy this pattern for `test_render_report.py`: build `_make_row`/`_make_cross_row`-style fixtures matching the `comparison.csv` schema (`op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,rsd_pct,ci_width_pct,noise_gate,source,notes`), plus a `_make_baseline_row` variant for the before/after diff tests (paired rows from `comparison_baseline.csv` and `comparison.csv`).

**`unittest.TestCase` grouping by concern** (lines 75-296):
```python
class TestNoiseGate(unittest.TestCase):
    def test_noise_gate_flags_high_rsd(self):
        ...
class TestLeverageScore(unittest.TestCase):
    def test_leverage_slower_than_competitor(self):
        ...
class TestRanking(unittest.TestCase):
    def test_noisy_op_excluded_from_ranked_targets(self):
        ...
if __name__ == "__main__":
    unittest.main()
```
Mirror this: `TestSignificance` (non-overlapping-CI logic — boundary cases: touching CIs, fully overlapping, fully disjoint), `TestScalingExponent` (log-log slope fit against a known synthetic power-law grid, e.g. verify recovered exponent ≈2 for an O(N²) input), `TestBeforeAfterDiff` (row pairing by op/n/m between baseline and after CSVs; missing-baseline-row handling), and `TestReportEmission` (smoke-test that `emit_*` writes non-empty markdown with expected section headers) — same `unittest.TestCase` + plain-assert style as `test_rank_hotspots.py`, no pytest fixtures/parametrize despite the CONTEXT calling it "pytest unit tests" (existing suite runs as `unittest.main()`; keep consistent — pytest can still collect and run `unittest.TestCase` classes, satisfying the "pytest" framing without diverging style).

**Late-import pattern** (line 72):
```python
import rank_hotspots  # noqa: E402  (module in same directory)
```
Use `import render_report  # noqa: E402` identically (same-directory module import after `sys`/`unittest` stdlib imports).

---

### `benchmarks/results/comparison_baseline.csv` (data snapshot, file-I/O)

**Analog:** `benchmarks/results/comparison.csv` (currently produced by `assemble_comparison.py`, schema per `benchmarks/README.md` lines 83-101 and CONTEXT line 51)

Schema (must match exactly, copied verbatim so `render_report.py` can diff row-for-row):
```
op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,rsd_pct,ci_width_pct,noise_gate,source,notes
```
**Gitignore gap:** `benchmarks/results/*` is currently gitignored wholesale (confirmed: `git ls-files` shows zero tracked files under `benchmarks/results/`, despite `comparison.csv`/`env.json`/`hotspot_ranking.md` existing on disk from Phase 113 — they are NOT committed). CONTEXT explicitly requires `comparison_baseline.csv` to be **committed**. The plan must add a `.gitignore` carve-out (pattern like `!benchmarks/results/comparison_baseline.csv` alongside the existing `benchmarks/results/` ignore rule) — otherwise `git add` will silently no-op or need `-f`. Same treatment likely needed for `comparison.csv` itself if the "after" numbers are also meant to be committed evidence (CONTEXT line 23 implies `comparison.csv` gets overwritten by the re-run and is the "after" data referenced by the report — check whether it too needs a gitignore carve-out; the report's traceability claim in `<specifics>` ("every number traceable to a committed CSV") implies **both** CSVs must end up tracked).

---

### `documentation/PERFORMANCE.md` (doc)

**Analog:** `documentation/TESTING.md` and `documentation/ARCHITECTURE.md` — both tracked guides in the same directory, referenced from README's "Guides" list.

**README "Guides" list-entry pattern to extend** (README.md lines 160-166, current — see exact excerpt below): each guide is one bullet, `[Title](documentation/NAME.md) — one-line description`. Add:
```markdown
- [Performance](documentation/PERFORMANCE.md) — before/after benchmarks vs scikit-fda/R, scaling analysis, reproduction steps
```
**Doc structure convention (project CLAUDE.md "Comments" section):** module-level doc always states purpose + examples upfront; deviations from other-ecosystem baselines must be explicitly noted ("note when implementation differs from R/Python baselines") — this directly matches the CONTEXT requirement to disclose the `fda::pca.fd` vs Simpson-weighted-SVD footnote and non-win framing for elastic alignment. Structure `PERFORMANCE.md` with explicit headed sections mirroring `documentation/TESTING.md`'s convention of "layout → run commands → interpreting output" progression: Methodology → Before/After Tables → Rust-Internal Table (`face_covariance`) → Scaling Analysis → Reproduction Steps → Known Limitations/Noise-Gate Disclosures.

---

### `README.md` "## Performance" section (doc, edit)

**Analog:** the section itself (README.md, lines ~153-158, read in full above):
```markdown
## Performance

With the `parallel` feature (enabled by default), computationally intensive operations use `rayon` for multi-core performance. The library also supports WASM targets with sequential execution.
```
Replace with a short headline table (generated by `render_report.py`, per CONTEXT) + 2-3 hand-written honest takeaway lines + link to `documentation/PERFORMANCE.md`, keeping the existing sentence about `parallel`/rayon/WASM as trailing qualitative context (CONTEXT: "the `parallel` feature is mentioned qualitatively in prose, no multi-thread tables").

**README Guides list edit point** (lines 163-166, exact excerpt above) — add the PERFORMANCE.md bullet following the existing `- [Title](documentation/NAME.md) — description` pattern used for Getting-Started/Architecture/Configuration/Development/Testing.

---

### `fdars-core/benches/depth_benchmarks.rs` `functional_boxplot` group (config, criterion tuning)

**Analog:** `fdars-core/benches/audit_hotpaths.rs` `bench_p3_elastic_cross` (lines 600-615), which already documents per-cell noise characteristics and sets `sample_size`/`measurement_time`/`warm_up_time` explicitly with an inline comment table:
```rust
/// Timing by cell (O(n²·m²) DP, all pairs):
///   - n100_m50:  ~2× self → sample_size(20), measurement_time(30s)
///   - n100_m200: very slow → sample_size(10), measurement_time(60s)
///   - n500_m50:  very slow → sample_size(10), measurement_time(60s)
///   - n500_m200: extreme  → sample_size(10), measurement_time(60s)
fn bench_p3_elastic_cross(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_p3_elastic_cross");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(30));
    group.warm_up_time(std::time::Duration::from_secs(5));
    ...
```
Current `functional_boxplot` group (`fdars-core/benches/depth_benchmarks.rs` lines 348-357) has NO explicit `sample_size`/`measurement_time` override (uses criterion defaults — likely contributing to the baseline's RSD ~28% flagged in CONTEXT):
```rust
fn bench_functional_boxplot(c: &mut Criterion) {
    let mut group = c.benchmark_group("functional_boxplot");
    let t = 200;
    for &n in &[50, 200, 500, 1000, 2300] {
        let data = generate_centered_data(n, t);
        group.bench_with_input(BenchmarkId::new("N", n), &data, |b, data| {
            b.iter(|| functional_boxplot(black_box(data), DepthMethod::ModifiedBand, 1.5))
        });
    }
    group.finish();
}
```
Copy the `audit_hotpaths.rs` idiom exactly: add `group.sample_size(N)` + `group.measurement_time(Duration::from_secs(S))` (values at Claude's discretion per CONTEXT) right after `benchmark_group(...)`, with an inline doc comment stating the RSD-driven rationale, consistent with the project convention of documenting *why* a tuning value was chosen. Other repo-wide precedents for the same knobs: `fdars-core/benches/frechet_benchmarks.rs:51-52` (`sample_size(20)` + `measurement_time(30s)`), `fdars-core/benches/coclustering_benchmarks.rs:41-42` (`sample_size(10)` + `measurement_time(60s)` for a slow group). Any other "noisy headline cell" group needing a bump (e.g. within `audit_p3_elastic_cross`/`audit_p3_elastic_cross_banded` if still noisy after Phase 114/115, or `mface_covariance` in `fdars-core/benches/face_benchmarks.rs`) should follow this same pattern — locate via `group.sample_size`/`group.measurement_time` grep shown above for existing precedent values before picking new ones.

---

### `benchmarks/README.md` (doc, edit)

**Analog:** itself — extend in place. Current "Running" section (lines 59-73) documents the `--op fpca` single-op invocation; the full/grid invocation is `run_all.py --grid` per its own module docstring (`benchmarks/run_all.py` lines 24-42):
```
python run_all.py --grid                       # full N×M grid baseline run
python run_all.py --grid --dry-run             # grid wiring check, no cargo bench
```
Add a "Full re-run" subsection to `benchmarks/README.md`'s "Running" block using the existing fenced-bash-block style (see lines 61-73 above) documenting: `benchmarks/.venv/bin/python benchmarks/run_all.py --grid` (full 6-cell grid, all three ecosystems, gate-before-timing per cell), followed by `benchmarks/.venv/bin/python benchmarks/render_report.py` to regenerate `documentation/PERFORMANCE.md` tables. Also extend the "Result Schema" section (lines 83-101) if `comparison_baseline.csv`'s schema needs documenting (it is identical to `comparison.csv`, so a one-line note suffices: "`comparison_baseline.csv` uses the identical schema, snapshotted from the Phase 113 baseline before re-run").

## Shared Patterns

### Thread-pinning fairness contract
**Source:** `benchmarks/run_all.py` lines 51-63 (`THREAD_ENV` dict) and `benchmarks/README.md` lines 75-81
```python
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}
```
**Apply to:** `documentation/PERFORMANCE.md` methodology section must restate this exact env-var list (matches CONTEXT's "thread pinning" methodology-disclosure requirement) and `render_report.py` should read the six keys back out of `env.json` (written by `benchmarks/capture_env.py`) rather than hardcoding, so the report can never drift from what the run actually used.

### Noise gate (RSD/CI-width) definition
**Source:** `benchmarks/rank_hotspots.py` lines 45-46, 55-114 (`apply_noise_gate`, `_RSD_THRESHOLD = 5.0`, `_CI_THRESHOLD = 5.0`)
**Apply to:** `render_report.py`'s "significant?" column and its own noise-gate flagging (CONTEXT: "Cells failing the noise gate ... are flagged and never headlined") — import/reuse `rank_hotspots.apply_noise_gate` and the same thresholds directly instead of reimplementing, to guarantee the two reports (`hotspot_ranking.md` and the new performance report) never disagree on what counts as noisy.

### Markdown-table-from-CSV generation
**Source:** `benchmarks/rank_hotspots.py` lines 279-510 (`emit_hotspot_ranking`) — build `lines: list[str]`, append header/table-row strings, `"\n".join(lines) + "\n"`, write with `open(path, "w", encoding="utf-8")`.
**Apply to:** every markdown-emitting function in `render_report.py` (headline table, before/after tables, scaling table, methodology section).

### Script entry-point / fail-closed-on-missing-input
**Source:** `benchmarks/rank_hotspots.py` lines 518-557 (`main()`), same idiom in `benchmarks/run_all.py` (e.g. lines 725-731: explicit existence check + `sys.exit(1)` with an actionable stderr message).
**Apply to:** `render_report.py main()` — check both `comparison.csv` and `comparison_baseline.csv` (and `env.json`) exist before doing any work; clear stderr message naming which upstream script/phase to re-run.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Scaling-exponent (log-log slope) fitting logic inside `render_report.py` | utility (statistical transform) | transform | No existing script in `benchmarks/` computes regression/slope fits — closest is `gate.py`'s Pearson-correlation check (`benchmarks/gate.py`, not read in full here but referenced in CONTEXT/README as `check_fpca_agreement`); a small in-house `numpy.polyfit(log(x), log(y), 1)` is idiomatic and needs no new dependency (numpy already in `requirements.lock` per `gate.py` usage in `run_all.py` line 293 `import numpy as np`). Implement as a small pure function analogous in style to `compute_leverage` (lines 117-144 above) — deterministic, guarded against non-positive/NaN inputs.

## Metadata

**Analog search scope:** `benchmarks/` (all `.py`/`.md`), `fdars-core/benches/*.rs`, `documentation/*.md`, `README.md`, `.planning/phases/113-*/`, `.gitignore`
**Files scanned:** `rank_hotspots.py`, `test_rank_hotspots.py`, `run_all.py`, `benchmarks/README.md`, `README.md` (Performance + Guides sections), `depth_benchmarks.rs`, `audit_hotpaths.rs`, `face_benchmarks.rs`, `frechet_benchmarks.rs`, `coclustering_benchmarks.rs`, `.gitignore`, `113-01-SUMMARY.md`, `113-02-SUMMARY.md`, `113-03-SUMMARY.md`
**Pattern extraction date:** 2026-09-24

### Full-grid re-run invocation & Phase 113 baseline duration (for reproduction-instructions section)

- Full grid invocation: `benchmarks/.venv/bin/python benchmarks/run_all.py --grid` (6 cells, N∈{100,300,500}×M∈{50,200}, strictly sequential, gate-before-timing per cell per `run_all.py`'s `run_grid()`, lines 560-653 read above). Dry-run wiring check: `... run_all.py --grid --dry-run`.
- `--env-only` flag re-captures `env.json` alone: `... run_all.py --env-only` (needed post-re-run per CONTEXT: "`env.json` refreshed for the after-run").
- Cargo bench for the Rust side is run separately from `--grid` (grid mode reuses whatever criterion output is already on disk — see `run_all.py` lines 26-27, 578-579): `cargo bench -p fdars-core --features linalg,parallel --bench <bench_name>` for each of `BENCH_FILES = ["regression_benchmarks", "depth_benchmarks", "alignment_benchmarks", "basis_benchmarks"]` (`run_all.py` lines 81-86, 249-260) — plus any additional bench files touched by Phase 114/115 optimizations (`perf_hotpaths`, `face_benchmarks`, `audit_hotpaths` per the grep above) must also be re-run for the "after" numbers to reflect the optimized code paths.
- Phase 113 baseline duration (from SUMMARY frontmatter `duration:` fields): 113-01 = 7 min, 113-03 (ranking write-up) = 3 min; 113-02 (the actual grid run + gate) duration was not recorded in its SUMMARY frontmatter (not found via grep) — flag this gap explicitly in `documentation/PERFORMANCE.md`'s "expected runtime" reproduction note rather than fabricating a number; the plan should instrument/time the Phase 116 re-run itself and report that measured wall-clock instead.
