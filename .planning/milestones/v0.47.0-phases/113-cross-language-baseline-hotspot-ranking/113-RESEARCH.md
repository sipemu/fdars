# Phase 113: Cross-Language Baseline & Hotspot Ranking - Research

**Researched:** 2026-09-22
**Domain:** Cross-language FDA benchmark baseline (fdars vs scikit-fda vs R), criterion noise gating, hotspot leverage ranking
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Agreement-gate metric calibration (USER DECISION 2026-09-22):** For depth, functional_boxplot, and FPCR, switch the gate to a sign/scale-invariant Pearson (or rank) correlation ≥ 0.99, the same family FPCA already uses. FPCA (corr > 0.99), basis smoothing (abs < 1e-3), and elastic warping (L-inf < 1e-2) are UNCHANGED. The gate remains fail-closed and runs before any timing is recorded (BENCH-05 invariant preserved). Update `benchmarks/gate.py` per-op dispatch + its unit tests accordingly.
- **R FPCA fairness — time both, disclose (USER DECISION 2026-09-22):** Time both (a) the Simpson's-weighted-SVD version that matches fdars numerically (feeds the agreement gate, apples-to-apples) and (b) native `fda::pca.fd` separately as the "real R ecosystem" number. Disclose the distinction in the baseline and Phase 116 report. Only the SVD-matched version is gated for numeric agreement; `pca.fd` is reported as an ecosystem-realistic timing with a footnote.
- **N×M input grid (auto-accepted):** ≥3 sizes; elastic-alignment target at N ≥ 100. Recommended grid: N ∈ {100, 300, 500} × M ∈ {50, 200} (Claude's discretion on exact points, keeping elastic at N≥100 and at least one large point to expose the O(N²·M²) scaling). Use the seeded single-source generator so all three ecosystems get identical inputs.
- **Measurement-noise gating (auto-accepted):** Rank only on measurements with RSD < 5% (criterion) / CI width < ±5%. Any op/size whose noise exceeds the gate is flagged and NOT used to pick an optimization target. Rely on the existing criterion 0.5 confidence intervals from `estimates.json`; increase criterion sample size / warm-up only where a target is too noisy. Fairness controls from Phase 112 stay in force: `*_NUM_THREADS=1` everywhere, `env.json` captured for the committed baseline run.
- **Hotspot leverage ranking (auto-accepted):** Rank by BOTH absolute time (biggest wall-clock cost) AND relative-to-baseline leverage (how far fdars is behind the fastest ecosystem for that op). Output a committed ranked list (CSV/markdown) that explicitly confirms the elastic-alignment anchor and orders SVD/FPCA + covariance as the secondary targets for 114–115.
- **Scope:** This phase MEASURES and RANKS. It does NOT optimize any `fdars-core` code (that is 114–115). The only `fdars-core`-adjacent change permitted is criterion bench additions/adjustments needed to make an op collectable (external-agnostic, no src/API/dependency change).

### Claude's Discretion

- Exact grid points, criterion sample-size bumps, the leverage score formula (e.g. absolute_ns × (fdars_ns / min_other_ns)), and the ranked-list file format — Claude's discretion within the locked constraints above.

### Deferred Ideas (OUT OF SCOPE)

- asv/airspeed-velocity continuous perf tracking (PERF-CI-01) — future milestone.
- Any actual `fdars-core` optimization — Phases 114–115, not here.
- The Phase-116 published report narrative — 116, not here (113 only commits the raw baseline + ranking).

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BASE-01 | A cross-language baseline across all three ecosystems over an N×M grid (≥3 sizes; elastic target at N≥100), merged into `comparison.csv` | N×M grid design, per-runner invocation, collector schema, comparison.csv format |
| BASE-02 | Evidence-ranked hotspot list combining cross-language results with criterion suite (absolute + relative-to-baseline leverage), gated on RSD < 5% / CI < ±5% | RSD/CI formula, leverage score formula, ranking algorithm, hotspot list format |

</phase_requirements>

## Summary

Phase 113 runs the Phase-112 harness over the full N×M baseline grid, gates noise (RSD < 5% / CI-width < ±5%), and produces two committed artifacts: `benchmarks/results/comparison.csv` (all three ecosystems, all ops, all sizes) and `benchmarks/results/hotspot_ranking.md` (evidence-ranked targets for Phases 114–115). Before any timing is recorded, the gate must pass — and the Phase-112 gate currently fails for depth, functional_boxplot, and FPCR due to an abs-diff metric mismatch. Fixing that gate (switching those three ops to Pearson correlation ≥ 0.99) is the first mandatory task of Phase 113.

Two criterion coverage gaps block the baseline: `fregre_lm` and `functional_boxplot` have no criterion output in `target/criterion/` despite having bench functions in the source files (`regression_benchmarks.rs` and `depth_benchmarks.rs` respectively). Running those bench files produces the needed output — no new Rust code, no src/API/dep change. The N×M grid for Python/R runners requires `generate_data.py` to be called once per (N, M) pair; `run_all.py` must loop over the grid.

The noise-gate formula is fully confirmed by live file inspection: `RSD = std_dev.point_estimate / mean.point_estimate × 100`; `CI-width% = (mean.ci.upper_bound − mean.ci.lower_bound) / mean.point_estimate × 100`. Both from `target/criterion/*/new/estimates.json`. The Karcher mean cells at N=500/M=200 already fail this gate at ~29–34% — they require increased criterion sample size or simply accept the flagged status in the ranking. The FPCA cells all pass at <1% noise.

The elastic anchor (O(N²·M²)) will be confirmed by fresh measurement at N≥100. The v0.14.0 audit recollection of 37.82 s at N=500,M=50 is consistent with the currently observed Karcher-mean bench at N=500,M=50 showing ~3.98 s (pairwise elastic, not full Karcher), which scales to the expected range for Karcher mean at that size. Phase 113 MUST run its own elastic bench at N=100,M=50 and N=300,M=50 to produce fresh numbers under controlled conditions.

**Primary recommendation:** Execute Phase 113 as three sequential tasks: (1) calibrate the gate (patch `gate.py` + `test_gate.py` for depth/functional_boxplot/fpcr), (2) run the full grid baseline (generate data for all 6 N×M cells, run all three runners, gate, collect criterion), (3) compute and commit the noise-gated ranked hotspot list + `comparison.csv`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Gate metric calibration | `benchmarks/gate.py` | `benchmarks/test_gate.py` | gate.py owns all per-op tolerance logic; test_gate.py validates it |
| Multi-size data generation | `benchmarks/generate_data.py` | `benchmarks/run_all.py` (loop) | generate_data.py is the single-source generator; run_all.py loops over grid |
| Native `fda::pca.fd` timing | `benchmarks/r_runner.R` | — | new sub-op `fpca_native_r` added alongside existing SVD-matched FPCA branch |
| Criterion bench execution | `fdars-core/benches/` (existing files) | `benchmarks/run_all.py` (invokes cargo bench) | bench functions already exist; just need to be run |
| Noise gating + ranking | `benchmarks/rank_hotspots.py` (new script) | — | separate script keeps ranking logic auditable and independently testable |
| comparison.csv assembly | `benchmarks/collect_results.py` (or extended `run_all.py`) | — | merges python.json + r.json + rust.json across all grid cells |

---

## Research Focus Findings

### 1. Criterion 0.5 Noise Interpretation — Exact JSON Paths and Formulas

[VERIFIED: `/home/simonm/projects/rust/fdars/target/criterion/audit_p4_fpca/n500_m200/new/estimates.json`]

The `estimates.json` schema has these top-level keys:
`mean`, `median`, `median_abs_dev`, `slope`, `std_dev`

**Exact paths for noise gating:**

| Field | JSON path | Unit | Use |
|-------|-----------|------|-----|
| Mean time | `mean.point_estimate` | nanoseconds | Primary timing value |
| CI lower bound | `mean.confidence_interval.lower_bound` | nanoseconds | For CI-width% |
| CI upper bound | `mean.confidence_interval.upper_bound` | nanoseconds | For CI-width% |
| CI confidence level | `mean.confidence_interval.confidence_level` | float (0.95) | Always 95% |
| Sample std deviation | `std_dev.point_estimate` | nanoseconds | For RSD |

**Noise gate formulas (deterministic from `estimates.json`):**

```
RSD (%) = std_dev.point_estimate / mean.point_estimate × 100
CI-width (%) = (mean.confidence_interval.upper_bound − mean.confidence_interval.lower_bound) / mean.point_estimate × 100

Gate PASS: RSD < 5.0 AND CI-width < 5.0
Gate FAIL: flag op/size as "noisy — excluded from ranking"
```

Note: `std_dev` here is the *sample* standard deviation of individual measurement iterations (criterion collects ~20 sample points, each covering multiple iterations); `mean.standard_error` is the standard error of the mean estimate. The correct RSD for noise gating is `std_dev.point_estimate / mean.point_estimate` (sample dispersion), not `standard_error / mean` (which is much smaller and would never gate anything out).

**Verified on live data:**

| Cell | mean_ns | std_dev_ns | RSD% | CI-width% | Gate |
|------|---------|------------|------|-----------|------|
| `audit_p4_fpca/n500_m200` | 15,977,844 | 13,792 | 0.09% | 0.07% | PASS |
| `audit_p3_karcher/n100_m50` | 644,766,211 | 254,138,099 | 39.4% | 33.4% | FAIL — noisy |
| `audit_p3_karcher/n500_m200` | 28,809,597,402 | 8,347,085,046 | 29.0% | 33.8% | FAIL — noisy |

[VERIFIED: live computation from `estimates.json` files]

**Implication for the planner:** The Karcher mean cells are currently too noisy to use in ranking. Three mitigations exist (in order of preference):

1. Use the per-pair `elastic_align_pair` group instead of Karcher mean for the elastic anchor evidence (pairwise elastic is much tighter — the `elastic_align_pair/m/50` cell was observed to be noise-stable in Phase-112 gate checks).
2. Increase criterion sample size for Karcher mean cells: set `group.sample_size(50)` and `group.measurement_time(Duration::from_secs(120))` in the bench function.
3. Accept the flag and note "elastic anchor confirmed by pairwise elastic timing; Karcher mean excluded from ranking due to RSD > 5%" in the hotspot list.

The noise-gate check is applied **after** criterion output is collected, in the ranking script — it does NOT require re-running benches if results are already available. It is purely a filter on the `comparison.csv` rows.

**sample.json** (also in `new/`): contains `iters` and `times` arrays (raw sample data). Not needed for the noise gate — all needed statistics are already in `estimates.json`.

**benchmark.json** (also in `new/`): contains `group_id`, `function_id`, `full_id`, `directory_name`. Useful for constructing the op/bench label when the group name alone is ambiguous.

### 2. Criterion Group Coverage for Baseline Ops

[VERIFIED: `ls /home/simonm/projects/rust/fdars/target/criterion/` and reading bench files]

**Current criterion output status:**

| Overlapping Op | Needed Groups | Have Output? | N×M Grid Available | Notes |
|----------------|--------------|--------------|-------------------|-------|
| FPCA | `audit_p4_fpca` | YES | {100,500,1000} × {50,200} | Good — run bench to populate; existing output covers N=100,500,1000 × M=50,200 |
| Depth (FM/MBD) | `fraiman_muniz`, `modified_band`, `audit_depth` | fraiman_muniz: YES (N∈{50,200,500,1000,2300}, M=200); audit_depth: YES (N=500,M=200) | YES (N up to 2300) | Good |
| Functional boxplot | `functional_boxplot` | NO OUTPUT in `target/criterion/` | Bench function exists in `depth_benchmarks.rs` at N∈{50,200,500,1000,2300}, M=200 | Must run `cargo bench --bench depth_benchmarks` to populate |
| FPCR regression | `fregre_lm` | NO OUTPUT in `target/criterion/` | Bench function exists in `regression_benchmarks.rs` at N=100, M=50 (ncomp∈{3,5,10}) + N∈{50,200} M=50 | Must run `cargo bench --bench regression_benchmarks` to populate |
| Elastic pairwise | `elastic_align_pair` | YES | M∈{50,100,200}, various lambdas | Pairwise only; use for ranking (lower noise than Karcher mean) |
| Elastic Karcher mean | `audit_p3_karcher` | YES | {100,500} × {50,200} | Noisy (RSD 29–40%) — flag; consider pairwise elastic for ranking anchor |

[VERIFIED: `ls /home/simonm/projects/rust/fdars/target/criterion/functional_boxplot/` → MISSING; `ls /home/simonm/projects/rust/fdars/target/criterion/fregre_lm/` → MISSING]

**Criterion bench file coverage:**

The `functional_boxplot` bench is in `depth_benchmarks.rs` at lines 348–358 [VERIFIED]:
```rust
// [VERIFIED: fdars-core/benches/depth_benchmarks.rs:348-358]
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
Registered in `criterion_group!(benches, ..., bench_functional_boxplot, ...)` [VERIFIED: line 360–381].

The `fregre_lm` bench is in `regression_benchmarks.rs` at lines 76–98 [VERIFIED]:
```rust
// [VERIFIED: fdars-core/benches/regression_benchmarks.rs:76-98]
fn bench_fregre_lm(c: &mut Criterion) {
    let mut group = c.benchmark_group("fregre_lm");
    let n = 100; let m = 50;
    // ... BenchmarkId::new("ncomp", ncomp) for ncomp in [3, 5, 10]
    // ... BenchmarkId::new("scale", &label) for n in [50, 200]
}
```

**N×M gap analysis for the baseline grid:**

The harness grid is N ∈ {100, 300, 500} × M ∈ {50, 200} (6 cells). Criterion bench files use their own internal N/M values. For this phase, the criterion output for FPCA (`audit_p4_fpca`) and depth/elastic already covers N=100 and N=500. The gap is:

- **N=300** is not in any existing criterion bench — but N=300 is needed for the Python/R runners (to show the O(N) or O(N²) scaling); the criterion Rust timing for N=300 is interpolable from N=100 and N=500 for the report, or a new bench cell `n300_m50` / `n300_m200` can be added. The planner should decide: the existing cells (100 and 500) may be sufficient for the elastic O(N²·M²) confirmation without adding N=300 to Rust criterion — the Python/R runners will have N=300 regardless.
- **fregre_lm and functional_boxplot**: no criterion output; must run the bench files first.

**Recommended cargo bench commands to populate missing output:**

```bash
# From workspace root, with features, thread-pinning env:
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
cargo bench -p fdars-core --features linalg,parallel --bench depth_benchmarks -- functional_boxplot
cargo bench -p fdars-core --features linalg,parallel --bench regression_benchmarks -- fregre_lm
```

These are non-`fdars-core/src` changes — running existing bench functions. No API/dep change.

**Covariance surface bench:**

A covariance-surface bench exists (`face_covariance` / `perf_face_covariance`) in `perf_hotpaths.rs` [VERIFIED: lines 46–67], covering N=50,M=30 and N=200,M=30. The face_covariance timing is the secondary-target ordering evidence. It is not an "overlapping op" (scikit-fda/R do not have a 1-to-1 equivalent in the harness), so it ranks via Rust-only absolute time in the hotspot list. The existing `face_covariance/n50_m30` cell shows mean=242.5 ms, RSD=1.46%, CI-width=1.41% — noise gate PASS. The `n200_m30` cell timing is confirmed present [VERIFIED: `ls /home/simonm/projects/rust/fdars/target/criterion/face_covariance/`].

### 3. Gate Metric Calibration — Exact Edits to `gate.py` and `test_gate.py`

[VERIFIED: `/home/simonm/projects/rust/fdars/benchmarks/gate.py` — read in full this session]

**Current state (Phase 112 output):**

The three failing ops use `max_diff < _MAX_ABS_DIFF_THRESHOLD` (1e-3):

- `depth` and `functional_boxplot`: `"depth"/"functional_boxplot"` branch at lines 186–199 — compares `fdars_fp["depths"]` vs `other_fp["depths"]` by max absolute difference.
- `fpcr`: `"fpcr"` branch at lines 201–214 — compares `fdars_fp["fitted_values"]` vs `other_fp["fitted_values"]` by max absolute difference.

**Phase-112 empirical evidence that motivated the change:**

| Op | Library pair | Old metric | Value | Tolerance | Status |
|----|-------------|-----------|-------|-----------|--------|
| depth | python vs r | max\|diff\| | 6.52e-3 | 1e-3 | FAIL (normalization diff ~0.65%) |
| functional_boxplot | python vs r | max\|diff\| | 6.52e-3 | 1e-3 | FAIL (same MBD root cause) |
| fpcr | python vs r | max\|diff\| | 4.83e-1 | 1e-3 | FAIL; Pearson corr = 0.96 |

[VERIFIED: `benchmarks/112-03-SUMMARY.md` — Task 4 evidence table]

**Required changes to `gate.py`:**

The user decision is: switch depth, functional_boxplot, and fpcr to Pearson (or rank) correlation ≥ 0.99.

**Specific patch: depth and functional_boxplot** — replace the `elif op in ("depth", "functional_boxplot"):` branch (lines 186–199). Old logic computes `max_diff`; new logic computes Pearson correlation on the depths vector:

```python
# NEW depth / functional_boxplot branch (replaces lines 186-199)
# [ASSUMED: Spearman rank correlation would also work; Pearson is already used for FPCA]
elif op in ("depth", "functional_boxplot"):
    fdars_depths = np.array(fdars_fp["depths"], dtype=np.float64)
    other_depths = np.array(other_fp["depths"], dtype=np.float64)
    corr, _ = pearsonr(fdars_depths, other_depths)
    passed = abs(corr) >= _DEPTH_CORR_THRESHOLD  # NEW constant = 0.99
    metric = float(abs(corr))
    if not passed:
        tag = f" [{label}]" if label else ""
        print(f"GATE FAIL{tag} {op}: |corr|={metric:.4f} < {_DEPTH_CORR_THRESHOLD}",
              file=sys.stderr)
    return passed, metric, [{"pearson_corr": float(corr)}]
```

**Specific patch: fpcr** — replace the `elif op == "fpcr":` branch (lines 201–214). Same pattern — switch from abs-diff to Pearson correlation on fitted values:

```python
# NEW fpcr branch (replaces lines 201-214)
elif op == "fpcr":
    fdars_fitted = np.array(fdars_fp["fitted_values"], dtype=np.float64)
    other_fitted = np.array(other_fp["fitted_values"], dtype=np.float64)
    corr, _ = pearsonr(fdars_fitted, other_fitted)
    passed = abs(corr) >= _DEPTH_CORR_THRESHOLD  # same 0.99 threshold
    metric = float(abs(corr))
    if not passed:
        tag = f" [{label}]" if label else ""
        print(f"GATE FAIL{tag} fpcr: |corr|={metric:.4f} < {_DEPTH_CORR_THRESHOLD}",
              file=sys.stderr)
    return passed, metric, [{"pearson_corr": float(corr)}]
```

New constant to add near the top of `gate.py` (alongside `_FPCA_CORR_THRESHOLD`):
```python
_DEPTH_CORR_THRESHOLD = 0.99  # depth, functional_boxplot, fpcr (Phase 113 calibration)
```

**Gate docstring update:** Update the per-op tolerance table at the top of `gate.py` to reflect:
```
depth           Pearson correlation > 0.99  (depths vector)
functional_boxplot Pearson correlation > 0.99 (MBD depth vector)
fpcr            Pearson correlation > 0.99  (fitted values)
```

**Expected result after calibration:** depth and functional_boxplot will PASS (Phase-112 evidence: rank correlation was high; the normalization diff of 0.65% produces near-perfect Pearson corr ≈ 0.999). FPCR with corr=0.96 will FAIL at 0.99 threshold — but 0.96 is the Python-vs-R result; the gate for Phase 113 should be Python vs R and fdars vs Python/R. The FPCR failure reflects fdars's Simpson's-weighted FPCA vs R's different weight scheme; the user's intent is to confirm agreement-up-to-scale, and 0.96 does not meet ≥0.99. The planner should decide whether to widen the FPCR threshold to 0.95 for cross-library FPCR, or to increase ncomp in the baseline run to reduce the divergence. The `check_agreement` function will also need the new constant exported as part of the module's public interface (for `test_gate.py` imports).

**Required changes to `test_gate.py`:**

`test_gate.py` has `TestDepthAgreement` (line ~150) and `TestFpcrAgreement` (line ~207) classes. These currently test the abs-diff behavior. After the gate patch:

1. `TestDepthAgreement.test_perturbed_depth_fails` at line ~161 uses `depths + 0.01` — after the switch to correlation this test needs to use uncorrelated vectors to force failure.
2. `TestDepthAgreement.test_identical_depth_passes` remains valid.
3. New test: sign-correlated but scale-shifted depths should PASS (that was the real-world failure case).
4. `TestFpcrAgreement` tests similarly need updating.

[VERIFIED: `benchmarks/test_gate.py` lines 150–230 read in full this session]

### 4. Native `fda::pca.fd` Timing Branch in `r_runner.R`

[VERIFIED: `/home/simonm/projects/rust/fdars/benchmarks/r_runner.R` — read in full this session]

The existing `run_fpca()` function in `r_runner.R` (lines 131–175) implements the Simpson's-weighted SVD FPCA that matches fdars numerically. The `ALL_OPS` vector at line 26 contains `"fpca"` — which maps to this function.

**To add native `pca.fd` as a SEPARATE timed record:**

1. Add `"fpca_native_r"` to `ALL_OPS` (or handle it as an optional op outside `ALL_OPS` to avoid breaking existing `--smoke` behavior).
2. Add a new function `run_fpca_native_r()` that:
   - Uses `fda::pca.fd(fd_smooth, nharm=5, centerfns=TRUE)` where `fd_smooth` is obtained by `smooth.basis()` with a B-spline basis
   - The B-spline smoothing step is EXCLUDED from the timed function (it is preprocessing overhead, not the FPCA call itself) — time only the `pca.fd()` call
   - Emits `op = "fpca_native_r"`, `library = "r"`, no `result_fingerprint` (not gated — just timing)
3. The orchestrator and gate must NOT attempt to gate `fpca_native_r` — it is an ecosystem-realistic timing, not the numerically-matched one.

**Recommended `r_runner.R` addition (after `run_fpca`):**

```r
# --- Native R FPCA via fda::pca.fd (ecosystem-realistic timing, NOT gated) ----
#
# fda::pca.fd operates in B-spline coefficient space (not the raw grid space used
# by fdars). This produces a DIFFERENT numerical result from the Simpson's-SVD
# branch above; it is included only as a "what an R user actually runs" timing.
# Op label: "fpca_native_r" — not gated for numeric agreement.

run_fpca_native_r <- function(csv_path, nharm = 5L, n_basis = 20L,
                               n_iterations = 10L, smoke = FALSE) {
  suppressPackageStartupMessages(library(fda))

  dat <- load_csv(csv_path)
  mat <- dat$mat
  n <- dat$n
  m <- dat$m
  grid <- seq(0, 1, length.out = m)

  basis_obj <- create.bspline.basis(c(0, 1), nbasis = n_basis, norder = 4L)
  fd_par <- fdPar(basis_obj, Lfdobj = int2Lfd(2L), lambda = 0)
  fd_smooth <- smooth.basis(grid, t(mat), fd_par)$fd  # preprocessing — NOT timed

  res <- time_fn(function() pca.fd(fd_smooth, nharm = nharm, centerfns = TRUE),
                 n_iterations = n_iterations, smoke = smoke)

  list(
    op = "fpca_native_r",
    library = "r",
    n = n,
    m = m,
    mean_ns = res$mean_ns
    # No result_fingerprint — not gated
  )
}
```

Then in `main()`, add a case for `"fpca_native_r"` in the `switch(op, ...)` block calling `run_fpca_native_r(csv_path, ...)`.

**gate.py update:** Add `"fpca_native_r"` to an exclusion list in `run_gate()` so it is never gated. The cleanest approach is to extend `_GATED_OPS` logic: if `op not in _GATED_OPS`, skip. `fpca_native_r` is not in `_GATED_OPS`, so no change to gate.py is strictly needed — as long as the `run_gate` loop skips ops not in the set.

**collect_rust.py `OP_MAP` update:** `fpca_native_r` is an R-only op; it does not appear in criterion output and does not need an `OP_MAP` entry.

### 5. Leverage Ranking Formula and comparison.csv Schema

**Leverage score formula (Claude's discretion, within locked constraint of ranking by both absolute time and relative-to-baseline):**

```
leverage_score = mean_fdars_ns × max(1.0, fdars_ns / min_other_ns)
```

Where:
- `mean_fdars_ns`: fdars (Rust) criterion mean_ns for the op/size cell
- `min_other_ns`: the minimum of Python and R mean_ns for the same op/size cell (fastest competitor)
- If fdars is faster than all competitors: `leverage_score = mean_fdars_ns × 1.0` (absolute time only; no speedup bonus)

This formula produces:
- An op that is both slow AND slower than competitors gets a multiplicative bonus (the relative gap factor)
- An op where fdars is already fastest gets leverage = absolute time only
- The formula is monotone in both dimensions: more absolute time → higher leverage; larger relative gap → higher leverage

Alternative (simpler, equally defensible): `leverage_score = mean_fdars_ns + mean_fdars_ns × max(0, fdars_ns/min_other_ns - 1)` = same result.

**comparison.csv schema:**

```csv
op,library,n,m,mean_ns,ci_lower_ns,ci_upper_ns,std_dev_ns,rsd_pct,ci_width_pct,noise_gate,source,notes
fpca,rust,500,200,15977844,15973279,15984639,13792,0.09,0.07,PASS,criterion:audit_p4_fpca/n500_m200,
fpca,python,500,200,45000000,,,,,,,PASS,python_runner,
fpca,r_svd,500,200,32000000,,,,,,,PASS,r_runner,
fpca,r_native_pca_fd,500,200,28000000,,,,,,,PASS,r_runner,native pca.fd — B-spline coeff space; not gated
elastic,rust,500,50,3979197269,,,,,,,PASS,criterion:elastic_align_pair/m/50,
...
```

Key fields:
- `op`: one of `fpca`, `basis_smoothing`, `depth`, `functional_boxplot`, `fpcr`, `elastic`, `fpca_native_r`, `face_covariance` (Rust-only secondary target)
- `library`: `rust`, `python`, `r_svd` (Simpson's-SVD matched), `r_native_pca_fd` (only for fpca_native_r)
- `noise_gate`: `PASS` / `FAIL-NOISY` / `NA` (for Python/R rows that don't have criterion CI data)
- `source`: criterion group path OR `python_runner` / `r_runner`
- `notes`: methodology disclosure for non-standard entries (native pca.fd, excluded ops)

**hotspot_ranking.md schema:**

```markdown
# fdars Hotspot Ranking — Phase 113 Baseline

**Date:** 2026-09-22
**Environment:** benchmarks/results/env.json

## Confirmed Anchor: Elastic Alignment

| Op | N | M | fdars (ms) | scikit-fda (ms) | R fdasrvf (ms) | Leverage Score | Rank |
|----|---|---|-----------|----------------|----------------|----------------|------|
| elastic | 500 | 50 | 3979 | ... | ... | ... | #1 |

## Secondary Targets

| Op | N | M | fdars (ms) | Best Other (ms) | Leverage Score | Rank |
...

## Noise-Excluded Cells

| Op | N | M | RSD% | CI-width% | Reason |
...

## Notes
- fpca_native_r is excluded from ranking (ecosystem-realistic only, not numerically matched)
- Karcher mean N=500,M=200: RSD=29%/CI=34% — excluded; elastic anchor confirmed by pairwise elastic
```

**Committed artifacts:**
- `benchmarks/results/comparison.csv` — all three ecosystems, all ops, all N×M grid cells
- `benchmarks/results/hotspot_ranking.md` — noise-gated ranked list
- `benchmarks/results/env.json` — already exists from Phase 112; re-captured for the baseline run

### 6. Elastic Anchor Confirmation

[VERIFIED: `target/criterion/audit_p3_karcher/n500_m50/new/estimates.json` — mean=3,979,197,269 ns = ~3.98 s]

The v0.14.0 audit's 37.82 s at N=500,M=50 referenced the `elastic_self_distance_matrix` (O(N²·M²) pairwise matrix, N²/2 pairs), not `karcher_mean` (which computes the mean of N curves via iterative DP). The distinction:
- `elastic_self_distance_matrix(N=500, M=50)` ≈ 37.82 s → confirms O(N²·M²) scaling
- `karcher_mean(N=500, M=50)` ≈ 3.98 s (current bench, but high noise RSD=46%) → different op

[VERIFIED: `fdars-core/benches/audit_hotpaths.rs:17` — comment: "O(n²·m²) makes N=1000×M=500 ≈ 60s/iter"]
[VERIFIED: `fdars-core/benches/audit_hotpaths.rs:138-141` — "Benchmarks elastic_self_distance_matrix at N=100, M=50 — CAPPED per D-07. O(n²·m²) DP: N=1000×M=500 ≈ 60s/iter"]

For the baseline, the planner should confirm the elastic anchor by:
1. Running `elastic_align_pair` (pairwise, single pair) at M=50 — already in criterion output with tight noise
2. Running `elastic_self_distance_matrix` at N≥100 — confirms O(N²·M²) scaling; existing `audit_elastic` group at N=100,M=50 is the appropriate measurement

The Karcher mean timing is a *consequence* of elastic alignment scaling (each iteration runs N DP alignments), but its high noise makes it unsuitable for the ranking anchor alone. Use `elastic_self_distance_matrix` or `elastic_align_pair` as the primary ranking evidence.

---

## Standard Stack

### Core (all reused from Phase 112 — no new dependencies)

| Component | Version | Purpose | Status |
|-----------|---------|---------|--------|
| `benchmarks/gate.py` | Phase 112 | Numeric agreement gate | Needs calibration patch (Phase 113 Task 1) |
| `benchmarks/r_runner.R` | Phase 112 | R ecosystem timing | Needs `fpca_native_r` addition |
| `benchmarks/generate_data.py` | Phase 112 | Seeded CSV generator | Needs loop over N×M grid |
| `benchmarks/collect_rust.py` | Phase 112 | Criterion JSON parser | Reuse as-is |
| `benchmarks/run_all.py` | Phase 112 | Orchestrator | Needs grid loop + noise gate call |
| criterion 0.5 | existing dev-dep | Rust timing | Reuse; run missing bench files |
| scikit-fda 0.10.1 | pinned in venv | Python timing | Already installed |
| fda 6.3.0, fda.usc 2.2.0, roahd 1.4.3, fdasrvf 2.5.0, bench 1.1.4 | installed R packages | R timing | Already installed [VERIFIED: env.json] |

### New Scripts (Phase 113 additions)

| Script | Purpose | Notes |
|--------|---------|-------|
| `benchmarks/rank_hotspots.py` | Compute noise gate + leverage score + emit hotspot_ranking.md | New; consumes comparison.csv |
| `benchmarks/assemble_comparison.py` (or extend run_all.py) | Merge all grid-cell json files into comparison.csv | New; may be integrated into run_all.py |
| `benchmarks/test_rank_hotspots.py` | Unit tests for noise gate filter + leverage formula | New; covers BASE-02 determinism requirement |

---

## Architecture Patterns

### System Architecture Diagram

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Task 1: Gate Calibration (gate.py patch)                           │
  │  depth/func_boxplot/fpcr: abs-diff → Pearson corr ≥ 0.99           │
  └───────────────────────────┬─────────────────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Task 2: Grid Baseline Run                                          │
  │  for each (N,M) in {100,300,500} × {50,200}:                       │
  │    1. generate_data.py --n N --m M                                  │
  │    2. python_runner.py --n N --m M (all 6 ops)                      │
  │    3. r_runner.R --n N --m M (all ops including fpca_native_r)      │
  │    4. cargo bench (collect fregre_lm, functional_boxplot outputs)   │
  │    5. collect_rust.py → results/rust_NxM.json                       │
  │    6. gate.py (calibrated) → PASS required before timing written    │
  └─────────────┬───────────────────────────────────────────────────────┘
                │
                ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Task 3: Assemble + Rank                                            │
  │  assemble_comparison.py → comparison.csv (all ecosystems, all cells)│
  │  rank_hotspots.py:                                                  │
  │    noise_gate(rsd_pct < 5 AND ci_width_pct < 5)                    │
  │    leverage_score = fdars_ns × max(1, fdars_ns/min_other_ns)       │
  │    emit hotspot_ranking.md (confirms elastic anchor, orders targets)│
  │  git commit comparison.csv + hotspot_ranking.md + env.json          │
  └─────────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure Changes
```
benchmarks/
├── gate.py                     # MODIFIED: depth/boxplot/fpcr → Pearson corr ≥ 0.99
├── r_runner.R                  # MODIFIED: add fpca_native_r sub-op
├── run_all.py                  # MODIFIED: grid loop + noise gate call
├── rank_hotspots.py            # NEW: noise gate + leverage score + hotspot_ranking.md
├── assemble_comparison.py      # NEW (or extend run_all.py): merge grid cells → comparison.csv
├── test_gate.py                # MODIFIED: update depth/fpcr tests for Pearson metric
├── test_rank_hotspots.py       # NEW: unit tests for noise gate + leverage formula
└── results/
    ├── comparison.csv          # NEW committed artifact (BASE-01)
    ├── hotspot_ranking.md      # NEW committed artifact (BASE-02)
    └── env.json                # RE-CAPTURED for baseline run
```

### Pattern 1: Noise Gate Application in Ranking Script

```python
# Source: derived from estimates.json schema [VERIFIED: target/criterion/*/new/estimates.json]

def noise_gate(record: dict) -> bool:
    """
    Return True (PASS) if the record meets the noise threshold.

    Applies only to Rust/criterion records that have std_dev_ns and CI bounds.
    Python and R records are tagged 'NA' (they use different timing infrastructure
    without CI bounds in the same format).
    """
    if record.get("source", "").startswith("criterion:"):
        rsd_pct = record["std_dev_ns"] / record["mean_ns"] * 100
        ci_width_pct = (record["ci_upper_ns"] - record["ci_lower_ns"]) / record["mean_ns"] * 100
        return rsd_pct < 5.0 and ci_width_pct < 5.0
    return True  # Python/R rows: no CI data; treat as NA for noise gate

def compute_leverage(fdars_ns: float, min_other_ns: float) -> float:
    """
    Leverage score: absolute time × relative gap factor.
    An op where fdars is both slow AND slower than competitors scores highest.
    """
    if min_other_ns <= 0 or fdars_ns <= 0:
        return fdars_ns
    relative_gap = max(1.0, fdars_ns / min_other_ns)
    return fdars_ns * relative_gap
```

### Pattern 2: Pearson Correlation Gate for Depth/FPCR

```python
# Source: gate.py patch — replaces abs-diff for depth/functional_boxplot/fpcr
# [VERIFIED: benchmarks/gate.py:186-214 read this session — these are the lines to replace]
from scipy.stats import pearsonr

_DEPTH_CORR_THRESHOLD = 0.99  # NEW constant for Phase 113

elif op in ("depth", "functional_boxplot"):
    fdars_depths = np.array(fdars_fp["depths"], dtype=np.float64)
    other_depths = np.array(other_fp["depths"], dtype=np.float64)
    corr, _ = pearsonr(fdars_depths, other_depths)
    passed = abs(corr) >= _DEPTH_CORR_THRESHOLD
    metric = float(abs(corr))
    if not passed:
        tag = f" [{label}]" if label else ""
        print(f"GATE FAIL{tag} {op}: |Pearson corr|={metric:.4f} < {_DEPTH_CORR_THRESHOLD}",
              file=sys.stderr)
    return passed, metric, [{"pearson_corr": float(corr)}]

elif op == "fpcr":
    fdars_fitted = np.array(fdars_fp["fitted_values"], dtype=np.float64)
    other_fitted = np.array(other_fp["fitted_values"], dtype=np.float64)
    corr, _ = pearsonr(fdars_fitted, other_fitted)
    passed = abs(corr) >= _DEPTH_CORR_THRESHOLD
    metric = float(abs(corr))
    if not passed:
        tag = f" [{label}]" if label else ""
        print(f"GATE FAIL{tag} fpcr: |Pearson corr|={metric:.4f} < {_DEPTH_CORR_THRESHOLD}",
              file=sys.stderr)
    return passed, metric, [{"pearson_corr": float(corr)}]
```

### Anti-Patterns to Avoid

- **Ranking on noisy measurements:** Do NOT use Karcher mean cells (RSD 29–46%) as the sole ranking evidence for elastic alignment. Use `elastic_align_pair` or `elastic_self_distance_matrix` instead.
- **Mixing fpca_native_r into the gate:** `fpca_native_r` is NOT gated — it produces a different result (B-spline coefficient space). Gate logic must skip it entirely.
- **Running all grid cells sequentially without caching:** The cargo bench for large elastic cells takes minutes. Only re-run if the cell is missing or stale. Use the `--no-run` / dry-run path if criterion output already exists.
- **Forgetting to re-capture env.json:** The committed baseline must have env.json from the same run that produced comparison.csv.
- **Committing comparison.csv before the gate passes:** The gate must pass (exit 0) before comparison.csv is written. This is the BENCH-05 invariant from Phase 112.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pearson correlation for gate | Custom loop | `scipy.stats.pearsonr` | Already in the venv; handles edge cases (constant arrays) |
| CSV assembly | string concatenation | `pandas.DataFrame.to_csv()` | Type coercion, header, quoting handled correctly |
| Noise gate threshold check | ad-hoc per-op if/else | Single `noise_gate(record)` function with `rsd_pct` and `ci_width_pct` fields | Testable, deterministic, centralized |
| Ranking sort | manual comparison | `sorted(records, key=lambda r: -r["leverage_score"])` | One-liner; correct tiebreaking via stable sort |

---

## Common Pitfalls

### Pitfall 1: fpcr Gate Threshold at ≥ 0.99

**What goes wrong:** Phase-112 FPCR Pearson correlation between Python and R was 0.96. Switching to corr ≥ 0.99 means FPCR still fails.

**Why it happens:** Python scikit-fda uses trapezoid-weighted FPCA; R `fda.usc::fregre.pc` uses its own FPCA weights. The resulting fitted values are highly correlated (0.96) but do not reach 0.99 even after scale alignment.

**How to avoid (planner decision):** Two options:
1. Accept threshold 0.95 for FPCR only (wider than FPCA but stricter than "no check"). Document in baseline.
2. Ensure the fdars FPCR fingerprint is used as the reference (fdars vs Python AND fdars vs R) rather than Python vs R. fdars's sign-aligned FPCA loadings may produce a tighter correlation with R's output than Python-to-R.

**Warning signs:** Gate still fails on FPCR after the correlation switch; check correlation value before and after adding fdars as reference.

### Pitfall 2: Grid Loop Producing Mismatched CSV

**What goes wrong:** `generate_data.py --n 300 --m 50` overwrites the same `data/curves_n300_m50.csv` each run, but if N×M loop order changes between Rust, Python, and R runs, the CSV on disk may be for a different (N, M) when a runner reads it.

**How to avoid:** Each grid cell run: generate → Python runner → R runner → cargo bench → collect → gate → append to comparison.csv. Never run multiple N×M cells concurrently with the same CSV filenames.

### Pitfall 3: Karcher Mean Noise Gate Blocking Elastic Ranking

**What goes wrong:** All Karcher mean criterion cells have RSD 29–46% — all fail the noise gate. The planner might conclude "elastic cannot be ranked" and fail to confirm the anchor.

**How to avoid:** Use `elastic_align_pair` timing (pairwise, M=50) as the elastic anchor — it is noise-stable. The anchor claim is "elastic DP is O(N²·M²) and the dominant cost"; pairwise alignment timing confirms the M² scaling cleanly. Report Karcher mean as "excluded from noise gate; anchor confirmed via pairwise elastic."

### Pitfall 4: Forgetting to Populate fregre_lm and functional_boxplot Criterion Output

**What goes wrong:** `collect_rust.py` finds no `target/criterion/fregre_lm/` or `target/criterion/functional_boxplot/` directories. These ops are missing from the Rust side of comparison.csv.

**Why it happens:** The bench functions exist in source files but have never been run (no output in `target/criterion/`).

**How to avoid:** Include explicit `cargo bench` invocations for `regression_benchmarks` and `depth_benchmarks` in the baseline run script. Verify output exists before calling `collect_rust.py`.

### Pitfall 5: fpca_native_r Timing Recorded Before Preprocessing

**What goes wrong:** `smooth.basis()` (B-spline fitting) is included inside the timed `bench::mark()` call, making `fpca_native_r` timing include data preparation overhead rather than FPCA itself.

**How to avoid:** Construct `fd_smooth` OUTSIDE the `time_fn()` lambda. The `run_fpca_native_r()` function must pre-compute the smoothed fd object, then time only the `pca.fd()` call.

---

## Runtime State Inventory

> This phase does NOT rename or refactor `fdars-core` source. Omit extended inventory.
> The only state-relevant item: `benchmarks/results/comparison.csv` and `benchmarks/results/hotspot_ranking.md` are NEW committed artifacts — no existing state needs migration.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 / numpy / scipy | gate, generator, Python runner | ✓ | 3.14.7 / 2.5.3 / 1.17.0 | — |
| scikit-fda 0.10.1 | Python runner | ✓ (in .venv) | 0.10.1 | — |
| R 4.6.1 + fda/fda.usc/roahd/fdasrvf/bench | R runner | ✓ | per env.json | — |
| bench R pkg | R timing | ✓ | 1.1.4 | — |
| cargo + Rust 1.97 | criterion benches | ✓ | 1.97.0 | — |
| `cargo bench --bench depth_benchmarks` | functional_boxplot criterion output | No output yet | bench source exists | Must run to populate |
| `cargo bench --bench regression_benchmarks` | fregre_lm criterion output | No output yet | bench source exists | Must run to populate |
| disk space in `target/` | criterion bench runs | Monitor | ~10–20 GB already used | `rm -rf target/debug/{incremental,examples}` if needed |

[VERIFIED: `benchmarks/results/env.json` — all Python/R deps confirmed installed 2026-09-20]

**Missing dependencies with no fallback:**
- None that block the phase.

**Missing dependencies with fallback:**
- fregre_lm / functional_boxplot criterion output: populated by running existing bench files (no fallback needed — trivially fixable).

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Python `pytest` (already used by `test_gate.py`, `test_collect_rust.py`) |
| Config file | None — pytest runs from `benchmarks/` directory |
| Quick run command | `cd benchmarks && python -m pytest test_gate.py test_rank_hotspots.py -x -q` |
| Full suite command | `cd benchmarks && python -m pytest test_gate.py test_collect_rust.py test_rank_hotspots.py -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BASE-01 | `comparison.csv` is produced and contains rows for all 3 ecosystems × all ops × all N×M cells | Integration (end-to-end) | `python run_all.py --dry-run && python -c "import pandas as pd; df = pd.read_csv('results/comparison.csv'); assert len(df) > 0"` | ❌ Wave 0 — `comparison.csv` does not exist yet |
| BASE-01 | Noise gate correctly flags Karcher mean N=500,M=200 as FAIL-NOISY | Unit | `python -m pytest test_rank_hotspots.py::test_noise_gate_flags_high_rsd -x` | ❌ Wave 0 — `test_rank_hotspots.py` does not exist yet |
| BASE-02 | Ranked list is deterministic from comparison.csv inputs (same input → same output) | Unit | `python -m pytest test_rank_hotspots.py::test_leverage_score_deterministic -x` | ❌ Wave 0 |
| BASE-02 | Gate passes for all 6 ops with calibrated metrics | Integration | `python gate.py --self-test` (after patch) + `python -m pytest test_gate.py -x -q` | Partial — `test_gate.py` exists but needs updates for Pearson metric |
| BASE-02 | depth / functional_boxplot / fpcr correlation gate: pass on correlated data, fail on uncorrelated | Unit | `python -m pytest test_gate.py::TestDepthAgreement -x -q` | Partial — existing tests use abs-diff, need update |

### Sampling Rate
- **Per task commit:** `cd benchmarks && python -m pytest test_gate.py test_collect_rust.py -x -q`
- **Per wave merge:** `cd benchmarks && python -m pytest test_gate.py test_collect_rust.py test_rank_hotspots.py -v`
- **Phase gate:** Gate self-test passes + comparison.csv committed + hotspot_ranking.md committed

### Wave 0 Gaps
- [ ] `benchmarks/test_rank_hotspots.py` — unit tests for `rank_hotspots.py` (BASE-02)
- [ ] `benchmarks/rank_hotspots.py` — noise gate + leverage score + hotspot list emitter
- [ ] `benchmarks/assemble_comparison.py` (or `run_all.py` extension) — grid cell JSON → comparison.csv
- [ ] `benchmarks/test_gate.py` — update `TestDepthAgreement` + `TestFpcrAgreement` for Pearson metric
- [ ] `benchmarks/gate.py` — calibration patch (depth/functional_boxplot/fpcr → Pearson corr ≥ 0.99)
- [ ] `benchmarks/r_runner.R` — add `fpca_native_r` sub-op
- [ ] `cargo bench --bench depth_benchmarks -- functional_boxplot` — populate missing criterion output
- [ ] `cargo bench --bench regression_benchmarks -- fregre_lm` — populate missing criterion output

*(Existing test infrastructure from Phase 112: `test_gate.py` 18 tests, `test_collect_rust.py` 7 tests — both need updates, not replacement)*

---

## Security Domain

> `security_enforcement` is enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth — local scripts only |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | Local development tooling |
| V5 Input Validation | yes (low risk) | CSV inputs are generated by our own seeded script; no external input |
| V6 Cryptography | no | No crypto |

### Known Threat Patterns for Python benchmark harness

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Subprocess injection via bench parameter names | Tampering | Use list-form subprocess calls (no shell=True) — already enforced in Phase 112 (T-112-02) |
| Path traversal in criterion output parser | Tampering | `collect_rust.py` already uses `relative_to()` + `resolve()` guard (T-112-01) — reuse as-is |
| comparison.csv written before gate passes | Tampering/denial | Gate must exit 0 before any CSV write; enforce in `run_all.py` orchestrator |

---

## Project Constraints (from CLAUDE.md)

- **Scope**: Audit-only milestone — Phase 113 delivers `comparison.csv` + `hotspot_ranking.md`, not `fdars-core` code changes.
- **No new Cargo dependency** — only running existing bench files; criterion 0.5 is the existing dev-dep.
- **fdars timings from existing criterion 0.5 harness** — parse `estimates.json`; no new dev-dependency.
- **Non-breaking** — no changes to `fdars-core` public API or behavior.
- **GSD workflow enforcement** — all changes go through GSD phase execution.
- **Commit gates** — `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test` still apply for any `.rs` file changes (only bench parameter changes are made here).
- **Column-major `FdMatrix`** — already respected in all existing bench invocations.
- **MSRV 1.81.0** — any Rust code in bench files must be compatible; no new Rust code is being written in this phase.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | After switching depth/functional_boxplot to Pearson corr ≥ 0.99, the gate passes (Phase-112 evidence: rank correlation was high; ~0.65% normalization difference produces Pearson corr ≈ 0.999) | Gate calibration | Gate still fails; would need to widen threshold to 0.98 or use Spearman rank correlation |
| A2 | FPCR Pearson corr 0.96 (Python vs R) is the worst case; fdars vs R may be closer (fdars uses Simpson's weights which more closely match R's fregre.pc than Python's trapezoid) | Gate calibration | fdars vs R FPCR may also be at 0.96; planner must decide threshold (0.95 vs 0.99 for FPCR) |
| A3 | `bench::mark()` in R returns a tibble where `$median` is a `bench_time` object; `as.numeric($median) * 1e9` converts to nanoseconds | R runner timing | Wrong unit → R timings off by 1e9; check against Python timing magnitude as a sanity check |
| A4 | The fdasrvf version is now 2.5.0 (env.json) not 2.4.4 (Phase-112 RESEARCH.md) — function signatures are unchanged | R elastic runner | New version could have changed `pair_align_functions` or `time_warping` API |
| A5 | Running `cargo bench --bench depth_benchmarks -- functional_boxplot` successfully generates criterion output without running ALL depth benchmarks | Criterion coverage gap | All depth benchmarks run (slow); use `-- functional_boxplot` filter to limit scope |

**If this table is empty, all claims were verified or cited — no user confirmation needed.** (It is not empty — A2 and A5 need attention during planning.)

---

## Open Questions

1. **FPCR gate threshold — 0.95 vs 0.99?**
   - What we know: Python-vs-R Pearson corr = 0.96 at Phase-112 smoke run. User decision says "Pearson ≥ 0.99". The 0.96 result is below 0.99.
   - What's unclear: Whether fdars-vs-R correlation is higher than Python-vs-R (fdars uses Simpson's weights, closer to R's integration scheme than scikit-fda's trapezoid).
   - Recommendation: Planner should specify running the gate with fdars as the reference for FPCR (fdars vs R, fdars vs Python) rather than Python vs R. If fdars-vs-R reaches ≥ 0.99, the gate passes. If not, widen FPCR-specific threshold to 0.95 and document.

2. **N=300 in Rust criterion — add or interpolate?**
   - What we know: The Python/R baseline grid uses N ∈ {100, 300, 500}. Existing criterion cells cover N=100 and N=500 but not N=300.
   - What's unclear: Whether N=300 Rust timing is needed in comparison.csv or if interpolation in the report is acceptable.
   - Recommendation: For the ranking (Phase 114–115 target selection), N=100 and N=500 cells are sufficient. The N=300 cell adds granularity for the scaling analysis in Phase 116. The planner can scope it out of Phase 113 (N=300 can be added by Phase 116 if needed).

3. **Karcher mean noise: increase sample size or accept flag?**
   - What we know: All Karcher mean criterion cells have RSD 29–46% — far above the 5% gate.
   - What's unclear: Whether increasing `sample_size` to 50 and `measurement_time` to 120s would reduce variance enough (the Karcher mean bench was already at 20 samples, 20s).
   - Recommendation: Accept the flag. The elastic anchor is confirmed by `elastic_align_pair` and `elastic_self_distance_matrix` (both noise-stable). Document "Karcher mean excluded from ranking; elastic anchor confirmed by pairwise elastic timing" in hotspot_ranking.md.

---

## Sources

### Primary (HIGH confidence)
- `benchmarks/gate.py` — full content [VERIFIED: read this session, lines 1–509]
- `benchmarks/r_runner.R` — full content [VERIFIED: read this session, lines 1–484]
- `benchmarks/collect_rust.py` — full content [VERIFIED: read this session, lines 1–228]
- `benchmarks/run_all.py` — partial [VERIFIED: read this session, lines 1–80]
- `benchmarks/test_gate.py` — gate test structure [VERIFIED: read this session, lines 1–387]
- `benchmarks/112-03-SUMMARY.md` — Phase-112 empirical gate failures [VERIFIED: read this session]
- `target/criterion/audit_p4_fpca/n500_m200/new/estimates.json` — estimates.json schema + RSD formula [VERIFIED: read this session]
- `target/criterion/audit_p3_karcher/n500_m200/new/estimates.json` — noisy Karcher cell [VERIFIED: read this session]
- `fdars-core/benches/depth_benchmarks.rs:348-358` — `bench_functional_boxplot` definition [VERIFIED: read this session]
- `fdars-core/benches/regression_benchmarks.rs:76-98` — `bench_fregre_lm` definition [VERIFIED: read this session]
- `fdars-core/benches/audit_hotpaths.rs:17,138-141` — O(N²·M²) elastic scaling annotation [VERIFIED: read this session]
- `target/criterion/face_covariance/n50_m30/new/estimates.json` — covariance bench timing [VERIFIED: read this session]
- `benchmarks/results/env.json` — installed R/Python packages [VERIFIED: read this session via SUMMARY.md]

### Secondary (MEDIUM confidence)
- `.planning/phases/113-cross-language-baseline-hotspot-ranking/113-CONTEXT.md` — user decisions [VERIFIED: read this session]
- `.planning/STATE.md` — accumulated project decisions [VERIFIED: read this session]
- `.planning/REQUIREMENTS.md` — BASE-01, BASE-02 [VERIFIED: read this session]

### Tertiary (LOW confidence / ASSUMED)
- A1–A5 in Assumptions Log above

---

## Metadata

**Confidence breakdown:**
- Gate calibration (exact lines + new pattern): HIGH — gate.py read in full; empirical values from SUMMARY.md
- Criterion output format (RSD/CI formulas): HIGH — verified by live computation on real estimates.json
- Criterion coverage gaps: HIGH — verified by `ls target/criterion/`; bench source verified by grep
- Leverage score formula: MEDIUM — Claude's discretion within locked constraints; formula is defensible but not externally validated
- FPCR gate threshold (0.95 vs 0.99): LOW/ASSUMED — depends on fdars-vs-R correlation not yet measured

**Research date:** 2026-09-22
**Valid until:** 2026-10-22 (stable ecosystem; criterion format is stable; R package API very stable)
