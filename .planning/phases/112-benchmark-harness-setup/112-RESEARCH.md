# Phase 112: Benchmark Harness Setup - Research

**Researched:** 2026-09-20
**Domain:** Cross-language benchmark harness (Rust/Python/R), numeric agreement gate, criterion output parsing
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Harness lives in workspace-root `benchmarks/` directory that is NOT a Cargo workspace member — zero new `fdars-core` dependency.
- Python deps pinned in `benchmarks/requirements.lock` (scikit-fda 0.10.1); R deps pinned in `benchmarks/renv.lock` (`fda`/`fda.usc`/`refund`/`fdapace`/`roahd`/`fdasrvf`).
- fdars timings come from the **existing criterion 0.5 harness** — parse `target/criterion/*/estimates.json` (`mean.point_estimate`, ns). No new dev-dependency, no `cargo-criterion` binary, no `dhat`.
- A **single seeded generator** emits matched curve-set files; **CSV is the default shared format** consumed bit-identically by all three runners.
- `.npy` only if `RcppCNPy` is reliably available on the R side — otherwise default to CSV.
- Overlapping operation set: FPCA, basis smoothing, functional depth, functional boxplot, FPCR regression, elastic alignment / Karcher mean.
- Each runner emits **structured timing JSON** in a shared comparison schema.
- Pin `OMP/OPENBLAS/MKL/RAYON_NUM_THREADS=1` on ALL sides for the baseline run.
- Capture full version + BLAS-backend + CPU + thread-env metadata into `benchmarks/results/env.json`.
- Numeric-agreement gate: sign-aligned FPCA correlation > 0.99; 1e-3 depth/regression; 1e-2 elastic warping.
- FPCA sign-canonicalization must match fdars `fdata_to_pc_1d` convention.
- Verify scikit-fda vs fdars elastic cost-function equivalence on a reference dataset in this phase.
- R `roahd` random-projection-depth default projection count must be pinned to match fdars.
- Default to CSV for shared data format.

### Claude's Discretion

- Orchestration script language/shape (shell vs Python driver), exact JSON schema field names, directory sub-layout under `benchmarks/`, curve-set generator RNG choice.

### Deferred Ideas (OUT OF SCOPE)

- asv/airspeed-velocity continuous perf tracking + CI perf-regression gating (PERF-CI-01).
- Producing the actual baseline comparison numbers (Phase 113).
- Any change to `fdars-core` public API or behavior.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BENCH-01 | External `benchmarks/` harness with seeded single-source generator emitting matched CSV curve-sets consumed identically by Rust/Python/R runners | CSV schema, RNG seed, bit-identity check protocol |
| BENCH-02 | Python runner (scikit-fda 0.10.1 + pytest-benchmark, `requirements.lock`) and R runner (`bench` + fda ecosystem, `renv.lock`) covering FPCA, basis smoothing, functional depth, functional boxplot, FPCR, elastic alignment/Karcher mean | Per-op API table with exact function names and parameter equivalences |
| BENCH-03 | fdars timings from existing criterion 0.5 harness — parse `target/criterion/*/estimates.json` into shared comparison format | Criterion directory layout documented; which overlapping ops have existing benches vs need new ones |
| BENCH-04 | Fairness controls: pin `OMP/OPENBLAS/MKL/RAYON_NUM_THREADS=1`, capture env metadata into `benchmarks/results/env.json` | BLAS backends confirmed; env capture commands documented |
| BENCH-05 | Numeric-agreement gate — sign-aligned FPCA correlation > 0.99; 1e-3 depth/regression; 1e-2 elastic warping — abort/flag on mismatch before any timing | Sign convention verified by code inspection; tolerance rationale documented; per-op agreement approach |

</phase_requirements>

## Summary

Phase 112 delivers a pure infrastructure milestone: an external `benchmarks/` directory (not a Cargo workspace member) that wires together three ecosystem runners (Rust via criterion, Python via pytest-benchmark, R via the `bench` package) on identical seeded CSV inputs, with a mandatory numeric-agreement gate before any timing is recorded. The dominant implementation risks are methodological, not technical: thread-count fairness, sign-convention alignment across FPCA implementations, and elastic cost-function equivalence. All three risks are resolvable by code inspection and explicit parameter pinning — this research documents the concrete answers the planner needs.

The `fdata_to_pc` sign convention is fully verified by code inspection: each loading column is flipped so its max-abs element is strictly positive, and both faer and nalgebra SVD paths apply the identical `fix_svd_signs` function. The R `pca.fd` (package `fda`) produces the same convention (max-abs element positive) on the same test data. The elastic cost function is the Fisher-Rao (SRSF-based) metric in both fdars and `fdasrvf::pair_align_functions`; `lambda=0.0` on both sides removes the warp-penalty term, making the two directly comparable. The criterion output format is fully confirmed by live file inspection: `mean.point_estimate` in nanoseconds at path `target/criterion/<group>/<bench>/new/estimates.json` — with one subtlety: BenchmarkId groups use slash-separated subdirectories (e.g., `elastic_align_pair/m/50/new/estimates.json`), while plain `bench_function` names produce flat paths.

**Primary recommendation:** Build the harness as a single Python orchestration script (`benchmarks/run_all.py`) that (1) generates CSV fixtures, (2) runs all three runners with explicit thread-pin env vars, (3) runs the numeric-agreement gate, and (4) collects results into structured JSON. The R and Python runners are separate scripts (`benchmarks/r_runner.R`, `benchmarks/python_runner.py`); the Rust side is parsed from the existing criterion output.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Seeded data generator | Python (orchestrator) | — | Single source of truth; Python's `numpy.random.default_rng(seed)` produces deterministic IEEE-754 floats written as full-precision CSV |
| CSV data consumption | All three runners | — | Each runner reads shared CSV independently; spot-check verifies bit-identity |
| fdars timing | Rust / criterion output files | Python orchestrator parses | No new Rust code; criterion 0.5 already writes `estimates.json` |
| Python timing | Python runner / pytest-benchmark | — | `pytest-benchmark` or `timeit`-based loop around scikit-fda calls |
| R timing | R runner / `bench::mark()` | — | `bench` package provides consistent ns-resolution timing |
| Numeric-agreement gate | Python orchestrator | — | Easiest to implement cross-language comparison in Python; reads all three outputs |
| Thread-count pinning | OS environment variables | Orchestrator sets before fork | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `RAYON_NUM_THREADS` |
| env.json capture | Python orchestrator | — | subprocess calls to `python3 --version`, `Rscript --version`, BLAS introspection |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-fda | 0.10.1 | Python FDA operations | Locked decision; pinned version |
| pytest-benchmark | ~4.0 | Python timing | `requirements.lock` compatible; ns-resolution |
| fda (R) | 6.3.0 installed | R FPCA, basis smoothing | Canonical R FDA package |
| fda.usc (R) | 2.2.0 installed | R FPCA, depth, FPCR | Covers depth.FM, depth.RP, fregre.pc |
| roahd (R) | 1.4.3 installed | R functional boxplot (MBD+fbplot) | fbplot() is the standard R functional boxplot |
| fdasrvf (R) | 2.4.4 installed | R elastic alignment + Karcher mean | time_warping(), pair_align_functions() |
| renv (R) | NOT INSTALLED | R lockfile capture | Needed; must be installed via `renv::install()` |
| bench (R) | NOT INSTALLED | R timing | Needed; install via `install.packages("bench")` |
| refund (R) | NOT INSTALLED | R FPCR regression | Needed for FPCR; install via `install.packages("refund")` |
| fdapace (R) | NOT INSTALLED | R FPCA alternative | Needed for Phase 113 coverage; install via `install.packages("fdapace")` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 2.3.5 (installed) | CSV generation, numeric comparison in gate | Always — data generation and agreement gate |
| scipy | 1.17.0 (installed) | Pearson correlation in FPCA gate | Agreement gate: `scipy.stats.pearsonr` |
| pandas | current | CSV I/O in Python runner | Reading shared CSV, emitting timing JSON |

**Installation (R — missing packages):**
```bash
Rscript -e "install.packages(c('renv', 'bench', 'refund', 'fdapace'), repos='https://cloud.r-project.org')"
```

**Installation (Python — in virtualenv):**
```bash
python3 -m venv benchmarks/.venv
benchmarks/.venv/bin/pip install scikit-fda==0.10.1 pytest-benchmark numpy scipy pandas
benchmarks/.venv/bin/pip freeze > benchmarks/requirements.lock
```

**Version verification:**
```bash
# R installed packages (verified 2026-09-20):
# fda 6.3.0, fda.usc 2.2.0, roahd 1.4.3, fdasrvf 2.4.4
# renv, bench, refund, fdapace: NOT INSTALLED — must be installed before Phase 113

# Python:
# scikit-fda: NOT INSTALLED — must be installed via venv
# numpy 2.3.5 ✓, scipy 1.17.0 ✓ (system Python 3.14.7)
```

## Package Legitimacy Audit

> All packages in this phase are well-established FDA ecosystem libraries verified via the live R environment and PyPI. No new Cargo packages are introduced.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| scikit-fda 0.10.1 | PyPI | ~5 yrs | High (FDA reference) | github.com/GAA-UAM/scikit-fda | OK | Approved |
| pytest-benchmark | PyPI | ~10 yrs | Very high | github.com/ionelmc/pytest-benchmark | OK | Approved |
| fda (R) | CRAN | ~20 yrs | Very high | — | OK | Approved — installed 6.3.0 |
| fda.usc (R) | CRAN | ~10 yrs | High | — | OK | Approved — installed 2.2.0 |
| roahd (R) | CRAN | ~8 yrs | Medium | — | OK | Approved — installed 1.4.3 |
| fdasrvf (R) | CRAN | ~8 yrs | Medium | github.com/jdtuck/fdasrvf_R | OK | Approved — installed 2.4.4 |
| renv (R) | CRAN | ~6 yrs | Very high | github.com/rstudio/renv | OK | Approved — needs install |
| bench (R) | CRAN | ~6 yrs | High | github.com/r-lib/bench | OK | Approved — needs install |
| refund (R) | CRAN | ~10 yrs | High | — | OK | Approved — needs install |
| fdapace (R) | CRAN | ~8 yrs | Medium | — | OK [ASSUMED] | Approved — needs install |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

---

## Architecture Patterns

### System Architecture Diagram

```
  ┌─────────────────────────────────────────────────────┐
  │             benchmarks/run_all.py (orchestrator)     │
  │  sets: OMP/OPENBLAS/MKL/RAYON_NUM_THREADS=1         │
  └────┬──────────┬──────────┬──────────────────────────┘
       │          │          │
       ▼          │          │
  ┌────────────┐  │          │
  │  generate  │  │          │
  │  CSV data  │  │          │
  │  (numpy    │  │          │
  │   seeded)  │  │          │
  └────┬───────┘  │          │
       │          │          │
       ▼          ▼          ▼
  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐
  │ Rust     │ │ Python   │ │ R runner             │
  │ criterion│ │ runner   │ │ (r_runner.R)         │
  │ output   │ │ (py_     │ │ fda/fda.usc/roahd/   │
  │ (pre-run │ │ runner.  │ │ fdasrvf              │
  │ or       │ │ py)      │ │                      │
  │ trigger  │ │ scikit-  │ │ bench::mark()        │
  │ cargo    │ │ fda 0.10 │ │                      │
  │ bench)   │ │ pytest-  │ │ outputs timing JSON  │
  │          │ │ benchmark│ │                      │
  └────┬─────┘ └────┬─────┘ └──────────┬───────────┘
       │             │                  │
       ▼             ▼                  ▼
  ┌──────────────────────────────────────────────────┐
  │     numeric-agreement gate (gate.py)              │
  │  - FPCA: sign-align loadings, Pearson corr > 0.99│
  │  - Depth/Regression: max abs diff < 1e-3          │
  │  - Elastic: warp L∞ < 1e-2                       │
  │  ABORT if mismatch; else proceed                  │
  └──────────────────┬───────────────────────────────┘
                     │
                     ▼
              ┌─────────────────┐
              │ benchmarks/     │
              │ results/        │
              │ ├── env.json    │
              │ ├── rust.json   │
              │ ├── python.json │
              │ └── r.json      │
              └─────────────────┘
```

### Recommended Project Structure
```
benchmarks/
├── run_all.py             # master orchestrator; sets thread env vars; calls gate
├── generate_data.py       # seeded curve-set generator; writes CSV to data/
├── python_runner.py       # scikit-fda 0.10.1 timing; reads CSV; emits results/python.json
├── r_runner.R             # R ecosystem timing; reads CSV; emits results/r.json
├── gate.py                # numeric-agreement gate; reads all three result JSONs
├── collect_rust.py        # parses target/criterion/*/new/estimates.json -> results/rust.json
├── data/                  # generated CSV fixtures (gitignored or small)
│   ├── curves_n50_m50.csv
│   ├── curves_n100_m100.csv
│   └── ...
├── results/               # output (gitignored during runs, committed after baseline)
│   ├── env.json
│   ├── rust.json
│   ├── python.json
│   └── r.json
├── requirements.lock      # pinned Python deps (pip freeze)
└── renv.lock              # pinned R deps (renv::snapshot())
```

### Pattern 1: Seeded CSV Generator
**What:** Python script generates `n` curves on an `m`-point uniform grid using `numpy.random.default_rng(seed)`, writes as CSV with full float64 precision.
**When to use:** Once per harness run; all three runners read the same files.
**Example:**
```python
# Source: numpy documentation / project convention
import numpy as np
import pandas as pd

def generate_curves(n: int, m: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    argvals = np.linspace(0, 1, m)
    # Each curve: random amplitude*sin + noise (matches fdars test patterns)
    amplitudes = rng.uniform(0.5, 2.0, size=n)
    phases = rng.uniform(0, 2 * np.pi, size=n)
    noise_scale = rng.uniform(0.0, 0.2, size=n)
    curves = np.array([
        amplitudes[i] * np.sin(2 * np.pi * argvals + phases[i])
        + noise_scale[i] * rng.standard_normal(m)
        for i in range(n)
    ])
    # CSV: first row = argvals header, then one row per curve
    df = pd.DataFrame(curves, columns=[f"t_{j}" for j in range(m)])
    df.insert(0, "curve_id", range(n))
    return df

# Write with full precision
df.to_csv(f"data/curves_n{n}_m{m}.csv", index=False, float_format="%.17g")
```

### Pattern 2: Thread-Pinning Env Vars
**What:** Set all four thread-count env vars before launching any subprocess.
**When to use:** All runner invocations in `run_all.py`.
**Example:**
```python
# Source: numpy/OpenBLAS/Intel MKL documentation [ASSUMED]
import os
import subprocess

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
env = {**os.environ, **THREAD_ENV}

subprocess.run(["Rscript", "r_runner.R"], env=env, check=True)
subprocess.run(["python3", "python_runner.py"], env=env, check=True)
# Rust: run `cargo bench` with env vars set before parsing output
subprocess.run(["cargo", "bench", "-p", "fdars-core", "--features", "linalg,parallel",
                "--bench", "regression_benchmarks", "--bench", "alignment_benchmarks",
                "--bench", "depth_benchmarks"], env=env, cwd="..", check=True)
```

### Pattern 3: Criterion Output Parsing
**What:** Parse `target/criterion/<group>/<bench>/new/estimates.json` to extract `mean.point_estimate` (nanoseconds). Handle both flat and BenchmarkId subdirectory layouts.
**When to use:** `collect_rust.py`; called after `cargo bench` completes.
**Example:**
```python
# Source: Live file inspection of target/criterion/ — confirmed layout [VERIFIED: /home/simonm/projects/rust/fdars/target/criterion/audit_fpca/n500_m200/new/estimates.json]
import json
import pathlib
from typing import Iterator

CRITERION_DIR = pathlib.Path("../target/criterion")

def iter_estimates() -> Iterator[dict]:
    """Yield {group, bench, mean_ns, ci_lower_ns, ci_upper_ns} for every new/estimates.json."""
    for est_path in CRITERION_DIR.rglob("new/estimates.json"):
        parts = est_path.relative_to(CRITERION_DIR).parts
        # parts example: ("elastic_align_pair", "m", "50", "new", "estimates.json")
        # or:            ("audit_fpca", "n500_m200", "new", "estimates.json")
        group = parts[0]
        bench = "/".join(parts[1:-2])  # everything between group and "new"
        with est_path.open() as f:
            data = json.load(f)
        mean = data["mean"]
        yield {
            "group": group,
            "bench": bench,
            "mean_ns": mean["point_estimate"],
            "ci_lower_ns": mean["confidence_interval"]["lower_bound"],
            "ci_upper_ns": mean["confidence_interval"]["upper_bound"],
            "std_err_ns": mean["standard_error"],
        }
```

**Key detail confirmed by live inspection:** The `estimates.json` schema is:
```json
{
  "mean": {
    "point_estimate": 16390770.837,   // nanoseconds (f64)
    "confidence_interval": {"lower_bound": ..., "upper_bound": ...},
    "standard_error": ...
  },
  "median": {...},
  "std_dev": {...},
  "slope": {...},
  "median_abs_dev": {...}
}
```
[VERIFIED: `/home/simonm/projects/rust/fdars/target/criterion/audit_fpca/n500_m200/new/estimates.json`]

### Pattern 4: FPCA Sign-Alignment Gate
**What:** For each component k, multiply the fdars loading column by `sign(pearson(fdars_loading_k, other_loading_k))`, then compute correlation. If any component's |corr| < 0.99, abort.
**When to use:** Numeric-agreement gate, FPCA block.
**Example:**
```python
# Source: derived from code inspection of fix_svd_signs (regression.rs:288-301)
import numpy as np
from scipy.stats import pearsonr

def sign_align_and_check_fpca(
    fdars_loadings: np.ndarray,   # shape (m, ncomp)
    other_loadings: np.ndarray,   # shape (m, ncomp)
    threshold: float = 0.99,
) -> bool:
    """Returns True if all components pass the correlation threshold."""
    ncomp = fdars_loadings.shape[1]
    for k in range(ncomp):
        l1 = fdars_loadings[:, k]
        l2 = other_loadings[:, k]
        # Sign-align l2 to l1
        if np.dot(l1, l2) < 0:
            l2 = -l2
        corr, _ = pearsonr(l1, l2)
        if abs(corr) < threshold:
            return False
    return True
```

### Anti-Patterns to Avoid
- **Running Python before pinning threads:** scikit-fda's numpy backend auto-uses all cores. Always set `OMP_NUM_THREADS=1` etc. *in the subprocess environment*, not just the parent process.
- **Comparing raw FPCA loadings without sign alignment:** SVD sign is arbitrary; a correlation of -1.0 means identical, not opposite. Always sign-align before computing the gate metric.
- **Using `elapsed_time` from bench::mark in R without specifying unit:** `bench::mark()` returns a tibble; extract `median` in nanoseconds via `as.numeric(result$median)` (the `bench_time` class uses ns internally).
- **Parsing criterion output from `base/` instead of `new/`:** The `base/` subdirectory stores the previous run for change detection. Always read from `new/estimates.json` for current timing.
- **Triggering `cargo bench` without `--features linalg,parallel`:** The production timing path uses faer SVD (under `linalg`) and rayon (under `parallel`). Omitting features gives a different, slower code path. [VERIFIED: CLAUDE.md project constraints]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| R timing | Custom `proc.time()` wrapper | `bench::mark()` | bench handles warmup, multiple iterations, gc, and produces consistent ns-resolution output |
| Python timing | `time.perf_counter()` loops | `pytest-benchmark` or `timeit.timeit()` | Consistent handling of warmup iterations; pytest-benchmark integrates with CI |
| Lockfile management | Manual dep version pinning | `pip freeze > requirements.lock` + `renv::snapshot()` | Proper lockfile semantics; renv handles transitive deps |
| BLAS detection (Python) | Parsing linker output | `numpy.__config__.blas_opt_info` / `np.show_config()` | Built-in numpy introspection |
| BLAS detection (R) | Parsing `sessionInfo()` strings | `sessionInfo()$BLAS` | Standard R mechanism |
| Sign ambiguity | Custom SVD post-processing | fdars `fix_svd_signs` convention (max-abs positive) | Already implemented; gate must match, not replicate |

---

## Research Focus Findings

### 1. fdars FPCA Sign-Canonicalization Convention

[VERIFIED: `/home/simonm/projects/rust/fdars/fdars-core/src/regression.rs:253-300`]

The sign convention is implemented in two functions:

**`dominant_sign_negative` (lines 266-276):** For column `k` of a matrix, finds the row index `j_max` with the largest absolute value (`max_by` on `.abs().partial_cmp()`), and returns `true` iff `col[(j_max, k)] < 0.0`. Tie-break: `unwrap_or(Equal)` with fallback index `0`.

Verbatim quote:
```rust
pub(crate) fn dominant_sign_negative(col: &FdMatrix, k: usize, nrows: usize) -> bool {
    let j_max = (0..nrows)
        .max_by(|&a, &b| {
            col[(a, k)]
                .abs()
                .partial_cmp(&col[(b, k)].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    col[(j_max, k)] < 0.0
}
```

**`fix_svd_signs` (lines 288-301):** For each component `k`, if `dominant_sign_negative` is true for the rotation column, flip both `rotation[:, k]` and `scores[:, k]` simultaneously.

**Convention in plain language:** For each loading vector (column of the `rotation` matrix), the element with the largest absolute value is made strictly positive. Both the faer and nalgebra SVD paths apply this identical function via `fix_svd_signs(&mut rotation, &mut scores, ncomp)` at line 503, called before the sqrt-weights unscaling step.

**This is applied to the `rotation` matrix (m × ncomp) stored before unscaling.** The sign decision is made on the scaled rotation; the flip is then carried through the unscaling.

**R `pca.fd` (package `fda`) comparison:** Verified empirically on a synthetic sinusoidal dataset (n=20, m=50): `pca.fd` produces loadings where the max-abs element is positive (value `+1.4141` at index 1), matching the fdars convention. [VERIFIED: live R session — both implementations place max-abs element positive on the same test data]

**Gate implementation:** For comparing fdars vs scikit-fda or R FPCA, the agreement gate must:
1. Extract the loading vectors (columns of rotation matrix)
2. Sign-align the reference library's loadings to fdars by flipping columns where `dot(fdars_col, other_col) < 0`
3. Compute Pearson correlation between aligned columns
4. Pass if all `|corr| > 0.99`

### 2. Cross-Language Operation Equivalence

#### FPCA

| Parameter | fdars | scikit-fda 0.10.1 | R |
|-----------|-------|-------------------|---|
| Function | `fdata_to_pc(data, ncomp, argvals)` | `FPCA(n_components=ncomp).fit_transform(fd)` | `pca.fd(fd, nharm=ncomp, centerfns=TRUE)` |
| Integration weights | Simpson's rule via `simpsons_weights(argvals)` | Numerical integration (trapezoid by default) | Numerical integration (quadrature) |
| Centering | Column means (eval-point means) | `FDataGrid` mean centering | `centerfns=TRUE` |
| Sign convention | Max-abs loading element positive | [ASSUMED] sklearn convention (max abs positive — same as fdars by coincidence) | Max-abs positive (verified) |
| Key divergence risk | fdars uses Simpson's weights; scikit-fda default may differ | Tolerance ~1e-2 for scores; >0.99 corr on loadings is achievable | Same as above |

[ASSUMED]: scikit-fda 0.10.1 exact sign convention — not installed to verify; use sign-alignment gate regardless.

#### Basis Smoothing

| Parameter | fdars | scikit-fda 0.10.1 | R |
|-----------|-------|-------------------|---|
| Function | `smooth_basis(data, argvals, fdpar)` | `BasisSmoother(basis=BSplineBasis(...))` | `smooth.basis(argvals, y, fdParobj)` |
| B-spline basis | `bspline_basis(argvals, nknots, order)` | `BSplineBasis(n_basis=nbasis, order=order)` | `create.bspline.basis(c(0,1), nbasis, norder)` |
| Penalty | GCV or fixed lambda | Regularization parameter | `fdPar(basis, Lfdobj, lambda)` |
| Key divergence risk | Different penalty parametrizations; tolerance 1e-3 for smoothed values only applies when using same lambda | Pin lambda=0 (no penalty) for the agreement check; penalized smoothing is NOT numerically equivalent across libraries | [ASSUMED] |

**Recommendation for agreement gate:** Compare smoothed curve values (not coefficients) with lambda=0 (interpolation). Use 1e-3 tolerance on max absolute difference.

#### Functional Depth

| Parameter | fdars | scikit-fda 0.10.1 | R |
|-----------|-------|-------------------|---|
| Fraiman-Muniz | `fraiman_muniz(data, data, integrated=true, dim=Dim::One)` | `ModifiedBandDepth()` or `FramanMunizDepth()` | `depth.FM(fdataobj, fdataori)` — fda.usc |
| Modified Band | `modified_band(data, data, Dim::One)` | `ModifiedBandDepth()` | `MBD(fData(grid, mat))` — roahd |
| Random Projection | `random_projection_1d_seeded(data, data, nproj=50, seed=Some(42))` | [ASSUMED] `RandomProjectionDepth(n_projections=50)` | `depth.RP(fdataobj, nproj=50)` — fda.usc |
| Key divergence risk | Must use `random_projection_1d_seeded` (not `random_projection`) for reproducibility; nproj must match |

**Critical: R `depth.RP` default `nproj=50`** [VERIFIED: live R session — `formals(depth.RP)$nproj = 50`]. fdars `random_projection` takes `nproj: usize` as explicit parameter; use 50.

**Critical: R `depth.RP` default `proj="vexponential"`** — this is the projection distribution. fdars uses Gaussian projections internally. [ASSUMED: these produce equivalent expected depth values at large nproj but may differ at small nproj — use seeded comparison on same data with nproj=200 for the gate.]

**roahd `MBD` returns a `numeric` vector of depth values** [VERIFIED: live R session]. roahd `fbplot()` with `display=FALSE` returns `list(Depth, Fvalue, ID_outliers)` [VERIFIED: live R session].

#### Functional Boxplot

| Parameter | fdars | R (roahd) |
|-----------|-------|-----------|
| Function | `functional_boxplot(data, DepthMethod::ModifiedBand, 1.5)` | `fbplot(fData(grid, mat), Depths="MBD", Fvalue=1.5, display=FALSE)` |
| Returns | `FunctionalBoxplotResult` with `{median, central_lower, central_upper, whisker_lower, whisker_upper, outliers, depths}` | `list(Depth, Fvalue, ID_outliers)` |
| Agreement check | Compare depth vectors (MBD values) with 1e-3 tolerance | — |

[VERIFIED: `src/depth/dispatch.rs:287-329`] fdars `functional_boxplot` signature: `fn functional_boxplot(data: &FdMatrix, method: DepthMethod, factor: f64)`.

**scikit-fda 0.10.1 functional boxplot:** [ASSUMED] `FDataBoxplot` class — not verified since scikit-fda is not installed.

#### FPCR Regression

| Parameter | fdars | scikit-fda 0.10.1 | R |
|-----------|-------|-------------------|---|
| Function | `fregre_lm(data, y, argvals=None, ncomp)` | [ASSUMED] `FPCRRegression(n_components=ncomp)` | `fregre.pc(fdataobj, y, l=1:ncomp)` — fda.usc |
| Approach | FPCA scores → OLS | FPCA scores → OLS | FPCA scores (via fdata2pc) → OLS |
| Agreement check | Compare fitted values (1e-3 tolerance on same ncomp and same FPCA) | — | — |
| Key divergence risk | FPCA weights differ (Simpson's vs trapezoid) → scores differ slightly → fitted values will differ; tolerance 1e-3 may be tight | Use sign-aligned FPCA scores from the gate step as input; check fitted value correlation rather than abs diff |

[VERIFIED: `src/scalar_on_function/fregre_lm.rs:64`] `pub fn fregre_lm(fdataobj: &FdMatrix, y: &[f64], argvals: Option<&[f64]>, ncomp: usize) -> ...`

#### Elastic Alignment / Karcher Mean

| Parameter | fdars | scikit-fda 0.10.1 | R (fdasrvf) |
|-----------|-------|-------------------|-------------|
| Pairwise align | `elastic_align_pair(f1, f2, argvals, lambda=0.0)` | [ASSUMED] `FisherRaoElasticRegistration(penalty=0.0)` | `pair_align_functions(f1, f2, time, lambda=0, pen="roughness", method="DP")` |
| Karcher mean | `karcher_mean(data, argvals, max_iter=20, tol=1e-4, lambda=0.0)` | [ASSUMED] `ElasticRegistration` mean | `time_warping(f, time, lambda=0, max_iter=20)` |
| Cost function | Fisher-Rao metric (SRSF-based, DP solver) | Fisher-Rao metric (SRSF-based) | Fisher-Rao metric (SRSF-based, DP solver) |
| Lambda meaning | Penalty on warp deviation from identity (0.0 = no penalty) | Penalty parameter (0.0 = no penalty) | `lambda=0` — same meaning |

[VERIFIED: `src/alignment/pairwise.rs:13-16`] fdars uses Fisher-Rao (SRSF-based) metric:
```
/// Align curve f2 to curve f1 using the elastic framework.
/// Computes the optimal warping γ such that f2∘γ is as close as possible
/// to f1 in the elastic (Fisher-Rao) metric.
```

[VERIFIED: live R session — `formals(pair_align_functions)$lambda = 0`, `$pen = "roughness"`, `$method = "DP"`]

**Cost function alignment: lambda=0.0 on both sides removes the warp-penalty term; both use Fisher-Rao (SRSF) distance with DP alignment. These are equivalent algorithms and should produce numerically close results (1e-2 tolerance on warping functions is appropriate given different DP grid implementations).

**Karcher mean divergence risk:** `time_warping` in fdasrvf defaults `max_iter=20` — same as fdars default. Both converge to the same Karcher mean but may differ at intermediate convergence states. Run both to convergence with the same tolerance for the gate.

### 3. Existing Criterion Harness Coverage

**Confirmed by live inspection of `target/criterion/` directory:**

Operations with existing criterion benchmarks that map to overlapping ops:

| Overlapping Op | Existing Criterion Group | Bench IDs | Notes |
|----------------|--------------------------|-----------|-------|
| FPCA | `audit_fpca` | `n500_m200` | Only one size; Phase 112 needs smaller sizes for cross-lang comparison |
| FPCA | `audit_p4_fpca` | `n100_m50`, `n100_m200`, `n500_m50`, `n500_m200`, `n1000_m50`, `n1000_m200` | Good coverage |
| FPCA | `fdata_to_pc` (regression_benchmarks) | via `audit_fpca` | Yes, exists |
| Elastic align pair | `elastic_align_pair` | `m/50`, `m/100`, `m/200`, `m_band0.1/50..200`, `m100_lambda0.1` | Pairwise — good |
| Karcher mean | `karcher_mean` | `n20_m50`, `n10_m50` | Only small sizes; need N≥100 for Phase 113 |
| Karcher mean | `audit_p3_karcher` | `n100_m50`, `n100_m200`, `n500_m50`, `n500_m200` | Good |
| Functional depth (FM) | `audit_depth` | `n500_m200` | Fraiman-Muniz depth (via `streaming_depth`) |
| Functional depth | `audit_streaming_depth` | `n500_m200` | Streaming FM |
| Basis smoothing | `audit_smooth` | `n500_m200` | Kernel smoother (Nadaraya-Watson) — NOT basis smoothing |
| FPCR regression | — | — | **MISSING**: `fregre_lm` has no criterion bench in any current group |
| Functional boxplot | — | — | **MISSING**: `functional_boxplot` has no criterion bench |
| Random projection depth | `depth_benchmarks` (via `random_projection`) | has results? | bench file has `bench_random_projection` but no `target/criterion/random_projection` dir found |

[VERIFIED: `ls /home/simonm/projects/rust/fdars/target/criterion/`]

**Missing benchmarks (must add new criterion benches for complete Phase 113 coverage):**
1. `fregre_lm` (FPCR regression) — no existing criterion bench group
2. `functional_boxplot` — no existing criterion bench group
3. `random_projection` depth — no `target/criterion/random_projection` output found (bench file exists but hasn't been run recently)
4. `smooth_basis` / B-spline basis smoothing — `audit_smooth` covers kernel smoothing, not basis smoothing

**These new benches are added in `benchmarks/` as standalone scripts that invoke `cargo bench`, NOT as new `fdars-core/benches/` files. The fdars-core bench files need the new groups added** (since criterion collects timings only for groups that exist in `fdars-core/benches/`). However, the CONTEXT.md says "no new dev-dependency" — adding a new bench group to an existing `.rs` bench file in `fdars-core/benches/` is NOT a new dev-dependency.

**Recommended approach:** Add `fregre_lm`, `functional_boxplot`, and `random_projection` bench groups to existing bench files in `fdars-core/benches/` (e.g., `regression_benchmarks.rs` already has `bench_fregre_lm`). The `random_projection` bench is already in `depth_benchmarks.rs` — it just needs to be run. `functional_boxplot` needs a new bench function added to `depth_benchmarks.rs`.

### 4. Fairness Controls and env.json

**Thread-count pinning required vars:**
- `OMP_NUM_THREADS=1` — OpenMP (affects LAPACK/BLAS threaded routines, R's reference BLAS)
- `OPENBLAS_NUM_THREADS=1` — OpenBLAS specifically
- `MKL_NUM_THREADS=1` — Intel MKL
- `RAYON_NUM_THREADS=1` — Rust rayon (affects fdars parallel feature)
- `NUMEXPR_NUM_THREADS=1` — numpy numexpr (less critical)
- `VECLIB_MAXIMUM_THREADS=1` — macOS vecLib (not relevant on Linux but harmless)

**Environment on this machine (2026-09-20):**
- CPU: 13th Gen Intel Core i9-13900H, 20 logical cores [VERIFIED: `/proc/cpuinfo`]
- OS: Manjaro Linux 6.18.45-1 [VERIFIED: env context]
- Python: 3.14.7 [VERIFIED: `python3 --version`]
- numpy: 2.3.5 [VERIFIED: `python3 -c "import numpy; print(numpy.__version__)"`]
- numpy BLAS: scipy-openblas (OpenBLAS 0.3.30, 64-bit int, dynamic arch) [VERIFIED: `numpy.show_config()`]
- R: 4.6.1 (2026-06-24) [VERIFIED: `Rscript --version`]
- R BLAS: `/usr/lib/libblas.so.3.12.0` (reference BLAS) [VERIFIED: `sessionInfo()$BLAS`]
- R LAPACK: `/usr/lib/liblapack.so.3.12.0`, version 3.12.0 [VERIFIED: `sessionInfo()`]
- Rust: 1.97.0 (from CLAUDE.md)

**env.json schema:**
```json
{
  "timestamp": "2026-09-20T...",
  "cpu": {"model": "13th Gen Intel Core i9-13900H", "cores_logical": 20},
  "rust": {"version": "1.97.0", "features": "linalg,parallel"},
  "python": {"version": "3.14.7", "numpy_version": "2.3.5", "numpy_blas": "scipy-openblas/OpenBLAS-0.3.30"},
  "scikit_fda": {"version": "0.10.1"},
  "r": {"version": "4.6.1", "blas": "/usr/lib/libblas.so.3.12.0", "lapack_version": "3.12.0"},
  "r_packages": {"fda": "6.3.0", "fda.usc": "2.2.0", "roahd": "1.4.3", "fdasrvf": "2.4.4", "refund": "...", "fdapace": "..."},
  "thread_env": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"},
  "criterion_version": "0.5"
}
```

**BLAS detection commands:**
```python
# Python: get numpy BLAS backend
import numpy
import io
f = io.StringIO()
numpy.show_config()  # prints to stdout — capture with subprocess or redirect
# Or: numpy.__config__.blas_opt_info (older numpy)

# R: get BLAS path
Rscript -e "cat(sessionInfo()$BLAS, '\n')"
```

### 5. Agreement Tolerance Validation

#### FPCA Loadings Correlation > 0.99
**Rationale:** Functional loadings from SVD-based FPCA on the same data should be nearly identical. The dominant source of divergence is the integration-weight scheme (Simpson's vs trapezoid). On a uniform grid with m=50, the Simpson's vs trapezoid difference in the weighted covariance is O(h²) ≈ O(1/m²) ≈ 4e-4 relative error. This propagates to the loadings at roughly the same relative scale. A Pearson correlation of 0.99 allows for ~14% absolute angle difference, which is far more than numerical differences can produce between correctly implemented FPCA algorithms. [ASSUMED: tolerance analysis based on numerical integration theory]

#### Depth/Regression 1e-3
**Rationale:** Fraiman-Muniz depth is computed as an empirical mean of indicator functions; on n=100 curves, the discrete approximation error is O(1/n) = 1e-2. The 1e-3 tolerance assumes both implementations use the same definition. If they do not (e.g., different normalization), this will flag — which is correct behavior (flag then investigate). [ASSUMED]

For FPCR fitted values, 1e-3 in the response scale is appropriate when both sides use the same sign-aligned FPC scores. [ASSUMED]

#### Elastic Warping 1e-2
**Rationale:** The DP grid resolution is O(1/m) ≈ 2e-2 for m=50. Two DP implementations on identical input should agree to within one grid step. 1e-2 is appropriate for m=50; it may be too tight for m=20 or too loose for m=200. [ASSUMED: DP grid error analysis]

**Practical recommendation:** For the harness gate, use L∞ norm on the warping functions (max absolute deviation `max|γ_fdars(t) - γ_R(t)|`). Karcher mean warping comparison should use average L2 norm across curves.

---

## Common Pitfalls

### Pitfall 1: Thread-Count Mismatch on Python Side
**What goes wrong:** numpy/scipy auto-uses all cores via OpenBLAS (detected: 20 cores). A single benchmark call may use all 20 cores for LAPACK, giving 10–20× faster times vs Rust's sequential path.
**Why it happens:** numpy links against OpenBLAS which reads `OMP_NUM_THREADS` at fork time, not at import time.
**How to avoid:** Set `OMP_NUM_THREADS=1` etc. *before* launching the Python subprocess. Setting it within Python after import does NOT work for already-linked BLAS.
**Warning signs:** Python timings for FPCA on n=500, m=200 are significantly faster than Rust timings.

### Pitfall 2: FPCA Sign Flip Causing False Gate Failure
**What goes wrong:** FPCA loadings have sign ambiguity; direct comparison yields correlation -1.0, tripping the gate with a false failure.
**Why it happens:** Different SVD implementations choose arbitrary sign; fdars applies `fix_svd_signs` but other libraries may not use the same convention.
**How to avoid:** Always sign-align before correlating: if `dot(fdars_col, other_col) < 0`, flip `other_col`. Then compute correlation.
**Warning signs:** Gate reports correlation = -0.99x (near -1, not +1).

### Pitfall 3: Criterion `new/` vs `base/` Confusion
**What goes wrong:** Parsing `base/estimates.json` (previous run) instead of `new/estimates.json` (current run).
**Why it happens:** Criterion writes both for change detection. The directory structure is `<group>/<bench>/new/` and `<group>/<bench>/base/`.
**How to avoid:** Always use `rglob("new/estimates.json")` pattern, not `rglob("estimates.json")`.
**Warning signs:** Rust timings look stale (same as previous run after a rerun).

### Pitfall 4: Criterion BenchmarkId Slash-Separated Paths
**What goes wrong:** Bench ID `BenchmarkId::new("m", 50)` produces path `<group>/m/50/new/estimates.json`, not `<group>/m_50/new/estimates.json`. Flat `bench_function("n20_m50", ...)` produces `<group>/n20_m50/new/estimates.json`.
**Why it happens:** Criterion uses the BenchmarkId parameter as a subdirectory component.
**How to avoid:** Use `rglob("new/estimates.json")` and reconstruct the bench name from relative path parts, not by guessing.
**Warning signs:** Parser finds some groups but misses BenchmarkId-parameterized groups.
[VERIFIED: live inspection of `target/criterion/elastic_align_pair/m/50/new/estimates.json`]

### Pitfall 5: fda.usc `depth.RP` vs fdars `random_projection` Projection Distribution
**What goes wrong:** `depth.RP` default `proj="vexponential"` (exponential projection vectors) while fdars uses Gaussian projections. Even with the same `nproj=50`, results differ due to the projection distribution.
**Why it happens:** Different communities made different implementation choices.
**How to avoid:** Use a large `nproj` (e.g., 200) for the agreement check and accept a looser tolerance (1e-2) for random projection depth, or skip random projection depth from the gate and use Fraiman-Muniz (which is deterministic and equivalent across libraries).
**Warning signs:** Random projection depth values differ by more than 1e-3 consistently.

### Pitfall 6: R reference BLAS is Single-Threaded by Default
**What goes wrong:** R's BLAS (`/usr/lib/libblas.so.3.12.0`) is the reference BLAS — single-threaded by default. Setting `OMP_NUM_THREADS=1` has no effect because R isn't multi-threaded through BLAS anyway on this machine. However, setting it is still correct practice (guards against alternative BLAS installs) and is required for `env.json` documentation.
**Why it happens:** Manjaro defaults to reference BLAS, not OpenBLAS, for R.
**How to avoid:** Document this in `env.json`; the R timings are effectively already single-threaded.
**Warning signs:** N/A — this is actually fine; just document it.

---

## Code Examples

### Complete Agreement Gate (FPCA Section)
```python
# Source: derived from regression.rs:253-300 sign convention
import numpy as np
from scipy.stats import pearsonr

def check_fpca_agreement(fdars_loadings, other_loadings, threshold=0.99, label=""):
    """
    fdars_loadings: np.ndarray shape (m, ncomp), rotation matrix from fdata_to_pc
    other_loadings: np.ndarray shape (m, ncomp), from scikit-fda or R
    Returns: (passed: bool, min_corr: float, details: list)
    """
    ncomp = fdars_loadings.shape[1]
    details = []
    min_corr = 1.0
    for k in range(ncomp):
        l1 = fdars_loadings[:, k]
        l2 = other_loadings[:, k]
        if np.dot(l1, l2) < 0:
            l2 = -l2  # sign-align
        corr, _ = pearsonr(l1, l2)
        details.append({"component": k, "correlation": corr})
        min_corr = min(min_corr, abs(corr))
    passed = min_corr >= threshold
    if not passed:
        print(f"GATE FAIL [{label}] FPCA: min |corr| = {min_corr:.4f} < {threshold}")
    return passed, min_corr, details
```

### Criterion Output Collection
```python
# Source: confirmed from live file inspection
# [VERIFIED: /home/simonm/projects/rust/fdars/target/criterion/elastic_align_pair/m/50/new/estimates.json]
import json, pathlib

def collect_criterion_results(criterion_dir: pathlib.Path) -> list[dict]:
    results = []
    for est_path in criterion_dir.rglob("new/estimates.json"):
        rel = est_path.relative_to(criterion_dir)
        parts = rel.parts  # e.g. ("audit_fpca", "n500_m200", "new", "estimates.json")
        group = parts[0]
        bench = "/".join(parts[1:-2])  # join middle parts
        data = json.loads(est_path.read_text())
        m = data["mean"]
        results.append({
            "group": group, "bench": bench,
            "mean_ns": m["point_estimate"],
            "ci_lower_ns": m["confidence_interval"]["lower_bound"],
            "ci_upper_ns": m["confidence_interval"]["upper_bound"],
        })
    return results
```

### R Timing Template
```r
# Source: bench package documentation [ASSUMED]
library(bench)
library(fda)
library(fdasrvf)

run_timings <- function(csv_path, output_path) {
  # Load data
  df <- read.csv(csv_path)
  n <- nrow(df); m <- ncol(df) - 1
  mat <- as.matrix(df[, -1])
  grid <- seq(0, 1, length.out = m)
  
  results <- list()
  
  # FPCA
  basis <- create.bspline.basis(c(0, 1), nbasis = min(20, m))
  fd_smooth <- smooth.basis(grid, t(mat), basis)$fd
  t_fpca <- bench::mark(
    pca.fd(fd_smooth, nharm = 5, centerfns = TRUE),
    iterations = 10, memory = FALSE
  )
  results$fpca <- list(
    op = "fpca", n = n, m = m,
    mean_ns = as.numeric(t_fpca$median)  # bench::mark uses ns internally
  )
  
  # Write JSON
  jsonlite::write_json(results, output_path, auto_unbox = TRUE)
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual timing with `proc.time()` in R | `bench::mark()` with ns resolution | ~2020 | Consistent warmup, GC isolation |
| Shared `.npy` format for cross-lang data | CSV as default (`.npy` optional) | Phase 112 decision | R doesn't need `RcppCNPy`; simpler setup |
| faer SVD path not in criterion | faer path now default under `linalg` | v0.44.0 (Phase 94+) | Existing `audit_p4_fpca` benchmarks use faer path |
| `fdata_to_pc_1d` (old name) | `fdata_to_pc` (current public name) | v0.42.0 API unification | Use `fdata_to_pc` in all new code |

**Deprecated/outdated:**
- `fdata_to_pc_1d`: Renamed to `fdata_to_pc` in the v0.42.0 API unification; the old name is a backward-compat alias. Use `fdata_to_pc` in the harness.
- Parsing `base/estimates.json`: Always use `new/estimates.json` for current measurements.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | scikit-fda 0.10.1 FPCA sign convention is max-abs positive (same as fdars) | Cross-language op table: FPCA | Gate may produce false failures; mitigated by always doing sign-alignment regardless |
| A2 | scikit-fda 0.10.1 uses `FisherRaoElasticRegistration` class for elastic alignment | Cross-language op table: Elastic | Wrong class name → Python runner fails; resolve at install time |
| A3 | `bench::mark()` returns a tibble where `$median` is a `bench_time` object extractable as ns via `as.numeric()` | R Timing Template | Timing extraction wrong → R results in wrong unit |
| A4 | fdapace CRAN package installs cleanly on Manjaro R 4.6.1 | Package legitimacy | Could fail to install; fallback = skip fdapace from R runner |
| A5 | fda.usc random projection depth `proj="vexponential"` produces results within 1e-2 of fdars Gaussian projection at nproj=200 | Agreement tolerance | Gate may not be passable for random projection depth; mitigation = exclude random projection depth from gate, include only FM and MBD |
| A6 | 1e-2 warp tolerance is achievable between fdars DP and fdasrvf DP at lambda=0 | Elastic agreement tolerance | Gate always fails; would block Phase 113; mitigation = verify on reference dataset during Phase 112 |
| A7 | Python `pytest-benchmark` version ~4.0 is compatible with scikit-fda 0.10.1 requirements | Package stack | Dependency conflict; fallback = use `timeit` loops instead of pytest-benchmark |

---

## Open Questions

1. **scikit-fda 0.10.1 exact elastic API class name**
   - What we know: Fisher-Rao elastic registration exists in scikit-fda; the milestone context names it `FisherRaoElasticRegistration`
   - What's unclear: Exact module path (`skfda.preprocessing.registration.FisherRaoElasticRegistration` or similar) and whether Karcher mean is a separate class
   - Recommendation: Verify at Python environment setup time; the task "verify elastic cost-function equivalence on a reference dataset" in BENCH-05 covers this

2. **Whether to add new bench functions to `fdars-core/benches/` for FPCR and functional_boxplot**
   - What we know: These ops have no existing criterion output; `fregre_lm` already has a bench function in `regression_benchmarks.rs` but it hasn't been run recently (no output in `target/criterion/`)
   - What's unclear: Whether the CONTEXT "no new dev-dependency" prohibition also covers adding new bench functions to existing bench files (it shouldn't — these are already in the file)
   - Recommendation: Add `functional_boxplot` bench group to `depth_benchmarks.rs`; run `regression_benchmarks.rs` to populate criterion output for `fregre_lm`

3. **fdasrvf Karcher mean function name**
   - What we know: `time_warping` aligns n curves to a Karcher mean; `multiple_align_functions` takes an explicit mu
   - What's unclear: Whether `time_warping` IS the Karcher mean computation or just returns aligned curves
   - Recommendation: Use `time_warping` with `centroid_type="mean"` (default) — it computes the Karcher mean internally and returns it

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3 | Generator, gate, Python runner | ✓ | 3.14.7 | — |
| numpy | Generator, gate | ✓ | 2.3.5 | — |
| scipy | Agreement gate (pearsonr) | ✓ | 1.17.0 | — |
| scikit-fda | Python runner | ✗ | — | Must install 0.10.1 in venv; blocking |
| pytest-benchmark | Python runner | ✗ | — | Fallback: timeit-based loops |
| R | R runner | ✓ | 4.6.1 | — |
| fda (R) | R FPCA, basis smoothing | ✓ | 6.3.0 | — |
| fda.usc (R) | R depth, FPCR | ✓ | 2.2.0 | — |
| roahd (R) | R functional boxplot | ✓ | 1.4.3 | — |
| fdasrvf (R) | R elastic alignment | ✓ | 2.4.4 | — |
| renv (R) | Lockfile snapshot | ✗ | — | Must install; blocking for BENCH-02 |
| bench (R) | R timing | ✗ | — | Must install; fallback: microbenchmark |
| refund (R) | R FPCR (alternative) | ✗ | — | fda.usc::fregre.pc suffices |
| fdapace (R) | R FPCA alternative | ✗ | — | fda::pca.fd suffices |
| cargo + fdars-core | Rust criterion | ✓ | Rust 1.97.0 | — |

**Missing dependencies with no fallback:**
- scikit-fda 0.10.1 (BENCH-02 Python runner) — must install in venv before any Python runner task
- renv (R) — needed for `renv::snapshot()` to produce `renv.lock`

**Missing dependencies with fallback:**
- pytest-benchmark → timeit-based loops
- bench (R) → microbenchmark package
- refund → fda.usc::fregre.pc
- fdapace → fda::pca.fd

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Python `unittest` / `pytest` for gate tests; criterion 0.5 for Rust timing |
| Config file | None — gate.py is run directly; criterion config in Cargo.toml |
| Quick run command | `python3 benchmarks/gate.py --dry-run` (generates data, runs agreement check only) |
| Full suite command | `python3 benchmarks/run_all.py` (generates + times + gate + collects) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BENCH-01 | Generator produces identical CSV consumed by all runners | Integration (spot-check) | `python3 benchmarks/gate.py --check-csv-identity` | ❌ Wave 0 |
| BENCH-02 | Python + R runners cover all 6 ops and emit valid timing JSON | Integration | `python3 benchmarks/python_runner.py --smoke && Rscript benchmarks/r_runner.R --smoke` | ❌ Wave 0 |
| BENCH-03 | Criterion parser correctly extracts mean_ns from estimates.json | Unit | `python3 -m pytest benchmarks/test_collect_rust.py` | ❌ Wave 0 |
| BENCH-04 | Thread vars are set and captured in env.json | Integration | `python3 benchmarks/run_all.py --env-only` | ❌ Wave 0 |
| BENCH-05 | Agreement gate passes on synthetic data; fails on deliberately mismatched data | Unit | `python3 -m pytest benchmarks/test_gate.py` | ❌ Wave 0 |

### Wave 0 Gaps
- [ ] `benchmarks/test_collect_rust.py` — unit tests for criterion output parser (BENCH-03)
- [ ] `benchmarks/test_gate.py` — unit tests for agreement gate logic (BENCH-05)
- [ ] `benchmarks/generate_data.py` — seeded curve generator
- [ ] `benchmarks/gate.py` — numeric-agreement gate
- [ ] `benchmarks/collect_rust.py` — criterion output parser
- [ ] `benchmarks/python_runner.py` — scikit-fda timing runner
- [ ] `benchmarks/r_runner.R` — R timing runner
- [ ] `benchmarks/run_all.py` — master orchestrator
- [ ] R package installs: `renv`, `bench` (blocking)
- [ ] Python venv setup + scikit-fda 0.10.1 install (blocking)

---

## Security Domain

> `security_enforcement` is enabled. Phase 112 creates external scripts and a `benchmarks/` directory.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth — local scripts only |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | Local development tooling |
| V5 Input Validation | yes (low risk) | CSV inputs are generated by our own script; no external input |
| V6 Cryptography | no | No crypto |

### Known Threat Patterns for Python script harness

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| subprocess injection via bench parameter names | Tampering | Use list-form subprocess calls (no shell=True); bench names are hardcoded strings |
| Path traversal in criterion output parser | Tampering | Use `pathlib` with `relative_to()` and validate that `est_path` is under `CRITERION_DIR` |
| Malicious R package via renv.lock | Spoofing | Use `renv::restore()` from a lockfile created on this machine; do not fetch from untrusted sources |

---

## Project Constraints (from CLAUDE.md)

- **Scope**: Audit-only milestone — deliverables are infrastructure + report, not `fdars-core` changes.
- **No new Cargo dependency** — harness is entirely external; any criterion bench additions use existing bench infrastructure.
- **fdars timings** from existing criterion 0.5 harness only — parse `estimates.json`, no new dev-dependency.
- **Non-breaking** — no changes to `fdars-core` public API.
- **GSD workflow enforcement** — all file changes must go through GSD phase execution.
- **Commit gates** — `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`.
- **No `commit --no-verify`** for Phase 112 (pure external scripts, no Rust code changes, so pre-commit hook speed is not an issue).
- **Rust edition 2021, MSRV 1.81.0** — any Rust code in bench files must be compatible.
- **Column-major `FdMatrix`** — all fdars bench invocations use `FdMatrix::from_column_major`.

---

## Sources

### Primary (HIGH confidence)
- `fdars-core/src/regression.rs:253-300` — `fix_svd_signs` and `dominant_sign_negative` — FPCA sign convention [VERIFIED by direct Read]
- `target/criterion/audit_fpca/n500_m200/new/estimates.json` — criterion output layout [VERIFIED by direct Read]
- `target/criterion/elastic_align_pair/m/50/new/estimates.json` — BenchmarkId path layout [VERIFIED by direct Read]
- `fdars-core/src/alignment/pairwise.rs:1-44` — Fisher-Rao metric confirmed [VERIFIED by direct Read]
- `fdars-core/src/depth/dispatch.rs:287-329` — `functional_boxplot` signature [VERIFIED by direct Read]
- `fdars-core/src/depth/random_projection.rs:75-84` — `random_projection` takes explicit `nproj` [VERIFIED by direct Read]
- Live R session — `fda 6.3.0`, `fda.usc 2.2.0`, `roahd 1.4.3`, `fdasrvf 2.4.4` [VERIFIED]
- Live R session — `pca.fd` sign convention (max-abs positive) [VERIFIED]
- Live R session — `depth.RP` default `nproj=50`, `proj="vexponential"` [VERIFIED]
- Live R session — `pair_align_functions` default `lambda=0`, `pen="roughness"`, `method="DP"` [VERIFIED]
- Live R session — `time_warping` default `lambda=0`, `max_iter=20` [VERIFIED]
- Live R session — R BLAS: `/usr/lib/libblas.so.3.12.0` [VERIFIED]
- `fdars-core/benches/` — 24 bench files; criterion groups with output documented [VERIFIED]

### Secondary (MEDIUM confidence)
- `fdars-core/src/scalar_on_function/fregre_lm.rs:64` — `fregre_lm` function signature [VERIFIED by direct Read]
- Live R session — `roahd::fbplot` output structure [VERIFIED]
- CLAUDE.md project constraints [VERIFIED by direct Read]

### Tertiary (LOW confidence / ASSUMED)
- scikit-fda 0.10.1 API for FPCA, elastic alignment, basis smoothing — not installed [ASSUMED]
- Tolerance analysis for agreement gate thresholds [ASSUMED]
- `bench::mark()` ns extraction via `as.numeric()` [ASSUMED — bench not installed]

---

## Metadata

**Confidence breakdown:**
- FPCA sign convention: HIGH — verified by direct code inspection + live R test
- Criterion output format: HIGH — verified by live file inspection
- R package function signatures: HIGH — verified via live R session
- Cross-language op equivalence (elastic): HIGH for fdars+R, MEDIUM for Python
- Agreement tolerances: MEDIUM/LOW — analytical estimates, not empirically verified
- scikit-fda API details: LOW — package not installed

**Research date:** 2026-09-20
**Valid until:** 2026-10-20 (stable — package APIs change slowly; criterion format is stable)
