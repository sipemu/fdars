# Architecture Research

**Domain:** Cross-language benchmark harness + optimization pass for a Rust FDA library
**Milestone:** v0.47.0 Performance — Cross-Language Benchmarks & Optimization Pass
**Researched:** 2026-09-19
**Confidence:** HIGH (codebase read + project history read; toolchain integration patterns from v0.14.0 and v0.30.0 milestones)

---

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     EXTERNAL HARNESS  (no Cargo dep)                         │
│  benchmarks/                                                                  │
│  ├── shared/                  # single-source-of-truth synthetic datasets     │
│  │   └── generate.py          # writes .npy / .csv shared by all runners      │
│  ├── runners/                                                                 │
│  │   ├── python/              # scikit-fda runner (reads shared data)         │
│  │   └── r/                   # R ecosystem runner (reads shared data)        │
│  ├── collect.py               # aggregates JSON timing files → comparison CSV │
│  └── report/                  # Markdown / README speed-story generation       │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ reads shared data files (npy/csv)
                                │ reads fdars JSON timing output
┌───────────────────────────────▼──────────────────────────────────────────────┐
│                IN-CRATE CRITERION HARNESS  (existing, extended)               │
│  fdars-core/benches/                                                          │
│  ├── alignment_benchmarks.rs  # extended with cross-language cell sizes       │
│  ├── perf_hotpaths.rs         # PERMANENT before/after guard (v0.30.0)        │
│  ├── audit_hotpaths.rs        # v0.14.0 audit cells (preserved)               │
│  └── ... (20+ existing bench files)                                           │
│                                                                               │
│  fdars-core/src/              # optimization targets (NO public API change)   │
│  ├── alignment/               # elastic O(N²·M²) — anchor target              │
│  ├── regression.rs            # SVD/FPCA (faer path already exists)           │
│  └── ... (only internals change)                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | Responsibility | Location | New or Existing |
|-----------|----------------|----------|-----------------|
| Dataset generator | Emit a single set of `FdMatrix`-equivalent curve sets (N×M grids, multiple sizes) as `.npy` + `.csv` files that every ecosystem reads identically | `benchmarks/shared/generate.py` | NEW |
| Python runner | Time scikit-fda on each operation at each (N, M) cell; emit `timings_python.json` | `benchmarks/runners/python/run.py` | NEW |
| R runner | Time R packages (`fda`, `fda.usc`, `refund`, `fdapace`, `roahd`) on each operation; emit `timings_r.json` | `benchmarks/runners/r/run.R` | NEW |
| Rust criterion runner | Time fdars via criterion (existing benches, extended with matched cells); emit criterion JSON | `fdars-core/benches/` | EXISTING + EXTENDED |
| Timing collector | Merge `timings_python.json`, `timings_r.json`, criterion JSON → one comparison CSV / table | `benchmarks/collect.py` | NEW |
| Report generator | Render the comparison table into a Markdown speed story; driven from collected CSV | `benchmarks/report/` | NEW |
| Hotspot optimizer | Behavior-preserving code changes inside `fdars-core/src/` on the ranked targets | `fdars-core/src/alignment/`, `src/regression.rs`, `src/covariance.rs` | EXISTING MODIFIED |
| Before/after guard | Criterion cells that freeze the pre-optimization baseline and prove the after improvement | `fdars-core/benches/perf_hotpaths.rs` extended; new bench cells | EXISTING + EXTENDED |
| dhat allocation probe | Dev-dep `dhat` integration test that verifies heap-allocation reduction on optimization targets | `fdars-core/tests/` (feature-gated `dhat-heap`) | EXISTING PATTERN |
| Golden-value checker | Test(s) asserting numeric outputs are bit-for-bit (or ≤1e-10) identical before/after optimization | `fdars-core/tests/` or inline `#[test]` | NEW (per optimization) |

---

## Recommended Project Structure

```
fdars/                                  # workspace root (unchanged)
├── fdars-core/                         # Rust crate (only internals change)
│   ├── benches/
│   │   ├── alignment_benchmarks.rs     # EXTEND: add N=50/100/200 × M=50/100 cells
│   │   ├── perf_hotpaths.rs            # EXTEND: add elastic alignment before-cell
│   │   └── ...                         # all other existing benches unchanged
│   ├── src/
│   │   ├── alignment/                  # OPTIMIZE: elastic O(N²·M²) hot path
│   │   ├── regression.rs               # OPTIMIZE: SVD/FPCA (faer path widening)
│   │   └── ...                         # all other src unchanged
│   └── Cargo.toml                      # NO new [dependencies]; dhat already in [dev-dependencies]
│
├── benchmarks/                         # NEW top-level directory (not a Cargo member)
│   ├── README.md                       # reproduction instructions
│   ├── shared/
│   │   ├── generate.py                 # emits shared_N50_M50.npy, shared_N100_M100.npy, etc.
│   │   └── requirements.txt            # numpy, scipy (data gen only)
│   ├── runners/
│   │   ├── python/
│   │   │   ├── run.py                  # times scikit-fda on shared data
│   │   │   └── requirements.txt        # scikit-fda==0.10.1, numpy, timeit
│   │   └── r/
│   │       ├── run.R                   # times fda, fda.usc, refund, fdapace, roahd
│   │       └── renv.lock               # pinned R package versions (renv)
│   ├── collect.py                      # merges JSON outputs → results/comparison.csv
│   └── results/                        # gitignored raw outputs, committed summary CSV
│       ├── .gitignore                  # ignore *.json (raw), keep *.csv (summary)
│       └── comparison.csv              # COMMITTED: the final table (updated each run)
│
├── scripts/                            # existing; add bench driver script
│   └── run-crosslang-bench.sh          # orchestrates generate → Rust → Python → R → collect
│
└── Cargo.toml                          # workspace unchanged (fdars-core only member)
```

### Structure Rationale

- **`benchmarks/` at workspace root, not inside `fdars-core/`:** Keeps zero Cargo dependency. Python/R toolchains are invoked by shell scripts, never by `cargo`. The directory is excluded from `Cargo.toml` workspace members. This is identical to the v0.14.0 audit pattern where scripts lived in `scripts/` with no Cargo coupling.
- **`benchmarks/shared/` with a single generator:** One Python script emits the shared dataset files that all three ecosystems read. This is the single source of truth; if any ecosystem reads different data the comparison is invalid. The generator uses numpy's deterministic seeding so files are reproducible.
- **`.npy` + `.csv` as the shared format:** `.npy` is the lingua franca of numerical Python; `numpy.load` is available in any Python benchmark runner. R reads `.csv` natively (`read.csv`). Rust reads `.csv` in the generator script or directly converts from the same deterministic formula — the simplest approach is for the Rust bench to re-generate the identical data in-process using the same formula that `generate.py` documents, cross-checked against the saved `.csv` at setup time.
- **`results/comparison.csv` committed, raw JSON gitignored:** The final merged table is the artifact that feeds the report. Raw per-run criterion JSON and Python/R timing JSON are ephemeral and large.
- **`scripts/run-crosslang-bench.sh`:** A single entry point orchestrates the full run. Documents the exact sequence so any developer can reproduce it from scratch.

---

## Architectural Patterns

### Pattern 1: Shared-Input Deterministic Dataset

**What:** A single Python script generates the synthetic curve sets and writes them to numbered files (`shared_N50_M50.npy`, `shared_N100_M100.npy`, etc.) using a seeded RNG and a documented generation formula (sinusoidal with per-curve phase/amplitude variation, matching the formula in `alignment_benchmarks.rs`). All three ecosystem runners read these same files.

**When to use:** Any time you need apples-to-apples timing — ensuring that data shape, scale, and content are identical across ecosystems is the only way to make the comparison meaningful.

**Trade-offs:** The format must be readable by all three ecosystems without a heavy dependency. `.npy` requires numpy (always available in scikit-fda setups) and `RcppCNPy` or a csv fallback in R. The `.csv` fallback is slower to parse but has zero new dependencies in R. Use `.npy` for Python runner and `.csv` for R runner to avoid the R numpy reader dependency.

**Concrete cell sizes:** Match the existing criterion cells already in `alignment_benchmarks.rs`:
- N=10, M=50 (light cell; elastic pair)
- N=50, M=50 (elastic self-matrix baseline)
- N=100, M=100 (medium; regression and FPCA)
- N=200, M=200 (heavy; only elastic with band)

### Pattern 2: Ecosystem Runner as a Standalone Script with JSON Output

**What:** Each ecosystem runner (Python `run.py`, R `run.R`) is a self-contained script that reads the shared `.npy`/`.csv` files, times each operation using the ecosystem's idiomatic timer (`timeit.timeit` for Python, `proc.time()` or `microbenchmark::microbenchmark()` for R), and writes structured JSON:

```json
{
  "ecosystem": "python",
  "package_versions": {"scikit-fda": "0.10.1", "numpy": "1.26.0"},
  "env": {"python": "3.11.0", "cpu": "...", "date": "2026-09-19"},
  "results": [
    {"operation": "fpca_n100_m100", "mean_ms": 42.3, "std_ms": 1.1, "n_reps": 20}
  ]
}
```

**When to use:** This decouples the timing logic from the reporting logic and makes reruns trivial. The `collect.py` script merges all JSONs into a single comparison table.

**Trade-offs:** Requires maintaining three scripts in three languages. Keep each runner minimal — just timing calls, no data transformation or processing beyond what the benchmark function requires.

### Pattern 3: Criterion JSON + Cross-Language Merge

**What:** Criterion already supports `--output-format json` via the unstable `criterion.toml` output configuration. For v0.47.0 the simplest approach is to capture criterion's stdout text output for specific bench groups using `cargo bench --bench alignment_benchmarks -- --output-format bencher 2>/dev/null` and parse the bencher-format lines. Alternatively, use `criterion`'s `--save-baseline` and read the saved JSON from `target/criterion/`. The collector script reads criterion's saved baseline files directly from `target/criterion/<bench_name>/<group>/<id>/new/estimates.json`.

**When to use:** Avoids duplicating timing logic for Rust. Criterion's output is the authoritative Rust measurement; no separate Rust runner script is needed.

**Trade-offs:** Criterion's saved JSON format is semi-internal (not a public API). Path: `target/criterion/<bench_name>/<group>/<function_id>/new/estimates.json`. The `mean.point_estimate` field (in nanoseconds) is stable across criterion 0.5.x. The collector script should document the field path and note it may need updating on criterion upgrades.

### Pattern 4: measure → optimize → prove (v0.30.0 pipeline, extended)

**What:** The established v0.30.0 pipeline: (1) run criterion before-cells to record baseline, (2) implement the optimization change inside `fdars-core/src/`, (3) run criterion after-cells to prove speedup, (4) run `cargo test` + golden-value assertions to prove numeric identity. For v0.47.0 this extends to: (5) re-run the cross-language harness after optimization to update the comparison table.

**When to use:** Every optimization change. The before-cell must be committed before the code change so the comparison is honest.

**Concretely for elastic alignment:**
```
Phase A (measure): run alignment_benchmarks.rs N=50/100 cells, record baseline
Phase B (optimize): modify alignment/ internals (see known hot paths below)
Phase C (prove): run same cells again → speedup %, run cargo test → green,
                 run golden assertions → bit-identical outputs
Phase D (update): re-run cross-language harness → updated comparison.csv
```

---

## Data Flow

### Cross-Language Benchmark Flow

```
scripts/run-crosslang-bench.sh
    │
    ├──► python benchmarks/shared/generate.py
    │       writes: benchmarks/results/shared_N{n}_M{m}.npy
    │               benchmarks/results/shared_N{n}_M{m}.csv
    │
    ├──► cargo bench --bench alignment_benchmarks --save-baseline crosslang
    │       writes: target/criterion/elastic_align_pair/.../estimates.json
    │               target/criterion/karcher_mean/.../estimates.json
    │               target/criterion/fpca/.../estimates.json
    │
    ├──► python benchmarks/runners/python/run.py
    │       reads:  benchmarks/results/shared_*.npy
    │       writes: benchmarks/results/timings_python.json
    │
    ├──► Rscript benchmarks/runners/r/run.R
    │       reads:  benchmarks/results/shared_*.csv
    │       writes: benchmarks/results/timings_r.json
    │
    └──► python benchmarks/collect.py
            reads:  target/criterion/**/estimates.json
                    benchmarks/results/timings_python.json
                    benchmarks/results/timings_r.json
            writes: benchmarks/results/comparison.csv  (committed)
                    benchmarks/results/report.md       (fed into README)
```

### Optimization Proof Flow

```
git stash (or branch) — record clean baseline
    │
    ▼
cargo bench --bench alignment_benchmarks --save-baseline before_opt
    │ records: target/criterion/.../before_opt/
    ▼
[code change in fdars-core/src/alignment/]
    │
    ▼
cargo test -p fdars-core --features linalg,parallel
    │ must be: green (all ~2942 tests)
    ▼
cargo bench --bench alignment_benchmarks --save-baseline after_opt
    │ records: target/criterion/.../after_opt/
    ▼
criterion compare: before_opt vs after_opt
    │ must show: statistically significant speedup on target cells
    ▼
golden-value assertion test (new test, per optimization)
    │ must show: outputs bit-identical (or ≤1e-10 for float reordering)
    ▼
dhat probe (existing dhat-heap pattern, if allocation is the target)
    │ must show: allocation count/bytes reduced (not required for compute-bound)
    ▼
[commit with before/after numbers in commit message body]
```

---

## Known Hot Paths (Integration Targets)

These are the confirmed optimization targets from the v0.14.0 audit evidence (`AUDIT-REPORT.md` §Phase 4–6) and the project's criterion output:

| Target | Location | Complexity | Known Cost | Optimization Direction |
|--------|----------|------------|------------|------------------------|
| Elastic alignment pair | `src/alignment/mod.rs` (`elastic_align_pair`) | O(M²) per pair, O(N²·M²) self-matrix | N=500,M=50 → 37.82 s (v0.14.0 audit); anchor target | Sakoe-Chiba band already ships (`elastic_align_pair_banded`); investigate SIMD inner loop or better DP ordering |
| Karcher mean | `src/alignment/mod.rs` (`karcher_mean`) | O(N·M²) per iteration | Thread-scaling ~4.73× at 8 threads already; rayon path exists | Profile remaining serial sections; per-iteration parallelism gap |
| SVD/FPCA | `src/regression.rs` (`fdata_to_pc_1d`) | O(N·M²) or O(M·N²) for thin SVD | faer path 1.8–4.1× faster than nalgebra (Phase 6 audit) under `linalg` feature | Widen faer path to 2D FPCA; audit `elastic_fpca.rs` SVD call |
| Covariance surface | `src/covariance.rs` | O(N·M²) | `face_covariance` N=200,M=30 → 984 ms (Phase 47 audit) | Kernel separability; vectorize inner product loop |

---

## Integration Points

### Boundary: External Harness ↔ In-Crate Criterion

| Concern | Decision | Rationale |
|---------|----------|-----------|
| No Cargo dependency | `benchmarks/` is NOT a workspace member | `Cargo.toml` lists only `fdars-core`; adding `benchmarks/` as a member would require it to be a Rust crate |
| Shared data format | `.npy` for Python, `.csv` for R, re-generated in-process for Rust using the documented formula | Rust benches don't read files; they call `generate_curves()` which must produce numerically identical data to what `generate.py` writes |
| Data identity verification | A setup script compares Rust in-process outputs against the `.csv` for a spot-check N=10,M=50 cell | Ensures the formula is synchronized across languages before the full benchmark run |
| Timing unit normalization | `collect.py` normalizes all timings to milliseconds (ms) before merging | criterion reports ns, Python timeit reports seconds, R microbenchmark reports ms — all normalized at the collector layer |

### Boundary: Optimization Changes ↔ Public API / R / WASM Bindings

| Constraint | Enforcement |
|------------|-------------|
| No public signature changes | All optimization changes are inside private `_impl` functions or internals; public function signatures must compile identically |
| R binding safety | The `fdars-r` R package uses `default-features = false` (no `linalg`/`parallel`); optimization changes must compile under the default-only feature set or be gated behind `linalg`/`parallel` |
| WASM safety | No allocation of thread-local state; no `std::thread`; no `rayon` in the non-`parallel` path |
| 28 examples | Must all compile and pass after each optimization change |
| pmat quality gate | No new cognitive complexity violations; all optimization helpers must meet ≤23 threshold |

### Boundary: Before/After Criterion Cells ↔ Committed Code

The v0.30.0 pattern (`perf_hotpaths.rs`) must be extended: add a new bench group for each optimization target that captures both the pre-optimization and post-optimization cell in the same bench file, with a comment block documenting the before/after numbers at commit time. This creates a permanent regression guard — if a future change regresses the optimized path, criterion will show it.

---

## Build Order / Phase Decomposition

This order respects the dependency chain: cross-language harness must exist before you can rank hotspots; hotspot ranking must exist before you can choose optimization targets; optimization must be measured before/after.

```
Phase 112: Cross-Language Harness Setup
    - Create benchmarks/ directory structure
    - Write generate.py (shared dataset generator, all N×M cell sizes)
    - Write runners/python/run.py (scikit-fda, pinned scikit-fda==0.10.1)
    - Write runners/r/run.R (fda, fda.usc, refund, fdapace, roahd; renv.lock)
    - Write collect.py (merge JSONs + criterion files → comparison.csv)
    - Write scripts/run-crosslang-bench.sh (full orchestration)
    - Deliverable: harness runs end-to-end; comparison.csv produced

Phase 113: Baseline Measurement & Hotspot Ranking
    - Extend existing criterion benches with matched N×M cells (same sizes as shared data)
    - Run full cross-language harness; produce initial comparison.csv
    - Combine cross-language results with existing criterion data
    - Produce evidence-ranked hotspot list (absolute ms + fdars/scikit-fda ratio)
    - Deliverable: BENCH-RESULTS.md with ranked targets; Phase 114+ targets confirmed

Phase 114: Elastic Alignment Optimization (Anchor Target)
    - Record before-baseline criterion cells (alignment_benchmarks.rs)
    - Implement optimization inside src/alignment/ (internals only)
    - Prove: cargo test green, golden-value assertion, criterion after-cells
    - Deliverable: alignment optimization committed with before/after criterion numbers

Phase 115: SVD/FPCA & Covariance Optimization (if ranking warrants)
    - Record before-baseline for FPCA and covariance cells
    - Implement optimization (widening faer path and/or covariance vectorization)
    - Prove: same pipeline as Phase 114
    - Deliverable: committed with before/after proof

Phase 116: Cross-Language Re-Run & Perf Report
    - Re-run full cross-language harness after all optimizations
    - Update comparison.csv with after-optimization numbers
    - Generate published perf report (Markdown speed story)
    - Deliverable: benchmarks/results/report.md + README update

Phase 117: Release Preparation
    - Version bump 0.46.0 → 0.47.0
    - CHANGELOG entries (both root and crate-level)
    - All whole-crate gates green (fmt, clippy --all-targets, cargo test, serde build,
      28 examples, doctests, cargo package, pmat quality-gate)
    - Deliverable: release-ready crate
```

---

## Anti-Patterns

### Anti-Pattern 1: Making `benchmarks/` a Cargo Workspace Member

**What people do:** Add `benchmarks/` to the `[workspace] members = [...]` list in `Cargo.toml` so "everything is in one place."

**Why it's wrong:** Immediately forces the Python/R harness to be a Rust crate. The `benchmarks/` directory contains Python and R scripts — there is no `Cargo.toml` inside it. Adding it to the workspace is not possible and would pollute the workspace with a Rust shim crate that serves no purpose.

**Do this instead:** Keep `benchmarks/` as a plain directory at the workspace root. It is discovered by humans and CI via the orchestration script, not by Cargo.

### Anti-Pattern 2: Running Python/R Benchmarks from Inside `build.rs` or a Test

**What people do:** Invoke `Command::new("python")` from a Rust `build.rs` or `#[test]` to get "unified" timing.

**Why it's wrong:** This couples the Rust build to Python/R being installed on the build machine, violates the no-new-dep rule's spirit, breaks the WASM build, and causes `cargo test` to fail in CI environments without Python. It also makes it impossible to control Python/R package versions independently.

**Do this instead:** The cross-language runs are always invoked from the shell orchestration script, never from Rust code.

### Anti-Pattern 3: Skipping the Before-Baseline Commit

**What people do:** Make the optimization change and only then run criterion, recording the "after" numbers as if they are the first measurement.

**Why it's wrong:** Without a committed before-baseline, you cannot prove the optimization improved anything. Future reviewers (and the GSD audit) will reject the claim.

**Do this instead:** Run criterion with `--save-baseline before_opt` before touching a single line of `src/`, commit or stash the baseline results reference in the bench file's header comment, then make the change and run `--save-baseline after_opt`.

### Anti-Pattern 4: Relaxing Test Tolerances to Pass the Golden-Value Gate

**What people do:** Change a golden-value assertion from bit-exact comparison to a wide tolerance (e.g., 1e-3) to accommodate floating-point reordering in the optimized path.

**Why it's wrong:** The v0.47.0 constraint is "strictly non-breaking + numeric-identical." A 1e-3 tolerance change is a behavioral change that could break user code silently. The v0.30.0 and v0.17.0 milestones held this to 1e-10 or bit-exact.

**Do this instead:** Design the optimization so it does not change the order of floating-point operations (e.g., use the same accumulation order, just vectorized). If exact ordering is unavoidable, document the numerical equivalence proof explicitly and keep the tolerance tight (≤1e-10).

### Anti-Pattern 5: Adding a New `[dependency]` for the Optimization

**What people do:** Reach for a SIMD crate (`packed_simd`, `wide`, `pulp`) or a different linear-algebra crate to implement the hot-path optimization.

**Why it's wrong:** The v0.47.0 constraint is "no new crate dependency." The existing stack (`nalgebra`, `faer` via `linalg` feature, `rayon` via `parallel` feature) already covers the optimization surface. Manual loop unrolling and compiler-auto-vectorization are the first tools; the `linalg` feature's `faer` path is the second.

**Do this instead:** Use manual loop restructuring, cache-friendly access patterns (respecting column-major `FdMatrix` layout), and where applicable the existing `linalg` feature's `faer` path. If a specific SIMD intrinsic is genuinely needed, use `std::simd` (stabilized Rust nightly) only if it lands in stable before the milestone; otherwise defer to a future milestone.

---

## Additive/Non-Breaking Constraint Checklist

All changes in v0.47.0 must satisfy every row:

| Constraint | How Honored |
|------------|-------------|
| No new `[dependencies]` in `fdars-core/Cargo.toml` | `benchmarks/` is external; Python/R deps are in their own requirements files |
| No public signature changes in `fdars-core/src/` | All optimization changes are inside private internals |
| R binding (`default-features = false`) still builds | Only modify functions that compile without `linalg`/`parallel`, or gate changes behind those features |
| WASM target still builds | No `std::thread`, no unconditional `rayon`; gated as today |
| 28 examples still compile and pass | Verified in the release-prep phase gate |
| pmat quality-gate stays at 0 violations | New helper functions from decomposition must stay ≤23 cognitive complexity |
| `cargo test` green throughout | Golden-value + suite-wide tests run after each optimization commit |
| `--features serde` build green | No new public types without serde-conditional derives |
| `cargo package` clean | Verified in release-prep phase |

---

## Sources

- `fdars-core/benches/alignment_benchmarks.rs` — existing bench cells and generate_curves formula (direct codebase read, HIGH confidence)
- `fdars-core/benches/perf_hotpaths.rs` — v0.30.0 before/after pattern, dhat probe pattern (direct codebase read, HIGH confidence)
- `fdars-core/Cargo.toml` — workspace structure, feature flags, existing dev-dependencies (direct codebase read, HIGH confidence)
- `.planning/PROJECT.md` §v0.47.0 — milestone constraints, target features, phase numbering (direct file read, HIGH confidence)
- `.planning/research/AUDIT-REPORT.md` §Phase 4–6 — confirmed hot-path evidence: SVD ~99.8% wall-clock, faer 1.8–4.1× faster, elastic N=500,M=50 → 37.82 s (cited findings from v0.14.0 audit, HIGH confidence)
- `.planning/PROJECT.md` §v0.30.0 — established measure→optimize→prove pipeline: criterion before/after + dhat allocation probe + golden-value numeric-identity check (direct file read, HIGH confidence)
- `Cargo.toml` (workspace root) — confirmed only `fdars-core` is a workspace member (direct file read, HIGH confidence)

---

*Architecture research for: v0.47.0 Cross-Language Benchmark Harness + Optimization Pass*
*Researched: 2026-09-19*
