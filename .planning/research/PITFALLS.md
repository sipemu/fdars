# Pitfalls Research

**Domain:** Performance auditing and feature-parity gap analysis — Rust numerical library (fdars) vs Python reference (scikit-fda)
**Researched:** 2026-08-07
**Confidence:** HIGH

---

## Critical Pitfalls

### Pitfall 1: Benchmarking in Debug Mode

**What goes wrong:**
Criterion benchmarks are run without `--release`, or `cargo bench` is run inside an IDE terminal that injects `CARGO_PROFILE=dev`. Numbers look real (wall-clock, throughput) but are 5–50× slower than release. The audit then flags hot paths that are not actually hot, and misses the real bottlenecks.

**Why it happens:**
`cargo bench` defaults to the `bench` profile (which is release), but developers who call `cargo test --bench` or use IDE run buttons may get debug builds. The mismatch is invisible unless you inspect the binary path in Criterion's output header.

**How to avoid:**
Always run: `cargo bench --release -p fdars-core --features linalg 2>&1 | head -5` and confirm the binary path contains `/release/`. Add this as the first line of the benchmark-confirmation phase procedure. Record the rustc version and profile in every benchmark result table.

**Warning signs:**
- Criterion shows throughput < 1 MB/s for simple matrix ops
- SVD on a 50×50 matrix takes > 10 ms
- The `/target/debug/` path appears in Criterion HTML report URL

**Phase to address:**
Benchmark Confirmation phase — include a mandatory build-mode verification step in the phase plan before recording any numbers.

---

### Pitfall 2: Benchmarking Without `--features linalg` When Ridge/Faer Paths Are Being Timed

**What goes wrong:**
The `linalg` feature gates faer-based ridge regression and Cholesky solvers. Without it, those code paths fall back to a slower path or are absent. A benchmark omitting `--features linalg` correctly measures the non-linalg path but wrongly represents performance for users who do enable it — and vice versa. Comparing the two as if they are equivalent inflates or deflates gaps.

**Why it happens:**
The feature-flag matrix (`default=["parallel"]`, optional `linalg`, `serde`, `js`) means the compiled binary is not the same across feature combinations. Developers forget that criterion benchmarks compiled without `linalg` literally do not call the faer code paths.

**How to avoid:**
Run benchmarks for each relevant feature combination and tag results explicitly. The audit's benchmark table must have a `features` column. Minimum required runs: `--features ""`, `--features linalg`, `--features linalg,parallel`. Never merge results across feature sets.

**Warning signs:**
- A benchmark result for `ridge_regression_fit` that doesn't mention which features were active
- Criterion output that doesn't show the feature flags used (check: `rustc --print cfg` inside the bench binary, or just document explicitly in the result table)

**Phase to address:**
Benchmark Confirmation phase — the phase plan must enumerate the feature-flag matrix for each benchmark target.

---

### Pitfall 3: Missing `std::hint::black_box` Allows Dead-Code Elimination

**What goes wrong:**
The Rust compiler (and LLVM) can prove that a benchmark's computation has no observable effect and eliminate it entirely. The benchmark then measures near-zero time — appearing to show the function is extremely fast — when in fact it never ran.

**Why it happens:**
When writing a quick benchmark directly against fdars functions that return `f64` or `Vec<f64>`, it is tempting to write `let _ = heavy_function(data)` without wrapping inputs or outputs in `black_box`. Criterion's `iter` closure provides some protection but does not guarantee elimination is blocked for all compiler optimization levels.

**How to avoid:**
Wrap benchmark inputs in `criterion::black_box(input)` and consume outputs via `criterion::black_box(result)`. For the existing 8 criterion benchmarks in fdars, audit each one to confirm `black_box` is present on both input and output sides. When adding new benchmarks in the Benchmark Confirmation phase, make this a code-review criterion.

**Warning signs:**
- Benchmark time < 10 ns for any matrix operation on n > 10
- Build with `--emit=llvm-ir` and search for the function name — if absent, it was eliminated
- Criterion shows 0% variance (too good to be true)

**Phase to address:**
Benchmark Confirmation phase — review existing benchmarks first; flag any missing `black_box` as a benchmark correctness issue before recording results.

---

### Pitfall 4: Unrepresentative Input Sizes (Microbenchmark Mismatch)

**What goes wrong:**
The existing 8 criterion benchmarks use small inputs convenient for fast CI. The audit records their numbers and draws conclusions about scalability. But fdars' real workloads involve n=500–5,000 curves and m=100–1,000 evaluation points. Small-input benchmarks are dominated by fixed overhead (thread pool spinup, allocation setup) and miss the O(n²) or O(m³) growth that matters.

**Why it happens:**
Small inputs run fast in CI without timeouts. Nobody updates them as the library matures.

**How to avoid:**
For each benchmark, add a large-input variant at a realistic workload size. Specifically: elastic alignment at n=200 (O(n²) concern), FPCA/SVD at m=500 (O(m³) concern), basis CV at n=50 curves × 100 points (quadrature cost). Record results for both small and large inputs and note which scaling regime each falls into. The Static Hot-Path Analysis phase should determine "realistic workload" definitions before the Benchmark Confirmation phase runs.

**Warning signs:**
- All existing benchmarks use n ≤ 20 or m ≤ 50
- Benchmark results show linear scaling in n where the algorithm is O(n²)
- No benchmark covers elastic alignment at n > 50

**Phase to address:**
Static Hot-Path Analysis (define workload targets) → Benchmark Confirmation (run at those sizes).

---

### Pitfall 5: Ignoring Allocation Cost vs CPU Cost

**What goes wrong:**
The audit reports "SVD takes 12 ms" but the majority of that time is memory allocation: the `FdMatrix → nalgebra::DMatrix` copy described in ARCHITECTURE.md and CONCERNS.md. A fix that removes the copy would halve the time, but the audit missed this because it only measured wall-clock time without heap profiling. Future implementers then try to optimize the SVD algorithm itself instead.

**Why it happens:**
Wall-clock benchmarks do not distinguish CPU from allocator. The known issue (dense matrix reconstruction, DMatrix round-trips) is documented in CONCERNS.md but easy to overlook when reading numbers.

**How to avoid:**
Augment Criterion benchmarks with `dhat` or `cargo-heaptrack` for the top-3 hottest paths. At minimum, add a comment in the benchmark recording the expected allocation pattern (e.g., "allocates 1 DMatrix per call, O(n*m) bytes"). Static analysis should flag all `DMatrix::from_column_slice` and `FdMatrix::from_column_major` call sites as allocation hotspots before benchmark confirmation.

**Warning signs:**
- Benchmark time is dominated by a single `unsafe` copy or `clone`
- `cargo-flamegraph` shows allocator (`jemalloc`/`ptmalloc`) consuming > 20% of samples
- CONCERNS.md already notes this pattern — if the audit doesn't address it, it has missed something documented

**Phase to address:**
Static Hot-Path Analysis — include allocation analysis as a required deliverable alongside CPU-path analysis.

---

### Pitfall 6: Warm vs. Cold Cache Invalidation

**What goes wrong:**
Criterion runs each benchmark multiple times to stabilize variance. After the first few iterations, CPU L2/L3 caches are warm with the working data. For fdars' real use case (called once per batch from R or WASM), the cold-cache cost matters and is 2–5× higher than the warm measurement.

**Why it happens:**
Criterion is designed for throughput measurement and naturally warms the cache. Cold-cache measurement requires flushing caches manually (mmap tricks, or re-allocating data each iteration at a cost that must be subtracted).

**How to avoid:**
Note in every benchmark result whether it reflects warm or cold cache. For the audit, warm-cache numbers are acceptable as the primary comparison metric — just add a note that first-call latency may be higher for large inputs. If a function is specifically called once per session (e.g., `fregre_lm` in a WASM pipeline), flag it as a cold-cache concern in the backlog item.

**Warning signs:**
- Benchmark result for a large-input path seems implausibly fast vs. theoretical memory bandwidth
- Flamegraph shows no cache-miss stalls despite large matrix sizes

**Phase to address:**
Benchmark Confirmation — add a "cache regime" column to the results table (warm/cold/N/A).

---

### Pitfall 7: Noisy Machine / Missing Baseline Variance Control

**What goes wrong:**
Benchmarks are run on a laptop with background processes (IDE, browser, compiling other crates). Criterion reports ±15% variance and the audit treats the mean as authoritative. A "30% speedup" finding later turns out to be within noise.

**Why it happens:**
Developers bench on whatever machine they have. Criterion's outlier detection helps but does not prevent systematic interference from OS scheduling.

**How to avoid:**
Before the Benchmark Confirmation phase runs, close non-essential processes. Run each benchmark at least twice in separate `cargo bench` invocations and confirm the means are within ±5%. If Criterion reports > 10% variance on a measurement, mark that result as LOW CONFIDENCE in the audit report and do not use it to justify a backlog priority. Optionally use `cpupower` (Linux) to disable frequency scaling during the benchmark run.

**Warning signs:**
- Criterion "change" reports between two identical runs show > 5% difference
- Standard deviation > 10% of mean in Criterion output

**Phase to address:**
Benchmark Confirmation — add a variance threshold to the phase acceptance criteria.

---

### Pitfall 8: Linker/Toolchain Flakiness Masking Real Test Failures

**What goes wrong:**
This environment currently produces criterion/doctest linker "bus errors" unrelated to fdars code. If the audit runs `cargo test` and sees failures, it may misclassify linker-flakiness failures as code bugs, inflating the apparent defect count. Conversely, if the flakiness causes the test run to abort, real failures may be hidden.

**Why it happens:**
Linker bus errors on Linux can arise from memory-mapped file limits (`vm.max_map_count`), toolchain mismatches between the system linker and the Rust toolchain, or from doctest infrastructure bugs in Criterion 0.5 (a known issue). These are infrastructure failures, not fdars failures.

**How to avoid:**
Before recording any test-failure count, run `cargo test -p fdars-core --features linalg -- --test-threads=1 2>&1 | grep -E "^(test |FAILED|error)"` and distinguish: (a) `FAILED` lines naming a specific test = code failure, (b) `error: process didn't exit successfully` without a test name = toolchain/linker failure. Only category (a) counts toward the audit's defect list. Document the environment's known linker issue explicitly in the audit report's methodology section.

**Warning signs:**
- `cargo bench` or `cargo test --doc` exits with a signal (bus error, segfault) rather than a failed test count
- The failure disappears on `--test-threads=1` or with `RUSTFLAGS=-C link-arg=-fuse-ld=lld`
- The failure is in Criterion harness infrastructure, not in fdars code

**Phase to address:**
All phases involving test execution — include a "distinguish infrastructure failures from code failures" checklist item.

---

### Pitfall 9: Counting API Names Instead of Capabilities in Parity Analysis

**What goes wrong:**
The gap analysis counts scikit-fda's public function names and compares them 1:1 against fdars exports. This produces a large-looking gap (fdars has 40 public functions, scikit-fda has 120) that is mostly noise: scikit-fda has separate functions for fit/predict/transform/inverse_transform that fdars bundles into one call returning a result struct. The audit then recommends "add 80 functions" when the capability already exists under a different shape.

**Why it happens:**
API surface is easy to enumerate programmatically. Capability mapping requires understanding what each function does.

**How to avoid:**
Structure the parity analysis as a capability matrix, not a function-count comparison. Group by user task: "smooth a set of curves", "compute FPCA", "classify new curves", etc. For each task, determine: (a) can scikit-fda do this? (b) can fdars do this? Accept that fdars may accomplish the same task via a different call shape. Treat builder structs + a single function call as equivalent to scikit-fda's `fit()/transform()` pattern. Document the mapping explicitly.

**Warning signs:**
- Gap count is > 50 items before any capability grouping
- The gap list includes entries like "fdars missing `BasisFDA.fit()`" when `fdata_to_basis()` exists
- Parity matrix has a row per scikit-fda class name rather than per user task

**Phase to address:**
Gap Analysis phase — define the capability taxonomy before beginning enumeration.

---

### Pitfall 10: Treating "scikit-fda has X" as "fdars must have X"

**What goes wrong:**
scikit-fda is a Python library targeting data scientists who write scripts, use matplotlib, and call sklearn pipelines. fdars is a Rust numeric crate targeting performance-sensitive systems and language bindings (R, WASM). Features that make sense in scikit-fda — sklearn-compatible estimator API, matplotlib plot methods, pandas DataFrame integration — are irrelevant anti-features for fdars. An audit that counts these as gaps inflates the backlog with low-value work and misleads prioritization.

**Why it happens:**
It is easier to enumerate everything scikit-fda has than to evaluate each item against fdars' stated purpose.

**How to avoid:**
Before the gap analysis, write a one-page "design goal filter": what kinds of capabilities are in scope for fdars (numeric algorithms, memory layout, correctness) vs. out of scope (plotting, sklearn API compatibility, DataFrame IO). Apply this filter explicitly to every gap finding. The PROJECT.md already establishes this: "Plotting/visualization parity with scikit-fda — a numeric Rust library does not need matplotlib-style output; treat as low-priority."

**Warning signs:**
- Gap items reference matplotlib, seaborn, or sklearn pipeline interfaces
- A gap item is "fdars has no `plot()` method"
- Gap items outnumber the actual algorithm categories in scikit-fda by 2:1

**Phase to address:**
Gap Analysis phase — apply design-goal filter as a required pre-step before the analysis begins.

---

### Pitfall 11: Missing That fdars Already Has a Capability Under a Different Name

**What goes wrong:**
scikit-fda has `FPCATransformer.fit_transform()`. The analyst doesn't find `fit_transform` in fdars exports and marks it as a gap. fdars actually provides `fdata_to_pc_1d()` which computes and returns scores directly. The gap is a naming mismatch, not a missing capability.

**Why it happens:**
fdars uses domain-idiomatic naming (FDA jargon, Rust conventions) rather than sklearn-style method names. An analyst reading scikit-fda docs without deep fdars knowledge will miss these correspondences.

**How to avoid:**
Build the capability matrix from both sides: for each scikit-fda capability, search fdars by description/behavior rather than name. Use the `.planning/codebase/STRUCTURE.md` module map as the fdars side of the search. For any gap candidate, explicitly write: "searched fdars for: [what it does]. Closest match: [fdars function]. Verdict: [equivalent / partial / missing]." Partial matches should be a separate backlog category from missing.

**Warning signs:**
- Gap items use scikit-fda class/method names verbatim without a "searched for equivalent in fdars" note
- The gap list does not distinguish "capability absent" from "API shape differs"

**Phase to address:**
Gap Analysis phase — require explicit "fdars equivalent search" for every gap candidate.

---

### Pitfall 12: Ignoring Numerical Accuracy Parity vs. Mere Feature Presence

**What goes wrong:**
fdars implements FPCA, elastic alignment, and B-spline smoothing — same as scikit-fda. The analyst marks these as "present, no gap." But fdars' B-spline CV had a silent correctness bug (GH #33, fixed in v0.14.0) that produced wrong n_basis selection. If the audit does not check numerical accuracy, correctness gaps are invisible.

**Why it happens:**
Parity analyses default to "does the function exist?" because correctness verification requires test data and reference outputs.

**How to avoid:**
For the top-10 most-used capability areas, include a numerical accuracy check: run fdars and scikit-fda on the same small dataset and compare outputs within a tolerance. This is not full validation — it is a smoke-test for systematic discrepancies. Use the existing CONCERNS.md "Known Bugs" list as the starting point for which areas have correctness risk. Flag any capability as "present but accuracy not verified" rather than just "present."

**Warning signs:**
- Parity matrix has only ✓/✗ columns with no accuracy note
- Known buggy areas (elastic alignment level encoding, basis CV) are marked ✓ without a "fixed in v0.14.0, needs verification" note
- No test data or reference output is attached to any gap finding

**Phase to address:**
Gap Analysis phase — add an "accuracy verified?" column to the parity matrix.

---

### Pitfall 13: Ranking Gaps by Ease Instead of User Value

**What goes wrong:**
The backlog sorts findings by implementation effort: easy wins first. This produces a backlog that front-loads cosmetic improvements (add a `Display` impl, add a convenience wrapper) and defers the high-value, high-effort items (elastic alignment at scale, functional-on-functional regression). Users and stakeholders then see months of activity with no meaningful improvement.

**Why it happens:**
Effort is visible and objective. Value requires user research or domain judgment.

**How to avoid:**
Use a 2×2 matrix: value vs. effort. Populate "value" from: (a) GitHub issues and user pain points, (b) algorithm coverage gaps (things scikit-fda has that genuinely block fdars use cases), (c) known performance bottlenecks at realistic scales (O(n²) elastic alignment is a concrete blocker at n > 1,000). Rank the backlog by `value / sqrt(effort)` as a simple heuristic. Explicitly mark any convenience item as "low value" even if it is easy.

**Warning signs:**
- The top 5 backlog items are all "add `impl Display` / add `From` conversion / rename parameter"
- No item in the top 10 addresses a known O(n²) or O(m³) bottleneck
- Effort column is present but value column is absent

**Phase to address:**
Consolidated Audit Report + Prioritized Backlog phase — include a value-estimation step before ranking.

---

### Pitfall 14: Letting Plotting and IO Features Inflate the Gap Count

**What goes wrong:**
scikit-fda has extensive plotting integration (`FDataGrid.plot()`, `FDataBasis.plot()`, various `Visualization` classes) and pandas/DataFrame IO. An uncritical gap analysis counts all of these, inflating the gap from ~20 meaningful algorithm items to ~60 total. The report then looks alarming, and stakeholders ask why fdars is "so far behind."

**Why it happens:**
The comparison is done at the API-surface level without filtering for relevance to fdars' design goals.

**How to avoid:**
Apply the PROJECT.md exclusion explicitly: plotting and visualization are out of scope. IO helpers (DataFrame round-trips) are out of scope unless fdars explicitly targets that workflow. In the parity matrix, add a "Relevance" column with values: In-Scope Algorithm, In-Scope API Ergonomics, Out-of-Scope (plotting), Out-of-Scope (IO). Report gap counts separately for in-scope vs. out-of-scope.

**Warning signs:**
- More than 20% of gap items relate to visualization or IO
- Gap count drops dramatically when filtering to "numeric algorithm" items only

**Phase to address:**
Gap Analysis phase — apply relevance filter before finalizing the parity matrix.

---

### Pitfall 15: Findings Too Vague to Action in the Backlog

**What goes wrong:**
A backlog item reads: "Improve elastic alignment performance." No function name, no current measurement, no target, no suggested approach. A future implementer opens it and must redo the audit to understand what to do.

**Why it happens:**
Audit findings are written at the analytical level ("this is slow") without translating into implementation tasks.

**How to avoid:**
Every backlog item must include: (1) the specific function or code path, (2) the measured or estimated current cost, (3) the root cause (from the audit), (4) the suggested fix approach, and (5) the expected benefit. Use the CONCERNS.md format as a model: it has "Problem / Files / Cause / Improvement path" for every bottleneck. Backlog items that cannot be filled out this way are not ready to promote.

**Warning signs:**
- Backlog item has < 3 sentences of description
- No function name or file path in the item
- No "current state" measurement attached

**Phase to address:**
Prioritized Backlog phase — gate each item on a completeness checklist before finalizing.

---

### Pitfall 16: No Severity/Effort Estimate on Backlog Items

**What goes wrong:**
All findings are listed flat. `/gsd-new-milestone` promoter cannot tell which items to bundle into a sprint and which require a full milestone. Future milestones are either under-scoped (a single easy item) or over-scoped (a bundle of hard items that misses the deadline).

**Why it happens:**
Severity and effort estimation feels speculative and is skipped to save time.

**How to avoid:**
Use a 3-level scale for each dimension:
- Severity: P1 (blocks meaningful use), P2 (impairs real workloads), P3 (nice to have)
- Effort: S (< 1 week), M (1–3 weeks), L (> 3 weeks)

Each backlog item must have both fields. The Severity definition must be anchored to fdars' actual user base (R and WASM consumers, performance-sensitive pipelines) not hypothetical users. The O(n²) elastic alignment issue at n=1,000 is P1 for users with large corpora; a missing `Display` impl is P3.

**Warning signs:**
- Backlog has > 20 items with no relative ordering
- No item is explicitly labeled P1
- All items cluster at the same effort level

**Phase to address:**
Prioritized Backlog phase — severity/effort tagging is a required field, not optional.

---

### Pitfall 17: No Reproducible Evidence Attached to Findings

**What goes wrong:**
A performance finding says "SVD is slow." The evidence is "I ran the benchmark and it seemed slow." No command, no output, no environment. A future implementer cannot reproduce the finding, cannot verify whether a fix helps, and cannot be confident the finding was real.

**Why it happens:**
Audit findings are written from memory after running experiments. The raw output is not saved.

**How to avoid:**
For every performance finding: save the full `cargo bench` output (Criterion HTML or text output) to a file in `.planning/research/bench/` and reference it in the backlog item. For every gap finding: save the scikit-fda API reference URL and the fdars source location that was checked. For correctness findings: save the test case that reproduces the discrepancy. This is the "reproducible evidence" requirement.

**Warning signs:**
- Backlog items reference "the benchmark" without naming which benchmark or what it showed
- No `.planning/research/bench/` directory exists after the Benchmark Confirmation phase
- Gap findings cite scikit-fda features without a URL or version number

**Phase to address:**
Benchmark Confirmation phase (bench evidence) + Gap Analysis phase (gap evidence) — require evidence artifacts as phase deliverables.

---

### Pitfall 18: Feature-Flag Matrix Confusion (parallel / linalg / serde / js)

**What goes wrong:**
The static analysis or benchmark run uses default features (`parallel` enabled, `linalg` disabled). The audit concludes that a hot path "lacks parallelism" when in fact `iter_maybe_parallel!` is already there — it just compiled out because `--features parallel` was not explicitly passed alongside `--features linalg`. Or conversely, a finding claims ridge regression is fast without realizing the non-linalg fallback path was measured.

**Why it happens:**
Cargo's feature system means the same source file compiles differently depending on features. The 5 parallel macros in `parallel.rs` are a no-op under `cfg(not(feature="parallel"))`. It is easy to forget this when reading benchmark output.

**How to avoid:**
Create a benchmark matrix table at the start of the Benchmark Confirmation phase:

| Feature set | What it tests |
|-------------|--------------|
| `--features ""` | Sequential, no linalg (WASM / minimal build) |
| `--features parallel` | Default (most users) |
| `--features linalg` | Ridge/Cholesky paths, sequential |
| `--features linalg,parallel` | Full capability build |

Run at least `--features parallel` and `--features linalg,parallel` for every benchmark. Tag all results with the feature set used. In static analysis, note which code paths exist only under `linalg` and which only under `parallel`.

**Warning signs:**
- A benchmark result for a function that uses `iter_maybe_parallel!` doesn't note the `parallel` feature state
- Static analysis says "this function is sequential" for a function whose hot loop is wrapped in `iter_maybe_parallel!`

**Phase to address:**
Static Hot-Path Analysis (identify feature-gated paths) + Benchmark Confirmation (run all relevant feature combinations).

---

## Audit-Specific Technical Debt Patterns

Shortcuts that seem reasonable during the audit but create misleading conclusions.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Single feature-flag run for all benchmarks | Faster benchmark phase | Results only valid for one feature combination; misleads parity and perf findings | Never for the primary audit; acceptable for quick sanity checks if flagged |
| Reusing existing 8 benchmarks without large-input variants | No new code to write | Masks O(n²) and O(m³) scaling issues at realistic sizes | Never — at minimum add one large-input variant per hot path |
| Counting scikit-fda API names directly | Automatable, fast | 2–3× inflated gap count; pollutes backlog with out-of-scope items | Never without a relevance filter applied afterward |
| Skipping numerical accuracy check for "existing" capabilities | Saves time | Misses silent correctness gaps like GH #33 (B-spline CV) | Never for capabilities flagged as fragile in CONCERNS.md |
| Vague backlog items ("improve X") | Writes quickly | Not actionable; forces re-audit before implementation | Never — all items must pass the completeness checklist |
| Recording benchmark numbers from a single run | Fast | High variance; findings may reverse on re-run | Never — require 2 independent runs within ±5% before recording |

---

## Integration Gotchas

Specific to the fdars audit context — interactions between tools and the codebase.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Criterion 0.5 + doctest harness | Criterion 0.5 has a known linker issue with doctests on some Linux configurations; `cargo test --doc` may bus-error | Run `cargo test -p fdars-core --lib --features linalg` first; isolate doctest failures separately |
| scikit-fda API scrape | Scraping `scikit-fda.readthedocs.io` without pinning a version may mix stable and dev API | Pin to the latest stable release tag (check `scikit-fda.__version__` in PyPI); document version in the gap matrix |
| nalgebra SVD vs faer Cholesky | Both are used for different operations; a static analysis that conflates them will misattribute bottlenecks | Trace each hot path to its specific linear algebra call: `nalgebra::SVD` (FPCA) vs faer `Cholesky` (ridge, gated by `linalg`) |
| `--features linalg` requires Rust 1.84 | Running on Rust 1.81 with `--features linalg` fails at compile time | Check `rustc --version` before the benchmark phase; document which Rust version was used |
| cargo-flamegraph on Linux | Requires `perf` permissions; may fail without `echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid` | Document this requirement in the benchmark phase plan; use `--root` or adjust perf paranoia level |

---

## Performance Traps Specific to This Audit

Patterns where the measurement itself is misleading.

| Trap | Symptoms | Prevention | Notes |
|------|----------|------------|-------|
| Measuring `FdMatrix::row()` in isolation | Appears O(1); misses that callers call it in an O(n) loop making total O(n*m) | Measure at the call-site loop level, not the primitive | The `row_to_buf` variant avoids allocation; check callers use it |
| Benchmarking small n for elastic alignment | O(n²) looks linear for n ≤ 20 | Add n=50, 100, 200 variants to confirm quadratic growth | Sakoe-Chiba band (v0.14.0) may change the slope |
| SVD benchmark without the copy overhead | Extracting just the `nalgebra::SVD::new()` call misses the `FdMatrix→DMatrix` conversion cost that dominates | Benchmark the full function call including data prep | CONCERNS.md already flags this; the benchmark must reproduce it |
| Rayon overhead on small n | Small-n benchmarks with `--features parallel` appear slower than sequential; analysts conclude "parallelism is broken" | Always test at the threshold size where rayon helps (n ≈ 100+) | CONCERNS.md notes parallel overhead for n < ~100 |

---

## "Looks Done But Isn't" Checklist

Items that appear complete in the audit but are missing critical pieces.

- [ ] **Benchmark results table:** Includes feature-flag column, Rust version, machine spec, and 2-run variance check — verify these fields are present before finalizing the report
- [ ] **Parity matrix:** Has a "Relevance" filter applied (in-scope algorithm vs. out-of-scope plotting/IO) — verify gap count drops when filtering to in-scope only
- [ ] **Parity matrix:** Has an "fdars equivalent searched" note for every gap item — verify no item is "gap" solely because the name doesn't match
- [ ] **Parity matrix:** Has an "accuracy verified?" column — verify fragile areas from CONCERNS.md are not marked ✓ without a verification note
- [ ] **Backlog items:** Every item has function name, current cost, root cause, fix approach, severity, and effort — verify using the completeness checklist
- [ ] **Backlog items:** Evidence artifact (bench output or gap reference) is linked — verify `.planning/research/bench/` exists and is non-empty
- [ ] **Test failure triage:** Infrastructure failures (linker bus errors) are separated from code failures — verify the methodology section documents this distinction
- [ ] **Build mode verification:** Every benchmark result is tagged with `--release`, features used, and rustc version — verify by checking Criterion output headers

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Benchmarks run in debug mode | LOW | Rerun with `--release`; results available in < 1 hour |
| Wrong feature flags on benchmarks | LOW | Rerun with correct feature matrix; update results table |
| Inflated gap count from API-name counting | MEDIUM | Apply capability taxonomy retroactively; re-review ~60 items and collapse to ~20 meaningful groups |
| Vague backlog items | MEDIUM | Return to audit artifacts and fill completeness fields; requires re-reading source code per item |
| Missing `black_box` in benchmarks | LOW | Add `black_box` wrappers; rebuild and rerun |
| Infrastructure failures counted as code failures | LOW | Re-triage failure list; likely removes 3–5 items from the defect count |
| Accuracy gaps missed | HIGH | Requires running fdars + scikit-fda on reference datasets; may find correctness issues requiring code changes (out of audit scope) |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Debug mode benchmarks | Benchmark Confirmation | Build mode check is first line of phase procedure |
| Wrong feature flags | Benchmark Confirmation | Feature matrix table is a required phase artifact |
| Missing `black_box` | Benchmark Confirmation | Code review of existing benchmarks before recording results |
| Unrepresentative input sizes | Static Hot-Path Analysis (define sizes) + Benchmark Confirmation (run them) | Large-input variant present for each hot path |
| Ignoring allocation cost | Static Hot-Path Analysis | Allocation hotspot list is a required deliverable |
| Warm vs. cold cache | Benchmark Confirmation | Cache regime column in results table |
| Noisy machine | Benchmark Confirmation | ±5% variance threshold; 2-run requirement |
| Linker flakiness masking failures | All test-execution phases | Infrastructure vs. code failure triage documented in methodology |
| API-name counting | Gap Analysis | Capability taxonomy defined before enumeration begins |
| "scikit-fda has X → must have X" | Gap Analysis | Design-goal filter applied before analysis |
| Missed equivalent under different name | Gap Analysis | "Searched fdars for equivalent" field required per gap item |
| Missing accuracy parity | Gap Analysis | "Accuracy verified?" column in parity matrix |
| Effort-over-value ranking | Prioritized Backlog | Value estimation step before ranking |
| Plotting/IO inflating gap count | Gap Analysis | Relevance filter; separate gap counts for in-scope vs. out-of-scope |
| Vague backlog items | Prioritized Backlog | Completeness checklist gates each item |
| Missing severity/effort | Prioritized Backlog | Both fields required; definitions anchored to fdars user base |
| Missing evidence artifacts | Benchmark Confirmation + Gap Analysis | `.planning/research/bench/` non-empty; URLs in gap items |
| Feature-flag matrix confusion | Static Hot-Path Analysis + Benchmark Confirmation | Feature matrix table; feature-gated path annotations in static analysis |

---

## Sources

- Codebase knowledge: `.planning/codebase/CONCERNS.md` (performance bottlenecks, known bugs, fragile areas)
- Codebase knowledge: `.planning/codebase/ARCHITECTURE.md` (DMatrix round-trip pattern, parallel macro system)
- Project scope: `.planning/PROJECT.md` (audit constraints, out-of-scope plotting/IO, baseline decisions)
- Known environment issue: Criterion 0.5 linker bus errors documented in environment context
- Domain knowledge: Rust benchmarking best practices (criterion `black_box`, release mode, feature-flag matrix)
- Domain knowledge: API parity analysis methodology (capability vs. name counting, design-goal filtering)

---
*Pitfalls research for: fdars AUDIT milestone — performance auditing and scikit-fda parity analysis*
*Researched: 2026-08-07*
