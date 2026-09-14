# Phase 108: Entropy Extraction & SATD/Unwrap Cleanup — Research

**Researched:** 2026-09-14
**Domain:** pmat quality-gate remediation — entropy, SATD, defect
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
All implementation choices are at Claude's discretion — pure cleanup phase.
Constraints (non-negotiable):

- **Strictly behavior-preserving + numeric-preserving.** Entropy extraction = factor repeated
  `.map(...)`/transformation closures into a shared private helper (or a test-helper fn), CALLED
  identically — same computation, same numeric result. Do NOT reorder floating-point accumulations.
  SATD is prose-only (zero code change). The unwrap fix keeps the same result on valid input.
- **No public API / signature / visibility change** where it can be avoided; new entropy helpers
  are private `fn` (or `#[cfg(test)]` test helpers). If a shared helper must be crate-visible to
  be reused across modules, prefer a private module-local helper per file, or a `pub(crate)` helper
  only if truly shared — but never widen an existing PUBLIC signature.
- **`fdars-core` only. No new crate dependency.**
- Proof = existing suite green (2942 lib + 215 doctests + 28 examples, numeric-identical) +
  `pmat analyze entropy` → 0, `pmat analyze satd` → 0, `pmat analyze defects` (the unwrap) → 0.

### Claude's Discretion
All implementation choices are at Claude's discretion — pure cleanup phase.

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
Complexity tail (Phase 109), CI guardrail (110), release (111) are out of scope for this phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| QUAL-04 | Extract repeated `DataTransformation` patterns into shared helpers so `pmat analyze entropy` → 0 violations | Workstream 1: entropy file-by-file analysis, helper placement, non-determinism note |
| QUAL-05 | Reword 2 SATD doc-comment false-positives AND replace the 1 critical `.unwrap()` | Workstream 2: exact SATD lines confirmed; Workstream 3: unwrap invariant analysis |
</phase_requirements>

---

## Summary

Phase 108 targets three independent cleanup workstreams that collectively drive the non-complexity
`pmat quality-gate` findings to zero. Workstream 1 (entropy) is the broadest: pmat 3.18.0 flags
repeated `DataTransformation` patterns across 10–13 files per run (the count is non-deterministic
— see below). Workstream 2 (SATD) is two one-line prose rewrites in doc-comments, confirmed.
Workstream 3 (unwrap) is a single `.expect()` addition at `depth/rpd.rs:229` — an invariant-
guaranteed construct that cannot fail on valid input.

**Critical observation on pmat entropy non-determinism:** Running `pmat quality-gate` (or
`pmat analyze entropy`) five consecutive times from `fdars-core/` produced 10–13 entropy
violations per run with a *different file list every time*. The stable core (appearing in all 5
runs) is: `covariance.rs`, `detrend/tests.rs`, `explain_generic/friedman.rs`, `fem_smoothing.rs`,
`metric/tests.rs`, `seasonal/tests.rs`, plus the project-level diversity finding. Additional files
that appear in most (3–4/5) runs include `clustering.rs`, `elastic_regression/scalar_on_shape.rs`,
`regression.rs`, `simulation.rs`, `landmark.rs`, `explain/tests.rs`,
`scalar_on_function/tests.rs`. The executor must address ALL identified repeated-pattern sites
broadly (not just the 11 from one gate run) to ensure the gate stays green on repeated runs.

**Primary recommendation:** Address all known repeated-pattern files; verify with 3 consecutive
`pmat analyze entropy` runs from `fdars-core/` before calling entropy done.

**pmat quality-gate pass criteria (VERIFIED):** ALL violations — including `warning` severity —
count toward the total. The gate runs 9 checks; it emits `Quality Gate: FAILED` and exits non-zero
if `total_violations > 0`. Both entropy (severity `warning`) and SATD (severity `error`/`info`)
count equally. There is no "warnings-only pass" mode.
[VERIFIED: pmat 3.18.0 JSON output `results.passed: false` when entropy_violations > 0, quality-gate --format json]

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Entropy helper extraction | `fdars-core` private helpers / `#[cfg(test)]` test helpers | `src/test_helpers.rs` (for cross-module test helpers) | Helpers are module-local where possible; cross-module test helpers extend `test_helpers.rs` |
| SATD doc-comment reword | `depth/dispatch.rs` + `metric/lp.rs` | — | Pure `///` text edit, no code layer involved |
| Unwrap fix | `depth/rpd.rs` private fn `build_augmented` | — | Local invariant; fix stays in the same private fn |

---

## Workstream 1 — Entropy

### pmat entropy non-determinism

`pmat analyze entropy` uses stochastic pattern matching — the reported file list differs each run.
Run the gate 3× after fixing and require 0 violations in ALL runs before marking QUAL-04 complete.
[VERIFIED: 5 consecutive runs from `fdars-core/` showed 10–13 violations with different file sets]
[VERIFIED: pmat 3.18.0, `pmat quality-gate --format json` from `/home/simonm/projects/rust/fdars/fdars-core`]

### Quality-gate verdict

```
Quality Gate: FAILED
Total violations: ~110 (varies per run)
entropy_violations: 11–13 per run (all severity: "warning")
```
[VERIFIED: pmat quality-gate --format json from fdars-core, 5 runs, 2026-09-14]

### Complete entropy violation inventory

Files appearing in the stable core (all 5 runs) and additional files seen in 3–5/5 runs are listed
below. Each entry includes: pattern, context (production / cfg(test) / doctest), occurrence count,
and proposed extraction.

---

#### E-01: `src/explain/tests.rs` — y-median binary conversion (cfg(test))

**Pattern (verbatim from pmat example):**
```rust
.map(|&v| if v >= y_median { 1.0 } else { 0.0 })
```
**Context:** `#[cfg(test)]` module (gated via `explain/mod.rs` line 119: `#[cfg(test)] mod tests;`)
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/explain/mod.rs:119-120]

**Occurrences:** 13 lines — all follow the three-line pattern:
```rust
let y_bin: Vec<f64> = y_cont
    .iter()
    .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
    .collect();
```
[VERIFIED: grep in explain/tests.rs returned 13 lines (lines 64,128,146,209,347,459,504,585,649,742,785,1087,1557)]

**Proposed extraction:** Add to `src/test_helpers.rs`:
```rust
/// Binarize a continuous response at its median: values >= median → 1.0, else 0.0.
pub fn binarize_at_median(y: &[f64], threshold: f64) -> Vec<f64> {
    y.iter().map(|&v| if v >= threshold { 1.0 } else { 0.0 }).collect()
}
```
Call site becomes:
```rust
let y_bin = crate::test_helpers::binarize_at_median(&y_cont, y_median);
```
**Risk:** Low — pure value transformation, no floating-point accumulation.

---

#### E-02: `src/metric/tests.rs` — flat sin matrix construction (cfg(test))

**Pattern (verbatim):**
```rust
FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(), n, m)
```
**Context:** `#[cfg(test)]` (gated via `metric/mod.rs` line 33: `#[cfg(test)] mod tests;`)
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/metric/mod.rs:33-36]

**Occurrences:** 9× across lines 125, 162, 200, 302, 332, 375, 402, 423, 458, 496
[VERIFIED: grep in metric/tests.rs]

**Proposed extraction:** Add to `src/test_helpers.rs`:
```rust
/// Build a flat n×m FdMatrix with values `(idx * scale).sin()` (column-major).
pub fn sin_matrix(n: usize, m: usize, scale: f64) -> crate::matrix::FdMatrix {
    crate::matrix::FdMatrix::from_column_major(
        (0..(n * m)).map(|i| (i as f64 * scale).sin()).collect(),
        n,
        m,
    )
    .expect("sin_matrix: n*m matches data length by construction")
}
```
Call site: `sin_matrix(n, m, 0.1)` — same numeric result.
**Risk:** Low.

---

#### E-03: `src/seasonal/tests.rs` — scaled uniform argvals (cfg(test))

**Pattern (verbatim):**
```rust
let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 10.0 / (m - 1) as f64).collect();
```
**Context:** `#[cfg(test)]` (gated via `seasonal/mod.rs` line 23)
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/seasonal/mod.rs:23-24]

**Occurrences:** 8× lines 57, 100, 131, 163, 195, 225, 260, 2757
[VERIFIED: grep in seasonal/tests.rs]

**Proposed extraction:** The existing `test_helpers::uniform_grid(m)` returns `[0,1]` — this one
returns `[0,10]`. Add to `src/test_helpers.rs`:
```rust
/// Generate a uniform grid on [0, end] with n points.
pub fn uniform_grid_range(n: usize, end: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 * end / (n - 1) as f64).collect()
}
```
Call site: `uniform_grid_range(m, 10.0)`.
**Risk:** Low. Note: `uniform_grid(n)` already exists in `test_helpers.rs` (returns `[0,1]`).
`uniform_grid_range(n, 1.0)` is identical to `uniform_grid(n)` — can consolidate later or keep
both for clarity.

---

#### E-04: `src/fem_smoothing.rs` — bounding-box extraction (PRODUCTION)

**Pattern (verbatim, lines 111–114):**
```rust
let x_min = nodes.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
let x_max = nodes.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
let y_min = nodes.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
let y_max = nodes.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
```
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/fem_smoothing.rs:111-114]

**Context:** Production code (test section begins at line 747). This 4-line block appears only
ONCE in the file — pmat is flagging the pattern across the entire codebase with 8 repetitions
(likely including occurrences in other files counted together by pmat's file-level report). The
file itself has only 1 bounding-box block.

**Proposed extraction:** Add a private helper in `fem_smoothing.rs`:
```rust
/// Axis-aligned bounding box of a 2D point set.
fn node_bbox(nodes: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    let x_min = nodes.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    let x_max = nodes.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
    let y_min = nodes.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let y_max = nodes.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
    (x_min, x_max, y_min, y_max)
}
```
Call site:
```rust
let (x_min, x_max, y_min, y_max) = node_bbox(nodes);
```
**Risk:** Low — pure min/max fold, no accumulation order change.
**Note on node type:** Verify the actual type of `nodes` elements at the call site (may be
`Vec<f64>` slices rather than `[f64; 2]`) — adjust generic accordingly. The fold computation
is unchanged.

---

#### E-05: `src/explain_generic/friedman.rs` — scalar covariate row extraction (PRODUCTION)

**Pattern (verbatim):**
```rust
scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
```
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/explain_generic/friedman.rs:34,67,146,164]

**Context:** Production code, repeated 4× within two private helper functions. The pattern extracts
row `i` from an optional `&FdMatrix` into `Option<Vec<f64>>`.

**Proposed extraction:** Add private helper in `src/explain_generic/friedman.rs`:
```rust
/// Extract row `i` of the optional scalar-covariate matrix as an owned Vec.
fn extract_scalar_row(
    scalar_covariates: Option<&crate::matrix::FdMatrix>,
    i: usize,
    p_scalar: usize,
) -> Option<Vec<f64>> {
    scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
}
```
Call site: `extract_scalar_row(scalar_covariates, i, p_scalar)` — identical output type.
**Risk:** Low — no floating-point accumulation, just index reads.

---

#### E-06: `src/metric/soft_dtw.rs` — row extraction (PRODUCTION)

**Pattern (verbatim):**
```rust
let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
```
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/metric/soft_dtw.rs:251,280,822]

**Context:** Production code (test section begins at line 982). Three occurrences in separate
public/private functions.

**Proposed extraction:** Add a private helper in `soft_dtw.rs` (or extend `helpers.rs`):
```rust
fn extract_rows(data: &crate::matrix::FdMatrix) -> Vec<Vec<f64>> {
    (0..data.nrows()).map(|i| data.row(i)).collect()
}
```
Note: `helpers::extract_curves(data)` already does exactly this
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/helpers.rs:45-48 — `data.rows()`].
Check whether `data.rows()` or `helpers::extract_curves(data)` can be substituted directly at
each call site — if `data.row(i)` and `data.rows()` return identical `Vec<Vec<f64>>`, use the
existing helper instead of adding a new one.
**Risk:** Low, but VERIFY that `data.row(i)` and `helpers::extract_curves(data)` produce
numerically identical output before substituting.

---

#### E-07: `src/elastic_regression/scalar_on_shape.rs` — residual computation (PRODUCTION)

**Pattern (verbatim):**
```rust
.map(|(&yi, &gi)| yi - gi)
```
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/elastic_regression/scalar_on_shape.rs:143,185,204]

**Context:** Production code (cfg(test) begins at line 688). Three occurrences computing
`y_i - fitted_i` residuals.

**Proposed extraction:** Add a private helper:
```rust
fn compute_residuals(y: &[f64], fitted: &[f64]) -> Vec<f64> {
    y.iter().zip(fitted.iter()).map(|(&yi, &gi)| yi - gi).collect()
}
```
Call site: `compute_residuals(&y, &fitted)` — identical subtraction order.
**Risk:** Low. Element-wise subtraction, no accumulation.

---

#### E-08: `src/detrend/tests.rs` — scaled uniform argvals variant (cfg(test))

**Pattern (verbatim):**
```rust
let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();
```
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/detrend/tests.rs:7,29,97,127,142,171,200,216,250,268]

**Context:** `#[cfg(test)]` (gated via `detrend/mod.rs` line 23)
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/detrend/mod.rs:23-24]

**Note:** This is algebraically identical to E-03 (seasonal) but written as
`i as f64 / (m-1) as f64 * 10.0` vs `i as f64 * 10.0 / (m-1) as f64`. The two expressions are
NOT floating-point identical (multiplication order differs). Use SEPARATE helpers or verify
bit-identity before sharing. Safest: share `uniform_grid_range(m, 10.0)` from `test_helpers.rs`
and verify that `i * 10.0 / (m-1)` == `i / (m-1) * 10.0` for all valid `m` (they differ by FP
rounding in some cases — inspect both call sites and the test assertions before committing to one
form).

**Risk:** Medium — floating-point multiplication order. Do NOT silently consolidate without
verifying test passes.

---

#### E-09: `src/covariance.rs` — argvals grid in tests (cfg(test))

**Pattern (verbatim):**
```rust
let argvals: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
```
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/covariance.rs:823]

**Context:** `#[cfg(test)]` (gated at line 626)
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/covariance.rs:626]

**Occurrences vary per run (pmat sees 6 total** including other argvals variants at lines 846, 890,
915, 939, 1168, 1184). These are DIFFERENT idioms (different `n` and `step`), not all the same
pattern. Extract the most common variant as `uniform_grid_range(20, 1.9)` or inline
`(0..20).map(|i| i as f64 * 0.1)` is equivalent to `[0.0, 0.1, ..., 1.9]`. If pmat counts
these as a cluster, adding a `test_helpers::uniform_step_grid(n, step)` helper may reduce them.
**Risk:** Low for the first variant; verify each variant before consolidating.

---

#### E-10: `src/landmark.rs` — sin curve in tests (cfg(test))

**Pattern (verbatim):**
```rust
let curve: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
```
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/landmark.rs:574,592,610,665,739,749]

**Context:** `#[cfg(test)]` (gated at line 492)
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/landmark.rs:492]

**Proposed extraction:** Add to `src/test_helpers.rs`:
```rust
/// Sine curve evaluated on a grid: sin(2π·t[i]).
pub fn sine_curve(t: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect()
}
```
**Risk:** Low.

---

#### E-11: `src/scalar_on_function/tests.rs` — y-median binary conversion (cfg(test))

**Pattern:** Same as E-01.
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/scalar_on_function/tests.rs:202,226,256,350,494,519]

**Context:** `#[cfg(test)]` (gated via `scalar_on_function/mod.rs` line 35)
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/scalar_on_function/mod.rs:35]

**Proposed extraction:** Same helper as E-01 (`test_helpers::binarize_at_median`). Both
`explain/tests.rs` and `scalar_on_function/tests.rs` use the same three-line idiom.

---

#### E-12: Occasional-appearing files (2–4/5 runs)

These files surface in some runs but not all. Address them to ensure 3-for-3 post-fix runs pass:

| File | Pattern | Context |
|------|---------|---------|
| `src/simulation.rs` | `let t: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect()` | `#[cfg(test)]` (line 712) — use `uniform_grid(100)` from test_helpers |
| `src/regression.rs` | `///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(),` | doc-comment `///` examples (NOT production, NOT test — compiled as doctests) |
| `src/clustering.rs` | `///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(),` | doc-comment `///` examples (doctests) |
| `src/fof_regression.rs` | `/// let tx: Vec<f64> = (0..mx).map(|j| j as f64 / (mx - 1) as f64)` | doc-comment `///` examples (doctests) |
| `src/pda.rs` | `(0..n_pts).map(|i| i as f64 / (n_pts - 1) as f64)` | mixed: module doc `//!` + `#[cfg(test)]` test code |
| `src/classification/fit.rs` | `(0..d).map(|j| scores[(i, j)])` | production code — score row extraction |
| `src/explain_generic/tests.rs` | y-median binary conversion | `#[cfg(test)]` — same E-01 helper |

**Doc-comment pattern risk:** `regression.rs`, `clustering.rs`, `fof_regression.rs` have repeated
`map` expressions inside `///` doc-comment examples. These ARE compiled and run as doctests.
Reducing them requires editing the doc-comment example code. Two options:
1. **Simplify the examples** — use shorter literal `vec![]` instead of `(0..N).map(...)` where
   possible, making examples less repetitive without breaking doctests.
2. **Accept doc-comment entropy** — pmat may not reduce this to 0 without deeply restructuring
   examples. This is the highest-risk change (breaking a doctest = CI failure). Prioritize
   option 1 for the most-repeated patterns; verify each doctest compiles after editing.

**Risk for doc-comments:** HIGH — must run `cargo test --doc -p fdars-core` after any doc-comment
change to verify doctests still pass.

---

#### E-13: "Low pattern diversity 76.9%" project-level finding

This is a **separate, softer metric** — the project-level entropy score. It does count as a
violation in `pmat quality-gate` (severity `warning`). Fixing the per-file repeated patterns
(E-01 through E-12) will raise the diversity score, but whether it reaches the 100% threshold
is uncertain. **This may require additional consolidation after the file-level fixes.** The
executor should re-check the project-level finding after all file fixes are committed and address
any remaining diversity gap separately.
[VERIFIED: pmat quality-gate JSON shows `project` as an entropy violation with `pattern_diversity: 0.766`]

---

### Entropy grouping recommendation

Because pmat's pattern-detection is non-deterministic, group fixes by natural file clusters rather
than by pmat output:

**Group A — test helpers in `src/test_helpers.rs`** (new or extended):
- `binarize_at_median` (used by explain/tests.rs + scalar_on_function/tests.rs + explain_generic/tests.rs)
- `sine_curve` (landmark.rs tests)
- `uniform_grid_range(n, end)` (seasonal/tests.rs, detrend/tests.rs, simulation.rs)
- `sin_matrix(n, m, scale)` (metric/tests.rs)

**Group B — production-file private helpers** (one per file):
- `fem_smoothing.rs`: `fn node_bbox(nodes)` → returns `(x_min, x_max, y_min, y_max)`
- `explain_generic/friedman.rs`: `fn extract_scalar_row(sc, i, p)` → `Option<Vec<f64>>`
- `metric/soft_dtw.rs`: verify `helpers::extract_curves` equivalence, then substitute
- `elastic_regression/scalar_on_shape.rs`: `fn compute_residuals(y, fitted)` → `Vec<f64>`

**Group C — doc-comment simplification** (careful, doctest risk):
- `regression.rs`, `clustering.rs`, `fof_regression.rs`: simplify `///` examples to reduce
  repeated `(0..N).map(|i| ...)` patterns; run `cargo test --doc` after each file

---

## Workstream 2 — SATD

### Confirmed findings

`pmat analyze satd` from `fdars-core/`: **2 violations in 2 files**.
[VERIFIED: pmat 3.18.0 output, 2026-09-14, from `/home/simonm/projects/rust/fdars/fdars-core`]

```
Severity Distribution
  Critical: 0
  High: 1   ← depth/dispatch.rs:281 ("Defect:" keyword)
  Low:  1   ← metric/lp.rs:175 ("Performance:" keyword)
```

#### SATD-1: `depth/dispatch.rs:281`

**Exact current line:**
```rust
    // Deepest 50% of rows (ceil(n/2)), ties broken by index for determinism.
```
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/depth/dispatch.rs:281]

**Context:** This is a `//` inline comment (NOT a `///` doc-comment — the CONTEXT.md description
of "Defect: Deepest 50% of rows…" refers to what pmat's SATD scanner sees as the SATD text. Let
me reconcile: the quality-gate output says line 281 is:
```
Defect: Deepest 50% of rows (ceil(n/2)), ties broken by index for determinism.
```
Reading the actual file shows that the pmat output is quoting the comment text starting right
after where a doc-comment block ends. The pmat `satd` violation at `dispatch.rs:281` is:
```
./src/depth/dispatch.rs:281 - Defect: Deepest 50% of rows... (severity: High/error)
```
The actual file line 281 reads:
```rust
    // Deepest 50% of rows (ceil(n/2)), ties broken by index for determinism.
```
The word "Defect" does not appear on line 281 itself. The pmat SATD scanner is reporting that the
SATD keyword "Defect" is located at line 281 — which suggests the surrounding context (possibly
from the quality-gate output message which prepends the SATD type keyword). The word that triggers
pmat may be at an adjacent line or is the SATD category label.

**Resolution:** The CONTEXT.md says the SATD keyword is `"Defect:"` in a `///`. Run
`grep -n "Defect:" fdars-core/src/depth/dispatch.rs` to find the exact line at execution time.
The fix is: remove the trigger keyword prefix while keeping the comment meaning. E.g.:
- Before: `/// Defect: Deepest 50% of rows…` (if that's the actual line)
- After: `/// Deepest 50% of rows…` (or `/// Note: deepest 50% of rows…`)

[VERIFIED: pmat analyze satd confirms violation at dispatch.rs:281, severity error (High)]
[ASSUMED: exact line text — executor must `grep -n "Defect:" src/depth/dispatch.rs` at implementation time]

#### SATD-2: `metric/lp.rs:175`

**Exact current lines (VERIFIED):**
```rust
    // Inline the self-distance pattern rather than going through self_distance_matrix
    // to ensure LLVM can fully optimize the tight p=2 inner loop.
```
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/metric/lp.rs:174-175]

**Context:** The pmat SATD output says `Performance:` at column 5 at line 175. Line 175 reads:
`    // to ensure LLVM can fully optimize the tight p=2 inner loop.`
Line 174 reads: `    // Inline the self-distance pattern rather than going through self_distance_matrix`

The word "Performance" does NOT appear on lines 174–175. Similar to SATD-1, the pmat output
message prepends the SATD category keyword. The actual trigger is `grep -n "Performance:"`.

**Resolution:** Run `grep -n "Performance:" src/metric/lp.rs` at execution time to find the exact
trigger. Reword to remove the keyword while keeping the meaning:
- Before: `/// Performance: to ensure LLVM can fully optimize…` (if that's the text)
- After: `/// Ensures LLVM can fully optimize the tight p=2 inner loop.`

[VERIFIED: pmat analyze satd confirms violation at lp.rs:175, severity info (Low)]
[ASSUMED: exact trigger-line text — executor must grep at implementation time]

### SATD verification after fix

After each file edit:
```bash
cd fdars-core && pmat analyze satd
```
Must return: `Found 0 SATD violations in 0 files`

---

## Workstream 3 — Unwrap

### Exact unwrap location

**File:** `fdars-core/src/depth/rpd.rs`
**Line 229 (verbatim):**
```rust
    FdMatrix::from_column_major(aug_data, n, aug_dim).unwrap()
```
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/depth/rpd.rs:229]

### Context

This line is at the end of the private fn `build_augmented(data, argvals, nderiv, aug_dim)`:
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/depth/rpd.rs:129-230]

```rust
fn build_augmented(data: &FdMatrix, argvals: &[f64], nderiv: usize, aug_dim: usize) -> FdMatrix {
    let n = data.nrows();
    // ...
    let mut aug_data = vec![0.0; n * aug_dim];  // ← allocated with exact size
    // ... fills aug_data with n * aug_dim elements ...
    FdMatrix::from_column_major(aug_data, n, aug_dim).unwrap()
}
```

### What the unwrap is on

`FdMatrix::from_column_major(data, nrows, ncols)` returns `Err(FdarError::InvalidDimension)`
iff `data.len() != nrows * ncols`.
[VERIFIED: /home/simonm/projects/rust/fdars/fdars-core/src/matrix.rs:50-63]

**Is the invariant guaranteed?** YES. `aug_data` is allocated as `vec![0.0; n * aug_dim]`
(line 135), giving exactly `n * aug_dim` elements. The subsequent loops only write into it
(never reallocate or truncate). `from_column_major` is called with the same `n` and `aug_dim`.
Therefore `aug_data.len() == n * aug_dim` is a construction invariant — the `Err` branch is
unreachable.

### Proposed fix

Replace the `.unwrap()` with `.expect()` documenting the invariant:
```rust
FdMatrix::from_column_major(aug_data, n, aug_dim)
    .expect("aug_data length == n * aug_dim by construction")
```

**Why `.expect()` and not `?`:** The function `build_augmented` returns `FdMatrix` (not
`Result<FdMatrix, ...>`). Changing the signature to return `Result` would require changing the
caller (`rpd_depth_1d_seeded`) and potentially surfacing through the public API. The CONTEXT.md
constraint ("no public API/signature change") and the fact that this is genuinely invariant-
guaranteed make `.expect()` the correct choice.

**Behavior on valid input:** Identical. `.expect()` panics only if the invariant is violated
(which cannot happen). On the successful path, the returned `FdMatrix` is bit-identical.

### Defect verification after fix

```bash
cd fdars-core && pmat analyze defects
# Must return: Total Defects: 0
```
[VERIFIED: pmat analyze defects (exit 1) confirms exactly 1 CRITICAL defect at depth/rpd.rs:229]

---

## Standard Stack

No new dependencies. All fixes use existing crate primitives only.

| Tool | Version | Purpose |
|------|---------|---------|
| `pmat` | 3.18.0 | Analysis and gate verification |
| `cargo` | (project toolchain) | Build, test, clippy, fmt |

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Grid generation | New `linspace` utility | `test_helpers::uniform_grid(n)` (already exists) or `uniform_grid_range` extension |
| Row extraction from FdMatrix | Custom loop | `data.rows()` or `helpers::extract_curves(&data)` [VERIFIED: helpers.rs:45-48] |
| Test matrix construction | Ad-hoc vec! macros | `sin_matrix` helper in test_helpers |

---

## Common Pitfalls

### Pitfall 1: Floating-Point Order Difference Between E-03 and E-08

**What goes wrong:** `(0..m).map(|i| i as f64 * 10.0 / (m-1) as f64)` (seasonal) vs
`(0..m).map(|i| i as f64 / (m-1) as f64 * 10.0)` (detrend) differ in FP multiplication order.
Sharing one helper that uses the wrong order will silently change test numerics.
**How to avoid:** Implement `uniform_grid_range(n, end)` as `i as f64 * end / (n-1) as f64`
(the seasonal form) and verify that detrend tests still pass — if they fail, keep separate forms
or use `(i as f64 / (n-1) as f64) * end`.

### Pitfall 2: Doctest Breakage from Doc-Comment Edits

**What goes wrong:** Simplifying `///` example code in `regression.rs`, `clustering.rs`, or
`fof_regression.rs` to reduce repeated `.map(|i| ...)` patterns can break the doctest if the
simplified form doesn't compile or produces different output.
**How to avoid:** Run `cargo test --doc -p fdars-core --features linalg,parallel` after EVERY
doc-comment edit. Do not batch doc-comment edits across files.

### Pitfall 3: Sharing `extract_curves` When `row(i)` and `rows()` Differ

**What goes wrong:** `data.row(i)` returns `Vec<f64>` for row `i`; `helpers::extract_curves(data)`
calls `data.rows()`. If these return different orderings (e.g., one transposed), substituting
`extract_curves` in `soft_dtw.rs` silently breaks DTW computation.
**How to avoid:** Before substituting, write a one-line test asserting
`data.row(0) == helpers::extract_curves(&data)[0]` for a known matrix.

### Pitfall 4: pmat Non-Determinism Causing False "Done" Assessment

**What goes wrong:** After fixing files, a single run of `pmat analyze entropy` returns 0 but the
next run returns 3 violations (from the stochastic sampling).
**How to avoid:** Run `pmat quality-gate` 3 consecutive times from `fdars-core/`. Require 0
entropy violations in ALL three runs before marking QUAL-04 complete.

### Pitfall 5: Changing `build_augmented` Return Type for the Unwrap

**What goes wrong:** Changing `build_augmented` to return `Result<FdMatrix, FdarError>` and
propagating `?` changes callers (`rpd_depth_1d_seeded`) from returning `Vec<f64>` to potentially
returning `Result`, cascading into public API changes.
**How to avoid:** Use `.expect("<invariant message>")` — not `?` — since the error is
unreachable and the function signature must not change.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + criterion 0.5 |
| Config file | none |
| Quick run command | `cargo test -p fdars-core --features linalg,parallel` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` (same) |
| Doctest command | `cargo test --doc -p fdars-core --features linalg,parallel` |
| Clippy | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` |
| Fmt | `cargo fmt --check` |
| pmat gate | `cd fdars-core && pmat quality-gate` (run 3× for entropy stability) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| QUAL-04 | Entropy violations → 0 | pmat | `cd fdars-core && pmat analyze entropy` (must say 0 violations, 3× runs) |
| QUAL-04 | Behavior-preservation | existing suite | `cargo test -p fdars-core --features linalg,parallel` (all pass) |
| QUAL-04 | Doctest preservation | doctests | `cargo test --doc -p fdars-core --features linalg,parallel` |
| QUAL-05a | SATD violations → 0 | pmat | `cd fdars-core && pmat analyze satd` (0 violations) |
| QUAL-05b | Defect violations → 0 | pmat | `cd fdars-core && pmat analyze defects` (0 violations) |
| QUAL-05b | Unwrap behavior identical | existing tests | depth module tests pass — `cargo test -p fdars-core --features linalg,parallel depth` |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg,parallel` + `pmat analyze satd` (fast)
- **Per wave merge:** Full suite + `pmat quality-gate` 3× from `fdars-core/`
- **Phase gate:** All 3 pmat sub-metrics → 0 (3× gate runs), full test suite green

### Wave 0 Gaps

None — existing test infrastructure covers all phase requirements. No new test files needed;
entropy helpers become call sites within existing test functions.

---

## Security Domain

No security-relevant changes. This phase makes no networking, auth, crypto, or data-handling
changes. ASVS categories V2/V3/V4/V6 are not applicable. V5 (input validation): no changes to
public-facing validation logic — internal helper extraction only.

---

## Proposed Plan Grouping

**Plan 01 — SATD + Unwrap (tracer, tiny, ~15 min)**
- Edit `depth/dispatch.rs`: find and reword the SATD trigger line
- Edit `metric/lp.rs`: find and reword the SATD trigger line
- Edit `depth/rpd.rs:229`: replace `.unwrap()` with `.expect("aug_data length == n * aug_dim by construction")`
- Verify: `pmat analyze satd` → 0, `pmat analyze defects` → 0, `cargo test depth metric`
- Files: `src/depth/dispatch.rs`, `src/metric/lp.rs`, `src/depth/rpd.rs`
- Zero overlap with Plan 02.

**Plan 02 — Entropy: test-helper extensions**
- Extend `src/test_helpers.rs` with: `binarize_at_median`, `sine_curve`, `uniform_grid_range`,
  `sin_matrix`
- Update all `#[cfg(test)]` call sites across:
  - `explain/tests.rs` (E-01)
  - `scalar_on_function/tests.rs` (E-11)
  - `explain_generic/tests.rs` (if flagged)
  - `metric/tests.rs` (E-02)
  - `seasonal/tests.rs` (E-03)
  - `detrend/tests.rs` (E-08) — verify FP order first
  - `landmark.rs` test section (E-10)
  - `simulation.rs` test section (use `uniform_grid(100)`)
  - `covariance.rs` test section (E-09)
  - `pda.rs` test section
- Files: `src/test_helpers.rs` + test-module call sites
- Zero overlap with Plan 01 and Plan 03.

**Plan 03 — Entropy: production-code private helpers**
- `src/fem_smoothing.rs`: add `fn node_bbox`, update call site (E-04)
- `src/explain_generic/friedman.rs`: add `fn extract_scalar_row`, update 4 call sites (E-05)
- `src/metric/soft_dtw.rs`: substitute `helpers::extract_curves` or add `fn extract_rows` (E-06)
- `src/elastic_regression/scalar_on_shape.rs`: add `fn compute_residuals`, update 3 call sites (E-07)
- `src/classification/fit.rs`: extract score-row pattern if flagged
- Files: production-only, no test helper changes
- Zero overlap with Plan 01 and Plan 02.

**Plan 04 — Entropy: doc-comment simplification (optional / last)**
- Simplify `///` example code in `regression.rs`, `clustering.rs`, `fof_regression.rs` to reduce
  repeated `.map(|i| ...)` patterns
- Run `cargo test --doc` after EACH file
- Address project-level diversity finding if still present after Plans 02–03
- This plan is optional IF the gate is green after Plans 01–03 on 3 consecutive runs.

---

## Verify Commands (Complete List)

```bash
# From fdars-core/ directory
pmat analyze entropy
pmat analyze satd
pmat analyze defects
pmat quality-gate        # run 3× for entropy stability
pmat quality-gate        # second run
pmat quality-gate        # third run

# From workspace root
cargo test -p fdars-core --features linalg,parallel
cargo test --doc -p fdars-core --features linalg,parallel
cargo clippy --all-targets --features linalg,parallel -- -D warnings
cargo fmt --check
```

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Exact trigger text on `depth/dispatch.rs:281` is `"Defect:"` in a `///` comment | Workstream 2 / SATD-1 | Executor may grep wrong keyword; fix: run `grep -n "Defect:" src/depth/dispatch.rs` first |
| A2 | Exact trigger text on `metric/lp.rs:175` is `"Performance:"` in a `///` comment | Workstream 2 / SATD-2 | Same — run `grep -n "Performance:" src/metric/lp.rs` first |
| A3 | `data.rows()` / `helpers::extract_curves(data)` returns same values as `(0..n).map(|i| data.row(i)).collect()` in soft_dtw.rs | Workstream 1 / E-06 | Silent numeric change in DTW computation; verify with one-line assertion test |
| A4 | The project-level "Low pattern diversity" finding will clear after per-file extractions | Workstream 1 / E-13 | Gate still fails on diversity even after file fixes; requires Plan 04 doc-comment work |

---

## Environment Availability

| Dependency | Required By | Available | Version |
|------------|------------|-----------|---------|
| pmat | All 3 workstreams (verification) | ✓ | 3.18.0 |
| cargo | Build + test | ✓ | (project toolchain 1.97.0) |
| Rust stable | Compilation | ✓ | 1.97.0 |

[VERIFIED: `pmat --version` returns `pmat 3.18.0`, 2026-09-14]

---

## Sources

### Primary (HIGH confidence)
- [VERIFIED: pmat 3.18.0 JSON output] — quality-gate full report, entropy violations, SATD violations, defect violations
- [VERIFIED: source file reads] — rpd.rs:229, dispatch.rs:278-285, lp.rs:172-178, matrix.rs:50-63, test_helpers.rs, helpers.rs:45-48
- [VERIFIED: grep across fdars-core src] — occurrence counts for all repeated patterns

### Secondary (MEDIUM confidence)
- [CITED: 108-CONTEXT.md] — workstream definitions, constraints, guidance
- [CITED: REQUIREMENTS.md] — QUAL-04, QUAL-05 acceptance criteria

## Metadata

**Confidence breakdown:**
- Entropy file list: MEDIUM — non-deterministic between runs; stable core HIGH, occasional files MEDIUM
- SATD exact lines: HIGH for file/line, MEDIUM for exact trigger text (grep recommended at execution time)
- Unwrap invariant analysis: HIGH — construction invariant is clear from source
- Quality-gate pass criteria: HIGH — JSON output confirms all violations count

**Research date:** 2026-09-14
**Valid until:** Permanent for this milestone (pmat 3.18.0 thresholds stable within v0.46.0)
