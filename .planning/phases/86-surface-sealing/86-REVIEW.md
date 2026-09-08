---
phase: 86-surface-sealing
reviewed: 2026-09-08T21:40:00Z
depth: deep
files_reviewed: 41
files_reviewed_list:
  - fdars-core/benches/boosting_regression_benchmarks.rs
  - fdars-core/benches/shapelet.rs
  - fdars-core/examples/27_spm/main.rs
  - fdars-core/src/alignment/bayesian.rs
  - fdars-core/src/alignment/clustering.rs
  - fdars-core/src/alignment/diagnostics.rs
  - fdars-core/src/alignment/lambda_cv.rs
  - fdars-core/src/alignment/multires.rs
  - fdars-core/src/alignment/outlier.rs
  - fdars-core/src/alignment/partial_match.rs
  - fdars-core/src/alignment/robust_karcher.rs
  - fdars-core/src/alignment/shape_ci.rs
  - fdars-core/src/alignment/transfer.rs
  - fdars-core/src/boosting_regression/boost_fofr.rs
  - fdars-core/src/boosting_regression/boost_fosr.rs
  - fdars-core/src/boosting_regression/mod.rs
  - fdars-core/src/classification/fit.rs
  - fdars-core/src/elastic_regression/mod.rs
  - fdars-core/src/kshape.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/outliers.rs
  - fdars-core/src/peer.rs
  - fdars-core/src/shapelet/classifier.rs
  - fdars-core/src/shapelet/discovery.rs
  - fdars-core/src/shapelet/transform.rs
  - fdars-core/src/smooth_basis.rs
  - fdars-core/src/spm/amewma.rs
  - fdars-core/src/spm/arl.rs
  - fdars-core/src/spm/cusum.rs
  - fdars-core/src/spm/elastic_spm.rs
  - fdars-core/src/spm/ewma.rs
  - fdars-core/src/spm/frcc.rs
  - fdars-core/src/spm/iterative.rs
  - fdars-core/src/spm/mewma.rs
  - fdars-core/src/spm/mfpca.rs
  - fdars-core/src/spm/mod.rs
  - fdars-core/src/spm/partial.rs
  - fdars-core/src/spm/phase.rs
  - fdars-core/src/spm/profile.rs
  - fdars-core/src/wire.rs
  - fdars-core/tests/validate_spm_math.rs
findings:
  critical: 0
  warning: 1
  info: 1
  total: 2
status: issues_found
---

# Phase 86: Code Review Report

**Reviewed:** 2026-09-08T21:40:00Z
**Depth:** deep (cross-file semantic equivalence trace)
**Files Reviewed:** 41
**Status:** issues_found — 1 Warning, 1 Info; no correctness or semantic-equivalence defects found

## Summary

Reviewed commit ae0fbe34. The refactor seals `pub mod wire` to `pub(crate)`, adds `#[non_exhaustive]` to 38 public `*Config` structs, and migrates every external-crate construction site. All migration bases were verified to be `Default::default()` — no silent behavioral change introduced. Attribute placement (`#[non_exhaustive]` above `#[derive(...)]` above `#[cfg_attr(...)]`) is correct throughout. No `derive` or `cfg_attr` was dropped.

One apparent alarming artifact in `validate_spm_math.rs` turns out to be valid Rust; see WR-01 for the rationale and the case for fixing it regardless. The `__cfg17`/`__cfg18`/`__cfg19` variable names in the bench files are unusual but scoped to inner blocks and carry no risk.

## Warnings

### WR-01: Inline comment embedded inside field-access expression (`__cfg1.// …\nweighted`)

**File:** `fdars-core/tests/validate_spm_math.rs:342-343`

**Issue:** The automated migration converted

```rust
let config = MfpcaConfig {
    ncomp: n - 1, // use all available
    weighted: true,
};
```

into

```rust
let config = {
    let mut __cfg1 = MfpcaConfig::default();
    __cfg1.ncomp = n - 1;
    __cfg1.// use all available
    weighted = true;
    __cfg1
};
```

The inline comment `// use all available` was copied verbatim but placed after the dot of `__cfg1.`, making the statement read as `__cfg1.<line-comment><newline>weighted = true`. Rust's parser does accept whitespace and line comments between a dot and the field identifier, so this compiles and is semantically equivalent to `__cfg1.weighted = true;`. However, the statement is visually indistinguishable from a broken expression — a future reader (or a naive linter) will be alarmed, and `cargo fmt` cannot normalize it. The comment intent (explaining `ncomp`) is also now attached to the wrong statement.

**Fix:** Split cleanly into two statements, moving the comment to the preceding line:

```rust
__cfg1.ncomp = n - 1; // use all available
__cfg1.weighted = true;
```

## Info

### IN-01: Generated variable names in bench files (`__cfg17`, `__cfg18`, `__cfg19`) are double-underscore-prefixed

**File:** `fdars-core/benches/boosting_regression_benchmarks.rs:78`, `fdars-core/benches/shapelet.rs:45-54`

**Issue:** The migration tool generated names with `__` prefix (e.g. `let mut __cfg17 = BoostingConfig::default()`). In Rust, `__`-prefixed identifiers are reserved by convention for compiler/macro internals. They are not keyword-reserved at the language level, so this compiles cleanly, but the names will look odd to any contributor who opens these files. Because the variables are scoped to anonymous blocks (`let config50 = { ... }`), there is no shadowing risk.

**Fix:** Rename to readable names at next touch:

```rust
// benches/boosting_regression_benchmarks.rs
let config50 = {
    let mut cfg = BoostingConfig::default();
    cfg.mstop = 50;
    cfg
};

// benches/shapelet.rs
let cfg = {
    let mut discovery = ShapeletDiscoveryConfig::default();
    discovery.min_length = 3;
    discovery.max_length = 6;
    discovery.max_candidates = Some(500);
    discovery.max_shapelets = 4;
    discovery.seed = 0;
    let mut cfg = ShapeletClassifierConfig::default();
    cfg.discovery = discovery;
    cfg
};
```

---

_Reviewed: 2026-09-08T21:40:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
