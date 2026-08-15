# Phase 20: Two-Sample Functional Tests & `inference/` Module - Context

**Gathered:** 2026-08-15
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — two material API decisions resolved by user; reuse anchors verified against the codebase.

<domain>
## Phase Boundary

Create fdars' first standalone functional-inference surface: a new `fdars-core/src/inference/` module exposing two-sample functional hypothesis tests. Covers INF-01. Additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exported. Does NOT cover FLM inference (INF-02 → Phase 21) or ITP (INF-03, deferred).
</domain>

<decisions>
## Implementation Decisions

### SCB construction (user-resolved)
- **`mean_scb` reuses the existing `tolerance::degras::scb_mean_degras`** (already implemented in `tolerance/degras.rs:82`). Surface it as an inference-module entry point (thin wrapper / re-export with an inference-friendly signature); do NOT reimplement the band math. The SCB-based two-sample test builds on the same degras machinery (band around the difference of means).

### Permutation-test conventions (user-resolved)
- `t_perm_test` / `f_perm_test`: default **`n_perm = 999`**, a **deterministic `seed` parameter** (fdars `StdRng::seed_from_u64(seed)` convention; per-thread `seed + k` if parallelized), returning a **`TestResult { statistic: f64, p_value: f64 }`** struct (Debug/Clone/PartialEq, serde-gated). All `Result`-returning with input validation.

### Reuse anchors (verified in codebase — do NOT re-derive)
- **`f_perm_test`**: lift the permutation-F machinery already in `function_on_scalar::fanova(data, groups, n_perm) -> FanovaResult` (`function_on_scalar.rs:771`, integrated F-statistic + permutation p-value). Two-sample is the k=2 case; expose a standalone `f_perm_test(data_a, data_b, n_perm, seed)` that assembles a 2-group problem and reuses the integrated-F permutation logic (factor the core out of `fanova` if cleaner, but keep `fanova`'s public signature unchanged).
- **`t_perm_test`**: functional two-sample t-permutation — integrated |t(t)| or L2-of-difference statistic, permutation null by relabeling. Simpson weights via `helpers::simpsons_weights`; sample means via `fdata::mean_1d`.
- **`two_sample_mean_test`**: expose `spm::stats::hotelling_t2` (`spm/stats.rs:86`, on FPC scores + eigenvalues) as a standalone two-sample mean test (project both samples to a shared FPC basis, Hotelling T² on the score difference). Reuse `regression::fdata_to_pc_1d` for the projection.
- **`mean_scb` + SCB two-sample test**: `tolerance::degras::scb_mean_degras`.

### API shape
- Module `inference/` with `mod.rs` re-exporting: `t_perm_test`, `f_perm_test`, `two_sample_mean_test`, `mean_scb`, `scb_two_sample_test`, and `TestResult`. Crate-root re-export in `lib.rs`.
- Parameter order follows fdars convention: `(data.../groups, [argvals,] n_perm, seed)`.

### Claude's Discretion
- Exact statistic form (integrated-F vs L2), whether to factor a shared helper out of `fanova`, the `TestResult` field set (may add `n_perm`/`reject` if useful), and internal file split within `inference/`.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets (verified)
- `function_on_scalar::fanova(data, groups, n_perm) -> FanovaResult` — permutation-F machinery (`function_on_scalar.rs:771`); `FanovaResult` carries p-value + n_perm.
- `spm::stats::hotelling_t2(scores, eigenvalues) -> Result<Vec<f64>>` (`spm/stats.rs:86`) + `hotelling_t2_regularized`.
- `tolerance::degras::scb_mean_degras(...)` (`tolerance/degras.rs:82`) — bootstrap SCB for the mean.
- `regression::fdata_to_pc_1d` — FPCA projection for the Hotelling path.
- `fdata::mean_1d` (`fdata.rs:167`), `helpers::simpsons_weights` (`helpers.rs:57`).
- RNG convention: `StdRng::seed_from_u64(seed)`; parallel loops seed `seed + k` (see `parallel.rs` macros).

### Established Patterns
- New public fns: `Result<T, FdarError>`, dimension/param validation at entry, `#[derive(Debug, Clone, PartialEq)]` on result structs, `#[cfg_attr(feature = "serde", ...)]`, crate-root re-export. NO `#[must_use]` on `Result`-returning fns (clippy::double_must_use under `-D warnings`).
- Inline `#[cfg(test)] mod tests`. CI runs `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (lint test code too).

### Integration Points
- `inference/` module is created here and REUSED by Phase 21 (INF-02 adds `flm_gof_test`/`flm_f_test`/`oneway_anova_vstat` into the same module).
</code_context>

<specifics>
## Specific Ideas

- Statistical sanity in inline tests: p-value ≈ 0 (small) for clearly-separated samples; ~large/uniform under the null (identical or exchangeable samples); determinism under a fixed seed; SCB covers the true mean at the requested level.
</specifics>

<deferred>
## Deferred Ideas

- FLM goodness-of-fit / F-test + asymptotic ANOVA V-statistic → Phase 21 (INF-02).
- Interval Testing Procedure (ITP) family → INF-03 (v2, deferred).
</deferred>
