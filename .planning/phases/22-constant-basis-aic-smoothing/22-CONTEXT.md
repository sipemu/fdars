# Phase 22: Constant Basis & AIC Smoothing Selection - Context

**Gathered:** 2026-08-16
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — API defaults follow existing `basis/`/`smoothing` conventions; reuse anchors verified.

<domain>
## Phase Boundary

Two small additive capabilities (T-01), both wrapping existing infrastructure: (1) a named **constant/intercept basis** constructor in `basis/`, and (2) an **AIC criterion** in the automatic smoothing-parameter selection (kernel-bandwidth via `CvCriterion`, and basis-roughness-penalty via `smooth_basis`), which today do GCV/CV only. Covers T-01. Does NOT touch depth/boxplot (T-02 → Phase 23). Independent of Phase 23.
</domain>

<decisions>
## Implementation Decisions

### Constant/intercept basis
- Add `constant_basis(t: &[f64]) -> Vec<f64>` to `basis/` (a new `basis/constant.rs` or into an existing basis file) returning a column-major m×1 matrix of ones — matching the `Vec<f64>`-returning convention of `bspline_basis`/`fourier_basis` (basis functions return flattened basis matrices, NOT a trait object). Crate-root re-export. This is the intercept column usable in a regression design matrix.
- Rationale: fdars' basis layer is function-based (no `Basis` trait), so a `constant_basis` function is the idiomatic form; keep it trivial and correct (m rows × 1 column of 1.0).

### AIC smoothing criterion
- **Kernel-bandwidth path:** add an `Aic` variant to `CvCriterion` (`smoothing.rs:527`) and implement its criterion value in the `optim_bandwidth` dispatch (`smoothing.rs:687`), reusing the hat-matrix trace already computed for GCV (`smoothing.rs:627`). Standard smoother AIC: `AIC = n·ln(RSS/n) + 2·tr(S)` (df = trace of the smoother matrix). (`CvCriterion` is a plain public enum, not `#[non_exhaustive]`; adding a variant is an acceptable additive change for this 0.x **minor** release — 0.19→0.20. Optionally add `#[non_exhaustive]` to it going forward — executor's discretion.)
- **Basis-roughness-penalty path:** add an AIC-based λ selector alongside `smooth_basis_gcv` (e.g. a `smooth_basis_aic`, or a criterion parameter) that searches the same log-λ grid but minimizes AIC (using the penalized-fit trace as df). Keep `smooth_basis_gcv`'s signature unchanged.
- **Latitude:** exact AIC formula variant (n·ln(RSS/n)+2·df vs the equivalent RSS/(n·(1−df/n)²)-adjacent forms) and whether the basis-path AIC is a new function vs a criterion arg are executor's discretion — but the inline tests MUST validate the AIC selection against a brute-force AIC grid search (the selected λ/bandwidth = argmin of the explicitly-computed AIC over the grid).

### Conventions
- Public fns `Result<T, FdarError>` where fallible (constant_basis may be infallible → return `Vec<f64>` directly like its neighbors, or `Result` if it validates `t`); validate inputs; inline `#[cfg(test)] mod tests`; crate-root re-export new public items; keep existing public signatures unchanged (GCV/CV paths + `smooth_basis_gcv` untouched). No `#[must_use]` on `Result`-returning fns.

### Claude's Discretion
- File placement within `basis/` and `smoothing`/`smooth_basis`; whether `constant_basis` returns `Vec<f64>` or `Result`; exact AIC formula; new-function-vs-criterion-arg for the basis path.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets (verified)
- Basis constructors (functions → `Vec<f64>`): `bspline_basis` (`basis/bspline.rs:100`), `fourier_basis` (`basis/fourier.rs:22`). Constant basis mirrors these.
- `CvCriterion { Cv, Gcv }` (`smoothing.rs:527`, plain public enum); `optim_bandwidth` dispatch (`smoothing.rs:687`); GCV hat-matrix trace `trace_s` (`smoothing.rs:627`).
- `smooth_basis` (`smooth_basis.rs:174`) + `smooth_basis_gcv` (log-λ grid GCV selection); `bspline_penalty_matrix`/`fourier_penalty_matrix` for the penalized fit.

### Established Patterns
- `Result<T, FdarError>`, input validation, inline tests, crate-root re-export. CI: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`. Build/test: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` (MEMORY).
- Docs-only commits `--no-verify`; code commits pass the gate (or `--no-verify` after manual test+clippy confirmation if the pre-commit hook times out / hits the /tmp doctest issue).

### Integration Points
- Independent of Phase 23 (disjoint modules). Both additive.
</code_context>

<specifics>
## Specific Ideas

- Test correctness (mandatory): constant basis is an m×1 all-ones column; an intercept-only fit using it reproduces the response mean; `CvCriterion::Aic` selection == argmin of an explicitly-computed AIC grid (and differs from GCV on a case where they diverge); AIC prefers a smoother fit than an over-fit λ on noisy data.
</specifics>

<deferred>
## Deferred Ideas

- Depth-fence functional boxplot + `functional_depth` dispatcher → Phase 23 (T-02).
- Other smoothing criteria (FPE/Shibata/Rice) → out of scope (separate backlog item).
</deferred>
