---
phase: 22-constant-basis-aic-smoothing
status: passed
verified: 2026-08-16
requirements: [T-01]
plans: ["22-01"]
must_haves_verified: 5
must_haves_total: 5
independently_verified: true
---

# Phase 22 — Constant Basis & AIC Smoothing Selection · Verification

**Verdict: PASSED** — ROADMAP success criteria satisfied; T-01 delivered. Additive/non-breaking. Independently re-verified by the orchestrator.

**Deliverable:** `constant_basis` (`basis/constant.rs`), `CvCriterion::Aic` + `aic_smoother` (`smoothing.rs`), `smooth_basis_aic` (`smooth_basis.rs`); all crate-root re-exported.

## Success Criteria

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| SC1 | Named constant/intercept basis in `basis/`, crate-root re-exported | ✅ | `constant_basis` (`basis/constant.rs`), m×1 all-ones column mirroring `bspline_basis`/`fourier_basis`; re-exported at `lib.rs:442`. |
| SC2 | Constant basis integrates into a design matrix (intercept-only fit = response mean) | ✅ | Inline `intercept-mean identity` test. |
| SC3 | AIC criterion in the automatic smoothing-parameter selector (kernel + basis paths) | ✅ | `CvCriterion::Aic` + `aic_smoother` (`AIC = n·ln(RSS/n)+2·tr(S)`, reuses GCV hat-matrix trace) in `optim_bandwidth`; `smooth_basis_aic` λ selector over the log-λ grid minimizing `SmoothBasisResult.aic`. Re-exported. |
| SC4 | AIC selection matches a brute-force AIC grid search | ✅ | Brute-force AIC-grid argmin tests for both `aic_smoother` and `smooth_basis_aic`; AIC-vs-GCV divergence test; hand-computed AIC fixture. |
| SC5 | Additive/`Result`-returning/inline tests/re-export; existing signatures unchanged | ✅ | `cv_smoother`/`gcv_smoother`/`optim_bandwidth`/`smooth_basis`/`smooth_basis_gcv` untouched (dedicated GCV regression-guard test); `CvCriterion` marked `#[non_exhaustive]` so the variant add is future-proof. |

## Independent verification (orchestrator-run, 2026-08-16)

- `cargo test -p fdars-core --features linalg,parallel --lib` → **2049 passed, 0 failed** (2039 after v0.19.0 + 10 new).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → **clean**.
- Crate-root re-exports (`constant_basis`, `smooth_basis_aic`, `aic_smoother`, `CvCriterion`) + `#[non_exhaustive]` confirmed by grep.

## Notes

- No new dependency. `BasisCriterion` helper added for the basis-path selector.
- A `state.advance-plan` schema-mismatch note from the executor (older prose-format STATE handler) was benign — the frontmatter progress block is correct and other handlers applied.
- Nyquist VALIDATION.md not produced (carried-forward draft posture).
