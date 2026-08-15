---
phase: 20-two-sample-tests-inference-module
status: passed
verified: 2026-08-16
requirements: [INF-01]
plans: ["20-01"]
must_haves_verified: 5
must_haves_total: 5
independently_verified: true
---

# Phase 20 — Two-Sample Functional Tests & `inference/` Module · Verification

**Verdict: PASSED** — 5/5 ROADMAP success criteria satisfied; INF-01 delivered. Additive/non-breaking. Independently re-verified by the orchestrator (not just trusting the executor).

**Deliverable:** new `fdars-core/src/inference/` module (`mod.rs`, `permutation.rs`, `hotelling.rs`, `scb.rs`); 5 public fns + `TestResult` crate-root re-exported.

## Success Criteria

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| SC1 | New `inference/` module; `t_perm_test`, `f_perm_test`, mean/cov test, `mean_scb` crate-root re-exported | ✅ | `lib.rs:92` `pub mod inference`; `lib.rs:224` `pub use inference::{f_perm_test, mean_scb, scb_two_sample_test, t_perm_test, two_sample_mean_test, TestResult}`. |
| SC2 | `t_perm_test`/`f_perm_test` p≈0 for separated, large under null | ✅ | 9 permutation inline tests (separated→p<0.05, null→p>0.1, fixed-seed determinism, `f_perm`↔`fanova` agreement). |
| SC3 | Two-sample mean test rejects on differing means, fails to reject on equal | ✅ | `two_sample_mean_test` (Hotelling T² on shared FPC basis, χ² p-value); 4 tests incl. differ→reject / coincide→fail-to-reject, χ² SF vs known quantiles. |
| SC4 | `mean_scb` bands cover the true mean; SCB two-sample flags a difference | ✅ | 4 SCB tests: coverage at every grid point + difference detection (thin wrapper over `tolerance::degras::scb_mean_degras`). |
| SC5 | All new fns `Result`-returning, input-validated, existing signatures unchanged (additive) | ✅ | invalid-input `Err` tests for all fns; `fanova`/`hotelling_t2`/`scb_mean_degras` public signatures confirmed unchanged (fanova body refactored to call `pub(crate) integrated_f_statistic`, its 4 tests green). |

## Independent verification (orchestrator-run, 2026-08-16)

- `cargo test -p fdars-core --features linalg,parallel --lib inference` → **17 passed, 0 failed**.
- `cargo test -p fdars-core --features linalg,parallel --lib` → **2027 passed, 0 failed** (2010 prior + 17 new; no regressions).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → **clean** (CI gate; lints test code too).
- Crate-root re-exports + unchanged existing signatures confirmed by grep.

## Notes

- No new crate dependency: a self-contained χ² survival function (regularized upper incomplete gamma + Lanczos ln_gamma) was implemented and validated against known χ² quantiles, avoiding a `statrs` API addition.
- Executor self-caught + fixed a non-zero-mean test-noise bug in the SCB coverage helper (`2u−1`); harmless to the permutation/Hotelling tests.
- Nyquist VALIDATION.md not produced (carried-forward `draft` posture from prior implementation milestones).
