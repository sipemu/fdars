---
phase: 21-functional-linear-model-inference
status: passed
verified: 2026-08-16
requirements: [INF-02]
plans: ["21-01"]
must_haves_verified: 4
must_haves_total: 4
independently_verified: true
---

# Phase 21 — Functional-Linear-Model Inference · Verification

**Verdict: PASSED** — 4/4 ROADMAP success criteria satisfied; INF-02 delivered. Additive/non-breaking. Independently re-verified by the orchestrator.

**Deliverable:** `flm_f_test`, `flm_gof_test`, `oneway_anova_vstat` added to `fdars-core/src/inference/` (`flm.rs`, `anova.rs`, shared `dist.rs`); all crate-root re-exported.

## Success Criteria

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| SC1 | `flm_f_test` rejects a genuine functional effect, fails to reject a null effect | ✅ | Overall-significance F `(R²/p)/((1−R²)/(n−p−1))` via self-contained `f_sf`; tests p<0.05 (real effect) / p>0.20 (null). |
| SC2 | `flm_gof_test` fails to reject a well-specified FLM, flags lack-of-fit otherwise | ✅ | Ramsey-RESET-style residual lack-of-fit (residuals ~ cubic of fitted, joint-F vs F(3,n−4)), documented in rustdoc; tests p>0.10 (well-specified) / p<0.05 (mis-specified). |
| SC3 | `oneway_anova_vstat` (asymptotic V-stat) agrees with permutation ANOVA, added alongside `fanova` | ✅ | Simpson-integrated between-group V-stat + Satterthwaite scaled-χ² p-value; direction-agrees with `fanova` (rejects separated, fails-to-reject pooled); `fanova` unchanged. |
| SC4 | Reuses fitted residuals + integration weights; `Result`-returning, crate-root re-exported, inline tests, non-breaking | ✅ | Reuses `FregreLmResult.{residuals,fitted_values,r_squared,ncomp}`, `simpsons_weights`, `integrated_f_statistic`; all `Result`; re-exported at `lib.rs:225`. |

## Independent verification (orchestrator-run, 2026-08-16)

- `cargo test -p fdars-core --features linalg,parallel --lib` → **2039 passed, 0 failed** (2027 after Phase 20 + 12 new).
- `cargo test ... --lib inference` → **29 inference-module tests pass** (17 INF-01 + 12 INF-02).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → **clean**.
- `fanova` public signature byte-identical (only `compute_group_means` widened private→`pub(crate)`, additive); 3 fns re-exported confirmed by grep.

## Notes

- No new crate dependency: F-distribution SF (regularized incomplete beta) added self-contained in a new crate-internal `inference/dist.rs`, which also absorbs the Phase-20 gamma/χ² SF (private refactor, zero behavior change; `hotelling.rs` re-points).
- One rustdoc intra-doc link demoted to plain text (public fn must not link a crate-internal item under `-D warnings`).
- Nyquist VALIDATION.md not produced (carried-forward `draft` posture).
