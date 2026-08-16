---
phase: 23-functional-boxplot-depth-dispatcher
status: passed
verified: 2026-08-16
requirements: [T-02]
plans: ["23-01"]
must_haves_verified: 5
must_haves_total: 5
independently_verified: true
---

# Phase 23 — Functional Boxplot & Depth Dispatcher · Verification

**Verdict: PASSED** — ROADMAP success criteria satisfied; T-02 delivered. Additive/non-breaking, numeric-only. Independently re-verified by the orchestrator.

**Deliverable:** `DepthMethod`, `functional_depth`, `functional_boxplot`, `FunctionalBoxplotResult` in new `depth/dispatch.rs`; all crate-root re-exported.

## Success Criteria

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| SC1 | Unified `functional_depth(data, DepthMethod)` dispatcher over existing depth fns | ✅ | `functional_depth` (self-depth) + `#[non_exhaustive]` `DepthMethod` (FraimanMuniz/Band/ModifiedBand/RandomProjection); re-exported (`lib.rs:414–418`). |
| SC2 | Dispatcher == underlying fn per method | ✅ | Per-method equality tests: `functional_depth(data, FraimanMuniz{scale})` == `fraiman_muniz_1d(data,data,scale)`, etc.; seed reproducibility for RandomProjection. |
| SC3 | Depth-fence functional boxplot with numeric central-region/whisker/outlier outputs | ✅ | `functional_boxplot` (canonical López-Pintado–Romo: median=deepest, 50% central envelope, fence=factor×central width, outliers=exceed fence); `FunctionalBoxplotResult` (7 fields, serde-gated). Numeric only, no plotting. |
| SC4 | Boxplot flags planted outliers, not inliers; median/central ordering correct | ✅ | Planted-gross-outlier test (flagged), inliers spared, median==deepest curve, central brackets median, fence contains central. |
| SC5 | Additive/`Result`-returning/inline tests/re-export; existing signatures unchanged | ✅ | Dispatcher only wraps `fraiman_muniz_1d`/`band_1d`/`modified_band_1d`/`random_projection_1d_seeded` (signatures byte-identical, grep-confirmed); `outliergram` untouched; error-path tests incl. non-finite `factor`. |

## Independent verification (orchestrator-run, 2026-08-16)

- `cargo test -p fdars-core --features linalg,parallel --lib` → **2061 passed, 0 failed** (2049 after Phase 22 + 12 new).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → **clean** (also gate-enforced on both code commits).
- Crate-root re-exports + `depth/dispatch.rs` + unchanged depth signatures confirmed by grep.

## Notes

- No new dependency. Both public items cohesive in one `depth/dispatch.rs`.
- Nyquist VALIDATION.md not produced (carried-forward draft posture).
