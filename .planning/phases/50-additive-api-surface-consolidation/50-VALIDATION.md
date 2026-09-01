---
phase: 50
slug: additive-api-surface-consolidation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-09-01
---

# Phase 50 — Validation Strategy

> ADDITIVE-ONLY API pass. Every change is a new symbol (a `Default` impl, `fanova_seeded`, a `Dim`
> dispatcher) or a `#[deprecated]` attribute — never a rename or removal. The old forms keep compiling
> and passing (deprecation warnings only). Two behavior-touching paths (fanova, the 5 `_2d` shims) are
> pinned **bit-identical** (`assert_eq!`) to their current output; the `Default` impls have no behavior
> golden (configs only take effect when explicitly constructed). All gates run under BOTH feature configs
> (`parallel` ON vs OFF), plus the 28-example build and the wasm compile. `since = "0.30.0"` on all
> `#[deprecated]` (crate is v0.29.0 → next is 0.30.0). No new crate dependency.

**Scope (3 PROF-03 items shipped):**
- **Item #1 (API-01):** `impl Default` for `BoostingConfig`, `BayesianConfig`, `StabilityConfig`
  (`boosting_regression/mod.rs`). **StlConfig EXCLUDED** — already `#[derive(Default)]` (E0119 if added).
- **Item #2 (API-01/API-02):** `fanova_seeded(…, seed)` (keeps the hand-rolled LCG) + `fanova` as a
  `#[deprecated]` seed=42 shim, bit-identical golden.
- **Item #4 (API-02):** `Dim` enum (`src/dim.rs`) + 5 unified dispatchers (4 depth + 1 fdata `mean`) +
  `#[deprecated]` on the 5 redundant `_2d` shims only.
- **Item #3 (API-01):** DOC-ONLY canonical-vocabulary note. No field rename.

**Deliberately excluded (documented, not dropped):** StlConfig `Default` (already derived); switching
fanova to `StdRng` (breaking p-value change → APIB-01); deprecating any `_1d` workhorse; unifying
regression (no `_2d` exists), fdata `deriv`/`geometric_median` (divergent sigs), depth
`functional_spatial`/`kernel_functional_spatial` (arity mismatch), `band`/`modified_band`/etc. (no `_2d`);
touching genuine `_nd` algorithms (`pca_nd`, `karcher_*_nd`, `srsf_*_nd`); all field renames (APIB-01).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `#[test]` + integration tests in `fdars-core/tests/` |
| **Quick run (parallel ON)** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --test equivalence_phase50` |
| **Parallel-OFF run** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --no-default-features --features linalg` |
| **Examples gate (API-03)** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo build -p fdars-core --examples --features linalg,parallel` (28 examples; on LINK/disk error `rm -rf target/debug/{incremental,examples}` and retry) |
| **WASM gate (API-03)** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo build -p fdars-core --target wasm32-unknown-unknown --features js --no-default-features` (target confirmed installed) |
| **Full clippy gate** | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` (lints test/bench/example code; deprecation warnings become errors — internal callers must be migrated or `#[allow(deprecated)]`) |

---

## Sampling Rate

- **After every task commit:** targeted `cargo test -p fdars-core <touched_module or --test equivalence_phase50 <filter>>` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` (parallel ON).
- **After every wave:** full suite under BOTH feature configs + `cargo build --examples` + wasm compile.
- **Before verify:** both feature configs green + 28-example build + wasm compile + `cargo fmt -p fdars-core` sweep, with all behavior-touching goldens bit-identical.

---

## Per-Requirement Verification Map

| Req | Behavior | Test Type | Command | Plan | Status |
|-----|----------|-----------|---------|------|--------|
| API-01 (#1) | BoostingConfig/BayesianConfig/StabilityConfig `::default()` == documented values; StlConfig untouched (no E0119) | unit | `cargo test … boosting_regression` (both configs) | 50-01 | ⬜ |
| API-01/02 (#2) | `fanova_seeded(…, 42)` AND the `fanova` shim reproduce the CURRENT fanova output BIT-IDENTICALLY (global_statistic + p_value, LCG preserved not StdRng) | integration golden | `cargo test … --test equivalence_phase50 fanova` (both configs) | 50-02 | ⬜ |
| API-02 (#2) | `fanova_seeded` with a different seed changes p_value but not global_statistic (seed actually threads) | integration golden | `cargo test … --test equivalence_phase50 fanova` | 50-02 | ⬜ |
| API-02 (#4) | 5 unified dispatchers: 3 deterministic (modal/fraiman_muniz/fdata mean) == their `_1d` AND `_2d` outputs bit-identically; 2 RNG (random_projection/random_tukey) forward-by-construction to `_1d` (thread_rng → no runtime equality) verified STRUCTURALLY (len + [0,1]) | integration golden + structural | `cargo test … --test equivalence_phase50 dispatch` (both configs) | 50-03 | ⬜ |
| API-02 (#4) | ONLY the 5 `_2d` shims deprecated; NO `_1d` deprecated; `_nd`/regression/deriv/geometric_median/functional_spatial untouched | static + clippy | grep `#[deprecated]` set == {fanova, modal_2d, fraiman_muniz_2d, random_projection_2d, random_tukey_2d, mean_2d}; `clippy --all-targets -D warnings` | 50-03 | ⬜ |
| API-01 (#3) | Canonical FpcaResult vocabulary documented; NO field rename | static | grep the doc note in regression.rs; grep confirms no field renamed | 50-03 | ⬜ |
| API-03 | 28 examples build with deprecation warnings only (example 21 migrated to fanova_seeded) | smoke | `cargo build -p fdars-core --examples --features linalg,parallel` | 50-01/02/03 | ⬜ |
| API-03 | wasm32-unknown-unknown compiles (`js` feature = getrandom/js only) | smoke | `cargo build -p fdars-core --target wasm32-unknown-unknown --features js --no-default-features` | 50-01/02/03 | ⬜ |
| API-03 | R bindings compile unchanged — by construction (external `fdars-r`, not in-repo; all changes additive, no signature/field change) | construction | documented reasoning (no in-repo compile possible) | all | ⬜ |
| all | Existing suite green — BOTH feature configs; no public signature change; no new dep; `-D warnings` clean | integration | `cargo test --features linalg,parallel` AND `--no-default-features --features linalg` + clippy `--all-targets` | all | ⬜ |

---

## Wave 0 Requirements

- [ ] `fdars-core/tests/equivalence_phase50.rs` — created by plan 50-02 (fanova golden), extended by 50-03 (dispatch-equality goldens). Mirror `equivalence_phase49.rs` header + `#![allow(clippy::excessive_precision)]` + `const` f64 goldens + `assert_eq!` (NOT tolerance).
- [ ] fanova golden captures the CURRENT (pre-change) `fanova` output (global_statistic + p_value) from a DETERMINISTIC fixture (≥2 groups, n≥3) BEFORE any src/ edit; the golden test fn carries `#[allow(deprecated)]`.
- [ ] Dispatch goldens: the 3 deterministic pairs (modal/fraiman_muniz/mean) use `assert_eq!` against a fixed fixture; the 2 RNG pairs (random_projection/random_tukey) use a STRUCTURAL valid-depth-vector check (len + [0,1]) — NOT `assert_eq!` — because their `_1d` calls `_seeded(…, None)` → thread_rng (no public seed → not reproducible).
- [ ] One-line `::default()` smoke test (50-01) asserting the documented config values under both configs.

---

## Deprecation-Hygiene Ledger (API-03 — the `-D warnings` linchpin)

No `#[deprecated]` existed in-crate before this phase. Each in-crate caller of a newly-deprecated fn is
either MIGRATED (user-facing/integration surfaces) or `#[allow(deprecated)]` (behavior-pinning tests):

| Deprecated symbol | In-crate callers | Disposition |
|-------------------|------------------|-------------|
| `fanova` | example 21 (:83); validate_new_modules (:543/:576) | **MIGRATE** → `fanova_seeded(…, 42)` |
| `fanova` | function_on_scalar unit tests (:1031/:1063/:1094/:1104/:1107); inference/anova (:240/:260); inference/permutation (:400); equivalence_phase50 golden | **ALLOW** (`#[allow(deprecated)]` on the specific test fn) |
| `modal_2d` | depth/tests.rs (test_modal_2d_delegates) | **ALLOW** |
| `fraiman_muniz_2d` | depth/tests.rs (test_fraiman_muniz_2d_delegates); validate_against_r.rs:2284 | **ALLOW** |
| `random_projection_2d` | depth/tests.rs (test_random_projection_2d_returns_valid) | **ALLOW** |
| `random_tukey_2d` | (none found — re-exports only; grep to confirm) | **VERIFY** (no allow needed if truly no caller) |
| `mean_2d` | fdata.rs (test_mean_2d_delegates); validate_against_r.rs:2515/:2518 | **ALLOW** |
| — | `depth/dispatch.rs` (calls `_1d` only) | **NO CHANGE** (verify warning-free) |

> `pub use` re-export lines (lib.rs / prelude.rs / depth/mod.rs) do NOT emit deprecation warnings — leave
> them. Each plan re-greps `grep -rn '<symbol>' fdars-core/{src,tests,examples}` before annotating, since
> line numbers may shift.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Instructions |
|----------|-------------|------------|--------------|
| R bindings (`fdars-r`) compile unchanged | API-03 | External CRAN package, not in this repo — cannot be compiled from this tree | Verify by CONSTRUCTION: every change is additive (a trait impl, a new fn, a new `Dim` fn, a `#[deprecated]` attribute); `fanova` keeps its exact signature + return type; the 5 `_2d` shims keep exact signatures; no field renamed → the R FFI surface is source-and-binary compatible. If a local `fdars-r` checkout is available, optionally `cargo build` it against a path-override of `fdars-core`. |

---

## Validation Sign-Off

- [ ] Item #1: 3 boosting `Default` impls compile; `::default()` == documented values under BOTH configs; StlConfig untouched (no E0119)
- [ ] Item #2: `fanova_seeded` keeps the LCG (NOT StdRng); `fanova` deprecated shim delegates with seed=42; `equivalence_phase50` fanova golden bit-identical (global_statistic + p_value) under BOTH configs
- [ ] Item #4: `Dim` enum in src/dim.rs re-exported at crate root; 5 unified dispatchers ship — 3 deterministic (modal/fraiman_muniz/mean) == their `_1d`/`_2d` outputs (assert_eq!, fixed fixture); 2 RNG (random_projection/random_tukey) structural valid-depth-vector check — under BOTH configs
- [ ] Item #4: ONLY the 5 `_2d` shims are `#[deprecated(since="0.30.0")]`; NO `_1d` deprecated; regression/deriv/geometric_median/functional_spatial and all `_nd` algorithms untouched
- [ ] Item #3: canonical-vocabulary doc note in regression.rs; NO field renamed
- [ ] Deprecation hygiene: every in-crate deprecated-fn caller migrated or `#[allow(deprecated)]`; `depth/dispatch.rs` warning-free; `clippy --all-targets --features linalg,parallel -- -D warnings` clean
- [ ] API-03: all 28 examples build (deprecation warnings only); wasm32-unknown-unknown compiles with `--features js`; R compat argued by construction
- [ ] Full suite green under BOTH `--features linalg,parallel` AND `--no-default-features --features linalg`; `cargo fmt` clean; no public signature change; no new dependency
- [ ] `nyquist_compliant: true` set once all above hold

**Approval:** pending
