# PROF-03 — API-Inconsistency Inventory

**Phase:** 46 (Whole-Crate Profiling & Measurement) · **Requirement:** PROF-03 · **Consumer:** Phase 50 (API-01 / API-02 / API-03)
**Method:** static grep + source read against the live tree (2026-08-30). No benches, no `src/` edits.
**Ranking metric (locked):** user-facing impact + breadth. Every proposed form is classified **additive-safe** (add + `#[deprecated]`, in scope for v0.30.0) vs **breaking** (rename/remove — OUT OF SCOPE, deferred to APIB-01) to protect R/WASM bindings + 28 examples.

---

## Inventory Table

| # | Item | Current form(s) | Proposed canonical form | Impact | Breadth | Class | API target |
|---|------|-----------------|-------------------------|--------|---------|-------|-----------|
| 1 | Config structs missing `Default` | `BoostingConfig`, `BayesianConfig`, `StabilityConfig` (`src/boosting_regression/mod.rs:44/76/103`), `StlConfig` (`src/detrend/stl.rs:49`) | `impl Default for …` with documented field defaults | **HIGH** (public API ergonomics; 52/56 configs already have it — these 4 are the outliers) | Low (4 structs) | **additive-safe** | API-01 |
| 2 | `fanova` lacks `seed` (non-reproducible) | `fanova(data, groups, n_perm)` (`src/function_on_scalar.rs:791`) vs siblings all `(…, n_perm, seed)` | `fanova_seeded(data, groups, n_perm, seed)` (add; `#[deprecated]` note on `fanova`) | **HIGH** (reproducibility bug — permutation p-value not seedable) | Low (1 fn, but a correctness/repro gap) | **additive-safe** | API-01 / API-02 |
| 3 | Result-field naming variance (`fitted` vs `fitted_values`) | `FtsmResult.fitted` (`src/fts/mod.rs:191`), `BoostFosrResult.fitted` (`src/boosting_regression/mod.rs:127`) vs `FpcaResult` (`src/regression.rs:25`) canonical `scores/rotation/mean/weights` | Document canonical naming; add doc-aliases where cheap (no field rename — breaking) | Medium | Medium | **mostly breaking** (field rename) → defer; additive doc-alias only | API-01 (doc) |
| 4 | `_1d`/`_2d` paired families without unified dispatch | 13 families with both `_1d` and `_2d` (e.g. `mean_1d`/`mean_2d`, `modal_1d`/`modal_2d`, `random_tukey_1d`/`_2d`, `random_projection_1d`/`_2d`, `lp_self_1d`/`_2d`) | Optional unified `fn mean(data, dim: Dim)`-style dispatch for **high-impact** families only (depth, regression, fdata); keep `_1d`/`_2d` (`#[deprecated]` only if a unified form ships) | Medium | **High** (30+ public fns) | **additive-safe** (add dispatcher) — but scope-limit | API-02 |

---

## Proposed Canonical Forms

### 1. Config `Default` impls (additive-safe, API-01)

```rust
// src/boosting_regression/mod.rs
impl Default for BoostingConfig { /* mstop, nu (learning rate), … documented defaults */ }
impl Default for BayesianConfig { /* prior scale, n_iter, burn_in, … */ }
impl Default for StabilityConfig { /* n_subsamples, subsample_frac, threshold, … */ }
// src/detrend/stl.rs
impl Default for StlConfig { /* period, seasonal/trend spans, robust=false, … */ }
```
Pull the default values from each struct's field doc-comments / the constructor most callers use. Purely additive — no existing signature changes.

### 2. Seedable `fanova` (additive-safe, API-01/API-02)

```rust
// add alongside the existing fn; keep fanova as a #[deprecated] shim delegating with a fixed seed
pub fn fanova_seeded(data: &FdMatrix, groups: &[usize], n_perm: usize, seed: u64) -> Result<FanovaResult, FdarError>
#[deprecated(note = "use fanova_seeded for reproducible permutation p-values")]
pub fn fanova(data: &FdMatrix, groups: &[usize], n_perm: usize) -> Result<FanovaResult, FdarError>
```
Sibling signatures to match (all already `(…, n_perm, seed)`): `t_perm_test` (`src/inference/permutation.rs:152`), `f_perm_test` (`src/inference/permutation.rs:214`), `frechet_anova` (`src/frechet/anova.rs:122`), `generic_permutation_importance` (`src/explain_generic/importance.rs:22`).

### 3. Result-field naming (defer field renames — breaking)

Canonical reference is `FpcaResult` (`scores`, `rotation`, `mean`, `weights`, `singular_values`, `centered`). `FtsmResult`/`BoostFosrResult` reuse `mean`/`rotation`/`scores`/`weights` consistently; the only variance is `fitted` (both) — acceptable. **Renaming a public field is breaking → out of scope (APIB-01).** Phase 50 action: document the canonical vocabulary; the `FpcPredictor` trait (`src/explain_generic/mod.rs`) already provides the unification layer for cross-model access.

### 4. `_1d`/`_2d` dispatch (additive-safe but scope-limited, API-02)

30+ `_1d`/`_2d`/`_nd` public fns exist. **Do NOT bulk-deprecate.** Phase 50 should add a unified dispatcher only for the highest-impact families (depth, regression, fdata) and leave the rest. Genuinely-different `_nd` *algorithms* must never be deprecated (see below).

---

## Additive-Safe vs Breaking

| Item | Classification | Rationale |
|------|---------------|-----------|
| Config `Default` impls (#1) | **additive-safe** | Adds a trait impl; no signature change |
| `fanova_seeded` (#2) | **additive-safe** | Adds a fn; old `fanova` becomes a `#[deprecated]` shim |
| `_1d`/`_2d` unified dispatch (#4) | **additive-safe** (if pursued) | Adds a dispatcher; `_1d`/`_2d` kept and only `#[deprecated]` if a unified form ships |
| Result-field rename (#3) | **BREAKING → defer (APIB-01)** | Renaming a public struct field breaks R/WASM bindings + 28 examples |
| `_nd` alignment/FPCA algorithms | **DO NOT DEPRECATE** | `pca_nd`, `karcher_mean_nd`, `karcher_covariance_nd`, `srsf_transform_nd`, `srsf_inverse_nd` are *different algorithms* (genuine multivariate/shape variants), not dimension conveniences |

---

## Config Default Gap

Confirmed count against the live tree: **56 `pub struct *Config`, 52 `impl Default` → exactly 4 missing** (matches RESEARCH):

| Struct | Anchor | Notes |
|--------|--------|-------|
| `BoostingConfig` | `src/boosting_regression/mod.rs:44` | boosting hyperparams (mstop, nu, …) |
| `BayesianConfig` | `src/boosting_regression/mod.rs:76` | MCMC / prior params |
| `StabilityConfig` | `src/boosting_regression/mod.rs:103` | stability-selection params |
| `StlConfig` | `src/detrend/stl.rs:49` | STL period + span params |

Three of the four live in `boosting_regression` (a v0.29-era module) — a localized ergonomics gap. `StlConfig` is the seasonal-decomposition config.

---

## Phase 50 Hand-off

Top target: **#1 the 4 missing-Default configs** (HIGH impact, additive-safe, small, self-contained) → API-01.
Then **#2 `fanova_seeded`** (HIGH — reproducibility gap) → API-01/API-02. **#4 `_1d`/`_2d` dispatch** is large-breadth — prioritize high-impact families only, do not unify all 30+ at once. **#3 field renames are breaking** → defer to APIB-01; Phase 50 does documentation only. Every in-scope item is additive (add + `#[deprecated]`) so the 28 examples + R/WASM bindings keep compiling with deprecation warnings only (API-03).
