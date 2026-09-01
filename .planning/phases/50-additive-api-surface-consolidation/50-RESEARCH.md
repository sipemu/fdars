# Phase 50: Additive API-Surface Consolidation - Research

**Researched:** 2026-09-01
**Domain:** Rust public-API design — additive deprecation, trait `Default` impls, dispatcher patterns, RNG-reproducible permutation tests
**Confidence:** HIGH (all findings verified by reading the live source tree this session)

## Summary

Phase 50 is an **additive-only** API pass over `fdars-core`: for each of four inventoried inconsistencies, add the new canonical form and `#[deprecated]` the old one — never rename or remove. All findings below are verified against the live tree (2026-09-01). The 28 examples, the `js`-feature WASM surface, and the external `fdars-r` bindings must keep compiling with **deprecation warnings only**.

Three of the four items are mechanically simple and low-risk. The fourth (`_1d`/`_2d` unified dispatch) is where the phase's real design judgment lives, and the honest finding is that **the "unified dispatcher" opportunity is much smaller than the CONTEXT framing implies**: of the three named families, **regression has no `_2d` variants at all** (nothing to unify), **fdata has exactly one cleanly-unifiable pair** (`mean_1d`/`mean_2d`) plus two genuinely-divergent pairs that must be left alone, and **depth has five cleanly-unifiable pairs** whose `_2d` members are already thin shims delegating to `_1d`. The crate already ships a dispatcher precedent (`depth::DepthMethod` in `dispatch.rs`), which the new `Dim` dispatch should mirror stylistically rather than invent a new idiom.

**Primary recommendation:** Land item #1 (config `Default` impls — but only **3** structs, not 4: `StlConfig` already derives `Default`) as the tracer. Then #2 `fanova_seeded` using the **LCG-preserving** approach (a) so the deprecated shim reproduces the current output bit-identically. Then #4, shipping unified `Dim`-dispatch **only** for the 6 cleanly-unifiable pairs (5 depth + 1 fdata `mean`), and explicitly deprecating **only those** `_1d`/`_2d` pairs. Fold #3 (doc-only) into the #4 plan or a small standalone doc commit.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Config `Default` impls | Domain module (boosting_regression, detrend) | — | Trait impl lives beside the struct it targets |
| `fanova_seeded` + shim | Domain module (`function_on_scalar.rs`) | Public API (`lib.rs` re-export) | New fn added beside `fanova`; re-exported at crate root |
| Result-field doc vocabulary | Doc layer (rustdoc) | — | Doc-only; no code change |
| `Dim` dispatch wrapper | Shared infra (`dim.rs` at crate root) + per-family module | Public API | `Dim` enum shared; per-family dispatchers live in their module |
| Golden equivalence tests | Test layer (`tests/equivalence_phase50.rs`) | — | Mirrors Phase 47/48/49 pattern |

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| API-01 | Config/result consistency | Item #1 (3 `Default` impls — verified values below), item #2 (`fanova_seeded` matches sibling `(…, n_perm, seed)` convention), item #3 (doc canonical vocab) |
| API-02 | Redundant-function unification | Item #2 (`fanova`/`fanova_seeded`), item #4 (`Dim` dispatch for 6 verified cleanly-unifiable pairs) |
| API-03 | Back-compat: old forms compile + bindings/examples pass | Deprecation-hygiene call-site list below; examples/WASM/R gate strategy below |

## Standard Stack

No new dependencies. Everything uses in-crate primitives and the existing toolchain.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `std` (`#[deprecated]`) | Rust 1.81+ | Additive deprecation vehicle | Built-in; `note`-carrying attribute; the crate's chosen additive-deprecation idiom [VERIFIED: no prior `#[deprecated]` in-crate, grep 2026-09-01] |
| `rand` / `StdRng` | via existing dep | (NOT introduced for fanova — see reproduction resolution) | fanova uses a hand-rolled LCG, NOT `StdRng` [VERIFIED: src/function_on_scalar.rs:839-848] |
| `criterion` 0.5 | existing | Not needed this phase (no benches) | — |

**Installation:** none. `#[deprecated]` is a `std` attribute; no `Cargo.toml` change.

## Package Legitimacy Audit

Not applicable — this phase installs **zero** new packages (explicit CONTEXT constraint: "No new crate dependency"). No registry verification required.

## Item #1 — Config `Default` impls (API-01, tracer)

### Correction to inventory: only THREE structs need work, not four

`StlConfig` **already derives `Default`** — verified at `src/detrend/stl.rs:47-49`:

```rust
#[derive(Debug, Clone, PartialEq, Default)]   // ← Default is ALREADY here
#[non_exhaustive]
pub struct StlConfig { … }
```

[VERIFIED: src/detrend/stl.rs:47-49] The derive works because every field is `Option<usize>` or `bool`, all of which have a `std` `Default`. **No action needed for `StlConfig`.** The PROF-03 inventory listed it as missing; the live tree contradicts that. The planner should note this and NOT add a hand-written `impl Default for StlConfig` (that would conflict with the derive and fail to compile).

The three that genuinely lack `Default`, all in `src/boosting_regression/mod.rs`:

| Struct | Anchor | Derive line | Has `Default`? |
|--------|--------|-------------|----------------|
| `BoostingConfig` | `src/boosting_regression/mod.rs:44` | `#[derive(Debug, Clone, PartialEq)]` (line 43) | ❌ |
| `BayesianConfig` | `src/boosting_regression/mod.rs:76` | `#[derive(Debug, Clone, PartialEq)]` (line 75) | ❌ |
| `StabilityConfig` | `src/boosting_regression/mod.rs:103` | `#[derive(Debug, Clone, PartialEq)]` (line 102) | ❌ |

[VERIFIED: src/boosting_regression/mod.rs:43-112] None of these three are `#[non_exhaustive]` (unlike `StlConfig` and `BoostFosrResult`), so external struct-literal construction is currently allowed. Adding `impl Default` is still purely additive — a new trait impl never breaks existing construction.

### Default field values — use the DOC-COMMENT stated defaults, not test values

The field doc-comments name canonical defaults; the test-module constructors use *different* ad-hoc values. The planner must use the **doc-comment values** as the authoritative `Default`, because those are the documented API contract. Verbatim from the source:

**`BoostingConfig`** [VERIFIED: src/boosting_regression/mod.rs:44-65]:
- `mstop: usize` — "Number of boosting iterations (must be ≥ 1)." No stated numeric default in doc. Doc-example uses `mstop: 10` (boost_fosr.rs:257). **Recommend `mstop: 100`** (FDboost convention) — but this is `[ASSUMED]`; the doc gives no number. Planner should confirm.
- `nu: f64` — "Learning rate ν ∈ (0, 1] (FDboost default: 0.1)." → **`nu: 0.1`** [VERIFIED, doc states "FDboost default: 0.1"].
- `nbasis: usize` — "must be ≥ 4"; doc-example uses `nbasis: 8` or `10`. **Recommend `nbasis: 10`** (doc text at :52-53 gives the `nbasis = 10` worked example). `[ASSUMED]` numeric.
- `order: usize` — "typically 4 for cubic splines." → **`order: 4`** [VERIFIED, doc "typically 4"].
- `lfd_order: usize` — "typically 2 for roughness." → **`lfd_order: 2`** [VERIFIED, doc "typically 2"].
- `lambda: f64` — "Smoothing parameter λ > 0." Doc-example uses `lambda: 1.0`. **Recommend `lambda: 1.0`.** `[ASSUMED]` numeric.
- `ncomp_x: usize` — "Number of predictor FPC components for FoFR." Doc-example uses `3`. **Recommend `ncomp_x: 3`.** `[ASSUMED]` numeric.
- `seed: u64` — "unused in pure boosting." → **`seed: 0`** (matches `stability.rs:202` `default_boost`).

**`BayesianConfig`** [VERIFIED: src/boosting_regression/mod.rs:76-95] — the doc explicitly cites the reference paper's recommended values:
- `ncomp: usize` — "must be ≥ 1." Test `default_config` uses `4` (bayesian.rs:342). **Recommend `ncomp: 4`.** `[ASSUMED]` numeric.
- `tau2: f64` — "default: 100.0" → **`tau2: 100.0`** [VERIFIED, doc "default: 100.0"].
- `ig_a0: f64` — "default: 0.001 — weakly informative" → **`ig_a0: 0.001`** [VERIFIED].
- `ig_b0: f64` — "default: 0.001 — weakly informative" → **`ig_b0: 0.001`** [VERIFIED].
- `n_iter: usize` — "must be ≥ 1." Test uses `400`. **Recommend `n_iter: 400`.** `[ASSUMED]` numeric.
- `burn_in: usize` — Test uses `200`. **Recommend `burn_in: 200`.** `[ASSUMED]` numeric.
- `thin: usize` — "must be ≥ 1." Test uses `1`. → **`thin: 1`.**
- `seed: u64` — deterministic chain. Test uses `20260824`. **Recommend a fixed seed** (e.g. `0` or `20260824`). `[ASSUMED]`.

**`StabilityConfig`** [VERIFIED: src/boosting_regression/mod.rs:103-112] — doc gives explicit defaults:
- `n_resamples: usize` — "must be ≥ 1; default: 100" → **`n_resamples: 100`** [VERIFIED, doc "default: 100"].
- `pi_thr: f64` — "π ∈ (0.5, 1.0] (default: 0.9)" → **`pi_thr: 0.9`** [VERIFIED, doc "default: 0.9"].
- `seed: u64` — replicate isolation via `wrapping_add`. **Recommend `seed: 0`.** `[ASSUMED]`.

> **Planner action:** the `[VERIFIED]` values above are locked by the doc-comments. For the `[ASSUMED]` numerics (`mstop`, `nbasis`, `lambda`, `ncomp_x`, `ncomp`, `n_iter`, `burn_in`, and seeds) the doc gives a "must be ≥ N" constraint but no single number — the plan should adopt the values from the most-representative existing constructor (the ones cited above) and note them as documentation-completing decisions. There is no behavior-golden to break here (these configs are only *used* when explicitly constructed today), so any valid-range default is back-compat-safe.

### Why this is the tracer

Smallest, most self-contained, zero behavior change, exercises the full pipeline: additive trait impl → `cargo build --examples` gate → `clippy --all-targets -D warnings` → `cargo fmt` → commit. No golden needed (a new `Default` impl cannot change any existing output). Proves the additive+examples-gate machinery before the riskier items.

## Item #2 — `fanova_seeded` (API-01/API-02) — THE fanova-reproduction resolution

### The mechanism `fanova` uses today (verified verbatim)

`fanova` does **NOT** use `StdRng`. It uses a hand-rolled LCG seeded with a hardcoded `42` [VERIFIED: src/function_on_scalar.rs:839-848]:

```rust
let mut rng_state: u64 = 42;
for _ in 0..n_perm {
    for i in (1..n).rev() {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let j = (rng_state >> 33) as usize % (i + 1);
        perm_groups.swap(i, j);
    }
    let perm_stat = integrated_f_statistic(data, &perm_groups, &labels);
    if perm_stat >= observed_stat { n_ge += 1; }
}
let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
```

The multiplier `6_364_136_223_846_793_005` and increment `1` are the Knuth/PCG LCG constants; the bit-extraction is `(rng_state >> 33) as usize % (i + 1)`. There is a design comment at :836-838 explicitly noting this LCG was **deliberately NOT migrated** to the `StdRng`-based `permutation_test` scaffold in Phase 49 precisely because migrating "would change the p-value."

### RESOLUTION: use approach (a) — keep the LCG, add a `seed` param feeding `rng_state`

This is the ONLY approach that lets the deprecated `fanova` shim delegate to `fanova_seeded` and reproduce the current output bit-identically.

- **Approach (a) — CHOSEN.** `fanova_seeded` is a copy of the current `fanova` body with the single change `let mut rng_state: u64 = seed;` (instead of `= 42`). The deprecated `fanova` becomes a one-line shim: `fanova_seeded(data, groups, n_perm, 42)`. Because seed=42 drives the *identical LCG stream*, every `perm_groups` permutation, every `perm_stat`, and thus `n_ge` and `p_value` are bit-identical. **This is provably bit-identical** — same arithmetic, same seed, same loop.
- **Approach (b) — REJECTED.** If `fanova_seeded` switched to `StdRng::seed_from_u64(seed)` (matching how `frechet_anova`/`t_perm_test` seed), then `fanova(…)` delegating with seed=42 would run `StdRng::seed_from_u64(42)` — a *completely different* byte stream from the LCG — producing different permutations and a **different p-value**. The old output would NOT be reproduced. Rejected.

> **Consistency caveat (document, do not fix):** the sibling seeded fns (`frechet_anova`, `t_perm_test`, `f_perm_test`, `generic_permutation_importance`) all use `StdRng`. `fanova_seeded` will be the odd one out, using an LCG. This is a **deliberate, documented divergence** required for bit-identical back-compat. Migrating fanova to `StdRng` is a **breaking behavior change** (different p-values for the same seed) and belongs in APIB-01, not here. The `fanova_seeded` doc-comment must state: "uses the legacy LCG stream for exact back-compat with the deprecated `fanova`; a future `StdRng` migration is deferred (APIB-01)."

### Signatures

```rust
// NEW — src/function_on_scalar.rs, added directly above the existing fanova.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fanova_seeded(
    data: &FdMatrix,
    groups: &[usize],
    n_perm: usize,
    seed: u64,
) -> Result<FanovaResult, FdarError> { /* current fanova body, rng_state = seed */ }

// CHANGED — fanova becomes a #[deprecated] delegating shim.
#[deprecated(
    since = "0.30.0",
    note = "use `fanova_seeded` for reproducible permutation p-values; `fanova` delegates with the legacy fixed seed 42"
)]
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fanova(
    data: &FdMatrix,
    groups: &[usize],
    n_perm: usize,
) -> Result<FanovaResult, FdarError> {
    #[allow(deprecated)]
    fanova_seeded(data, groups, n_perm, 42)   // #[allow] not needed here; call target is not deprecated
}
```

`FanovaResult` type (unchanged — reused by both) [VERIFIED: src/function_on_scalar.rs:72-92]:

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FanovaResult {
    pub group_means: FdMatrix,      // k × m
    pub overall_mean: Vec<f64>,     // length m
    pub f_statistic_t: Vec<f64>,    // length m (pointwise F)
    pub global_statistic: f64,      // integrated F
    pub p_value: f64,               // permutation p-value
    pub n_perm: usize,
    pub n_groups: usize,
    pub group_labels: Vec<usize>,   // sorted unique
}
```

### Golden capture (mirror Phase 49 pattern)

Add `tests/equivalence_phase50.rs`. Capture the CURRENT `fanova` output as `const` goldens from a deterministic fixture, then assert `assert_eq!` (bit-identical, NOT tolerance) against BOTH:
1. `fanova(&data, &groups, n_perm)` (the deprecated shim) — proves the shim path is preserved.
2. `fanova_seeded(&data, &groups, n_perm, 42)` — proves the new path with seed=42 reproduces the old output.

The fixture must be deterministic and cover ≥2 groups, `n ≥ 3`. Capture `global_statistic` (deterministic, seed-independent) AND `p_value` (seed-dependent — the linchpin). Because the test file itself calls the deprecated `fanova`, add `#[allow(deprecated)]` on that test fn (see hygiene section).

## Item #3 — Result-field naming (DOC-ONLY, API-01)

**No code change.** Field renames are breaking (deferred to APIB-01). Phase 50 action: document the canonical vocabulary in rustdoc. [VERIFIED: src/regression.rs — `FpcaResult` canonical fields; src/fts/mod.rs:191 `FtsmResult.fitted`; src/boosting_regression/mod.rs:127-137 `BoostFosrResult.fitted`].

Canonical FPCA vocabulary (from `FpcaResult`, per CONTEXT + MEMORY): `scores`, `rotation`, `mean`, `weights`, `singular_values`, `centered`. The `fitted` field (on `FtsmResult`, `BoostFosrResult`, `FosrResult`, `FosrFpcResult`) is the documented-acceptable variance vs a hypothetical `fitted_values`. The `FpcPredictor` trait (`explain_generic/mod.rs`) already provides the cross-model unification layer, so no field rename is needed for programmatic access.

**Deliverable:** a short doc note (module-level `//!` on `regression.rs` or a `docs/` line, planner's discretion) stating the canonical vocabulary and that `fitted` is the accepted response-field name across result structs. Optionally add `#[doc(alias = "fitted_values")]` on the `fitted` fields (cheap, additive, non-breaking) — this is the "doc-alias where cheap" from PROF-03. `[VERIFIED: field names above by grep]`.

## Item #4 — `_1d`/`_2d` unified dispatch (API-02) — THE CRUX

### Honest per-family assessment

The CONTEXT names three in-scope families (depth, regression, fdata). Reading every signature this session shows the unifiable surface is **much smaller** than "30+ fns":

#### Family: regression — ZERO unifiable pairs

Only `_1d` functions exist; there is **no `_2d` counterpart at all** [VERIFIED: grep src/regression.rs — only `fdata_to_pc_1d` (:307) and `fdata_to_pls_1d` (:634), no `_2d`]. The `_1d` suffix here is vestigial (these functions only ever operated on 1D data). There is nothing to dispatch between. **Recommendation: DO NOTHING for regression.** Deprecating `fdata_to_pc_1d` in favor of a renamed `fdata_to_pc` is an additive-rename that would require shipping a new `fdata_to_pc` fn — possible but not a *dispatcher*, and it adds churn for near-zero user value. Leave regression alone; note it in RESEARCH as "no 1d/2d pair exists."

#### Family: fdata — ONE cleanly-unifiable pair; TWO must be left alone

| Pair | `_1d` signature | `_2d` signature | Unifiable? |
|------|-----------------|-----------------|------------|
| `mean` | `mean_1d(data: &FdMatrix) -> Vec<f64>` [VERIFIED :167] | `mean_2d(data: &FdMatrix) -> Vec<f64>` — body is literally `mean_1d(data)` [VERIFIED :184-187] | **YES — clean.** Identical signature; `_2d` is a pure delegating shim. |
| `deriv` | `deriv_1d(data, argvals: &[f64], nderiv: usize) -> FdMatrix` [VERIFIED :852] | `deriv_2d(data, argvals_s, argvals_t, m1, m2) -> Option<Deriv2DResult>` [VERIFIED :934-940] | **NO.** Different arity (3 vs 5 args), different return type (`FdMatrix` vs `Option<Deriv2DResult>`). Genuinely different computations. LEAVE ALONE. |
| `geometric_median` | `geometric_median_1d(data, argvals, max_iter, tol) -> Vec<f64>` [VERIFIED :1000-1005] | `geometric_median_2d(data, argvals_s, argvals_t, max_iter, tol) -> Vec<f64>` [VERIFIED :1025-1031] | **NO.** Different arity (4 vs 5 args — `_2d` splits `argvals` into `argvals_s`/`argvals_t`). Same return type but the 2D grid needs two axis vectors. Not cleanly unifiable under a single `Dim`. LEAVE ALONE. |

Also present in fdata but single-dimension-only (no pair): `center_1d`, `norm_lp_1d` — no `_2d`, nothing to unify. [VERIFIED: grep — no `center_2d`/`norm_lp_2d` in fdata.rs].

**Recommendation for fdata: ship a unified `mean` dispatcher ONLY.** Because `mean_1d`/`mean_2d` have *identical* signatures and `mean_2d` already just calls `mean_1d`, a `Dim`-dispatch is trivial and honest.

#### Family: depth — FIVE cleanly-unifiable pairs

Every depth `_2d` is a **thin shim that delegates to `_1d`** with an identical signature — these are pure redundancy, exactly what API-02 targets:

| Pair | Signature (identical for _1d/_2d) | `_2d` body | Unifiable? |
|------|-----------------------------------|-----------|------------|
| `modal` | `(data_obj, data_ori, h: f64) -> Vec<f64>` [VERIFIED modal.rs:17,44] | `modal_1d(data_obj, data_ori, h)` [VERIFIED modal.rs:45] | **YES** |
| `fraiman_muniz` | `(data_obj, data_ori, scale: bool) -> Vec<f64>` [VERIFIED fraiman_muniz.rs:32,43] | `fraiman_muniz_1d(...)` [VERIFIED :45] | **YES** |
| `random_projection` | `(data_obj, data_ori, nproj: usize) -> Vec<f64>` [VERIFIED random_projection.rs:32,57] | `random_projection_1d(...)` [VERIFIED :58] | **YES** |
| `random_tukey` | `(data_obj, data_ori, nproj: usize) -> Vec<f64>` [VERIFIED random_tukey.rs:11,36] | `random_tukey_1d(...)` [VERIFIED :36] | **YES** |
| `functional_spatial` | `_1d(data_obj, data_ori, argvals: Option<&[f64]>)`; `_2d(data_obj, data_ori)` [VERIFIED spatial.rs:18-22,77] | `_2d` differs: NO `argvals` param | **PARTIAL — see note** |

Note on `functional_spatial`: `_1d` takes `argvals: Option<&[f64]>`, `_2d` takes only `(data_obj, data_ori)`. Signatures **differ in arity**. A `Dim`-dispatch `functional_spatial(data_obj, data_ori, argvals, dim)` could pass `argvals` and ignore it for `Dim::Two`, but that's awkward (a param that's meaningless in one branch). **Recommend: LEAVE `functional_spatial` alone** — its arity mismatch makes a clean unified form awkward, matching the CONTEXT guidance "left alone because their signatures can't cleanly unify." Same reasoning excludes `kernel_functional_spatial_1d/2d` (spatial.rs:199,219 — `_1d` may take extra params).

Depth single-dimension-only (no pair, nothing to unify): `band_1d`, `modified_band_1d`, `extremal_depth_1d`, `hypograph_index_1d`, `epigraph_index_1d`, etc. [VERIFIED: depth/mod.rs:30-48 — these have no `_2d` export].

### FINAL dispatcher-design recommendation

**Ship a shared `Dim` enum + per-function unified dispatchers for exactly 6 pairs** (5 depth + 1 fdata `mean`). Deprecate ONLY those 6 pairs' `_1d`/`_2d` members. Everything else in the three families is left untouched.

**`Dim` enum placement:** create `src/dim.rs` (tiny) and re-export `Dim` at the crate root, so depth and fdata share one type:

```rust
// src/dim.rs
/// Dimensionality selector for the unified depth/fdata dispatchers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Dim {
    /// 1D functional data (curves).
    One,
    /// 2D functional data (surfaces, flattened column-major).
    Two,
}
```

**Dispatcher shape** (per family, mirroring the existing `DepthMethod`/`functional_depth` precedent in `depth/dispatch.rs` — the crate's established dispatcher idiom [VERIFIED: depth/dispatch.rs:1-31]):

```rust
// depth — one unified fn per unifiable pair. Example for modal:
#[must_use = "expensive computation whose result should not be discarded"]
pub fn modal(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64, dim: Dim) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => modal_1d(data_obj, data_ori, h), // 2d already == 1d
    }
}
```

Because every in-scope `_2d` currently just calls `_1d`, the unified fn's two match arms are identical today — but the `Dim` parameter makes the *intent* explicit and gives a single future seam if the 2D path ever diverges. Deprecate both `modal_1d` and `modal_2d`... **NO — do not deprecate `_1d`.** See the important refinement below.

> **IMPORTANT refinement on WHICH members to deprecate.** Deprecating `modal_1d` is user-hostile: `modal_1d` is the *real workhorse*, called directly by `depth/dispatch.rs` and by user code, and the unified `modal` just forwards to it. Recommend deprecating **only the redundant `_2d` shim** (`modal_2d`, `fraiman_muniz_2d`, `random_projection_2d`, `random_tukey_2d`, `mean_2d`) and adding the unified `modal(…, dim)` form. Rationale: the `_2d` shims are the genuine redundancy (they add nothing over `_1d`); `_1d` is the canonical implementation. The unified `Dim` form is offered as the *new ergonomic entry point*, `_2d` is deprecated as redundant, and `_1d` stays un-deprecated as the primitive. This keeps `depth/dispatch.rs`'s internal `modal_1d`/`fraiman_muniz_1d`/`random_projection_1d_seeded` calls warning-free with **zero** `#[allow(deprecated)]` churn.

**Net item-#4 deliverable:**
- Add `src/dim.rs` with `Dim`, re-export at crate root.
- Add 6 unified dispatchers: `depth::modal`, `depth::fraiman_muniz`, `depth::random_projection`, `depth::random_tukey`, `fdata::mean` (each taking `dim: Dim`).
- `#[deprecated]` the 5 `_2d` shims + `mean_2d` (note: "redundant with the unified `<name>` — use `<name>(…, Dim::Two)`").
- Do **NOT** deprecate any `_1d`. Do **NOT** touch regression, `deriv`, `geometric_median`, `functional_spatial`, `band`, or genuine `_nd` algorithms.

This is a smaller, safer, more honest scope than "unify 30+ fns" — and it's defensible: it deprecates only genuine redundancy and never a workhorse primitive.

## Runtime State Inventory

Not a rename/refactor/migration phase in the state-carrying sense — this is additive API surface. No stored data, live-service config, OS-registered state, secrets, or build artifacts embed any renamed string (nothing is renamed). **None — verified: all changes are additive symbol additions + `#[deprecated]` attributes; no existing symbol name, file path, or datastore key changes.**

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Deprecation warnings | A custom `#[cfg]`-gated "legacy" module or runtime warning | `#[deprecated(since, note)]` | Standard, compiler-enforced, IDE-surfaced, zero runtime cost [VERIFIED: std attribute] |
| fanova reproducibility | A new `StdRng` path + a "compatibility shim" that tries to mimic the LCG | Keep the exact LCG, parameterize its seed | Any non-LCG stream produces different p-values; only the LCG reproduces bit-identically [VERIFIED: src:839-848] |
| Dimension dispatch | A trait with associated types, or macro-generated variants | A plain `Dim` enum + `match` (mirror `DepthMethod`) | The crate already has this exact idiom in `depth/dispatch.rs`; consistency > cleverness [VERIFIED: dispatch.rs] |
| Golden equivalence | Tolerance-based `approx` asserts | `assert_eq!` on captured `const` f64 | Additive/behavior-preserving changes must be bit-identical (Phase 47/48/49 precedent) [VERIFIED: equivalence_phase49.rs] |

**Key insight:** the highest-risk trap in this phase is switching fanova to `StdRng` "for consistency with siblings" — that silently breaks p-value reproducibility for the deprecated path. The LCG must be preserved verbatim.

## Common Pitfalls

### Pitfall 1: Deprecating a fn the crate's own code calls → `-D warnings` fails
**What goes wrong:** `#[deprecated]` emits a warning at every call site, including in-crate callers and tests. Under `clippy --all-targets -- -D warnings` (the CI gate per MEMORY `ci-clippy-all-targets-gate`), those warnings become errors → false-red build.
**Why it happens:** `-D warnings` promotes `deprecated` lint to error; `--all-targets` includes test/bench/example code.
**How to avoid:** For each deprecated symbol, silence every internal call site with `#[allow(deprecated)]` (on the fn/mod/test) OR migrate the caller to the new form. See the exhaustive call-site list below.
**Warning signs:** `warning: use of deprecated function` in `cargo clippy --all-targets` output.

### Pitfall 2: Adding `impl Default` where a `derive(Default)` already exists → conflicting impls
**What goes wrong:** Writing `impl Default for StlConfig` fails to compile because `StlConfig` already `#[derive(..., Default)]`.
**Why it happens:** PROF-03 inventory listed StlConfig as missing Default; the live tree shows it already has the derive [VERIFIED: stl.rs:47].
**How to avoid:** Only add `impl Default` for the 3 boosting configs. Skip StlConfig entirely.
**Warning signs:** `error[E0119]: conflicting implementations of trait Default`.

### Pitfall 3: `fanova_seeded` using `StdRng` → deprecated shim produces different p-values
**What goes wrong:** Golden equivalence test fails: `fanova(…)` (via shim delegating to `StdRng`-based `fanova_seeded(…, 42)`) yields a p-value that differs from the captured pre-change golden.
**Why it happens:** `StdRng::seed_from_u64(42)` ≠ the LCG stream seeded at 42.
**How to avoid:** `fanova_seeded` must keep the exact LCG (`rng_state = seed`); only the seed source changes.
**Warning signs:** `assert_eq!` p_value mismatch in `equivalence_phase50.rs`.

### Pitfall 4: `target/` fills /home during the examples gate
**What goes wrong:** `cargo build --examples` links 28 binaries; `target/` balloons and `/home` fills → "linking with cc failed" (a disk error, not a code bug).
**Why it happens:** MEMORY `target-dir-fills-home-partition` + `tmp-exhaustion-blocks-precommit`.
**How to avoid:** Prefix cargo with `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`; `rm -rf target/debug/{incremental,examples}` if a link fails; use `git commit --no-verify` (doctest link needs /tmp) then a `cargo fmt` sweep (MEMORY `noverify-commits-leave-fmt-drift`).
**Warning signs:** "No space left on device" or "linking with cc failed" during the examples build.

## Deprecation-Hygiene Call-Site List (API-03)

No `#[deprecated]` exists anywhere in the crate today [VERIFIED: grep src/ tests/ examples/, 2026-09-01 — zero hits]. This milestone introduces the pattern. Below is every in-repo call site of each to-be-deprecated symbol.

### `fanova` (deprecated shim) — call sites that will warn under `-D warnings`

| File | Line(s) | Kind | Fix |
|------|---------|------|-----|
| `src/function_on_scalar.rs` | 1031, 1063, 1094, 1104, 1107 | unit tests | `#[allow(deprecated)]` on the test fns (they intentionally test the old path) OR migrate to `fanova_seeded(…, 42)` |
| `src/inference/anova.rs` | 240, 260 | unit tests (cross-check vs vstat) | `#[allow(deprecated)]` on the two test fns |
| `src/inference/permutation.rs` | 400 | unit test | `#[allow(deprecated)]` on the test fn |
| `tests/validate_new_modules.rs` | 543, 576 (import at 414) | integration tests | `#[allow(deprecated)]` on the test fns or migrate to `fanova_seeded` |
| `examples/21_function_on_scalar/main.rs` | 83 (import at 8) | example | **Migrate to `fanova_seeded(&data, &groups, 500, 42)`** — examples should demonstrate the *new* canonical API, not the deprecated one. (An `#[allow(deprecated)]` in an example teaches the wrong thing.) |
| `tests/equivalence_phase50.rs` | (new) | golden test — deliberately calls `fanova` | `#[allow(deprecated)]` on the golden test fn (must exercise the deprecated path) |

[VERIFIED: grep fanova call sites, 2026-09-01]. **Recommended fix strategy:** migrate the *example* (user-facing, teaches the new API) and the *integration tests* to `fanova_seeded`; `#[allow(deprecated)]` the *unit tests* that specifically pin old behavior and the equivalence golden.

### `_2d` depth shims + `mean_2d` (the deprecated members of item #4)

Because the recommendation deprecates only the `_2d` shims (NOT `_1d`), and `depth/dispatch.rs` calls only the `_1d` forms [VERIFIED: dispatch.rs:13-18 imports `fraiman_muniz_1d`, `random_projection_1d_seeded`, etc. — no `_2d`], internal warning exposure is minimal. Call sites to check for each `_2d`:

| Symbol | Search command | Expected callers | Fix |
|--------|----------------|------------------|-----|
| `modal_2d` | `grep -rn 'modal_2d' src tests examples` | tests only (if any) | `#[allow(deprecated)]` on those tests |
| `fraiman_muniz_2d` | same | tests only | `#[allow(deprecated)]` |
| `random_projection_2d` | same | tests only | `#[allow(deprecated)]` |
| `random_tukey_2d` | same | tests only | `#[allow(deprecated)]` |
| `mean_2d` | `grep -rn 'mean_2d' src tests examples` | tests + possibly a 2D example | `#[allow(deprecated)]` on tests; migrate any example to `mean(…, Dim::Two)` |

> **Planner action:** before writing the plan, run `grep -rn '<symbol>' fdars-core/src fdars-core/tests fdars-core/examples` for each of the 5 deprecated `_2d` symbols to produce the exact per-symbol call-site list. This RESEARCH verified the *depth internal dispatcher* does not use them; the plan's Wave-0 grep confirms test/example exposure. (These were not exhaustively grepped this session because the deprecation set for item #4 depends on the plan confirming the "deprecate `_2d` only" recommendation.)

## Config Default Values (consolidated — item #1)

See the per-struct verbatim tables under **Item #1** above. Summary of the locked (doc-VERIFIED) values:

| Struct | Field | Value | Provenance |
|--------|-------|-------|------------|
| BoostingConfig | nu / order / lfd_order / seed | 0.1 / 4 / 2 / 0 | doc-comment [VERIFIED] |
| BoostingConfig | mstop / nbasis / lambda / ncomp_x | 100? / 10 / 1.0 / 3 | doc-example / `[ASSUMED]` numeric |
| BayesianConfig | tau2 / ig_a0 / ig_b0 / thin | 100.0 / 0.001 / 0.001 / 1 | doc-comment [VERIFIED] |
| BayesianConfig | ncomp / n_iter / burn_in / seed | 4 / 400 / 200 / 0 | representative constructor / `[ASSUMED]` |
| StabilityConfig | n_resamples / pi_thr | 100 / 0.9 | doc-comment [VERIFIED] |
| StabilityConfig | seed | 0 | `[ASSUMED]` |
| ~~StlConfig~~ | — | **already derives Default — no action** | [VERIFIED: stl.rs:47] |

## Examples / Bindings Gate Strategy (API-03)

### 28 examples

`cargo build --examples` is the explicit API-03 gate (28 `[[example]]` entries in Cargo.toml per CONTEXT). Only **example 21** (`examples/21_function_on_scalar/main.rs:83`) calls a to-be-deprecated fn (`fanova`) [VERIFIED: grep]. **Migrate it to `fanova_seeded(&data, &groups, 500, 42)`** so the examples build stays warning-free and teaches the new API. All other examples are unaffected by items #1–#4 (no example constructs `BoostingConfig`/`BayesianConfig`/`StabilityConfig` via `default()`, and the `_2d` depth shims / `mean_2d` need a per-symbol grep — see hygiene section).

**Gate command (disk-safe, per MEMORY):**
```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo build --examples -p fdars-core --features linalg,parallel
# if "linking with cc failed": rm -rf target/debug/{incremental,examples} and retry
```

### WASM (`js` feature)

**There is NO in-crate `wasm_bindgen` surface** [VERIFIED: grep — zero `wasm_bindgen` in fdars-core/src/]. The `js` feature is solely `js = ["getrandom/js"]` [VERIFIED: Cargo.toml:29] — it only wires `getrandom`'s browser entropy backend. So "WASM surface compiles" = **the crate compiles for `wasm32-unknown-unknown` with `--features js`**. None of items #1–#4 touch RNG-entropy or platform code, so this gate is a formality but should still run:

```bash
rustup target add wasm32-unknown-unknown   # if not present
cargo build -p fdars-core --target wasm32-unknown-unknown --features js --no-default-features
```

Additive symbol additions + `#[deprecated]` attributes cannot break a WASM compile (deprecation is warn-level, and the crate does not `-D warnings` on the wasm build unless the plan adds it). **Recommend the plan runs the wasm build as a non-`-D-warnings` compile check** (deprecation warnings acceptable there).

### R bindings (`fdars-r`, external)

`fdars-r` is an **external CRAN package** [VERIFIED: CLAUDE.md] not in this repo — it cannot be built from this tree. API-03 verification for R is by **construction, not compilation-in-repo**: because every change is additive (`impl Default` adds a trait, `fanova_seeded` adds a fn, `fanova` keeps its exact signature + return type, the `_2d` shims keep their exact signatures and only gain a `#[deprecated]` attribute, and no field is renamed), the external R FFI surface is **binary-and-source compatible**. The plan should document this reasoning explicitly and, if the R crate is locally available, note the manual verification step `cargo build` in the `fdars-r` checkout against a path-override of `fdars-core`. Otherwise: **document that no public signature or field changed → R bindings compile unchanged (deprecation warnings only).**

## Code Examples

### Item #2 — the fanova_seeded body change (the only line that differs)
```rust
// Source: src/function_on_scalar.rs:839 (current) — fanova_seeded changes ONLY this line.
// CURRENT (fanova):
let mut rng_state: u64 = 42;
// NEW (fanova_seeded): seed the SAME LCG from the param.
let mut rng_state: u64 = seed;
// everything below (the LCG update + Fisher-Yates + stat compare) is byte-for-byte identical.
```

### Item #4 — a unified depth dispatcher (mirror of depth/dispatch.rs idiom)
```rust
// Source: pattern from src/depth/dispatch.rs:1-31 (functional_depth / DepthMethod).
use crate::dim::Dim;
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fraiman_muniz(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool, dim: Dim) -> Vec<f64> {
    match dim {
        // Both branches call _1d today; the 2d path never diverged (fraiman_muniz_2d just forwards).
        Dim::One | Dim::Two => fraiman_muniz_1d(data_obj, data_ori, scale),
    }
}
#[deprecated(since = "0.30.0", note = "redundant with `fraiman_muniz(…, Dim::Two)`; body just forwards to `fraiman_muniz_1d`")]
pub fn fraiman_muniz_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64> {
    fraiman_muniz_1d(data_obj, data_ori, scale)   // call target NOT deprecated → no #[allow] needed
}
```

### Item #1 — a Default impl (the boosting configs are NOT non_exhaustive, so a struct literal works)
```rust
// Source: src/boosting_regression/mod.rs:44-65 (field docs give the values).
impl Default for BoostingConfig {
    fn default() -> Self {
        Self { mstop: 100, nu: 0.1, nbasis: 10, order: 4, lfd_order: 2, lambda: 1.0, ncomp_x: 3, seed: 0 }
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Public fn with no deprecation path | `#[deprecated(since, note)]` additive shim | This milestone (0.30.0) | First use of deprecation in-crate |
| Redundant `_2d` shims duplicating `_1d` | `Dim`-enum dispatch + deprecate the `_2d` shim | This phase | Fewer redundant public symbols |
| Hardcoded-42 LCG in `fanova` | Seed-parameterized LCG (`fanova_seeded`) | This phase | Reproducible p-values; old path preserved via fixed seed |

**Deprecated/outdated in this phase:** `fanova` (→ `fanova_seeded`), `modal_2d`/`fraiman_muniz_2d`/`random_projection_2d`/`random_tukey_2d`/`mean_2d` (→ unified `Dim` forms). All remain callable (warnings only).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `BoostingConfig::default().mstop = 100` | Item #1 | Low — any value ≥1 is valid; doc gives no number. Planner confirms. |
| A2 | `nbasis=10`, `lambda=1.0`, `ncomp_x=3` for BoostingConfig | Item #1 | Low — doc gives ranges, these are the doc-example values |
| A3 | `BayesianConfig` `ncomp=4`, `n_iter=400`, `burn_in=200` | Item #1 | Low — from representative constructor; doc gives only ranges |
| A4 | Default seeds = 0 for the three configs | Item #1 | Low — seed is documented as arbitrary/isolation-only |
| A5 | Deprecate `_2d` only (not `_1d`) is the right split for item #4 | Item #4 | Medium — affects which symbols get `#[deprecated]`; planner should confirm the "keep `_1d` as primitive" call. Alternative (deprecate both, keep only unified) is more aggressive and churns `depth/dispatch.rs`. |
| A6 | R bindings compile unchanged by construction (external, not built in-repo) | Bindings gate | Medium — can't compile-verify in this tree; reasoning is sound (all additive) but unverified against actual `fdars-r` FFI |
| A7 | The `since = "0.30.0"` version string | Items #2/#4 | Low — planner should set to the actual next-release version |

## Open Questions

1. **Exact `#[deprecated(since = "…")]` version string.**
   - What we know: this is v0.14.0 crate per CLAUDE.md header, but MEMORY/PROF-03 reference v0.29/v0.30 module eras and "in scope for v0.30.0."
   - What's unclear: the actual next published version.
   - Recommendation: planner reads `fdars-core/Cargo.toml` `version` and uses `<current>+1` or the milestone target; or omit `since` (it's optional).

2. **Deprecate `_2d`-only vs deprecate-both for item #4 (A5).**
   - What we know: `_2d` shims are the genuine redundancy; `_1d` are the workhorses used by `depth/dispatch.rs`.
   - What's unclear: operator preference on aggressiveness.
   - Recommendation: deprecate `_2d`-only (keeps `_1d` warning-free internally, zero `#[allow(deprecated)]` churn in `dispatch.rs`). Documented as the recommendation.

3. **Should regression get an additive `fdata_to_pc` rename?**
   - What we know: regression has no `_1d`/`_2d` pair — the `_1d` suffix is vestigial.
   - What's unclear: whether "consistency" warrants adding `fdata_to_pc` + deprecating `fdata_to_pc_1d`.
   - Recommendation: NO — out of the item-#4 "dispatch" intent, adds churn for ~zero value, and it's a rename-flavored change closer to APIB-01. Leave regression alone.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | all | ✓ (assumed) | 1.97.0 dev / 1.81 MSRV | — |
| `wasm32-unknown-unknown` target | WASM gate | ? | — | `rustup target add wasm32-unknown-unknown`; if unavailable, document R/WASM as construction-verified only |
| `fdars-r` checkout | R gate | ✗ (external CRAN pkg, not in repo) | — | Verify by construction (all-additive reasoning) |

**Missing dependencies with fallback:** WASM target (install via rustup); R crate (construction-verify). Neither blocks — both have documented fallbacks.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]` / `#[cfg(test)]`) + integration tests in `tests/` |
| Config file | none (standard cargo) |
| Quick run command | `cargo test -p fdars-core --features linalg,parallel <testname>` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` AND `cargo test -p fdars-core --no-default-features --features linalg` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-01 (#1) | 3 configs get `Default` | unit | `cargo test -p fdars-core boosting_regression` | ✅ (add a small assert-`::default()` test) |
| API-01/02 (#2) | `fanova_seeded` reproduces `fanova` bit-identically | golden | `cargo test -p fdars-core --test equivalence_phase50` | ❌ Wave 0 — new file |
| API-02 (#4) | unified `Dim` dispatch == `_1d`/`_2d` outputs | unit/golden | `cargo test -p fdars-core depth::` / `fdata::` | ❌ Wave 0 — add equality asserts |
| API-03 | examples + WASM + R compile (warnings only) | smoke | `cargo build --examples`; `cargo build --target wasm32-… --features js` | ✅ existing examples; gate scripted |

### Sampling Rate
- **Per task commit:** targeted `cargo test -p fdars-core <touched_module>` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- **Per wave merge:** full suite under BOTH feature configs + `cargo build --examples`.
- **Phase gate:** full suite green (both configs) + examples build + WASM compile + `cargo fmt` sweep, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `tests/equivalence_phase50.rs` — captures CURRENT `fanova` golden (global_statistic + p_value from a deterministic fixture), asserts bit-identical via both the deprecated shim and `fanova_seeded(…, 42)`. Mirror `equivalence_phase49.rs` structure.
- [ ] Depth/fdata dispatch-equality asserts — `assert_eq!(modal(a, b, h, Dim::One), modal_1d(a, b, h))` etc. for the 6 unified pairs (with `#[allow(deprecated)]` where a `_2d` is referenced).
- [ ] A one-line `BoostingConfig::default()`/`BayesianConfig::default()`/`StabilityConfig::default()` smoke test asserting the documented values.
- [ ] Scripted API-03 gate: `cargo build --examples` + wasm compile, with the disk-safety `TMPDIR` prefix.

## Security Domain

`security_enforcement` is not indicated for this crate (pure numerical library, no auth/session/network/untrusted-input surface). This phase adds no I/O, no deserialization path, no new parsing. The only "security-adjacent" element is RNG: `fanova_seeded` uses a **non-cryptographic LCG** — correct and intended (permutation-test reproducibility, NOT security). No ASVS category applies. **Section omitted as not applicable** beyond this note.

## Sources

### Primary (HIGH confidence — source read this session)
- `src/function_on_scalar.rs:72-92` (FanovaResult), `:791-868` (fanova + LCG) — verbatim mechanism
- `src/boosting_regression/mod.rs:43-112` (3 configs + field docs), `bayesian.rs:340-351`, `stability.rs:193-211`, `boost_fosr.rs:256-259` (constructor values)
- `src/detrend/stl.rs:47-49` (StlConfig ALREADY derives Default — inventory correction)
- `src/depth/{modal,fraiman_muniz,random_projection,random_tukey,spatial}.rs` (1d/2d signatures + shim bodies), `depth/dispatch.rs:1-60` (DepthMethod precedent), `depth/mod.rs:30-48` (exports)
- `src/fdata.rs:167-187,852-875,934-989,1000-1040` (mean/deriv/geometric_median signatures — divergence proof)
- `src/regression.rs:307,634` (only `_1d`, no `_2d`)
- `src/metric/lp.rs` (lp_self in metric module — out of scope confirmation)
- `Cargo.toml:29,44,50` (`js` feature = getrandom/js only; no wasm_bindgen)
- `tests/equivalence_phase49.rs` (golden pattern to mirror)
- grep: zero `#[deprecated]` in-crate; all fanova/`_2d` call sites

### Secondary (MEDIUM)
- PROF-03 inventory (2026-08-30) — corrected re StlConfig
- CLAUDE.md, MEMORY.md (disk/CI/commit operational constraints)

### Tertiary (LOW)
- FDboost/refund/paper default conventions (for the `[ASSUMED]` numeric config defaults)

## Metadata

**Confidence breakdown:**
- Config Default (item #1): HIGH — verified StlConfig already-derived (inventory correction), 3 target structs + doc-comment values read verbatim. Numeric defaults for a few fields are `[ASSUMED]` (doc gives ranges only).
- fanova reproduction (item #2): HIGH — LCG mechanism read verbatim; approach (a) is provably bit-identical.
- `_1d`/`_2d` dispatch (item #4): HIGH — every in-scope signature read; unifiable set (6 pairs) vs divergent set (deriv/geometric_median/functional_spatial) established from source.
- Bindings gate: MEDIUM — WASM = getrandom-only (verified); R external (construction-verified, not compile-verified in-repo).

**Research date:** 2026-09-01
**Valid until:** ~2026-10-01 (stable library; source anchors may shift line numbers on unrelated edits — re-grep anchors before planning if the tree changed).
