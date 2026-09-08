# Phase 86: Surface Sealing — Research

**Researched:** 2026-09-08
**Domain:** Rust API visibility, `#[non_exhaustive]`, `pub(crate)` module sealing
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- `pub mod wire` → `pub(crate) mod wire`. Wire is sealed, NOT wired up as a supported public interchange API. No current JS/R consumer; re-expose deliberately later.
- Config escape hatches use `Default` / hand-written builders, NOT a derive-builder crate (no new dependency).
- SEAL-02 + SEAL-03 are inseparable — a config struct cannot be sealed `#[non_exhaustive]` without a non-literal construction path or external construction breaks; they land together.
- API-shape-only: no numeric or behavioral change. `fdars-core` only, no new crate dependency.

### Claude's Discretion
All implementation choices are at Claude's discretion — this is a pure infrastructure/refactor phase. Guided by the v0.41.0 sealing/dispatch pattern already established in the crate and existing conventions.

### Deferred Ideas (OUT OF SCOPE)
- Wiring `wire` up as a supported public interchange API — deliberately deferred; re-expose only when JS/R bindings need it.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SEAL-01 | Seal `pub mod wire` → `pub(crate) mod wire`; remove all crate-root/prelude re-exports of wire types; crate + all 28 examples + all doctests still compile | Complete enumeration below: zero examples use wire, zero lib.rs/prelude.rs re-exports, one internal doctest requires path fix |
| SEAL-02 | Mark every public config struct without `#[non_exhaustive]` with `#[non_exhaustive]` | 38 unsealed `pub struct *Config` structs enumerated precisely — see table below |
| SEAL-03 | Every struct sealed under SEAL-02 gets a non-literal construction escape hatch (`Default` + documented `..Default::default()` path) | All 38 unsealed configs already have `impl Default` or `derive(Default)` — no new impls needed |
</phase_requirements>

---

## Summary

Phase 86 is a **codebase enumeration and attribute-insertion** task, not a design task. All decisions are resolved by the existing code:

1. **SEAL-01 (wire sealing):** `wire` is declared `pub mod wire` at `src/lib.rs:125` with no `pub use wire::*` re-exports anywhere in `lib.rs` or `prelude.rs`. No examples and no external-crate code reference wire types. The only breakage from `pub(crate)` is the module-level doctest at `src/wire.rs:17-36` which uses `use fdars_core::wire::*;` — this path becomes invalid once the module is crate-private, so the doctest must be updated to `use crate::wire::*;` (or annotated `no_run`/removed).

2. **SEAL-02 (non_exhaustive on configs):** There are exactly **66 `pub struct *Config`** structs in the codebase. **28 are already sealed** with `#[non_exhaustive]`. **38 need the attribute added** — enumerated precisely below. The CONTEXT.md's "15 already sealed" figure is stale; the actual count from reading the tree is 28.

3. **SEAL-03 (construction escape hatch):** All 38 unsealed configs already have a `Default` implementation (`derive(Default)` or hand-written `impl Default for ...`). Zero structs need a new `Default` impl. The planner needs only to add the attribute, update documentation to show the `..Default::default()` pattern, and verify the Default path works for each struct.

**Primary recommendation:** Three atomic tasks — (A) seal wire + fix internal doctest, (B) add `#[non_exhaustive]` to 38 config structs, (C) verify compilation gates pass. No new code logic required.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `pub mod wire` → `pub(crate)` | `src/lib.rs` (declaration) | `src/wire.rs` (internal doctest fix) | Module visibility is declared in `lib.rs`; the internal doctest must also be fixed |
| Remove wire re-exports | `src/lib.rs`, `src/prelude.rs` | — | No wire re-exports exist in either file — zero changes needed here |
| `#[non_exhaustive]` on config structs | Individual domain module files (38 sites) | — | Attributes go directly above the struct definition in each file |
| Escape-hatch verification | Build gate | — | `cargo test --features linalg` confirms existing Default impls suffice |
| Doctest compile-gate | `cargo test --doc` | All 28 examples | Catch any wire reference surviving the seal |

---

## SEAL-01: Wire Module Enumeration

### `pub mod wire` Declaration Site

`src/lib.rs:125` — verbatim: `pub mod wire;` [VERIFIED: src/lib.rs:125]

**Change required:** Replace `pub mod wire;` with `pub(crate) mod wire;`

### Public Types Defined in `src/wire.rs`

All types are listed with their line numbers. [VERIFIED: src/wire.rs]

**Structs (15):**

| Line | Type | Notes |
|------|------|-------|
| 49 | `FdaData` | Main container; has constructors `from_curves`, `from_tabular`, `empty` |
| 74 | `NamedVec` | Named scalar vector |
| 82 | `GroupVar` | Named grouping variable |
| 157 | `FpcaLayer` | FPCA decomposition layer |
| 174 | `PlsLayer` | PLS decomposition layer |
| 187 | `AlignmentLayer` | Elastic alignment result; has `to_karcher_mean_result()` method |
| 224 | `DistancesLayer` | Precomputed distance matrix |
| 234 | `DepthLayer` | Depth scores layer |
| 244 | `OutlierLayer` | Outlier detection result |
| 270 | `ClusterLayer` | Cluster assignments |
| 288 | `RegressionLayer` | Regression fit layer |
| 328 | `FosrLayer` | Function-on-scalar fit layer |
| 340 | `ToleranceLayer` | Tolerance band |
| 354 | `MeanLayer` | Mean curve |
| 362 | `SpmChartLayer` | SPM Phase I chart; has `from_chart()` and `can_monitor()` methods |
| 414 | `SpmMonitorLayer` | SPM Phase II result |
| 442 | `ExplainLayer` | Explainability result |
| 467 | `CustomLayer` | User-defined extension |

**Enums (2):**

| Line | Type | Already `#[non_exhaustive]`? |
|------|------|------------------------------|
| 97 | `LayerKey` | YES — already has `#[non_exhaustive]` at line 96 |
| 134 | `Layer` | YES — already has `#[non_exhaustive]` at line 133 |

**Type aliases (4, cfg-gated):**

| Lines | Type | Notes |
|-------|------|-------|
| 434/437 | `ExplainExtra` | `serde_json::Value` (serde) / `HashMap<String, Vec<f64>>` (no-serde) |
| 459/462 | `CustomData` | `serde_json::Value` (serde) / `HashMap<String, Vec<f64>>` (no-serde) |

**Total public items in wire:** 18 structs + 2 enums + 4 type aliases (cfg-gated pairs) = **24 public items** [VERIFIED: src/wire.rs]

### Crate-Root Re-Exports of Wire Types

**Result: ZERO.** `src/lib.rs` contains no `pub use wire::*` or any `pub use wire::{...}` statement. [VERIFIED: src/lib.rs]

### Prelude Re-Exports of Wire Types

**Result: ZERO.** `src/prelude.rs` contains no reference to `wire`. [VERIFIED: src/prelude.rs]

### Wire Usage in 28 Examples

**Result: ZERO.** All 28 example directories (`examples/01_simulation/` through `examples/28_berkeley_growth/`) contain no reference to `wire`, `FdaData`, `LayerKey`, or `Layer::`. [VERIFIED: grep across examples/]

### Wire Usage in Integration Tests

**Result: ZERO.** `fdars-core/tests/` contains no reference to `wire`. [VERIFIED: grep across tests/]

### Internal Doctest That Must Be Fixed

**Location:** `src/wire.rs:17-36` — module-level `//!` doc comment containing a fenced code block.

The doctest uses:
```
use fdars_core::wire::*;
use fdars_core::matrix::FdMatrix;
```

After `wire` becomes `pub(crate)`, this external-path `use fdars_core::wire::*;` will fail to compile because `wire` is no longer accessible from the external perspective that `rustdoc` uses to compile module-level doctests.

**Fix:** Change the doctest to use `use crate::wire::*;` — this is the correct path for an internal doctest in a `pub(crate)` module — or annotate with `no_run` if the test intent is illustrative only. [ASSUMED — the correct doctest path for a pub(crate) module in Rust; rustdoc compiles module-level doctests in the crate's internal context when the module is crate-private]

**External-facing example blocker:** NONE. No external-facing example uses wire. [VERIFIED: grep across examples/]

---

## SEAL-02: Config Struct Enumeration

### Summary Counts [VERIFIED: grep + Python analysis across src/]

| Category | Count |
|----------|-------|
| Total `pub struct *Config` | **66** |
| Already sealed with `#[non_exhaustive]` | **28** |
| Need `#[non_exhaustive]` added | **38** |
| `pub(crate) struct *Config` | **0** (none found) |

Note: CONTEXT.md and STATE.md reference "15 already sealed" — this figure appears to be a stale estimate from an earlier audit. The actual count from reading the current tree is **28 already sealed**.

### Already-Sealed Config Structs (28) — No Action Needed

[VERIFIED: grep + Python analysis — `#[non_exhaustive]` confirmed in the 8 lines preceding each struct]

| File | Struct |
|------|--------|
| `clustering_advanced.rs:1267` | `AlignClusterConfig` |
| `clustering_advanced.rs:277` | `KcfcConfig` |
| `clustering_advanced.rs:635` | `FunFemConfig` |
| `clustering_advanced.rs:67` | `DbscanConfig` |
| `coclustering.rs:153` | `CoClusterConfig` |
| `conformal/mod.rs:118` | `ConformalConfig` |
| `detrend/stl.rs:49` | `StlConfig` |
| `famm.rs:1274` | `MultiFammConfig` |
| `famm.rs:1443` | `FastFmmConfig` |
| `famm.rs:927` | `DenseFlmmConfig` |
| `fof_regression.rs:528` | `FofReConfig` |
| `gmm/cluster.rs:51` | `GmmClusterConfig` |
| `gmm/subspace.rs:47` | `FunHddcConfig` |
| `kernel_kmeans.rs:57` | `KernelKmeansConfig` |
| `kshape.rs:70` | `KShapeConfig` |
| `metric/gak.rs:53` | `GakConfig` |
| `optimal_design.rs:208` | `OptDesConfig` |
| `pace_fpca.rs:53` | `PaceFpcaConfig` |
| `scalar_on_function/additive.rs:1038` | `PermTestConfig` |
| `scalar_on_function/additive.rs:1077` | `HistoryIndexConfig` |
| `scalar_on_function/additive.rs:119` | `GsamConfig` |
| `scalar_on_function/additive.rs:67` | `FamConfig` |
| `scalar_on_function/additive.rs:93` | `GkamConfig` |
| `scalar_on_function/additive.rs:966` | `VarSelectConfig` |
| `tolerance/conformal_anomaly.rs:64` | `ConformalAnomalyConfig` |
| `tolerance/types.rs:104` | `ElasticToleranceConfig` |
| `wavelet/regression.rs:729` | `WnetConfig` |
| `wavelet/regression.rs:92` | `WcrConfig` |

---

## SEAL-02 + SEAL-03: Unsealed Config Structs Requiring `#[non_exhaustive]`

All 38 structs below need `#[non_exhaustive]` added immediately before the `#[derive(...)]` line (or at the attribute position). All already have `Default` — **zero new escape hatches needed**.

[VERIFIED: grep + Python analysis across src/ for both missing `#[non_exhaustive]` and presence of Default]

| File | Line | Struct | Escape Hatch Present |
|------|------|--------|----------------------|
| `alignment/bayesian.rs` | 20 | `BayesianAlignConfig` | `impl Default` |
| `alignment/clustering.rs` | 38 | `KMedoidsConfig` | `impl Default` |
| `alignment/diagnostics.rs` | 39 | `DiagnosticConfig` | `impl Default` |
| `alignment/lambda_cv.rs` | 13 | `LambdaCvConfig` | `impl Default` |
| `alignment/multires.rs` | 18 | `MultiresConfig` | `impl Default` |
| `alignment/outlier.rs` | 14 | `ElasticOutlierConfig` | `impl Default` |
| `alignment/partial_match.rs` | 17 | `PartialMatchConfig` | `impl Default` |
| `alignment/robust_karcher.rs` | 20 | `RobustKarcherConfig` | `impl Default` |
| `alignment/shape_ci.rs` | 17 | `ShapeCiConfig` | `impl Default` |
| `alignment/transfer.rs` | 16 | `TransferAlignConfig` | `impl Default` |
| `boosting_regression/mod.rs` | 44 | `BoostingConfig` | `impl Default` |
| `boosting_regression/mod.rs` | 94 | `BayesianConfig` | `impl Default` |
| `boosting_regression/mod.rs` | 140 | `StabilityConfig` | `impl Default` |
| `classification/fit.rs` | 559 | `ClassifCvConfig` | `impl Default` |
| `elastic_regression/mod.rs` | 39 | `ElasticConfig` | `impl Default` |
| `elastic_regression/mod.rs` | 63 | `ElasticPcrConfig` | `impl Default` |
| `elastic_regression/mod.rs` | 90 | `ScalarOnShapeConfig` | `impl Default` |
| `outliers.rs` | 467 | `TvdMssConfig` | `impl Default` |
| `outliers.rs` | 564 | `MuodConfig` | `impl Default` |
| `outliers.rs` | 734 | `SeqTransformConfig` | `impl Default` |
| `outliers.rs` | 870 | `DepthgramConfig` | `impl Default` |
| `peer.rs` | 132 | `PeerConfig` | `derive(Default)` |
| `shapelet/classifier.rs` | 63 | `ShapeletClassifierConfig` | `derive(Default)` |
| `shapelet/discovery.rs` | 51 | `ShapeletDiscoveryConfig` | `impl Default` |
| `smooth_basis.rs` | 840 | `SmoothBasisGcvConfig` | `impl Default` |
| `smooth_basis.rs` | 916 | `BasisNbasisCvConfig` | `impl Default` |
| `spm/amewma.rs` | 41 | `AmewmaConfig` | `impl Default` |
| `spm/arl.rs` | 43 | `ArlConfig` | `impl Default` |
| `spm/cusum.rs` | 29 | `CusumConfig` | `impl Default` |
| `spm/elastic_spm.rs` | 65 | `ElasticSpmConfig` | `impl Default` |
| `spm/ewma.rs` | 46 | `EwmaConfig` | `impl Default` |
| `spm/frcc.rs` | 57 | `FrccConfig` | `impl Default` |
| `spm/iterative.rs` | 50 | `IterativePhase1Config` | `impl Default` |
| `spm/mewma.rs` | 39 | `MewmaConfig` | `impl Default` |
| `spm/mfpca.rs` | 48 | `MfpcaConfig` | `impl Default` |
| `spm/partial.rs` | 81 | `PartialDomainConfig` | `impl Default` |
| `spm/phase.rs` | 29 | `SpmConfig` | `impl Default` |
| `spm/profile.rs` | 61 | `ProfileMonitorConfig` | `impl Default` |

---

## Standard Stack

No external packages are added. This is a pure attribute-insertion and visibility-change phase.

### Rust Language Features Used

| Feature | Usage | Reference |
|---------|-------|-----------|
| `pub(crate)` | Module visibility restriction | Rust Reference: visibility |
| `#[non_exhaustive]` | Prevents external struct literal construction; allows field addition in future | RFC 2008 |

### Key Rule: `#[non_exhaustive]` on Structs

`#[non_exhaustive]` on a struct prevents external crates from using struct literal syntax (`Config { field: value }`). External code must use a constructor (`Config::default()`, `Config::new()`, `..Default::default()` spread). Internal code (same crate) is unaffected — can still use literals. [ASSUMED — standard Rust 1.40+ behavior; well-established pattern already used in this codebase on 28 structs]

### Attribute Placement Convention

Based on existing sealed structs in the codebase (e.g., `GmmClusterConfig`, `StlConfig`), the convention is: [VERIFIED: grep across src/ for existing #[non_exhaustive] + pub struct *Config patterns]

```rust
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SomeConfig {
    ...
}
```

`#[non_exhaustive]` goes FIRST, before `#[derive(...)]` lines. This matches the existing pattern in `GmmClusterConfig`, `StlConfig`, `ConformalConfig`, etc.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Construction escape hatch | Custom builder struct | `Default` + `..Default::default()` spread | Already implemented on all 38 structs; no-op for SEAL-03 |
| Attribute insertion | sed/awk scripts | Direct `Edit` tool | Surgical edits per file preserve surrounding context |

---

## Architecture Patterns

### Recommended Task Structure

This phase has three natural atomic tasks:

**Task A — SEAL-01: Wire Module Seal**
1. `src/lib.rs:125` — change `pub mod wire;` → `pub(crate) mod wire;`
2. `src/wire.rs:17-18` — fix internal doctest path from `fdars_core::wire::*` to `use crate::wire::*;`
3. Verify: `cargo build` + `cargo test --doc` pass

**Task B — SEAL-02 + SEAL-03: Config Struct Sealing (38 sites)**
- Insert `#[non_exhaustive]` above the `#[derive(...)]` line on each of the 38 unsealed structs
- Group by file to minimize context switches (SPM has 9 structs, alignment has 10, etc.)
- No new Default impls needed
- Verify: `cargo build --features linalg,parallel,serde` passes

**Task C — Documentation Update**
- Update each sealed config struct's doc comment to show `..Default::default()` usage pattern where not already present
- This is the "documented path" requirement of SEAL-03

### System Architecture Diagram

```
SEAL-01 (wire seal)
  src/lib.rs:125  →  pub(crate) mod wire
  src/wire.rs:17  →  fix internal doctest path
  [no re-exports to remove — lib.rs and prelude.rs have zero wire pub use]

SEAL-02+03 (config sealing, 38 sites)
  alignment/ (10 structs)
  boosting_regression/ (3 structs)
  classification/ (1 struct)
  elastic_regression/ (3 structs)
  outliers.rs (4 structs)
  peer.rs (1 struct)
  shapelet/ (2 structs)
  smooth_basis.rs (2 structs)
  spm/ (9 structs)
  └─ All already have impl Default → zero new escape hatch code needed
```

### File Groupings for Efficient Editing

| Group | Files | Struct Count |
|-------|-------|-------------|
| SPM | `spm/amewma.rs`, `spm/arl.rs`, `spm/cusum.rs`, `spm/elastic_spm.rs`, `spm/ewma.rs`, `spm/frcc.rs`, `spm/iterative.rs`, `spm/mewma.rs`, `spm/mfpca.rs`, `spm/partial.rs`, `spm/phase.rs`, `spm/profile.rs` | 12 (but `IterativePhase1Config` already in unsealed list → 9 in SEAL-02 scope) |
| Alignment | `alignment/bayesian.rs`, `alignment/clustering.rs`, `alignment/diagnostics.rs`, `alignment/lambda_cv.rs`, `alignment/multires.rs`, `alignment/outlier.rs`, `alignment/partial_match.rs`, `alignment/robust_karcher.rs`, `alignment/shape_ci.rs`, `alignment/transfer.rs` | 10 |
| Elastic regression | `elastic_regression/mod.rs` | 3 |
| Outliers | `outliers.rs` | 4 |
| Boosting | `boosting_regression/mod.rs` | 3 |
| Shapelet | `shapelet/classifier.rs`, `shapelet/discovery.rs` | 2 |
| Smooth basis | `smooth_basis.rs` | 2 |
| Classification | `classification/fit.rs` | 1 |
| Peer | `peer.rs` | 1 |

---

## Common Pitfalls

### Pitfall 1: Doctest in `pub(crate)` Module Using External Path

**What goes wrong:** `src/wire.rs` module-level doctest uses `use fdars_core::wire::*;`. After making `wire` `pub(crate)`, `cargo test --doc` fails with "module `wire` is private".

**Why it happens:** Module-level doctests (`//!` docs) are compiled as if from an external crate perspective. A `pub(crate)` module is not accessible from that perspective.

**How to avoid:** Change the doctest `use` statement to `use crate::wire::*;` — which is valid in an `//!`-level doctest of a `pub(crate)` module since it compiles in the crate's own context. Alternatively, annotate the doctest `` ``` `` block with `no_run` if the test is illustrative-only.

**Warning signs:** `cargo test --doc` failing on `wire.rs` after the visibility change.

### Pitfall 2: Wrong `#[non_exhaustive]` Placement

**What goes wrong:** `#[non_exhaustive]` placed AFTER `#[derive(...)]` or AFTER `#[cfg_attr(...)]` lines — this is valid Rust (attribute order doesn't matter for most), but inconsistent with codebase convention.

**How to avoid:** Always place `#[non_exhaustive]` as the first attribute before the struct, matching the existing convention in `GmmClusterConfig`, `StlConfig`, `ConformalConfig`, etc.

### Pitfall 3: Confusing "Already-Sealed Count" from Stale Estimates

**What goes wrong:** Planning from the CONTEXT.md estimate of "15 already sealed" and targeting only ~22 unsealed leads to missing 6 already-sealed + 16 unsealed structs.

**How to avoid:** Use this research's enumeration — **28 already sealed, 38 to seal**.

### Pitfall 4: Missing `--all-targets` on Clippy

**What goes wrong:** Running `cargo clippy -p fdars-core -- -D warnings` (without `--all-targets`) passes but CI fails because test/bench code has warnings.

**How to avoid:** Always run `cargo clippy --all-targets --features linalg,parallel -- -D warnings` per CLAUDE.md and STATE.md. [VERIFIED: .claude/CLAUDE.md, .planning/STATE.md]

### Pitfall 5: `cargo build --features serde` Regressing

**What goes wrong:** The `serde` feature build was repaired in v0.40.0 after `ShapeletTransformClassifier` broke it. Any new struct that gets `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` added without all its fields implementing `Serialize`/`Deserialize` breaks this gate.

**How to avoid:** For SEAL-02/03, we're only adding `#[non_exhaustive]` — no new serde derives. The risk is zero for this phase, but the gate must still be verified.

---

## Project Constraints (from CLAUDE.md)

- **Scope**: `fdars-core` only. No new crate dependency.
- **Build gates**: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, `cargo build --features serde`.
- **MSRV**: 1.81.0 — `#[non_exhaustive]` has been stable since Rust 1.40.0; no MSRV impact.
- **All 28 examples + doctests must compile** — the compile-time proof the breaking changes are complete.
- **No behavioral change** — `#[non_exhaustive]` only restricts struct literal construction from external crates; in-crate construction (tests, doctests inside the crate) is unaffected.
- **Column-major `FdMatrix`** — no matrix operations in this phase.
- **`Result<T, FdarError>`** — no error handling changes in this phase.
- **All public types derive `Debug, Clone, PartialEq`** — no change to derives in this phase (adding `#[non_exhaustive]` does not affect existing derives).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`) + `criterion 0.5` (benchmarks) + `rustdoc` (doctests) |
| Config file | None — `cargo test` discovers automatically |
| Quick run command | `cargo build --features linalg,parallel` |
| Full suite command | `cargo test --features linalg,parallel` |

### Phase Validation Strategy

This phase is **API-shape-only**. Validation is entirely **compile-time** — no new test logic is required. The existing test suite provides the behavioral non-regression proof (byte-identical logic, `#[non_exhaustive]` is transparent inside the crate). The gates are:

| Gate | Command | What It Proves |
|------|---------|----------------|
| Compile (all features) | `cargo build --features linalg,parallel` | Wire sealed, attribute syntax correct |
| Serde build | `cargo build --features serde` | No serde regression |
| Fmt | `cargo fmt --check` | No formatting drift |
| Clippy (all targets) | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | No warnings in test/bench code |
| Doc tests | `cargo test --doc --features linalg,parallel` | Wire doctest path fix works |
| Full test suite | `cargo test --features linalg,parallel` | No behavioral change (existing tests pass) |
| Examples (all 28) | `cargo build --examples` | All examples still compile with sealed surfaces |
| Package | `cargo package -p fdars-core` | Publishable artifact correct |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| SEAL-01 | `wire` not accessible externally | Compile gate | `cargo build` | Rust compiler enforces visibility |
| SEAL-01 | All 28 examples still compile | Compile gate | `cargo build --examples` | No example uses wire — all should pass without changes |
| SEAL-01 | Doctests still pass | Doctest gate | `cargo test --doc` | Requires wire.rs doctest path fix |
| SEAL-02 | Config structs have `#[non_exhaustive]` | Compile + manual verify | `cargo build` + code review | Rust compiler accepts attribute; no test for "attribute is present" |
| SEAL-02 | No external literal construction | Compile gate | External downstream test not in scope | `#[non_exhaustive]` enforced by compiler for external callers |
| SEAL-03 | `Default` path works | Existing test suite | `cargo test --features linalg,parallel` | Existing tests exercise `Default::default()` calls already |

### Wave 0 Gaps

None — existing test infrastructure covers all phase requirements. No new test files needed. The compile gates are the proof.

---

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1` per config.

### Applicable ASVS Categories

| ASVS Category | Applies | Rationale |
|---------------|---------|-----------|
| V2 Authentication | No | No auth logic in this phase |
| V3 Session Management | No | No session logic |
| V4 Access Control | No | Rust module visibility is a language-level control, not an application-level access control concern |
| V5 Input Validation | No | No new input paths |
| V6 Cryptography | No | No cryptographic operations |

**Security assessment:** This phase has no meaningful security surface. `pub(crate)` visibility and `#[non_exhaustive]` are compile-time language features that affect API shape, not runtime behavior. No ASVS requirements apply.

---

## Runtime State Inventory

> Not applicable — this is a greenfield attribute-insertion phase (not a rename/refactor of stored data).

Nothing found in any category — this phase only changes Rust source code attributes and visibility modifiers. No stored data, live service config, OS-registered state, secrets, or build artifacts are affected by adding `#[non_exhaustive]` or changing module visibility.

---

## Environment Availability

| Dependency | Required By | Available | Notes |
|------------|------------|-----------|-------|
| `cargo` | All build/test gates | Confirmed (Rust 1.97.0 in dev) | MSRV 1.81.0; `#[non_exhaustive]` stable since 1.40.0 |
| `rustfmt` | `cargo fmt --check` | Standard toolchain component | |
| `clippy` | `cargo clippy` | Standard toolchain component | |

No missing dependencies.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Module-level `//!` doctests in a `pub(crate)` module must use `use crate::wire::*;` (not `fdars_core::wire::*;`) | SEAL-01 Doctest section | `cargo test --doc` fails; easily diagnosed and fixed during Task A |
| A2 | The "15 already sealed" figure in CONTEXT.md/STATE.md is a stale estimate from Phase 81 audit | SEAL-02 summary | If wrong, the count-based planning notes would be off, but the concrete struct list is independently verified |

**If this table is shrinks to 0:** All claims in this research were verified by direct file reads — the Assumptions Log above reflects the only two claims not confirmed by a direct `Read` of the authoritative definition.

---

## Open Questions

1. **`//!` doctest path in `pub(crate)` module**
   - What we know: `rustdoc` compiles module-level doctests; the path `fdars_core::wire::*` will fail post-sealing
   - What's unclear: Whether `use crate::wire::*;` is the right internal path in a `//!` module doctest, or whether the test should be annotated `no_run`
   - Recommendation: Use `use crate::wire::*;` — this is the standard approach for internal module doctests. If compilation still fails, add `no_run` and convert to a `#[cfg(test)]` unit test.

2. **Documentation update scope for SEAL-03**
   - What we know: All 38 structs already have `Default` impls
   - What's unclear: How many of the 38 already document `..Default::default()` usage in their doc comments
   - Recommendation: Planner should scope a documentation-update sub-task; a brief `# Example` section showing `let cfg = SomeConfig { extra_field: val, ..Default::default() };` satisfies SEAL-03's "documented path" requirement

---

## Sources

### Primary (HIGH confidence)
- `src/wire.rs` — direct read; all 24 public types enumerated verbatim
- `src/lib.rs` — direct read; wire re-export status confirmed (zero re-exports)
- `src/prelude.rs` — direct read; wire re-export status confirmed (zero)
- `src/` tree — grep + Python analysis; 66 `pub struct *Config` identified, 28/38 sealed/unsealed split confirmed
- `examples/` tree — grep confirmed zero wire references across all 28 examples
- `tests/` tree — grep confirmed zero wire references

### Secondary (MEDIUM confidence)
- CONTEXT.md, REQUIREMENTS.md, STATE.md — project planning artifacts; used for constraint extraction

### Tertiary (LOW confidence)
- A1, A2 in Assumptions Log — training knowledge about Rust doctest compilation behavior for `pub(crate)` modules

---

## Metadata

**Confidence breakdown:**
- Wire type enumeration: HIGH — direct file read, verbatim
- Re-export status: HIGH — direct file read confirmed zero
- Example/test wire usage: HIGH — grep confirmed zero
- Config struct enumeration: HIGH — grep + Python analysis across full src/ tree
- Already-sealed vs unsealed split: HIGH — Python script reading each file
- Escape hatch status: HIGH — Python script searching `impl Default for X` and `derive(Default)` patterns
- Doctest fix requirement: MEDIUM (A1) — training knowledge about Rust `pub(crate)` doctest behavior

**Research date:** 2026-09-08
**Valid until:** Indefinite — based on direct codebase reads, not web sources; re-run if src/ is substantially modified before planning
