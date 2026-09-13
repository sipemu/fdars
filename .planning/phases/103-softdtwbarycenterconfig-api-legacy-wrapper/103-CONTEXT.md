# Phase 103: SoftDtwBarycenterConfig API & Legacy Wrapper - Context

**Gathered:** 2026-09-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Expose the Phase 101/102 optimizer through a public `SoftDtwBarycenterConfig`
builder-convention config struct + a new config-taking entry point, while preserving the
existing `soft_dtw_barycenter(data, gamma, max_iter, tol)` signature as a thin,
behavior-compatible wrapper so NO existing call site, prelude re-export, R/WASM binding, or
example changes (SDTW-03). New public items are additively re-exported (crate root +
prelude) with a running doctest. This is the non-breaking API surface; the escape-from-
local-minimum quality evidence is Phase 104 and release prep is Phase 105.

</domain>

<decisions>
## Implementation Decisions

### Config API Design
- **New entry point:** `soft_dtw_barycenter_with_config(data: &FdMatrix, config: &SoftDtwBarycenterConfig) -> SoftDtwBarycenterResult` — mirrors the crate's `stl_decompose_with_config` naming convention.
- **`SoftDtwBarycenterConfig`** fields + defaults: `gamma = 1.0`, `max_iter = 100`,
  `tol = 1e-6`, `n_restarts = 10`, `seed = 0`. Derives `#[derive(Debug, Clone, PartialEq)]`
  and is `#[non_exhaustive]` — matching the sealed-config convention (external construction
  via `SoftDtwBarycenterConfig::default()` + field-assign). **`Default` is a MANUAL `impl`,
  NOT `#[derive(Default)]`** — a derived Default on the numeric fields would give
  `gamma = 0.0` / `max_iter = 0` (a broken soft-DTW config: gamma=0 is exactly the invalid
  input Phase 101's guard rejects). The manual impl returns the documented non-zero defaults
  above. (This corrects the incidental "derive Default" shorthand from the original discuss —
  the substantive locked decision is the default VALUES, which only a manual impl can deliver;
  `StlConfig` can derive Default only because its fields are all `Option`/`bool`.) It
  internally drives `soft_dtw_barycenter_multirestart` (Phase 102).
- **Legacy wrapper:** `soft_dtw_barycenter(data, gamma, max_iter, tol)` keeps its exact 4-arg
  signature and delegates to the config path with `n_restarts = 1` — bit-identical to the
  current single-init behavior (a single unperturbed-mean L-BFGS restart). No behavior change
  for any existing caller.
- **`SoftDtwBarycenterResult` derives:** add `PartialEq` (convention: all public result types
  derive `Debug, Clone, PartialEq`; resolves the deferred Phase 102 IN-02). Do NOT add serde
  derives — not required by SDTW-03 and avoids the known crate-wide serde-build hazard.

### Claude's Discretion
- Whether the config struct/entry point live in `soft_dtw.rs` or a small sibling section.
- Builder ergonomics: plain public fields + `Default` (StlConfig style) is sufficient; no
  fluent setter methods required unless trivial.
- Exact doctest example content (must compile + run under `cargo test --doc`).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/metric/soft_dtw.rs` — Phase 102's private `soft_dtw_barycenter_multirestart(data, gamma, max_iter, tol, n_restarts, seed)` (currently `#[allow(dead_code)]` — this phase wires it through the public config API, removing the allow), `lbfgs_refine_from_init`, `soft_dtw_barycenter` (4-arg), `SoftDtwBarycenterResult`.
- Convention reference: `fdars-core/src/detrend/stl.rs` — `StlConfig` (`#[derive(Debug, Clone, PartialEq, Default)] #[non_exhaustive]`, public fields) + `stl_decompose_with_config`.

### Established Patterns
- Sealed configs: `#[non_exhaustive]` on config structs (v0.42.0 SEAL-02); external
  construction is `Config::default()` then field assignment (only valid external ctor form).
- Re-export chain: `metric/mod.rs` → `lib.rs` (crate root). Prelude (`prelude.rs`) currently
  re-exports `soft_dtw_distance_generic` but NOT `soft_dtw_barycenter`.

### Integration Points
- Add `SoftDtwBarycenterConfig` + `soft_dtw_barycenter_with_config` to the `metric/mod.rs`
  `pub use`, the crate-root `lib.rs` re-export, AND `prelude.rs` (SDTW-03 requires crate root
  + prelude). Keep the existing `soft_dtw_barycenter` / `SoftDtwBarycenterResult` re-exports
  intact (byte-compatible).

</code_context>

<specifics>
## Specific Ideas

- Non-breakingness is the CENTRAL constraint: all 28 examples + R/WASM bindings + every call
  site must compile unchanged. The 4-arg fn signature survives byte-for-byte; the config API
  is purely additive.
- Behavior-compatibility of the legacy wrapper: `n_restarts = 1` reproduces the current
  single-init result exactly (assert parity in a test).
- A running doctest on `soft_dtw_barycenter_with_config` (compiles + runs) satisfies the
  SDTW-03 doctest requirement.

</specifics>

<deferred>
## Deferred Ideas

- Escape-from-local-minimum quality evidence vs MM + legacy f64 parity example (SDTW-04) → Phase 104.
- Release prep / CHANGELOG / version bump (REL-01) → Phase 105.
- serde derives on the config/result — intentionally out of scope (not required by SDTW-03).

</deferred>
