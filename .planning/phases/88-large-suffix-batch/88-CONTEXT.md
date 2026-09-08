# Phase 88: Large Suffix Batch - Context

**Gathered:** 2026-09-08
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure/rename phase — smart discuss skipped)

<domain>
## Phase Boundary

The remaining crate-wide lone-`_1d`/`_2d` suffix functions are unified onto `Dim` dispatch and the entire example/doc surface is migrated. This is NAME-04 / AUD-22 — the **highest-risk, highest-blast-radius** item, isolated in its own phase. API-shape-only, no numeric/behavioral change: every consolidated public signature routes to **byte-identical private `_impl` bodies** (the v0.41.0 + Phase 87 pattern).

Recon (2026-09-08): ~46 lone `pub fn *_1d` + 3 lone `pub fn *_2d` remain. **No true `_1d`/`_2d` pairs remain** (Phase 87 consolidated the last pairs) — every remaining `_1d` is lone (no `_2d` sibling). Some already have a bare `Dim`-dispatcher (`mean`, `modal`, `fraiman_muniz`, `random_tukey`) with a **redundant public `_1d` still exposed**; others (`center`, `band`, `norm_lp`, `deriv_self`, `dtw_self`, `fourier_self`, …) have no dispatcher yet.

</domain>

<decisions>
## Implementation Decisions

### Established convention (NOT a grey area — the milestone + crate already decided this)
`src/dim.rs` documents the pattern verbatim: historically `_2d` was "a thin shim that simply forwarded to the `_1d` primitive"; the unified dispatchers take an explicit `Dim` and callers use `name(…, Dim::Two)`. `fdata::mean` / `depth::modal` / `fraiman_muniz` / `random_tukey` already implement it as `pub fn name(.., dim: Dim) { match dim { Dim::One | Dim::Two => name_1d_impl(..) } }`.

**Strategy per category (research confirms/refines the exact set):**
1. **Lone `_1d` WITH an existing bare dispatcher** (mean, modal, fraiman_muniz, random_tukey, …): the dispatcher already IS the public API → seal the redundant public `_1d` to a private `_1d_impl` (or point the dispatcher at the existing body); remove the public `_1d` name.
2. **Lone `_1d` WITHOUT a dispatcher:** add a public `name(.., dim: Dim)` dispatcher routing `Dim::One | Dim::Two` to a private `name_1d_impl` (body byte-identical); remove the public `_1d`. Preserve `#[must_use]`/attributes on the public dispatcher.
3. **`_self`/`_cross` families** (dtw_self/dtw_cross, fourier_self/fourier_cross, hshift_self/hshift_cross, kl_self/kl_cross, pca_self/pca_cross, basis_coef_self/…, soft_dtw_self/cross/div): each `_self`/`_cross` member gets its own dispatcher (`dtw_self(.., dim)`, `dtw_cross(.., dim)`).
4. **Lone `_2d` (no `_1d`)** — `fosr_2d`, `predict_fosr_2d` (genuinely 2D function-on-scalar; there is a separate 1D FOSR under a different name), `simpsons_weights_2d` (helper): research must decide per case — likely **drop the `_2d` suffix** (`fosr_2d`→`fosr`, `predict_fosr_2d`→`predict_fosr`) since there is no `_1d` ambiguity, OR leave if a name collision exists. Do NOT force a `Dim` param onto an inherently-2D-only function.
5. **Exclusions:** any `_1d`/`_2d` that is NOT public (`pub(crate)`/private), a struct field, or where the suffix is load-bearing and not a dimensionality selector — leave untouched. Research must flag these (e.g. `from_1d`, `basis_to_fdata_1d`, `simpsons_weights_2d` if it is a `pub(crate)` helper).

### Claude's Discretion
Pure infrastructure/refactor — all mechanics at Claude's discretion, following the above convention. Byte-identical `_impl` bodies (code-review gate confirms zero drift). `fdars-core` only, no new dependency.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/dim.rs` — `enum Dim { One, Two }` (data-free, `#[non_exhaustive]`) + the documented convention.
- The Phase 87 consolidation + call-site-rewrite tooling (scripts) is the reference: move `_1d` body verbatim → private `_impl`, add `Dim` dispatcher, crate-wide rewrite of call sites, grep+compiler proves completeness.
- Existing dispatchers to copy: `fdata::mean`, `depth::modal`, `depth::fraiman_muniz`, `depth::random_tukey`.

### Integration Points
- `src/lib.rs` + `src/prelude.rs` re-export blocks (drop removed `_1d`/`_2d` names, add dispatchers).
- Per-module `mod.rs` re-exports (depth, metric, basis, regression, etc.).
- **All 28 examples + all doctests + `tests/` + `benches/`** — the compile-time proof (highest blast radius of the milestone).

</code_context>

<specifics>
## Specific Ideas

Research MUST produce: (1) the COMPLETE enumeration of every public `_1d`/`_2d` function with `file:line`, grouped by family/module; (2) a per-function category (1–5 above) + the exact new public name + private `_impl` name; (3) the decision for each lone `_2d`; (4) the list of exclusions (non-public / load-bearing-suffix) with justification; (5) the module re-export blocks that need editing; (6) confirmation of the gate set. Given the scale, the planner should decompose into grouped tasks (e.g. by module: depth, metric, basis, regression/fpca, fdata, soft-dtw, function-on-scalar) each verified by compile.

</specifics>

<deferred>
## Deferred Ideas

None — this is the terminal naming-unification phase. Release prep (REL-01) is Phase 89.

</deferred>
