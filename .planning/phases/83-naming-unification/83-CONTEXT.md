# Phase 83: Naming Unification - Context

**Gathered:** 2026-09-07
**Status:** Ready for planning
**Mode:** Scope-bound (approved at Phase 81 gate) + one design decision resolved with the user

<domain>
## Phase Boundary

Apply the APPROVED Scope-D naming unification (API-04) from the Phase 81 audit inventory: `AUD-14`–`AUD-18` ONLY. Breaking is limited to API shape (names/dispatchers); NO numeric or behavioral change. All call sites, all 28 examples, and all doctests migrate to the new names; whole-crate gates green. The largest/highest-risk phase — scope was deliberately reduced to the recommended set at the approval gate.

</domain>

<decisions>
## Implementation Decisions

### Approved change set (from `81-AUDIT-INVENTORY.md` ## Approval)

- **`AUD-14`** — rename `funhddC_cluster` → `fun_hddc_cluster` (fix camelCase-in-snake_case Rust-convention violation). Def `src/gmm/subspace.rs:554`; crate-root re-export `lib.rs:481`; update all ~15 in-crate/test/doc references + the module doc + doctest at `subspace.rs:546`. Hard rename (no deprecated shim — still 0.x).
- **`AUD-15`** — rename type `FosrResult2d` → `Fosr2dResult` (adjective-before-noun, matches `Grid2d`/`Deriv2DResult` house style). Def `src/function_on_scalar_2d.rs:69`; re-export `lib.rs:399`; ~7 references.
- **`AUD-16`** — rename type `GmmResult` → `GmmFitResult` (single-K fit result; disambiguates from `GmmClusterResult` for K-selection). Def `src/gmm/mod.rs:39`; re-export `lib.rs:482`; ~13 references.
- **`AUD-17`** — collapse `deriv_1d`/`deriv_2d` → a single `deriv` dispatcher (see grid-enum decision below). Def `src/fdata.rs:862,944`; ~22 (`deriv_1d`) + ~7 (`deriv_2d`) references.
- **`AUD-18`** — collapse `lp_self_1d`/`lp_cross_1d`/`lp_self_2d`/`lp_cross_2d` → `lp_self`/`lp_cross` dispatchers (grid-enum). Def `src/metric/lp.rs:40,94,126,160`; ~19+14+6+5 references.

### DESIGN DECISION — grid-enum dispatcher (resolved with user, 2026-09-07)
The `_1d`/`_2d` forms of `deriv`/`lp` have DIFFERENT signatures (2D takes separate `argvals_s`/`argvals_t`; `deriv_2d` returns `Option<Deriv2DResult>` vs `deriv_1d` → `FdMatrix`), so the crate's existing `name(…, Dim)` last-arg pattern (used by mean/depth, where 2D was a thin shim over one flattened `argvals`) does NOT fit. **User chose: grid-enum dispatcher.**
- Introduce a small domain/grid enum that carries the dimension-specific grid data, e.g.:
  - for lp: a `LpDomain` (or reuse a shared grid enum) with `OneD(&[f64])` / `TwoD(&[f64], &[f64])`.
  - for deriv: same grid concept, carrying `nderiv` for 1D and `m1`/`m2` for 2D as the enum payload (or a paired param) — planner picks the cleanest shape.
- `deriv` returns a **unified result enum** (e.g. `DerivResult::OneD(FdMatrix)` / `DerivResult::TwoD(Deriv2DResult)`) so the single function covers both without an awkward bare `Option`. `lp_self`/`lp_cross` return `FdMatrix` (both arities already do).
- Naming: prefer a `#[non_exhaustive]`-free simple enum is fine here (these are inputs/outputs the user constructs/matches; keep ergonomic). The dispatcher fns take the grid enum in the argvals position.
- Behavior MUST be identical to the current `_1d`/`_2d` implementations — the new dispatcher just routes to the existing computation bodies (keep them as private helpers). This is a pure shape change; assert numeric parity via the existing tests (which migrate to the new API).

### OUT of scope (deferred to STAB-03 / Phase 84)
- `AUD-19` (geometric_median), `AUD-20` (hausdorff), `AUD-21` (functional_spatial), `AUD-22` (the large lone-`_1d`/`_2d` suffix batch), `AUD-23` (`LpeerResult`) — all deferred.
- Do NOT touch `_nd` alignment fns, `fosr_2d`, `_seeded` suffix, `FdCurveSet::from_1d` (correct as-is per inventory).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- The `Dim` enum (`src/dim.rs`) and existing dispatchers (`mean`, `fraiman_muniz`, `modal`, `random_tukey`, `random_projection`) are the naming template for the SUFFIX-FREE public name — but their single-`argvals` shape does not transfer to deriv/lp (hence the grid enum).
- `Grid2d`, `Deriv2DResult` (`src/function_on_scalar_2d.rs`, `src/fdata.rs`) show the crate's `<Adjective>2d`/`2D` house style for 2D types (informs AUD-15 target name).

### Established Patterns
- Public renames update: definition site, the `pub use` re-export in `lib.rs` (and `prelude.rs`/barrel `mod.rs` if present), all in-crate callers, unit tests, doctests (```rust fences + intra-doc `[...]` links), and any of the 28 examples that use the symbol.
- Doctests and intra-doc links are compile-time-checked — a stale link fails `cargo test`/`cargo doc`.

### Integration Points
- `deriv`/`lp` are used across metric/depth/regression/examples — the grid-enum change ripples to every caller; each must be migrated to construct the enum. Blast radius (grounded): deriv ~29 refs, lp ~44 refs, funhddC ~15, GmmResult ~13, FosrResult2d ~7.
- Check `src/prelude.rs` for any of these symbols and update re-exports there too.

</code_context>

<specifics>
## Specific Ideas

- Whole-crate gates (must stay green): `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, `cargo test`, `cargo build --features serde`, all 28 examples compile. The 28-example + doctest compile IS the proof the rename is complete.
- Build/CI hazards (MEMORY): pre-commit hook full-test-suite times out at 30s → `cargo fmt` then `git commit --no-verify`, gates out-of-band. `/home` disk pressure — `rm -rf target/debug/{incremental,examples}` if spurious "No space left"/link failure. Full `cargo test` may show the pre-existing co_cluster/svd_sign golden flake — re-run the specific test binary in isolation to confirm it's the env flake, not a rename regression (see MEMORY: cocluster-svdsign-golden-flake-fullrun).
- This is the highest-risk phase: sequence the renames so the crate compiles after each logical rename (e.g. one symbol/family per commit), making regressions easy to localize.

</specifics>

<deferred>
## Deferred Ideas

- `AUD-19`–`AUD-23` optional naming → STAB-03 1.0 gap checklist (Phase 84).

</deferred>
