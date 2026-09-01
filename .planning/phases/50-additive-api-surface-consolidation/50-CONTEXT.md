# Phase 50: Additive API-Surface Consolidation - Context

**Gathered:** 2026-09-01
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — additive API pass driven by the PROF-03 inventory

<domain>
## Phase Boundary

Give users a single canonical, consistent entry point for previously-inconsistent config/result
patterns and redundant public functions — **additively only**: add the new form + `#[deprecated]` the
old, NEVER remove or rename an existing public signature. The 28 examples, R/WASM bindings, and
external callers must all keep compiling and passing, emitting **deprecation warnings only** (zero
breakage). Requirements: API-01 (config/result consistency), API-02 (redundant-function unification),
API-03 (back-compat: old forms compile + bindings/examples pass). Consumes
`.planning/phases/46-whole-crate-profiling-measurement/PROF-03-api-inventory.md`.

Out of scope: any BREAKING change — public field renames, function renames/removals — deferred to
APIB-01 (future 1.0-readiness milestone). No new crate dependency.

</domain>

<decisions>
## Implementation Decisions

### Item #1 — Config `Default` impls (API-01, additive-safe, HIGH)
- Add `impl Default` to the 4 config structs currently missing it (52/56 already have it):
  `BoostingConfig` (`src/boosting_regression/mod.rs:44`), `BayesianConfig` (`:76`),
  `StabilityConfig` (`:103`), `StlConfig` (`src/detrend/stl.rs:49`). Pull default values from each
  struct's field doc-comments / the constructor most callers use. Purely additive (adds a trait impl).

### Item #2 — Seedable `fanova` (API-01/API-02, additive-safe, HIGH reproducibility gap)
- Add `pub fn fanova_seeded(data, groups, n_perm, seed) -> Result<FanovaResult, FdarError>` matching
  the sibling `(…, n_perm, seed)` convention (`t_perm_test`, `f_perm_test`, `frechet_anova`,
  `generic_permutation_importance`).
- Keep `fanova(data, groups, n_perm)` as a `#[deprecated(note = "use fanova_seeded for reproducible
  permutation p-values")]` shim that DELEGATES to `fanova_seeded` with a FIXED seed that reproduces
  the current `fanova` output bit-identically (capture a golden — `fanova` currently uses a
  hardcoded-42 LCG per PROF-02; the shim must preserve that exact output). Behavior-preserving for the
  old path; new path is seedable.

### Item #3 — Result-field naming (API-01, DOC-ONLY)
- Field renames are BREAKING → DEFER to APIB-01. Phase 50 action is documentation only: document the
  canonical vocabulary (`FpcaResult`: `scores`/`rotation`/`mean`/`weights`/`singular_values`/`centered`)
  and note the `fitted` vs `fitted_values` variance as acceptable. The `FpcPredictor` trait already
  provides the cross-model unification layer. NO field renames.

### Item #4 — `_1d`/`_2d` unified dispatch (API-02, additive-safe) — OPERATOR-CONFIRMED SCOPE
- Add a unified `Dim`-dispatch wrapper (e.g. `mean(data, dim: Dim)` with `Dim::One`/`Dim::Two`) for the
  **HIGH-IMPACT families only: depth, regression, fdata**. Keep the existing `_1d`/`_2d` functions;
  `#[deprecated]` an `_1d`/`_2d` pair ONLY where a clean unified form actually ships for it.
- **NEVER deprecate or touch genuine `_nd` ALGORITHMS** — `pca_nd`, `karcher_mean_nd`,
  `karcher_covariance_nd`, `srsf_transform_nd`, `srsf_inverse_nd` are *different multivariate/shape
  algorithms*, not dimension conveniences. Do NOT bulk-unify all 30+ `_1d`/`_2d` fns — only the three
  named families. The remaining families are deferred (APIB-01 or a later additive pass).

### API-03 — Back-compat gate (all items)
- After every change: the 28 examples build, R/WASM binding surfaces compile, and the full suite
  passes — with **deprecation warnings only**, zero errors. Deprecation warnings on the crate's OWN
  internal call sites of newly-deprecated fns must be silenced (`#[allow(deprecated)]` at the call
  site or migrate the internal caller to the new form) so `-D warnings` gates stay green.

### Gates (Claude's Discretion on exact mechanics)
- Golden-capture the pre-change output of `fanova` (and any behavior-touching path) and assert
  bit-identical via the deprecated shim (mirror the Phase 47/48/49 equivalence pattern —
  `tests/equivalence_phase50.rs` or reuse existing tests).
- Full suite green under BOTH feature configs; `clippy --all-targets --features linalg,parallel -- -D
  warnings` clean (note: deprecation warnings become errors under `-D warnings` — the crate must not
  warn on its own deprecated items).
- Build the 28 examples (`cargo build --examples`) as an explicit API-03 gate — WATCH DISK: `target/`
  can fill /home; `rm -rf target/debug/{incremental,examples}` if example LINK fails ("linking with cc
  failed" — MEMORY pointer). Prefix cargo with `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`.
- `cargo fmt -p fdars-core` + `git commit --no-verify` per commit.

</decisions>

<code_context>
## Existing Code Insights

### PROF-03 anchors
- 4 missing-Default configs: `boosting_regression/mod.rs:44/76/103`, `detrend/stl.rs:49`.
- `fanova` at `function_on_scalar.rs:791` (no seed; hardcoded-42 LCG per PROF-02 — the shim must
  reproduce it). `FanovaResult` is its result type.
- `_1d`/`_2d` families: depth (`mean_1d/2d`, `modal_1d/2d`, `band_*`, `random_tukey_1d/2d`,
  `random_projection_1d/2d`, `lp_self_1d/2d`), regression, fdata. 13 families total; only depth +
  regression + fdata are in scope.
- Canonical result vocab: `FpcaResult` (`regression.rs:25`). `FtsmResult.fitted` (`fts/mod.rs:191`),
  `BoostFosrResult.fitted` (`boosting_regression/mod.rs:127`).

### Established patterns to reuse
- Config structs are `#[non_exhaustive]` with builder-style field mutation after `default()` — the new
  `Default` impls must follow suit (and existing `#[non_exhaustive]` means external struct-literal
  construction is already disallowed, so adding `Default` is safe).
- `#[deprecated(note = "…")]` is the standard additive-deprecation vehicle (no prior use in-crate —
  confirm none exist; this milestone introduces the pattern).
- `FpcPredictor` trait (`explain_generic/mod.rs`) = the cross-model unification precedent.
- Equivalence-golden-then-change pattern (Phases 47/48/49).

### Must-still-pass surfaces (API-03)
- 28 `[[example]]` entries in `fdars-core/Cargo.toml`.
- R bindings (`fdars-r`, external) + WASM (`js` feature) binding surfaces — at minimum they must
  compile against the changed crate with deprecation warnings only.

</code_context>

<specifics>
## Specific Ideas

- Introduce a `Dim` enum (`Dim::One` / `Dim::Two`) for the unified dispatchers; place it where the
  depth/regression/fdata modules can share it (crate root or a small `dim.rs`).
- `fanova` deprecated shim delegates to `fanova_seeded` with the seed that reproduces its current
  hardcoded-LCG output bit-identically (golden-verified).
- Tracer-first: land item #1 (4 Default impls — smallest, self-contained) as the tracer proving the
  additive+deprecation+examples-gate pipeline, THEN #2 (fanova_seeded), THEN #4 (the dispatch families),
  with #3 docs folded in.

</specifics>

<deferred>
## Deferred Ideas

- Result-field RENAMES (`fitted` → `fitted_values` etc.) — breaking → APIB-01. Doc-only this phase.
- Genuine `_nd` algorithms (`pca_nd`, `karcher_mean_nd`, `karcher_covariance_nd`, `srsf_transform_nd`,
  `srsf_inverse_nd`) — different algorithms, NEVER deprecated.
- `_1d`/`_2d` families OTHER than depth/regression/fdata — deferred (APIB-01 or later additive pass).
- Any breaking removal of the newly-`#[deprecated]` forms — APIB-01.

</deferred>
