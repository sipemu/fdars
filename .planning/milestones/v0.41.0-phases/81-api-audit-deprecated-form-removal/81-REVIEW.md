---
phase: 81-api-audit-deprecated-form-removal
reviewed: 2026-09-07T00:00:00Z
depth: deep
files_reviewed: 15
files_reviewed_list:
  - fdars-core/examples/21_function_on_scalar/main.rs
  - fdars-core/src/depth/fraiman_muniz.rs
  - fdars-core/src/depth/mod.rs
  - fdars-core/src/depth/modal.rs
  - fdars-core/src/depth/random_projection.rs
  - fdars-core/src/depth/random_tukey.rs
  - fdars-core/src/depth/tests.rs
  - fdars-core/src/fdata.rs
  - fdars-core/src/function_on_scalar.rs
  - fdars-core/src/inference/anova.rs
  - fdars-core/src/inference/permutation.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
  - fdars-core/tests/equivalence_phase50.rs
  - fdars-core/tests/validate_against_r.rs
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: findings
---

# Phase 81: Code Review Report

**Reviewed:** 2026-09-07
**Depth:** deep
**Files Reviewed:** 15
**Status:** issues_found (2 warnings, 1 info — all documentation drift; zero correctness/security defects)

## Summary

Phase 81 (plan 81-02) hard-removes 6 deprecated API forms — `mean_2d`, `fanova`,
`random_tukey_2d`, `random_projection_2d`, `fraiman_muniz_2d`, `modal_2d` — and migrates
their callers to the surviving replacements. This is a net-deletion mechanical refactor
(-320/+63). I independently verified all 5 declared correctness risks and found the core
migration to be correct:

1. **Seed preservation (highest risk) — CLEAN.** Every migrated `fanova(...)` caller now
   calls `fanova_seeded(..., 42)` with the literal `42` as the final `seed: u64` argument.
   Confirmed at all 8 production/test call sites across `function_on_scalar.rs`,
   `inference/anova.rs`, `inference/permutation.rs`, and the two integration-test files. The
   only non-42 seed (`equivalence_phase50.rs:84`, seed `7`) is the intentional
   "different-seed-changes-p-value" test, not a migration site. The `fanova_seeded` LCG body
   threads `seed` directly into `rng_state`, so `seed = 42` reproduces the legacy hand-rolled
   LCG stream bit-for-bit — the bit-identical golden test `fanova_seeded_seed42_bit_identical`
   remains and pins this.

2. **No dangling code references — CLEAN.** Grepped `src/`, `tests/`, `examples/`, `benches/`
   for all 6 removed symbols with word boundaries: zero code references remain. All surviving
   `fanova` occurrences are in prose comments/docstrings (see WR-01/IN-01). No intra-doc
   `[...]` links to bare `fanova` survive (all migrated to `fanova_seeded`), so rustdoc
   resolution is intact.

3. **Re-export removal — CLEAN.** The 6 forms are gone from `lib.rs`, `prelude.rs`, and
   `depth/mod.rs`. All 6 now-dead `#[allow(deprecated)]` guards were removed; a repo-wide grep
   confirms zero `#[allow(deprecated)]` remain in `src/tests/examples`. Zero `#[deprecated]`
   attributes remain on any function (the only match is a prose mention in `dim.rs` — see WR-01).

4. **Replacement-map / `Dim::Two` correctness — CLEAN.** Verified each replacement signature:
   `mean(data, dim)`, `modal(.., dim)`, `fraiman_muniz(.., dim)`, `random_projection(.., dim)`,
   `random_tukey(.., dim)` all take `dim: Dim` as the final argument, and every migrated call
   site passes `Dim::Two` in that position.

5. **Test-removal justification — CLEAN (spot-checked 3).** The removed
   `dispatch_modal_equals_2d`, `dispatch_fraiman_muniz_equals_2d`, and `dispatch_mean_equals_2d`
   asserted `unified(.., Dim::Two) == <name>_2d(..)`, where each `_2d` shim merely forwarded to
   its `_1d` primitive. The retained `dispatch_*_equals_1d` tests assert
   `unified(.., Dim::Two) == <name>_1d(..)` — transitively identical coverage. The removed
   `depth/tests.rs`, `fdata.rs`, and `validate_against_r.rs` delegation tests likewise only
   pinned `_2d == _1d` for a shim that no longer exists. No real behavior coverage was dropped.

The two warnings and one info item below are documentation drift only — stale prose references
to the now-removed `fanova` in doc comments. They do not affect compilation, rustdoc resolution,
or runtime behavior, but they mislead a reader of the source.

## Warnings

### WR-01: Stale `dim.rs` module doc claims the `_2d` shims are "retired via `#[deprecated]`" — they are now hard-removed

**File:** `fdars-core/src/dim.rs:4-9`
**Issue:** The module-level doc comment states: *"the redundant `_2d` shims are retired via
`#[deprecated]`."* As of this phase the shims are fully deleted, not merely deprecated. A
reader of `dim.rs` (the canonical explainer for the `Dim` dispatch design) will incorrectly
believe the `_2d` functions still exist behind a deprecation gate and try to call them. This
is the most authoritative doc surface for the `Dim` mechanism, so the drift is the highest-value
one to fix.
**Fix:** Update the sentence to reflect removal, e.g.:
```rust
//! dispatchers take an explicit [`Dim`] argument so callers get one ergonomic
//! `name(…, dim)` entry point. The redundant `_2d` shims were removed in v0.41.0
//! (API-01) after a deprecation cycle; `name(…, Dim::Two)` is the sole entry point.
```
Also revise the type-level doc at `dim.rs:15-16` ("because the `_2d` path never diverged") if
you want it to read in past tense, though that line is defensible as-is.

### WR-02: Stale prose reference to removed `function_on_scalar::fanova` in `permutation_test.rs` module doc

**File:** `fdars-core/src/permutation_test.rs:20`
**Issue:** The module doc lists *"the fixed-42 LCG site (`function_on_scalar::fanova`)"* among
sites documented-and-excluded from the `StdRng` permutation-helper migration. `fanova` no longer
exists; the fixed-42 LCG now lives in `fanova_seeded`. The reference is plain prose in backticks
(not a `[...]` intra-doc link), so it does not break `cargo doc`, but it points a reader at a
symbol that is no longer in the crate.
**Fix:** Rename the reference to the surviving function:
```rust
//! (`function_on_scalar::fanova_seeded` with the legacy seed 42) are documented-and-excluded
```

## Info

### IN-01: Residual `fanova` mentions in inference test comments/assert messages are acceptable but slightly ambiguous

**File:** `fdars-core/src/inference/anova.rs:242,246,267` (and similar in `inference/permutation.rs`)
**Issue:** Several test comments and `assert!` failure messages still say bare "fanova" (e.g.
`"fanova should also reject separated groups"`). These are informal prose describing the
seed-42 permutation ANOVA behavior — now provided by `fanova_seeded(.., 42)` — and the
migrated comments already clarify "legacy seed-42 `fanova`". No action strictly required; if a
consistency pass is cheap, consider "seeded fanova" in the assert-message strings so a failing
test does not name a function that no longer exists. Not a defect.

---

_Reviewed: 2026-09-07_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
