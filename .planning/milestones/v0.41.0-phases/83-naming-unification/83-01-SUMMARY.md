---
type: summary
phase: 83
plan: 01
requirements: [API-04]
status: complete
commit: ce5c3a6e
---

# 83-01 SUMMARY — Simple hard renames (AUD-14/15/16)

## What was done

Three hard renames (no deprecated shims, still 0.x):

- **AUD-14** — fn `funhddC_cluster` → `fun_hddc_cluster` (fix camelCase-in-snake_case).
  Migrated: def (`src/gmm/subspace.rs:554`), module-doc intra-doc link (`:21`),
  doctest `use` line (`:546`), 7 in-crate test callers, barrel re-export
  (`src/gmm/mod.rs:81`), crate-root re-export (`src/lib.rs`). The `operation:
  "funhddC_cluster"` error-string literal was intentionally left as-is (not an API
  symbol; explicitly out of scope per plan).
- **AUD-15** — type `FosrResult2d` → `Fosr2dResult` (adjective-before-noun house
  style). Whole-word rename across `src/function_on_scalar_2d.rs` (struct def, impl
  block, return type, constructor, doc mention, fn param) + crate-root re-export.
- **AUD-16** — type `GmmResult` → `GmmFitResult` (disambiguate single-K fit result
  from `GmmClusterResult`). Whole-word rename across `src/gmm/em.rs`,
  `src/gmm/cluster.rs` (imports, returns, `Option<..>`, `&..`, doc) + `GmmClusterResult.best`
  field type + crate-root re-export. `GmmClusterResult` untouched (verified).

## New public symbols

None (pure renames). No new types/enums.

## Gate results (whole-crate, from workspace root)

- `cargo fmt --check` — clean (exit 0)
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean, zero warnings
- `cargo test` (full) — lib: `test result: ok. 2850 passed; 0 failed`.
  The only failing binary is the PRE-EXISTING env golden flake
  `golden_co_cluster_parallel` / `golden_co_cluster_below_threshold`
  (tests/equivalence_phase48.rs). Confirmed pre-existing by stashing all
  `fdars-core/src` changes and re-running on the clean checkout — still FAILED, so it
  is machine/BLAS-dependent golden drift, NOT a rename regression (touches
  coclustering, none of the renamed symbols).
- `cargo build --features serde` — green (exit 0; the historic shapelet/ClassifFit
  serde break no longer reproduces)
- `cargo build --examples` — all 28 compile (exit 0)
- Grep gate — no API/re-export/intra-doc reference to `funhddC_cluster`,
  `FosrResult2d`, or bare `GmmResult` remains in src/examples/tests (only the allowed
  `operation:` error string).

## Files changed

- `fdars-core/src/gmm/subspace.rs`
- `fdars-core/src/gmm/mod.rs`
- `fdars-core/src/gmm/em.rs`
- `fdars-core/src/gmm/cluster.rs`
- `fdars-core/src/function_on_scalar_2d.rs`
- `fdars-core/src/lib.rs`

## Commit

`ce5c3a6e` — refactor(83-01): hard-rename funhddC_cluster, FosrResult2d, GmmResult (API-04)
