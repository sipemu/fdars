---
phase: 79-serde-feature-repair
plan: "01"
status: complete
provides:
  - BUILD-01
key-files:
  - fdars-core/src/classification/fit.rs
  - fdars-core/src/classification/mod.rs
  - fdars-core/src/tolerance/types.rs
  - fdars-core/src/elastic_fpca.rs
  - fdars-core/tests/serde_feature_roundtrip.rs
completed: 2026-09-07
---

## Accomplishments

Restored `cargo build --features serde` (RED since Phase 60, 12 trait-bound errors) to GREEN. Added `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` to all five crate-owned types that were missing it. No new dependency, no API change, no behavior change. Both build configs (`--features serde` and `--features linalg,parallel,serde`) compile cleanly.

## Types that received serde derives

All five named types in the plan were needed. No cascade beyond these five was surfaced by the build:

| Type | File | Notes |
|------|------|-------|
| `ClassifFit` | `fdars-core/src/classification/fit.rs:47` | Primary offender (Phase 60 ShapeletTransformClassifier embeds it) |
| `ClassifMethod` | `fdars-core/src/classification/fit.rs:19` | Embedded by `ClassifFit`; needed for cascade |
| `ClassifResult` | `fdars-core/src/classification/mod.rs:33` | Embedded by `ClassifFit`; needed for cascade |
| `NonConformityScore` | `fdars-core/src/tolerance/types.rs:26` | Fieldless C-like enum; no cascade |
| `JointFpcaResult` | `fdars-core/src/elastic_fpca.rs:72` | Embeds only `FdMatrix` (already serde) and std types; no cascade |

No `#[serde(skip)]` shim was needed. All embedded field types were either already serde-enabled or are std types.

## Round-trip test

- **File:** `fdars-core/tests/serde_feature_roundtrip.rs`
- **Test fn:** `serde_roundtrip::classiffit_serde_roundtrip`
- **Gated by:** `#[cfg(feature = "serde")]`
- **Approach:** Constructs a `ClassifFit` via `fclassif_lda_fit` on a 20-obs 2-class deterministic dataset, serializes to JSON via `serde_json::to_string`, deserializes back, then asserts all integer/structural fields are exactly equal and all f64 fields are within 1e-12 (matching the `src/spm/phase.rs:777` convention — JSON f64 serialization truncates at ~15-16 significant digits, so strict PartialEq fails at the last ULP).

## Re-breakage guard

`rust-ci.yml:55` runs `cargo test --features linalg,parallel,serde` on every push to main and every PR. This step was RED on serde since Phase 60 (ea39c623 added `ShapeletTransformClassifier` embedding non-serde `ClassifFit`). It is now GREEN. The new test file `serde_feature_roundtrip.rs` is compiled as part of this step — a future missing-derive gap will fail `classiffit_serde_roundtrip` before it can merge.

No explicit serde-only `cargo build --features serde` CI step was added — the existing `cargo test --features linalg,parallel,serde` (which compiles everything) is sufficient as the durable guard.

## Task Commits

| Task | Commit | Message |
|------|--------|---------|
| Task 1+2 (derives) | `440eb2cc` | `fix(79-01): add serde derives to ClassifFit, ClassifMethod, ClassifResult, NonConformityScore, JointFpcaResult` |
| Task 1 (round-trip test) | `02f4d652` | `test(79-01): add serde round-trip regression test for ClassifFit` |

## Files Modified

- `fdars-core/src/classification/fit.rs` — added `cfg_attr` serde derive on `ClassifMethod` (line 20) and `ClassifFit` (line 48)
- `fdars-core/src/classification/mod.rs` — added `cfg_attr` serde derive on `ClassifResult` (line 34)
- `fdars-core/src/tolerance/types.rs` — added `cfg_attr` serde derive on `NonConformityScore` (line 27)
- `fdars-core/src/elastic_fpca.rs` — added `cfg_attr` serde derive on `JointFpcaResult` (line 73)
- `fdars-core/tests/serde_feature_roundtrip.rs` — new file: serde-gated round-trip integration test

## Verification

All gates passed:

```
# Gate 1: minimal serde build
cargo build -p fdars-core --features serde
→ Finished dev profile in 19.20s  [PASS]

# Gate 2: CI feature-set serde build
cargo build -p fdars-core --features linalg,parallel,serde
→ Finished dev profile in 18.86s  [PASS]

# Gate 3: round-trip test
cargo test -p fdars-core --features linalg,parallel,serde --test serde_feature_roundtrip
→ test serde_roundtrip::classiffit_serde_roundtrip ... ok
→ test result: ok. 1 passed; 0 failed  [PASS]

# Gate 4: fmt check
cargo fmt --manifest-path fdars-core/Cargo.toml -- --check
→ (no output, clean)  [PASS]

# Gate 5: clippy (CI allow set)
cargo clippy --all-targets --features linalg,parallel,serde -- -D warnings \
  -A clippy::too_many_arguments -A clippy::useless_vec -A clippy::type_complexity \
  -A clippy::manual_memcpy -A clippy::wildcard_in_or_patterns
→ Finished dev profile in 17.69s  [PASS]

# Gate 6: full serde test matrix
cargo test -p fdars-core --features linalg,parallel,serde
→ test result: ok. 2862 passed; 0 failed; 0 ignored  [PASS]
  (plus 209 doc tests: ok. 209 passed; 0 failed; 4 ignored)
```
