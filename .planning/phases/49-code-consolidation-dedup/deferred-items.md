# Phase 49 — Deferred (out-of-scope) items

## Pre-existing clippy warning (NOT introduced by plan 49-05)

- **File:** `fdars-core/src/parallel.rs:172` (test module)
- **Lint:** `clippy::useless_vec` — `let vec = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];` → suggests array.
- **Surfaces only under:** `cargo clippy --all-targets --no-default-features --features linalg -- -D warnings`
- **Status:** Pre-existing (confirmed by stashing all 49-05 changes — warning persists without them).
  Last commit to `parallel.rs` is `bc7baefa`, unrelated to Phase 49. The CI-representative gate
  (`--features linalg,parallel --all-targets`, per project MEMORY) is CLEAN.
- **Decision:** Out of scope for 49-05 (seeded-RNG gap-closure touches only 5 files, none is
  `parallel.rs`). Not fixed here to respect the executor scope boundary. Candidate for a future
  clippy-sweep cleanup.

## Permanent `__equivalence_test_support` public surface (code-review WR-01)

- **File:** `fdars-core/src/lib.rs` — `pub mod __equivalence_test_support` (13 `#[doc(hidden)]` `pub fn`
  forwarders that expose `pub(crate)` internals — `seed_for_thread`, `distributions::*`,
  `dominant_sign_negative`, etc. — to the external integration test crates in `tests/`).
- **Concern:** `#[doc(hidden)]` hides docs but does NOT restrict visibility, so these forwarders are
  technically reachable by downstream crates → grows the permanent public surface. Additive and
  milestone-compliant (v0.30.0 allows additive/non-breaking API), so NOT a blocker; flagged by the
  Phase-49 code review as WARNING WR-01.
- **Why not fixed now:** the clean fix is a non-default `_internal-test` feature gating the module,
  but that would force every `equivalence_phase*` integration test (and CI) to pass the feature, and
  omitting it would silently disable the golden tests in CI — real regression risk for a doc-only tidy.
- **Decision:** DEFER to the future breaking/1.0-readiness cleanup (APIB-01). At that point either
  feature-gate the module or replace the forwarders with in-crate `#[cfg(test)]` unit tests.
