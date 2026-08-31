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
