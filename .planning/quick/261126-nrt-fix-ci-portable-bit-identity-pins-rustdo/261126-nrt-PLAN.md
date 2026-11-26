---
quick_id: 261126-nrt
status: complete
---
# Plan: fix "Rust CI" — rustdoc errors + portable bit-identity pins

**Goal:** CI green without weakening what the tests guard.

**Problem 1 (Documentation job):** `RUSTDOCFLAGS=-Dwarnings cargo doc --no-deps --features linalg,parallel,serde`
fails on intra-doc links from the #73 merge: `metric/dba.rs` (`dtw_path`, `dtw_distance_windowed` not in scope)
and `metric/elastic_family/mod.rs:776` (`[module docs](self)` expands inside private submodules).
Fix: path-qualified links only (no allow attributes, no visibility changes).

**Problem 2 (4 tests fail on CI only):** literal FNV bit-hash pins / exact seed equality that depend on
last-ulp results of the platform libm (glibc IFUNC-dispatched transcendental kernels).
- Diagnose: same binary under Ubuntu 24.04 / 26.04 glibc, CI-resolved dependency versions (Cargo.lock is
  gitignored), gdb on all runtime-dispatch crates, `GLIBC_TUNABLES` hwcaps masking.
- Fix: replace literal cross-machine hashes with (a) in-process bit-identity against an explicit classic /
  default configuration where one exists, (b) stored per-field value summaries compared at rtol 1e-10.
  Closed Karcher seeds: the ellipse is centrally symmetric (seed s and s + (m-1)/2 are an exact tie), so
  a seed mismatch is accepted only when proven to be that exact symmetry; mean/distances stay at 1e-8.
- Scan: every other pin that flips under a different libm variant (full lib suite with SSE2 libm
  variants); convert the pins, leave non-pin behaviour tests untouched (report them).

**Gates:** cargo fmt --all; clippy --all-targets -D warnings (linalg,parallel,serde); converted tests pass
with FMA libm *and* with SSE2 libm (GLIBC_TUNABLES) *and* in ubuntu:26.04; exact CI doc command exits 0;
`cargo test -p fdars-core --features linalg,parallel,serde --lib alignment::` passes.
