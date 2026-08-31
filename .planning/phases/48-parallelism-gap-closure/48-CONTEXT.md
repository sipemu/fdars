# Phase 48: Parallelism-Gap Closure - Context

**Gathered:** 2026-08-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Close feature-gated rayon parallelism gaps in the newer v0.19–v0.29 subsystems (PERF-03): loops that
currently run only sequentially but are embarrassingly parallel gain multi-threaded speedups on large
inputs via the existing `parallel.rs` macros — with **bit-identical** results versus the sequential
path (tested with the `parallel` feature both ON and OFF), a **payback-threshold N guard** where a
small-input regression is possible, and **preserved per-thread RNG determinism**
(`StdRng::seed_from_u64(seed + k)`). No new crate dependency. Additive/behavior-preserving.

</domain>

<decisions>
## Implementation Decisions

### Target Selection & Granularity
- Parallelize the PROF-flagged **sequential permutation loops** — `t_perm_test` / `f_perm_test`
  (`src/inference/permutation.rs`), `frechet_anova` (`src/frechet/anova.rs`) — and the **`co_cluster`
  n_init loop** (`src/coclustering.rs`). These are high-cost, embarrassingly-parallel outer loops that
  PROF-01/02 flagged as sequential (parallel already exists in function_on_scalar/famm/explain_generic).
- Parallelize at the **outer independent-iteration level** (permutation replicas / random inits) using
  the existing `iter_maybe_parallel!` macro — not inner/nested loops.
- **Research confirms** which loops are genuinely sequential AND worth it (payback positive); marginal
  or tiny loops are **deferred with a documented note**, not force-parallelized.
- Reuse the Phase 47 harness: extend `benches/perf_hotpaths.rs` (or a sibling) for thread-scaling and
  add an equivalence test file.

### Equivalence & Determinism
- **Bit-identical** parallel-ON vs parallel-OFF output (SC2) — not merely within tolerance.
- Determinism holds via **per-iteration seeding**: replica/init `k` uses `StdRng::seed_from_u64(seed + k)`,
  so its RNG stream depends only on `k`, never on thread or execution order → result is independent of
  thread count. Any parallelized randomized loop MUST preserve this exact pattern (SC4).
- **Test both feature configs:** a committed equivalence/golden test asserts the seeded result; run the
  suite under both `--features linalg,parallel` and parallel-off (`--no-default-features` + needed
  features) to prove both branches produce the same output.
- Permanent golden regression guard for each parallelized fn's output.

### Payback Threshold & Benchmark Evidence
- **Outer-if payback-threshold guard**: `if work >= THRESHOLD { parallel } else { sequential }` where a
  small-input regression is possible (matching the v0.17.0 `SCORES_PARALLEL_THRESHOLD` precedent). The
  threshold is a documented `const` per parallelized fn, keyed on the real work size (e.g. `n_perm`,
  `n_perm * n`, or `n_init`).
- Derive the threshold from **criterion thread-scaling** measurement (1 vs N threads) — set conservatively.
- **Speedup evidence:** criterion thread-scaling cells (RAYON_NUM_THREADS 1 vs 20) showing large-input
  speedup; record before/after (+ governor caveat) in a `PERF-PARALLEL-RESULTS.md`.
- Register the thread-scaling benches **permanently** (`[[bench]]`) — they become Phase 51 BENCH-02 guards.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/parallel.rs` — 5 macros: `iter_maybe_parallel!`, `slice_maybe_parallel!`,
  `slice_maybe_parallel_mut!`, `maybe_par_chunks_mut!`, `maybe_par_chunks_mut_enumerate!`
  (feature-gate rayon: `parallel`-on uses `par_iter`, off falls back to sequential `iter`).
- Per-thread RNG seeding pattern `StdRng::seed_from_u64(seed + k as u64)` used crate-wide (10+ sites).
- v0.17.0 `SCORES_PARALLEL_THRESHOLD` — the payback-threshold precedent to mirror.
- Phase 47 `benches/perf_hotpaths.rs` + `tests/equivalence_phase47.rs` — patterns for permanent
  bench + golden-equivalence tests.
- PROF-01 (Phase 46): `frechet_anova` 133ms and `co_cluster` 13ms are the named parallelism candidates;
  PROF-02 lists the 3 sequential permutation sites (inference/permutation.rs, frechet/anova.rs, explain/importance.rs).

### Established Patterns
- Parallel-first, sequential-compatible: the macros keep both paths equivalent by construction.
- Deterministic seeding is the linchpin — it makes parallel output order-independent.

### Integration Points
- Consumes PROF-01/02 parallelism candidates (Phase 46). Feeds Phase 51 (BENCH-02 guards the thread-scaling benches).

</code_context>

<specifics>
## Specific Ideas

- Honor MEMORY.md operational pointers: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`; free
  `target/debug/{incremental,examples}` before bench builds; full clippy gate
  `cargo clippy --all-targets --features linalg,parallel -- -D warnings`; commit `--no-verify` +
  `cargo fmt -p fdars-core` per commit; capture governor/RAYON_NUM_THREADS (powersave LOW-CONFIDENCE).
- Thread-scaling benches should vary `RAYON_NUM_THREADS` (or a rayon thread-pool) to show 1-vs-N speedup honestly.
- Behavior-changing phase: suite must stay green at every commit; parallel-OFF build must also compile + pass.

</specifics>

<deferred>
## Deferred Ideas

- Loops already parallel (function_on_scalar, famm, explain_generic) — no work.
- Marginal/tiny sequential loops where payback is negative → documented defer, not parallelized.
- Formalizing the thread-scaling benches as documented regression guards → Phase 51 (BENCH-02).
- Any parallelization that would change numeric output (non-deterministic reduction order) → out of scope (must stay bit-identical).

</deferred>
