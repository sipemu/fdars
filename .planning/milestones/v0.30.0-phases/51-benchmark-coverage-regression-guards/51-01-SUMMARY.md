---
phase: 51-benchmark-coverage-regression-guards
plan: 01
subsystem: inference
tags: [benchmark, criterion, tracer, bench-coverage, BENCH-01]
requires:
  - phase: 47
    provides: "benches/perf_hotpaths.rs — the criterion structure + generate_curves generator this bench mirrors"
provides:
  - "fdars-core/benches/inference_benchmarks.rs — criterion bench for inference::t_perm_test (na30_nb30_m50_nperm999 cell)"
  - "fdars-core/Cargo.toml — [[bench]] name=\"inference_benchmarks\" harness=false"
  - "The full add-bench pipeline proven end-to-end (Phase-51 tracer): create bench → Cargo register → build --benches → clippy --all-targets → fmt → commit --no-verify"
affects: [51-02, 51-03]
actuals:
  tokens: 5500
  tasks: 1
  commits: 2
tech-stack:
  added: []
  patterns:
    - "each bench file is a separate compilation unit — no shared helper module, so the deterministic non-RNG generate_curves generator is copied verbatim into each bench file"
    - "criterion pattern: build data OUTSIDE b.iter(); black_box on every input AND the returned result; group.sample_size/measurement_time/warm_up_time tuning per cost class"
key-files:
  created:
    - fdars-core/benches/inference_benchmarks.rs
  modified:
    - fdars-core/Cargo.toml
key-decisions:
  - "Import path is `fdars_core::inference::{t_perm_test, DEFAULT_N_PERM}` — VERIFIED at src/inference/mod.rs:41 (re-export of permutation::{...}). Signature `t_perm_test(&FdMatrix, &FdMatrix, &[f64], usize, u64) -> Result<TestResult, FdarError>` VERIFIED at src/inference/permutation.rs:152."
  - "Cell size kept at the plan's medium estimate (na=nb=30, m=50, DEFAULT_N_PERM=999): the --quick run measured ~2.196 ms median — well within budget, so NO down-sizing was needed (RESEARCH A1 executor-tunability not exercised)."
  - "New [[bench]] block placed adjacent to the two PERMANENT perf_* blocks; their comments left intact. No src/ edit, no new dependency (criterion 0.5 is a pre-existing dev-dep)."
requirements-completed: [BENCH-01]
coverage:
  - id: D1
    description: "benches/inference_benchmarks.rs benchmarks inference::t_perm_test; registered as [[bench]] harness=false; compiles under --benches and lints clean under clippy --all-targets"
    requirement: BENCH-01
    verification:
      - kind: integration
        ref: "cargo build -p fdars-core --benches --features linalg,parallel => Finished (all benches incl. new one compiled)"
        status: pass
      - kind: lint
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings => clean (primary gate, lints bench code)"
        status: pass
      - kind: integration
        ref: "cargo bench --bench inference_benchmarks -- --quick => ran t_perm_test once, ~2.196 ms median baseline captured"
        status: pass
    human_judgment: false
status: complete
---

# Phase 51 Plan 01: Inference Benchmarks (BENCH-01 Tracer) Summary

Added the first new criterion bench of Phase 51 — `benches/inference_benchmarks.rs` covering
`inference::t_perm_test` — and proved the full add-bench pipeline end-to-end. `t_perm_test` is the
cheapest, most self-contained target (two sinusoid `FdMatrix` from the existing generator, no
feature gate, no generic bound, no multi-input assembly), so it exercises the exact create → Cargo
register → `build --benches` → `clippy --all-targets` → `fmt` → `commit --no-verify` machinery that
the remaining 8 module benches (plans 51-02, 51-03) will repeat, at minimum risk.

## What shipped

- **`benches/inference_benchmarks.rs`** — one criterion bench group `inference_t_perm_test` with a
  single cell `na30_nb30_m50_nperm999`:
  - Imports `fdars_core::inference::{t_perm_test, DEFAULT_N_PERM}` and `fdars_core::matrix::FdMatrix`.
  - Copies the deterministic non-RNG `generate_curves(n, m) -> (FdMatrix, Vec<f64>)` sinusoid
    generator verbatim from `perf_hotpaths.rs` (bench files are separate compilation units).
  - Builds both samples OUTSIDE `b.iter()`: `(a, argvals) = generate_curves(30, 50)` and
    `(b_data, _) = generate_curves(30, 50)` (second sample shares argvals).
  - `black_box` on all five inputs and on the returned `TestResult`.
  - `group.sample_size(20); group.measurement_time(30s); group.warm_up_time(3s)` (medium cost class).
  - `criterion_group!(benches, bench_t_perm_test); criterion_main!(benches);`.
- **`Cargo.toml`** — new `[[bench]] name = "inference_benchmarks" harness = false` placed adjacent to
  the two PERMANENT `perf_hotpaths` / `perf_parallelism` blocks, their comments untouched.
- **No src/ edit; no new dependency.**

## Baseline captured (for BENCH-RESULTS.md)

| Bench cell | Sample | Median (--quick) |
|------------|--------|------------------|
| `inference_t_perm_test/na30_nb30_m50_nperm999` | na=nb=30, m=50, n_perm=999 (DEFAULT_N_PERM), seed=42 | **~2.196 ms** (range 2.1937–2.2055 ms) |

The medium cost estimate ran short (~2.2 ms per iter), so NO cell down-sizing was applied
(RESEARCH A1 executor-tunability was available but not needed).

## Commit count

2 atomic commits:
- `a0f6999b` feat(51-01): add inference_benchmarks bench (t_perm_test) — BENCH-01 tracer
- (this SUMMARY commit)

## Gate results

| Gate | Result |
|------|--------|
| `cargo build -p fdars-core --benches --features linalg,parallel` | clean — all benches incl. new one compiled (7.88s) |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | clean (primary gate — lints bench code) |
| `cargo bench --bench inference_benchmarks -- --quick` | ran t_perm_test once, ~2.196 ms median captured |
| `cargo fmt -p fdars-core` | clean (no reformatting of the new file) |
| commit | `git commit --no-verify` (avoids the slow full hook that stalls executor watchdogs) |

Full create → register → build → clippy → fmt → commit pipeline is proven green. The tracer is
complete and plans 51-02 / 51-03 can build on it.

## Tracer feedback gate

`type="tracer"` — the tracer feedback gate re-ran the `<verify>` chain end-to-end after the commit
(build ✓, clippy ✓, --quick bench ✓, all passing). Verified end-to-end; no expansion tasks exist in
this plan.

## Deviations from Plan

None — plan executed exactly as written. The `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` prefix
was applied to every cargo command and a pre-emptive `rm -rf target/debug/{incremental,examples}`
was run before the build; no link/disk failure occurred, so no retry was needed. No `v*` tag was
created (audit milestone — a tag would trigger a phantom crates.io publish; crate version stays
0.29.0).

## Known Stubs

None. The bench calls the real `t_perm_test` with concrete deterministic inputs and asserts nothing
placeholder; there are no TODOs, empty returns, or mock data.

## Threat Flags

None. Bench inputs are hard-coded deterministic generators — no external/untrusted input, no
network, no auth/crypto (RESEARCH Security Domain: attack surface nil). Threat T-51-02 (phantom
crates.io publish from a `v*` tag) mitigated exactly as the register prescribed: no `v*` tag pushed,
crate version unchanged at 0.29.0. T-51-01 (dhat allocator) and T-51-SC (package install) untouched
— no allocator change, no dependency added.

## Self-Check: PASSED

- `fdars-core/benches/inference_benchmarks.rs` — FOUND
- `fdars-core/Cargo.toml` `[[bench]] name = "inference_benchmarks"` — FOUND
- Commit `a0f6999b` — FOUND
