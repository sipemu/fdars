# Phase 47: Hot-Path & Allocation Performance - Context

**Gathered:** 2026-08-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Optimize the top-ranked compute-bound hot paths and allocation hotspots from PROF-01
(`.planning/phases/46-whole-crate-profiling-measurement/PROF-01-hotpath-targets.md`)
**behavior-preservingly**, proven by before/after criterion benchmarks + allocation profiles +
equivalence tests. Covers **PERF-01** (hot-path compute) and **PERF-02** (allocation). Parallelism
gaps (PERF-03) are Phase 48; permanent regression-guard formalization (BENCH-02) is Phase 51.
Numeric outputs must stay identical or provably-equivalent within documented tolerance; no public
signature changes; no new crate dependency.

</domain>

<decisions>
## Implementation Decisions

### Optimization Scope & Target Selection
- Optimize the highest-leverage subset with clear safe wins, not all top-10:
  - **`fts::dpca`** — allocation hotspot #1 (42 MB total / 8.6 MB peak / 17 739 blocks @ n200_m50,
    `src/fts/spectral.rs:203`). PERF-02 primary.
  - **`irreg_fdata::face_covariance`** — compute hotspot #1 (984 ms @ n200_m30, `src/irreg_fdata/face.rs:128`).
  - **`fem_smoothing::fem_smooth`** — compute #2 (452 ms @ 576 nodes, `src/fem_smoothing.rs:475`).
  - **Opportunistic** FdMatrix↔DMatrix copy removals surfaced by PROF-01 (fsvd `:488`, ssvd `:740`,
    long_run_covariance `acf.rs:337`) where mechanical and safe.
- "Measurable improvement" bar: **≥15% wall-time** (non-overlapping criterion CIs) OR **≥25%
  allocation reduction**.
- A target with **no safe behavior-preserving win** is **documented + deferred** — do not force a
  risky rewrite. (face_covariance / fem_smooth are inherently O(n·m²) / superlinear; target
  constant-factor wins — redundant allocation/recompute removal, row-method reuse — not asymptotic.)
- Attack order: **allocation-reduction first** (mechanical, low-risk), then compute paths
  (algorithmic, higher-risk).

### Behavior-Preservation & Equivalence Testing
- Tolerance: **exact** for counting/integer paths; **relative ≤ 1e-10** for float SVD/eigen paths,
  documented per change.
- Add **permanent `#[test]` equivalence/golden tests** capturing old→new output so future changes are
  guarded (not temporary checks).
- The **existing full suite must stay green at every commit**, in addition to the new equivalence tests.
- PERF-02 allocation proof: **re-add a committed feature-gated `dhat-heap` alloc-audit test** (mirror
  `tests/alloc_audit_fpca.rs`) showing before→after fewer/smaller allocations.

### Benchmark Evidence & BENCH-02 Overlap
- **Register the PERF-proof benches permanently now** (`[[bench]]`) for the optimized paths — they
  become BENCH-02's regression guards; Phase 51 documents/formalizes them (avoids re-authoring).
- Record before/after numbers in the phase SUMMARY **+ a `PERF-RESULTS.md`** (folded into BENCH-02).
- Capture governor + `RAYON_NUM_THREADS` for every before/after; **pin the `performance` governor if
  permitted**, else note the `powersave` LOW-CONFIDENCE caveat (v0.14.0).
- **No public signature changes**; keep `linalg`/non-`linalg` branches producing equivalent results (SC3).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- PROF-01 inventory (Phase 46) — ranked targets with file:line anchors + before numbers to beat.
- `tests/alloc_audit_fpca.rs` — the committed dhat-heap allocation-test pattern to copy for PERF-02.
- `benches/audit_hotpaths.rs` — the criterion bench pattern for before/after proof.
- `FdMatrix` row methods (`row_to_buf`, `row_dot`, `row_l2_sq`) — avoid materializing DMatrix copies.
- `parallel.rs` macros exist but PERF-03 parallelism is Phase 48 — this phase is single-thread compute + allocation.

### Established Patterns
- Column-major `FdMatrix`; `Result<T, FdarError>`; `#[cfg(feature = "linalg")]` gating; SVD via nalgebra.
- Allocation reduction = remove needless `to_dmatrix()` / `DMatrix::from_column_slice` copies and
  per-iteration `Vec`/matrix allocations in hot loops (reuse buffers).

### Integration Points
- Consumes: PROF-01 (Phase 46). Feeds: Phase 48 (shares perf harness), Phase 51 (BENCH-02 guards these benches).

</code_context>

<specifics>
## Specific Ideas

- fts::dpca (42MB churn) is the clearest PERF-02 win — investigate `spectral_density` temporary
  `DMatrix` churn at `src/fts/spectral.rs:203` first (mechanical, low-risk).
- Honor MEMORY.md operational pointers: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`; free
  `target/debug/{incremental,examples}` before bench builds; full clippy gate
  `cargo clippy --all-targets --features linalg,parallel -- -D warnings`; commit with `--no-verify`
  (long-cargo hook) then run `cargo fmt` per commit to avoid CI fmt drift.
- Behavior-changing phase (unlike 46) — every commit must keep the suite green.

</specifics>

<deferred>
## Deferred Ideas

- Parallelism (feature-gated rayon) for these hot paths → Phase 48 (PERF-03).
- Documenting the benches as formal regression guards with a before/after table → Phase 51 (BENCH-02).
- Any target with no safe behavior-preserving win → deferred with a documented rationale.
- Breaking/asymptotic rewrites of inherently O(n·m²) paths → out of scope (behavior-preserving only).

</deferred>
