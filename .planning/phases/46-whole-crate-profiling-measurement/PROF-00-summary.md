# PROF-00 — Profiling & Measurement Summary Index

**Phase 46 (Whole-Crate Profiling & Measurement) · v0.30.0 · 2026-08-30**

This is the one-page index to the three ranked inventories produced this phase. Downstream planners
(Phases 47/49/50/51) read this first, then jump to the inventory they consume. Measure-only: zero
`fdars-core/src/` edits, no permanent `[[bench]]` registered, no new crate dependency.

---

## The Three Inventories

| Inventory | File | Requirement | Consumer phase(s) | Top-ranked target |
|-----------|------|-------------|-------------------|-------------------|
| Hot-path optimization targets | [`PROF-01-hotpath-targets.md`](./PROF-01-hotpath-targets.md) | PROF-01 | **Phase 47** (PERF) + **Phase 51** (BENCH-01 module list) | `irreg_fdata::face_covariance` — 984 ms @ n200_m30 |
| Duplication / consolidation | [`PROF-02-dedup-inventory.md`](./PROF-02-dedup-inventory.md) | PROF-02 | **Phase 49** (CONS) | χ²/F survival — 2 independent gamma kernels |
| API inconsistency | [`PROF-03-api-inventory.md`](./PROF-03-api-inventory.md) | PROF-03 | **Phase 50** (API) | 4 Config structs missing `Default` |

---

## Headline Findings

**PROF-01 (hot-path) → Phase 47.** The two dominant compute-bound paths are
`irreg_fdata::face_covariance` (**984 ms** @ n200_m30, `src/irreg_fdata/face.rs:128`, ~O(n·m²)) and
`fem_smoothing::fem_smooth` (**452 ms** @ 576 nodes, `src/fem_smoothing.rs:475`, superlinear FEM
assembly+solve). The dominant **allocation** hotspot is `fts::dpca` — **42 MB total / 8.6 MB peak /
17 739 blocks** at n200_m50 (`src/fts/spectral.rs:203`), ~70× the next-largest. Sequential
permutation loops in `frechet_anova` / `t_perm_test` and `co_cluster` inits are PERF-03 parallelism
candidates.

**PROF-02 (dedup) → Phase 49.** Highest-leverage target is the **χ²/F survival machinery**: two
independent regularized-incomplete-gamma kernels (`src/inference/dist.rs:99` vs
`src/spm/chi_squared.rs:164`) → factor into a shared `src/distributions.rs` (CONS-01). Then
permutation-loop scaffolding (6 sites, 3 sequential / 3 parallel, CONS-02), per-thread seeded-RNG (10
thread-offset sites, CONS-02), and the SVD sign-fix mirror (`src/regression.rs:180` vs
`src/pace_fpca.rs:219`, CONS-01, correctness-critical). `simpsons_weights`, Cholesky, and FPCA scoring
are already consolidated — explicitly out of scope.

**PROF-03 (API) → Phase 50.** Top target is the **4 Config structs missing `Default`**
(`BoostingConfig`/`BayesianConfig`/`StabilityConfig` in `src/boosting_regression/mod.rs:44/76/103`,
`StlConfig` in `src/detrend/stl.rs:49`) — 52/56 already have it. Then a reproducibility gap:
`fanova` (`src/function_on_scalar.rs:791`) lacks a `seed` param while every sibling permutation test
has one → additive `fanova_seeded`. Field renames and bulk `_1d`/`_2d` unification are **breaking**
(deferred to APIB-01); the genuinely-different `_nd` algorithms must not be deprecated. Every in-scope
item is additive (add + `#[deprecated]`) so the 28 examples + R/WASM bindings keep compiling.

---

## Measurement Environment Caveat

All PROF-01 timings were taken with the CPU governor at **`powersave`** (unpinned) on a 20-core host
with `RAYON_NUM_THREADS` defaulting to 20. This is a **LOW-CONFIDENCE** setting for multi-thread cells
(v0.14.0 audit caveat): absolute numbers inflate and de-stabilize. Phase 47 before/after comparisons
MUST re-capture the environment and should pin the governor to `performance` for honest deltas. Use
the rankings (relative order) with more confidence than the absolute millisecond figures.

---

## Scope Confirmation

- **Zero `fdars-core/src/` edits** across the whole phase (every task asserted `git status --porcelain fdars-core/src/` empty).
- **No permanent `[[bench]]` registered** — all 9 throwaway probe benches + the dhat probe file were removed; Cargo.toml is back to its 10 pre-phase benches. Permanent bench coverage is Phase 51 (BENCH-01).
- **No new crate dependency** — profiling used only existing dev-deps (criterion 0.5, dhat 0.3 behind `dhat-heap`).
- **Full suite green** before and after (zero behavior change).
