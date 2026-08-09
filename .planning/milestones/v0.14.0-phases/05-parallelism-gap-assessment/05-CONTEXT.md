# Phase 5: Parallelism Gap Assessment - Context

**Gathered:** 2026-08-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Measure how well `fdars` uses available cores and identify the safe, high-leverage sequential loops worth parallelizing. Delivers **analysis artifacts only** — a report section + backlog slice; **no changes to `fdars-core/src`** (audit-only milestone).

Concretely, this phase produces, appended to the single `.planning/research/AUDIT-REPORT.md`:
1. A **rayon thread-scaling table** (`RAYON_NUM_THREADS` sweep 1/2/4/8) for representative already-parallel hot paths, including the **payback-threshold N** at which parallel overhead is recovered (SC1).
2. A list of **sequential loops confirmed safe to parallelize** with evidence and the thread-safe RNG-seeding note where relevant (SC2).
3. A record of **where parallelism/banding is opt-in rather than automatic** and the **measured cost of the default unaccelerated path** (SC3).
4. **Backlog entries** (parallelization opportunities) with function / current-cost / root-cause fields (SC4).

**Not in scope:** editing `fdars-core/src` to actually parallelize any loop (deferred to a future implementation milestone); re-running the elastic or FPCA deep sweeps owned by Phases 3–4; the nalgebra-vs-faer SVD comparison (Phase 6).

</domain>

<decisions>
## Implementation Decisions

### Thread-Scaling Sweep (SC1)
- **D-01:** Sweep `RAYON_NUM_THREADS` ∈ {1, 2, 4, 8} over **two representative already-parallel targets chosen to bracket the crossover**: `karcher_mean` (compute-heavy, seconds-scale → clean speedup curve and the large-N regime) and `StreamingFraimanMuniz::depth_batch` (lightweight, sub-millisecond → the overhead-dominated small-N regime). Rationale: a single heavy target shows scaling but never exposes where rayon overhead stops paying off; pairing it with a light target lets the table show the payback-threshold N that SC1 requires. — **Reversibility:** reversible — bench-only target selection, no shipped-code impact.
- **D-02:** **Payback-threshold-N method.** For each sweep target, hold threads at the machine default and sweep N downward, comparing against a `RAYON_NUM_THREADS=1` (single-thread rayon) run of the same build; report the smallest N at which the parallel path first beats the single-thread path. Use `RAYON_NUM_THREADS=1` rather than `--no-default-features` for the crossover baseline so the *only* variable is thread count (same codegen, same feature set). The `--no-default-features` rayon-off cost is captured separately under SC3 (D-06).
- **D-03:** `pairwise` / `nadaraya_watson` are **not** additional sweep targets — noted as covered by the Phase-2 static inventory (ALREADY PARALLEL) and represented by the two chosen sentinels. Keeps the sweep matrix small enough to run under the stricter stability controls (D-04).

### Measurement Stability (methodology, Phase-5-specific)
- **D-04:** **Escalate machine-stability controls for the sweep** beyond the Phase-1 ±5%/2-run discipline: pin worker cores (`taskset`), set a fixed CPU governor (`cpupower`/`performance`), and run **3 independent runs per sweep cell** reporting the median and the run spread. Rationale: Phase 3 measured 34–58% two-run variance on `karcher_mean` and Phase 2 explicitly recommended re-measuring under `taskset`/`cpupower`; a thread-scaling curve is indistinguishable from scheduler noise without this. Document the exact controls applied (core set, governor, thread pinning) in the report methodology note. — **Reversibility:** costly — the stability protocol is the confidence basis for the whole SC1 table; loosening it later would require re-running the sweep to keep the numbers comparable.
- **D-05:** Retain all other Phase-1 discipline unchanged: `black_box` on inputs+outputs, primary build `linalg,parallel`, artifact naming `p5_<target>_<features>_run<N>.txt` under `.planning/research/bench/`, single-file append to `AUDIT-REPORT.md` (Phase-1 D-05/D-06).

### Sequential-Gap Evidence (SC2, audit-only)
- **D-06 (SC2 evidence standard):** "Confirmed safe to parallelize with evidence" = a **static safety argument + projected speedup**, with **no `fdars-core/src` edits**. For each candidate loop, document: (a) independence / absence of shared mutable state across iterations, (b) the thread-safe per-thread RNG-seeding note where an RNG is involved (`StdRng::seed_from_u64(seed + k)` pattern; note explicitly where no RNG is in the loop body, e.g. CV fold assignment happens once before the loop), and (c) a **projected speedup** extrapolated from the *measured* scaling of that loop's already-parallel analogue (e.g. project CV-fold and elastic-FPCA N-loops from `karcher_mean`'s `iter_maybe_parallel!` scaling in the SC1 table). Rationale: the milestone is audit-only; prototyping in `src` — even on a scratch branch — exceeds scope and risks the "no src changes" constraint. — **Reversibility:** reversible — the projection method is a report convention; a future implementation phase can measure real speedups when it actually wraps the loops.
- **D-07 (SC2 candidate set):** The candidates are exactly the Phase-2 SEQUENTIAL gap-list entries, no re-derivation: `classification/cv.rs:76` (`fclassif_cv` fold loop — independent folds, fold-assignment RNG outside the loop body), `elastic_fpca.rs:701` (`shooting_vectors_from_psis`), `elastic_fpca.rs:720` (`build_augmented_srsfs`), `elastic_fpca.rs:764` (`svd_scores_and_eigenvalues`) (the three elastic-FPCA inner N-loops), and `regression.rs:167` (`center_columns` inside `fdata_to_pc_1d` — distinct from the already-parallel `fdata.rs:center_1d`). Cite the Phase-2 AUDIT-REPORT "Parallelism Gap List" as the source.

### Unaccelerated-Path Cost (SC3)
- **D-08:** Report **both "acceleration off by default" dimensions**, citing existing artifacts rather than re-measuring: (a) **rayon-off cost** via Phase-1's `karcher_mean` feature-combo data (`""` ≈ 1555 ms vs `parallel` ≈ 162 ms → ~10×; artifacts `p1_karcher_none_run1.txt` / `p1_karcher_parallel_run1.txt`), and (b) **banding opt-in cost** via Phase-3's `karcher_mean` full-DP-vs-`karcher_mean_banded` (~7× at `band_frac=0.1`), since `karcher_mean()` defaults to `band_frac=0.0` (Phase-3 D-05, Anti-Pattern 2). Rationale: both numbers already exist at high enough confidence for a "cost of the default path" statement; re-measuring would add bench time and risk cross-phase inconsistency. Note the LOW-CONFIDENCE caveat on the raw Phase-3 karcher variance when citing (b). — **Reversibility:** reversible — a citation choice; a later phase can re-measure if a specific claim needs tightening.

### Claude's Discretion
- Exact N grid for the payback-N downward sweep per target (e.g. karcher over N∈{10,50,100,...}; streaming over N_obj∈{1,10,50,...}) — planner picks values that actually bracket each target's crossover, documented in the report.
- Criterion `sample_size` / `measurement_time` tuning per sweep cell within the D-04 protocol (heavy karcher cells will need long measurement time / reduced samples; light streaming cells keep defaults).
- Exact bench-function/group naming inside `audit_hotpaths.rs` and the `bench/` artifact filenames, following the `p5_*` convention.
- Whether the thread sweep reuses existing `bench_p3_karcher` / streaming bench cells (parameterized by an env-read of `RAYON_NUM_THREADS`) or adds dedicated `p5` cells — a bench-organization call, as long as artifacts are `p5`-tagged and linked.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition & requirements
- `.planning/ROADMAP.md` §"Phase 5: Parallelism Gap Assessment" — the 4 success criteria (thread-scaling table + payback-N, safe-to-parallelize list with RNG note, opt-in/unaccelerated-cost record, backlog fields).
- `.planning/REQUIREMENTS.md` — PERF-05 (this phase's requirement).

### Prior evidence to build on / cite (do NOT re-derive)
- `.planning/research/AUDIT-REPORT.md` §"Parallelism Gap List" (Phase 2) — the authoritative SEQUENTIAL vs ALREADY-PARALLEL inventory with exact `file:line` anchors and gate tags. **This is the input for SC2** (D-07). Also §"Feature-combo sentinel" / karcher combo table (Phase 2/1) — the rayon-off cost cited in SC3 (D-08a).
- `.planning/research/AUDIT-REPORT.md` §"Phase 3: Elastic Alignment Hot Path" — the `karcher_mean` vs `karcher_mean_banded` numbers cited for SC3 banding-opt-in cost (D-08b), and the 34–58% two-run variance that motivates the stricter stability controls (D-04).
- `.planning/phases/01-measurement-discipline-baselines/01-CONTEXT.md` — Phase-1 measurement discipline (D-01..D-07): report append convention, bench naming, feature matrix, ±5%/2-run baseline this phase escalates from.
- `.planning/phases/03-elastic-alignment-hot-path/03-CONTEXT.md` — Phase-3 D-07 backlog-entry field format (function/current-cost/root-cause + one-line candidate fix) reused for SC4; D-05 banding-opt-in root cause.

### Measurement discipline
- `.planning/research/PITFALLS.md` — Pitfall 7 (noisy machine / variance / pinning — governs D-04), Pitfall 3 (`black_box`), Pitfall 17 (reproducible artifacts under `bench/`). Rayon-overhead-on-small-N is the SC1 payback-threshold pitfall.

### fdars-side inputs
- `.planning/codebase/CONCERNS.md` §"Performance Bottlenecks" / §"Scaling Limits" — parallel-overhead-below-n≈100 note (informs the payback-N sweep range) and elastic cost that caps karcher cell sizes.
- `fdars-core/src/parallel.rs` — the 5 `iter_maybe_parallel!` / `slice_maybe_parallel!`-family macros and their `parallel` feature gate; the mechanism every SC2 candidate would use and every SC1 target already uses.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`fdars-core/benches/audit_hotpaths.rs`** — the audit bench file (Phases 1–4) already contains `karcher` and streaming/`p1`-era sentinel cells and a seeded `FdMatrix` generator; the `RAYON_NUM_THREADS` sweep can parameterize existing karcher + streaming-depth cells rather than writing new algorithm harnesses.
- **Phase-1/2/3 bench artifacts** under `.planning/research/bench/` (`p1_karcher_none_run1.txt`, `p1_karcher_parallel_run1.txt`, `p3_karcher_*`) — cited directly for the SC3 unaccelerated-path costs; no re-measurement.
- **Phase-2 Parallelism Gap List** — a ready-made, source-verified SC2 candidate list (each entry already grep-confirmed against `src/`).

### Established Patterns
- `parallel` feature gates all `*_maybe_parallel!` macros (`default = ["parallel"]`); `RAYON_NUM_THREADS` controls the rayon pool size at runtime without recompiling — the sweep varies threads via env, holding the `linalg,parallel` build fixed.
- Per-thread deterministic RNG seeding `StdRng::seed_from_u64(seed + k)` — the thread-safety pattern the SC2 RNG note references.
- Two-run ±5% variance methodology (Phase 1) — escalated here to pinned 3-run for sweep cells (D-04).

### Integration Points
- Append a new `## Phase 5: Parallelism Gap Assessment` section to `.planning/research/AUDIT-REPORT.md` (single-file convention, D-05).
- New `p5_*` raw artifacts under `.planning/research/bench/`.
- Possibly parameterize existing `audit_hotpaths.rs` bench cells with an env-read of `RAYON_NUM_THREADS` (Claude's discretion, D-08/D-01).

</code_context>

<specifics>
## Specific Ideas

- Sweep sentinels chosen to **bracket** the payback crossover: one heavy (`karcher_mean`), one light (`StreamingFraimanMuniz::depth_batch`).
- Payback baseline is `RAYON_NUM_THREADS=1` (single-thread rayon), not `--no-default-features`, so thread count is the sole variable (D-02).
- SC3 numbers are reused, not re-measured: rayon-off ~10× (Phase 1 karcher combo), banding-opt-in ~7× (Phase 3 karcher banded).
- Stability protocol: `taskset` core-pin + `cpupower` performance governor + 3 runs (median + spread), documented in the report.

</specifics>

<deferred>
## Deferred Ideas

- **Actually parallelizing any SEQUENTIAL loop** (wrapping CV folds / elastic-FPCA N-loops / `center_columns` in `iter_maybe_parallel!` and measuring real speedup) — a future implementation milestone; this audit only projects the speedup statically (D-06).
- **Thread counts beyond 8 / NUMA-aware scaling** — out of scope for an assessment on the current dev machine; note as a backlog consideration only if the 1→8 curve is still climbing steeply at 8.
- **Re-measuring the elastic/FPCA deep sweeps under the new pinned protocol** — Phases 3–4 own those tables; Phase 5 cites them.
- **nalgebra-vs-faer SVD comparison** — Phase 6 (PERF-06).

None of these are scope creep — they are downstream/other-phase work, noted to keep Phase 5's assessment lean.

</deferred>

---

*Phase: 5-Parallelism Gap Assessment*
*Context gathered: 2026-08-08*
