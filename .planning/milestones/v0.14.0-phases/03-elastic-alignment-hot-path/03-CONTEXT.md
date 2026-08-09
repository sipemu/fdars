# Phase 3: Elastic Alignment Hot Path - Context

**Gathered:** 2026-08-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Run the **deep criterion benchmark sweep** for the elastic-alignment hot path and confirm with real numbers whether it is fdars' top bottleneck, then write this slice's report section + backlog entries. This is an **analysis phase** — the only code added is measurement infrastructure (new benches in the audit bench harness), never a change to `fdars-core/src` algorithms.

Concretely, this phase delivers (per ROADMAP Phase 3 success criteria):
1. A criterion **results table** for three targets — `karcher_mean`, `elastic_self_distance_matrix`, `elastic_cross_distance_matrix` — swept over N∈{100,500} × M∈{50,200}, release build with `linalg,parallel` and `black_box`, tagged with feature set + toolchain version, appended to `.planning/research/AUDIT-REPORT.md`.
2. **Banded-vs-unbanded** measurements at a fixed band fraction quantifying the expected ~7× reduction, and confirming `karcher_mean()` defaults to the unbanded path (Anti-Pattern 2).
3. **Reproducible evidence:** raw criterion output under `.planning/research/bench/` (Phase-1 naming convention), each finding links to its artifact, two-run variance within ±5% (>10% = LOW CONFIDENCE).
4. **Backlog entries** (elastic-alignment perf items) with function / current-cost / root-cause fields, drafted GSD-ready for final ranking in Phase 9.

**MVP mode:** the phase is one vertical slice — *measure → report section → backlog slice* — end to end, not horizontal layers.

**Not in scope:** implementing any fix (defaulting to banded, exposing `band_frac` in a higher-level API, etc. — those are backlog items for a future milestone); benchmarking non-elastic modules (Phases 4–6 own those); the final cross-module ranking (Phase 9, RPT-01); allocation profiling / dhat (Phase 4).

</domain>

<decisions>
## Implementation Decisions

### Targets & Sweep Grid
- **D-01:** Three targets in the results table, all swept over N∈{100,500} × M∈{50,200} at `release + linalg,parallel` with `black_box`: `karcher_mean` (`alignment/karcher.rs:293`), `elastic_self_distance_matrix` (`alignment/pairwise.rs:194`), `elastic_cross_distance_matrix` (`alignment/pairwise.rs:266`). Honors the Phase-1 elastic cap (N≤500, M≤200) — no extra cells. — **Reversibility:** costly — the grid is the shared size contract Phase 9 ranks against and that Phase 1 already fixed; changing cells breaks cross-phase comparability.
- **D-02:** `elastic_cross_distance_matrix` is benched **square N×N** (`data1 = data2 = N` curves from the grid, N∈{100,500}), so its cost is directly comparable to the N×N self-distance matrix. Not a train/test reference-set shape.

### Banded vs Unbanded (SC2)
- **D-03:** Fixed **`band_frac = 0.1`** (10% Sakoe–Chiba corridor; `band_frac` is a domain fraction 0..1 converted via `band_radius(band_frac, m)`). At M=200 this is ≈20 points — ≈10× theoretical DP reduction, ~7× expected after overhead. Quantify the observed reduction against this value; caveat any alignment-quality implications lightly (fix framing is Phase-9/backlog territory).
- **D-04:** Measure the banded-vs-unbanded pair for **all three targets** — `karcher_mean` vs `karcher_mean_banded`, `elastic_self_distance_matrix` vs `_banded`, `elastic_cross_distance_matrix` vs `_banded` — not just karcher. Gives full evidence for the ~7× claim across the elastic hot path.
- **D-05:** Explicitly confirm and record that `karcher_mean()` passes `band_frac = 0.0` → unbanded full DP by default (source: `karcher.rs:300` → `karcher_mean_impl(..., 0.0)`), i.e. banding is opt-in via `karcher_mean_banded()` — this is Anti-Pattern 2 from Phase 2's AUDIT-REPORT and the root cause of the top backlog item.

### karcher_mean Fixed Parameters (reproducibility)
- **D-06:** Lock `karcher_mean(max_iter = 20, tol = 1e-4, lambda = 0.0)` for every karcher cell. Matches the Phase-1 elastic baseline for cross-phase comparability; `lambda = 0.0` (no warp penalty) is the standard elastic default; deterministic on seeded synthetic data. Distance-matrix targets take the same `lambda = 0.0`.

### Backlog Root-Cause Depth (SC4)
- **D-07:** Each backlog entry carries the SC4 fields — **function, current-cost** (Phase-3 measured numbers), **root-cause** citing the relevant AUDIT-REPORT anti-pattern / complexity row — **plus a one-line candidate fix** (e.g. "default `karcher_mean` to a banded path", "expose `band_frac` on the high-level API"). Cite Phase 2's static analysis rather than re-deriving it; keep entries GSD-ready for Phase 9 ranking, not fully-specified fixes.

### Claude's Discretion (planner/researcher)
- Criterion `sample_size` / `measurement_time` tuning per cell within the Phase-1 methodology (workload matrix already prescribes `measurement_time = 60s` for the borderline N=500×M=200 elastic cell; smaller cells may keep defaults). Document whatever is applied.
- Seeded synthetic input generator for the audit benches (Phase 1 established the audit bench needs its own seeded N×M generator built on `FdMatrix` column-major layout; reuse/extend that generator).
- Exact bench-function/group naming inside the audit bench file and the per-target `bench/` artifact filenames, following the Phase-1 `p<phase>_<target>_<features>_run<N>.txt` convention (here `p3_*`).
- Whether banded and unbanded share one criterion group per target or are split — a presentation call, as long as both land in the table with linked artifacts.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition & requirements
- `.planning/ROADMAP.md` §"Phase 3: Elastic Alignment Hot Path" — the 4 success criteria (results table, banded-vs-unbanded ~7×, reproducibility, backlog slice) and the `**Mode:** mvp` marker.
- `.planning/REQUIREMENTS.md` — PERF-03 (this phase's requirement).

### Inherited measurement discipline (MANDATORY — governs this sweep)
- `.planning/phases/01-measurement-discipline-baselines/01-CONTEXT.md` — the locked conventions this phase inherits: single growing `AUDIT-REPORT.md` (D-05), `bench/` artifact naming `p<phase>_<target>_<features>_run<N>.txt` (D-06), elastic size cap N≤500/M≤200 (D-07), `black_box` + feature-tag + toolchain capture, ±5%/>10% variance rule.
- `.planning/research/PITFALLS.md` — Pitfall 1 (release binary path), 3 (`black_box`), 4 (representative sizes), 7 (±5% two-run variance), 8 (linker/toolchain = infra failure not code failure), 17 (reproducible evidence under `bench/`).
- `.planning/research/AUDIT-REPORT.md` — Phase 1 methodology + workload matrix (append this phase's section here, do not create a new file); the elastic complexity row (O(max_iter·N·m²) unbanded / O(max_iter·N·m·band) banded) and the **Anti-Pattern 2 / banding-opt-in** note that this phase confirms and turns into backlog root-cause; the Phase-1 elastic baseline (`elastic_self_distance_matrix` N=100×M=50 ≈ 790 ms) for sanity-checking the new numbers.

### fdars-side source (targets & API shape)
- `fdars-core/src/alignment/karcher.rs` — `karcher_mean` (:293, defaults `band_frac=0.0`), `karcher_mean_banded` (:312, `band_frac` domain fraction), `karcher_mean_impl` (:323).
- `fdars-core/src/alignment/pairwise.rs` — `elastic_self_distance_matrix` (:194) + `_banded` (:205), `elastic_cross_distance_matrix` (:266) + `_banded` (:278); `band_radius(band_frac, m)` conversion.
- `.planning/codebase/CONCERNS.md` §"Performance Bottlenecks"/"Scaling Limits" — O(n²·m²) elastic cost that justifies the N≤500/M≤200 cap.
- `.planning/codebase/TESTING.md` §"Criterion benchmarks" — existing bench files + `harness = false` convention the audit bench mirrors.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase-1 audit bench harness** (`fdars-core/benches/audit_hotpaths.rs` + its `[[bench]]` entry and seeded N×M generator, created in Phase 1): extend it with the Phase-3 elastic sweep rather than starting a new file. Phase 1 already benched `elastic_self_distance_matrix` at N=100×M=50 — this phase completes the grid and adds karcher + cross + banded twins.
- **Banded API twins already exist** — every target has a `_banded(band_frac)` public function, so the banded-vs-unbanded comparison needs no new library code, only new bench cases.
- **Criterion 0.5** already a dev-dependency; `iter_maybe_parallel!` in the elastic inner N-loop means `parallel` is exercised — run under `linalg,parallel` per D-01.

### Established Patterns
- `[[bench]]` with `harness = false` in `fdars-core/Cargo.toml`; `black_box` on inputs AND outputs (Phase-1 D-02).
- Column-major `FdMatrix` — synthetic inputs built respecting `row + col*nrows` layout; distance-matrix functions take `&FdMatrix` + `argvals: &[f64]` + `lambda: f64`.
- `band_frac` semantics: `≤ 0` or `≥ 1` reproduces the unbanded path; the banded functions convert to a point radius via `band_radius`.

### Integration Points
- Append benches to `fdars-core/benches/audit_hotpaths.rs` (+ Cargo.toml `[[bench]]` if a new file is chosen instead).
- Append the Phase-3 results section to `.planning/research/AUDIT-REPORT.md`.
- Write raw criterion output to `.planning/research/bench/p3_*` and link each finding.

</code_context>

<specifics>
## Specific Ideas

- band_frac = 0.1 chosen as the fixed Sakoe–Chiba corridor for SC2; report the observed reduction vs the ~7× expectation and vs the ~10× theoretical (m/band at M=200).
- karcher_mean bench params locked at max_iter=20, tol=1e-4, lambda=0.0 (Phase-1-consistent).
- Cross-distance benched square N×N for direct comparability with self-distance.
- Backlog top item is expected to be "default `karcher_mean`/distance matrices to banding (or expose `band_frac`)" — root-caused to Anti-Pattern 2, quantified by this phase's banded-vs-unbanded numbers.

</specifics>

<deferred>
## Deferred Ideas

- **Implementing the banding default / API change** — a future implementation milestone, not this audit. Phase 3 only measures and drafts the backlog item.
- **Allocation profiling (dhat) of elastic SVD copies** — Phase 4 (PERF-04); AUDIT-REPORT already lists the `elastic_fpca.rs` `to_dmatrix()` copies as Phase-4 dhat candidates.
- **Parallelizing the sequential elastic FPCA loops** (`elastic_fpca.rs:701/720/764`) — Phase 5 (PERF-05) candidate per AUDIT-REPORT.
- **RAYON_NUM_THREADS thread-scaling sweep** on karcher's parallel N-loop — Phase 5 (PERF-05).
- **Final cross-module bottleneck ranking** — Phase 9 (RPT-01); this phase supplies the elastic slice only.

None are scope creep — all are downstream roadmap phases, noted only to keep Phase 3 focused on the elastic measurement.

</deferred>

---

*Phase: 3-Elastic Alignment Hot Path*
*Context gathered: 2026-08-07*
