# Phase 12: Elastic Feasibility — Banded Alignment Default & `band_frac` - Context

**Gathered:** 2026-08-11
**Status:** Ready for planning
**Mode:** Autonomous smart-discuss + one user grey-area decision

<domain>
## Phase Boundary

Expose a **banded** Sakoe-Chiba dynamic-programming path through the high-level elastic alignment API — `karcher_mean`, `elastic_self_distance_matrix`, `elastic_cross_distance_matrix` — via a `band_frac` control, so previously-infeasible large grids (N=500, M=200) become tractable. The banded implementations (`karcher_mean_banded`, `elastic_self_distance_matrix_banded`, `elastic_cross_distance_matrix_banded`) and `band_radius(band_frac, m)` already exist and are correct; this phase is API surfacing, NOT a new algorithm.

Covers requirement **PERF-03** (audit: PERF-ELASTIC-BAND, rank 10, P1/M).
</domain>

<decisions>
## Implementation Decisions

### LOCKED — user decision (grey area)
- **API shape: opt-in, non-breaking.** Expose `band_frac: Option<f64>` on the three high-level functions (or via their config), where `None` (the default) preserves today's **exact unbanded** behavior. Existing callers are unchanged and get identical numerical results; they opt into the 4–6× banded path explicitly by passing `Some(0.1)`.
  - Do **NOT** silently flip the default to banded (`band_frac=0.1`) — that would change existing callers' results (band-approximation) and was explicitly declined. The default-flip may be revisited in a future milestone.

### Claude's discretion (guided by codebase conventions + the audit)
- Prefer threading `band_frac` through the existing `ElasticConfig`/config struct if one is already the parameter-passing idiom for these functions, rather than adding a positional argument (positional additions are breaking — avoid). If a config struct exists, add `band_frac: Option<f64>` (default `None`) to it; if the public functions take loose params, add `band_frac: Option<f64>` as a trailing optional in a non-breaking way (or a new `*_with_band` wrapper if a clean non-breaking signature is otherwise impossible — but a config field is strongly preferred).
- `None` → call the existing unbanded impl (current path). `Some(f)` with `f > 0` → `band_radius(f, m)` → banded impl. `Some(0.0)` → treat as exact/unbanded (equivalent to `None`).
- rustdoc on all three functions must document: default `None` = exact; `Some(0.1)` ≈ 4–6× faster with small band-approximation error; band width is a fraction of M.
</decisions>

<code_context>
## Existing Code Insights

- `alignment/karcher.rs:~300` — `karcher_mean()` currently calls `karcher_mean_impl(.., 0.0)` (hard-coded `band_frac=0.0` → `band_radius(0.0, m) = None` → full O(m²) unbanded DP).
- `alignment/` — `elastic_self_distance_matrix` / `elastic_cross_distance_matrix` plus their correct `_banded` variants; `band_radius(band_frac, m)` helper already exists.
- Complexity: karcher O(max_iter·N·m²) unbanded vs O(max_iter·N·m·band) banded; distance matrices O(N²·m²) vs O(N²·m·band). Measured banded speedup 4–6× at representative cells.
- Convention: public fns return `Result<T, FdarError>`, column-major `FdMatrix`, feature-gated parallelism. Config structs (e.g. `ElasticConfig`) are the established builder-style idiom — plan-phase research should confirm the exact current signatures.
</code_context>

<specifics>
## Specific Ideas

- Success criteria (from ROADMAP): `band_frac` threaded into the DP warp search across `karcher.rs` + both distance-matrix functions; full unbanded path remains available/exact; an inline test asserts wide-band output matches unbanded within numerical tolerance at a small M where exact comparison is feasible; a benchmark/timing test demonstrates feasibility at a large (N, M) where the unbanded default was infeasible.
- Numerical-equivalence test: at a sufficiently wide band (band_frac large enough that the band covers the full warp), banded output must equal unbanded within tolerance.
</specifics>

<deferred>
## Deferred Ideas

- Flipping the default to `band_frac=0.1` (banded-by-default) — declined for this milestone; candidate for a future milestone once users have adopted the opt-in.
- Parallelizing the elastic-FPCA inner loops (PERF-PAR-ELFPCA) — separate deferred backlog item.
</deferred>
