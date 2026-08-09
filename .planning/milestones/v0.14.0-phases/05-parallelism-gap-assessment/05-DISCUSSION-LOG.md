# Phase 5: Parallelism Gap Assessment - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-08
**Phase:** 5-Parallelism Gap Assessment
**Areas discussed:** Thread-sweep targets, Machine-stability controls, Sequential-gap evidence, Unaccelerated-path cost

---

## Thread-sweep targets (SC1)

| Option | Description | Selected |
|--------|-------------|----------|
| karcher (heavy) + streaming depth (light) | One compute-heavy loop + one lightweight loop bracket the payback-threshold N | ✓ |
| karcher + pairwise (both heavy) | Two heavy loops — clearest raw scaling but no small-N overhead regime | |
| streaming depth + nadaraya_watson (both light) | Two fast loops — good for overhead regime but sub-ms jitter | |

**User's choice:** karcher (heavy) + streaming depth (light)
**Notes:** Chosen to bracket the crossover — heavy target gives the large-N speedup curve, light target exposes the overhead-dominated small-N regime where the payback-N lives.

---

## Machine-stability controls (methodology)

| Option | Description | Selected |
|--------|-------------|----------|
| Escalate: pin + governor + 3 runs | taskset + cpupower/performance + 3 runs (median+spread) for sweep cells | ✓ |
| Keep Phase-1 ±5%/2-run as-is | Reuse existing discipline unchanged | |
| Pin only cells with >10% variance | Adaptive escalation | |

**User's choice:** Escalate: pin + governor + 3 runs
**Notes:** Motivated by Phase-3's 34–58% two-run variance on karcher and Phase-2's explicit taskset/cpupower recommendation; a thread-scaling curve is noise without it.

---

## Sequential-gap evidence (SC2, audit-only)

| Option | Description | Selected |
|--------|-------------|----------|
| Static safety argument + projected speedup | Independence/RNG note + projection from already-parallel analogue; no src edits | ✓ |
| Throwaway prototype-and-bench | Temporarily wrap each loop and bench real speedup | |
| Static + prototype top candidate only | Static for all; prototype just the highest-leverage loop | |

**User's choice:** Static safety argument + projected speedup
**Notes:** Keeps the milestone audit-only (no fdars-core/src edits, even on a scratch branch). Candidate set = the Phase-2 SEQUENTIAL gap list verbatim.

---

## Unaccelerated-path cost (SC3)

| Option | Description | Selected |
|--------|-------------|----------|
| Both dimensions, reuse Phase-1/3 numbers | rayon-off (~10×, Phase 1) + banding-opt-in (~7×, Phase 3), cite existing artifacts | ✓ |
| Rayon-off only, re-measure fresh | Focus on parallel feature-gate default, re-run | |
| Both, but re-measure under new controls | Cover both, re-run under Phase-5 pinning/3-run | |

**User's choice:** Both dimensions, reuse Phase-1/3 numbers
**Notes:** Both existing numbers are high-enough confidence for a "cost of the default path" statement; re-measuring adds bench time and risks cross-phase inconsistency. LOW-CONFIDENCE caveat noted when citing the Phase-3 banding number.

---

## Claude's Discretion

- Exact N grid for the payback-N downward sweep per target.
- Criterion `sample_size` / `measurement_time` tuning per sweep cell within the pinned protocol.
- Bench-function/group naming in `audit_hotpaths.rs` and `p5_*` artifact filenames.
- Whether to parameterize existing bench cells via a `RAYON_NUM_THREADS` env-read or add dedicated `p5` cells.

## Deferred Ideas

- Actually parallelizing any SEQUENTIAL loop (future implementation milestone).
- Thread counts beyond 8 / NUMA-aware scaling.
- Re-measuring elastic/FPCA deep sweeps under the pinned protocol (Phases 3–4 own them).
- nalgebra-vs-faer SVD comparison (Phase 6).
