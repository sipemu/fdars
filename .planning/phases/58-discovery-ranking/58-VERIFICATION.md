---
phase: 58
title: Discovery & Ranking — Verification
status: passed
requirements: [SHP-03, SHP-04, SHP-05]
commit: b18b0a7a
---

# Phase 58 Verification — Discovery & Ranking

Each ROADMAP success criterion → PASS/FAIL with evidence.

## Criterion 1 — Tractable candidate generation (exhaustive OR contracted/seeded), n=100/m=200 well under 10s
**PASS.** `generate_candidates` enumerates `(series_idx,start,length)` over `[min_length,max_length]` and, when exhaustive_count > `max_candidates`, deterministically random-samples via `seed_for_thread(seed,0)` (rejection over flat indices, decoded + sorted). `test_discover_tractable_contracted` (n=100, m=200, `max_candidates=Some(800)`) asserts `elapsed < 10s` and `len ≤ max_shapelets` — green. Exhaustive path exercised by `test_discover_known_motif` (`max_candidates=None`). Naive full O(n²·M³) search is never the only path (bounded by `max_candidates`).

## Criterion 2 — Optimal distance-split quality (IG default, F-statistic alternative via QualityMeasure); discriminative ≈ max entropy, random ≈ 0
**PASS.** `information_gain` sorts the orderline (`total_cmp`), sweeps all distinct-distance midpoints incrementally, returns `max_θ IG` — no fixed threshold. `f_statistic_1d` provides the ANOVA alternative selectable via `QualityMeasure::{InfoGain,FStatistic}`. Evidence: `test_infogain_optimal_split` (clean split → IG=1.0 == max entropy; degenerate → 0.0); `test_discover_known_motif` (top shapelet IG > 0.9); `test_fstatistic_measure` (discriminative F-stat > noise F-stat, and end-to-end FStatistic path scores > 0). All green.

## Criterion 3 — Top-K with self-similarity pruning (overlapping same-series candidates dropped)
**PASS.** Greedy selection over quality-ranked candidates tracks per-series accepted `[start,end)` intervals and skips any not-yet-selected same-series candidate whose range overlaps (`!(end<=s || e<=start)`); stops at resolved `max_shapelets`. `test_self_similarity_pruning` asserts no two selected same-series shapelets overlap — green. (Column-correlation/series-diversity is a Phase 59 transform-level property; the position-overlap guard that produces it is verified here.)

## Criterion 4 — Reproducible: seeded sampling + total_cmp tie-break → byte-identical fits, no partial_cmp().unwrap()
**PASS.** Candidate set fixed by seed before scoring; scoring pure; ranking uses `b.0.total_cmp(&a.0)` with `(series_idx,start,length)` tie-break (no `partial_cmp().unwrap()` anywhere in the module). `test_discover_deterministic` (over-budget → sampling active) asserts two same-seed fits are byte-identical (`assert_eq!` on `ShapeletSet`) — green. Sequential (default-features run, `parallel` off) and parallel (`linalg,parallel`) both produce 14/14 passing, confirming sequential==parallel.

## Gate summary
- `cargo test -p fdars-core --features linalg shapelet` → 14 passed / 0 failed (lib).
- `cargo test -p fdars-core shapelet` (default features) → 14 passed / 0 failed.
- `cargo test -p fdars-core --features linalg --doc shapelet` → 3 passed (incl. discovery doctest).
- `cargo fmt --check` → clean.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → clean.

All 4 criteria PASS → status: passed.
