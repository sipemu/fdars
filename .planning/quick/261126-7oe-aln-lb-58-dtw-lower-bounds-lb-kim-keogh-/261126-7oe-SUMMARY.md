---
status: complete
quick_id: 261126-7oe
issue: 58
---
# Summary — ALN-LB (GH #58): DTW lower bounds, EAPrunedDTW, exact 1-NN

## Commits
- `87960934` feat(quick-261126-7oe): DTW lower bounds (LB_Kim/Keogh/Webb), EAPrunedDTW and exact 1-NN search (refs #58; body `closes #58`)

## Public API added (new `metric::dtw_lb`, re-exported from `metric` and crate root; `dtw_nearest_neighbor` also in prelude)
- `DtwEnvelope { lower, upper }` (non_exhaustive, serde)
- `lb_envelope(x, radius)` — tslearn `lb_envelope`; Lemire streaming min/max (Vec-backed monotone deque)
- `lb_envelope_windowed(y, query_len, window)` — envelope over the admissible row ranges of any `DtwWindow` (Itakura, aeon fractions, unequal lengths; Sakoe-Chiba unequal = dtaidistance's `lb_keogh` window with `window = radius + 1`)
- `lb_keogh(query, &envelope, p)`, `lb_kim(x, y, p)`, `lb_webb(x, y, p, window)` (requires p ≥ 1)
- `dtw_distance_early_abandon(x, y, p, window, cutoff) -> Result<Option<f64>>` — EAPrunedDTW
- `dtw_nearest_neighbor(queries, references, p, window) -> Result<DtwNearestNeighbors { indices, distances, stats: DtwSearchStats }>`
- Internal: `dtw::LocalCost` / `validate_dtw_inputs` made `pub(super)` (no behaviour change).

## Conventions / design
- **Squared vs unsquared:** every quantity uses the `dtw_distance_windowed` convention `Σ|x_i − y_j|^p`, no root. tslearn `dtw`/`lb_keogh` and dtaidistance report `sqrt` of the p=2 value — documented in module and fn docs; goldens compare `ours.sqrt()` and `ours` vs `tslearn²`.
- **LB_Kim** adapted to sum-of-costs DTW: `max(c(first)+c(last), c(max x, max y), c(min x, min y))`, window independent, valid for every p > 0.
- **LB_Webb** implemented from the paper (Theorem 2 / Alg. 2) with an independent correctness proof generalising it to arbitrary windows: envelopes of the candidate over row ranges R(i), of the query over column ranges C(j) (exact two-pointer column ranges for monotone staircase windows, contiguous hull otherwise — still valid). Returned value = max(LR variant with the paper's MinLRPaths (n, m ≥ 6, admissible 3-cell end paths only), parameter-free LB_Webb_NoLR) so LB_Keogh ≤ LB_Webb holds unconditionally (the LR variant can capture less than Keogh's six end rows). Requires p ≥ 1 (convexity used in the proof); returns InvalidParameter otherwise.
- **EAPrunedDTW:** pruned cells stored as +inf; row starts at the previous row's first kept cell, stops past its last kept cell once a cell is pruned; abandons when a row is fully pruned. Same recurrence/operand order as `dtw_distance_windowed` → bit-identical result whenever DTW ≤ cutoff (proved by induction: every cell with true value ≤ cutoff is computed exactly).
- **1-NN:** references' Keogh envelopes precomputed once (parallel); LB_Webb's envelope-of-envelope built lazily per reference (`OnceLock`), as most candidates die at LB_Keogh. Per query (parallel): candidates sorted by LB_Keogh (scan stops when LB_Keogh > best), then LB_Kim, LB_Webb (p ≥ 1), then EAPrunedDTW with cutoff = best and the UCR-suite cumulative LB_Keogh suffix bound. Bound comparisons use a relative slack `max(1e-10, 16(n+m)ε)` so rounding can never prune the true NN; the final DTW cutoff test is exact, ties resolved to the lowest index → identical to brute force.
- Max cognitive complexity in new code 16 (pmat).

## Validation
- tslearn 0.10.0 `lb_envelope` (radius 0–3 + docstring) exact, `lb_keogh` (sqrt, 1e-14) and `dtw` Sakoe-Chiba (sqrt, 1e-14).
- dtaidistance 2.5.1 `lb_keogh` for unequal lengths (both directions, window 1–3), 1e-13.
- LB_Webb vs the authors' Java `Bounds.lbWebb` (GIWebb/DTWBounds; harness `golden_lbwebb_Gold.java`): 36 pairs × w ∈ {1,2,3}, 1e-12 (summation order differs); also their lbKeogh and DTW. w = 0: NoLR equals the diagonal DTW.
- Invariants: LB_Kim ≤ DTW, LB_Keogh ≤ LB_Webb ≤ DTW on ~2000 random-walk pair/window/p combos (10 windows incl. Itakura and aeon fractions, equal and unequal lengths, p ∈ {1, 1.5, 2, 3}); LB_Webb strictly tighter than LB_Keogh in > 50 %. p = 0.5: Kim/Keogh still bounds, Webb rejected.
- EAPrunedDTW: bit-identical for cutoff ∈ {inf, DTW, 1.5·DTW, DTW+1}, `None` for cutoff ∈ {prev float of DTW, DTW/2, 0, −1}; all windows, p ∈ {0.5, 1, 1.5, 2}, sizes incl. 1×1, 1×6, 6×1.
- 1-NN == brute force (index + distance bits) over 3 length pairs × 10 windows × 4 p values; tie/duplicate test (lowest index); clustered-data pruning test (> 80 % pruned); stats sum to #queries × #refs.
- Envelopes vs naive scan, column ranges vs `DtwWindow::mask`.

## Benchmark (`cargo bench -p fdars-core --bench dtw_lb_benchmarks -- --quick`)
500 refs × length 128, 8 noisy random-walk prototypes, Sakoe-Chiba radius 12, p = 2 (CPU was clocked at ~0.9 GHz, so absolute times are inflated; ratios are what matters):
| case | brute force | LB cascade + EAPrunedDTW | speedup |
|---|---|---|---|
| 1 query (sequential DTW; NN precompute parallel) | 7.12 ms | 1.67 ms | 4.3× |
| 20 queries (both parallel over queries) | 18.7 ms | 2.70 ms | 6.9× |
Stats (20 queries): 8671 pruned by LB_Keogh, 77 by LB_Webb, 0 by LB_Kim, 1161 abandoned, 91 completed of 10000.

## Gates
- `cargo fmt --all` ✓
- `cargo test -p fdars-core --lib metric::` 192 passed ✓ (17 new in `metric::dtw_lb`)
- `cargo test -p fdars-core --doc dtw` 33 passed ✓ (7 new doctests)
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` clean ✓
- `cargo check -p fdars-core --features serde` ✓; `--no-default-features` ✓

## Deviations
- `dtw_nearest_neighbor` takes `p` (between `references` and `window`) for consistency with `dtw_distance_windowed`; the issue text listed `(queries, references, window)`.
- `lb_webb` returns max(LR variant, NoLR variant) rather than the paper's LR variant alone, to make LB_Keogh ≤ LB_Webb unconditional (the LR variant alone matches the Java reference and is what is golden-tested).
- LB_Kim pruned nothing in the bench (as expected after LB_Keogh ordering); kept as a cheap cascade stage per the issue.
- Not done (out of scope): LB_Improved / LB_Petitjean / LB_Enhanced, k-NN (k > 1), multivariate series.
