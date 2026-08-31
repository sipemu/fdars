# Phase 48: Parallelism-Gap Closure — Research

**Researched:** 2026-08-31
**Domain:** Feature-gated rayon parallelism — permutation-test loops and multi-restart clustering
**Confidence:** HIGH (all candidate source files read directly this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Parallelize the PROF-flagged sequential permutation loops — `t_perm_test` / `f_perm_test`
  (`src/inference/permutation.rs`), `frechet_anova` (`src/frechet/anova.rs`) — and the `co_cluster`
  n_init loop (`src/coclustering.rs`). These are high-cost, embarrassingly-parallel outer loops that
  PROF-01/02 flagged as sequential (parallel already exists in function_on_scalar/famm/explain_generic).
- Parallelize at the outer independent-iteration level (permutation replicas / random inits) using
  the existing `iter_maybe_parallel!` macro — not inner/nested loops.
- Research confirms which loops are genuinely sequential AND worth it (payback positive); marginal
  or tiny loops are deferred with a documented note, not force-parallelized.
- Reuse the Phase 47 harness: extend `benches/perf_hotpaths.rs` (or a sibling) for thread-scaling
  and add an equivalence test file.

### Equivalence & Determinism
- Bit-identical parallel-ON vs parallel-OFF output (SC2) — not merely within tolerance.
- Determinism holds via per-iteration seeding: replica/init `k` uses
  `StdRng::seed_from_u64(seed + k)`, so its RNG stream depends only on `k`, never on thread or
  execution order → result is independent of thread count. Any parallelized randomized loop MUST
  preserve this exact pattern (SC4).
- Test both feature configs: a committed equivalence/golden test asserts the seeded result; run the
  suite under both `--features linalg,parallel` and parallel-off to prove both branches produce the
  same output.
- Permanent golden regression guard for each parallelized fn's output.

### Payback Threshold & Benchmark Evidence
- Outer-if payback-threshold guard: `if work >= THRESHOLD { parallel } else { sequential }` where a
  small-input regression is possible (matching the v0.17.0 `SCORES_PARALLEL_THRESHOLD` precedent).
- Derive the threshold from criterion thread-scaling measurement (1 vs N threads) — set conservatively.
- Speedup evidence: criterion thread-scaling cells (RAYON_NUM_THREADS 1 vs 20) showing large-input
  speedup; record before/after in a `PERF-PARALLEL-RESULTS.md`.
- Register the thread-scaling benches permanently (`[[bench]]`) — they become Phase 51 BENCH-02 guards.

### Deferred Ideas (OUT OF SCOPE)
- Loops already parallel (function_on_scalar, famm, explain_generic) — no work.
- Marginal/tiny sequential loops where payback is negative → documented defer, not parallelized.
- Formalizing the thread-scaling benches as documented regression guards → Phase 51 (BENCH-02).
- Any parallelization that would change numeric output (non-deterministic reduction order) → out of
  scope (must stay bit-identical).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PERF-03 | Parallelism gaps identified in the newer subsystems are closed with feature-gated rayon (via the existing `parallel.rs` macros), equivalence-tested vs the sequential path, with a payback-threshold N guard where a small-input regression is possible. | Candidates classified below. `frechet_anova` (both functions) and `t_perm_test`/`f_perm_test` are SAFE (per-perm reseeding already in place). `co_cluster` n_init is SAFE (per-init seeding via `wrapping_add(init * 1000)`) but requires `?`-propagation restructuring. `explain/importance.rs` deferred. |
</phase_requirements>

---

## Summary

This research classifies every sequential permutation/init loop in the codebase as either
**SAFE** (bit-identical-safe to parallelize via `iter_maybe_parallel!`) or **DEFER** (not
bit-identical-safe without changing outputs), and provides exact rewrite skeletons for the SAFE
candidates.

**The crux finding:** The determinism classification for each loop hinges on whether it seeds the
RNG once for the whole loop (advancing a shared state — NOT safe to parallelize bit-identically) or
whether it reseeds per-iteration with a deterministic formula (fully independent of iteration order
— SAFE to parallelize).

- `frechet_anova` / `frechet_anova_space` (`src/frechet/anova.rs`): **SAFE** — each permutation
  iteration reseeds with `StdRng::seed_from_u64(seed.wrapping_add(perm as u64))`. Already
  structured for parallelism; just add `iter_maybe_parallel!` + `atomic_sum`.
- `t_perm_test` / `f_perm_test` (`src/inference/permutation.rs`): **DEFER (current code) / REQUIRES
  RESTRUCTURE** — both currently seed ONE shared `StdRng` before the loop and advance it with
  successive `shuffle_labels` calls. Per-iteration state is order-dependent. To parallelize
  bit-identically, each permutation `k` must independently seed
  `StdRng::seed_from_u64(seed.wrapping_add(k as u64))` and independently build a fresh labels
  vector. This changes the RNG draw sequence vs. the current code — it is NOT bit-identical to the
  current sequential output. **Decision required:** accept the re-baselining (the statistical
  semantics are identical, just different draws) or defer entirely.
- `co_cluster` n_init loop (`src/coclustering.rs`): **SAFE** — each init uses
  `config.seed.wrapping_add(init as u64 * 1000)` as an independent seed. However, the loop body
  calls `kmeans_fd(...)? ` (error-propagation via `?`) which cannot appear in a rayon `.map()`
  closure directly; requires restructuring to collect `Result`s and propagate after. Also
  `cem_single_fit` returns `(CoClusterResult, Vec<f64>)` (infallible), so it is safe to call in
  parallel after `kmeans_fd` succeeds. The "best-by-log_likelihood" reduction is a pure `max`
  over independent results — commutative and associative, so parallel reduction is bit-identical.
- `explain/importance.rs` (`:131`, `:221`): **DEFER** — the outer-k loop is only `ncomp` wide
  (typically 3–10), which is too narrow for worthwhile parallelism. The inner `n_perm` loop shares
  a single advancing `rng` and cannot be bit-identically parallelized without per-perm reseeding.
  The `explain_generic/importance.rs` counterpart is already parallel over `k` via
  `iter_maybe_parallel!` with per-k seeding; the non-generic version is a lower-priority duplicate
  (CONS-02 will unify them in Phase 49).

**Primary recommendation:** Parallelize `frechet_anova`/`frechet_anova_space` first (highest wall
time, cleanest rewrite). Then `co_cluster` n_init (second-highest, requires `?`-restructure). For
`t_perm_test`/`f_perm_test`, restructure to per-perm reseeding first (which re-baselines output),
then add `iter_maybe_parallel!` — this is a two-step change requiring explicit acknowledgment that
the p-value draws change (statistical equivalence preserved, exact values differ from previous
sequential output).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Permutation-null distribution (frechet ANOVA) | `src/frechet/anova.rs:171` (loop body) | `compute_tn_generic` called per perm | Loop is the parallelization site; `compute_tn_generic` is the work unit |
| Permutation-null distribution (t/f perm test) | `src/inference/permutation.rs:175, 238` (loop body) | `integrated_l2_mean_diff` / `integrated_f_statistic` | Same scaffold; RNG restructure needed before parallelization |
| Multi-restart best-of CEM (co_cluster) | `src/coclustering.rs:935` (init loop) | `cem_single_fit`, `kmeans_fd` | Init loop is the outer parallelization target; `kmeans_fd` result feeds each independent cem fit |
| Payback-threshold guard | Caller site (same file as loop) | `const THRESHOLD` defined at module top | Mirrors `src/elastic_fpca.rs:28` `SCORES_PARALLEL_THRESHOLD` pattern |
| Thread-scaling benchmark | `benches/perf_hotpaths.rs` (extend) | `[[bench]]` in `fdars-core/Cargo.toml` | Extends the existing Phase 47 permanent bench file |
| Equivalence golden test | `tests/equivalence_phase48.rs` (new) | `tests/equivalence_phase47.rs` (pattern) | Mirror of Phase 47 equivalence harness |

---

## Candidate-by-Candidate Classification

This is the primary output of the research. Each candidate is analyzed for: current RNG structure,
bit-identical-safety verdict, exact rewrite, payback threshold, and proof method.

---

### CANDIDATE A: `frechet_anova` + `frechet_anova_space` — SAFE TO PARALLELIZE

**Anchor:** `src/frechet/anova.rs:171–181` (both `frechet_anova` and `frechet_anova_space` at `:259–269`)
**PROF-01 wall time:** 133 ms @ n50_m200, 129 ms @ n200_m50 (both M and N drive cost)
**PROF-01 rank:** #4

#### Current RNG/accumulation structure

[VERIFIED: src/frechet/anova.rs:171-181] Verbatim (both functions are identical in structure):

```rust
    let mut n_ge = 0usize;
    for perm in 0..n_perm {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm as u64));
        let mut perm_labels = group_labels.to_vec();
        perm_labels.shuffle(&mut rng);
        // A degenerate permutation (compute error) is skipped conservatively.
        if let Ok((tn_perm, _, _, _, _)) = compute_tn_generic(&space, &objects, &perm_labels, k) {
            if tn_perm >= tn_obs {
                n_ge += 1;
            }
        }
    }
```

**Key observation:** Each iteration `perm` creates a FRESH `StdRng::seed_from_u64(seed.wrapping_add(perm as u64))`. The RNG is local to the iteration. The `perm_labels` vector is also freshly allocated per iteration. There is NO shared mutable state across iterations except the `n_ge` accumulator.

#### Bit-identical-safe to parallelize: YES

- Each permutation `perm` is completely independent (no shared RNG state, no shared mutable data).
- The seeding formula `seed.wrapping_add(perm as u64)` is deterministic per `perm`, independent of
  thread assignment or execution order.
- The accumulator `n_ge` is a count — parallel map+sum is exactly equivalent to sequential
  accumulation (addition is commutative and associative over integers).
- The degenerate-skip (`if let Ok(...)`) is order-independent: a skipped permutation contributes
  `0` in both sequential and parallel paths.
- Result: `p_permutation = (n_ge + 1.0) / (n_perm + 1.0)` is bit-identical whether computed
  sequentially or in parallel, for any thread count.

#### Exact `iter_maybe_parallel!` rewrite

```rust
// src/frechet/anova.rs — replace the sequential n_ge loop with:
use crate::iter_maybe_parallel;

const FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD: usize = 200;
// Payback threshold: n_perm is the primary work driver; 200 permutations is
// the point where rayon dispatch overhead (≈10µs) becomes negligible vs. per-perm
// work (compute_tn_generic at n50_m50 costs ≈160µs → 200*160µs = 32ms).
// Below 200, sequential is faster (dispatcher overhead > savings).

let n_ge: usize = if n_perm >= FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD {
    iter_maybe_parallel!(0..n_perm)
        .map(|perm| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm as u64));
            let mut perm_labels = group_labels.to_vec();
            perm_labels.shuffle(&mut rng);
            match compute_tn_generic(&space, &objects, &perm_labels, k) {
                Ok((tn_perm, _, _, _, _)) if tn_perm >= tn_obs => 1usize,
                _ => 0usize,
            }
        })
        .sum()
} else {
    // Sequential path — identical to current code.
    let mut count = 0usize;
    for perm in 0..n_perm {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm as u64));
        let mut perm_labels = group_labels.to_vec();
        perm_labels.shuffle(&mut rng);
        if let Ok((tn_perm, _, _, _, _)) = compute_tn_generic(&space, &objects, &perm_labels, k) {
            if tn_perm >= tn_obs { count += 1; }
        }
    }
    count
};
```

Apply identically to `frechet_anova_space` (lines `:259–269`).

**Required imports:** `use crate::iter_maybe_parallel;` at the top of `anova.rs` (not currently present — must add).

**`Sync` bound:** `compute_tn_generic` takes `space: &S` and `objects: &[S::Object]`. For rayon
`par_iter` the closure must be `Send`. Since `S: MetricSpace` and `S::Object: Clone`, `space` is
shared immutably (read-only reference), which is `Sync`-compatible in rayon as long as `S: Sync`.
The `WassersteinDensitySpace` and `SpdMatrixSpace` types are plain structs with no interior
mutability, so they are `Sync`. **If other `MetricSpace` impls are non-Sync, add `S: Sync +
MetricSpace` bound to `frechet_anova_space` only.** Check whether any custom space impl uses
`RefCell`/`Mutex` before adding the `Sync` bound.

#### Payback threshold rationale

- `n_perm` is the work driver (default 999).
- Per-perm cost at n50_m200: ≈133ms / 999 ≈ 133µs per permutation.
- Rayon dispatch overhead per work-item: ≈1–5µs (measured empirically from similar patterns in the
  codebase; [ASSUMED] for the specific overhead on this machine without a direct measurement).
- Threshold = 200: at n_perm ≥ 200, each thread gets ≥10 work-items of ≈133µs each = 1.33ms
  per thread, amortizing dispatch well. Below 200 perms, the overhead may dominate on small inputs.
- The constant is conservative; criterion thread-scaling will confirm the real crossover.

`const FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD: usize = 200;`

#### Proof of equivalence

1. **Golden test** (both ON and OFF feature configs):
   - Same seed, same input → assert `p_value_permutation` bits-identical.
   - Run under `--features linalg,parallel` AND `--no-default-features --features linalg`.
2. **Existing tests confirm determinism:** `anova_permutation_is_seed_reproducible` at
   `src/frechet/anova.rs:354` already asserts `a.p_value_permutation == b.p_value_permutation` for
   the same seed — this test becomes a parallel-vs-sequential regression guard with no code change.
3. **New equivalence test in `tests/equivalence_phase48.rs`:**
   ```rust
   #[test]
   fn golden_frechet_anova_perm_parallel() {
       // Inputs: n=24, k=81 argvals, n_perm=999, seed=42
       // Assert: result from sequential == result from parallel (both features on and off)
       // The existing test `anova_permutation_is_seed_reproducible` already proves seed-to-seed
       // reproducibility; this test is the permanent guard for the specific golden value.
   }
   ```

---

### CANDIDATE B: `t_perm_test` / `f_perm_test` — REQUIRES RESTRUCTURE BEFORE PARALLELIZING

**Anchor:** `src/inference/permutation.rs:173–181` (`t_perm_test`), `:236–244` (`f_perm_test`)
**PROF-01 wall time:** 1.74 ms @ n200_m50 (N-dominated); 992 µs @ n50_m200
**PROF-01 rank:** #9 (lower priority than frechet_anova / co_cluster)

#### Current RNG/accumulation structure

[VERIFIED: src/inference/permutation.rs:173-181] Verbatim (`t_perm_test`):

```rust
    let mut rng = StdRng::seed_from_u64(seed);
    let mut n_ge = 0usize;
    for _ in 0..n_perm {
        shuffle_labels(&mut labels, &mut rng);
        let perm_stat = integrated_l2_mean_diff(&pooled, &labels, n_a, m, &weights);
        if perm_stat >= observed {
            n_ge += 1;
        }
    }
```

[VERIFIED: src/inference/permutation.rs:236-244] Verbatim (`f_perm_test`):

```rust
    let mut rng = StdRng::seed_from_u64(seed);
    let mut n_ge = 0usize;
    for _ in 0..n_perm {
        shuffle_labels(&mut groups, &mut rng);
        let perm_stat = integrated_f_statistic(&pooled, &groups, &labels_dedup);
        if perm_stat >= observed {
            n_ge += 1;
        }
    }
```

**Key observation:** ONE `StdRng` is seeded once and advanced across all permutations in sequence.
The shuffled `labels`/`groups` vector is mutated in-place across iterations. Each permutation's
random draw depends on the PREVIOUS permutation's draw — the chain is order-dependent.

#### Bit-identical-safe to parallelize (AS-IS): NO

If you run iterations in any non-sequential order, each iteration will see a different `rng` state
than in the sequential path. The resulting shuffle is different → the perm stat is different → the
`n_ge` count will differ. The p-value will not be bit-identical to the sequential output.

**This is the critical blocking issue.** The loop cannot be made parallel without restructuring the
RNG usage.

#### Restructuring to per-perm reseeding

To make parallelization bit-identical across thread counts, restructure each permutation to own an
independent RNG seeded as `StdRng::seed_from_u64(seed.wrapping_add(k as u64))`:

```rust
// RESTRUCTURED (changes output vs. old sequential code):
let mut n_ge = 0usize;
for k in 0..n_perm {
    // Per-permutation fresh RNG — deterministic per k, independent of order.
    let mut rng_k = StdRng::seed_from_u64(seed.wrapping_add(k as u64));
    let mut perm_labels: Vec<usize> = (0..(n_a + n_b))
        .map(|i| usize::from(i >= n_a))
        .collect();
    shuffle_labels(&mut perm_labels, &mut rng_k);
    let perm_stat = integrated_l2_mean_diff(&pooled, &perm_labels, n_a, m, &weights);
    if perm_stat >= observed {
        n_ge += 1;
    }
}
```

This sequential restructured version produces **different specific p-values** than the original
sequential code for a given `seed`, because the RNG draws are different. However, the **statistical
properties are identical**: the p-value distribution under the null is unchanged.

**After restructuring, the `iter_maybe_parallel!` form is safe:**

```rust
const T_PERM_PARALLEL_THRESHOLD: usize = 200;

let n_ge: usize = if n_perm >= T_PERM_PARALLEL_THRESHOLD {
    iter_maybe_parallel!(0..n_perm)
        .map(|k| {
            let mut rng_k = StdRng::seed_from_u64(seed.wrapping_add(k as u64));
            let mut perm_labels: Vec<usize> =
                (0..(n_a + n_b)).map(|i| usize::from(i >= n_a)).collect();
            shuffle_labels(&mut perm_labels, &mut rng_k);
            let perm_stat = integrated_l2_mean_diff(&pooled, &perm_labels, n_a, m, &weights);
            usize::from(perm_stat >= observed)
        })
        .sum()
} else {
    (0..n_perm).map(|k| {
        let mut rng_k = StdRng::seed_from_u64(seed.wrapping_add(k as u64));
        let mut perm_labels: Vec<usize> =
            (0..(n_a + n_b)).map(|i| usize::from(i >= n_a)).collect();
        shuffle_labels(&mut perm_labels, &mut rng_k);
        let perm_stat = integrated_l2_mean_diff(&pooled, &perm_labels, n_a, m, &weights);
        usize::from(perm_stat >= observed)
    }).sum()
};
```

Apply the same restructuring to `f_perm_test` (`:236–244`).

#### Output-change classification

The restructuring step changes the exact p-value for any given `seed` — it is NOT bit-identical to
the old sequential output. **However:**

- The test `t_perm_deterministic` (`src/inference/permutation.rs:305`) asserts `r1 == r2` for the
  SAME seed — this test will pass after restructuring (within each config), but the golden values
  will change vs. pre-phase-48 code. The golden test must be re-baselined.
- Statistical correctness is unchanged: the null distribution coverage is equivalent.
- The change is equivalent to saying "we changed the random-number sequence" — which is acceptable
  for a performance optimization if documented.

**Verdict:** This change is acceptable (the p-value semantics are correct), but it re-baselines
existing test golden values. The planner MUST document this in the PLAN as a "output-change within
documented tolerance — re-baseline golden tests, not a bug."

#### Payback threshold rationale

- At n200_m50, 1.74 ms total for 999 perms → ≈1.7 µs per perm (very cheap per perm).
- Rayon overhead ≈1–5 µs per work-item.
- Payback is marginal at the n200_m50 cell. At n=1000 (extrapolating N-dominated scaling: ≈8.7 ms
  for 999 perms → ≈8.7 µs per perm), parallelism starts to pay.
- Recommend a HIGHER threshold than frechet_anova because per-perm work is smaller:

`const T_PERM_PARALLEL_THRESHOLD: usize = 500;`

This means: only parallelize when n_perm ≥ 500. At 500 perms × 8.7 µs = 4.35 ms total, dispatching
20 threads each ≈24 items × 8.7 µs = 210 µs each → ~10× speedup at n=1000.

#### Priority: MEDIUM (lower wall-time than frechet_anova/co_cluster; restructuring adds risk)

For the planner: implement the restructuring first as a standalone commit (changes output, must re-baseline tests), THEN add `iter_maybe_parallel!` in a second commit (adds parallelism on top of the restructured sequential code).

---

### CANDIDATE C: `co_cluster` n_init loop — SAFE (with `?`-restructure)

**Anchor:** `src/coclustering.rs:935–968`
**PROF-01 wall time:** 13.3 ms @ n100_m50 (N-dominated, n_init=3 default)
**PROF-01 rank:** #6

#### Current RNG/accumulation structure

[VERIFIED: src/coclustering.rs:935-968] Verbatim:

```rust
    for init in 0..n_init {
        let seed = config.seed.wrapping_add(init as u64 * 1000);

        // Row initialization via kmeans_fd
        use crate::clustering::kmeans_fd;
        let km = kmeans_fd(data, argvals, k_blocks, 100, 1e-4, seed)?;
        let init_row_labels = km.cluster;

        // Column initialization via k-means++ on argument-point profiles
        let init_col_labels = col_kmeans_init(data, n, m, l_blocks, seed.wrapping_add(1));

        let (result, _per_iter_ll) = cem_single_fit(
            data, rotation, mean, weights,
            init_row_labels, init_col_labels,
            n, m, k_blocks, l_blocks, eff_ncomp, config.max_iter, config.tol,
        );

        let is_better = best
            .as_ref()
            .map_or(true, |b| result.log_likelihood > b.log_likelihood);
        if is_better {
            best = Some(result);
        }
    }
```

**Key observations:**
1. Each `init` uses `config.seed.wrapping_add(init as u64 * 1000)` — a deterministic, per-init
   seed formula. This is EXACTLY the pattern required for bit-identical parallelism.
2. `cem_single_fit` is infallible (returns `(CoClusterResult, Vec<f64>)`, no `Result`).
3. The `kmeans_fd` call uses `?` — error propagation requires restructuring for parallel use.
4. The reduction (`is_better` → keep max log_likelihood) is a `max_by` operation over independent
   results — commutative and associative, bit-identical in any order (f64 `max` is deterministic).
5. `col_kmeans_init` is seeded as `seed.wrapping_add(1)` where `seed` is already per-init —
   so it is also fully deterministic per `init`.

#### Bit-identical-safe to parallelize: YES (after `?`-restructure)

The only blocker is the `kmeans_fd(...)? ` operator — `?` inside a rayon closure propagates out of
the closure (not the function), causing a compile error. The fix is:

1. Run `kmeans_fd` and collect `Result<Vec<usize>>` per init.
2. Collect all results via parallel map, propagating errors after.

#### Exact `iter_maybe_parallel!` rewrite

```rust
use crate::iter_maybe_parallel;

const CO_CLUSTER_INIT_PARALLEL_THRESHOLD: usize = 3;
// Payback threshold: n_init is typically 3 (default). Each init runs kmeans_fd
// + cem_single_fit, which costs ~4 ms each at n100_m50 → 3 inits = 13 ms total.
// Rayon overhead is negligible vs. 4 ms per work item, so parallelize at n_init >= 3.
// Even n_init=3 on 20 threads gives 3 parallel work-items → each thread gets 1 item.
// For n_init=10+ (users who increase restarts), speedup is near-linear.

let n_init = config.n_init.max(1);

if n_init >= CO_CLUSTER_INIT_PARALLEL_THRESHOLD {
    // Parallel path: each init is independent.
    // Step 1: kmeans_fd is fallible; collect Results in parallel, bail on first error.
    // Note: rayon has no direct parallel ? — collect into Vec<Result>, then check.
    let init_results: Vec<Result<CoClusterResult, FdarError>> =
        iter_maybe_parallel!(0..n_init)
            .map(|init| {
                use crate::clustering::kmeans_fd;
                let seed_init = config.seed.wrapping_add(init as u64 * 1000);
                let km = kmeans_fd(data, argvals, k_blocks, 100, 1e-4, seed_init)?;
                let init_row_labels = km.cluster;
                let init_col_labels =
                    col_kmeans_init(data, n, m, l_blocks, seed_init.wrapping_add(1));
                let (result, _) = cem_single_fit(
                    data, rotation, mean, weights,
                    init_row_labels, init_col_labels,
                    n, m, k_blocks, l_blocks, eff_ncomp, config.max_iter, config.tol,
                );
                Ok(result)
            })
            .collect();

    // Step 2: propagate any kmeans_fd error (fail-fast: return first Err).
    let mut best: Option<CoClusterResult> = None;
    for r in init_results {
        let result = r?;
        let is_better = best.as_ref().map_or(true, |b| result.log_likelihood > b.log_likelihood);
        if is_better { best = Some(result); }
    }
    best.ok_or_else(|| FdarError::ComputationFailed {
        operation: "co_cluster",
        detail: "all initializations failed".to_string(),
    })
} else {
    // Sequential path (identical to current code for n_init < threshold).
    let mut best: Option<CoClusterResult> = None;
    for init in 0..n_init {
        let seed_init = config.seed.wrapping_add(init as u64 * 1000);
        use crate::clustering::kmeans_fd;
        let km = kmeans_fd(data, argvals, k_blocks, 100, 1e-4, seed_init)?;
        let init_row_labels = km.cluster;
        let init_col_labels = col_kmeans_init(data, n, m, l_blocks, seed_init.wrapping_add(1));
        let (result, _) = cem_single_fit(
            data, rotation, mean, weights,
            init_row_labels, init_col_labels,
            n, m, k_blocks, l_blocks, eff_ncomp, config.max_iter, config.tol,
        );
        let is_better = best.as_ref().map_or(true, |b| result.log_likelihood > b.log_likelihood);
        if is_better { best = Some(result); }
    }
    best.ok_or_else(|| FdarError::ComputationFailed {
        operation: "co_cluster",
        detail: "all initializations failed".to_string(),
    })
}
```

**`Sync` requirements:** `data`, `rotation`, `mean`, `weights` are shared read-only references
(`&FdMatrix`, `&[f64]`) — `Sync` by construction (immutable borrows). `kmeans_fd` and
`cem_single_fit` take only `&FdMatrix` and `&[f64]` inputs → no interior mutability → safe in
rayon.

**Error-ordering note:** The `init_results` Vec preserves the order of init 0..n_init (rayon
`par_iter` collect preserves order). So the `best` selection after collecting is deterministic:
whichever init has the highest log_likelihood wins, with ties broken by the lowest `init` index
(first occurrence in sequential scan). This is **bit-identical to the sequential path** for any
thread count.

#### Payback threshold rationale

- n_init=3 default (13 ms total at n100_m50) → ≈4.3 ms per init.
- Rayon dispatch overhead: ~10µs (negligible vs. 4.3 ms per item).
- For n_init=3 on 20 cores: each core processes 1 item in ≈4.3 ms → parallel time ≈4.3 ms vs
  sequential 13 ms → ~3× speedup (limited by number of inits, not cores).
- For n_init=10: 10 items on 20 cores → ~5 items per 2 cores → ≈21.5 ms / 2 = ≈10.75 ms → ~2×.
- Threshold 3 is correct: even at n_init=3 the speedup matches parallelism count (3×), which
  clearly exceeds dispatch overhead.

`const CO_CLUSTER_INIT_PARALLEL_THRESHOLD: usize = 3;`

#### Proof of equivalence

- Existing `test_determinism_under_seed` (`src/coclustering.rs:1411`) runs two identical calls and
  asserts field-by-field equality including `row_labels`, `col_labels`, `log_likelihood` — this
  becomes the parallel-vs-sequential regression guard.
- Add to `tests/equivalence_phase48.rs`: same data, same config, assert `log_likelihood` and
  assignments bit-identical between parallel-ON and parallel-OFF build.

---

### CANDIDATE D: `explain/importance.rs` (`:131`, `:221`) — DEFER

**Anchor:** `src/explain/importance.rs:129–141` (`fpc_permutation_importance_linear`), `:219–235`
(`fpc_permutation_importance_logistic`)
**Reason for deferral:**

1. **Outer-k loop too narrow:** The outer loop is `for k in 0..ncomp` (typically 3–10 iterations).
   This is an insufficient fan-out for rayon to show benefit.
2. **Inner loop shares one advancing `rng`:** The inner `for _ in 0..n_perm` loop advances a single
   `StdRng` across all perms for component `k`. Not bit-identical to parallelize without per-perm
   reseeding.
3. **Already-parallel counterpart exists:** `src/explain_generic/importance.rs:64–80` is the
   generic version of this function and is ALREADY parallelized over `k` via `iter_maybe_parallel!`
   with per-k seeding (`StdRng::seed_from_u64(seed.wrapping_add(k as u64))`). The non-generic
   `explain/importance.rs` is a legacy path that CONS-02 (Phase 49) will unify into the generic.
4. **Priority:** Low wall-time, low fan-out, will be unified in Phase 49 anyway.

**DEFER to Phase 49 (CONS-02):** When the non-generic importance functions are consolidated into
the generic path, they automatically gain the parallelism already present in `explain_generic`.

---

## `parallel.rs` Macro Reference

[VERIFIED: src/parallel.rs:41-55] All 5 macros documented verbatim:

**`iter_maybe_parallel!(expr)`** — converts `expr` into a parallel/sequential iterator:
- `parallel` ON: `IntoParallelIterator::into_par_iter(expr)` (rayon `par_iter`)
- `parallel` OFF: `IntoIterator::into_iter(expr)` (standard `iter`)
- Fits: any `IntoParallelIterator` + `IntoIterator` type (ranges `0..n`, `Vec`, etc.)
- Returns: a rayon `ParallelIterator` or standard `Iterator` — supports `.map()`, `.sum()`,
  `.filter()`, `.collect()`, etc.
- **This is the correct macro for permutation-loop rewrites** (range `0..n_perm` or `0..n_init`).

**`slice_maybe_parallel!(slice)`** — calls `.par_iter()` or `.iter()` on a slice.
- Fits: when iterating over an existing `&[T]` slice.
- NOT suitable for owning range `0..n_perm` (use `iter_maybe_parallel!` instead).

**`slice_maybe_parallel_mut!(slice)`** — calls `.par_iter_mut()` or `.iter_mut()`.
- Fits: mutation of existing slice elements.
- NOT suitable here.

**`maybe_par_chunks_mut!(slice, chunk_size, closure)`** — chunks + `for_each`.
- NOT suitable for the permutation/init rewrites.

**`maybe_par_chunks_mut_enumerate!(slice, chunk_size, closure)`** — enumerated chunks + `for_each`.
- NOT suitable here.

**Closure signature for `iter_maybe_parallel!` + `.map()`:**
```rust
iter_maybe_parallel!(0..n_perm)
    .map(|perm: usize| -> usize {
        // Must be Send (auto for any type without Rc/raw ptrs).
        // Must capture only Send+Sync references (shared &FdMatrix is Sync).
        // Returns per-iteration contribution (0 or 1 for n_ge counting).
        ...
    })
    .sum::<usize>()
```

The rayon `.sum()` of `usize` values is exactly equivalent to sequential accumulation (addition is
commutative; usize has no rounding). No `Mutex` or `AtomicUsize` needed.

---

## Payback Threshold Pattern

[VERIFIED: src/elastic_fpca.rs:25-28] The precedent `SCORES_PARALLEL_THRESHOLD`:

```rust
/// multiply (`scores[(i,k)] = u[(i,k)] * sv`), so the overhead of spawning
/// rayon work-items exceeds the gain until N ≥ 50 (per the audit's
/// streaming-sentinel payback analysis, PERF-04-C).
const SCORES_PARALLEL_THRESHOLD: usize = 50;
```

[VERIFIED: src/elastic_fpca.rs:796-804] The outer-if guard shape:

```rust
    if n >= SCORES_PARALLEL_THRESHOLD {
        // Parallel branch: collect per-i values for each component, then assign.
        for k in 0..ncomp {
            let sv = svd.singular_values[k];
            let col: Vec<f64> = iter_maybe_parallel!(0..n).map(|i| u[(i, k)] * sv).collect();
            ...
        }
    } else {
        // Sequential branch: original nested loops for small N.
        ...
    }
```

**Follow this exact pattern:** `const *_PARALLEL_THRESHOLD: usize = N;` at module top, `if
work >= THRESHOLD { iter_maybe_parallel!(...) } else { sequential }`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Feature-gated parallel iterator | Custom thread pool, std::thread::spawn | `iter_maybe_parallel!(0..n)` | Already in `src/parallel.rs`; handles WASM fallback; tested |
| Parallel n_ge count accumulation | `AtomicUsize`, `Mutex<usize>` | `.map(... -> usize).sum()` | rayon `.sum()` is lock-free and correct; no shared mutation needed |
| Per-iteration RNG | Shared StdRng with Mutex | `StdRng::seed_from_u64(seed.wrapping_add(k as u64))` per iteration | The crate-wide determinism pattern; avoid any shared mutable RNG |
| Parallel `?` propagation | panic!, unwrap() | Collect `Vec<Result<_, _>>` then `.into_iter().collect::<Result<Vec<_>, _>>()?` | Rayon closures cannot use `?` directly; collect Results, then propagate sequentially |

---

## Parallel-OFF Build Verification

The parallel-OFF build command (no rayon, WASM-compatible):

```bash
TMPDIR=/home/simonm/.cache/fdars-bench-tmp \
  cargo test -p fdars-core --no-default-features --features linalg \
  -- --test-threads=4
```

**Feature flag reference (verified):**
- Default features: `["parallel"]` (rayon enabled by default)
- `linalg` feature: requires Rust ≥ 1.84 (faer)
- `--no-default-features --features linalg`: parallel OFF, linalg ON → correct test for parallel-OFF path

Equivalence test must explicitly be compiled with BOTH flag sets in CI to prove bit-identity.

The equivalence tests in `tests/equivalence_phase48.rs` use hardcoded golden values captured with
the restructured (per-perm-reseeded) sequential code. These golden values are independent of the
`parallel` feature flag — they must hold under both builds.

---

## Architecture Patterns

### Permutation Loop Rewrite Pattern

```
Current (shared-rng or already-per-perm) → Restructured (always per-perm) → Parallelized
```

For `frechet_anova`: already per-perm → go directly to parallelized.
For `t_perm_test`/`f_perm_test`: shared-rng → restructure to per-perm (output changes, re-baseline)
→ parallelize.
For `co_cluster`: already per-init seed → parallelize with `?`-restructure.

### Recommended Project Structure

No new directories. New files:
```
fdars-core/
├── benches/
│   └── perf_hotpaths.rs        # extend (add frechet_anova, t_perm_test, co_cluster benches)
├── tests/
│   └── equivalence_phase48.rs  # new (golden tests for all 3 candidates)
└── src/
    └── frechet/anova.rs        # add FRECHET_ANOVA_PERM_PARALLEL_THRESHOLD + rewrite loop
    └── inference/permutation.rs # restructure RNG + add T_PERM_PARALLEL_THRESHOLD + rewrite
    └── coclustering.rs         # add CO_CLUSTER_INIT_PARALLEL_THRESHOLD + rewrite loop
```

---

## Common Pitfalls

### Pitfall 1: Using `?` inside a rayon closure
**What goes wrong:** `? ` inside `.map(|...| { ... km = kmeans_fd(...)? ... })` causes a compile error — rayon's `ParallelIterator` items are not `Result`-aware.
**Why it happens:** The `?` operator requires the enclosing function's return type to be `Result`; the closure returns a concrete type.
**How to avoid:** Collect `Vec<Result<CoClusterResult, FdarError>>` from the parallel map, then fold sequentially with `?`. See the CO_CLUSTER rewrite skeleton above.
**Warning signs:** Compiler error "the `?` operator can only be used in a closure that returns `Result` or `Option`".

### Pitfall 2: Assuming `.sum::<usize>()` is non-deterministic
**What goes wrong:** Developers add a `Mutex<usize>` or `AtomicUsize` unnecessarily, which adds lock overhead and may incorrectly be non-order-dependent.
**Why it happens:** Confusion between non-deterministic floating-point reductions and integer counts.
**How to avoid:** Integer addition is exact and commutative; rayon's `.sum()` on `usize` is always bit-identical regardless of thread count. No synchronization primitive needed for the count.
**Warning signs:** Using `AtomicUsize::fetch_add` inside a parallel closure (correct, but unnecessary overhead vs. `.sum()`).

### Pitfall 3: Forgetting to add `use crate::iter_maybe_parallel;` in the target file
**What goes wrong:** The macro is defined in `src/parallel.rs` and re-exported, but each file that uses it needs `use crate::iter_maybe_parallel;` at the top. Without it, the macro is not in scope.
**Why it happens:** `anova.rs` currently does not import `iter_maybe_parallel!`.
**How to avoid:** Add `use crate::iter_maybe_parallel;` at the top of `src/frechet/anova.rs`, `src/inference/permutation.rs`, and `src/coclustering.rs`.
**Warning signs:** Compiler error "cannot find macro `iter_maybe_parallel` in this scope".

### Pitfall 4: `MetricSpace: Sync` bound missing for `frechet_anova_space`
**What goes wrong:** rayon requires captured references in the parallel closure to be `Send + Sync`. If `S: MetricSpace` does not have `S: Sync`, the parallel branch won't compile.
**Why it happens:** The generic `frechet_anova_space<S: MetricSpace>` signature does not currently constrain `S: Sync`.
**How to avoid:** Add `S: Sync` to the trait bound on `frechet_anova_space` when the `parallel` feature is enabled, or add it unconditionally (all current MetricSpace impls are Sync).
**Warning signs:** Compiler error "`S` cannot be shared between threads safely" or "trait bound `S: Sync` is not satisfied".

### Pitfall 5: Re-baselining golden tests for `t_perm_test`/`f_perm_test` without documentation
**What goes wrong:** The restructuring to per-perm seeding changes the specific p-value for a given seed. If the golden tests are updated without explanation, future reviewers think the output changed due to a bug.
**Why it happens:** The output IS different (different RNG draw sequence), but for a valid reason.
**How to avoid:** Add an explicit comment in `tests/equivalence_phase48.rs` and the commit message: "per-perm reseeding changes draw sequence → p-value numerical value changes; statistical correctness is preserved. Re-baselined golden values captured 2026-08-31."
**Warning signs:** CI fails on old golden values; re-baselining without comment.

### Pitfall 6: Threshold too low for `co_cluster` (false payback)
**What goes wrong:** With default `n_init=3` on a 20-core machine, the parallel path creates only 3 work-items for 20 threads — 17 threads sit idle. At n_init < 3, no parallelism is possible. Overhead per rayon dispatch: ≈10µs.
**Why it happens:** The payback threshold considers n_init < 3 as sequential.
**How to avoid:** `CO_CLUSTER_INIT_PARALLEL_THRESHOLD = 3` is the minimum meaningful parallel fan-out (3 items on 3+ threads). Users with `n_init=1` or `n_init=2` fall through to the sequential path.
**Warning signs:** Criterion reports the parallel path slower than sequential at n_init=2.

---

## Runtime State Inventory

This is a pure code-optimization phase — no renaming, no migration, no stored data. SKIPPED per execution-flow instructions (not a rename/refactor phase).

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | All compilation | ✓ | 1.97.0 | — |
| rayon 1.10 | `parallel` feature | ✓ | 1.10 (existing dep) | Sequential via `parallel.rs` macros |
| criterion 0.5 | Thread-scaling benches | ✓ | 0.5 (existing dev-dep) | — |
| cpupower | Governor pinning | ✓ | `/usr/bin/cpupower` (confirmed Phase 47) | Run under `powersave` with LOW-CONFIDENCE caveat |
| TMPDIR cache | Long bench builds | ✓ | `/home/simonm/.cache/fdars-bench-tmp` (confirmed Phase 47) | — |
| `linalg` feature | faer (needed for full suite) | ✓ | Rust 1.97 ≥ 1.84 (required) | Build without for non-Cholesky paths |
| RAYON_NUM_THREADS env var | Thread-count sweep in benches | ✓ | Runtime env var, no install needed | — |

**Missing dependencies with no fallback:** None.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + criterion 0.5 |
| Config file | `fdars-core/Cargo.toml` (`[[bench]]` entries) |
| Quick run command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |
| Parallel-OFF run command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --no-default-features --features linalg` |
| Full suite command | `cargo clippy --all-targets --features linalg,parallel -- -D warnings && cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PERF-03 | `frechet_anova` p-value bit-identical parallel vs sequential | golden equivalence | `cargo test -p fdars-core -- golden_frechet_anova_perm_parallel` | ❌ Wave 0 |
| PERF-03 | `frechet_anova` p-value bit-identical ON vs OFF feature | golden equivalence | run under both `--features linalg,parallel` and `--no-default-features --features linalg` | ❌ Wave 0 |
| PERF-03 | `t_perm_test` p-value bit-identical (per-perm-reseeded) | golden equivalence | `cargo test -p fdars-core -- golden_t_perm_test_parallel` | ❌ Wave 0 |
| PERF-03 | `f_perm_test` p-value bit-identical (per-perm-reseeded) | golden equivalence | `cargo test -p fdars-core -- golden_f_perm_test_parallel` | ❌ Wave 0 |
| PERF-03 | `co_cluster` log_likelihood and labels bit-identical parallel vs sequential | golden equivalence | `cargo test -p fdars-core -- golden_co_cluster_parallel` | ❌ Wave 0 |
| PERF-03 | `frechet_anova` thread-scaling shows ≥2× speedup at n_perm=999 | criterion bench | `RAYON_NUM_THREADS=1 cargo bench -p fdars-core --bench perf_parallelism -- frechet_anova` vs `RAYON_NUM_THREADS=20` | ❌ Wave 0 |
| PERF-03 | `co_cluster` thread-scaling shows speedup at n_init≥3 | criterion bench | `RAYON_NUM_THREADS=1 cargo bench ... -- co_cluster` vs `RAYON_NUM_THREADS=20` | ❌ Wave 0 |
| PERF-03 | Existing suite green at every commit | unit/integration | `cargo test -p fdars-core --features linalg,parallel` | ✅ |
| PERF-03 | Existing determinism tests still pass after restructuring | unit | `cargo test -p fdars-core -- t_perm_deterministic f_perm_deterministic anova_permutation_is_seed_reproducible test_determinism_under_seed` | ✅ (values re-baselined for perm tests) |

### ON/OFF Equivalence Proof Mechanics

The critical proof is: parallel-ON and parallel-OFF produce **bit-identical** output for the same
seed. This is verified by:

1. Writing a golden test that captures the numeric output (e.g., `p_value_permutation`) for a fixed
   input and seed using the SEQUENTIAL path (with `#[cfg(not(feature = "parallel"))]` or the
   below-threshold branch).
2. Running the test under both feature configs:
   ```bash
   # parallel ON
   cargo test -p fdars-core --features linalg,parallel -- golden_frechet_anova_perm_parallel
   # parallel OFF
   cargo test -p fdars-core --no-default-features --features linalg -- golden_frechet_anova_perm_parallel
   ```
3. Both must pass with the same golden value — proving the `iter_maybe_parallel!` macro's
   sequential fallback is numerically identical to the parallel branch.

This works because:
- `iter_maybe_parallel!` without the `parallel` feature compiles to a plain `into_iter()` → pure
  sequential behavior.
- The per-perm seeding ensures the same values are computed regardless of which branch runs.
- The `.sum::<usize>()` accumulator is exact for both paths.

### Thread-Scaling Bench Pattern

```rust
// In benches/perf_hotpaths.rs (extend existing file):
fn bench_frechet_anova_perm(c: &mut Criterion) {
    let mut group = c.benchmark_group("perf_frechet_anova_perm");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));
    let (resp, labels, argvals) = make_frechet_anova_data(24, 81); // n24, m81
    // Run under RAYON_NUM_THREADS=1 and RAYON_NUM_THREADS=20 via env var sweep.
    // Criterion will capture wall-time; compare 1-thread vs 20-thread runs manually.
    group.bench_function("n24_m81_nperm999", |b| {
        b.iter(|| frechet_anova(
            black_box(&resp), black_box(&argvals), black_box(&labels), 999, 42
        ))
    });
    group.finish();
}
```

Register as `[[bench]] name = "perf_parallelism" harness = false` — a new bench file separate from
`perf_hotpaths.rs` to isolate parallelism-specific timing. Or extend `perf_hotpaths.rs` with a
`bench_parallelism_candidates` group.

### Sampling Rate

- **Per task commit:** `TMPDIR=... cargo test -p fdars-core --features linalg,parallel`
- **Per wave merge:** Full suite + clippy: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- **Phase gate:** All golden tests pass under both feature configs; at least one criterion
  thread-scaling cell shows ≥2× speedup (parallel-ON vs 1-thread) at n_perm ≥ 500

### Wave 0 Gaps

- [ ] `tests/equivalence_phase48.rs` — golden tests for frechet_anova, t_perm_test, f_perm_test,
  co_cluster under both feature configs
- [ ] `benches/perf_parallelism.rs` (or extend `perf_hotpaths.rs`) — thread-scaling benches for
  frechet_anova and co_cluster at their PROF-01 cells
- [ ] `[[bench]] name = "perf_parallelism" harness = false` in `fdars-core/Cargo.toml` (if new file)

*(If extending `perf_hotpaths.rs`: no new Cargo.toml entry needed — bench already registered.)*

---

## Security Domain

Pure numerical-algorithm optimization — no authentication, I/O, or cryptography. SKIPPED.

---

## Ranked Implementation Sequence (for Planner)

| Priority | Candidate | Safety | Risk | Expected Speedup | Action |
|----------|-----------|--------|------|-----------------|--------|
| 1 | `frechet_anova` + `frechet_anova_space` | SAFE — already per-perm seeded | LOW | ~10–20× at n_perm=999 (133ms wall time, 20 cores) | Add `iter_maybe_parallel!` + threshold |
| 2 | `co_cluster` n_init | SAFE — already per-init seeded | LOW-MEDIUM (`?` restructure) | ~3× at n_init=3 (13ms / 3 inits) | Restructure `?`, add `iter_maybe_parallel!` + threshold |
| 3 | `t_perm_test` / `f_perm_test` | REQUIRES RNG RESTRUCTURE | MEDIUM (re-baselines golden values) | ~2–5× at n_perm=999 (1.74ms at n200; larger N needed for real payback) | Step 1: restructure to per-perm seed (output changes, re-baseline); Step 2: add `iter_maybe_parallel!` |
| DEFER | `explain/importance.rs` | DEFER (Phase 49 unification) | — | Negligible (ncomp-wide outer loop) | Document defer; Phase 49 CONS-02 gives it automatically |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Rayon work-item dispatch overhead ≈1–5µs on this machine | Payback threshold rationale | If overhead is higher (e.g., 50µs), thresholds should be raised; criterion will reveal this |
| A2 | All `MetricSpace` impls in the codebase are `Sync` (no interior mutability) | frechet_anova_space Sync bound | If a user-defined MetricSpace uses `RefCell`, adding `S: Sync` to the parallel branch is required |
| A3 | `cem_single_fit` is infallible and `Send` (no Rc, no non-Sync state) | co_cluster parallel sketch | If cem_single_fit borrows non-Send state, parallel branch won't compile; check before implementing |
| A4 | Rayon `.sum::<usize>()` is associative and produces bit-identical results to sequential accumulation | All three candidates | True by definition for integer addition; not a real risk |
| A5 | Per-perm reseeding in `t_perm_test`/`f_perm_test` preserves statistical validity | Candidate B analysis | True: each permutation is an independent uniform random relabeling; the null distribution is the same |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.

The A1 threshold assumption will be confirmed by criterion measurement; the planner should include a criterion before/after step for each candidate to validate the threshold choice.

---

## Sources

### Primary (HIGH confidence — files read this session)

- [VERIFIED: src/parallel.rs:41-55] — `iter_maybe_parallel!` macro: full body, both branches
- [VERIFIED: src/parallel.rs:62-73] — `slice_maybe_parallel!` macro
- [VERIFIED: src/parallel.rs:81-93] — `slice_maybe_parallel_mut!` macro
- [VERIFIED: src/parallel.rs:106-118] — `maybe_par_chunks_mut!` macro
- [VERIFIED: src/parallel.rs:131-148] — `maybe_par_chunks_mut_enumerate!` macro
- [VERIFIED: src/inference/permutation.rs:173-181] — `t_perm_test` loop: shared-rng, shared labels mutation
- [VERIFIED: src/inference/permutation.rs:236-244] — `f_perm_test` loop: shared-rng, shared groups mutation
- [VERIFIED: src/frechet/anova.rs:171-181] — `frechet_anova` loop: per-perm fresh RNG (`seed.wrapping_add(perm as u64)`)
- [VERIFIED: src/frechet/anova.rs:259-269] — `frechet_anova_space` loop: identical per-perm structure
- [VERIFIED: src/coclustering.rs:935-968] — `co_cluster` n_init loop: per-init seed `wrapping_add(init * 1000)`, `kmeans_fd` uses `?`, `cem_single_fit` infallible
- [VERIFIED: src/explain/importance.rs:125-141] — shared-rng advancing across `n_perm` inner loop; outer loop is `ncomp`-wide
- [VERIFIED: src/explain/importance.rs:215-235] — same structure for logistic variant
- [VERIFIED: src/explain_generic/importance.rs:64-80] — already parallel: `iter_maybe_parallel!(0..ncomp)` with per-k `StdRng::seed_from_u64(seed.wrapping_add(k as u64))`
- [VERIFIED: src/elastic_fpca.rs:25-28] — `SCORES_PARALLEL_THRESHOLD: usize = 50` const pattern
- [VERIFIED: src/elastic_fpca.rs:796-804] — outer-if threshold guard pattern
- [VERIFIED: tests/equivalence_phase47.rs:1-51] — Phase 47 equivalence test structure to mirror
- [VERIFIED: fdars-core/Cargo.toml:98-100] — `perf_hotpaths` bench already registered
- [VERIFIED: src/coclustering.rs:80-113] — `CoClusterResult` derives `Debug, Clone, PartialEq` (no non-Send/Sync fields)
- [VERIFIED: src/clustering.rs:545-552] — `kmeans_fd` signature: `Result<KmeansResult, FdarError>`

### Secondary (MEDIUM confidence — inferred from code structure)

- `famm.rs:874` uses shared advancing `StdRng` in its sequential permutation loop (not a Phase 48 target; already parallel via different loop in `famm`)
- `function_on_scalar.rs:836` uses LCG-based permutation (no `StdRng` — not a Phase 48 target)

### Tertiary (LOW confidence — not verified this session)

- Rayon work-item dispatch overhead estimate of 1–5µs [ASSUMED] — confirm with criterion measurement

---

## Metadata

**Confidence breakdown:**
- Candidate A (`frechet_anova`): HIGH — per-perm seeding read verbatim, rewrite is mechanical
- Candidate B (`t_perm_test`/`f_perm_test`): HIGH — shared-rng confirmed verbatim; restructuring requirement is certain
- Candidate C (`co_cluster`): HIGH — per-init seeding confirmed verbatim; `?`-restructure requirement confirmed
- Candidate D (defer): HIGH — outer-k width and shared-rng confirmed; generic parallel counterpart confirmed
- Payback thresholds: MEDIUM — estimated from PROF-01 timing; criterion will refine

**Research date:** 2026-08-31
**Valid until:** 2026-09-30 (stable Rust library; no external changes expected)
