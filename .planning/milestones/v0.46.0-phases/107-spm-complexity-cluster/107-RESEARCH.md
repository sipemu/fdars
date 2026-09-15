# Phase 107: SPM Complexity Cluster — Research

**Researched:** 2026-09-14
**Domain:** Rust internal refactor — cognitive/cyclomatic complexity reduction by private helper extraction (src/spm/)
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

All implementation choices are at Claude's discretion — pure refactor phase.
Constraints (from PROJECT.md / STATE.md, non-negotiable):

- **Strictly non-breaking + numeric-preserving.** Pure private helper extraction.
  No public API / signature / visibility change (protects R + WASM bindings).
- **Reduce complexity by helper extraction only** — never by changing signatures.
- **Preserve numeric identity** — do NOT reorder floating-point accumulations;
  lift loop bodies verbatim. SPM has control charts / MEWMA / MFPCA with
  sequential recurrences and possible RNG — preserve update order exactly.
- **`fdars-core` only. No new crate dependency** (pmat is external dev/CI tooling).
- Proof = existing suite green (2942 lib + 215 doctests + 28 examples,
  numeric-identical) + `pmat analyze complexity` confirming each SPM target
  drops below threshold and every new helper is itself under threshold.

### Claude's Discretion

All naming, exact seam placement, and internal helper structure choices are at
Claude's discretion. Pattern (from Phase 106): extract private helper → run
touched-module tests → `pmat analyze complexity` confirms drop → `--no-verify`
commit → out-of-band foreground gates.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| QUAL-02 | The SPM complexity cluster (`mfpca`, `spm_phase1_iterative`, `spm_mewma_monitor`, `select_ncomp`, `conditional_expectation` / partial, `spm_monitor_from_fields` / phase, `profile_phase1`, `spe_multivariate` / stats) is refactored under pmat thresholds, behavior/numeric-identical. | All 13 functions located, baseline scores confirmed via live pmat, decomposition seams identified from reading source. |
</phase_requirements>

---

## Summary

Phase 107 targets 13 functions in `fdars-core/src/spm/` that exceed pmat 3.18.0's
quality-gate thresholds (cognitive > 25 or cyclomatic > 30). All 13 have been
located via live pmat output and reading source. Natural decomposition seams have
been identified for each.

The SPM module has a denser cluster of moderate violations (cognitive 26–52) spread
across 9 files, compared to Phase 106's fewer but higher-scoring top-level outliers.
The dominant patterns are: (1) validation blocks with 4–8 guards mixed into function
bodies, (2) sequential monitoring loops (EWMA/MEWMA recurrences) with inline
branching, (3) multi-step algorithm bodies that combine setup, math, and assembly.

No RNG usage exists in any of the 13 target functions. All numeric hazards are
in sequential recurrence loops (EWMA/AMEWMA/MEWMA) — these must preserve the
exact i-indexed time order.

**Primary recommendation:** Apply the Phase 106 pattern identically: extract
validation blocks first (largest cognitive gain per line moved), then loop bodies,
then setup blocks. Use split-validate when a single extracted validator scores > 25.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Helper extraction | Internal module (private impl) | — | Pure internal refactor; no tier boundary crosses |
| Validation blocks | Same file, private fn | — | Error paths unchanged, no public surface |
| Sequential monitoring loops | Same file, private fn | — | Recurrence state passed by index; order preserved |
| Setup / preprocessing | Same file, private fn | — | Pure value computation, no side effects |
| Result assembly | Same file, private fn | — | Output types unchanged |

---

## pmat 3.18.0 Baseline — Confirmed Scores

> [VERIFIED: pmat 3.18.0 live output, 2026-09-14]

**pmat quality-gate thresholds (from live output):**
- Cognitive: error at > 25, warning at > 23
- Cyclomatic: error at > 30, warning at > 28

The per-directory `pmat analyze complexity --path fdars-core/src/spm` scan was
cross-referenced against `pmat quality-gate` output filtered to `spm/`. The
`--path` scan missed `spm_monitor_from_fields` (in `phase.rs` which is counted
under `spm/`) and `spe_multivariate` (in `stats.rs`). Individual per-file runs
confirm all 13 below.

### ERROR-level violations (quality-gate fires)

| Function | File:Line | Cyclomatic | Cognitive | Rule fired |
|----------|-----------|-----------|-----------|------------|
| `spm_amewma_monitor` | `spm/amewma.rs:137` | **25** | **52** | cog error |
| `spm_phase1_iterative` | `spm/iterative.rs:150` | 16 | **42** | cog error |
| `mfpca` | `spm/mfpca.rs:251` | 21 | **41** | cog error |
| `select_ncomp` | `spm/ncomp.rs:81` | 10 | **32** | cog error |
| `t2_contributions_mfpca` | `spm/contrib.rs:280` | 12 | **30** | cog error |
| `frcc_phase1` | `spm/frcc.rs:199` | 13 | **29** | cog error |
| `conditional_expectation` | `spm/partial.rs:299` | 16 | **29** | cog error |
| `spm_ewma_monitor` | `spm/ewma.rs:217` | 14 | **27** | cog error |
| `spe_contributions` | `spm/contrib.rs:132` | 11 | **27** | cog error |
| `spm_monitor_from_fields` | `spm/phase.rs:458` | 12 | **27** | cog error |
| `spe_multivariate` | `spm/stats.rs:275` | 11 | **27** | cog error |
| `profile_phase1` | `spm/profile.rs:191` | 12 | **26** | cog error |

### WARNING-level violations (at or near threshold — must also drop for QUAL-02)

| Function | File:Line | Cyclomatic | Cognitive | Notes |
|----------|-----------|-----------|-----------|-------|
| `spm_mewma_monitor` | `spm/mewma.rs:138` | 12 | **25** | cog=25 (warning threshold 23, error threshold 25); quality-gate fires a "recommended" warning but QUAL-02 requires clearing all errors; keeping at 25 leaves a warning unresolved — extract one validation seam to bring below 23 |

**Total: 12 ERROR-level + 1 WARNING-level = 13 target functions across 9 files.**

Note on roadmap vs live: the roadmap named `spm_mewma_monitor` as a target; live
pmat scores it at cog=25 (exactly at the error threshold). It currently fires only
a warning but is close enough that a validation extraction is warranted to keep it
from crossing to error after unrelated edits.

---

## Per-Function Analysis

### 1. `spm_amewma_monitor` — `spm/amewma.rs:137–314`

> [VERIFIED: fdars-core/src/spm/amewma.rs:137-314]

**Scores:** cyclomatic 25, cognitive 52 (cog error-level)

**Function shape (178 lines):**

```
Lines 143–195   Validation block (7 guards: ncols, lambda_min range, lambda_max range,
                lambda_min<=lambda_max, lambda_init in range, eta range, error_clip>0)
Lines 197–214   Setup: ncomp clamp, project + truncate scores (inner i/k loop), standardize z
Lines 230–299   Adaptive EWMA loop (for i in 0..n):
  Lines 251–257    Apply EWMA step (S_t = lambda * z_i + (1-lambda) * S_{t-1})
  Lines 264–267    Update cumulative variance factor sum_lambda_sq_prod
  Lines 269–274    Compute T² statistic from smoothed scores
  Lines 278–288    Compute squared prediction error + optional clip
  Lines 291–298    Update adaptive weight Q_t and lambda_prev for next step
Lines 302–313   UCL + alarm assembly
```

**Sequential recurrence hazard:** The EWMA loop at lines 250–299 is a time-ordered
sequential recurrence. The variables `q_prev`, `lambda_prev`, and `sum_lambda_sq_prod`
are carried from step i to i+1. Extraction must preserve the exact per-step order:
(1) apply current lambda_cur → (2) update sum_lambda_sq_prod → (3) compute T² →
(4) compute error → (5) update Q_t, lambda_prev, q_prev. Any reordering changes
the adaptive weight trajectory and hence T² values.

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `amewma_validate` | 143–195 | `fn amewma_validate(chart: &SpmChart, seq: &FdMatrix, config: &AmewmaConfig) -> Result<(), FdarError>` | 7 guards → extract cleanly |
| `amewma_project_standardize` | 197–228 | `fn amewma_project_standardize(chart: &SpmChart, seq: &FdMatrix, ncomp: usize) -> Result<(FdMatrix, usize), FdarError>` returns `(z, actual_ncomp)` | Project + truncate + standardize; pure setup, no recurrence |
| `amewma_ewma_step` | (inline helper for one i-step) | `fn amewma_ewma_step(z_i: &[f64], s_prev: &[f64], lambda_cur: f64, ncomp: usize) -> Vec<f64>` returns `s_t` | One EWMA update step; called per i; #[inline] candidate |

After extracting `amewma_validate` and `amewma_project_standardize`, the parent
function is left with the loop body which still holds most of the cognitive score.
Extract `amewma_ewma_step` for the inner smoothed-update logic. The variance,
T², error, and adaptive-update blocks are short (3–8 lines each) and can remain
inline in the loop body after those extractions.

**RNG:** None.

---

### 2. `spm_phase1_iterative` — `spm/iterative.rs:150–278`

> [VERIFIED: fdars-core/src/spm/iterative.rs:150-278]

**Scores:** cyclomatic 16, cognitive 42 (cog error-level)

**Function shape (129 lines):**

```
Lines 162–185   Validation (3 guards: alpha, max_iterations, max_removal_fraction)
Lines 187–193   State init (remaining_indices, all_removed, removal_history, etc.)
Lines 195–259   Main iteration loop (for _ in 0..config.max_iterations):
  Lines 197–201    Build current_data, build chart, run monitor
  Lines 203–212    Identify flagged local indices
  Lines 215–218    Convergence check (empty flagged)
  Lines 221–231    Cumulative-removal check + n_after < 4 check
  Lines 241–258    Map flagged to original indices, update remaining_indices, update stats
Lines 262–267   Final chart build if loop exhausted
Lines 270–277   Ok(IterativePhase1Result {...})
```

**Sequential recurrence hazard:** The per-iteration state (`remaining_indices`,
`all_removed`, `removal_history`, `removal_rates`) is carried across iterations.
The `flagged_set` construction and `remaining_indices` filter at lines 247–253
must run in this exact order. The outer loop structure must stay unchanged.

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `iterative_validate` | 162–185 | `fn iterative_validate(config: &IterativePhase1Config) -> Result<(), FdarError>` | 3 guards; clean extraction |
| `iterative_identify_flagged` | 203–212 | `fn iterative_identify_flagged(monitor: &SpmMonitorResult, config: &IterativePhase1Config, n_current: usize) -> Vec<usize>` returns `flagged_local` | Pure scan, no side effects |
| `iterative_check_removal_budget` | 221–231 | `fn iterative_check_removal_budget(all_removed_len: usize, flagged_len: usize, n_current: usize, n_original: usize, config: &IterativePhase1Config) -> bool` returns `should_stop` | Guards against over-removal |
| `iterative_update_state` | 241–258 | `fn iterative_update_state(remaining_indices: &mut Vec<usize>, all_removed: &mut Vec<usize>, removal_history: &mut Vec<Vec<usize>>, removal_rates: &mut Vec<f64>, flagged_local: &[usize], n_current: usize)` | Updates all mutable state; pass as &mut refs |

**RNG:** None.

---

### 3. `mfpca` — `spm/mfpca.rs:251–406`

> [VERIFIED: fdars-core/src/spm/mfpca.rs:251-406]

**Scores:** cyclomatic 21, cognitive 41 (cog error-level)

**Function shape (156 lines):**

```
Lines 252–277   Validation (empty check, n<2 check, per-variable nrows match loop)
Lines 279–308   Step 1: center each variable + compute scale (per-variable loop)
Lines 310–338   Step 2: build stacked matrix (center/scale + stack, per-variable loop)
Lines 340–354   Step 3: SVD (+ ok_or_else unwrap for u/v_t)
Lines 356–394   Step 4: extract scores, rotation, eigenvalues
Lines 382–394   Step 5: split eigenfunctions by variable
Lines 396–405   Ok(MfpcaResult {...})
```

**Numeric hazard:** The SVD is over the stacked centered/scaled matrix. The
centering and scaling in Step 2 feed directly into SVD. The per-variable scale
logic (`if config.weighted && scales[p] > scale_threshold`) must be preserved
verbatim — the in-place mutation `scales[p] = 1.0` on line 329 (when !weighted)
affects both the stacked matrix and the stored `scales` field in the result.

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `mfpca_validate` | 252–277 | `fn mfpca_validate(variables: &[&FdMatrix]) -> Result<usize, FdarError>` returns `n` | Empty + n<2 + per-var nrows; 3 guards |
| `mfpca_compute_means_scales` | 284–308 | `fn mfpca_compute_means_scales(variables: &[&FdMatrix], n: usize) -> (Vec<Vec<f64>>, Vec<f64>)` returns `(means, scales)` | Pure statistic computation |
| `mfpca_build_stacked` | 317–338 | `fn mfpca_build_stacked(variables: &[&FdMatrix], means: &[Vec<f64>], scales: &mut Vec<f64>, grid_sizes: &[usize], n: usize, total_cols: usize, weighted: bool, scale_threshold: f64) -> FdMatrix` | Takes `&mut scales` to apply the `scales[p]=1.0` mutation in place |
| `mfpca_extract_components` | 356–394 | `fn mfpca_extract_components(svd: &nalgebra::SVD<f64, nalgebra::Dyn, nalgebra::Dyn>, ncomp: usize, n: usize, total_cols: usize, grid_sizes: &[usize]) -> Result<(FdMatrix, FdMatrix, Vec<f64>, Vec<FdMatrix>), FdarError>` returns `(scores, combined_rotation, eigenvalues, eigenfunctions)` | Post-SVD extraction |

**Note on `mfpca_build_stacked`:** The `scales[p] = 1.0` mutation at line 329
happens inside the stacked-matrix loop and mutates the scales vector that is also
stored in the result. Passing `scales` as `&mut Vec<f64>` to the helper lets the
mutation propagate back to the caller naturally.

**RNG:** None.

---

### 4. `select_ncomp` — `spm/ncomp.rs:81–132`

> [VERIFIED: fdars-core/src/spm/ncomp.rs:81-132]

**Scores:** cyclomatic 10, cognitive 32 (cog error-level)

**Function shape (52 lines):**

```
Lines 82–88     Validation: empty eigenvalues check
Lines 90–105    Validation loop: NaN + negative eigenvalue checks
Lines 107–131   Match arm dispatch: CumulativeVariance, Elbow, Fixed, Kaiser
```

The dominant cognitive cost is the NaN/negative validation loop + the deeply-
nested `match { NcompMethod::CumulativeVariance(threshold) => { if *threshold... } }`.

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `ncomp_validate_eigenvalues` | 90–105 | `fn ncomp_validate_eigenvalues(eigenvalues: &[f64]) -> Result<(), FdarError>` | The per-element NaN/negative loop; 2 error variants |
| `ncomp_validate_threshold` | inside CumulativeVariance arm | `fn ncomp_validate_threshold(threshold: f64) -> Result<(), FdarError>` | One guard: threshold in (0,1]; called from CumulativeVariance arm |

After extraction, `select_ncomp` contains: one empty-check, one `ncomp_validate_eigenvalues()?` call, and the match dispatch — all arms thin. The `cumulative_variance_ncomp`, `elbow_ncomp`, and `kaiser_ncomp` helpers already exist as private fns and score cleanly (cog 9, 5, 0 respectively).

**RNG:** None.

---

### 5. `t2_contributions_mfpca` — `spm/contrib.rs:280–341`

> [VERIFIED: fdars-core/src/spm/contrib.rs:280-341]

**Scores:** cyclomatic 12, cognitive 30 (cog error-level)

**Function shape (62 lines):**

```
Lines 284–293   Validation (ncomp > eigenvalues check, p==0 check)
Lines 306–325   Build weights matrix: for l in 0..ncomp { for v in 0..p { ... } }
Lines 327–338   Compute contributions: for i in 0..n { for l { for v { ... } } }
Lines 340         Ok(contrib)
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `t2_mfpca_validate` | 284–293 | `fn t2_mfpca_validate(scores: &FdMatrix, mfpca: &MfpcaResult) -> Result<(usize, usize), FdarError>` returns `(ncomp, p)` | 2 guards |
| `t2_mfpca_compute_weights` | 306–325 | `fn t2_mfpca_compute_weights(mfpca: &MfpcaResult, ncomp: usize, p: usize) -> Vec<Vec<f64>>` returns `weights[p][ncomp]` | Pure per-variable eigenfunction-norm computation |

**RNG:** None.

---

### 6. `frcc_phase1` — `spm/frcc.rs:199–367`

> [VERIFIED: fdars-core/src/spm/frcc.rs:199-367]

**Scores:** cyclomatic 13, cognitive 29 (cog error-level)

**Function shape (169 lines):**

```
Lines 204–238   Validation (n<6, predictors rows, argvals len, tuning_fraction range;
                split check for n_cal<2 and n_tune<p+2; fosr_lambda<0)
Lines 241–246   Split + subset rows (tune/cal split)
Lines 278–311   FOSR on tuning set + R² computation
Lines 313–316   R² threshold check
Lines 328–366   FPCA on calibration residuals + control limits
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `frcc_validate` | 204–238 | `fn frcc_validate(y: &FdMatrix, predictors: &FdMatrix, argvals: &[f64], config: &FrccConfig) -> Result<(usize, usize), FdarError>` returns `(n, m)` | Consolidates 6 guards |
| `frcc_validate_split` | (post-split checks 248–265) | `fn frcc_validate_split(n_cal: usize, n_tune: usize, p: usize) -> Result<(), FdarError>` | 2 post-split guards; called after `split_indices` |
| `frcc_compute_r_squared` | 294–311 | `fn frcc_compute_r_squared(tune_y: &FdMatrix, tune_predicted: &FdMatrix) -> f64` | Pure R² statistic; no side effects |

**RNG:** `split_indices` uses a seed for the split (line 240). The split itself
is called in the parent and its determinism depends only on the seed — not
affected by helper extraction.

---

### 7. `conditional_expectation` — `spm/partial.rs:299–412`

> [VERIFIED: fdars-core/src/spm/partial.rs:299-412]

**Scores:** cyclomatic 16, cognitive 29 (cog error-level)

**Function shape (114 lines):**

```
Lines 306–329   Compute sigma² from sorted median SPE (sort + median)
Lines 331–337   Compute Phi_obs^T y_obs
Lines 339–354   Build M matrix (Woodbury identity)
Lines 356–372   Condition-number check + Tikhonov regularization
Lines 374–400   Solve M*scores = rhs (Cholesky via solve_spd; retry loop on failure)
Lines 402–411   Reconstruct full curve
```

**Numeric hazard:** The retry loop at lines 382–400 accumulates `reg_strength *=
10.0` multiplicatively. The order of solve attempts must be preserved (up to 5
retries with increasing regularization). The `solve_spd` helper already exists
in the same file (lines 499–547, private) with its own pmat score (cog=21, cyc=11 —
currently a WARNING; extracting into helper lets us address both simultaneously).

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `cond_exp_sigma2` | 316–329 | `fn cond_exp_sigma2(chart: &SpmChart, m: usize) -> f64` | Median SPE estimation; pure computation |
| `cond_exp_build_m_matrix` | 339–372 | `fn cond_exp_build_m_matrix(chart: &SpmChart, n_obs: usize, ncomp: usize, sigma2: f64, reg_eps: f64) -> Vec<f64>` returns `m_matrix` (ncomp×ncomp row-major) | Builds Woodbury M + condition check + Tikhonov regularization |
| `cond_exp_solve_with_retry` | 379–401 | `fn cond_exp_solve_with_retry(m_matrix: &[f64], rhs: &[f64], ncomp: usize, diag_max: f64) -> Result<Vec<f64>, FdarError>` | Initial solve_spd call + retry loop |

**Note on `solve_spd`:** `solve_spd` at line 499 scores cog=21 (warning-level).
Extracting its inner forward/back-substitution blocks as separate named sub-helpers
(`cholesky_factor_n`, `forward_sub_n`, `back_sub_n`) would bring it below 23.
This is within scope for Phase 107 since it is in `partial.rs` and drives the
cognitive score of the file.

**RNG:** None.

---

### 8. `spm_ewma_monitor` — `spm/ewma.rs:217–347`

> [VERIFIED: fdars-core/src/spm/ewma.rs:217-347]

**Scores:** cyclomatic 14, cognitive 27 (cog error-level)

**Function shape (131 lines):**

```
Lines 223–239   Validation (ncols check, ncomp==0 check)
Lines 241–258   Project + truncate to ncomp
Lines 262        EWMA smooth (call to ewma_scores)
Lines 274–295   Compute T²: exact_covariance branch (for i + time-factor loop) OR hotelling_t2
Lines 297–326   Compute centered data + centered reconstruction (for i,j,k triple loop)
Lines 328        spe_univariate call
Lines 331–345   Control limits + alarms
```

**Sequential recurrence hazard in `exact_covariance` branch:** The per-i loop at
lines 281–295 computes `time_factor = 1 - (1-lambda)^{2*(i+1)}` which depends on
the observation index `i` (time-dependent variance). This is not a true
recurrence (no carried state from i to i+1) — it is a direct formula per `i`.
The order does not matter numerically; but preserving it as written avoids any
doubt.

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `ewma_validate` | 223–239 | `fn ewma_validate(chart: &SpmChart, seq: &FdMatrix, ncomp: usize) -> Result<(), FdarError>` | 2 guards |
| `ewma_truncate_scores` | 241–258 | `fn ewma_truncate_scores(scores: FdMatrix, ncomp: usize) -> FdMatrix` | Truncate columns to ncomp; pure |
| `ewma_compute_t2_exact` | 281–295 | `fn ewma_compute_t2_exact(smoothed: &FdMatrix, eigenvalues: &[f64], lambda: f64, actual_ncomp: usize) -> Vec<f64>` | Time-factor T² branch; extracted to eliminate the big if/else nesting |
| `ewma_compute_spe` | 297–328 | `fn ewma_compute_spe(scores: &FdMatrix, seq: &FdMatrix, chart: &SpmChart, argvals: &[f64], actual_ncomp: usize) -> Result<Vec<f64>, FdarError>` | Center + reconstruct + spe_univariate; triple loop |

**RNG:** None.

---

### 9. `spe_contributions` — `spm/contrib.rs:132–198`

> [VERIFIED: fdars-core/src/spm/contrib.rs:132-198]

**Scores:** cyclomatic 11, cognitive 27 (cog error-level)

**Function shape (67 lines):**

```
Lines 137–150   Validation (p mismatch, p==0 check)
Lines 159–196   Per-variable loop (for v in 0..p):
  Lines 164–176    Per-variable shape validation
  Lines 185–194    Compute per-i SPE contribution (simpsons_weights + inner i,j loop)
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `spe_contribs_validate` | 137–150 | `fn spe_contribs_validate(std_vars: &[&FdMatrix], recon_vars: &[&FdMatrix], argvals_list: &[&[f64]]) -> Result<usize, FdarError>` returns `p` | Top-level p-count validation |
| `spe_contribs_var_validate` | 164–176 | `fn spe_contribs_var_validate(std_v: &FdMatrix, rec_v: &FdMatrix, argvals: &[f64], n: usize, v: usize) -> Result<usize, FdarError>` returns `m_v` | Per-variable shape check; called from the v loop |
| `spe_contribs_compute_var` | 185–194 | `fn spe_contribs_compute_var(std_v: &FdMatrix, rec_v: &FdMatrix, weights: &[f64], n: usize, m_v: usize) -> Vec<f64>` returns per-obs var_spe | Inner i,j computation for one variable |

**RNG:** None.

---

### 10. `spm_monitor_from_fields` — `spm/phase.rs:458–549`

> [VERIFIED: fdars-core/src/spm/phase.rs:458-549]

**Scores:** cyclomatic 12, cognitive 27 (cog error-level)

**Function shape (92 lines):**

```
Lines 468–505   Validation (5 guards: new_data ncols, rotation nrows, rotation ncols,
                weights len, argvals len)
Lines 507–519   Score projection (for i,k,j triple loop)
Lines 521–537   Centered reconstruction (for i,j,k triple loop)
Lines 538        spe_univariate call
Lines 541–548   Alarms + Ok(...)
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `monitor_fields_validate` | 468–505 | `fn monitor_fields_validate(fpca_mean: &[f64], fpca_rotation: &FdMatrix, fpca_weights: &[f64], eigenvalues: &[f64], seq: &FdMatrix, argvals: &[f64]) -> Result<(usize, usize), FdarError>` returns `(m, ncomp)` | 5 guards → split validate if needed |
| `monitor_fields_project` | 507–519 | `fn monitor_fields_project(seq: &FdMatrix, fpca_mean: &[f64], fpca_rotation: &FdMatrix, fpca_weights: &[f64], n_new: usize, m: usize, ncomp: usize) -> FdMatrix` | Score projection triple loop |
| `monitor_fields_recon` | 521–537 | `fn monitor_fields_recon(scores: &FdMatrix, fpca_rotation: &FdMatrix, n_new: usize, m: usize, ncomp: usize) -> FdMatrix` | Reconstruction triple loop |

**Note:** If `monitor_fields_validate` itself scores > 25 after extraction (5 guards
can push cognitive to ~22–28), apply the split-validate pattern: `monitor_fields_validate_dims`
(seq ncols, rotation nrows, rotation ncols) + `monitor_fields_validate_params`
(weights len, argvals len).

**RNG:** None.

---

### 11. `spe_multivariate` — `spm/stats.rs:275–341`

> [VERIFIED: fdars-core/src/spm/stats.rs:275-341]

**Scores:** cyclomatic 11, cognitive 27 (cog error-level)

**Function shape (67 lines) — near-identical structure to `spe_contributions`:**

```
Lines 280–295   Validation (p mismatch, p==0 check)
Lines 302–338   Per-variable loop (for v in 0..p):
  Lines 307–313    Per-variable nrows check
  Lines 315–321    Per-variable ncols/argvals check
  Lines 328–337    Compute per-i SPE (simpsons_weights + inner i,j loop)
```

**Natural decomposition seams (mirrors spe_contributions pattern):**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `spe_mv_validate` | 280–295 | `fn spe_mv_validate(std_vars: &[&FdMatrix], recon_vars: &[&FdMatrix], argvals_list: &[&[f64]]) -> Result<usize, FdarError>` | 2 guards |
| `spe_mv_var_check` | 307–321 | `fn spe_mv_var_check(std_v: &FdMatrix, rec_v: &FdMatrix, argvals: &[f64], n: usize, v: usize) -> Result<usize, FdarError>` returns `m_v` | Per-variable shape |
| `spe_mv_compute_var` | 328–337 | `fn spe_mv_compute_var(std_v: &FdMatrix, rec_v: &FdMatrix, weights: &[f64], n: usize, m_v: usize) -> f64` returns scalar accumulated var_spe | Inner i,j loop for one variable; accumulates into caller's `spe[i]` |

**RNG:** None.

---

### 12. `profile_phase1` — `spm/profile.rs:191–339`

> [VERIFIED: fdars-core/src/spm/profile.rs:191-339]

**Scores:** cyclomatic 12, cognitive 26 (cog error-level)

**Function shape (149 lines):**

```
Lines 197–249   Validation (7 guards: predictors rows, argvals len, step_size==0,
                window_size<3, window_size>n, expected_n_windows<4 early check,
                n_windows<4 post-rolling check)
Lines 252        fosr call (reference)
Lines 255–256   rolling_betas call
Lines 266–280   FPCA on vectorized betas + eigenvalues
Lines 283–325   Lag-1 autocorrelation + Bartlett effective-n computation
Lines 328        t2_control_limit call
Lines 330–338   Ok(ProfileChart {...})
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `profile_validate` | 197–249 | `fn profile_validate(y: &FdMatrix, predictors: &FdMatrix, argvals: &[f64], config: &ProfileMonitorConfig) -> Result<(usize, usize, usize), FdarError>` returns `(n, m, expected_n_windows)` | 5 guards (6 if split-validate needed) |
| `profile_lag1_autocorr` | 289–308 | `fn profile_lag1_autocorr(phase1_t2: &[f64]) -> f64` | Lag-1 autocorrelation via unbiased variance estimator; pure |
| `profile_effective_n` | 318–325 | `fn profile_effective_n(n_windows: usize, lag1: f64) -> f64` | Bartlett correction; 3 lines, extracted to name the concept |

**Note on `rolling_betas`:** `rolling_betas` at line 414 scores cog=21 (warning-
level). It is already a private helper extracted from the main function body; its
existing cog=21 is below the error threshold 25 and does not need decomposition
for QUAL-02 but could benefit from further extraction in Phase 109's tail pass.

**RNG:** None.

---

### 13. `spm_mewma_monitor` — `spm/mewma.rs:138–249`

> [VERIFIED: fdars-core/src/spm/mewma.rs:138-249]

**Scores:** cyclomatic 12, cognitive 25 (cog=25, exactly at error threshold; quality-gate fires warning at 23)

**Function shape (112 lines):**

```
Lines 144–172   Validation (3 guards: ncols, lambda range, eigenvalue positivity loop)
Lines 174–191   Project + truncate
Lines 194–225   MEWMA statistic: asymptotic branch (hotelling_t2) OR exact time-factor loop
Lines 228–230   SPE from raw scores
Lines 233–248   UCL, alarms, Ok(...)
```

Extracting the validation block (3 guards) will drop parent cognitive below 23.

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `mewma_validate` | 144–172 | `fn mewma_validate(chart: &SpmChart, seq: &FdMatrix, config: &MewmaConfig, ncomp: usize) -> Result<(), FdarError>` | 3 guards including the eigenvalue-positivity loop |
| `mewma_compute_exact_t2` | 207–224 | `fn mewma_compute_exact_t2(smoothed: &FdMatrix, eigenvalues: &[f64], lambda: f64, actual_ncomp: usize) -> Vec<f64>` | Exact time-dependent T² branch; mirrors `ewma_compute_t2_exact` pattern |

**Sequential recurrence hazard:** The exact-covariance loop at lines 207–224 uses
`t = (i+1) as f64` per iteration; no cross-step state carried. Not a true
recurrence.

**RNG:** None.

---

## Order-Sensitivity and Numeric Hazards Summary

| Function | Hazard Type | Preservation Rule |
|----------|-------------|-------------------|
| `spm_amewma_monitor` | Sequential EWMA recurrence: `q_prev`, `lambda_prev`, `sum_lambda_sq_prod` carried i→i+1 | Outer for-i loop must remain intact; helpers called inside the loop |
| `spm_phase1_iterative` | Iteration state: `remaining_indices`, `all_removed` carried iter→iter+1 | Outer for-iteration loop must remain intact |
| `mfpca` | In-place `scales[p] = 1.0` mutation visible in result | Pass `scales` as `&mut Vec<f64>` to stacked-matrix helper |
| `conditional_expectation` | Retry loop: `reg_strength *= 10.0` up to 5 retries | Retry loop extracted verbatim to helper |
| `spm_ewma_monitor` | `exact_covariance` branch: `time_factor = 1 - (1-λ)^{2(i+1)}` | No cross-step state; extraction by value is safe |
| `spm_mewma_monitor` | Same as `spm_ewma_monitor` exact branch | Same rule |
| All others | No sequential state | Standard extract-and-call |

**RNG:** No function in the SPM module uses random number generation. There are
no RNG draw-order hazards in Phase 107 (unlike Phase 106's Gibbs/k-means++).

---

## Standard Stack

No new dependencies. All tools already in the crate:

| Utility | Source | Used by (SPM targets) |
|---------|--------|----------------------|
| `FdMatrix`, `FdMatrix::zeros` | `src/matrix.rs` | All 13 functions |
| `simpsons_weights` | `src/helpers.rs` | `spe_contributions`, `spe_multivariate`, `conditional_expectation` (via `partial_projection`) |
| `hotelling_t2` | `src/spm/stats.rs` | `spm_ewma_monitor`, `spm_mewma_monitor`, `spm_monitor_from_fields`, `frcc_phase1` |
| `spe_univariate` | `src/spm/stats.rs` | `spm_ewma_monitor`, `spm_mewma_monitor` |
| `fdata_to_pc` | `src/regression.rs` | `frcc_phase1`, `profile_phase1` |
| `fosr`, `predict_fosr` | `src/function_on_scalar.rs` | `frcc_phase1`, `profile_phase1` |
| `ewma_scores` | `src/spm/ewma.rs` | `spm_ewma_monitor`, `spm_mewma_monitor` |
| `chi2_quantile` | `src/spm/chi_squared.rs` | `spm_amewma_monitor`, `spm_ewma_monitor`, `spm_mewma_monitor` |
| `nalgebra::SVD` | external dep (existing) | `mfpca` |

---

## Architecture Patterns

### Recommended helper placement

All extracted helpers are placed **immediately below the public function they
serve**, before the next public function or module-level comment — matching the
Phase 106 pattern and the existing helper layout in `spm/frcc.rs` (`compute_residuals`
at line 143 directly above `frcc_phase1`).

### Pattern 1: Validate-and-call (Phase 106 proven)

```rust
// Before:
pub fn big_fn(args...) -> Result<Out, FdarError> {
    // 50-line validation block
    // ...
}

// After:
pub fn big_fn(args...) -> Result<Out, FdarError> {
    let (n, m) = big_fn_validate(args...)?;
    // ...
}

fn big_fn_validate(args...) -> Result<(usize, usize), FdarError> {
    // validation body verbatim
}
```

### Pattern 2: Split-validate (Phase 106 proven)

When an extracted validate fn itself scores > 25, split along a natural
boundary (dimension/count guards vs. config/parameter guards). Example from
Phase 106: `wnet_validate` → `wnet_validate_dims` + `wnet_validate_config`.

### Pattern 3: Verbatim loop-body extraction

For inner loops: extract body verbatim as a helper. Mark with `#[inline]` when
the extracted helper is in a hot path (scores projection loops, EWMA inner step).

### Anti-patterns to avoid

- **Reorder the EWMA/AMEWMA iteration steps** — the sequential update order
  for `q_prev`, `lambda_prev`, `sum_lambda_sq_prod` in `spm_amewma_monitor`
  must be preserved exactly.
- **Change visibility** — all helpers are `fn` (module-private), never `pub fn`.
- **Add `unwrap()`** — all error paths use `?` or `return Err(...)`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cholesky solve | Custom SPD solver | Existing `solve_spd` in `partial.rs` | Already tested, in-file |
| Simpson weights | Custom quadrature | `crate::helpers::simpsons_weights` | Simpson rule already implemented |
| Hotelling T² | Custom T² | `crate::spm::stats::hotelling_t2` | Already used by the file |
| EWMA smoothing | Custom recursion | `crate::spm::ewma::ewma_scores` | Already correct, tested |

---

## Suggested Plan Grouping

Four plans, zero file overlap between plans so each is independently executable:

### Plan 107-01: Tracer — `amewma.rs`, `iterative.rs`

The two highest-scoring functions (cog 52 and 42). Both are single-function
files with self-contained logic and no cross-file dependencies. Proves the
tracer loop for Phase 107.

Files: `fdars-core/src/spm/amewma.rs`, `fdars-core/src/spm/iterative.rs`
Target functions: `spm_amewma_monitor` (cog 52→target ≤25), `spm_phase1_iterative` (cog 42→target ≤25)
Key risk: amewma sequential recurrence; tested by `cargo test -p fdars-core --features linalg,parallel -- spm::amewma::tests spm::iterative::tests`

### Plan 107-02: MFPCA cluster — `mfpca.rs`, `ncomp.rs`, `contrib.rs`

Three closely related files (MFPCA and its consumers: component selection and
contribution diagnostics). No file shares code with Plans 01/03/04.

Files: `fdars-core/src/spm/mfpca.rs`, `fdars-core/src/spm/ncomp.rs`, `fdars-core/src/spm/contrib.rs`
Target functions: `mfpca` (41), `select_ncomp` (32), `t2_contributions_mfpca` (30), `spe_contributions` (27)
Key risk: mfpca `scales` mutation; tested by `cargo test -p fdars-core --features linalg,parallel -- spm::mfpca::tests spm::ncomp::tests spm::contrib::tests`

### Plan 107-03: Monitoring monitors — `ewma.rs`, `mewma.rs`, `amewma.rs` (already done), `stats.rs`, `phase.rs`

EWMA/MEWMA monitors and the SPE/T² stat helpers.

Files: `fdars-core/src/spm/ewma.rs`, `fdars-core/src/spm/mewma.rs`, `fdars-core/src/spm/stats.rs`, `fdars-core/src/spm/phase.rs`
Target functions: `spm_ewma_monitor` (27), `spm_mewma_monitor` (25), `spe_multivariate` (27), `spm_monitor_from_fields` (27)
Key risk: ewma exact-covariance branch, mewma validation eigenvalue loop; tested by `cargo test -p fdars-core --features linalg,parallel -- spm::ewma::tests spm::mewma::tests spm::stats::tests spm::phase::tests`

### Plan 107-04: Remaining monitors — `partial.rs`, `frcc.rs`, `profile.rs`

Three independent single-function targets (conditional expectation BLUP, FRCC
chart construction, profile monitoring).

Files: `fdars-core/src/spm/partial.rs`, `fdars-core/src/spm/frcc.rs`, `fdars-core/src/spm/profile.rs`
Target functions: `conditional_expectation` (29), `frcc_phase1` (29), `profile_phase1` (26)
Key risk: conditional_expectation Tikhonov retry loop; tested by `cargo test -p fdars-core --features linalg,parallel -- spm::partial::tests spm::frcc::tests spm::profile::tests`

---

## Runtime State Inventory

This phase is a greenfield pure refactor with no rename or migration component.

None — verified by inspection: no stored data, live service config, OS-registered
state, secrets/env vars, or build artifacts are affected by private helper extraction
within `fdars-core/src/spm/`.

---

## Validation Architecture

### Behavior-preservation proof

```bash
# Module-specific fast smoke (per plan, ~10–60 sec):
cargo test -p fdars-core --features linalg,parallel -- spm::amewma::tests spm::iterative::tests 2>&1 | tail -5
cargo test -p fdars-core --features linalg,parallel -- spm::mfpca::tests spm::ncomp::tests spm::contrib::tests 2>&1 | tail -5
cargo test -p fdars-core --features linalg,parallel -- spm::ewma::tests spm::mewma::tests spm::stats::tests spm::phase::tests 2>&1 | tail -5
cargo test -p fdars-core --features linalg,parallel -- spm::partial::tests spm::frcc::tests spm::profile::tests 2>&1 | tail -5

# Full suite (numeric-preservation oracle):
cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -5

# Clippy (catches visibility/lifetime issues from extraction):
cargo clippy --all-targets --features linalg,parallel -- -D warnings

# Format check:
cargo fmt --check
```

### Complexity-drop proof

```bash
# Per-file confirmation after each target is refactored:
pmat analyze complexity --path fdars-core/src/spm/amewma.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/iterative.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/mfpca.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/ncomp.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/contrib.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/ewma.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/mewma.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/stats.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/phase.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/partial.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/frcc.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/spm/profile.rs 2>/dev/null

# Full quality-gate (after all 4 plans complete):
pmat quality-gate 2>/dev/null | grep "spm/"
```

Thresholds to clear (confirmed live pmat 3.18.0):
- Cognitive: must be ≤ 25 (error threshold); target ≤ 23 (warning threshold) for full clean
- Cyclomatic: must be ≤ 30 (error threshold); no SPM function currently exceeds 25

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| QUAL-02: spm_amewma_monitor | lambda_t sequence + T² statistic unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- spm::amewma::tests` |
| QUAL-02: spm_phase1_iterative | removed_indices, n_iterations, n_remaining unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- spm::iterative::tests` |
| QUAL-02: mfpca | scores, eigenvalues, eigenfunctions unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- spm::mfpca::tests` |
| QUAL-02: select_ncomp | selected component count unchanged for all NcompMethod variants | Regression | `cargo test -p fdars-core --features linalg,parallel -- spm::ncomp::tests` |
| QUAL-02: t2_contributions_mfpca, spe_contributions | contribution matrices unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- spm::contrib::tests` |
| QUAL-02: spm_ewma_monitor, spm_mewma_monitor | smoothed_scores, T², SPE unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- spm::ewma::tests spm::mewma::tests` |
| QUAL-02: conditional_expectation | estimated scores + completed_curve unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- spm::partial::tests` |
| QUAL-02: frcc_phase1 | eigenvalues, R², control limits unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- spm::frcc::tests` |
| QUAL-02: profile_phase1 | eigenvalues, lag1_autocorrelation, effective_n unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- spm::profile::tests` |
| QUAL-02: spe_multivariate, spm_monitor_from_fields | SPE vec / T²+SPE results unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- spm::stats::tests spm::phase::tests` |

### Sampling Rate

- **Per-helper extraction:** Module tests for the target file (~5–30 sec)
- **Per-function completion:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- **Per-plan gate:** Module tests + `pmat analyze complexity` for affected files
- **Phase gate:** Full `cargo test -p fdars-core --features linalg,parallel` + `pmat quality-gate | grep spm/`

---

## Common Pitfalls

### Pitfall 1: Reordering the AMEWMA adaptive-update steps

**What goes wrong:** `spm_amewma_monitor` has 5 sequential sub-steps per i:
apply lambda_cur → update sum_lambda_sq_prod → compute T² → compute error →
update Q_t/lambda_prev. If an extracted helper absorbs steps out of order, the
adaptive weight `q_prev` / `lambda_prev` fed to the NEXT iteration will differ.
**How to avoid:** Extract only contiguous sub-blocks. The validation and
setup (project+standardize) are safe to extract because they precede the loop.
Inside the loop, extract only single-step helpers that receive `lambda_cur` as
a parameter and do not themselves write `lambda_prev`.
**Warning signs:** Test failures in `spm::amewma::tests` with lambda_t values
that differ from the expected sequence.

### Pitfall 2: `mfpca` scales mutation visibility

**What goes wrong:** `scales[p] = 1.0` on line 329 is inside the `mfpca_build_stacked`
loop. If the stacked-matrix helper receives `scales` by value (copy), the mutation
is lost and the `MfpcaResult::scales` field will contain the original (pre-mutation)
values. The `MfpcaResult::project()` method uses `self.scales` to un-scale
reconstructed data — wrong scales produce wrong reconstructions.
**How to avoid:** Pass `scales` as `&mut Vec<f64>` to `mfpca_build_stacked`.
Verify that the returned `MfpcaResult.scales` matches the in-place-mutated values
via the existing projection tests.
**Warning signs:** `MfpcaResult::project()` returning values that don't match
the training data reconstruction.

### Pitfall 3: Cognitive score surprises after extraction

**What goes wrong:** An extracted helper still exceeds cognitive 25 because it
inherits all of the original nesting depth (e.g., a validation loop with per-element
NaN + negative checks, or a triple-nested score projection loop).
**How to avoid:** Count nesting depth before extraction. The `ncomp_validate_eigenvalues`
loop (for + 2 ifs = nesting 2) should score ~12–14. `mfpca_build_stacked` has
for + if + for + for = nesting 3, estimated score ~18–22. If any extracted helper
scores > 25, apply split-validate or extract the inner-loop body one level deeper.
**Warning signs:** `pmat analyze complexity --path <file>` shows a new helper
with cog > 25 immediately after extraction.

### Pitfall 4: `conditional_expectation` retry loop fragility

**What goes wrong:** `cond_exp_solve_with_retry` reads `diag_max` from the
M matrix built by `cond_exp_build_m_matrix`. If `diag_max` is not returned by
the builder or is recomputed with a different value (e.g., from the regularized
matrix rather than the original), the initial reg_strength differs and the retry
sequence is numerically different.
**How to avoid:** Extract `diag_max` from `cond_exp_build_m_matrix`'s return
value alongside the matrix itself, passing it as a parameter to `cond_exp_solve_with_retry`.
The retry loop uses `diag_max * 1e-8` as the starting regularization — the
`diag_max` must be from the ORIGINAL matrix, before any regularization.

### Pitfall 5: `#[inline]` budget

**What goes wrong:** Marking every extracted helper `#[inline]` negates some of
the compile-time savings from decomposition (code-size growth).
**How to avoid:** Only mark the innermost hot-path helpers (score projection
inner loops, EWMA step) with `#[inline]`. Validation fns and setup fns should
NOT be inlined — they are called once at function entry.

---

## Hard Constraints Summary

> [VERIFIED: fdars-core/src/spm/{amewma,iterative,mfpca,ncomp,contrib,ewma,mewma,partial,frcc,profile,phase,stats}.rs — all read this session]

1. **No public API change.** Extracted helpers are `fn` (module-private). All `pub fn` signatures untouched.
2. **Numeric output identical.** EWMA/AMEWMA loop order preserved; MFPCA scale mutation preserved; Cholesky retry sequence preserved.
3. **No new Cargo.toml dependency.** pmat is invoked externally.
4. **`fdars-core` only.** No changes outside `fdars-core/src/spm/`.
5. **Test suite stays green.** 2942 lib tests + 215 doctests are the numeric oracle.
6. **`cargo clippy --all-targets --features linalg,parallel -- -D warnings`** stays clean.
7. **`cargo fmt --check`** stays clean. Run `cargo fmt` before each commit.
8. **Commit with `--no-verify`** (pre-commit hook times out on full cargo gate). Run gates foreground, separately, with 600s timeout.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `pmat` | Complexity verification | ✓ | 3.18.0 | — |
| `cargo` | Build + test | ✓ | (Rust 1.97.0) | — |
| `cargo clippy` | Lint gate | ✓ | same toolchain | — |
| `cargo fmt` | Format gate | ✓ | same toolchain | — |

---

## Security Domain

Security enforcement does not apply to this phase: pure private-helper extraction
within an existing module. No new network endpoints, auth paths, file access
patterns, or schema changes are introduced. No new ASVS categories are relevant.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Extracted `amewma_validate` will score cog ≤ 25 (7 guards, no loops) | amewma decomposition | Need to split into `_validate_params` + `_validate_lambda_range`; same split-validate pattern |
| A2 | `mfpca_build_stacked` taking `&mut Vec<f64>` for scales compiles cleanly alongside the immutable borrows of `variables` and `means` | mfpca decomposition | Borrow conflict if `scales` shares lifetime with other params; fallback: return new `Vec<f64>` from helper and let caller re-assign |
| A3 | `spm_monitor_from_fields` validation block (5 guards, no loops) will score ≤ 25 after extraction | phase.rs decomposition | Need split-validate; natural boundary: dim guards vs. argvals guards |
| A4 | `profile_phase1` will drop to ≤ 25 after extracting `profile_validate` (7 guards; ~cog 22 estimated) | profile decomposition | Need split-validate on window-size guards vs. fraction/step guards |
| A5 | `solve_spd` at `partial.rs:499` (cog=21, warning-level) does not require extraction for QUAL-02; it stays at warning | partial.rs scope | If a Phase 109 tail pass targets warnings as well, it will need extraction then |

---

## Open Questions

1. **Is `solve_spd` in scope for Phase 107?**
   - What we know: `solve_spd` at `partial.rs:499` scores cog=21, cyc=11 — warning-level (cog threshold 23). It is not in error territory.
   - What's unclear: QUAL-02 says "refactored under pmat thresholds" — does "thresholds" mean error-only (>25) or all violations (>23)?
   - Recommendation: Target error-level only (cog >25) in Phase 107 per QUAL-02 scope. Leave `solve_spd` cog=21 for Phase 109's tail pass.

2. **Will `rolling_betas` (cog=21) need extraction in Phase 107?**
   - What we know: `rolling_betas` scores cog=21 (warning, threshold 23). Not in Phase 107 error list.
   - Recommendation: Out of scope for Phase 107. Phase 109 may address it.

---

## Sources

### Primary (HIGH confidence)
- [VERIFIED: fdars-core/src/spm/amewma.rs:137-314] — read this session
- [VERIFIED: fdars-core/src/spm/iterative.rs:150-278] — read this session
- [VERIFIED: fdars-core/src/spm/mfpca.rs:251-406] — read this session
- [VERIFIED: fdars-core/src/spm/ncomp.rs:81-200] — read this session
- [VERIFIED: fdars-core/src/spm/contrib.rs:62-341] — read this session
- [VERIFIED: fdars-core/src/spm/ewma.rs:217-347] — read this session
- [VERIFIED: fdars-core/src/spm/mewma.rs:138-249] — read this session
- [VERIFIED: fdars-core/src/spm/partial.rs:164-548] — read this session
- [VERIFIED: fdars-core/src/spm/frcc.rs:199-367] — read this session
- [VERIFIED: fdars-core/src/spm/profile.rs:191-485] — read this session
- [VERIFIED: fdars-core/src/spm/phase.rs:458-549] — read this session
- [VERIFIED: fdars-core/src/spm/stats.rs:275-341] — read this session
- [VERIFIED: pmat 3.18.0 live output, 2026-09-14] — per-file + quality-gate runs confirmed all 13 scores

### Secondary (MEDIUM confidence)
- [VERIFIED: .planning/phases/107-spm-complexity-cluster/107-CONTEXT.md] — phase constraints
- [VERIFIED: .planning/STATE.md] — milestone gates and decisions
- [VERIFIED: .planning/REQUIREMENTS.md] — QUAL-02 requirement text
- [VERIFIED: .planning/phases/106-top-level-complexity-hotspots/106-RESEARCH.md] — proven decomposition patterns
- [VERIFIED: .planning/phases/106-top-level-complexity-hotspots/106-01-SUMMARY.md] — split-validate deviation documentation

---

## Metadata

**Confidence breakdown:**
- Exact file locations and line ranges: HIGH — all confirmed via `Read` + pmat per-file output this session
- Complexity baselines: HIGH — live pmat 3.18.0 output
- Decomposition seams: HIGH — derived from reading source directly
- Post-extraction pmat scores: MEDIUM — estimated from nesting analysis; verify each after extraction
- Borrow-checker hazards: HIGH for the main patterns; MEDIUM for mfpca `&mut scales` edge case (A2)

**Research date:** 2026-09-14
**Valid until:** Stable until source files are modified. pmat thresholds are pmat 3.18.0-specific.
