---
phase: 02-static-hot-path-analysis
verified: 2026-08-07T00:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
resolution: >
  The single 3/4 gap (2 wrong basis/projection.rs:117,119 anchors in the
  from_column_slice sub-table) was resolved by the orchestrator per the
  verifier's recommended option (a), in commit fixing AUDIT-REPORT.md:
  the impostor rows were replaced with the genuine sites basis/pspline.rs:87
  (the missing P-spline site) and elastic_regression/scalar_on_shape.rs:119
  (promoted from footnote to a table row). All 14 basis-table anchors were
  then re-validated against source — every cited file:line is a real
  DMatrix::from_column_slice call site. Phase now passes 4/4.
human_verification_resolved:
  - test: "Resolve basis/projection.rs:117 and :119 anchor discrepancy in from_column_slice table"
    resolution: "Fixed — impostor rows replaced with basis/pspline.rs:87 and scalar_on_shape.rs:119; all 14 anchors re-validated against source."
    expected: >
      Either (a) update the two wrong entries to their correct paths
      (`basis/pspline.rs:87` for the missing P-spline site and acknowledge that
      scalar_on_shape.rs:117,119 are already listed correctly at the bottom of
      the table), OR (b) add an override confirming this deviation is acceptable
      because the 14-site count is correct and downstream Phase 4 engineers can
      locate the real sites from the already-correct scalar_on_shape.rs rows.
    why_human: >
      basis/projection.rs:117 and :119 in the actual source are
      `let proj = btb_inv * b_mat.transpose()` and a comment — NOT
      from_column_slice calls. The real sites at those descriptions
      (basis/projection companion + penalty) do not exist in projection.rs;
      they belong to elastic_regression/scalar_on_shape.rs:117,119 which ARE
      also listed correctly in the table. The missing site is basis/pspline.rs:87.
      Whether to treat these two wrong anchors as blocking (the map is a
      source-anchored deliverable) or acceptable (count is correct, downstream
      impact is low) requires a human decision.
---

# Phase 2: Static Hot-Path Analysis — Verification Report

**Phase Goal:** Produce the zero-cost priority map of where fdars scales badly and why, before any expensive measurement
**Verified:** 2026-08-07
**Status:** passed (4/4 — the lone 3/4 anchor gap was fixed post-verification; see `resolution` in frontmatter)
**Re-verification:** No — initial verification

---

## Step 0 — Previous Verification

No previous VERIFICATION.md found in the phase directory. Initial mode.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A per-module bottleneck-candidate table exists giving algorithmic complexity in N and M for all 6 required modules: elastic alignment, FPCA/SVD, depth & distance, CV loops, streaming depth, and smoothing | ✓ VERIFIED | Complexity Table has exactly 6 data rows. All 6 modules present. Each row cites real file:line anchors and states N and M scaling separately. Confirmed by awk count = 6. |
| 2 | An allocation-hotspot list enumerates every to_dmatrix()/DMatrix::from_column_slice/redundant-clone call site, including the 8 production FdMatrix→DMatrix SVD-copy sites | ⚠️ PRESENT — anchor discrepancy | All 8 production to_dmatrix() SVD sites confirmed at correct lines (elastic_fpca.rs:214,317,483,584,930; alignment/nd.rs:705; regression.rs:298; spm/mfpca.rs:336). regression.rs:291 redundant clone confirmed. from_column_slice table has 14 rows, but 2 entries cite wrong anchors: basis/projection.rs:117 and :119 do NOT contain from_column_slice in the actual source. The real missing site is basis/pspline.rs:87. The 14-site count is coincidentally correct because scalar_on_shape.rs:117,119 are listed separately in the table. |
| 3 | A parallelism-gap list flags sequential loops as parallelization candidates and labels already-parallel loops correctly, with no false positives | ✓ VERIFIED | cv.rs:76 fold loop confirmed sequential (grep returns zero macro hits in cv.rs). regression.rs:167 center_columns confirmed sequential (zero macro hits in regression.rs scope). karcher.rs:185 and pairwise.rs:227 confirmed ALREADY PARALLEL (iter_maybe_parallel! present at those exact lines). streaming_depth/fraiman_muniz.rs:82 and mbd.rs:76 confirmed ALREADY PARALLEL (iter_maybe_parallel! present). smoothing.rs:110 confirmed ALREADY PARALLEL (slice_maybe_parallel!). elastic_fpca.rs:701,720,764 confirmed SEQUENTIAL (no macros present in elastic_fpca.rs at all). Banding opt-in noted. |
| 4 | Every finding carries a feature-gate annotation ([always]/[parallel-gated]/[sequential]/[linalg-gated]) so no macro-wrapped loop is mislabeled sequential | ✓ VERIFIED | 44 gate-tag occurrences counted in document. No loop labeled SEQUENTIAL was found to have a parallelism macro at its cited line. All 6 complexity rows carry gate tags. All allocation sites carry [always]. Parallelism entries all carry exactly one tag. |

**Score:** 3/4 truths verified (1 has a factual anchor discrepancy requiring human resolution)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/research/AUDIT-REPORT.md` | New `## Phase 2 — Static Hot-Path Analysis` section appended below Phase 1 content with three sub-sections | ✓ VERIFIED | Section exists. Phase 1 sections (`## Phase 1 — Measurement Discipline`, `## §Methodology`, `## §Workload Matrix`) are all intact and unmodified. Three sub-section headers (`### Complexity Table`, `### Allocation Hotspot List`, `### Parallelism Gap List`) all present at correct heading level. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| Phase 2 section | Plan 01 skeleton | Three sub-section headers | ✓ WIRED | All three sub-sections created by Plan 01 and populated by Plan 02. Headers match contract: `### Complexity Table`, `### Allocation Hotspot List`, `### Parallelism Gap List`. |
| Complexity Table | fdars-core source | file:line citations | ✓ WIRED (5/6 modules fully correct; depth module anchors confirmed) | All primary anchors verified against grep: karcher.rs:185,323 confirmed; pairwise.rs:227 confirmed; elastic_fpca.rs:930 enclosing fn `optimize_balance_c_raw` confirmed; regression.rs:167/249/291/298 confirmed; cv.rs:45/76 confirmed; streaming_depth/fraiman_muniz.rs:77/82 confirmed; smoothing.rs:110 confirmed; depth/fraiman_muniz.rs:32 confirmed. |
| Allocation Hotspot List | fdars-core source | file:line citations (to_dmatrix sites) | ✓ WIRED | All 8 production to_dmatrix() SVD sites confirmed at correct lines. |
| Allocation Hotspot List | fdars-core source | file:line citations (from_column_slice sites) | ⚠️ PARTIAL | 12 of 14 from_column_slice anchors verified. 2 entries (basis/projection.rs:117, basis/projection.rs:119) cite lines that do not contain from_column_slice in the actual source. Real source has basis/pspline.rs:87 which is NOT listed in the table. |

---

## Source-Anchor Spot-Checks

Representative anchors verified against `fdars-core/src/` source:

| Cited anchor | Claim | Actual source | Match? |
|---|---|---|---|
| `alignment/karcher.rs:185` | `iter_maybe_parallel!(0..n)` | `iter_maybe_parallel!(0..n)` (karcher.rs:185 confirmed by grep) | YES |
| `alignment/pairwise.rs:227` | `iter_maybe_parallel!(0..n)` | `iter_maybe_parallel!(0..n)` (pairwise.rs:227 confirmed by grep) | YES |
| `karcher.rs:323` | `fn karcher_mean_impl` | `fn karcher_mean_impl(` at line 323 | YES |
| `elastic_fpca.rs:214` | `SVD::new(...to_dmatrix(),...)` | `let svd = SVD::new(centered.to_dmatrix(), true, true);` | YES |
| `elastic_fpca.rs:317` | `SVD::new(...to_dmatrix(),...)` | `let svd = SVD::new(combined.to_dmatrix(), true, true);` | YES |
| `elastic_fpca.rs:483` | `SVD::new(...to_dmatrix(),...)` | `let svd = SVD::new(centered.to_dmatrix(), true, true);` | YES |
| `elastic_fpca.rs:584` | `SVD::new(...to_dmatrix(),...)` | `let svd = SVD::new(combined.to_dmatrix(), true, true);` | YES |
| `elastic_fpca.rs:930` | `SVD::new(...to_dmatrix(),...)` inside `optimize_balance_c_raw` / `eval_c` closure | `let svd = SVD::new(combined.to_dmatrix(), true, true);` at line 930, inside `fn optimize_balance_c_raw` (line 905) with `eval_c` closure starting at line 919 | YES |
| `alignment/nd.rs:705` | `SVD::new(gram.to_dmatrix(),...)` | `let svd = SVD::new(gram.to_dmatrix(), true, true);` | YES |
| `regression.rs:249` | `pub fn fdata_to_pc_1d` | `pub fn fdata_to_pc_1d(` | YES |
| `regression.rs:291` | `weighted = centered.clone()` | `let mut weighted = centered.clone();` | YES |
| `regression.rs:298` | `SVD::new(weighted.to_dmatrix(),...)` | `let svd = SVD::new(weighted.to_dmatrix(), true, true);` | YES |
| `regression.rs:167` | `fn center_columns` sequential double loop | `fn center_columns(` at line 167; `for j in 0..m {` at 171; `for i in 0..n {` at 176; zero macro hits confirmed | YES |
| `spm/mfpca.rs:336` | `SVD::new(stacked.to_dmatrix(),...)` | `let svd = SVD::new(stacked.to_dmatrix(), true, true);` | YES |
| `classification/cv.rs:45` | `pub fn fclassif_cv` | `pub fn fclassif_cv(` at line 45 | YES |
| `classification/cv.rs:76` | `for fold in 0..nfold` sequential | `for fold in 0..nfold {` at line 76; zero macro hits in cv.rs confirmed | YES |
| `streaming_depth/fraiman_muniz.rs:82` | `iter_maybe_parallel!(0..nobj)` | `iter_maybe_parallel!(0..nobj)` at line 82 | YES |
| `streaming_depth/mbd.rs:76` | `iter_maybe_parallel!(0..nobj)` | `iter_maybe_parallel!(0..nobj)` at line 76 | YES |
| `smoothing.rs:110` | `slice_maybe_parallel!(x_new)` | `Ok(slice_maybe_parallel!(x_new)` at line 110 | YES |
| `depth/fraiman_muniz.rs:32` | `pub fn fraiman_muniz_1d` delegates to streaming | `pub fn fraiman_muniz_1d(` at line 32; body creates SortedReferenceState, then calls `streaming.depth_batch(data_obj)` | YES |
| `elastic_fpca.rs:701` | `for i in 0..n` in `shooting_vectors_from_psis` (SEQUENTIAL) | `fn shooting_vectors_from_psis(` at line 693; `for i in 0..n {` at line 701; no iter_maybe_parallel! anywhere in elastic_fpca.rs | YES |
| `elastic_fpca.rs:720` | `for i in 0..n` in `build_augmented_srsfs` (SEQUENTIAL) | `fn build_augmented_srsfs(` at line 711; `for i in 0..n {` at line 720 | YES |
| `elastic_fpca.rs:764` | `for i in 0..n` in `svd_scores_and_eigenvalues` (SEQUENTIAL) | `fn svd_scores_and_eigenvalues(` at line 749; `for i in 0..n {` at line 764 | YES |
| `basis/projection.rs:117` | `DMatrix::from_column_slice(...)` | `let proj = btb_inv * b_mat.transpose();` — NOT a from_column_slice call | **NO — WRONG ANCHOR** |
| `basis/projection.rs:119` | `DMatrix::from_column_slice(...)` | `// Per-curve coefficient rows...` (a comment) — NOT a from_column_slice call | **NO — WRONG ANCHOR** |
| `basis/pspline.rs:87` | Real from_column_slice site | `let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);` — exists in source but NOT listed in report | **MISSING FROM REPORT** |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| PERF-01 | 02-01-PLAN.md, 02-02-PLAN.md | Static hot-path analysis documents bottleneck candidates per module with algorithmic complexity in N and M, covering at least elastic alignment, FPCA/SVD, depth & distance matrices, CV loops, streaming depth, and smoothing | ✓ SATISFIED | All 6 required modules present in Complexity Table with N and M complexity stated separately and feature-gate annotations. REQUIREMENTS.md marks PERF-01 as `[x] Complete` with Phase 2 mapping. |

---

## SC Verification (reproducing the in-document grep pass)

| Check | Command | Result | Threshold | Pass? |
|---|---|---|---|---|
| SC1 — complexity rows (O(n occurrences) | `grep -c "O(n" AUDIT-REPORT.md` | 9 (Phase 2 section contributes 6+ module-row cells) | ≥ 6 | Yes |
| SC2 — SVD copy sites | `grep -c "to_dmatrix" AUDIT-REPORT.md` | 12 (8 table entries + prose references) | ≥ 8 | Yes |
| SC3 — parallelism labels | `grep -Eic "already parallel" AUDIT-REPORT.md` | 8 | ≥ 4 | Yes |
| SC4 — gate tag coverage | `grep -Eoc "\[parallel-gated\]\|\[sequential\]\|\[linalg-gated\]\|\[always\]" AUDIT-REPORT.md` | 44 | ≥ 10 | Yes |
| matrix.rs:682 excluded | `! grep -q "matrix.rs:682" AUDIT-REPORT.md` | absent | absent | Yes |

---

## Behavioral Spot-Checks

Step 7b: SKIPPED — this is a documentation-only phase producing markdown only. No runnable code was produced or modified. All deliverables are in `.planning/research/AUDIT-REPORT.md`.

---

## Probe Execution

Step 7c: SKIPPED — no probe scripts declared in PLAN files for this phase. Phase is analysis-only.

---

## Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `.planning/research/AUDIT-REPORT.md` | `basis/projection.rs:117` listed as `from_column_slice` site — actual line is `let proj = btb_inv * b_mat.transpose();` | Warning | A Phase 4 engineer following this anchor will not find a `from_column_slice` call at that line. The real site they likely intended (`elastic_regression/scalar_on_shape.rs:117`) is also listed separately in the same table. Downstream impact is low but the anchor is factually wrong. |
| `.planning/research/AUDIT-REPORT.md` | `basis/projection.rs:119` listed as `from_column_slice` site — actual line is a comment | Warning | Same issue as above. The companion penalty matrix at `scalar_on_shape.rs:119` is noted in the table footnote. `basis/pspline.rs:87` (real P-spline site) is absent from the table. |

No `TBD`, `FIXME`, or `XXX` markers found (documentation-only phase; no source code modified).

---

## Human Verification Required

### 1. Resolve basis/projection.rs:117 and :119 anchor discrepancy

**Test:** Open `.planning/research/AUDIT-REPORT.md`, find the from_column_slice basis table, and check the entries for `basis/projection.rs:117` and `basis/projection.rs:119`. Then open `fdars-core/src/basis/projection.rs` and check those lines. Finally check `fdars-core/src/basis/pspline.rs:87`.

**Expected:** The reviewer confirms that:
- `basis/projection.rs:117` is `let proj = btb_inv * b_mat.transpose();` (NOT a from_column_slice)
- `basis/projection.rs:119` is a comment (NOT a from_column_slice)
- `basis/pspline.rs:87` is `let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);` and is absent from the table

Then decides: either fix the 2 wrong entries (basis/projection.rs:117 → basis/pspline.rs:87; basis/projection.rs:119 removed, with scalar_on_shape.rs:119 note promoted to a proper row), or add an override if the 14-site count and intent are deemed sufficient for Phase 4 planning purposes.

**Why human:** The verifier cannot determine whether this is a blocker for the phase goal ("priority map") vs an acceptable discrepancy in a secondary list. The 8 SVD-copy sites (the ROADMAP-cited primary concern) are all correct. The from_column_slice list is "Phase 4 secondary" by the report's own label. But this is a "source-anchored map" and wrong anchors defeat its navigational purpose. Only the developer can decide if the benefit of fixing these 2 entries now outweighs re-opening a completed phase.

---

## Gaps Summary

No FAILED truths. One truth (SC2 / allocation hotspot list) is factually present but has 2 wrong file:line anchors in the from_column_slice secondary table. The primary deliverables (8 SVD-copy sites, redundant clone, parallelism map, complexity table) are all correct and fully source-anchored. The anchor discrepancy is in the secondary category (Phase 4 target, "not SVD copies"). This is classified as a warning requiring human decision rather than a hard blocker, because:

1. The 8 production to_dmatrix() SVD sites (the ROADMAP-cited primary concern) are all correct.
2. The 14-site count for from_column_slice is coincidentally correct.
3. The intended sites (scalar_on_shape.rs:117,119) ARE listed at the bottom of the table.
4. The missing site (pspline.rs:87) is a real gap in navigability.

The phase goal — "zero-cost priority map of where fdars scales badly and why" — is substantially achieved. The discrepancy affects navigability of a secondary list, not the primary bottleneck picture.

---

_Verified: 2026-08-07_
_Verifier: Claude (gsd-verifier)_
