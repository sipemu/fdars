# Survey: tidyfun / refund (R) Ecosystem (TDY-01)

**Milestone:** v0.31.0 Multi-Ecosystem Gap Audit — Phase 52
**Survey date:** 2026-09-02
**Ecosystem:** R tidyfun data-representation/workflow slice + refund methods NOT already captured in v0.18.0
**Packages surveyed (version-pinned as of 2026-09):**
- `tf@0.3.x` — S3 classes/methods for tidy functional data (`tfd` evaluated, `tfb` basis) on the `vctrs` framework (tidyfun/tf).
- `tidyfun@0.x` (dev) — tidyverse-native wrangling/visualization over `tf` vectors (tidyfun.github.io).
- `refund@0.1-38` — Regression with Functional Data (CRAN).

**v0.18.0 refund boundary (EXCLUSIONS — already captured, out of scope here):**
refund methods already surveyed/backlogged in `R-AUDIT-REPORT.md` / `R-BACKLOG.md` are EXCLUDED: `pffr`/`pfr` (REG-01/04/05), `fosr`/`fosr2s`/`bayes_fosr` (REG-05/06), `fgam`/GKAM/GSAM (REG-04), mixed-effects `denseFLMM`/`multiFAMM`/`fastFMM` (REG-05), `fbps`/sandwich smoother + `fpca.face`/`fpca.sc`/`fpca.ssvd`/`fpca2s` (REP/FPCA-02, `R-BACKLOG` fbps hits=11), boosting/Bayesian FOSR (REG-06). Only refund methods with ZERO hits in both v0.18.0 artifacts are eligible.

**De-dup baselines:** shipped fdars, `BACKLOG.md`, `R-BACKLOG.md`, `R-AUDIT-REPORT.md`.
**Audit-only fence:** zero `fdars-core/src/` edits.

---

## Capability Inventory & fdars Parity Mapping

### tidyfun / tf — data representation & workflow slice

| Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes |
|---|---|---|---|---|---|
| `tfd`/`tfb` first-class functional-vector types (evaluated vs basis) on `vctrs`, operator overloading (`+`, `mean()`, `c()`) | tf@0.3.x | present (as data model) | `grep matrix.rs fdata.rs basis/`: `FdMatrix` (evaluated) + basis coefficients | No | fdars represents evaluated (`FdMatrix`) and basis forms; the tf *ergonomics* (R vctrs integration, tidyverse verbs) is an API-layer concern, not a numeric capability. |
| `tf_smooth`, `tf_derive`, `tf_integrate` (functional ops) | tf@0.3.x | present | `grep smoothing.rs pda.rs utility.rs scoring.rs`: smoothing, derivative, integration present | No | Numeric ops covered. |
| `tf_gather`/`tf_nest`/`tf_spread`/`tf_unnest` (wide↔long reshaping in data frames) | tidyfun@0.x | n/a | — | No | Data-frame reshaping / IO ergonomics — **out of scope** (data/IO parity fence). |
| `tf_ggplot` + `geom_*` functional visualization | tidyfun@0.x | n/a | — | No | Plotting — **out of scope** (visualization fence). |
| dplyr verbs (`filter`/`mutate`/`group_by`/`summarize`) over functional columns | tidyfun@0.x | n/a | — | No | Ergonomics layer, not a numeric method. |

### refund — methods NOT captured in v0.18.0

| Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes |
|---|---|---|---|---|---|
| `peer` / `lpeer` — Partially Empirical Eigenvectors for Regression (structured a-priori-penalty scalar-on-function regression; longitudinal PEER) | refund@0.1-38 | absent | `grep -riE "peer\|lpeer\|partially.empirical" src/` → none; `R-BACKLOG`=0, `R-AUDIT`=0 | **Yes** | Net-new: penalized SoF regression with a structured/decomposition-informed penalty distinct from FPCR/`pfr`. Not surveyed in v0.18.0. |
| `wcr` / `wnet` — wavelet-domain scalar-on-function regression (wavelet compression + ridge/lasso/elastic-net) | refund@0.1-38 | absent | `grep -riE "wcr\|wnet\|wavelet.*regress" src/` → none (only `detect_amplitude_modulation_wavelet` in seasonal); `R-BACKLOG`=0, `R-AUDIT`=0 | **Yes** | Net-new: sparse wavelet-domain functional regression for spiky/localized signals. Not surveyed in v0.18.0. |
| `fbps` / sandwich smoother, `fpca.face`/`fpca.sc`/`fpca.ssvd`/`fpca2s` | refund@0.1-38 | present/tracked | `grep fpca_variants.rs irreg_fdata/face.rs`: FACE present; `R-BACKLOG` fbps hits=11 | No | Already captured/backlogged in v0.18.0. Excluded. |
| `ccb.fpc` (bootstrap CI for FPC) | refund@0.1-38 | present | `grep spm/bootstrap.rs` + FPCA bootstrap | No | Bootstrap CI machinery present. |
| `rlrt.pfr` (restricted likelihood ratio test for functional regression) | refund@0.1-38 | present/tracked | inference suite `R-BACKLOG` INF-02 (FLM inference) | No | FLM inference already backlogged in v0.18.0. |

---

## Net-New Gap List (tidyfun / refund)

| Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes |
|---|---|---|---|---|---|
| **PEER / longitudinal PEER (`peer`/`lpeer`)** | refund@0.1-38 | absent | `peer/lpeer/partially.empirical` → none in `src/`, backlogs, or R-AUDIT | Yes | Value: structured-penalty SoF regression (incorporate a-priori signal structure). Effort: M (penalty-matrix SoF regression atop existing FPCR/`pfr`-style machinery). Reference: refund `peer`/`lpeer`. |
| **Wavelet-domain functional regression (`wcr`/`wnet`)** | refund@0.1-38 | absent | `wcr/wnet/wavelet.*regress` → none in `src/`, backlogs, or R-AUDIT | Yes | Value: sparse regression for spiky/localized functional predictors via wavelet compression + lasso/elastic-net. Effort: M–L (needs a wavelet transform + regularized-regression path; fdars has `rustfft` but no DWT). Reference: refund `wcr`/`wnet`. |

**Net-new gap count: 2.**

tidyfun's contribution is a data-representation & tidyverse-workflow ergonomics layer whose numeric operations fdars already provides; the reshaping/plotting pieces fall under the data-IO and visualization scope fences. refund was thoroughly audited in v0.18.0, leaving only the two structured/wavelet regression methods above as net-new.

---

## Reverse-Parity Note (where fdars LEADS tidyfun/refund)

- **Numeric depth & breadth**: fdars ships elastic/SRVF shape analysis, depth measures, SPM, conformal prediction, streaming depth, seasonal decomposition, explainability — none in tidyfun (a representation layer) and only partially in refund (a regression package).
- **Performance & deployment**: compiled Rust column-major core, WASM/JS + R bindings; tidyfun/refund are R-interpreter-bound.
- **Representation duality**: fdars carries evaluated (`FdMatrix`) and basis representations with efficient row ops (`row_dot`, `row_l2_sq`) — the numeric substrate under tidyfun's `tfd`/`tfb`.
