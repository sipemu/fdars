# Survey: Julia FDA Ecosystem (JUL-01)

**Milestone:** v0.31.0 Multi-Ecosystem Gap Audit — Phase 52
**Survey date:** 2026-09-02
**Ecosystem:** Julia functional data analysis + modern/performance-oriented Julia idioms
**Packages surveyed (version-pinned as of 2026-09):**
- `ElasticFDA.jl@1.x` — SRVF/SRSF elastic FDA: pairwise & groupwise alignment, elastic FPCA, elastic regression (jdtuck).
- `FDA.jl@0.x` — small general FDA package (LewisHein); mostly basis/smoothing basics, low activity.
- `MultivariateStats.jl@0.10.x` (JuliaStats) — PCA/PPCA/KPCA/CCA/factor analysis/MDS (dimension reduction, not FDA-specific).
- `registr` (julia-wrobel; R package with a Julia port lineage) — curve registration for exponential-family functional data (`bfpca`, `gfpca`).
- Supporting idiom sources: `KernelFunctions.jl`, `ForwardDiff.jl`/`Zygote.jl`, `CUDA.jl` broadcast patterns (surveyed for *patterns*, not as FDA packages).

**Note:** Julia has no single dominant FDA package equivalent to R's `fda`. Per JUL-01, modern/performance-oriented Julia PATTERNS (type-generic APIs, autodiff-through-FDA, GPU-friendly broadcast) are captured as candidate gaps where they represent a capability fdars lacks.

**De-dup baselines:** shipped fdars, `BACKLOG.md`, `R-BACKLOG.md`, `R-AUDIT-REPORT.md` (rigor).
**Audit-only fence:** zero `fdars-core/src/` edits.

---

## Capability Inventory & fdars Parity Mapping

| Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes |
|---|---|---|---|---|---|
| SRVF/SRSF elastic pairwise & groupwise alignment | ElasticFDA.jl@1.x | present | `grep elastic.rs warping.rs alignment/`: SRVF/SRSF, karcher mean, groupwise align present | No | fdars matches or exceeds. |
| Elastic FPCA (vertical/horizontal/joint) | ElasticFDA.jl@1.x | present | `grep elastic_fpca.rs` | No | Covered. |
| Elastic function/curve regression | ElasticFDA.jl@1.x | present | `grep elastic_regression/` | No | Covered. |
| SRVF curve (n-D) alignment | ElasticFDA.jl@1.x | present | `grep alignment/` (n-D elastic) | No | Covered. |
| Basis/smoothing basics | FDA.jl@0.x | present | `grep basis/ smooth_basis.rs smoothing.rs` | No | Covered. |
| PCA/PPCA/KPCA/CCA/MDS dimension reduction | MultivariateStats.jl@0.10.x | present | `grep dim.rs regression.rs (FPCA) frechet/`: FPCA + dimension reduction present | No | Multivariate not functional-specific; FPCA covers the functional case. |
| Exponential-family curve registration (binary/generalized FPCA: `bfpca`, `gfpca`) | registr | absent | `grep -riE "bfpca\|gfpca\|binary.*fpca" src/` → only "generalized FPCA tolerance bands" (`tolerance/types.rs`) | No | "exponential family registr" appears in `R-AUDIT-REPORT.md` (registr is R) — already-considered in v0.18.0, not net-new. |
| **Autodiff-compatible / differentiable FDA (gradients through warping, FPCA, regression via generic number types)** | Julia idiom (ForwardDiff/Zygote through generic code) | absent | `grep -riE "autodiff\|differentiable\|dual.number\|generic.*<T" src/`: only `soft_dtw` gradient + `face` bootstrap; core is concrete `f64` | **Yes** | Julia's generic-programming idiom lets AD flow through FDA ops (differentiable registration/FPCA) for embedding in larger optimization/ML pipelines. fdars is `f64`-concrete. Modern-pattern gap. |
| GPU-friendly / batched-broadcast FDA kernels | Julia idiom (CUDA.jl broadcast) | absent | `grep -riE "gpu\|cuda\|simd\|batched" src/` → none | Partial/flagged | Speculative; arguably out-of-scope for a CPU numeric core. Recorded honestly, low priority. |
| Composable kernel/metric abstractions (KernelFunctions-style) | KernelFunctions.jl idiom | present | `grep metric/ distance.rs`: metric module + distances present | No | fdars has a metric module + soft-DTW/DTW/Lp. |

---

## Net-New Gap List (Julia FDA)

| Capability | Reference (pkg@ver) | fdars status | Searched-for | Net-new? | Notes |
|---|---|---|---|---|---|
| **Autodiff-compatible / differentiable FDA core** | Julia idiom (ElasticFDA.jl + ForwardDiff/Zygote pattern) | absent | `autodiff/differentiable/dual.number/generic<T>` → concrete f64 only in `src/` | Yes | Value: gradients through warping/FPCA/regression enable end-to-end optimization & ML embedding. Effort: L (generics-over-scalar refactor of hot paths is invasive; likely a scoped subset — e.g. a differentiable elastic-distance API). Reference: Julia generic-programming + ElasticFDA.jl. |
| **GPU-friendly / batched-broadcast FDA kernels** | Julia idiom (CUDA.jl) | absent | `gpu/cuda/simd/batched` → none | Yes (flagged) | Value: throughput on large curve sets. Effort: L. **Flagged as likely out-of-scope** for a portable CPU/WASM numeric core — recorded for completeness, expected to be triaged out in RPT-03. |

**Net-new gap count: 2** (1 solid architectural gap + 1 honestly-flagged likely-out-of-scope).

Julia's FDA package surface (ElasticFDA.jl, FDA.jl, MultivariateStats.jl) is method-wise covered by or behind fdars. Its distinctive contribution is *architectural* — generic/differentiable/GPU-friendly design — which is where the net-new candidates sit.

---

## Reverse-Parity Note (where fdars LEADS Julia FDA)

- **Breadth**: fdars ships classification, depth, SPM, conformal, seasonal decomposition, Fréchet regression, FTS, boosting, co-clustering, density-object FDA, explainability — Julia has no single package near this coverage; ElasticFDA.jl is elastic-only.
- **Elastic completeness**: fdars adds elastic changepoint, elastic explainability, phase boxplots, and shape CIs beyond ElasticFDA.jl.
- **Deployment**: Rust core compiles to WASM/JS and binds to R; Julia's FDA tooling is Julia-runtime-bound.
- **Determinism**: per-thread seeded RNG + reproducible parallelism vs Julia's looser defaults.
