---
status: complete
quick_id: 261126-ic6
issue: 65
commits: [2fc65ed4]
---
# 261126-ic6 — ALN-CAV elastic DP residual floor (#65)

## Diagnosis (ignored test `curve_shape::optim_tests::residual_floor_table`)
d(β, β∘γ0), γ0 = t + 0.25t(1−t), analytic warped curve (open planar, default quotient).
"oracle" = exact inverse warp pushed through the pipeline (linear curve resampling,
finite-difference SRVF, Procrustes, trapezoid distance).

SRVF:
| m | oracle | dp7 (default) | dp10+lin | dp20+lin | rbfgs (it2) | rbfgs it6 | rbfgs no-rot | |γ_dp7−γ*|∞ |
|---|---|---|---|---|---|---|---|---|
| 41 | 0.00397 | 0.03653 | 0.02684 | 0.02684 | 0.00424 | 0.00362 | 0.00362 | 0.0125 |
| 81 | 0.00185 | 0.01829 | 0.01122 | 0.00907 | 0.00258 | 0.00187 | 0.00187 | 0.0052 |
| 161 | 0.00090 | 0.01864 | 0.01091 | 0.00583 | 0.00209 | 0.00094 | 0.00093 | 0.0057 |
| 321 | 0.00046 | 0.01866 | 0.01106 | 0.00466 | 0.00192 | 0.00051 | 0.00051 | 0.0044 |

(a,b) = (2, 0.5) (λ = 2): oracle 0.0076/0.0035/0.0017/0.0009; dp7 0.056/0.025/0.020/0.019;
dp10+lin 0.030/0.013/0.012/0.013; rbfgs it6 0.0067/0.0036/0.0018/0.0009.
1-D elastic distance (#49 curve): oracle 0.034/0.0082/0.0054/0.0021; dp7 0.127/0.077/0.064/0.044;
dp10+lin 0.087/0.034/0.030/0.026; dp20+lin 0.087/0.026/0.016/0.011; rbfgs 0.039/0.016/0.0074/0.0044.
Closed contour (warp + integer seed shift), m = 61/121/241: dp7 0.044/0.029/0.023;
dp10+lin 0.027/0.023/0.014; rbfgs it4 0.014/0.013/0.0035.

Sources:
1. **Warp-slope quantisation (dominant, m-independent).** The DP path uses moves with slope
   dr/dc, dr,dc ≤ nbhd_dim, so √γ' is quantised with resolution ~1/nbhd_dim regardless of m;
   warp *positions* are accurate (|γ−γ*|∞ < h) but the √γ' factor is wrong by a few percent →
   floor 0.0186 (nbhd 7), 0.011 (nbhd 10 + linear), decreasing further with nbhd 20.
2. **Unconverged rotation ↔ warp alternation (m-independent, ≈ 0.0015–0.0035).** With
   `align_iter = 2` the warp is optimised under a slightly wrong rotation; masked by (1) for DP,
   visible with RBFGS (0.0019 at m = 321 vs 0.0005 with 6 rounds or rotation off). Converges
   geometrically: it 1/2/3/4/6 → 0.0056/0.0021/0.0012/0.0010/0.0009 at m = 161.
3. **Resampling / finite-difference SRVF / quadrature** (oracle): O(h), not a floor.
   RBFGS's own discrete objective is not a floor either: it converges (gnorm ~1e-8) at the
   alternation-limited value; max_iter / tolerance changes had no effect.

## Fixes (additive; defaults bit-identical)
- `CurveShapeConfig::warp_optim: WarpOptimConfig` (serde default). Method `Rbfgs` →
  multi-channel RBFGS (coarse R^d classic DP start on ≤ 51 points, `L2Gam`/`L2Psi` start penalty
  as in 1-D); `Dp` with non-classic options → new R^d DP (`nbhd_dim`, `dp_scl`, `dp_cost`,
  `L2Gam`/`L2Psi`); classic options → unchanged `geometry::dp_warp`. Penalty weight =
  `CurveShapeConfig::lambda`; `warp_optim.lambda ≠ 0` is rejected; DP + Roughness/Geodesic with
  λ > 0 rejected (via `WarpOptimConfig::validate`). Single DP call site → all curve_shape_*
  functions (Karcher, PCA, TSRVF, depth, boxplot, CI, closed, (a,b)) get the option.
- `RbfgsProblem::new_multi` / `rbfgs_warp_multi`: data term Σ_c ‖q1_c − (q2_c∘γ)l‖², total-norm
  normalisation; 1-D path bit-identical (FNV hash recorded before the change).
- `elastic_align_pair_closed_with_config` / `elastic_distance_closed_with_config` (1-D closed,
  shift search + final alignment with the configured optimiser); `WarpOptimConfig::default()`
  with λ == classic `elastic_align_pair_closed` (struct equality test).
- serde derives on `WarpOptimConfig`, `WarpOptimMethod`, `WarpObjectivePenalty`, `DpEdgeCost`.
- (2) DP2-style default: **not changed** (bit-identity). Recommended setting documented on the
  field: `warp_optim.method = Rbfgs` + `align_iter = 4` (best); `nbhd_dim = 10` + PiecewiseLinear
  stays global but only lowers the floor 0.0186 → 0.011 at ≈ 5× DP cost. RBFGS is also faster than
  DP at large m (closed m = 241, debug: 15 s vs 39 s).

## Residuals before / after (new tests)
- open SRVF m = 80 rigid+warp: DP 0.0187 → RBFGS 0.00189 (bound 0.003 vs existing 0.03).
- (a,b) reparam invariance m = 101: DP 0.014–0.042 → RBFGS 0.0010–0.012 (bound 0.015 and < DP/3;
  existing DP bound 0.05).
- closed contour m = 121: 0.029 → 0.0125 (bound 0.02, < DP/2).
- Karcher mean residuals: max 0.0196 → 0.0021 (bound 0.01).
- 1-D closed m = 81: DP 0.100, DP2 0.074, RBFGS 0.043.

## Tests (fdars-core/src/alignment/curve_shape/optim_tests.rs, optim_tests.rs)
Bit-identity hash of default curve-shape outputs (pre-change), explicit default + dp_prune=false
hash, generic R^d DP == classic R^d DP, R^d DP options with a vanishing coordinate == 1-D engine
(DP2 golden-tested vs fdasrsf) for (10,linear), (7,scl 3), (12,const); floor regression (dp7
does not shrink with m, DP2 < 0.7·dp7, RBFGS within 1.5× oracle and shrinking); tight
invariance (rigid+warp, warp recovery < 5e-3, rotation recovery), (a,b) metrics, closed contour,
Karcher; validation; 1-D RBFGS bit-identity hash; closed with_config == classic; closed refined.
Existing default-path tolerances left unchanged (they still describe the default DP floor).

## Gates
- `cargo test -p fdars-core --lib alignment::` 561 passed, 3 ignored (diagnostics).
- doctests for the new examples pass; `cargo clippy --all-targets --features linalg,parallel
  -D warnings` clean; `cargo check --features serde` clean; `RUSTDOCFLAGS=-Dwarnings cargo doc`
  clean; `cargo fmt --check` clean.

## Deviations / remaining
- Closed-contour residual remains above the oracle at small m because seeds are integer samples
  (sub-sample start-point offset can't be absorbed by an endpoint-fixed warp); out of scope.
- Default `align_iter` not changed (bit-identity); RBFGS users should set ≥ 4.
