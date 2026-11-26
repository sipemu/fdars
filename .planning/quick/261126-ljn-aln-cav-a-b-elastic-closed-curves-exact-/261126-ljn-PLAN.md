---
quick_id: 261126-ljn
issue: 68
---
# Plan — ALN-CAV (GH #68): (a,b) elastic metrics — closed curves, exact PL distance, branch search

## Goal
1. Closed planar curves under `CurveShapeMetric::Elastic { a, b }`.
2. Bauer et al. (J. Nonlinear Sci. 2024, arXiv 2209.09862, §4.1) exact distance for
   piecewise-linear curves (via the Lahiri–Robinson–Klassen 2015 precise-matching structure).
3. Optional `rotation = false` branch search over the F-angle ambiguity.

## Math
### Closure in F-space
`F = r e^{iφ}`, `r = 2b|c'|^{1/2}`, `φ = λθ` (continuous lift), λ = a/(2b). Inverse:
`c' = |F|²/(4b²) · e^{iφ/λ}`. Hence `c(1) − c(0) = (4b²)^{-1} G_λ(F)` with
`G_λ(F) = ∫ |F|² e^{iφ/λ} dt ∈ C ≅ R²` (λ = 1: `∫|q|q`, the SRVF condition).
Independent of the starting branch (a branch shift rotates the curve, closure is rotation-
invariant), homogeneous of degree 2 (norm rescaling keeps closure).
L² gradients (variation `δF = (u + i w) e^{iφ}`, `u = δr`, `w = r δφ`):
`δG = ∫ r e^{iφ/λ} (2u + (i/λ) w)` ⇒
`b_x = 2cos(φ/λ) F − (1/λ) sin(φ/λ) iF`, `b_y = 2 sin(φ/λ) F + (1/λ) cos(φ/λ) iF`
(λ = 1 reduces to fdasrsf `Basis_Normal_A`: `|q| e_k + q_k q/|q|`).
Projection: Gauss–Newton along the normals, `J = [⟨b_i,b_k⟩]`, `x = −J⁻¹G`, backtracking on
`|G|`, on the unit-norm copy, norm restored. Discrete consistency: trapezoid weights = the
cumulative-trapezoid reconstruction endpoint, so the reconstructed curve closes exactly.
λ = 1 dispatches to the existing SRVF projector (bit-identical SRVF path).

### Exact PL distance
Corollary 2.1 + Thm 3.1 of Bauer et al.: in arc-length parametrisation
`d² = 4b² (L1 + L2 − 2 sup_γ ∫ κ(γ1,γ2) √(γ1'γ2'))`, `κ_ij = cos(λθ_ij)` if `λθ_ij < π/2` else 0,
θ_ij ∈ [0,π] the angle between segment i of c1 and segment j of c2: rectangular weight.
Lahiri et al. Thm 6: an optimal matching is a chain of P-segments (vertex → vertex, straight
in positive cells, refraction `H'/H = (κ'/κ)^{±2}`, horizontal/vertical through zero cells) and
N-segments (edge moves, value 0). Algorithm: longest path over grid vertices; from each
vertex whose NE cell is positive, sweep the P-segment rays by initial slope h with the
"straightening" coordinates (column scale α, row scale β, αβ = (κ/κ0)²; zero cells collapse):
the ray is `Y = hX`, events = corner ratios `Y/X`, the next vertex hit is the smallest ratio
above the current h, its value `κ0 √(XY)`. Polynomial: O(N1²N2²(N1+N2)).

## Public API
- `ab_closure_residual(f, a, b) -> Vec<f64>`, `ab_project_closed(f, a, b) -> FdMatrix`.
- `CurveShapeConfig::closed = true` allowed with `Elastic` (all curve_shape_* functions).
- `CurveShapeConfig::branch_search: bool` (default false): with `Elastic` and `rotation = false`,
  minimise over the relative branch `e^{2πiλk}`, k ∈ {−1, 0, 1} (curve itself not rotated).
- `ab_elastic_distance_pl(c1, c2, a, b) -> f64`, `ab_elastic_match_pl(...) -> AbPlMatching`
  (distance + optimal matching polyline in normalised arc length).

## Validation
- Closure: residual < 1e-9 after projection, idempotent, norm kept, reconstructed curve closed,
  general Newton at λ = 1 vs SRVF projector, finite-difference check of the normals.
- Elastic{1,1/2} closed = SRVF closed (distance/rotation/seed/γ + Karcher mean).
- Invariance: rotation + seed + reparam of closed contours, Karcher mean of closed contours
  stays closed, tangent vectors orthogonal to the F normals.
- geomstats 2.8.0 has no closed-curve support (checked: no `closed` in discrete_curves.py) →
  invariants only.
- PL: hand closed forms (two segments; one-segment target `√(L2 Σκ_i²ℓ_i)`; scaled copy
  `2b|√L1 − √L2|`; subdivision invariance → 0), symmetry, translation, √s scaling,
  independent test-side DP on the rectangular weight (valid lower bound of the sup → upper
  bound of d, converging to the exact value), ≤ curve_shape DP distance (rotation=false,
  scale=false) and convergence of the latter as m grows.

## Tasks
1. geometry: `closure_vec_ab`, `closure_normals_ab`, `project_closed_ab`; λ-aware
   `project_closed` / `project_tangent` call sites.
2. mod.rs: lift validation, public helpers, `branch_search`.
3. new `pl_exact.rs` + tests; re-exports; CHANGELOG; gates; commits.
