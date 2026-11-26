---
status: complete
quick_id: 261126-ljn
issue: 68
---
# Summary — ALN-CAV (GH #68): (a,b) elastic metrics — closed curves, exact PL distance, branch search

## Commits
- `5e128f4a` feat(quick-261126-ljn): closed contours + exact PL distance for (a,b) elastic metrics (body `Closes #68`)
- docs commit (this PLAN + SUMMARY)

Worktree `/home/simonm/projects/rust/fdars-wt-68`, branch `cav-68`; not pushed. STATE.md intentionally not touched (orchestrator override).

## Public API added (re-exported from `alignment` and the crate root)
- `CurveShapeConfig::closed = true` with `CurveShapeMetric::Elastic { a, b }` — no longer rejected; works in every `curve_shape_*` function.
- `ab_closure_residual(f, a, b) -> Vec<f64>` (displacement of the rebuilt curve) and `ab_project_closed(f, a, b) -> FdMatrix`.
- `CurveShapeConfig::branch_search: bool` (default `false`, serde default).
- `ab_elastic_distance_pl(c1, c2, a, b) -> f64`, `ab_elastic_match_pl(c1, c2, a, b) -> AbPlMatching { distance, similarity, length1, length2, path }` (d × n vertex matrices, any d ≥ 1).

## Math / design
- **Closure condition (derived from the inverse transform).** `c' = |F|²/(4b²)·e^{iφ/λ}` ⇒ `c(1) − c(0) = (4b²)⁻¹ ∫|F|² e^{iφ/λ}` (`φ` the lifted arg F; λ = 1 gives `∫|q|q`). Branch-independent (a branch shift rotates the rebuilt curve), homogeneous of degree 2. L² gradients `b_x = 2cos(φ/λ)F − λ⁻¹sin(φ/λ)iF`, `b_y = 2sin(φ/λ)F + λ⁻¹cos(φ/λ)iF` (= fdasrsf `Basis_Normal_A` at λ = 1). Projection: Gauss–Newton with the exact Gram Jacobian + backtracking on the scale-free residual `|G|/‖F‖²`, norm restored. Trapezoid weights = cumulative-trapezoid endpoint, so the rebuilt curve closes to the projection tolerance.
- λ = 1 (any b) dispatches to the existing SRVF projector → SRVF path and Elastic{1,1/2} bit-identical to before. `project_tangent` gained a λ argument (SRVF callers pass 1.0 → unchanged normals).
- Seed search on closed contours shifts the *curve* and recomputes F (F of a closed curve is not periodic for non-integer λ: `F(1) = F(0)e^{2πiλ·turning}`), so exact sample shifts are recovered exactly.
- **Branch search:** with `rotation = false`, F2 is also tried at `e^{±2πiλ}` (every relative initial-angle lift in (−π, π]); the reported curve rotation stays the identity.
- **Exact PL distance** (Bauer et al. 2024, Cor. 2.1 + Thm 3.1 + §4.1; Lahiri–Robinson–Klassen 2015 Thm 6 + §8): `d² = 4b²(L1 + L2 − 2S)`, rectangular weight `κ_ij = cos(λθ_ij)` if `λθ_ij < π/2` else 0. Optimal matching = chain of N-segments (edge moves, value 0) and P-segments. Implementation: longest-path DP over grid vertices (row-major); from each vertex with a positive NE cell, sweep the P-segment rays by initial slope `h` in "straightened" coordinates (column scale α, row scale β, αβ = (κ/κ0)², zero cells collapse) where the ray is `Y = hX`; the next vertex hit is the smallest corner ratio above `h`, value `κ0√(XY)`. All initial slopes are enumerated (no Theorem-6(3) slope pruning — pruning is only an optimisation; without it the DP is still exact). `O(N1²N2²(N1+N2))`. Path backtracking retraces each P-segment (bit-reproducible stop at the event ratio).
- The PL distance uses the pointwise angle θ ∈ [0, π] (global geodesic distance for d ≥ 3; geodesic completion for d = 2). It coincides with the F-transform DP integrand while the lifted tangent-angle difference stays within π/(2λ); outside that range the F-transform chord can be smaller (it is only a local isometry) — documented on the function.

## Validation
- `ab_closed_tests.rs` (14 tests): residual = rebuilt displacement (1e-12, 4 λ's) and = SRVF residual at (1,1/2); closure normals vs central finite differences (1e-7 rel, λ ∈ {¼,½,1,3/2,2}); λ = 1 functional/normals = SRVF ones (1e-12); projection closes (< 1e-9‖F‖²), keeps norm (1e-12), rebuilt curve closed (1e-9), idempotent (1e-12) for 3 open curves × 4 λ, barely moves a sampled ellipse (< 1e-3 rel); `ab_project_closed(·,1,½)` bit-identical to `project_closed_srvf`, general GN at λ = 1 lands on the SRVF manifold and agrees with the SRVF projector to < 5% of the move for a nearly closed contour; Elastic{1,½} closed = SRVF closed (seed equal, distance/rotation/γ/aligned curve/aligned transform 1e-10, with and without reparam) and closed Karcher mean (1e-8, same seeds/iterations); exact seed + rotation + scale recovery (distance < 1e-9, seed exact, rotation for λ ≤ 1), seed search needed; distance invariance to warp+seed+rotation+scale (within 0.03, the DP / sub-sample-seed floor), self-distance of a moved copy < 0.08 and < 0.3·d, near symmetry; aligned transforms rebuild closed curves; closed Karcher mean closed (1e-9) with shooting vectors ⟂ F-normals and mean (1e-10); PCA mode shape closed, TSRVF round trip, depth; branch search never worse, no curve rotation, equals the distance of the π-rotated pair (no cut) to 1e-9, strictly better for non-integer λ; ignored for SRVF / rotation = true (identical results); validation.
- `pl_exact_tests.rs` (10 tests): two single segments closed form (1e-12, incl. κ = 0); one segment vs polygon `S = √(L Σκ_j²ℓ_j)` both orders (1e-12, incl. zero cells); scaled copy `2b|√L1 − √L2|` (1e-7); subdivision + repeated vertices → 0; symmetry (1e-12), translation, `√s` scaling (on d²), triangle inequality; the returned path is monotone from (0,0) to (1,1) and its exact value equals `similarity` (1e-9); **independent brute-force DP** (straight moves on a node grid, each evaluated exactly on the rectangular weight → admissible → lower bound): never above the exact supremum, gap shrinking as nodes + slopes refine, < 2e-3 relative at (48 nodes, 24-move) for 5 (a,b) pairs × 2 polygon pairs; 2×2-segment hand case within 1e-4 at (64, 32); curve_shape F-transform DP (rotation = false, scale = false) on densely sampled polygons: relative gap −1.3% → −0.3% (m = 101 → 401, the O(1/m) corner-rounding of central differences) and < 5% (DP slope floor) — exact ≤ DP up to that sampling error; R³ polygons symmetric, self-distance 0; validation errors.
- geomstats 2.8.0: **no closed-curve support** for `ElasticMetric` (no "closed" anywhere in `geometry/discrete_curves.py`) → closed case validated by identities, finite differences and invariants only. No reference implementation of the exact PL algorithm was available (Bauer et al. use Bruveris' libsrvf C++ code, not packaged); validated against closed forms and the independent brute-force DP instead.
- Existing curve_shape tests (SRVF + open (a,b), 79 + goldens) unchanged and passing; one assertion in `ab_tests::elastic_metric_validation` updated from "closed + elastic is an error" to "is ok" (the behaviour change requested by the issue).

## Gates
- `cargo fmt --all` clean.
- `cargo test -p fdars-core --lib curve_shape`: 93 passed, 1 ignored (pre-existing diagnostic `#[ignore]` in optim_tests).
- `cargo test -p fdars-core --doc curve_shape`: 23 passed (incl. new doc examples of `ab_closure_residual`, `ab_project_closed`, `branch_search`, `ab_elastic_match_pl`, `ab_elastic_distance_pl`).
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`: clean.
- `cargo check -p fdars-core --features serde`: clean.
- `RUSTDOCFLAGS=-D warnings cargo doc -p fdars-core --no-deps`: clean.
- pmat complexity: no new function above cognitive 23 (`trace` / `solve` decomposed into `cell_step` / `relax_from`).

## Deviations / notes
- With λ > 1 the F-transform is not injective modulo rotation (curves rotated by 2π/λ share F up to branch), so the *reported* curve rotation of a closed alignment can differ from the true one by a multiple of 2π/λ (pre-existing convention `Rot(ψ/λ)` from #59); distances and the F-space outputs are unaffected. Tests check rotation recovery for λ ≤ 1 only.
- Lahiri et al.'s Theorem-6(3) slope-window pruning of the searchlight is not implemented (all P-segments are enumerated; still exact, polynomial).
- The exact PL function has no rotation/scale quotient (as in Bauer et al. §4.1); closed PL curves (seed over vertices) not added.
- Branch search covers k ∈ {−1, 0, 1} (relative initial lifts); larger |k| would only matter for curves whose lifted angle differences exceed 2π, where the F chord is not a meaningful distance anyway.

## Remaining
- Nothing required by #68. Optional future work: rotation-quotiented PL distance, closed PL curves (vertex seed search), Theorem-6(3) pruning for speed.
