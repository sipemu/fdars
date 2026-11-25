---
status: complete
quick_id: 261126-0tx
issues: [46]
commits: [06c5c990]
---
# Summary — ALN-SHIFT (#46)

## Commits
- `06c5c990` feat(quick-261126-0tx): iterative least-squares shift registration + landmark shift registration (closes #46)

## Public API added (additive; re-exported from `alignment` and crate root)
- `iterative_shift_registration(data, argvals, ShiftTemplate, &IterativeShiftConfig) -> Result<IterativeShiftResult>`
- `ShiftTemplate<'a> { Mean (default), Fixed(&[f64]), Custom(&dyn Fn(&FdMatrix) -> Vec<f64>) }`
- `IterativeShiftConfig { max_iter = 5, tol = 1e-2, step_size = 1, restrict_domain = false, extrapolation = Boundary, initial = None }` (sealed, serde) + `validate()`
- `IterativeShiftResult { registered_data, registered_argvals, shifts, template, n_iter, converged }`
- `ShiftExtrapolation { Boundary (default), Linear, Periodic, Zeros }`; `shift_curves(...) -> ShiftedCurves { data, argvals }`
- `LandmarkLocation<'a> { Mean (default), MidRange, Fixed(f64), Custom(&dyn Fn(&[f64]) -> f64) }`
- `landmark_shift_deltas`, `landmark_shift_deltas_multi` (K landmarks/curve → LS common shift),
  `landmark_shift_registration(data, argvals, landmarks, location, &LandmarkShiftConfig) -> LandmarkShiftResult`
- Files: `fdars-core/src/alignment/shift_iterative.rs`, `fdars-core/src/alignment/landmark_shift.rs`.
- Existing `least_squares_shift_registration` (golden-section, single pass) not modified → identical results.

## Validation
- scikit-fda 0.10.1 golden values (scipy 1.18.1, findiff 0.16.1; script `golden_shift.py` in this dir):
  default (mean, linear extrap.), bounds 10 iters, fixed template + periodic, median callable + zeros +
  step 0.8 + initial, restrict_domain; shifts/registered values/templates/n_iter match to < 1e-10.
- restrict_domain: < 1e-12 vs scikit-fda with `scipy.integrate.simpson` patched to fdars' even-N rule
  (proves algorithm identity); < 1e-5 vs raw scikit-fda (observed 6.1e-6, quadrature-rule only).
- Landmark deltas (1e-15) and registration (mean/linear; fixed 0.3 + bounds + restrict) vs scikit-fda.
- Invariants: relative-shift recovery with mean template; template curve → exactly zero shift;
  max_iter = 0; constant curve; extrapolation hand values; NumPy `gradient(edge_order=2)` values;
  input validation. 5 doctests.

## Gates
- `cargo fmt --all` ✓
- `cargo test -p fdars-core --lib shift` (50 passed), `--lib landmark` (36 passed), `--doc shift` (6 passed) ✓
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` ✓ clean
- `cargo check -p fdars-core --features serde` ✓
- pmat complexity: max cognitive 14 in new files (≤ 23).

## Deviations from scikit-fda (documented in module docs)
- Sign: fdars registers fᵢ(t − δᵢ) ⇒ δ = −d_skfda (also for `initial` and landmark deltas).
- One extrapolation setting for iterations and output; default Boundary (skfda FDataGrid default is linear).
- restrict_domain interval recomputed each iteration (skfda restricts cumulatively and raises
  AssertionError when the interval re-widens — hit with max_iter=10 on the golden data).
- Even-N quadrature: fdars Simpson + trapezoid vs SciPy Cartwright.
- Zero Newton step when ∫Df² = 0 (skfda divides by zero). Template returned on full grid.
- skfda doc says landmark default location is mid-range; its code uses the mean — fdars follows code,
  offers `MidRange` explicitly.
- skfda 0.10.1 bug noted: ndarray `initial` crashes (`== "zeros"`), golden script passes a list.

## Remaining
None — issue fully addressed.
