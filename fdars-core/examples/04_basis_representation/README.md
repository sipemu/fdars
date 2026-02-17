# Example 04: Basis Representation and Smoothing

## What this demonstrates

Representing functional data in terms of basis function expansions (B-splines and Fourier), projecting noisy data onto a basis, reconstructing curves from coefficients, and using penalized B-splines (P-splines) for smooth fitting. Also demonstrates automatic Fourier basis selection using GCV.

Basis representations reduce functional data to a finite set of coefficients, enabling downstream analyses like regression and clustering.

## API functions used

- `basis::bspline_basis()` — evaluate B-spline basis on a grid
- `basis::fourier_basis()` — evaluate Fourier basis on a grid
- `basis::fdata_to_basis_1d()` — project data onto a basis
- `basis::basis_to_fdata_1d()` — reconstruct data from coefficients
- `basis::pspline_fit_1d()` — P-spline smoothing with penalty (returns GCV/AIC/BIC)
- `basis::fourier_fit_1d()` — Fourier basis fitting
- `basis::select_fourier_nbasis_gcv()` — automatic selection of Fourier basis dimension

## How to run

```bash
cargo run --example basis_representation
```

## Expected output

Basis matrix dimensions, projection and reconstruction errors, P-spline results at different penalty levels (λ) with GCV/AIC/BIC scores, Fourier fitting results, and automatic basis selection.

## Key concepts

- **B-splines**: piecewise polynomial basis with local support, good for general curves
- **Fourier basis**: sine/cosine functions, ideal for periodic data
- **P-splines**: B-spline basis + roughness penalty on coefficient differences
- **GCV**: generalized cross-validation, estimates prediction error without leave-one-out
