"""Golden values for ALN-SHIFT (GH #46).

Reference: scikit-fda 0.10.1 (scipy 1.18.1, findiff 0.16.1), run in
~/.cache/fdars-golden-venv.

NOTE on sign convention: scikit-fda registers x_i(t + d_i); fdars registers
f_i(t - delta_i), so delta_fdars = -d_skfda. This script prints skfda's raw d
values; the Rust tests negate them.

Data (reproduced exactly in Rust): m = 51 points on [0, 1],
f_i(t) = sin(2 pi (t - c_i)) + a_i * cos(4 pi (t - c_i)) + b_i * t
"""
import numpy as np
import skfda
from skfda.preprocessing.registration import (
    LeastSquaresShiftRegistration,
    landmark_shift_deltas,
    landmark_shift_registration,
)

np.set_printoptions(precision=17)

m = 51
t = np.linspace(0.0, 1.0, m)
c = np.array([0.03, -0.05, 0.08, -0.02, 0.06])
a = np.array([0.3, 0.25, 0.35, 0.2, 0.4])
b = np.array([0.1, -0.1, 0.05, 0.0, -0.05])
X = np.array(
    [np.sin(2 * np.pi * (t - ci)) + ai * np.cos(4 * np.pi * (t - ci)) + bi * t
     for ci, ai, bi in zip(c, a, b)]
)
fd = skfda.FDataGrid(X, t)


def show(name, reg, out):
    print(f"== {name}")
    print("deltas_skfda =", repr(reg.deltas_.tolist()))
    print("n_iter =", reg.n_iter_)
    print("grid_len =", len(out.grid_points[0]), "grid0 =", out.grid_points[0][0],
          "gridN =", out.grid_points[0][-1])
    print("row0[:3] =", repr(out.data_matrix[0, :3, 0].tolist()))
    print("row4[-3:] =", repr(out.data_matrix[4, -3:, 0].tolist()))
    print("template[::10] =", repr(reg.template_.data_matrix[0, ::10, 0].tolist()))


# 1. defaults (mean template, max_iter=5, tol=1e-2, data's linear extrapolation)
reg = LeastSquaresShiftRegistration()
out = reg.fit_transform(fd)
show("default_linear", reg, out)

# 2. fixed 10 iterations (mean template does not pin the common shift, so it drifts slowly), bounds extrapolation (data extrapolation also "bounds")
fdb = skfda.FDataGrid(X, t, extrapolation="bounds")
reg = LeastSquaresShiftRegistration(max_iter=10, tol=1e-12)
out = reg.fit_transform(fdb)
show("bounds_10iter", reg, out)

# 3. fixed template sin(2 pi t), periodic extrapolation
fdp = skfda.FDataGrid(X, t, extrapolation="periodic")
tmpl = skfda.FDataGrid(np.sin(2 * np.pi * t)[None, :], t)
reg = LeastSquaresShiftRegistration(template=tmpl, max_iter=20, tol=1e-8)
out = reg.fit_transform(fdp)
show("fixed_periodic", reg, out)

# 4. callable template (pointwise median), zeros extrapolation, step 0.8, initial
fdz = skfda.FDataGrid(X, t, extrapolation="zeros")


def median_template(x):
    return x.copy(data_matrix=np.median(x.data_matrix, axis=0, keepdims=True),
                  sample_names=(None,))


init = np.array([0.01, -0.02, 0.0, 0.02, -0.01])
reg = LeastSquaresShiftRegistration(template=median_template, max_iter=7,
                                    tol=1e-6, step_size=0.8, initial=init.tolist())  # list: skfda 0.10.1 crashes on ndarray `initial` (`== "zeros"` on an array)
out = reg.fit_transform(fdz)
show("median_zeros_step", reg, out)

# 5. restrict_domain, linear (defaults; with more iterations skfda 0.10.1 hits an
#    AssertionError because it restricts fd_deriv cumulatively and cannot re-widen)
reg = LeastSquaresShiftRegistration(restrict_domain=True)
out = reg.fit_transform(fd)
show("restrict_linear", reg, out)

# 6. landmark shift: landmark = argmax location (grid), mean location
lm = np.array([0.31, 0.19, 0.37, 0.24, 0.33])
print("== landmark")
print("deltas_mean =", repr(landmark_shift_deltas(fd, lm).tolist()))
print("deltas_fixed_0.3 =", repr(landmark_shift_deltas(fd, lm, location=0.3).tolist()))
out = landmark_shift_registration(fd, lm)
print("reg_mean row1[:3] =", repr(out.data_matrix[1, :3, 0].tolist()))
print("reg_mean row2[-3:] =", repr(out.data_matrix[2, -3:, 0].tolist()))
out = landmark_shift_registration(fdb, lm, location=0.3, restrict_domain=True)
print("reg_fixed_restrict grid_len =", len(out.grid_points[0]),
      out.grid_points[0][0], out.grid_points[0][-1])
print("reg_fixed_restrict row1[:3] =", repr(out.data_matrix[1, :3, 0].tolist()))
print("reg_fixed_restrict row2[-3:] =", repr(out.data_matrix[2, -3:, 0].tolist()))

# 7. restrict_domain again, but with scipy.integrate.simpson patched inside skfda
#    to use fdars' even-N rule (Simpson on the first N-1 points + one trapezoid
#    panel). Proves that the difference to case 5 is purely the quadrature rule.
import scipy.integrate
import skfda.representation.grid as _grid

_orig_simpson = scipy.integrate.simpson


def _fdars_simpson(y, x=None, axis=-1, **kw):
    y = np.asarray(y)
    if y.shape[axis] % 2 == 1:
        return _orig_simpson(y, x=x, axis=axis)
    y = np.moveaxis(y, axis, -1)
    h = x[1] - x[0]
    return _orig_simpson(y[..., :-1], x=x[:-1], axis=-1) + 0.5 * h * (y[..., -2] + y[..., -1])


_grid.scipy = type("m", (), {"integrate": type("i", (), {"simpson": staticmethod(_fdars_simpson)})})
reg = LeastSquaresShiftRegistration(restrict_domain=True)
out = reg.fit_transform(fd)
show("restrict_linear_fdars_quadrature", reg, out)
