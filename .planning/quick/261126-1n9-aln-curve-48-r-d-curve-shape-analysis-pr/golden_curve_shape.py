"""Golden values for ALN-CURVE (#48) from fdasrsf 2.7.5.

Run: ~/.cache/fdars-golden-venv/bin/python golden_curve_shape.py
Curves are reproduced bit-for-bit in fdars-core/src/alignment/curve_shape/tests.rs.
"""
import io
import contextlib
import numpy as np
import fdasrsf
import fdasrsf.curve_functions as cf
import fdasrsf.curve_stats as cs


def open1(T):
    t = np.linspace(0, 1, T)
    return np.vstack([t, 0.3 * np.sin(2 * np.pi * t) + 0.1 * t ** 2])


def open2(T):
    t = np.linspace(0, 1, T)
    s = t + 0.15 * t * (1 - t)
    return np.vstack([s + 0.05 * np.sin(np.pi * s), 0.35 * np.sin(2 * np.pi * s) + 0.2 * s ** 3])


def open3d_1(T):
    t = np.linspace(0, 1, T)
    return np.vstack([t, 0.3 * np.sin(2 * np.pi * t), 0.2 * np.cos(np.pi * t)])


def open3d_2(T):
    t = np.linspace(0, 1, T)
    s = t ** 1.2
    return np.vstack([s, 0.25 * np.sin(2 * np.pi * s) + 0.05 * s, 0.25 * np.cos(np.pi * s)])


def closed1(T):
    th = np.linspace(0, 2 * np.pi, T)
    return np.vstack([1.5 * np.cos(th), 0.8 * np.sin(th)])


def closed2(T):
    th = np.linspace(0, 2 * np.pi, T)
    r = 1.0 + 0.2 * np.cos(3 * th)
    return np.vstack([r * np.cos(th), r * np.sin(th)])


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def family(T, i):
    t = np.linspace(0, 1, T)
    a = 0.2 + 0.05 * i
    return np.vstack([t, a * np.sin(np.pi * t) + 0.1 * i * t ** 2])


print("fdasrsf", fdasrsf.__version__)
T = 100
print("open2d rot  ", repr(cf.elastic_distance_curve(open1(T), open2(T), 0, True, False)[0]))
print("open2d norot", repr(cf.elastic_distance_curve(open1(T), open2(T), 0, False, False)[0]))
print("open3d norot", repr(cf.elastic_distance_curve(open3d_1(T), open3d_2(T), 0, False, False)[0]))
print("closed rot  ", repr(cf.elastic_distance_curve(closed1(T), closed2(T), 1, True, False)[0]))
b2 = rot(0.4) @ cf.shift_f(closed2(T), 12)
print("closed rot shifted", repr(cf.elastic_distance_curve(closed1(T), b2, 1, True, False)[0]))

# Karcher mean of a small open family (fdacurve resamples to N=T, no rotation search off)
Tk = 50
beta = np.stack([family(Tk, i) for i in range(4)], axis=2)
obj = cs.fdacurve(beta, mode="O", N=Tk)
with contextlib.redirect_stdout(io.StringIO()):
    obj.karcher_mean(rotation=True)
print("karcher beta_mean x", np.array2string(obj.beta_mean[0], separator=", ", max_line_width=10**6))
print("karcher beta_mean y", np.array2string(obj.beta_mean[1], separator=", ", max_line_width=10**6))
print("karcher E", obj.E, "qun", obj.qun)

# DP-floor references used to justify invariance tolerances in tests.rs
for TT in (40, 80):
    t = np.linspace(0, 1, TT)
    g = t + 0.25 * t * (1 - t)
    b = open1(TT)
    bw = np.vstack([np.interp(g, t, b[0]), np.interp(g, t, b[1])])
    print("floor open self T=%d" % TT, cf.elastic_distance_curve(b.copy(), bw.copy(), 0, False, False)[0])
t = np.linspace(0, 1, 61)
g = t + 0.2 * t * (1 - t)
b = closed2(61)
bw = np.vstack([np.interp(g, t, b[0]), np.interp(g, t, b[1])])
bm = rot(-0.7) @ cf.shift_f(bw, 23)
print("floor closed self moved T=61", cf.elastic_distance_curve(b.copy(), bm.copy(), 1, True, False)[0])
