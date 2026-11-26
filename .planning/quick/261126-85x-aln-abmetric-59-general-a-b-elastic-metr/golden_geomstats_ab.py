"""Golden values for ALN-ABMETRIC (GH #59): geomstats 2.8.0 FTransform / ElasticMetric(a, b).

Run: ~/.cache/fdars-golden-venv/bin/python golden_geomstats_ab.py
(geomstats 2.8.0, numpy backend; numpy 2.3.5)

geomstats discretisation: curve points p_0 = 0, p_1..p_{k-1}; velocity
v_i = (k-1)(p_{i+1} - p_i), i = 0..k-2; F_i = 2b |v_i|^{1/2} exp(i * lam * theta_i),
lam = a/(2b), theta_0 = atan2(v_0), theta_i = theta_{i-1} + signed angle(v_{i-1}, v_i);
L2 = left Riemann sum with dt = 1/(k-1).
"""
import numpy as np
import geomstats.backend as gs
from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    ElasticMetric,
    FTransform,
)
from geomstats.geometry.euclidean import Euclidean

np.set_printoptions(precision=17)


def spiral(k):
    t = np.linspace(0.0, 1.0, k)
    r = 0.3 + t
    ang = 3.0 * np.pi * t + 2.5  # starts near pi, turns 1.5 revolutions
    pts = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=-1)
    return pts - pts[0]


ab_list = [(1.0, 0.5), (1.0, 2.0), (1.0, 1.0), (3.0, 1.0), (2.0, 0.5)]

# 1. pointwise transform on a 9-point spiral
k = 9
pts = spiral(k)
vel = (k - 1) * np.diff(pts, axis=0)
print("// velocity (x row then y row), k-1 =", k - 1)
print("vx", ",".join(repr(v) for v in vel[:, 0]))
print("vy", ",".join(repr(v) for v in vel[:, 1]))
for a, b in ab_list:
    ft = FTransform(Euclidean(2), k, a, b)
    f = np.asarray(ft(gs.array(pts[1:])))
    print(f"F a={a} b={b}")
    print(" fx", ",".join(repr(v) for v in f[:, 0]))
    print(" fy", ",".join(repr(v) for v in f[:, 1]))
    inv = np.asarray(ft.inverse(gs.array(f)))
    inv = np.vstack([np.zeros(2), inv])
    print(" inverse max err vs input", np.abs(inv - pts).max())

# 2. distance without reparametrisation, smooth curves, fine grid
def c1(t):
    return np.stack([t, 0.3 * np.sin(2 * np.pi * t)], axis=-1)


def c2(t):
    return np.stack([np.cos(1.2 * t) + t, 0.5 * t * t - 0.2 * np.sin(5 * t)], axis=-1)


for kk in (9, 2001):
    t = np.linspace(0.0, 1.0, kk)
    p1 = c1(t) - c1(t)[0]
    p2 = c2(t) - c2(t)[0]
    space = DiscreteCurvesStartingAtOrigin(ambient_dim=2, k_sampling_points=kk, equip=False)
    for a, b in ab_list:
        space.equip_with_metric(ElasticMetric, a=a, b=b)
        d = float(space.metric.dist(gs.array(p1[1:]), gs.array(p2[1:])))
        print(f"dist k={kk} a={a} b={b}: {d!r}")
