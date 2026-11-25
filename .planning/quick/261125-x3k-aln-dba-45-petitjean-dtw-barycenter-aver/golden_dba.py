"""Golden values for fdars-core DBA (ALN-DBA, GH #45).

tslearn 0.10.0, dtaidistance 2.5.1 (numpy, python3 venv ~/.cache/fdars-golden-venv).
"""
import math
import warnings

import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from dtaidistance import dtw_barycenter

warnings.simplefilter("ignore")


def gen_d1(n, m):
    X = np.zeros((n, m))
    for i in range(n):
        for t in range(m):
            u = t / (m - 1)
            shift = ((i * 37) % 11) / 11.0 * 0.15
            noise = ((i * 7919 + t * 104729) % 1009) / 1009.0
            X[i, t] = (math.sin(2.0 * math.pi * (u - shift) * 1.5 + 0.08 * i)
                       + 0.25 * math.cos(5.0 * u + i) + 0.05 * noise)
    return X


def gen_k(n, m):
    X = np.zeros((n, m))
    for i in range(n):
        for t in range(m):
            u = t / (m - 1)
            noise = ((i * 7919 + t * 104729) % 1009) / 1009.0
            if i % 2 == 0:
                c = 0.3 + 0.05 * (i % 5)
                v = math.exp(-((u - c) / 0.08) ** 2)
            else:
                c = 0.35 + 0.05 * (i % 5)
                v = -math.exp(-((u - c) / 0.1) ** 2) + 0.5 * math.exp(-((u - c - 0.3) / 0.06) ** 2)
            X[i, t] = v + 0.05 * noise
    return X


def p(name, arr):
    arr = np.asarray(arr).ravel()
    print(f"{name} = [" + ", ".join(repr(float(v)) for v in arr) + "]")


def cost_of(X, bary, mp=None, w=None):
    mp = dict(mp or {})
    if mp.get("global_constraint") == 2:
        mp["global_constraint"] = "sakoe_chiba"
    d = cdist_dtw(bary[None, :], X, **mp)[0] ** 2
    w = np.ones(len(X)) if w is None else w
    return float((d * w).sum() / w.sum())


X = gen_d1(6, 20)
print("# D1 checksum", repr(float(X.sum())))
for label, kw in [
    ("MEAN_NONE", {}),
    # NB: DBA forwards metric_params raw to the numba kernel, which needs the
    # integer constraint code (2 = sakoe_chiba); the string form is silently ignored.
    ("MEAN_SC1", {"metric_params": {"global_constraint": 2, "sakoe_chiba_radius": 1}}),
    ("MEAN_IT2", {"metric_params": {"itakura_max_slope": 2.0}}),
    ("MEAN_SIZE12", {"barycenter_size": 12}),
    ("WEIGHTED", {"weights": np.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.25])}),
    ("CUSTOM_INIT_ROW3", {"init_barycenter": X[3].copy()}),
]:
    b = dtw_barycenter_averaging(X, max_iter=30, tol=1e-5, **kw)[:, 0]
    p(label, b)
    print(f"{label}_COST = {cost_of(X, b, kw.get('metric_params'), kw.get('weights'))!r}")

# dtaidistance: one plain DBA step and a 10-step loop from a fixed init (row 2).
c0 = X[2].copy()
step1 = dtw_barycenter.dba(list(X), c0.copy())
p("DTAI_STEP1_FROM_ROW2", step1)
loop = dtw_barycenter.dba_loop(list(X), c=c0.copy(), max_it=10, thr=None)
p("DTAI_LOOP10_FROM_ROW2", loop)

# k-means with DBA centroids from a fixed init (rows 0 and 1).
K = gen_k(10, 24)
print("# K checksum", repr(float(K.sum())))
km = TimeSeriesKMeans(n_clusters=2, metric="dtw", init=K[[0, 1]][:, :, None], n_init=1,
                      max_iter=50, tol=1e-6, random_state=0).fit(K)
print("KM_LABELS =", list(int(v) for v in km.labels_))
p("KM_C0", km.cluster_centers_[0, :, 0])
p("KM_C1", km.cluster_centers_[1, :, 0])
d = cdist_dtw(K, km.cluster_centers_) ** 2
print("KM_INERTIA_FINAL =", repr(float(d[np.arange(len(K)), km.labels_].mean())))
print("KM_N_ITER =", km.n_iter_)


# Medoid init: argmin_i sum_k DTW^2(x_i, x_k), then DBA from that series.
D = cdist_dtw(X, X) ** 2
med = int(np.argmin(D.sum(axis=1)))
print("MEDOID_INDEX =", med)
bm = dtw_barycenter_averaging(X, max_iter=30, tol=1e-5, init_barycenter=X[med].copy())[:, 0]
p("MEDOID_INIT", bm)
print(f"MEDOID_INIT_COST = {cost_of(X, bm)!r}")
