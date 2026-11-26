"""Golden reference for fdars-core `bayesian_align_group` (GH #52 ALN-BAYESGRP).

fdasrsf 2.7.5 `fdawarp.time_warping_bayes` on the synthetic set used in
`alignment/bayesian_group_tests.rs` (6 curves g0 ∘ γ_a⁻¹, exponential warps, m = 41,
no noise). Prints the posterior-mean warps and template (fmean) on the 41-point grid,
averaged over 4 seeds (chains) to reduce Monte-Carlo error.

Run: ~/.cache/fdars-golden-venv/bin/python golden_time_warping_bayes.py
"""
import numpy as np
import fdasrsf as fs

A = [-0.9, -0.5, -0.15, 0.2, 0.55, 0.85]
m = 41
t = np.linspace(0, 1, m)


def g0(s):
    return np.exp(-(((s - 0.35) / 0.1) ** 2)) + 0.7 * np.exp(-(((s - 0.7) / 0.08) ** 2))


def winv(a, s):
    return s if abs(a) < 1e-12 else np.log1p(s * np.expm1(a)) / a


f = np.column_stack([g0(winv(a, t)) for a in A])
gams, fmeans, sig = [], [], []
for seed in range(4):
    np.random.seed(seed)
    obj = fs.fdawarp(f, t)
    obj.time_warping_bayes(iter=20000, burnin=10000, nbasis=16, npoints=m,
                           initcoef=np.zeros((6, len(A))), verbose=False)
    gams.append(obj.gam)
    fmeans.append(obj.fmean)
    sig.append(np.mean(obj.mcmc["sigma1"][10000:] ** 2))
gam = np.mean(gams, axis=0)
fmean = np.mean(fmeans, axis=0)
np.set_printoptions(precision=5, suppress=True, linewidth=120)
print("fdasrsf", fs.__version__ if hasattr(fs, "__version__") else "2.7.5")
print("sigma2 per seed", sig)
print("seed spread gam (max abs)", np.max(np.abs(np.array(gams) - gam)))
print("GAM_REF (n x m, row = curve):")
for i in range(len(A)):
    print("    [" + ", ".join(f"{v:.5f}" for v in gam[:, i]) + "],")
print("FMEAN_REF:")
print("    [" + ", ".join(f"{v:.5f}" for v in fmean) + "],")
