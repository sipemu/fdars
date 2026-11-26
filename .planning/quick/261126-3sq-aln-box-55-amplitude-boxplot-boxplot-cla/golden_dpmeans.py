# ALN-BOX (#55): fdasrsf 2.7.5 kmeans.dpmeans_align on the Rust test's three_groups data.
import numpy as np, io, contextlib
from fdasrsf.kmeans import dpmeans_align
def three_groups(per, m):
    t = np.linspace(0, 1, m); n = 3 * per; f = np.zeros((m, n))
    for i in range(n):
        g = i % 3; r = i // 3; a = 1.0 + 0.07 * r; c = 0.15 * (r / per - 0.5)
        x = t + c * t * (1 - t)
        f[:, i] = [a * np.sin(np.pi * x), a * np.sin(4 * np.pi * x), 3 * a * x * x][g]
    return f, t
for per, m, lams in [(5, 41, [1.0, 1e6, 0.0]), (4, 31, [0.0, 0.05, 0.2, 1.0, 5.0, 100.0])]:
    f, t = three_groups(per, m)
    for lam in lams:
        with contextlib.redirect_stdout(io.StringIO()):
            out = dpmeans_align(f, t, lam=lam, showplot=False, omethod="DP2")
        print(per, m, lam, "K", out["K"], "labels", list(out["labels"]), "qun", out["qun"][-1])
