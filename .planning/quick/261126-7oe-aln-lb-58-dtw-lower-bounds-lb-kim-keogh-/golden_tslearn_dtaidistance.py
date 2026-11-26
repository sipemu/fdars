# Goldens for fdars-core/src/metric/dtw_lb/tests.rs (run in ~/.cache/fdars-golden-venv)
# tslearn 0.10.0, dtaidistance 2.5.1
import numpy as np
from tslearn.metrics import lb_envelope, lb_keogh, dtw
from dtaidistance import dtw as ddtw
x = [0.5, -1.2, 0.3, 2.0, 1.1, -0.4, 0.0, 0.9, -2.1, 0.7]
y = [0.1, 0.4, -0.8, 1.5, 2.2, 0.3, -0.6, 0.2, 0.5, -1.0]
for r in [0, 1, 2, 3]:
    lo, up = lb_envelope(y, radius=r)
    print(r, lo[:, 0].tolist(), up[:, 0].tolist())
    print(" keogh", repr(float(lb_keogh(x, y, radius=r))),
          "dtw", repr(float(dtw(x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=r))))
xa, ya = np.array(x), np.array(y)
for w in [1, 2, 3]:  # dtaidistance window = radius + 1; unequal lengths
    print(w, repr(ddtw.lb_keogh(xa[:7], ya, window=w) ** 2), repr(ddtw.lb_keogh(xa, ya[:6], window=w) ** 2))
