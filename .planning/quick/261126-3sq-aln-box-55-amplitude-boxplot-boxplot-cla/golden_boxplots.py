# Golden values for ALN-BOX (#55): fdasrsf 2.7.5 ampbox / phbox construct_boxplot + classify.
# Inputs are analytic so the Rust tests regenerate them bit-for-bit (no DP involved:
# fdawarp is mocked with the aligned arrays directly).
import numpy as np
import fdasrsf
from fdasrsf.boxplots import ampbox, phbox
np.set_printoptions(precision=17)
print("fdasrsf", fdasrsf.__version__ if hasattr(fdasrsf, "__version__") else "2.7.5")

M = 41
t = np.linspace(0, 1, M)
TP = 2 * np.pi

# amplitude: f_i = a_i sin(2πt) + b_i t^2,  q = sign(f') sqrt|f'|
A = [1.0, 1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85, 1.0, 1.3, 0.7, 1.02, 2.5, 1.08]
B = [0.5, 0.4, 0.6, 0.55, 0.45, 0.3, 0.7, 0.5, 0.52, 0.8, 0.2, 0.5, 0.48, 0.5, 0.35]
def fq(a, b):
    f = a * np.sin(TP * t) + b * t**2
    d = a * TP * np.cos(TP * t) + 2 * b * t
    return f, np.sign(d) * np.sqrt(np.abs(d))
fn = np.column_stack([fq(a, b)[0] for a, b in zip(A, B)])
qn = np.column_stack([fq(a, b)[1] for a, b in zip(A, B)])
fmed, qmed = fq(1.01, 0.49)

# phase: gam_i = t + c t(1-t) + e sin(2πt)/(2π)
C = [0.1, -0.2, 0.3, -0.1, 0.05, 0.2, -0.3, 0.15, -0.05, 0.25, -0.15, 0.85, 0.12, -0.22, 0.02]
E = [0.05, 0.1, -0.05, 0.02, -0.1, 0.0, 0.05, -0.08, 0.12, 0.03, -0.02, 0.1, -0.04, 0.06, 0.15]
def gm(c, e):
    return t + c * t * (1 - t) + e * np.sin(TP * t) / TP
gam = np.column_stack([gm(c, e) for c, e in zip(C, E)])

class W:
    pass
w = W()
w.fn = fn; w.qn = qn; w.mqn = qmed; w.fmean = fmed; w.time = t
w.rsamps = False; w.type = "median"; w.gam = gam

ab = ampbox(w); ab.construct_boxplot(alpha=0.05, k_a=1)
print("AMP dist", repr(ab.dist))
print("AMP r50 ralpha whisker", repr(ab.r50), repr(ab.ralpha), repr(ab.whisker_dis))
print("AMP idx Q1 Q3 Q1a Q3a", ab.Q1_index, ab.Q3_index, ab.Q1a_index, ab.Q3a_index)
print("AMP outliers", ab.outlier_index)
mn = [i for i in range(fn.shape[1]) if np.allclose(fn[:, i], ab.minn)]
mx = [i for i in range(fn.shape[1]) if np.allclose(fn[:, i], ab.maxx)]
print("AMP min max idx", mn, mx)

# new functions for classify
AN = [1.0, 1.25, 3.0, 0.75]; BN = [0.5, 0.5, 0.5, 0.5]
fnn = np.column_stack([fq(a, b)[0] for a, b in zip(AN, BN)])
qnn = np.column_stack([fq(a, b)[1] for a, b in zip(AN, BN)])
CN = [0.0001, 0.2, 0.95, -0.4]; EN = [0.01, 0.1, 0.0, 0.1]
gnn = np.column_stack([gm(c, e) for c, e in zip(CN, EN)])
w.align_new = lambda f_new, s=None: (fnn, qnn, gnn)
out = ab.classify(fnn)
print("AMPCLS dist", repr(out.dist)); print("AMPCLS level", out.level); print("AMPCLS side", out.side)

pb = phbox(w); pb.construct_boxplot(alpha=0.05, k_a=1)
print("PH dist", repr(pb.dist))
print("PH r50 ralpha whisker", repr(pb.r50), repr(pb.ralpha), repr(pb.whisker_dis))
print("PH idx Q1 Q3 Q1a Q3a", pb.Q1_index, pb.Q3_index, pb.Q1a_index, pb.Q3a_index)
print("PH outliers", pb.outlier_index)
print("PH median_x[::10]", repr(pb.median_x[::10]))
out = pb.classify(fnn)
print("PHCLS dist", repr(out.dist)); print("PHCLS level", out.level); print("PHCLS side", out.side)
