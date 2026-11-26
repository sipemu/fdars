import distance.Bounds;
import distance.elastic.DTW;
public class Gold {
  static double[][] S = {
{-0.59, -0.92, -2.44, -4.33, -3.27, -2.99, -3.49, -2.83, -0.97, -2.79, -2.44, -3.42, -3.1, -3.44, -4.56, -4.49},
{1.14, 1.7, 1.02, 1.69, 2.88, 3.76, 3.94, 4.14, 4.72, 4.09, 2.86, 1.82, 1.97, -0.23, 0.36, 1.28},
{1.09, 2.25, 2.36, 3.14, 3.13, 4.72, 2.86, 3.11, 1.99, 1.28, 0.85, 0.06, -0.61, -0.12, 0.48, 2.56},
{-0.85, -1.55, -2.0, -2.43, -1.58, -2.38, -2.84, -1.37, -2.93, -2.79, -1.04, 0.74, 1.12, 0.77, 0.51, 0.48}};
  public static void main(String[] a) {
    int l = 16;
    for (int w = 0; w <= 3; w++) for (int qi = 0; qi < 4; qi++) for (int ti = 0; ti < 4; ti++) {
      if (qi == ti) continue;
      double[] q = S[qi], t = S[ti];
      double[] uq = new double[l], lq = new double[l], ut = new double[l], lt = new double[l];
      Bounds.simpleGetEnvelopes(q, w, uq, lq); Bounds.simpleGetEnvelopes(t, w, ut, lt);
      double[] d1 = new double[l], d2 = new double[l];
      double[] ulq = new double[l], luq = new double[l], ult = new double[l], lut = new double[l];
      Bounds.simpleGetEnvelopes(lq, w, ulq, d1); Bounds.simpleGetEnvelopes(uq, w, d2, luq);
      Bounds.simpleGetEnvelopes(lt, w, ult, d1); Bounds.simpleGetEnvelopes(ut, w, d2, lut);
      double keogh = Bounds.lbKeogh(q, ut, lt, Double.MAX_VALUE);
      double webb = Bounds.lbWebb(q, uq, lq, luq, ulq, t, ut, lt, lut, ult, w, Double.MAX_VALUE);
      double dtw = DTW.distance(q, t, w);
      System.out.println("(" + w + ", " + qi + ", " + ti + ", " + keogh + ", " + webb + ", " + dtw + "),");
    }
  }
}
