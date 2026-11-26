// Sealed-config idiom (`default()` + field assignment) mirrors external usage.
#![allow(clippy::field_reassign_with_default)]

use super::*;
use crate::test_helpers::ValuePin;

fn grid(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

fn two_bump(s: f64) -> f64 {
    (-(s - 0.3).powi(2) / 0.02).exp() + 0.8 * (-(s - 0.7).powi(2) / 0.02).exp()
}

/// The golden example: `yᵢ(t) = g(t + aᵢ t (1 − t))`, `aᵢ ∈ {−0.3, −0.1, 0.15, 0.3}`.
fn golden_data() -> (FdMatrix, Vec<f64>) {
    let t = grid(101);
    let a = [-0.3, -0.1, 0.15, 0.3];
    let mut data = FdMatrix::zeros(4, 101);
    for (i, ai) in a.iter().enumerate() {
        for j in 0..101 {
            data[(i, j)] = two_bump(t[j] + ai * t[j] * (1.0 - t[j]));
        }
    }
    (data, t)
}

fn golden_config(
    crit: ContinuousRegistrationCriterion,
    lambda: f64,
) -> ContinuousRegistrationConfig {
    let mut cfg = ContinuousRegistrationConfig::default();
    cfg.criterion = crit;
    cfg.nbasis = 6;
    cfg.norder = 4;
    cfg.lambda = lambda;
    cfg.lfd_order = 2;
    cfg.n_fine = Some(431);
    cfg
}

// ---------------------------------------------------------------------------
// Golden values: R fda 6.3.0 (script golden_register_fd.R in
// .planning/quick/261126-1a8-aln-cc-47-ramsay-li-continuous-registrat/).
// yfd = smooth.basis(t, Y, create.bspline.basis(c(0,1), 43, 6))$fd,
// wbasis = create.bspline.basis(c(0,1), 6, 4),
// register.fd(mean.fd(yfd), yfd, fdPar(fd(matrix(0,6,4), wbasis), int2Lfd(2), lambda),
//             conv = 1e-6, iterlim = 50, dbglev = 0, crit).
// ---------------------------------------------------------------------------

/// `eval.basis(c(0, 0.1, 0.37, 0.5, 0.9, 1), wbasis)`.
const R_BASIS: [[f64; 6]; 6] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.343, 0.54225, 0.11025, 0.0045, 0.0, 0.0],
    [0.0, 0.17624225, 0.59648475, 0.22694025, 0.00033275, 0.0],
    [0.0, 0.03125, 0.46875, 0.46875, 0.03125, 0.0],
    [0.0, 0.0, 0.0045, 0.11025, 0.54225, 0.343],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
];

/// `getbasispenalty(wbasis, int2Lfd(2))`.
const R_PENALTY: [[f64; 6]; 6] = [
    [324.0, -445.5, 94.5, 27.0, 0.0, 0.0],
    [-445.5, 648.0, -182.25, -30.375, 10.125, 0.0],
    [94.5, -182.25, 121.5, -30.375, -30.375, 27.0],
    [27.0, -30.375, -30.375, 121.5, -182.25, 94.5],
    [0.0, 10.125, -30.375, -182.25, 648.0, -445.5],
    [0.0, 0.0, 27.0, 94.5, -445.5, 324.0],
];

struct RCase {
    crit: ContinuousRegistrationCriterion,
    lambda: f64,
    /// `r$Wfd$coefs`, 6 × 4 column-major (one curve per block of 6).
    coefs: [f64; 24],
    /// `eval.fd(t[c(11,26,51,76,91)], r$warpfd)`, 5 × 4 column-major.
    warp: [f64; 20],
    /// `eval.fd(t[c(11,26,51,76,91)], r$regfd)`, 5 × 4 column-major.
    reg: [f64; 20],
}

const R_IDX: [usize; 5] = [10, 25, 50, 75, 90];

const R_CASES: [RCase; 3] = [
    RCase {
        crit: ContinuousRegistrationCriterion::LeastSquares,
        lambda: 0.0,
        coefs: [
            0.0,
            -0.1413730894841189,
            -0.3775538028878155,
            -0.4315119475352712,
            -0.7387823276161214,
            -0.5490581132588078,
            0.0,
            0.004301882596398036,
            -0.1476790806094092,
            -0.04551826686812169,
            -0.3437623658547321,
            -0.1049499855752376,
            0.0,
            0.09475880480807228,
            0.005172754707445408,
            0.3121138985757422,
            0.09420268954760934,
            0.4215775694266682,
            0.0,
            0.1348279033997017,
            0.04797887429107384,
            0.4686863804642218,
            0.3212150034200454,
            0.7540669414340976,
        ],
        warp: [
            0.1377654378389051,
            0.3167872250474534,
            0.5761900794482374,
            0.8014307043861207,
            0.9212278641119824,
            0.1121249460883896,
            0.27152498643188,
            0.5278285891207879,
            0.7713313230103407,
            0.9078868112400863,
            0.08999475728220334,
            0.2254717102399063,
            0.4656834902680907,
            0.7237099553133222,
            0.8841400582547665,
            0.08026440628094164,
            0.2030874750805829,
            0.4295317658307863,
            0.6894226134948126,
            0.8642551447153796,
        ],
        reg: [
            0.1402689740522663,
            0.8919980716840394,
            0.2425629414482214,
            0.6890489696337883,
            0.1117747214128169,
            0.1391221397930693,
            0.893178644421595,
            0.2426956254600265,
            0.6880428420612577,
            0.1122263351169592,
            0.1381164538328844,
            0.8943667947240146,
            0.2427947747563526,
            0.6870288734924855,
            0.1126719907157147,
            0.1375336974356136,
            0.8953323648363852,
            0.2428860358970352,
            0.686261709389733,
            0.11313718992062,
        ],
    },
    RCase {
        crit: ContinuousRegistrationCriterion::MinEigenvalue,
        lambda: 0.0,
        coefs: [
            0.0,
            -0.3859492494185055,
            -0.6585640209217455,
            -0.3386301451836753,
            -1.3387090754035,
            -0.3099926156951125,
            0.0,
            -0.3055063739808616,
            -0.4297998090013314,
            -0.02957281848556458,
            -0.9285422986243372,
            0.06406146249901334,
            0.0,
            -0.288254049292846,
            -0.2647216763939755,
            0.251721149928424,
            -0.4686523413405855,
            0.5091122054865334,
            0.0,
            -0.2864540326344413,
            -0.2253590820615858,
            0.3637066502024304,
            -0.2171329427534635,
            0.7844117246352719,
        ],
        warp: [
            0.1487808300976003,
            0.3178868867640199,
            0.5739748519955022,
            0.8016137928231115,
            0.912237328791596,
            0.1225210987135516,
            0.2723806624242681,
            0.5260375439439606,
            0.7709127995849236,
            0.8979015797213656,
            0.09897974002307262,
            0.2253513474916189,
            0.4644183721051657,
            0.7229313373503656,
            0.8731580960061347,
            0.08869285727816495,
            0.2027501350812861,
            0.427738479985848,
            0.6874894225187679,
            0.8526690849174439,
        ],
        reg: [
            0.1645717257887017,
            0.8980911813012806,
            0.2447259262150576,
            0.6782060469035311,
            0.1466059136441789,
            0.1662905498183805,
            0.8982500642591505,
            0.2444940071375411,
            0.680765256734357,
            0.1449737141103853,
            0.1668425831618481,
            0.8950288757577645,
            0.2442195121889184,
            0.6817610095282245,
            0.1416754288690566,
            0.1678185540376735,
            0.8945283352248051,
            0.2446867496737699,
            0.6853065899004619,
            0.13991057547914,
        ],
    },
    RCase {
        crit: ContinuousRegistrationCriterion::MinEigenvalue,
        lambda: 1e-3,
        coefs: [
            0.0,
            -0.0847277708221473,
            -0.2565217050001986,
            -0.4570957317944274,
            -0.5693406442317382,
            -0.6263754659643266,
            0.0,
            -0.03054324329693026,
            -0.091891751832253,
            -0.1616104510185303,
            -0.2024663446227916,
            -0.2227888987970753,
            0.0,
            0.02406738963299699,
            0.071630130017416,
            0.1668217213147132,
            0.2398197046837599,
            0.276278944963548,
            0.0,
            0.05000016669867278,
            0.1477642447872447,
            0.3379356145014031,
            0.4922051341997905,
            0.5684278749138952,
        ],
        warp: [
            0.133054635968841,
            0.3145812189799999,
            0.5775932889428941,
            0.8029572181012417,
            0.9242390767708863,
            0.1110882019631137,
            0.2721366499371509,
            0.5275293271386586,
            0.769424394231289,
            0.909058620303687,
            0.08878913145289116,
            0.2256655512027032,
            0.4651640731882117,
            0.7220698478039583,
            0.8861093851430801,
            0.07786085883764664,
            0.2014756755230986,
            0.4288720457238994,
            0.6911744286644416,
            0.8700999632363607,
        ],
        reg: [
            0.1311055620042318,
            0.8821158644593696,
            0.2419129912153172,
            0.6857236734379982,
            0.1014634782618153,
            0.1386018121649228,
            0.8925930443433102,
            0.2424215998349743,
            0.699919742125515,
            0.1065685463121408,
            0.1378132313482495,
            0.8906377650940634,
            0.2424617850111357,
            0.6981751658332247,
            0.1055827959049983,
            0.1336937289649676,
            0.8812917909456396,
            0.2424997365561578,
            0.6869332648851725,
            0.09981423261756499,
        ],
    },
];

fn fine_target(setup: &CcSetup, t: &[f64], y0: &[f64]) -> Vec<f64> {
    let ts = NaturalSpline::new(t, y0);
    setup.xfine.iter().map(|&x| ts.eval(x).0).collect()
}

#[test]
fn clamped_basis_matches_r_eval_basis() {
    let knots = clamped_knots(0.0, 1.0, 6, 4);
    assert_eq!(
        knots,
        vec![0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0]
    );
    for (x, row) in [0.0, 0.1, 0.37, 0.5, 0.9, 1.0].iter().zip(R_BASIS.iter()) {
        let b = clamped_bspline_eval(*x, &knots, 4, 0);
        for (v, r) in b.iter().zip(row) {
            assert!((v - r).abs() < 1e-12, "x={x}: {v} vs {r}");
        }
        assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }
}

#[test]
fn bspline_derivative_matches_finite_difference() {
    let knots = clamped_knots(0.0, 2.0, 7, 4);
    for &x in &[0.05, 0.4, 0.93, 1.37, 1.97] {
        for d in 1..3 {
            let an = clamped_bspline_eval(x, &knots, 4, d);
            let h = 1e-5;
            let lo = clamped_bspline_eval(x - h, &knots, 4, d - 1);
            let hi = clamped_bspline_eval(x + h, &knots, 4, d - 1);
            for k in 0..an.len() {
                let fd = (hi[k] - lo[k]) / (2.0 * h);
                assert!(
                    (an[k] - fd).abs() < 1e-5 * (1.0 + fd.abs()),
                    "x={x} d={d} k={k}: {} vs {fd}",
                    an[k]
                );
            }
        }
    }
}

#[test]
fn penalty_matches_r_getbasispenalty() {
    let knots = clamped_knots(0.0, 1.0, 6, 4);
    let k = clamped_bspline_penalty(&knots, 4, 2);
    for p in 0..6 {
        for r in 0..6 {
            let want = R_PENALTY[p][r];
            assert!(
                (k[p * 6 + r] - want).abs() < 1e-9 * (1.0 + want.abs()),
                "K[{p},{r}]"
            );
        }
    }
}

#[test]
fn golden_r_register_fd() {
    let (data, t) = golden_data();
    for case in &R_CASES {
        let cfg = golden_config(case.crit, case.lambda);
        let res = continuous_registration(&data, &t, None, &cfg).unwrap();
        let setup = CcSetup::new(0.0, 1.0, 431, &cfg);
        let y0f = fine_target(&setup, &t, &res.target);
        for i in 0..4 {
            let sp = NaturalSpline::new(&t, &data.row(i));
            let rc = &case.coefs[i * 6..(i + 1) * 6];
            // fdars' criterion evaluated at R's optimum.
            let f_r = crit_eval(&setup, &sp, &y0f, rc, false).f;
            let f = res.criterion[i];
            assert!(res.converged[i]);
            assert!(
                f <= f_r + 1e-12,
                "{:?} curve {i}: fdars {f} > R {f_r}",
                case.crit
            );
            // Same basin: within 1 % of R's (R stops on an absolute conv = 1e-6
            // with an approximate gradient).
            assert!(
                (f_r - f) / f_r < 1e-2,
                "{:?} curve {i}: {f} vs {f_r}",
                case.crit
            );
            // Warps / registered curves vs R's basis-projected warpfd / regfd.
            for (q, &j) in R_IDX.iter().enumerate() {
                let hw = case.warp[i * 5 + q];
                let rw = case.reg[i * 5 + q];
                assert!((res.warps[(i, j)] - hw).abs() < 1e-2, "warp {i},{j}");
                assert!(
                    (res.registered_data[(i, j)] - rw).abs() < 2e-2,
                    "reg {i},{j}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Invariants
// ---------------------------------------------------------------------------

/// Curves `yᵢ = sᵢ · y₀ ∘ γᵢ⁻¹` with `γᵢ` drawn from the W-family itself, on [1, 4].
fn synthetic_warped(scales: &[f64]) -> (FdMatrix, Vec<f64>, Vec<f64>, Vec<Vec<f64>>, FdMatrix) {
    let (a, b, m) = (1.0, 4.0, 121);
    let t: Vec<f64> = (0..m)
        .map(|j| a + (b - a) * j as f64 / (m - 1) as f64)
        .collect();
    let y0f =
        |s: f64| (std::f64::consts::PI * (s - 1.0)).sin() + 0.6 * (s - 1.0) + 0.3 * (2.3 * s).cos();
    let coefs = vec![
        vec![0.0, 0.4, -0.3, 0.2, -0.4, 0.3],
        vec![0.0, -0.5, 0.2, 0.3, 0.1, -0.2],
        vec![0.0, 0.3, 0.5, -0.4, 0.2, 0.0],
    ];
    let cfg = ContinuousRegistrationConfig::default();
    let fine = CcSetup::new(a, b, 6001, &cfg);
    let mut data = FdMatrix::zeros(coefs.len(), m);
    let mut gam = FdMatrix::zeros(coefs.len(), m);
    for (i, c) in coefs.iter().enumerate() {
        let fw = fine_warp(&fine, c, false);
        let g = warp_at(&fine, &fw, &t);
        for j in 0..m {
            gam[(i, j)] = g[j];
            // γ⁻¹(t_j) by linear interpolation of the fine graph (h, x).
            let ginv = crate::helpers::linear_interp(&fw.h, &fine.xfine, t[j]);
            data[(i, j)] = scales[i] * y0f(ginv);
        }
    }
    let y0: Vec<f64> = t.iter().map(|&s| y0f(s)).collect();
    (data, t, y0, coefs, gam)
}

fn assert_valid_warps(res: &ContinuousRegistrationResult, t: &[f64]) {
    let m = t.len();
    for i in 0..res.warps.nrows() {
        assert_eq!(res.warps[(i, 0)], t[0]);
        assert_eq!(res.warps[(i, m - 1)], t[m - 1]);
        for j in 1..m {
            assert!(
                res.warps[(i, j)] > res.warps[(i, j - 1)],
                "warp {i} not increasing at {j}"
            );
        }
        assert_eq!(res.w_coefficients[(i, 0)], 0.0);
    }
}

#[test]
fn recovers_known_warps_both_criteria() {
    let (data, t, y0, coefs, gam) = synthetic_warped(&[1.0, 1.0, 1.0]);
    for crit in [
        ContinuousRegistrationCriterion::LeastSquares,
        ContinuousRegistrationCriterion::MinEigenvalue,
    ] {
        let mut cfg = ContinuousRegistrationConfig::default();
        cfg.criterion = crit;
        let res = continuous_registration(&data, &t, Some(&y0), &cfg).unwrap();
        assert_valid_warps(&res, &t);
        for i in 0..3 {
            assert!(res.converged[i]);
            assert!(
                res.criterion[i] < 1e-8,
                "{crit:?} curve {i}: {}",
                res.criterion[i]
            );
            for j in 0..t.len() {
                assert!(
                    (res.warps[(i, j)] - gam[(i, j)]).abs() < 2e-4,
                    "{crit:?} warp {i},{j}"
                );
                assert!(
                    (res.registered_data[(i, j)] - y0[j]).abs() < 1e-3,
                    "{crit:?} reg {i},{j}"
                );
            }
            for k in 1..6 {
                assert!(
                    (res.w_coefficients[(i, k)] - coefs[i][k]).abs() < 2e-2,
                    "{crit:?} c {i},{k}"
                );
            }
        }
    }
}

#[test]
fn min_eigenvalue_is_amplitude_invariant() {
    let scales = [0.5, 1.7, 2.5];
    let (data, t, y0, _, gam) = synthetic_warped(&scales);
    let mut cfg = ContinuousRegistrationConfig::default();
    cfg.criterion = ContinuousRegistrationCriterion::MinEigenvalue;
    let res = continuous_registration(&data, &t, Some(&y0), &cfg).unwrap();
    for i in 0..3 {
        assert!(res.criterion[i] < 1e-8 * scales[i] * scales[i]);
        for j in 0..t.len() {
            assert!((res.warps[(i, j)] - gam[(i, j)]).abs() < 2e-4);
            assert!((res.registered_data[(i, j)] - scales[i] * y0[j]).abs() < 1e-3 * scales[i]);
        }
    }
    // Least squares cannot absorb the amplitude difference.
    cfg.criterion = ContinuousRegistrationCriterion::LeastSquares;
    let ls = continuous_registration(&data, &t, Some(&y0), &cfg).unwrap();
    assert!(ls.criterion.iter().all(|&f| f > 1e-2));
}

#[test]
fn aligned_curves_give_identity_warps() {
    // Non-uniform grid on [-1, 2].
    let t: Vec<f64> = (0..80)
        .map(|j| -1.0 + 3.0 * (j as f64 / 79.0).powf(1.3))
        .collect();
    let row: Vec<f64> = t.iter().map(|&s| (2.0 * s).sin() + 0.2 * s * s).collect();
    let mut data = FdMatrix::zeros(3, t.len());
    for i in 0..3 {
        for j in 0..t.len() {
            data[(i, j)] = row[j];
        }
    }
    for crit in [
        ContinuousRegistrationCriterion::LeastSquares,
        ContinuousRegistrationCriterion::MinEigenvalue,
    ] {
        let mut cfg = ContinuousRegistrationConfig::default();
        cfg.criterion = crit;
        cfg.lambda = 1e-2;
        let res = continuous_registration(&data, &t, None, &cfg).unwrap();
        assert_valid_warps(&res, &t);
        for i in 0..3 {
            // crit 2 = (aa + cc) − √(…) carries an O(ε·aa) cancellation floor.
            assert!(res.criterion[i] < 1e-14, "{crit:?}: {}", res.criterion[i]);
            for j in 0..t.len() {
                assert!((res.warps[(i, j)] - t[j]).abs() < 1e-10);
                assert!((res.registered_data[(i, j)] - row[j]).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn gradient_matches_finite_differences() {
    let (data, t) = golden_data();
    let c = [0.0, 0.3, -0.2, 0.5, -0.1, 0.25];
    for crit in [
        ContinuousRegistrationCriterion::LeastSquares,
        ContinuousRegistrationCriterion::MinEigenvalue,
    ] {
        let cfg = golden_config(crit, 1e-3);
        let setup = CcSetup::new(0.0, 1.0, 301, &cfg);
        let y0f = fine_target(&setup, &t, &crate::fdata::mean_1d(&data));
        let sp = NaturalSpline::new(&t, &data.row(0));
        let g = crit_gradient(&setup, &sp, &y0f, &c);
        for k in 0..5 {
            let h = 1e-6;
            let (mut cp, mut cm) = (c, c);
            cp[k + 1] += h;
            cm[k + 1] -= h;
            let fd = (crit_eval(&setup, &sp, &y0f, &cp, false).f
                - crit_eval(&setup, &sp, &y0f, &cm, false).f)
                / (2.0 * h);
            assert!(
                (g[k] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
                "{crit:?} k={k}: {} vs {fd}",
                g[k]
            );
        }
    }
}

#[test]
fn criterion_history_is_non_increasing() {
    let (data, t) = golden_data();
    for case in &R_CASES {
        let cfg = golden_config(case.crit, case.lambda);
        let setup = CcSetup::new(0.0, 1.0, 431, &cfg);
        let y0f = fine_target(&setup, &t, &crate::fdata::mean_1d(&data));
        for i in 0..4 {
            let sp = NaturalSpline::new(&t, &data.row(i));
            let fit = register_curve(&setup, &sp, &y0f, &[0.0; 6], 50, 1e-8);
            assert!(fit.history.len() >= 2);
            assert!(fit.history.windows(2).all(|w| w[1] < w[0]));
            assert!(fit.history.iter().all(|&f| f >= 0.0));
        }
    }
}

#[test]
fn procrustes_template_iteration_reaches_fixed_point() {
    let (data, t) = golden_data();
    let mut cfg = golden_config(ContinuousRegistrationCriterion::MinEigenvalue, 1e-3);
    cfg.max_template_iter = 20;
    cfg.template_tol = 1e-3;
    let res = continuous_registration(&data, &t, None, &cfg).unwrap();
    assert!(
        res.n_template_iter > 1 && res.n_template_iter < 20,
        "{}",
        res.n_template_iter
    );
    let mean_reg = crate::fdata::mean_1d(&res.registered_data);
    assert!(rel_change(&mean_reg, &res.target) <= 1e-3);
    assert_valid_warps(&res, &t);
    // A single pass registers against the unregistered mean.
    cfg.max_template_iter = 1;
    let one = continuous_registration(&data, &t, None, &cfg).unwrap();
    assert_eq!(one.n_template_iter, 1);
    assert_eq!(one.target, crate::fdata::mean_1d(&data));
}

#[test]
fn warm_start_and_zero_iterations() {
    let (data, t) = golden_data();
    let mut cfg = golden_config(ContinuousRegistrationCriterion::LeastSquares, 0.0);
    let res = continuous_registration(&data, &t, None, &cfg).unwrap();
    cfg.initial_coefficients = Some(res.w_coefficients.clone());
    let warm = continuous_registration(&data, &t, None, &cfg).unwrap();
    for i in 0..4 {
        assert!(warm.n_iter[i] <= 2);
        assert!(warm.criterion[i] <= res.criterion[i]);
    }
    cfg.initial_coefficients = None;
    cfg.max_iter = 0;
    let id = continuous_registration(&data, &t, None, &cfg).unwrap();
    for i in 0..4 {
        assert!(!id.converged[i]);
        for j in 0..t.len() {
            assert!((id.warps[(i, j)] - t[j]).abs() < 1e-12);
            assert!((id.registered_data[(i, j)] - data[(i, j)]).abs() < 1e-12);
        }
    }
}

#[test]
fn constant_curve_does_not_break() {
    let t = grid(30);
    let mut data = FdMatrix::zeros(2, 30);
    for j in 0..30 {
        data[(0, j)] = 2.0;
        data[(1, j)] = t[j];
    }
    let target: Vec<f64> = t.iter().map(|s| s * s).collect();
    for crit in [
        ContinuousRegistrationCriterion::LeastSquares,
        ContinuousRegistrationCriterion::MinEigenvalue,
    ] {
        let mut cfg = ContinuousRegistrationConfig::default();
        cfg.criterion = crit;
        let res = continuous_registration(&data, &t, Some(&target), &cfg).unwrap();
        assert_valid_warps(&res, &t);
        assert!(res.criterion.iter().all(|f| f.is_finite()));
        // h with y(h) = h ≈ t² exists in the family up to basis error.
        assert!(res.criterion[1] < 1e-4, "{crit:?}: {}", res.criterion[1]);
    }
}

#[test]
fn rejects_invalid_input() {
    let (data, t) = golden_data();
    let base = ContinuousRegistrationConfig::default();
    let run = |cfg: &ContinuousRegistrationConfig| continuous_registration(&data, &t, None, cfg);
    let mut c = base.clone();
    c.nbasis = 3;
    assert!(run(&c).is_err());
    c = base.clone();
    c.norder = 1;
    c.nbasis = 4;
    assert!(run(&c).is_err());
    c = base.clone();
    c.lfd_order = 4;
    assert!(run(&c).is_err());
    c = base.clone();
    c.lambda = -1.0;
    assert!(run(&c).is_err());
    c = base.clone();
    c.tol = f64::NAN;
    assert!(run(&c).is_err());
    c = base.clone();
    c.n_fine = Some(2);
    assert!(run(&c).is_err());
    c = base.clone();
    c.max_template_iter = 0;
    assert!(run(&c).is_err());
    c = base.clone();
    c.initial_coefficients = Some(FdMatrix::zeros(4, 5));
    assert!(matches!(run(&c), Err(FdarError::InvalidDimension { .. })));
    assert!(continuous_registration(&data, &t[..100], None, &base).is_err());
    assert!(continuous_registration(&data, &t, Some(&t[..50]), &base).is_err());
    let mut tb = t.clone();
    tb[5] = tb[4];
    assert!(continuous_registration(&data, &tb, None, &base).is_err());
    let mut dn = data.clone();
    dn[(1, 3)] = f64::NAN;
    assert!(continuous_registration(&dn, &t, None, &base).is_err());
    let small = FdMatrix::zeros(2, 3);
    assert!(continuous_registration(&small, &t[..3], None, &base).is_err());
}

// ---------------------------------------------------------------------------
// Periodic registration and the R-default W basis (GH #72).
// Golden values: R fda 6.3.0, script golden_register_fd_periodic.R in
// .planning/quick/261126-m5e-aln-cav-missing-options-continuous-regis/:
// yfd = smooth.basis(t, Y, create.fourier.basis(c(0,1), 21))$fd,
// register.fd(mean.fd(yfd), yfd, NULL, conv = 1e-8, iterlim = 50, dbglev = 0,
//             periodic = TRUE, crit), nfine = 211.
// ---------------------------------------------------------------------------

fn periodic_g(s: f64) -> f64 {
    use std::f64::consts::PI;
    (2.0 * PI * s).sin() + 0.5 * (4.0 * PI * s).cos() + 0.3 * (6.0 * PI * s).sin()
}

/// `yᵢ(t) = g(t + dᵢ + 0.3·bᵢ·sin(2πt)/(2π))` — periodic, shifted and warped.
fn periodic_data() -> (FdMatrix, Vec<f64>) {
    use std::f64::consts::PI;
    let t = grid(101);
    let d = [-0.06, -0.02, 0.03, 0.07];
    let b = [0.25, -0.2, 0.1, -0.15];
    let mut data = FdMatrix::zeros(4, 101);
    for i in 0..4 {
        for j in 0..101 {
            let s = t[j] + d[i] + b[i] * (2.0 * PI * t[j]).sin() / (2.0 * PI) * 0.3;
            data[(i, j)] = periodic_g(s);
        }
    }
    (data, t)
}

struct RPeriodicCase {
    crit: ContinuousRegistrationCriterion,
    /// `r$Wfd$coefs[2, ]`.
    c2: [f64; 4],
    /// `r$shift`.
    shift: [f64; 4],
    /// `eval.fd(t[c(11,26,51,76,91)], r$regfd)`, 5 × 4 column-major.
    reg: [f64; 20],
}

const R_PERIODIC: [RPeriodicCase; 2] = [
    RPeriodicCase {
        crit: ContinuousRegistrationCriterion::LeastSquares,
        c2: [
            -0.10066631459907341,
            -0.01080554313155276,
            -0.00944770489492076,
            -0.08367422871830447,
        ],
        shift: [
            0.0587547418660501,
            0.0230967748649921,
            -0.0254602123421473,
            -0.0724340570922603,
        ],
        reg: [
            0.978041064241725,
            0.227155149919778,
            0.424495119752945,
            -1.199575316097089,
            -0.713475968113454,
            1.034759299763169,
            0.202300680235376,
            0.427974738097774,
            -1.198716157072242,
            -0.640815735302548,
            1.004580018590876,
            0.207881246670932,
            0.421272423703043,
            -1.199824735896626,
            -0.703330511020780,
            1.027127344788617,
            0.200630790183120,
            0.435107333015734,
            -1.198808564104242,
            -0.645223373454325,
        ],
    },
    RPeriodicCase {
        crit: ContinuousRegistrationCriterion::MinEigenvalue,
        c2: [
            -0.1026932574190101,
            -0.0143151901606594,
            -0.0161541461025375,
            -0.0891580302998799,
        ],
        shift: [
            0.0584047915807161,
            0.0229388989635364,
            -0.0258916965185344,
            -0.0726147717138622,
        ],
        reg: [
            0.979274556749657,
            0.226710197235556,
            0.425610840501757,
            -1.199567741272651,
            -0.716121421474605,
            1.034715939054851,
            0.202217358396455,
            0.424268713295645,
            -1.198675982192599,
            -0.640797501206357,
            1.005014995761218,
            0.208235159682759,
            0.416341135326355,
            -1.199804577308455,
            -0.704560987186977,
            1.026913026792511,
            0.200638556074814,
            0.428650477923216,
            -1.198744002666004,
            -0.644646612010571,
        ],
    },
];

fn periodic_config(crit: ContinuousRegistrationCriterion) -> ContinuousRegistrationConfig {
    let mut cfg = ContinuousRegistrationConfig::r_defaults();
    cfg.criterion = crit;
    cfg.periodic = true;
    cfg.max_iter = 50;
    cfg.n_fine = Some(211);
    cfg
}

#[test]
fn r_defaults_preset_matches_register_fd_defaults() {
    let r = ContinuousRegistrationConfig::r_defaults();
    let d = ContinuousRegistrationConfig::default();
    assert_eq!((r.nbasis, r.norder, r.lfd_order, r.max_iter), (2, 2, 0, 20));
    assert_eq!(r.lambda, 0.0);
    assert_eq!(r.criterion, ContinuousRegistrationCriterion::MinEigenvalue);
    assert!(!r.periodic);
    // Everything else is the (unchanged) fdars default.
    let mut r2 = r.clone();
    r2.nbasis = d.nbasis;
    r2.norder = d.norder;
    r2.lfd_order = d.lfd_order;
    r2.max_iter = d.max_iter;
    assert_eq!(r2, d);
    assert_eq!((d.nbasis, d.norder, d.lfd_order, d.max_iter), (6, 4, 2, 50));
    assert!(!d.periodic);
    assert!(r.validate().is_ok());
}

/// R `register.fd(mean.fd(yfd), yfd, NULL, conv = 1e-8, iterlim = 50, crit = 2)`
/// on the #47 golden data (same script): `r$Wfd$coefs[2, ]`.
const R_DEFAULT_C2: [f64; 4] = [
    -0.630366069588282,
    -0.223224215640469,
    0.276333466582626,
    0.562130195395612,
];

#[test]
fn golden_r_register_fd_default_w_basis() {
    let (data, t) = golden_data();
    let mut cfg = ContinuousRegistrationConfig::r_defaults();
    cfg.max_iter = 50;
    cfg.n_fine = Some(431);
    let res = continuous_registration(&data, &t, None, &cfg).unwrap();
    assert_eq!(res.w_coefficients.ncols(), 2);
    assert_eq!(res.w_knots, vec![0.0, 0.0, 1.0, 1.0]);
    assert!(res.shift.iter().all(|&s| s == 0.0));
    let setup = CcSetup::new(0.0, 1.0, 431, &cfg);
    let y0f = fine_target(&setup, &t, &res.target);
    for i in 0..4 {
        assert!(res.converged[i]);
        let sp = NaturalSpline::new(&t, &data.row(i));
        let f_r = crit_eval(&setup, &sp, &y0f, &[0.0, R_DEFAULT_C2[i]], false).f;
        let f = res.criterion[i];
        assert!(f <= f_r + 1e-12, "curve {i}: fdars {f} > R {f_r}");
        assert!((f_r - f) / f_r < 1e-2, "curve {i}: {f} vs {f_r}");
        let c2 = res.w_coefficients[(i, 1)];
        assert!((c2 - R_DEFAULT_C2[i]).abs() < 2e-2, "curve {i}: c2 {c2}");
    }
}

#[test]
fn golden_r_register_fd_periodic() {
    let (data, t) = periodic_data();
    for case in &R_PERIODIC {
        let cfg = periodic_config(case.crit);
        let res = continuous_registration(&data, &t, None, &cfg).unwrap();
        let setup = CcSetup::new(0.0, 1.0, 211, &cfg);
        let y0f = fine_target(&setup, &t, &res.target);
        for i in 0..4 {
            assert!(res.converged[i], "{:?} curve {i}", case.crit);
            let sp = curve_spline(&t, &data.row(i), true);
            let f_r = crit_eval(&setup, &sp, &y0f, &[case.shift[i], case.c2[i]], false).f;
            let f = res.criterion[i];
            assert!(
                f <= f_r + 1e-12,
                "{:?} curve {i}: fdars {f} > R {f_r}",
                case.crit
            );
            assert!(
                (f_r - f) / f_r < 1e-2,
                "{:?} curve {i}: {f} vs {f_r}",
                case.crit
            );
            let (s, c2) = (res.shift[i], res.w_coefficients[(i, 1)]);
            assert!(
                (s - case.shift[i]).abs() < 2e-3,
                "{:?} curve {i}: shift {s} vs {}",
                case.crit,
                case.shift[i]
            );
            assert!(
                (c2 - case.c2[i]).abs() < 1e-2,
                "{:?} curve {i}: c2 {c2}",
                case.crit
            );
            assert_eq!(res.w_coefficients[(i, 0)], 0.0);
            for (q, &j) in R_IDX.iter().enumerate() {
                let rw = case.reg[i * 5 + q];
                assert!(
                    (res.registered_data[(i, j)] - rw).abs() < 1e-2,
                    "{:?} reg {i},{j}: {} vs {rw}",
                    case.crit,
                    res.registered_data[(i, j)]
                );
                // Returned warp is h + δ.
                let h = res.warps[(i, j)] - res.shift[i];
                assert!(h > 0.0 && h < 1.0);
            }
            assert!((res.warps[(i, 0)] - res.shift[i]).abs() < 1e-15);
            assert!((res.warps[(i, 100)] - 1.0 - res.shift[i]).abs() < 1e-15);
        }
    }
}

#[test]
fn periodic_recovers_pure_shifts_including_wraparound() {
    // yᵢ(t) = g(t + dᵢ) against the explicit target g: the exact solution is
    // δᵢ = −dᵢ (mod 1), W = 0. Shifts up to ±0.3 wrap far past the endpoints.
    let t = grid(121);
    let d = [-0.3, -0.11, 0.07, 0.24];
    let mut data = FdMatrix::zeros(4, 121);
    for i in 0..4 {
        for j in 0..121 {
            data[(i, j)] = periodic_g(t[j] + d[i]);
        }
    }
    let target: Vec<f64> = t.iter().map(|&s| periodic_g(s)).collect();
    for crit in [
        ContinuousRegistrationCriterion::LeastSquares,
        ContinuousRegistrationCriterion::MinEigenvalue,
    ] {
        let mut cfg = periodic_config(crit);
        cfg.n_fine = None;
        // Start inside the basin (the criterion is multi-modal in δ for a
        // three-harmonic signal); R likewise relies on the initial value.
        let mut init = FdMatrix::zeros(4, 2);
        for i in 0..4 {
            init[(i, 0)] = -d[i] + 0.04;
        }
        cfg.initial_coefficients = Some(init);
        let res = continuous_registration(&data, &t, Some(&target), &cfg).unwrap();
        for i in 0..4 {
            assert!(
                (res.shift[i] + d[i]).abs() < 1e-4,
                "{crit:?} {i}: {}",
                res.shift[i]
            );
            assert!(res.w_coefficients[(i, 1)].abs() < 1e-3);
            for j in 0..121 {
                assert!((res.registered_data[(i, j)] - target[j]).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn periodic_spline_is_periodic_and_matches_samples() {
    let t = grid(81);
    let y: Vec<f64> = t.iter().map(|&s| periodic_g(s)).collect();
    let sp = curve_spline(&t, &y, true);
    for (j, &tj) in t.iter().enumerate() {
        assert!((sp.eval(tj).0 - y[j]).abs() < 1e-12);
    }
    // Value and slope agree across the wrap point.
    let (v0, d0) = sp.eval(0.0);
    let (v1, d1) = sp.eval(1.0);
    assert!((v0 - v1).abs() < 1e-12);
    assert!((d0 - d1).abs() < 1e-6 * (1.0 + d0.abs()), "{d0} vs {d1}");
    // Spline derivative ≈ analytic derivative of g (interpolation accuracy).
    let ga = |s: f64| {
        use std::f64::consts::PI;
        2.0 * PI * (2.0 * PI * s).cos() - 2.0 * PI * (4.0 * PI * s).sin()
            + 1.8 * PI * (6.0 * PI * s).cos()
    };
    assert!((d0 - ga(0.0)).abs() < 1e-2 * ga(0.0).abs());
}

#[test]
fn periodic_gradient_matches_finite_differences() {
    let (data, t) = periodic_data();
    for crit in [
        ContinuousRegistrationCriterion::LeastSquares,
        ContinuousRegistrationCriterion::MinEigenvalue,
    ] {
        let mut cfg = golden_config(crit, 1e-3);
        cfg.periodic = true;
        let setup = CcSetup::new(0.0, 1.0, 301, &cfg);
        let y0f = fine_target(&setup, &t, &crate::fdata::mean_1d(&data));
        let sp = curve_spline(&t, &data.row(1), true);
        // Includes a shift that wraps past b.
        let c = [0.93, 0.3, -0.2, 0.5, -0.1, 0.25];
        let g = crit_gradient(&setup, &sp, &y0f, &c);
        assert_eq!(g.len(), 6);
        for k in 0..6 {
            let h = 1e-6;
            let (mut cp, mut cm) = (c, c);
            cp[k] += h;
            cm[k] -= h;
            let fd = (crit_eval(&setup, &sp, &y0f, &cp, false).f
                - crit_eval(&setup, &sp, &y0f, &cm, false).f)
                / (2.0 * h);
            assert!(
                (g[k] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
                "{crit:?} k={k}: {} vs {fd}",
                g[k]
            );
        }
        // Shifting by a full period leaves the criterion unchanged.
        let mut c1 = c;
        c1[0] -= 1.0;
        let f0 = crit_eval(&setup, &sp, &y0f, &c, false).f;
        let f1 = crit_eval(&setup, &sp, &y0f, &c1, false).f;
        assert!((f0 - f1).abs() < 1e-10 * f0.abs().max(1e-12));
    }
}

#[test]
fn non_periodic_default_ignores_initial_shift_column() {
    // Bit-identity guard: with periodic = false, column 0 of the initial
    // coefficients stays ignored and `shift` is all zeros.
    let (data, t) = golden_data();
    let cfg = golden_config(ContinuousRegistrationCriterion::MinEigenvalue, 0.0);
    let base = continuous_registration(&data, &t, None, &cfg).unwrap();
    let mut cfg2 = cfg.clone();
    let mut init = FdMatrix::zeros(4, 6);
    for i in 0..4 {
        init[(i, 0)] = 0.37;
    }
    cfg2.initial_coefficients = Some(init);
    let res = continuous_registration(&data, &t, None, &cfg2).unwrap();
    assert_eq!(res, base);
    assert!(base.shift.iter().all(|&s| s == 0.0));
}

/// Default (non-periodic) results are unchanged from the pre-#72 code.
///
/// Originally an FNV-1a hash of the output bits pinned on commit 9c856e31; that hash
/// depends on the platform libm's last-ulp `exp` results (glibc selects a CPU-specific
/// kernel) and failed on CI runners. The reference summaries below were recorded on
/// 925ef514, which reproduced the 9c856e31 hash bit-for-bit on the recording machine,
/// and are compared at rtol 1e-10 (see [`ValuePin`]).
#[test]
fn default_results_bit_identical_to_pre_periodic_pin() {
    let m = 41;
    let t = grid(m);
    let n = 30;
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let a = 0.8 + 0.4 * ((i * 13 % 7) as f64 / 6.0);
        let sh = 0.1 * ((i * 7 % 5) as f64 / 4.0 - 0.5);
        for j in 0..m {
            data[(i, j)] = a * (-(t[j] - 0.5 - sh).powi(2) / 0.02).exp()
                + 0.3 * a * (-(t[j] - 0.2).powi(2) / 0.01).exp();
        }
    }
    let mut pin = ValuePin::new();
    for crit in [
        ContinuousRegistrationCriterion::LeastSquares,
        ContinuousRegistrationCriterion::MinEigenvalue,
    ] {
        let mut cfg = ContinuousRegistrationConfig::default();
        cfg.criterion = crit;
        cfg.lambda = 1e-3;
        cfg.max_template_iter = 3;
        let r = continuous_registration(&data, &t, None, &cfg).unwrap();
        pin.push(r.registered_data.as_slice());
        pin.push(r.warps.as_slice());
        pin.push(r.w_coefficients.as_slice());
        pin.push(&r.criterion);
        pin.push(&r.target);
    }
    pin.assert_close(PRE_PERIODIC_PIN, 1e-10, "continuous registration default");
}

const PRE_PERIODIC_PIN: &[f64] = &[
    364.24588120546883,
    364.24588120546883,
    235.91560643621625,
    164.12917424080433,
    614.0408937395684,
    614.0408937395684,
    415.0531392898845,
    412.2600522349258,
    1.6346921455864498,
    24.066030787971076,
    6.830318840845282,
    0.8801398047802601,
    0.12429088154900178,
    0.12429088154900178,
    0.0007771572408811497,
    0.06394401397632757,
    12.144669590402154,
    12.144669590402154,
    7.708831245023279,
    5.617447497900823,
    366.0599895261668,
    366.0599895261668,
    238.03940225363394,
    165.00719770519848,
    613.9668734457233,
    613.9668734457233,
    414.6834031309565,
    412.0838886450891,
    0.8025690662530468,
    22.77631730552719,
    6.405363193479252,
    0.4852514673330762,
    0.021584653302037918,
    0.021584653302037918,
    2.5541075635703238e-5,
    0.01090807810309888,
    12.208365754573414,
    12.208365754573414,
    7.760969478359123,
    5.648777403140646,
];
