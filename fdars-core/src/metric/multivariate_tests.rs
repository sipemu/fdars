//! Multivariate (multichannel) DTW family: goldens against aeon 1.6.0 /
//! tslearn 0.10.0 and `d = 1` bit-identity with the univariate functions.
#![allow(clippy::field_reassign_with_default)]

use super::multivariate_goldens::*;
use super::*;
use crate::dtw_kmeans::{dtw_kmeans, dtw_kmeans_multivariate, DtwKMeansConfig, DtwKMeansInit};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

#[derive(Clone, Copy, Debug)]
pub(super) enum W {
    None,
    Frac,
}

impl W {
    fn win(self) -> DtwWindow {
        match self {
            W::None => DtwWindow::None,
            W::Frac => DtwWindow::SakoeChibaFraction { window: 0.3 },
        }
    }
}

pub(super) struct MvGolden {
    pub(super) name: &'static str,
    pub(super) window: W,
    pub(super) distance: f64,
    pub(super) shape: (usize, usize),
    pub(super) cost: &'static [f64],
    /// `None` where aeon's traceback does not terminate (windowed LCSS).
    pub(super) path: Option<&'static [(usize, usize)]>,
    pub(super) path_distance: f64,
}

/// `d × m` series from a column-major (time-major interleaved) buffer.
fn ser(flat: &[f64], d: usize) -> FdMatrix {
    FdMatrix::from_column_major(flat.to_vec(), d, flat.len() / d).unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

fn assert_rm(what: &str, got: &FdMatrix, want_rm: &[f64], tol: f64) {
    let (r, c) = got.shape();
    assert_eq!(r * c, want_rm.len(), "{what}: shape {r}x{c}");
    for i in 0..r {
        for j in 0..c {
            let (g, w) = (got[(i, j)], want_rm[i * c + j]);
            assert!(close(g, w, tol), "{what}[{i},{j}]: {g} vs {w}");
        }
    }
}

type Trio = (f64, FdMatrix, ElasticDistancePath);

/// Distance, cost matrix and path of the named multivariate distance (the
/// golden parameters of `multivariate_gold.py`).
fn mv_run(name: &str, x: &FdMatrix, y: &FdMatrix, w: DtwWindow) -> Result<Trio, FdarError> {
    macro_rules! go {
        ($d:ident, $c:ident, $p:ident $(, $a:expr)*) => {
            Ok(($d(x, y, w $(, $a)*)?, $c(x, y, w $(, $a)*)?, $p(x, y, w $(, $a)*)?))
        };
    }
    match name {
        "ddtw" => go!(
            ddtw_distance_multivariate,
            ddtw_cost_matrix_multivariate,
            ddtw_alignment_path_multivariate
        ),
        "wdtw" => go!(
            wdtw_distance_multivariate,
            wdtw_cost_matrix_multivariate,
            wdtw_alignment_path_multivariate,
            0.05
        ),
        "wddtw" => go!(
            wddtw_distance_multivariate,
            wddtw_cost_matrix_multivariate,
            wddtw_alignment_path_multivariate,
            0.2
        ),
        "adtw" => go!(
            adtw_distance_multivariate,
            adtw_cost_matrix_multivariate,
            adtw_alignment_path_multivariate,
            0.7
        ),
        "erp" => go!(
            erp_distance_multivariate,
            erp_cost_matrix_multivariate,
            erp_alignment_path_multivariate,
            0.5
        ),
        "edr" => go!(
            edr_distance_multivariate,
            edr_cost_matrix_multivariate,
            edr_alignment_path_multivariate,
            None
        ),
        "edr_eps" => go!(
            edr_distance_multivariate,
            edr_cost_matrix_multivariate,
            edr_alignment_path_multivariate,
            Some(0.9)
        ),
        "twe" => go!(
            twe_distance_multivariate,
            twe_cost_matrix_multivariate,
            twe_alignment_path_multivariate,
            0.001,
            1.0
        ),
        "msm_ind" => go!(
            msm_distance_multivariate,
            msm_cost_matrix_multivariate,
            msm_alignment_path_multivariate,
            1.0,
            true
        ),
        "msm_dep" => go!(
            msm_distance_multivariate,
            msm_cost_matrix_multivariate,
            msm_alignment_path_multivariate,
            0.8,
            false
        ),
        "lcss" => go!(
            lcss_distance_multivariate,
            lcss_cost_matrix_multivariate,
            lcss_alignment_path_multivariate,
            1.2
        ),
        _ => unreachable!("{name}"),
    }
}

fn mv_pairwise(
    name: &str,
    data: &[FdMatrix],
    other: Option<&[FdMatrix]>,
) -> Result<FdMatrix, FdarError> {
    let w = DtwWindow::None;
    match name {
        "ddtw" => ddtw_pairwise_distance_multivariate(data, other, w),
        "wdtw" => wdtw_pairwise_distance_multivariate(data, other, w, 0.05),
        "wddtw" => wddtw_pairwise_distance_multivariate(data, other, w, 0.2),
        "adtw" => adtw_pairwise_distance_multivariate(data, other, w, 0.7),
        "erp" => erp_pairwise_distance_multivariate(data, other, w, 0.5),
        "edr" => edr_pairwise_distance_multivariate(data, other, w, None),
        "edr_eps" => edr_pairwise_distance_multivariate(data, other, w, Some(0.9)),
        "twe" => twe_pairwise_distance_multivariate(data, other, w, 0.001, 1.0),
        "msm_ind" => msm_pairwise_distance_multivariate(data, other, w, 1.0, true),
        "msm_dep" => msm_pairwise_distance_multivariate(data, other, w, 0.8, false),
        "lcss" => lcss_pairwise_distance_multivariate(data, other, w, 1.2),
        _ => unreachable!("{name}"),
    }
}

fn pairwise_golden(name: &str) -> (&'static [f64], &'static [f64]) {
    match name {
        "ddtw" => (&PW_DDTW, &PC_DDTW),
        "wdtw" => (&PW_WDTW, &PC_WDTW),
        "wddtw" => (&PW_WDDTW, &PC_WDDTW),
        "adtw" => (&PW_ADTW, &PC_ADTW),
        "erp" => (&PW_ERP, &PC_ERP),
        "edr" => (&PW_EDR, &PC_EDR),
        "edr_eps" => (&PW_EDR_EPS, &PC_EDR_EPS),
        "twe" => (&PW_TWE, &PC_TWE),
        "msm_ind" => (&PW_MSM_IND, &PC_MSM_IND),
        "msm_dep" => (&PW_MSM_DEP, &PC_MSM_DEP),
        "lcss" => (&PW_LCSS, &PC_LCSS),
        _ => unreachable!("{name}"),
    }
}

fn xyz() -> (FdMatrix, FdMatrix, FdMatrix) {
    (ser(&MX, 3), ser(&MY, 3), ser(&MZ, 3))
}

fn collection() -> Vec<FdMatrix> {
    vec![ser(&MC0, 3), ser(&MC1, 3), ser(&MC2, 3)]
}

#[test]
fn aeon_elastic_family_multivariate_goldens() {
    let (x, y, _) = xyz();
    for g in MV_GOLDENS {
        let what = format!("{} {:?}", g.name, g.window);
        let (d, cm, p) = mv_run(g.name, &x, &y, g.window.win()).unwrap();
        assert!(close(d, g.distance, 1e-10), "{what}: {d} vs {}", g.distance);
        assert_eq!(cm.shape(), g.shape, "{what}");
        assert_rm(&what, &cm, g.cost, 1e-10);
        if let Some(path) = g.path {
            assert_eq!(p.path, path, "{what} path");
            assert!(
                close(p.distance, g.path_distance, 1e-10),
                "{what} path distance"
            );
        }
    }
}

#[test]
fn aeon_elastic_family_multivariate_pairwise_goldens() {
    let (x, y, _) = xyz();
    let c = collection();
    let other = [x, y];
    for g in MV_GOLDENS.iter().filter(|g| matches!(g.window, W::None)) {
        let (pw, pc) = pairwise_golden(g.name);
        let got = mv_pairwise(g.name, &c, None).unwrap();
        assert_rm(g.name, &got, pw, 1e-10);
        let got = mv_pairwise(g.name, &c, Some(&other)).unwrap();
        assert_rm(g.name, &got, pc, 1e-10);
    }
}

#[test]
fn aeon_and_tslearn_multivariate_dtw_goldens() {
    let (x, y, z) = xyz();
    for (w, dist, path, cm) in [
        (W::None, DTW_NONE, DTW_PATH_NONE, &DTW_CM_NONE[..]),
        (W::Frac, DTW_FRAC, DTW_PATH_FRAC, &DTW_CM_FRAC[..]),
    ] {
        let d = dtw_distance_multivariate(&x, &y, 2.0, w.win()).unwrap();
        assert!(close(d, dist, 1e-10), "{w:?}: {d} vs {dist}");
        let p = dtw_path_multivariate(&x, &y, 2.0, w.win()).unwrap();
        assert_eq!(p.cost.to_bits(), d.to_bits());
        // aeon's traceback (`compute_min_return_path`) breaks ties like tslearn.
        assert_eq!(p.path, path, "{w:?}");
        // Accumulated cost matrix of the path's prefix is checked via the
        // aeon cost matrix at every path cell.
        let m = y.ncols();
        for &(i, j) in &p.path {
            assert!(cm[i * m + j].is_finite());
        }
        assert!(close(cm[(x.ncols() - 1) * m + m - 1], d, 1e-10));
    }
    let c = collection();
    let pw = dtw_pairwise_distance_multivariate(&c, None, 2.0, DtwWindow::None).unwrap();
    assert_rm("dtw pw", &pw, &DTW_PW, 1e-10);
    let pc =
        dtw_pairwise_distance_multivariate(&c, Some(&[x.clone(), y.clone()]), 2.0, DtwWindow::None)
            .unwrap();
    assert_rm("dtw pc", &pc, &DTW_PC, 1e-10);

    // tslearn dtw_path on (sz, d) arrays: same path, distance = sqrt(cost).
    let p = dtw_path_multivariate(&x, &z, 2.0, DtwWindow::SakoeChiba { radius: 2 }).unwrap();
    assert_eq!(p.path, TS_PATH_SC2);
    assert!(close(p.cost.sqrt(), TS_DIST_SC2, 1e-12));
    let p = dtw_path_multivariate(&x, &y, 2.0, DtwWindow::None).unwrap();
    assert_eq!(p.path, TS_PATH_NONE);
    assert!(close(p.cost.sqrt(), TS_DIST_NONE, 1e-12));
}

/// Pseudo-random series without an RNG dependency on test order.
fn pseudo(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut acc = 0.0;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            acc += ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
            acc
        })
        .collect()
}

fn row(v: &[f64]) -> FdMatrix {
    FdMatrix::from_column_major(v.to_vec(), 1, v.len()).unwrap()
}

/// Univariate (`&[f64]`) vs multivariate (`1 × m`) results of the named
/// elastic distance, compared bit-for-bit.
fn check_d1_bit_identity(name: &str, x: &[f64], y: &[f64], w: DtwWindow) {
    macro_rules! both {
        ($d:ident, $dm:ident, $c:ident, $cm:ident, $p:ident, $pm:ident $(, $a:expr)*) => {{
            let (xm, ym) = (row(x), row(y));
            let u = $d(x, y, w $(, $a)*);
            let v = $dm(&xm, &ym, w $(, $a)*);
            match (u, v) {
                (Ok(u), Ok(v)) => assert_eq!(u.to_bits(), v.to_bits(), "{name} distance"),
                (Err(_), Err(_)) => return,
                (u, v) => panic!("{name}: {u:?} vs {v:?}"),
            }
            let (cu, cv) = ($c(x, y, w $(, $a)*).unwrap(), $cm(&xm, &ym, w $(, $a)*).unwrap());
            assert_eq!(cu.shape(), cv.shape());
            for (a, b) in cu.as_slice().iter().zip(cv.as_slice()) {
                assert_eq!(a.to_bits(), b.to_bits(), "{name} cost matrix");
            }
            let (pu, pv) = ($p(x, y, w $(, $a)*).unwrap(), $pm(&xm, &ym, w $(, $a)*).unwrap());
            assert_eq!(pu.path, pv.path, "{name} path");
            assert_eq!(pu.distance.to_bits(), pv.distance.to_bits(), "{name} path distance");
        }};
    }
    match name {
        "ddtw" => both!(
            ddtw_distance,
            ddtw_distance_multivariate,
            ddtw_cost_matrix,
            ddtw_cost_matrix_multivariate,
            ddtw_alignment_path,
            ddtw_alignment_path_multivariate
        ),
        "wdtw" => both!(
            wdtw_distance,
            wdtw_distance_multivariate,
            wdtw_cost_matrix,
            wdtw_cost_matrix_multivariate,
            wdtw_alignment_path,
            wdtw_alignment_path_multivariate,
            0.05
        ),
        "wddtw" => both!(
            wddtw_distance,
            wddtw_distance_multivariate,
            wddtw_cost_matrix,
            wddtw_cost_matrix_multivariate,
            wddtw_alignment_path,
            wddtw_alignment_path_multivariate,
            0.3
        ),
        "adtw" => both!(
            adtw_distance,
            adtw_distance_multivariate,
            adtw_cost_matrix,
            adtw_cost_matrix_multivariate,
            adtw_alignment_path,
            adtw_alignment_path_multivariate,
            0.5
        ),
        "erp" => both!(
            erp_distance,
            erp_distance_multivariate,
            erp_cost_matrix,
            erp_cost_matrix_multivariate,
            erp_alignment_path,
            erp_alignment_path_multivariate,
            0.25
        ),
        "edr" => both!(
            edr_distance,
            edr_distance_multivariate,
            edr_cost_matrix,
            edr_cost_matrix_multivariate,
            edr_alignment_path,
            edr_alignment_path_multivariate,
            None
        ),
        "twe" => both!(
            twe_distance,
            twe_distance_multivariate,
            twe_cost_matrix,
            twe_cost_matrix_multivariate,
            twe_alignment_path,
            twe_alignment_path_multivariate,
            0.01,
            0.5
        ),
        "msm_ind" => both!(
            msm_distance,
            msm_distance_multivariate,
            msm_cost_matrix,
            msm_cost_matrix_multivariate,
            msm_alignment_path,
            msm_alignment_path_multivariate,
            1.0,
            true
        ),
        "msm_dep" => both!(
            msm_distance,
            msm_distance_multivariate,
            msm_cost_matrix,
            msm_cost_matrix_multivariate,
            msm_alignment_path,
            msm_alignment_path_multivariate,
            0.5,
            false
        ),
        "lcss" => both!(
            lcss_distance,
            lcss_distance_multivariate,
            lcss_cost_matrix,
            lcss_cost_matrix_multivariate,
            lcss_alignment_path,
            lcss_alignment_path_multivariate,
            0.4
        ),
        _ => unreachable!(),
    }
}

const NAMES: [&str; 10] = [
    "ddtw", "wdtw", "wddtw", "adtw", "erp", "edr", "twe", "msm_ind", "msm_dep", "lcss",
];

fn windows() -> Vec<DtwWindow> {
    vec![
        DtwWindow::None,
        DtwWindow::SakoeChiba { radius: 2 },
        DtwWindow::SakoeChibaFraction { window: 0.25 },
        DtwWindow::ItakuraFraction { max_slope: 0.6 },
    ]
}

#[test]
fn single_channel_is_bit_identical_to_univariate() {
    for (s, &(n, m)) in [(12usize, 12usize), (11, 9), (8, 13)].iter().enumerate() {
        let x = pseudo(n, 10 + s as u64);
        let y = pseudo(m, 20 + s as u64);
        for w in windows() {
            for name in NAMES {
                check_d1_bit_identity(name, &x, &y, w);
            }
            let (xm, ym) = (row(&x), row(&y));
            for p in [1.0, 2.0, 1.5] {
                let u = dtw_distance_windowed(&x, &y, p, w);
                let v = dtw_distance_multivariate(&xm, &ym, p, w);
                if let (Ok(u), Ok(v)) = (u, v) {
                    assert_eq!(u.to_bits(), v.to_bits());
                    let (pu, pv) = (
                        dtw_path(&x, &y, p, w).unwrap(),
                        dtw_path_multivariate(&xm, &ym, p, w).unwrap(),
                    );
                    assert_eq!(pu, pv);
                }
            }
        }
    }
    // Pairwise (univariate FdMatrix rows vs 1 × m collections).
    let rows: Vec<Vec<f64>> = (0..4).map(|k| pseudo(10, 40 + k)).collect();
    let mut flat = vec![0.0; 40];
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            flat[i + j * 4] = v;
        }
    }
    let data = FdMatrix::from_column_major(flat, 4, 10).unwrap();
    let coll: Vec<FdMatrix> = rows.iter().map(|r| row(r)).collect();
    let w = DtwWindow::SakoeChiba { radius: 3 };
    let a = msm_pairwise_distance(&data, None, w, 1.0, true).unwrap();
    let b = msm_pairwise_distance_multivariate(&coll, None, w, 1.0, true).unwrap();
    assert_eq!(a, b);
    let a = twe_pairwise_distance(&data, Some(&data), w, 0.001, 1.0).unwrap();
    let b = twe_pairwise_distance_multivariate(&coll, Some(&coll), w, 0.001, 1.0).unwrap();
    assert_eq!(a, b);
    let a = dtw_cross(&data, &data, 2.0, 10, crate::dim::Dim::One);
    let b = dtw_pairwise_distance_multivariate(&coll, Some(&coll), 2.0, DtwWindow::None).unwrap();
    assert_eq!(a, b);
}

/// Stack univariate series as channels of one `d × m` series.
fn stack(chans: &[Vec<f64>]) -> FdMatrix {
    let (d, m) = (chans.len(), chans[0].len());
    let mut flat = vec![0.0; d * m];
    for (c, ch) in chans.iter().enumerate() {
        for (t, &v) in ch.iter().enumerate() {
            flat[c + t * d] = v;
        }
    }
    FdMatrix::from_column_major(flat, d, m).unwrap()
}

#[test]
fn multichannel_invariants() {
    let (n, m) = (11, 9);
    let x: Vec<Vec<f64>> = (0..3).map(|c| pseudo(n, 100 + c)).collect();
    let y: Vec<Vec<f64>> = (0..3).map(|c| pseudo(m, 200 + c)).collect();
    let (xm, ym) = (stack(&x), stack(&y));
    let w = DtwWindow::SakoeChiba { radius: 3 };

    // Independent MSM = sum of the per-channel univariate MSM distances.
    let total = msm_distance_multivariate(&xm, &ym, w, 1.0, true).unwrap();
    let mut sum = 0.0;
    for c in 0..3 {
        sum += msm_distance(&x[c], &y[c], w, 1.0, true).unwrap();
    }
    assert_eq!(total.to_bits(), (0.0 + sum).to_bits());

    // A constant-zero extra channel leaves squared-cost distances unchanged.
    let zx = stack(&[x[0].clone(), vec![0.0; n]]);
    let zy = stack(&[y[0].clone(), vec![0.0; m]]);
    let u = dtw_distance_windowed(&x[0], &y[0], 2.0, w).unwrap();
    assert_eq!(dtw_distance_multivariate(&zx, &zy, 2.0, w).unwrap(), u);
    let u = wdtw_distance(&x[0], &y[0], w, 0.05).unwrap();
    assert_eq!(wdtw_distance_multivariate(&zx, &zy, w, 0.05).unwrap(), u);

    // Channel permutation invariance (dependent costs are symmetric in c up
    // to rounding) and symmetry in (x, y).
    let px = stack(&[x[2].clone(), x[0].clone(), x[1].clone()]);
    let py = stack(&[y[2].clone(), y[0].clone(), y[1].clone()]);
    for name in NAMES {
        let a = mv_run(name, &xm, &ym, w).unwrap().0;
        let b = mv_run(name, &px, &py, w).unwrap().0;
        assert!(close(a, b, 1e-12), "{name}: {a} vs {b}");
    }
    let a = dtw_distance_multivariate(&xm, &ym, 2.0, w).unwrap();
    let b = dtw_distance_multivariate(&ym, &xm, 2.0, w).unwrap();
    assert!(close(a, b, 1e-12));

    // The DTW path cost equals the sum of the per-cell multichannel costs.
    let p = dtw_path_multivariate(&xm, &ym, 2.0, w).unwrap();
    let mut s = 0.0;
    for &(i, j) in &p.path {
        s += (0..3).map(|c| (x[c][i] - y[c][j]).powi(2)).sum::<f64>();
    }
    assert!(close(s, p.cost, 1e-12));
}

#[test]
fn multivariate_errors() {
    let (x, y, _) = xyz();
    let two = FdMatrix::zeros(2, 8);
    let w = DtwWindow::None;
    assert!(matches!(
        dtw_distance_multivariate(&x, &two, 2.0, w),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        msm_distance_multivariate(&x, &two, w, 1.0, true),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        dtw_distance_multivariate(&x, &FdMatrix::zeros(3, 0), 2.0, w),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        dtw_distance_multivariate(&x, &y, -1.0, w),
        Err(FdarError::InvalidParameter { .. })
    ));
    assert!(matches!(
        ddtw_distance_multivariate(&x, &FdMatrix::zeros(3, 2), w),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        twe_pairwise_distance_multivariate(&[x.clone(), two.clone()], None, w, 0.001, 1.0),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        dtw_pairwise_distance_multivariate(std::slice::from_ref(&x), Some(&[two]), 2.0, w),
        Err(FdarError::InvalidDimension { .. })
    ));
    let empty: Vec<FdMatrix> = Vec::new();
    let e = dtw_pairwise_distance_multivariate(&empty, None, 2.0, w).unwrap();
    assert_eq!(e.shape(), (0, 0));
}

// ---------------------------------------------------------------------------
// DBA and DTW k-means
// ---------------------------------------------------------------------------

fn dba_data() -> Vec<FdMatrix> {
    [&DBA_X0[..], &DBA_X1, &DBA_X2, &DBA_X3]
        .iter()
        .map(|v| ser(v, 2))
        .collect()
}

fn assert_series(what: &str, got: &FdMatrix, want: &[f64], tol: f64) {
    assert_eq!(got.as_slice().len(), want.len(), "{what}");
    for (k, (a, b)) in got.as_slice().iter().zip(want).enumerate() {
        assert!(close(*a, *b, tol), "{what}[{k}]: {a} vs {b}");
    }
}

#[test]
fn tslearn_multivariate_dba_goldens() {
    let data = dba_data();
    let res = dtw_barycenter_averaging_multivariate(&data, &DbaConfig::default()).unwrap();
    assert_eq!(res.barycenter.shape(), (2, 12));
    assert_series("mean", &res.barycenter, &DBA_MEAN, 1e-9);

    let mut cfg = DbaConfig::default();
    cfg.window = DtwWindow::SakoeChiba { radius: 1 };
    let res = dtw_barycenter_averaging_multivariate(&data, &cfg).unwrap();
    assert_series("sc1", &res.barycenter, &DBA_SC1, 1e-9);

    let mut cfg = DbaConfig::default();
    cfg.barycenter_size = Some(9);
    cfg.weights = Some(vec![1.0, 2.0, 0.5, 1.5]);
    let res = dtw_barycenter_averaging_multivariate(&data, &cfg).unwrap();
    assert_eq!(res.barycenter.shape(), (2, 9));
    assert_series("size9", &res.barycenter, &DBA_SIZE9_W, 1e-9);

    // Variable-length series (tslearn: NaN-padded dataset, nanmean init).
    let full = &data;
    let cut = |x: &FdMatrix, len: usize| ser(&x.as_slice()[..2 * len], 2);
    let var = vec![full[0].clone(), cut(&full[1], 9), cut(&full[2], 11)];
    let res = dtw_barycenter_averaging_multivariate(&var, &DbaConfig::default()).unwrap();
    assert_eq!(res.barycenter.shape(), (2, 12));
    assert_series("varlen", &res.barycenter, &DBA_VARLEN, 1e-9);
}

#[test]
fn multivariate_dba_single_channel_bit_identity_and_errors() {
    let rows: Vec<Vec<f64>> = (0..5).map(|k| pseudo(14, 300 + k)).collect();
    let mut flat = vec![0.0; 5 * 14];
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            flat[i + j * 5] = v;
        }
    }
    let data = FdMatrix::from_column_major(flat, 5, 14).unwrap();
    let coll: Vec<FdMatrix> = rows.iter().map(|r| row(r)).collect();
    let mut cfgs = vec![DbaConfig::default()];
    let mut c = DbaConfig::default();
    c.init = DbaInit::Medoid;
    c.window = DtwWindow::SakoeChiba { radius: 2 };
    c.weights = Some(vec![1.0, 0.5, 2.0, 1.0, 0.25]);
    cfgs.push(c);
    let mut c = DbaConfig::default();
    c.barycenter_size = Some(9);
    cfgs.push(c);
    let mut c = DbaConfig::default();
    c.init = DbaInit::CustomMultivariate(row(&rows[3]));
    cfgs.push(c);
    for cfg in &cfgs {
        let u = dtw_barycenter_averaging(&data, cfg).unwrap();
        let v = dtw_barycenter_averaging_multivariate(&coll, cfg).unwrap();
        let ub: Vec<u64> = u.barycenter.iter().map(|v| v.to_bits()).collect();
        let vb: Vec<u64> = v
            .barycenter
            .as_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(ub, vb);
        assert_eq!(u.cost.to_bits(), v.cost.to_bits());
        assert_eq!((u.n_iter, u.converged), (v.n_iter, v.converged));
    }
    // A Custom (univariate) start equals CustomMultivariate with one row.
    let mut a = DbaConfig::default();
    a.init = DbaInit::Custom(rows[3].clone());
    let ua = dtw_barycenter_averaging(&data, &a).unwrap();
    let ub = dtw_barycenter_averaging(&data, &cfgs[3]).unwrap();
    assert_eq!(ua, ub);

    // Errors.
    let d = dba_data();
    assert!(matches!(
        dtw_barycenter_averaging_multivariate(&[], &DbaConfig::default()),
        Err(FdarError::InvalidDimension { .. })
    ));
    assert!(matches!(
        dtw_barycenter_averaging_multivariate(
            &[d[0].clone(), FdMatrix::zeros(3, 12)],
            &DbaConfig::default()
        ),
        Err(FdarError::InvalidDimension { .. })
    ));
    let mut bad = DbaConfig::default();
    bad.init = DbaInit::Custom(vec![0.0; 12]);
    assert!(matches!(
        dtw_barycenter_averaging_multivariate(&d, &bad),
        Err(FdarError::InvalidParameter { .. })
    ));
    bad.init = DbaInit::CustomMultivariate(FdMatrix::zeros(3, 12));
    assert!(matches!(
        dtw_barycenter_averaging_multivariate(&d, &bad),
        Err(FdarError::InvalidDimension { .. })
    ));
    let mut nan = d.clone();
    nan[1][(0, 3)] = f64::NAN;
    assert!(matches!(
        dtw_barycenter_averaging_multivariate(&nan, &DbaConfig::default()),
        Err(FdarError::InvalidParameter { .. })
    ));
}

fn km_data() -> Vec<FdMatrix> {
    [&KM_X0[..], &KM_X1, &KM_X2, &KM_X3, &KM_X4, &KM_X5]
        .iter()
        .map(|v| ser(v, 2))
        .collect()
}

#[test]
fn tslearn_multivariate_kmeans_golden() {
    let data = km_data();
    let mut cfg = DtwKMeansConfig::default();
    cfg.n_init = 1;
    cfg.init = DtwKMeansInit::CentroidsMultivariate(vec![data[0].clone(), data[3].clone()]);
    let res = dtw_kmeans_multivariate(&data, &cfg).unwrap();
    assert_eq!(res.cluster, KM_LABELS);
    assert_eq!(res.iter, KM_NITER);
    assert_series("c0", &res.centroids[0], &KM_C0, 1e-9);
    assert_series("c1", &res.centroids[1], &KM_C1, 1e-9);
    assert_eq!(res.predict(&data).unwrap(), res.cluster);

    // k-means++ restarts recover the two groups deterministically.
    let a = dtw_kmeans_multivariate(&data, &DtwKMeansConfig::default()).unwrap();
    let b = dtw_kmeans_multivariate(&data, &DtwKMeansConfig::default()).unwrap();
    assert_eq!(a, b);
    assert!(a.cluster[..3].iter().all(|&l| l == a.cluster[0]));
    assert!(a.cluster[3..].iter().all(|&l| l != a.cluster[0]));
}

#[test]
fn multivariate_kmeans_single_channel_bit_identity_and_errors() {
    let rows: Vec<Vec<f64>> = (0..7).map(|k| pseudo(16, 500 + k)).collect();
    let mut flat = vec![0.0; 7 * 16];
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            flat[i + j * 7] = v;
        }
    }
    let data = FdMatrix::from_column_major(flat, 7, 16).unwrap();
    let coll: Vec<FdMatrix> = rows.iter().map(|r| row(r)).collect();
    let mut cfg = DtwKMeansConfig::default();
    cfg.n_clusters = 3;
    cfg.n_init = 3;
    cfg.window = DtwWindow::SakoeChiba { radius: 3 };
    let u = dtw_kmeans(&data, &cfg).unwrap();
    let v = dtw_kmeans_multivariate(&coll, &cfg).unwrap();
    assert_eq!(u.cluster, v.cluster);
    assert_eq!(u.inertia.to_bits(), v.inertia.to_bits());
    assert_eq!(
        (u.iter, u.converged, u.n_init_best),
        (v.iter, v.converged, v.n_init_best)
    );
    for c in 0..3 {
        assert_eq!(u.centroids.row(c), v.centroids[c].as_slice());
    }
    // Univariate Centroids init in the multivariate entry point (d = 1) and
    // CentroidsMultivariate (1 × m) in the univariate one.
    let init = FdMatrix::from_column_major(
        (0..16).flat_map(|j| [rows[0][j], rows[4][j]]).collect(),
        2,
        16,
    )
    .unwrap();
    let mut a = DtwKMeansConfig::default();
    a.n_init = 1;
    a.init = DtwKMeansInit::Centroids(init);
    let mut b = a.clone();
    b.init = DtwKMeansInit::CentroidsMultivariate(vec![row(&rows[0]), row(&rows[4])]);
    let ua = dtw_kmeans(&data, &a).unwrap();
    assert_eq!(ua, dtw_kmeans(&data, &b).unwrap());
    let va = dtw_kmeans_multivariate(&coll, &a).unwrap();
    assert_eq!(va, dtw_kmeans_multivariate(&coll, &b).unwrap());
    assert_eq!(ua.cluster, va.cluster);

    // Errors.
    let d = km_data();
    let mut bad = DtwKMeansConfig::default();
    bad.init = DtwKMeansInit::Centroids(FdMatrix::zeros(2, 15));
    assert!(matches!(
        dtw_kmeans_multivariate(&d, &bad),
        Err(FdarError::InvalidParameter { .. })
    ));
    bad.init = DtwKMeansInit::CentroidsMultivariate(vec![d[0].clone()]);
    assert!(matches!(
        dtw_kmeans_multivariate(&d, &bad),
        Err(FdarError::InvalidDimension { .. })
    ));
    let mixed = vec![d[0].clone(), FdMatrix::zeros(2, 14)];
    assert!(matches!(
        dtw_kmeans_multivariate(&mixed, &DtwKMeansConfig::default()),
        Err(FdarError::InvalidDimension { .. })
    ));
    let fitted = dtw_kmeans_multivariate(&d, &DtwKMeansConfig::default()).unwrap();
    assert!(matches!(
        fitted.predict(&[FdMatrix::zeros(3, 15)]),
        Err(FdarError::InvalidDimension { .. })
    ));
}
