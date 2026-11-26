//! Two-step generalized FPCA (R registr 2.2.1 `gfpca_twoStep`, `cov_hall`;
//! Gertheiss, Goldsmith & Staicu 2017, *Stat* 6(1)), implemented natively:
//!
//! 1. **Covariance** (registr `cov_hall`, after Hall, Müller & Yao 2008): a
//!    P-spline GLM mean `μ(t)` (mgcv `gam(value ~ s(index, bs = "ps", k = Kt))`,
//!    smoothing parameter by GCV / UBRE), raw products of the centred
//!    observations, a tensor-product P-spline smooth of the off-diagonal
//!    products (mgcv `te(i1, i2, k = 10, bs = "ps")`, GCV), the delta-method
//!    transformation to the latent scale, a diagonal floor of `0.01` and the
//!    nearest positive-definite matrix (`Matrix::nearPD`).
//! 2. **Eigen-decomposition** on the common grid of (rounded) time points.
//! 3. **Scores and mean** from the mixed model `g(E[yᵢⱼ]) = f(tᵢⱼ) + Σₖ φₖ(tᵢⱼ)
//!    sᵢₖ` (registr: `gamm4`), see [`super::glmm`].

use nalgebra::DMatrix;

use crate::error::FdarError;
use crate::irreg_fdata::IrregFdata;
use crate::matrix::FdMatrix;

use super::basis::{glm_fit, glm_mean, template_knots, TemplateBasis};
use super::gfpca::{
    check_family_values, curve_data, CurveData, FpcaType, GfpcaConfig, GfpcaResult,
};
use super::glmm::{fit_glmm, GlmmData};
use super::mgcv::{gam_ps, near_pd, sym_eigen_desc, te_ps_gcv};
use super::ExpFamily;

/// Basis dimension of each margin of the covariance smooth (registr `Kc`).
const KC: usize = 10;
/// Diagonal floor of the latent covariance (registr `diag_epsilon`).
const DIAG_EPS: f64 = 0.01;

/// R `round(x, digits)` (half to even on the scaled value).
fn round_digits(x: f64, digits: i32) -> f64 {
    let p = 10f64.powi(digits);
    (x * p).round_ties_even() / p
}

/// registr `coarsen_index`: round to `digits` significant digits relative to
/// the largest magnitude (number of integer digits of `round(max |x|)`, or 0
/// when all `|x| < 1`).
pub(super) fn coarsen_index(x: &[f64], digits: u32) -> Vec<f64> {
    let max_abs = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let max_round = max_abs.round();
    let mut pre: i32 = if max_round >= 1.0 {
        max_round.log10().floor() as i32 + 1
    } else {
        1
    };
    if pre == 1 && max_abs < 1.0 {
        pre = 0;
    }
    let scale = 10f64.powi(pre - 1);
    x.iter()
        .map(|&v| {
            let sign = if v > 0.0 { 1.0 } else { -1.0 };
            sign * round_digits(v.abs() / scale, digits as i32 - 1) * scale
        })
        .collect()
}

/// Observations on the common output grid.
struct Gridded {
    /// Sorted unique (rounded) time points (registr `output_index`).
    grid: Vec<f64>,
    /// Rounded times and values per curve (all observations).
    t: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    /// Per curve: `(grid position, value)` with one entry per occupied grid
    /// point (the last observation wins, as registr's `Y.obs` pivot).
    cells: Vec<Vec<(usize, f64)>>,
    /// Grid position of every observation.
    pos: Vec<Vec<usize>>,
}

fn grid_data(data: &CurveData, digits: Option<u32>, family: ExpFamily) -> Gridded {
    let t: Vec<Vec<f64>> = match digits {
        Some(d) => {
            let flat: Vec<f64> = data.t.iter().flatten().copied().collect();
            let c = coarsen_index(&flat, d);
            let mut it = c.into_iter();
            data.t
                .iter()
                .map(|ti| it.by_ref().take(ti.len()).collect())
                .collect()
        }
        None => data.t.clone(),
    };
    let y: Vec<Vec<f64>> = data
        .y
        .iter()
        .map(|yi| match family {
            ExpFamily::Poisson => yi.iter().map(|v| v.round_ties_even()).collect(),
            _ => yi.clone(),
        })
        .collect();
    let mut grid: Vec<f64> = t.iter().flatten().copied().collect();
    grid.sort_by(f64::total_cmp);
    grid.dedup();
    // `partial_cmp`: registr's rounding maps 0 to −0, which equals +0.
    let find = |v: f64| {
        grid.binary_search_by(|g| g.partial_cmp(&v).expect("finite times"))
            .expect("grid holds every time")
    };
    let pos: Vec<Vec<usize>> = t
        .iter()
        .map(|ti| ti.iter().map(|&v| find(v)).collect())
        .collect();
    let cells = pos
        .iter()
        .zip(&y)
        .map(|(p, yi)| {
            let mut row: Vec<Option<f64>> = vec![None; grid.len()];
            for (&k, &v) in p.iter().zip(yi) {
                row[k] = Some(v);
            }
            row.iter()
                .enumerate()
                .filter_map(|(k, v)| v.map(|v| (k, v)))
                .collect()
        })
        .collect();
    Gridded {
        grid,
        t,
        y,
        cells,
        pos,
    }
}

/// Raw products of centred values: all ordered within-curve pairs of distinct
/// times (registr `crossprods_irregular`) or grid-pair averages over curves
/// (`crossprods_regular`, used unless more than half of the observations have
/// distinct times).
fn crossprods(g: &Gridded, centred: &[Vec<f64>]) -> Vec<(f64, f64, f64)> {
    let n_obs: usize = g.t.iter().map(Vec::len).sum();
    if g.grid.len() as f64 > 0.5 * n_obs as f64 {
        crossprods_irregular(g, centred)
    } else {
        crossprods_regular(g, centred)
    }
}

fn crossprods_irregular(g: &Gridded, centred: &[Vec<f64>]) -> Vec<(f64, f64, f64)> {
    let mut rows = Vec::new();
    for (ti, ci) in g.t.iter().zip(centred) {
        for (&ta, &ca) in ti.iter().zip(ci) {
            rows.extend(
                ti.iter()
                    .zip(ci)
                    .filter(|(&tb, _)| tb != ta)
                    .map(|(&tb, &cb)| (ta, tb, ca * cb)),
            );
        }
    }
    rows
}

fn crossprods_regular(g: &Gridded, centred: &[Vec<f64>]) -> Vec<(f64, f64, f64)> {
    let d = g.grid.len();
    let mut sum = vec![0.0; d * d];
    let mut cnt = vec![0usize; d * d];
    for (pos, ci) in g.pos.iter().zip(centred) {
        // Centred value per occupied grid cell (last observation wins).
        let mut val: Vec<Option<f64>> = vec![None; d];
        for (&k, &c) in pos.iter().zip(ci) {
            val[k] = Some(c);
        }
        let occupied: Vec<(usize, f64)> = val
            .iter()
            .enumerate()
            .filter_map(|(k, v)| v.map(|v| (k, v)))
            .collect();
        for &(a, va) in &occupied {
            for &(b, vb) in occupied.iter().filter(|(b, _)| *b != a) {
                sum[a * d + b] += va * vb;
                cnt[a * d + b] += 1;
            }
        }
    }
    (0..d * d)
        .filter(|&k| cnt[k] > 0)
        .map(|k| (g.grid[k / d], g.grid[k % d], sum[k] / cnt[k] as f64))
        .collect()
}

/// registr `cov_hall` on the gridded data: latent covariance on `grid`.
fn cov_hall(g: &Gridded, kt: usize, family: ExpFamily) -> Result<DMatrix<f64>, FdarError> {
    let mut y: Vec<Vec<f64>> = g.y.clone();
    if family == ExpFamily::Gamma {
        let min = y.iter().flatten().copied().fold(f64::INFINITY, f64::min);
        if min < 1.01 {
            y.iter_mut().flatten().for_each(|v| *v = *v - min + 1.01);
        }
    }
    let ts: Vec<f64> = g.t.iter().flatten().copied().collect();
    let ys: Vec<f64> = y.iter().flatten().copied().collect();
    let (basis, fit) = gam_ps(&ts, &ys, kt, family)?;
    let mut it = fit.eta.iter();
    let centred: Vec<Vec<f64>> = y
        .iter()
        .map(|yi| {
            yi.iter()
                .map(|&v| v - glm_mean(family, *it.next().expect("one eta per obs")).0)
                .collect()
        })
        .collect();
    let rows = crossprods(g, &centred);
    if rows.len() < KC * KC {
        return Err(FdarError::InvalidParameter {
            parameter: "data",
            message: format!(
                "two-step GFPCA needs at least {} within-curve cross products, got {}",
                KC * KC,
                rows.len()
            ),
        });
    }
    let te = te_ps_gcv(&rows, KC)?;
    let sm = te.predict_grid(&g.grid);
    let mut cov = (&sm + sm.transpose()) * 0.5;
    if family != ExpFamily::Gaussian {
        let eta: Vec<f64> = g
            .grid
            .iter()
            .map(|&x| {
                basis
                    .eval(x)
                    .iter()
                    .zip(&fit.coef)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();
        // registr: binomial `diag(1/μ'(η)) C diag(1/μ'(η))`; gamma / poisson
        // `diag(η) C diag(η)` with η the log-scale mean.
        let f: Vec<f64> = eta
            .iter()
            .map(|&e| match family {
                ExpFamily::Binomial => {
                    let p = 1.0 / (1.0 + (-e).exp());
                    1.0 / (p * (1.0 - p))
                }
                _ => e,
            })
            .collect();
        let d = g.grid.len();
        for a in 0..d {
            for b in 0..d {
                cov[(a, b)] *= f[a] * f[b];
            }
        }
    }
    for k in 0..g.grid.len() {
        if cov[(k, k)] < DIAG_EPS {
            cov[(k, k)] = DIAG_EPS;
        }
    }
    Ok(near_pd(&cov))
}

/// Least-squares (gamma: Gamma-log GLM on the mean scale) spline coefficients of
/// a latent curve given on `grid` (registr `registr_oneCurve` with a two-step
/// FPCA object).
fn spline_coefs(
    basis: &TemplateBasis,
    domain: [f64; 2],
    grid: &[f64],
    values: &[f64],
    family: ExpFamily,
) -> Result<Vec<f64>, FdarError> {
    let rows: Vec<Vec<f64>> = grid
        .iter()
        .map(|&x| basis.eval(x.clamp(domain[0], domain[1])))
        .collect();
    if family == ExpFamily::Gamma {
        let y: Vec<f64> = values
            .iter()
            .map(|&v| {
                let e = v.exp();
                if e > 0.0 && e < 1e-4 {
                    1e-4
                } else {
                    e
                }
            })
            .collect();
        glm_fit(&rows, &y, ExpFamily::Gamma)
    } else {
        glm_fit(&rows, values, ExpFamily::Gaussian)
    }
}

fn to_fd_rows(rows: &[Vec<f64>], ncol: usize) -> FdMatrix {
    let mut m = FdMatrix::zeros(rows.len(), ncol);
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            m[(i, j)] = v;
        }
    }
    m
}

/// Eigenpairs of the latent covariance with `npc` chosen.
struct Eigen {
    efunctions: Vec<Vec<f64>>,
    evalues: Vec<f64>,
    evalues_sum: f64,
}

fn eigen_npc(cov: &DMatrix<f64>, cfg: &GfpcaConfig) -> Result<Eigen, FdarError> {
    let (vals, vecs) = sym_eigen_desc(cov);
    let pos: Vec<usize> = (0..vals.len()).filter(|&k| vals[k] > 0.0).collect();
    let lambda: Vec<f64> = pos.iter().map(|&k| vals[k]).collect();
    let npc = cfg.npc_criterion.map_or(cfg.npc, |c| c.choose(&lambda));
    if npc == 0 || npc > lambda.len() {
        return Err(FdarError::InvalidParameter {
            parameter: "npc",
            message: format!(
                "must be in 1..={} (positive covariance eigenvalues), got {npc}",
                lambda.len()
            ),
        });
    }
    let d = cov.nrows();
    Ok(Eigen {
        efunctions: (0..d)
            .map(|r| (0..npc).map(|k| vecs[(r, pos[k])]).collect())
            .collect(),
        evalues: lambda[..npc].to_vec(),
        evalues_sum: lambda.iter().sum(),
    })
}

/// Core of [`gfpca_two_step`] on validated data; `domain` is the range of the
/// spline basis of the subject coefficients (default: range of the times) and
/// `start` optional starting variance parameters of the mixed model.
pub(super) fn two_step_core(
    data: &CurveData,
    family: ExpFamily,
    cfg: &GfpcaConfig,
    domain: Option<[f64; 2]>,
    start: Option<&[f64]>,
) -> Result<GfpcaResult, FdarError> {
    let pooled = data.pooled_sorted_t();
    let domain = domain.unwrap_or([pooled[0], pooled[pooled.len() - 1]]);
    let knots = template_knots(&pooled, cfg.kt, cfg.periodic);
    let g = grid_data(data, cfg.index_significant_digits, family);
    let cov = cov_hall(&g, cfg.kt, family)?;
    let eig = eigen_npc(&cov, cfg)?;
    let npc = eig.evalues.len();
    let glmm_data = glmm_rows(&g, &eig.efunctions, family);
    let start = start.filter(|s| s.len() == npc + 1);
    let fit = fit_glmm(&glmm_data, cfg.kt, npc, start)?;
    let d = g.grid.len();
    let alpha: Vec<f64> = g.grid.iter().map(|&x| fit.mean_at(x)).collect();
    let yhat: Vec<Vec<f64>> = fit
        .scores
        .iter()
        .map(|s| {
            (0..d)
                .map(|j| alpha[j] + (0..npc).map(|k| s[k] * eig.efunctions[j][k]).sum::<f64>())
                .collect()
        })
        .collect();
    let basis = TemplateBasis::new(domain[0], domain[1], &knots, cfg.periodic);
    let kt = basis.nbasis();
    let subject: Vec<Vec<f64>> = yhat
        .iter()
        .map(|yh| spline_coefs(&basis, domain, &g.grid, yh, family))
        .collect::<Result<_, _>>()?;
    let alpha_coefs = spline_coefs(&basis, domain, &g.grid, &alpha, ExpFamily::Gaussian)?;
    let psi_cols: Vec<Vec<f64>> = (0..npc)
        .map(|k| {
            let col: Vec<f64> = eig.efunctions.iter().map(|r| r[k]).collect();
            spline_coefs(&basis, domain, &g.grid, &col, ExpFamily::Gaussian)
        })
        .collect::<Result<_, _>>()?;
    let mut psi_coefs = FdMatrix::zeros(kt, npc);
    for (k, col) in psi_cols.iter().enumerate() {
        for (r, &v) in col.iter().enumerate() {
            psi_coefs[(r, k)] = v;
        }
    }
    let fitted = g
        .pos
        .iter()
        .zip(&yhat)
        .map(|(p, yh)| p.iter().map(|&k| yh[k]).collect())
        .collect();
    Ok(GfpcaResult {
        fpca_type: FpcaType::TwoStep,
        em_family: family,
        knots,
        domain,
        alpha_coefs,
        psi_coefs,
        mean: alpha,
        efunctions: to_fd_rows(&eig.efunctions, npc),
        evalues: eig.evalues,
        scores: to_fd_rows(&fit.scores, npc),
        subject_coefs: to_fd_rows(&subject, kt),
        fitted,
        sigma2: fit.sigma2,
        n_iter: fit.n_iter,
        converged: fit.converged,
        evalues_sum: Some(eig.evalues_sum),
        variance_components: Some(fit.theta),
        yhat: Some(to_fd_rows(&yhat, d)),
        t_vec: g.grid,
    })
}

fn glmm_rows(g: &Gridded, efun: &[Vec<f64>], family: ExpFamily) -> GlmmData {
    let (mut y, mut t, mut psi, mut curves) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for cells in &g.cells {
        let lo = y.len();
        for &(k, v) in cells {
            y.push(v);
            t.push(g.grid[k]);
            psi.push(efun[k].clone());
        }
        curves.push(lo..y.len());
    }
    GlmmData {
        y,
        t,
        psi,
        curves,
        family,
    }
}

/// Two-step generalized functional PCA of exponential-family functional data
/// (R registr 2.2.1 `gfpca_twoStep`; Gertheiss, Goldsmith & Staicu 2017).
///
/// Unlike [`super::gfpca_variational`], the decomposition is family-specific for
/// gaussian, binary, gamma and count data alike: the latent covariance is
/// estimated from smoothed cross products of the centred data (registr
/// `cov_hall`), its eigenvectors on the grid of time points (rounded to
/// [`GfpcaConfig::index_significant_digits`]) are the FPCs, and the mean and
/// scores come from a generalized linear mixed model with a P-spline mean and
/// independent random FPC scores. See the [module documentation](super) for the
/// deviations from R (no `mgcv` / `gamm4`; native GCV / UBRE, REML and Laplace
/// fits).
///
/// Output conventions follow registr: `t_vec` is the grid of rounded time
/// points, `efunctions` are unit-norm eigenvectors on it (`t_vec.len() × npc`,
/// arbitrary sign), `evalues` the matrix eigenvalues, `mean` the latent mean
/// `α` on `t_vec` and `yhat` the latent fits `α + Σₖ sᵢₖ φₖ`. `subject_coefs`
/// (`n × kt`) are the spline coefficients registr's registration step fits to
/// every row of `yhat` (least squares; gamma: Gamma-log GLM), `alpha_coefs` /
/// `psi_coefs` the least-squares spline coefficients of `mean` / `efunctions`.
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] / [`FdarError::InvalidParameter`] for fewer
/// than 2 curves, invalid curves or values, an invalid `config`, too few
/// cross products for the covariance smooth (< 100) or `npc` larger than the
/// number of positive eigenvalues; [`FdarError::ComputationFailed`] if a
/// penalised fit is singular.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{gfpca_two_step, ExpFamily, GfpcaConfig};
/// use fdars_core::irreg_fdata::IrregFdata;
///
/// let t: Vec<f64> = (0..25).map(|j| j as f64 / 24.0).collect();
/// let argvals: Vec<Vec<f64>> = (0..15).map(|_| t.clone()).collect();
/// let values: Vec<Vec<f64>> = (0..15)
///     .map(|i| {
///         let a = (i as f64 - 7.0) / 4.0;
///         let b = ((i * 7) % 5) as f64 / 4.0 - 0.5;
///         t.iter()
///             .enumerate()
///             .map(|(j, &s)| {
///                 let noise = 0.05 * (((i * 31 + j * 17) % 13) as f64 / 6.0 - 1.0);
///                 (2.0 * std::f64::consts::PI * s).sin() + a * s + b * (3.0 * s).cos() + noise
///             })
///             .collect()
///     })
///     .collect();
/// let data = IrregFdata::from_lists(&argvals, &values);
/// let mut cfg = GfpcaConfig::default();
/// cfg.npc = 2;
/// let fit = gfpca_two_step(&data, ExpFamily::Gaussian, &cfg).unwrap();
/// assert_eq!(fit.efunctions.shape(), (25, 2));
/// assert_eq!(fit.scores.shape(), (15, 2));
/// assert!(fit.evalues[0] >= fit.evalues[1]);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gfpca_two_step(
    data: &IrregFdata,
    family: ExpFamily,
    config: &GfpcaConfig,
) -> Result<GfpcaResult, FdarError> {
    config.validate()?;
    let cd = curve_data(data, "data")?;
    check_family_values(&cd, family)?;
    two_step_core(&cd, family, config, None, None)
}

/// Latent covariance on the output grid (registr `cov_hall` as called by
/// `gfpca_twoStep`), for tests.
#[cfg(test)]
pub(super) fn latent_covariance(
    data: &CurveData,
    family: ExpFamily,
    cfg: &GfpcaConfig,
) -> Result<(Vec<f64>, DMatrix<f64>), FdarError> {
    let g = grid_data(data, cfg.index_significant_digits, family);
    let cov = cov_hall(&g, cfg.kt, family)?;
    Ok((g.grid, cov))
}
