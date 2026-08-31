//! Shared χ² / regularized-incomplete-gamma numerical primitives (Phase-49 CONS-01).
//!
//! This module is the single crate-internal home for the special-function machinery that was
//! previously duplicated as two independent hand-rolled kernels — one SF-oriented
//! (`inference/dist.rs`) and one CDF-oriented (`spm/chi_squared.rs`). It hosts the ONE shared
//! [`ln_gamma`] Lanczos primitive plus the CDF-family regularized incomplete gamma
//! ([`reg_gamma_p`] / [`reg_gamma_q`]) and the χ² wrappers ([`chi2_cdf`], [`chi2_quantile`],
//! [`chi2_sf`]).
//!
//! # Why two tail wrappers, not one kernel
//!
//! A standalone bit-for-bit comparison (Phase-49 research) established that a SINGLE kernel CANNOT
//! reproduce both call families bit-identically: their shared `ln_gamma` is bit-identical, but the
//! regularized incomplete gamma diverges by up to ~51 ULP because the two kernels branch differently
//! between series and continued fraction — and in the far upper tail the divergence is
//! **catastrophic** (χ²-SF at x=70.59, k=1: the SF `Q`-direct path yields `4.397e-17`, the CDF `1−P`
//! route yields exactly `0.0`, a total loss of precision to upper-tail cancellation).
//!
//! Therefore this module **shares only the primitives and splits the tail policy**:
//! - the **SF family** ([`chi2_sf`]) keeps its OWN `Q`-direct continued fraction
//!   ([`gamma_q_cf_sf`] / [`gamma_p_series_sf`], `tiny = 1e-300`, inline `1e-15`, no underflow guard)
//!   — verbatim from `inference/dist.rs` — to avoid the `1 − P` far-tail cliff;
//! - the **CDF family** ([`reg_gamma_p`], and thus [`chi2_cdf`] / [`chi2_quantile`]) keeps its
//!   `P`-direct path (`tiny = 1e-30`, `eps = 1e-14`, `−700` underflow guard) — verbatim from
//!   `spm/chi_squared.rs`.
//!
//! This is a CODE-MOTION refactor: every operation is preserved in the same order as its source, so
//! all existing call sites stay BIT-IDENTICAL (locked by `tests/equivalence_phase49.rs`).
//!
//! # References
//!
//! - Pugh, G.R. (2004). *An Analysis of the Lanczos Gamma Approximation*. Ph.D. thesis, UBC.
//!   Table 4, g=7, n=9.
//! - Johnson, N.L., Kotz, S. & Balakrishnan, N. (1994). *Continuous Univariate Distributions*,
//!   Vol. 1. Wiley. §18.6.2 (Wilson–Hilferty).
//! - Abramowitz, M. & Stegun, I.A. (1964). *Handbook of Mathematical Functions*. Dover. 26.2.23.

use std::f64::consts::PI;

/// Natural logarithm of the gamma function via the Lanczos approximation (g = 7, n = 9).
///
/// Adopts the guarded reflection form (`sin().abs().ln()` with a `<1e-30 → INFINITY` guard) — the
/// strictly safer of the two pre-consolidation copies. For all χ² arguments `a = k/2 > 0` the
/// reflection branch (`x < 0.5`) is dead code, so this change is bit-identical for every χ² value
/// (confirmed by the equivalence goldens). Source: extracted verbatim from `spm/chi_squared.rs`;
/// the published Lanczos coefficients are intentionally full-precision.
#[allow(clippy::excessive_precision)]
pub(crate) fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients for g = 7, n = 9 (Pugh, 2004, Table 4)
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        let ln_pi = PI.ln();
        let sin_val = (PI * x).sin();
        if sin_val.abs() < 1e-30 {
            return f64::INFINITY;
        }
        return ln_pi - sin_val.abs().ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for i in 1..9 {
        sum += COEFFS[i] / (x + i as f64);
    }

    let t = x + G + 0.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// CDF family — P-direct regularized incomplete gamma (verbatim from spm/chi_squared.rs).
// Constants: tiny = 1e-30, eps = 1e-14, −700 underflow guard.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Regularized lower incomplete gamma function P(a, x).
///
/// `P(a, x) = gamma(a, x) / Gamma(a)`. Series expansion for `x < a + 1`, continued fraction
/// otherwise. Extracted verbatim from `spm/chi_squared.rs::regularized_gamma_p`.
pub(crate) fn reg_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    if a <= 0.0 {
        return 1.0;
    }

    if x < a + 1.0 {
        // Series expansion
        gamma_series(a, x)
    } else {
        // Continued fraction (upper tail), then complement
        1.0 - gamma_cf(a, x)
    }
}

/// Regularized upper incomplete gamma function Q(a, x) = 1 − P(a, x) (CDF family only).
pub(crate) fn reg_gamma_q(a: f64, x: f64) -> f64 {
    1.0 - reg_gamma_p(a, x)
}

/// Series expansion for the regularized lower incomplete gamma P(a, x).
///
/// `P(a, x) = exp(-x + a*ln(x) - ln(Gamma(a))) * sum_{n>=0} x^n / (a*(a+1)*...*(a+n))`.
fn gamma_series(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let max_iter = 200;
    let eps = 1e-14;

    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;

    for _ in 0..max_iter {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * eps {
            break;
        }
    }

    let log_prefix = a * x.ln() - x - ln_gamma_a;
    if log_prefix < -700.0 {
        return 0.0;
    }
    sum * log_prefix.exp()
}

/// Continued fraction for the regularized upper incomplete gamma Q(a, x) = 1 − P(a, x).
///
/// Modified Lentz algorithm; converges superlinearly for `x > a + 1`.
fn gamma_cf(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }

    let log_prefix = a * x.ln() - x - ln_gamma_a;
    if log_prefix < -700.0 {
        return 0.0;
    }
    log_prefix.exp() * h
}

/// Chi-squared CDF: P(χ²(k) <= x) = `reg_gamma_p(k/2, x/2)`.
///
/// Extracted verbatim from `spm/chi_squared.rs::chi2_cdf` (x<=0 → 0.0, k==0 → 1.0 guards).
pub(crate) fn chi2_cdf(x: f64, k: usize) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if k == 0 {
        return 1.0;
    }
    reg_gamma_p(k as f64 / 2.0, x / 2.0)
}

/// Chi-squared quantile function (inverse CDF).
///
/// Wilson–Hilferty initial approximation followed by Newton–Raphson refinement over [`chi2_cdf`].
/// Extracted verbatim from `spm/chi_squared.rs::chi2_quantile`.
pub(crate) fn chi2_quantile(p: f64, k: usize) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if k == 0 {
        return 0.0;
    }

    let df = k as f64;

    // Wilson-Hilferty initial approximation
    let z = normal_quantile_approx(p);
    let ratio = 2.0 / (9.0 * df);
    let cube = 1.0 - ratio + z * ratio.sqrt();
    let mut x = df * cube * cube * cube;
    if x <= 0.0 {
        x = 0.01;
    }

    // Newton-Raphson refinement
    let max_iter = 50;
    let tol = 1e-12;

    for _ in 0..max_iter {
        let cdf_val = chi2_cdf(x, k);
        let error = cdf_val - p;
        if error.abs() < tol {
            break;
        }

        // PDF of chi2: f(x) = x^{k/2-1} * exp(-x/2) / (2^{k/2} * Gamma(k/2))
        let log_pdf =
            (df / 2.0 - 1.0) * x.ln() - x / 2.0 - (df / 2.0) * 2.0_f64.ln() - ln_gamma(df / 2.0);
        let pdf = log_pdf.exp();

        if pdf < 1e-30 {
            break;
        }

        let delta = error / pdf;
        x -= delta;

        // Ensure x stays positive
        if x <= 0.0 {
            x = tol;
        }
    }

    x
}

/// Approximate normal quantile (probit) via the Abramowitz–Stegun 26.2.23 rational approximation.
///
/// Extracted verbatim from `spm/chi_squared.rs::normal_quantile_approx`; used only by
/// [`chi2_quantile`]'s Wilson–Hilferty seed.
fn normal_quantile_approx(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let p = if p < 0.5 { p } else { 1.0 - p };

    let t = (-2.0 * p.ln()).sqrt();

    // Rational approximation coefficients
    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;

    let num = c0 + c1 * t + c2 * t * t;
    let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;

    sign * (t - num / den)
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// SF family — Q-direct continued fraction (verbatim from inference/dist.rs).
// Constants: tiny = 1e-300, inline eps = 1e-15, NO underflow guard. Kept private to this module and
// used ONLY by chi2_sf, so the SF far tail avoids the CDF family's `1 − P` cancellation cliff.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Regularized lower incomplete gamma P(a, x) via the series expansion (converges for `x < a + 1`).
/// SF-family variant — verbatim from `inference/dist.rs::gamma_p_series` (inline `1e-15`, no guard).
fn gamma_p_series_sf(a: f64, x: f64) -> f64 {
    // Uses the series P(a,x) = x^a e^{-x} / Γ(a) · Σ_{n≥0} x^n / (a(a+1)...(a+n))
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..200 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Regularized upper incomplete gamma Q(a, x) via the continued fraction (converges for `x >= a + 1`).
/// SF-family variant — verbatim from `inference/dist.rs::gamma_q_cf` (tiny = 1e-300, inline `1e-15`).
fn gamma_q_cf_sf(a: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..200 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Chi-squared survival function: P(X > x) for X ~ χ²(df), `df: f64` (real degrees of freedom).
///
/// SF-direct: keeps its own `Q`-continued-fraction upper tail to avoid the `1 − P` far-tail cliff.
/// The single `df: f64` entry serves BOTH the integer-`k` site (`chi_square_sf`, called with
/// `k as f64`) and the real-`df` site (`chi_square_sf_df`) — both derive `a = <df>/2.0`, so at
/// integer df they produce identical f64 (equivalence golden A1). Body is verbatim from
/// `inference/dist.rs::chi_square_sf_df`.
pub(crate) fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 1.0;
    }
    let a = df / 2.0;
    let xx = x / 2.0;
    if xx < a + 1.0 {
        1.0 - gamma_p_series_sf(a, xx)
    } else {
        gamma_q_cf_sf(a, xx)
    }
}
