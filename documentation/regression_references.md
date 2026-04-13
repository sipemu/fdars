# Regression Methods -- References

This document maps each public regression function to the paper(s) that describe the underlying method.

## Scalar-on-Function (y ~ X(t))

| Function | Method | Reference |
|----------|--------|-----------|
| `fregre_lm` | FPC linear model: y = alpha + integral(beta(t) X(t)) dt | Ramsay & Silverman (2005), Ch. 12-13 |
| `fregre_pls` | PLS regression: y = alpha + PLS_scores(X) beta | Preda & Saporta (2005) |
| `fregre_l1` | L1 (median) robust via IRLS | Febrero-Bande & Oviedo de la Fuente (2012) |
| `fregre_huber` | Huber M-estimation via IRLS | Febrero-Bande & Oviedo de la Fuente (2012) |
| `fregre_np_mixed` | Nadaraya-Watson kernel regression with product kernels | Ferraty & Vieu (2006), Ch. 5 |
| `functional_logistic` | Logistic regression for binary outcomes | Ramsay & Silverman (2005), Ch. 14 |
| `fregre_lm_multi` | Multiple functional predictors: y = alpha + sum_k integral(beta_k(t) X_k(t)) dt | Ramsay & Silverman (2005), Ch. 15; Febrero-Bande & Oviedo de la Fuente (2012) |

## Function-on-Scalar (Y(t) ~ z)

| Function | Method | Reference |
|----------|--------|-----------|
| `fosr` | Penalized OLS: Y(t) = alpha(t) + Z beta(t) + eps(t) | Ramsay & Silverman (2005), Ch. 13 |
| `fosr_fpc` | FPC-based FOSR | Ramsay & Silverman (2005) |
| `fosr_2d` | 2D tensor-product FOSR for surfaces | Wood (2006) |
| `fanova` | Functional ANOVA with permutation tests | Cuevas et al. (2004) |

## Function-on-Function (Y(s) ~ X(t))

| Function | Method | Reference |
|----------|--------|-----------|
| `fof_regression` | Double FPCA: Y-scores regressed on X-scores, beta(s,t) reconstructed | Ramsay & Silverman (2005), Ch. 16-17; Yao et al. (2005b) |
| `fof_cv` | K-fold CV for (ncomp_x, ncomp_y) selection via integrated MSE | Ivanescu et al. (2015) |

## Elastic Regression

| Function | Method | Reference |
|----------|--------|-----------|
| `elastic_regression` | Scalar-on-function with alignment: y = integral(q beta) dt | Tucker et al. (2014) |
| `elastic_pcr` | Principal component regression on elastic FPCA scores | Tucker et al. (2013); Tucker et al. (2014) |
| `elastic_logistic` | Logistic regression with alignment (Armijo line search) | Tucker et al. (2014) |
| `scalar_on_shape` | y = h(shape) + g(amplitude) + eps (ScoSh) | Srivastava & Klassen (2016), S10.3 |

## Mixed Effects

| Function | Method | Reference |
|----------|--------|-----------|
| `fmm` | Functional additive mixed model: Y_ij(t) = mu(t) + X'beta(t) + b_i(t) + eps | Scheipl et al. (2015) |

## Cross-Validation

| Function | Method | Reference |
|----------|--------|-----------|
| `fregre_cv` | K-fold CV for ncomp selection in `fregre_lm` | -- |
| `fregre_lm_multi_cv` | K-fold CV for shared ncomp in multi-predictor regression | Febrero-Bande & Oviedo de la Fuente (2012) |
| `fof_cv` | K-fold CV for (ncomp_x, ncomp_y) in function-on-function | Ivanescu et al. (2015) |
| `model_selection_ncomp` | GCV/AIC/BIC model selection for `fregre_lm` | -- |
| `cv_fdata_with_metrics` | Generic K-fold + repeated CV with pluggable metrics | -- |

## FPCA / PLS Decomposition

| Function | Method | Reference |
|----------|--------|-----------|
| `fdata_to_pc_1d` | FPCA via weighted SVD (Simpson's integration weights) | Ramsay & Silverman (2005), Ch. 8 |
| `fdata_to_pls_1d` | PLS via NIPALS with integration weights | Preda & Saporta (2005) |
| `ridge_regression_fit` | Ridge regression (requires `linalg` feature) | Hoerl & Kennard (1970) |

---

## Bibliography

- Cuevas, A., Febrero, M. & Fraiman, R. (2004). An ANOVA test for functional data. *Computational Statistics & Data Analysis*, 47(1), 111--122. [doi:10.1016/j.csda.2003.10.021](https://doi.org/10.1016/j.csda.2003.10.021)
- Febrero-Bande, M. & Oviedo de la Fuente, M. (2012). Statistical Computing in Functional Data Analysis: The R Package fda.usc. *Journal of Statistical Software*, 51(4), 1--28. [doi:10.18637/jss.v051.i04](https://doi.org/10.18637/jss.v051.i04)
- Ferraty, F. & Vieu, P. (2006). *Nonparametric Functional Data Analysis*. Springer. [doi:10.1007/0-387-36620-2](https://doi.org/10.1007/0-387-36620-2)
- Hoerl, A. E. & Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55--67. [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Ivanescu, A. E., Staicu, A.-M., Scheipl, F. & Greven, S. (2015). Penalized function-on-function regression. *Computational Statistics*, 30(2), 539--568. [doi:10.1007/s00180-014-0548-4](https://doi.org/10.1007/s00180-014-0548-4)
- Preda, C. & Saporta, G. (2005). PLS regression on a stochastic process. *Computational Statistics & Data Analysis*, 48(1), 149--158. [doi:10.1016/j.csda.2003.10.003](https://doi.org/10.1016/j.csda.2003.10.003)
- Ramsay, J. O. & Silverman, B. W. (2005). *Functional Data Analysis*, 2nd ed. Springer. [doi:10.1007/b98888](https://doi.org/10.1007/b98888)
- Scheipl, F., Staicu, A.-M. & Greven, S. (2015). Functional additive mixed models. *Journal of Computational and Graphical Statistics*, 24(2), 477--501. [doi:10.1080/10618600.2014.901914](https://doi.org/10.1080/10618600.2014.901914)
- Srivastava, A. & Klassen, E. P. (2016). *Functional and Shape Data Analysis*. Springer. [doi:10.1007/978-1-4939-4020-2](https://doi.org/10.1007/978-1-4939-4020-2)
- Tucker, J. D., Wu, W. & Srivastava, A. (2013). Generative models for functional data using phase and amplitude separation. *Computational Statistics & Data Analysis*, 61, 50--66. [doi:10.1016/j.csda.2012.11.015](https://doi.org/10.1016/j.csda.2012.11.015)
- Tucker, J. D., Wu, W. & Srivastava, A. (2014). Analysis of proteomics data: Bayesian alignment of functions. *Electronic Journal of Statistics*, 8(2), 1769--1797. [doi:10.1214/14-EJS937](https://doi.org/10.1214/14-EJS937)
- Wood, S. N. (2006). *Generalized Additive Models: An Introduction with R*. Chapman & Hall/CRC. [doi:10.1201/9781420010404](https://doi.org/10.1201/9781420010404)
- Yao, F., Muller, H.-G. & Wang, J.-L. (2005b). Functional linear regression analysis for longitudinal data. *Annals of Statistics*, 33(6), 2873--2903. [doi:10.1214/009053605000000660](https://doi.org/10.1214/009053605000000660)
