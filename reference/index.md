# Package index

## Functional Data Objects

- [`fdata()`](https://sipemu.github.io/fdars/reference/fdata.md) :
  Create a functional data object
- [`fdata.cen()`](https://sipemu.github.io/fdars/reference/fdata.cen.md)
  : Center functional data
- [`deriv()`](https://sipemu.github.io/fdars/reference/deriv.md) :
  Compute functional derivative
- [`mean(`*`<fdata>`*`)`](https://sipemu.github.io/fdars/reference/mean.fdata.md)
  : Compute functional mean
- [`median()`](https://sipemu.github.io/fdars/reference/median.md) :
  Compute Functional Median
- [`trimmed()`](https://sipemu.github.io/fdars/reference/trimmed.md) :
  Compute Functional Trimmed Mean
- [`trimvar()`](https://sipemu.github.io/fdars/reference/trimvar.md) :
  Compute Functional Trimmed Variance
- [`var()`](https://sipemu.github.io/fdars/reference/var.md) :
  Functional Variance
- [`sd()`](https://sipemu.github.io/fdars/reference/sd.md) : Functional
  Standard Deviation
- [`normalize()`](https://sipemu.github.io/fdars/reference/normalize.md)
  : Normalize functional data
- [`standardize()`](https://sipemu.github.io/fdars/reference/standardize.md)
  : Standardize functional data (z-score normalization)
- [`scale_minmax()`](https://sipemu.github.io/fdars/reference/scale_minmax.md)
  : Min-Max scaling for functional data
- [`gmed()`](https://sipemu.github.io/fdars/reference/gmed.md) :
  Geometric Median of Functional Data
- [`inprod.fdata()`](https://sipemu.github.io/fdars/reference/inprod.fdata.md)
  : Inner Product of Functional Data
- [`int.simpson()`](https://sipemu.github.io/fdars/reference/int.simpson.md)
  : Utility Functions for Functional Data Analysis
- [`localavg.fdata()`](https://sipemu.github.io/fdars/reference/localavg.fdata.md)
  : Local Averages Feature Extraction
- [`fdata.bootstrap()`](https://sipemu.github.io/fdars/reference/fdata.bootstrap.md)
  : Bootstrap Functional Data
- [`fdata.bootstrap.ci()`](https://sipemu.github.io/fdars/reference/fdata.bootstrap.ci.md)
  : Bootstrap Confidence Intervals for Functional Statistics
- [`df_to_fdata2d()`](https://sipemu.github.io/fdars/reference/df_to_fdata2d.md)
  : Convert DataFrame to 2D functional data

## Basis Representation

- [`fdata2basis()`](https://sipemu.github.io/fdars/reference/fdata2basis.md)
  : Convert Functional Data to Basis Coefficients
- [`fdata2basis_2d()`](https://sipemu.github.io/fdars/reference/fdata2basis_2d.md)
  : Convert 2D Functional Data to Tensor Product Basis Coefficients
- [`fdata2basis_cv()`](https://sipemu.github.io/fdars/reference/fdata2basis_cv.md)
  : Cross-Validation for Basis Function Number Selection
- [`basis2fdata()`](https://sipemu.github.io/fdars/reference/basis2fdata.md)
  : Basis Representation Functions for Functional Data
- [`basis2fdata_2d()`](https://sipemu.github.io/fdars/reference/basis2fdata_2d.md)
  : Reconstruct 2D Functional Data from Tensor Product Basis
  Coefficients
- [`fdata2fd()`](https://sipemu.github.io/fdars/reference/fdata2fd.md) :
  Convert Functional Data to fd class
- [`fdata2pc()`](https://sipemu.github.io/fdars/reference/fdata2pc.md) :
  Convert Functional Data to Principal Component Scores
- [`fdata2pls()`](https://sipemu.github.io/fdars/reference/fdata2pls.md)
  : Convert Functional Data to PLS Scores
- [`basis.aic()`](https://sipemu.github.io/fdars/reference/basis.aic.md)
  : AIC for Basis Representation
- [`basis.bic()`](https://sipemu.github.io/fdars/reference/basis.bic.md)
  : BIC for Basis Representation
- [`basis.gcv()`](https://sipemu.github.io/fdars/reference/basis.gcv.md)
  : GCV Score for Basis Representation
- [`select.basis.auto()`](https://sipemu.github.io/fdars/reference/select.basis.auto.md)
  : Automatic Per-Curve Basis Type and Number Selection
- [`pspline()`](https://sipemu.github.io/fdars/reference/pspline.md) :
  P-spline Smoothing for Functional Data
- [`pspline.2d()`](https://sipemu.github.io/fdars/reference/pspline.2d.md)
  : P-spline Smoothing for 2D Functional Data

## Depth Functions

- [`depth()`](https://sipemu.github.io/fdars/reference/depth.md) : Depth
  Functions for Functional Data
- [`depth.BD()`](https://sipemu.github.io/fdars/reference/depth.BD.md) :
  Band Depth
- [`depth.FM()`](https://sipemu.github.io/fdars/reference/depth.FM.md) :
  Fraiman-Muniz Depth
- [`depth.FSD()`](https://sipemu.github.io/fdars/reference/depth.FSD.md)
  : Functional Spatial Depth
- [`depth.KFSD()`](https://sipemu.github.io/fdars/reference/depth.KFSD.md)
  : Kernel Functional Spatial Depth
- [`depth.MBD()`](https://sipemu.github.io/fdars/reference/depth.MBD.md)
  : Modified Band Depth
- [`depth.MEI()`](https://sipemu.github.io/fdars/reference/depth.MEI.md)
  : Modified Epigraph Index
- [`depth.mode()`](https://sipemu.github.io/fdars/reference/depth.mode.md)
  : Modal Depth
- [`depth.RP()`](https://sipemu.github.io/fdars/reference/depth.RP.md) :
  Random Projection Depth
- [`depth.RPD()`](https://sipemu.github.io/fdars/reference/depth.RPD.md)
  : Random Projection Depth with Derivatives
- [`depth.RT()`](https://sipemu.github.io/fdars/reference/depth.RT.md) :
  Random Tukey Depth

## Distance & Metrics

- [`metric()`](https://sipemu.github.io/fdars/reference/metric.md) :
  Distance Metrics for Functional Data
- [`metric.DTW()`](https://sipemu.github.io/fdars/reference/metric.DTW.md)
  : Dynamic Time Warping for Functional Data
- [`metric.hausdorff()`](https://sipemu.github.io/fdars/reference/metric.hausdorff.md)
  : Hausdorff Metric for Functional Data
- [`metric.kl()`](https://sipemu.github.io/fdars/reference/metric.kl.md)
  : Kullback-Leibler Divergence Metric for Functional Data
- [`metric.lp()`](https://sipemu.github.io/fdars/reference/metric.lp.md)
  : Lp Metric for Functional Data
- [`norm()`](https://sipemu.github.io/fdars/reference/norm.md) : Compute
  Lp Norm of Functional Data
- [`semimetric.basis()`](https://sipemu.github.io/fdars/reference/semimetric.basis.md)
  : Semi-metric based on Basis Expansion
- [`semimetric.deriv()`](https://sipemu.github.io/fdars/reference/semimetric.deriv.md)
  : Semi-metric based on Derivatives
- [`semimetric.fourier()`](https://sipemu.github.io/fdars/reference/semimetric.fourier.md)
  : Semi-metric based on Fourier Coefficients (FFT)
- [`semimetric.hshift()`](https://sipemu.github.io/fdars/reference/semimetric.hshift.md)
  : Semi-metric based on Horizontal Shift (Time Warping)
- [`semimetric.pca()`](https://sipemu.github.io/fdars/reference/semimetric.pca.md)
  : Semi-metric based on Principal Components
- [`group.distance()`](https://sipemu.github.io/fdars/reference/group.distance.md)
  : Compute Distance/Similarity Between Groups of Functional Data

## Clustering

- [`cluster.fcm()`](https://sipemu.github.io/fdars/reference/cluster.fcm.md)
  : Fuzzy C-Means Clustering for Functional Data
- [`cluster.init()`](https://sipemu.github.io/fdars/reference/cluster.init.md)
  : K-Means++ Center Initialization
- [`cluster.kmeans()`](https://sipemu.github.io/fdars/reference/cluster.kmeans.md)
  : Clustering Functions for Functional Data
- [`cluster.optim()`](https://sipemu.github.io/fdars/reference/cluster.optim.md)
  : Optimal Number of Clusters for Functional K-Means

## Outlier Detection

- [`outliergram()`](https://sipemu.github.io/fdars/reference/outliergram.md)
  : Outliergram for Functional Data
- [`outliers.boxplot()`](https://sipemu.github.io/fdars/reference/outliers.boxplot.md)
  : Outlier Detection using Functional Boxplot
- [`outliers.depth.pond()`](https://sipemu.github.io/fdars/reference/outliers.depth.pond.md)
  : Outlier Detection for Functional Data
- [`outliers.depth.trim()`](https://sipemu.github.io/fdars/reference/outliers.depth.trim.md)
  : Outlier Detection using Trimmed Depth
- [`outliers.lrt()`](https://sipemu.github.io/fdars/reference/outliers.lrt.md)
  : LRT-based Outlier Detection for Functional Data
- [`outliers.thres.lrt()`](https://sipemu.github.io/fdars/reference/outliers.thres.lrt.md)
  : LRT Outlier Detection Threshold
- [`magnitudeshape()`](https://sipemu.github.io/fdars/reference/magnitudeshape.md)
  : Magnitude-Shape Outlier Detection for Functional Data

## Regression

- [`fregre.basis()`](https://sipemu.github.io/fdars/reference/fregre.basis.md)
  : Functional Basis Regression
- [`fregre.basis.cv()`](https://sipemu.github.io/fdars/reference/fregre.basis.cv.md)
  : Cross-Validation for Functional Basis Regression
- [`fregre.np()`](https://sipemu.github.io/fdars/reference/fregre.np.md)
  : Nonparametric Functional Regression
- [`fregre.np.cv()`](https://sipemu.github.io/fdars/reference/fregre.np.cv.md)
  : Cross-Validation for Nonparametric Functional Regression
- [`fregre.np.multi()`](https://sipemu.github.io/fdars/reference/fregre.np.multi.md)
  : Nonparametric Regression with Multiple Functional Predictors
- [`fregre.pc()`](https://sipemu.github.io/fdars/reference/fregre.pc.md)
  : Functional Regression
- [`fregre.pc.cv()`](https://sipemu.github.io/fdars/reference/fregre.pc.cv.md)
  : Cross-Validation for Functional PC Regression
- [`optim.np()`](https://sipemu.github.io/fdars/reference/optim.np.md) :
  Optimize Bandwidth Using Cross-Validation
- [`flm.test()`](https://sipemu.github.io/fdars/reference/flm.test.md) :
  Statistical Tests for Functional Data
- [`pred.MAE()`](https://sipemu.github.io/fdars/reference/pred.MAE.md) :
  Mean Absolute Error
- [`pred.MSE()`](https://sipemu.github.io/fdars/reference/pred.MSE.md) :
  Mean Squared Error
- [`pred.R2()`](https://sipemu.github.io/fdars/reference/pred.R2.md) :
  R-Squared (Coefficient of Determination)
- [`pred.RMSE()`](https://sipemu.github.io/fdars/reference/pred.RMSE.md)
  : Root Mean Squared Error

## Seasonal Analysis

- [`estimate.period()`](https://sipemu.github.io/fdars/reference/estimate.period.md)
  : Estimate Seasonal Period using FFT
- [`detect.period()`](https://sipemu.github.io/fdars/reference/detect.period.md)
  : Seasonal Analysis Functions for Functional Data
- [`detect.periods()`](https://sipemu.github.io/fdars/reference/detect.periods.md)
  : Detect Multiple Concurrent Periods
- [`detect.peaks()`](https://sipemu.github.io/fdars/reference/detect.peaks.md)
  : Detect Peaks in Functional Data
- [`autoperiod()`](https://sipemu.github.io/fdars/reference/autoperiod.md)
  : Autoperiod: Hybrid FFT + ACF Period Detection
- [`cfd.autoperiod()`](https://sipemu.github.io/fdars/reference/cfd.autoperiod.md)
  : CFDAutoperiod: Clustered Filtered Detrended Autoperiod
- [`sazed()`](https://sipemu.github.io/fdars/reference/sazed.md) :
  SAZED: Spectral-ACF Zero-crossing Ensemble Detection
- [`lomb.scargle()`](https://sipemu.github.io/fdars/reference/lomb.scargle.md)
  : Lomb-Scargle Periodogram
- [`matrix.profile()`](https://sipemu.github.io/fdars/reference/matrix.profile.md)
  : Matrix Profile for Motif Discovery and Period Detection
- [`stl.fd()`](https://sipemu.github.io/fdars/reference/stl.fd.md) : STL
  Decomposition: Seasonal and Trend decomposition using LOESS
- [`ssa.fd()`](https://sipemu.github.io/fdars/reference/ssa.fd.md) :
  Singular Spectrum Analysis (SSA) for Time Series Decomposition
- [`seasonal.strength()`](https://sipemu.github.io/fdars/reference/seasonal.strength.md)
  : Measure Seasonal Strength
- [`seasonal.strength.curve()`](https://sipemu.github.io/fdars/reference/seasonal.strength.curve.md)
  : Time-Varying Seasonal Strength
- [`detect.seasonality.changes()`](https://sipemu.github.io/fdars/reference/detect.seasonality.changes.md)
  : Detect Changes in Seasonality
- [`detect.seasonality.changes.auto()`](https://sipemu.github.io/fdars/reference/detect.seasonality.changes.auto.md)
  : Detect Seasonality Changes with Automatic Threshold
- [`detect_amplitude_modulation()`](https://sipemu.github.io/fdars/reference/detect_amplitude_modulation.md)
  : Detect Amplitude Modulation in Seasonal Time Series
- [`instantaneous.period()`](https://sipemu.github.io/fdars/reference/instantaneous.period.md)
  : Estimate Instantaneous Period
- [`analyze.peak.timing()`](https://sipemu.github.io/fdars/reference/analyze.peak.timing.md)
  : Analyze Peak Timing Variability
- [`classify.seasonality()`](https://sipemu.github.io/fdars/reference/classify.seasonality.md)
  : Classify Seasonality Type
- [`detrend()`](https://sipemu.github.io/fdars/reference/detrend.md) :
  Remove Trend from Functional Data
- [`decompose()`](https://sipemu.github.io/fdars/reference/decompose.md)
  : Seasonal-Trend Decomposition

## Smoothing

- [`S.KNN()`](https://sipemu.github.io/fdars/reference/S.KNN.md) :
  K-Nearest Neighbors Smoother Matrix
- [`S.LCR()`](https://sipemu.github.io/fdars/reference/S.LCR.md) : Local
  Cubic Regression Smoother Matrix
- [`S.LLR()`](https://sipemu.github.io/fdars/reference/S.LLR.md) : Local
  Linear Regression Smoother Matrix
- [`S.LPR()`](https://sipemu.github.io/fdars/reference/S.LPR.md) : Local
  Polynomial Regression Smoother Matrix
- [`S.NW()`](https://sipemu.github.io/fdars/reference/S.NW.md) :
  Smoothing Functions for Functional Data
- [`CV.S()`](https://sipemu.github.io/fdars/reference/CV.S.md) :
  Cross-Validation for Smoother Selection
- [`GCV.S()`](https://sipemu.github.io/fdars/reference/GCV.S.md) :
  Generalized Cross-Validation for Smoother Selection
- [`h.default()`](https://sipemu.github.io/fdars/reference/h.default.md)
  : Default Bandwidth
- [`register.fd()`](https://sipemu.github.io/fdars/reference/register.fd.md)
  : Curve Registration (Alignment)

## Kernels (Smoothing)

- [`Kernel()`](https://sipemu.github.io/fdars/reference/Kernel.md) :
  Unified Symmetric Kernel Interface
- [`Kernel.asymmetric()`](https://sipemu.github.io/fdars/reference/Kernel.asymmetric.md)
  : Unified Asymmetric Kernel Interface
- [`Kernel.integrate()`](https://sipemu.github.io/fdars/reference/Kernel.integrate.md)
  : Unified Integrated Kernel Interface
- [`Ker.cos()`](https://sipemu.github.io/fdars/reference/Ker.cos.md) :
  Cosine Kernel
- [`Ker.epa()`](https://sipemu.github.io/fdars/reference/Ker.epa.md) :
  Epanechnikov Kernel
- [`Ker.norm()`](https://sipemu.github.io/fdars/reference/Ker.norm.md) :
  Kernel Functions
- [`Ker.quar()`](https://sipemu.github.io/fdars/reference/Ker.quar.md) :
  Quartic (Biweight) Kernel
- [`Ker.tri()`](https://sipemu.github.io/fdars/reference/Ker.tri.md) :
  Triweight Kernel
- [`Ker.unif()`](https://sipemu.github.io/fdars/reference/Ker.unif.md) :
  Uniform (Rectangular) Kernel
- [`AKer.cos()`](https://sipemu.github.io/fdars/reference/AKer.cos.md) :
  Asymmetric Cosine Kernel
- [`AKer.epa()`](https://sipemu.github.io/fdars/reference/AKer.epa.md) :
  Asymmetric Epanechnikov Kernel
- [`AKer.norm()`](https://sipemu.github.io/fdars/reference/AKer.norm.md)
  : Asymmetric Normal Kernel
- [`AKer.quar()`](https://sipemu.github.io/fdars/reference/AKer.quar.md)
  : Asymmetric Quartic Kernel
- [`AKer.tri()`](https://sipemu.github.io/fdars/reference/AKer.tri.md) :
  Asymmetric Triweight Kernel
- [`AKer.unif()`](https://sipemu.github.io/fdars/reference/AKer.unif.md)
  : Asymmetric Uniform Kernel
- [`IKer.cos()`](https://sipemu.github.io/fdars/reference/IKer.cos.md) :
  Integrated Cosine Kernel
- [`IKer.epa()`](https://sipemu.github.io/fdars/reference/IKer.epa.md) :
  Integrated Epanechnikov Kernel
- [`IKer.norm()`](https://sipemu.github.io/fdars/reference/IKer.norm.md)
  : Integrated Normal Kernel
- [`IKer.quar()`](https://sipemu.github.io/fdars/reference/IKer.quar.md)
  : Integrated Quartic Kernel
- [`IKer.tri()`](https://sipemu.github.io/fdars/reference/IKer.tri.md) :
  Integrated Triweight Kernel
- [`IKer.unif()`](https://sipemu.github.io/fdars/reference/IKer.unif.md)
  : Integrated Uniform Kernel

## Covariance Functions (GP)

- [`kernel.add()`](https://sipemu.github.io/fdars/reference/kernel.add.md)
  : Add Covariance Functions
- [`kernel.brownian()`](https://sipemu.github.io/fdars/reference/kernel.brownian.md)
  : Brownian Motion Covariance Function
- [`kernel.exponential()`](https://sipemu.github.io/fdars/reference/kernel.exponential.md)
  : Exponential Covariance Function
- [`kernel.gaussian()`](https://sipemu.github.io/fdars/reference/kernel.gaussian.md)
  : Gaussian (Squared Exponential) Covariance Function
- [`kernel.linear()`](https://sipemu.github.io/fdars/reference/kernel.linear.md)
  : Linear Covariance Function
- [`kernel.matern()`](https://sipemu.github.io/fdars/reference/kernel.matern.md)
  : Matern Covariance Function
- [`kernel.mult()`](https://sipemu.github.io/fdars/reference/kernel.mult.md)
  : Multiply Covariance Functions
- [`kernel.periodic()`](https://sipemu.github.io/fdars/reference/kernel.periodic.md)
  : Periodic Covariance Function
- [`kernel.polynomial()`](https://sipemu.github.io/fdars/reference/kernel.polynomial.md)
  : Polynomial Covariance Function
- [`kernel.whitenoise()`](https://sipemu.github.io/fdars/reference/kernel.whitenoise.md)
  : White Noise Covariance Function
- [`make.gaussian.process()`](https://sipemu.github.io/fdars/reference/make.gaussian.process.md)
  : Generate Gaussian Process Samples
- [`cov()`](https://sipemu.github.io/fdars/reference/cov.md) :
  Functional Covariance Function

## Simulation

- [`eFun()`](https://sipemu.github.io/fdars/reference/eFun.md) :
  Generate Eigenfunction Basis
- [`eVal()`](https://sipemu.github.io/fdars/reference/eVal.md) :
  Generate Eigenvalue Sequence
- [`simFunData()`](https://sipemu.github.io/fdars/reference/simFunData.md)
  : Simulate Functional Data via Karhunen-Loeve Expansion
- [`simMultiFunData()`](https://sipemu.github.io/fdars/reference/simMultiFunData.md)
  : Simulate Multivariate Functional Data
- [`addError()`](https://sipemu.github.io/fdars/reference/addError.md) :
  Add Measurement Error to Functional Data

## Irregular Functional Data

- [`irregFdata()`](https://sipemu.github.io/fdars/reference/irregFdata.md)
  : Create an Irregular Functional Data Object
- [`is.irregular()`](https://sipemu.github.io/fdars/reference/is.irregular.md)
  : Check if an Object is Irregular Functional Data
- [`sparsify()`](https://sipemu.github.io/fdars/reference/sparsify.md) :
  Convert Regular Functional Data to Irregular by Subsampling
- [`as.fdata()`](https://sipemu.github.io/fdars/reference/as.fdata.irregFdata.md)
  : Convert Irregular Functional Data to Regular Grid
- [`mean(`*`<irregFdata>`*`)`](https://sipemu.github.io/fdars/reference/mean.irregFdata.md)
  : Estimate Mean Function for Irregular Data

## Random Processes

- [`r.bridge()`](https://sipemu.github.io/fdars/reference/r.bridge.md) :
  Generate Brownian Bridge
- [`r.brownian()`](https://sipemu.github.io/fdars/reference/r.brownian.md)
  : Generate Brownian Motion
- [`r.ou()`](https://sipemu.github.io/fdars/reference/r.ou.md) :
  Generate Ornstein-Uhlenbeck Process

## Statistical Tests

- [`fmean.test.fdata()`](https://sipemu.github.io/fdars/reference/fmean.test.fdata.md)
  : Test for Equality of Functional Means
- [`group.test()`](https://sipemu.github.io/fdars/reference/group.test.md)
  : Permutation Test for Group Differences

## Plotting

- [`autoplot(`*`<fdata>`*`)`](https://sipemu.github.io/fdars/reference/autoplot.fdata.md)
  : Create a ggplot for fdata objects
- [`plot(`*`<fdata>`*`)`](https://sipemu.github.io/fdars/reference/plot.fdata.md)
  : Plot method for fdata objects
- [`boxplot(`*`<fdata>`*`)`](https://sipemu.github.io/fdars/reference/boxplot.fdata.md)
  : Functional Boxplot
- [`plot(`*`<fdata2pc>`*`)`](https://sipemu.github.io/fdars/reference/plot.fdata2pc.md)
  : Plot FPCA Results
- [`plot(`*`<basis.auto>`*`)`](https://sipemu.github.io/fdars/reference/plot.basis.auto.md)
  : Plot method for basis.auto objects
- [`plot(`*`<basis.cv>`*`)`](https://sipemu.github.io/fdars/reference/plot.basis.cv.md)
  : Plot method for basis.cv objects
- [`plot(`*`<cluster.fcm>`*`)`](https://sipemu.github.io/fdars/reference/plot.cluster.fcm.md)
  : Plot Method for cluster.fcm Objects
- [`plot(`*`<cluster.kmeans>`*`)`](https://sipemu.github.io/fdars/reference/plot.cluster.kmeans.md)
  : Plot Method for cluster.kmeans Objects
- [`plot(`*`<cluster.optim>`*`)`](https://sipemu.github.io/fdars/reference/plot.cluster.optim.md)
  : Plot Method for cluster.optim Objects
- [`plot(`*`<group.distance>`*`)`](https://sipemu.github.io/fdars/reference/plot.group.distance.md)
  : Plot method for group.distance
- [`plot(`*`<outliergram>`*`)`](https://sipemu.github.io/fdars/reference/plot.outliergram.md)
  : Plot Method for Outliergram Objects
- [`plot(`*`<outliers.fdata>`*`)`](https://sipemu.github.io/fdars/reference/plot.outliers.fdata.md)
  : Plot method for outliers.fdata objects
- [`plot(`*`<pspline>`*`)`](https://sipemu.github.io/fdars/reference/plot.pspline.md)
  : Plot method for pspline objects
- [`plot(`*`<pspline.2d>`*`)`](https://sipemu.github.io/fdars/reference/plot.pspline.2d.md)
  : Plot method for pspline.2d objects
- [`plot(`*`<register.fd>`*`)`](https://sipemu.github.io/fdars/reference/plot.register.fd.md)
  : Plot Method for register.fd Objects
- [`plot(`*`<magnitudeshape>`*`)`](https://sipemu.github.io/fdars/reference/plot.magnitudeshape.md)
  : Plot Method for magnitudeshape Objects

## Prediction

- [`predict(`*`<fregre.fd>`*`)`](https://sipemu.github.io/fdars/reference/predict.fregre.fd.md)
  : Predict Method for Functional Regression (fregre.fd)
- [`predict(`*`<fregre.np>`*`)`](https://sipemu.github.io/fdars/reference/predict.fregre.np.md)
  : Predict Method for Nonparametric Functional Regression (fregre.np)
- [`predict(`*`<fregre.np.multi>`*`)`](https://sipemu.github.io/fdars/reference/predict.fregre.np.multi.md)
  : Predict method for fregre.np.multi

## Print & Summary

- [`print(`*`<fdata>`*`)`](https://sipemu.github.io/fdars/reference/print.fdata.md)
  : Print method for fdata objects
- [`print(`*`<fdata2pc>`*`)`](https://sipemu.github.io/fdars/reference/print.fdata2pc.md)
  : Print Method for FPCA Results
- [`print(`*`<fdata.bootstrap.ci>`*`)`](https://sipemu.github.io/fdars/reference/print.fdata.bootstrap.ci.md)
  : Print method for bootstrap CI
- [`print(`*`<basis.auto>`*`)`](https://sipemu.github.io/fdars/reference/print.basis.auto.md)
  : Print method for basis.auto objects
- [`print(`*`<basis.cv>`*`)`](https://sipemu.github.io/fdars/reference/print.basis.cv.md)
  : Print method for basis.cv objects
- [`print(`*`<cluster.fcm>`*`)`](https://sipemu.github.io/fdars/reference/print.cluster.fcm.md)
  : Print Method for cluster.fcm Objects
- [`print(`*`<cluster.kmeans>`*`)`](https://sipemu.github.io/fdars/reference/print.cluster.kmeans.md)
  : Print Method for cluster.kmeans Objects
- [`print(`*`<cluster.optim>`*`)`](https://sipemu.github.io/fdars/reference/print.cluster.optim.md)
  : Print Method for cluster.optim Objects
- [`print(`*`<fbplot>`*`)`](https://sipemu.github.io/fdars/reference/print.fbplot.md)
  : Print Method for fbplot Objects
- [`print(`*`<fregre.fd>`*`)`](https://sipemu.github.io/fdars/reference/print.fregre.fd.md)
  : Print method for fregre objects
- [`print(`*`<fregre.np>`*`)`](https://sipemu.github.io/fdars/reference/print.fregre.np.md)
  : Print method for fregre.np objects
- [`print(`*`<fregre.np.multi>`*`)`](https://sipemu.github.io/fdars/reference/print.fregre.np.multi.md)
  : Print method for fregre.np.multi
- [`print(`*`<group.distance>`*`)`](https://sipemu.github.io/fdars/reference/print.group.distance.md)
  : Print method for group.distance
- [`print(`*`<group.test>`*`)`](https://sipemu.github.io/fdars/reference/print.group.test.md)
  : Print method for group.test
- [`print(`*`<kernel>`*`)`](https://sipemu.github.io/fdars/reference/print.kernel.md)
  : Print Method for Covariance Functions
- [`print(`*`<magnitudeshape>`*`)`](https://sipemu.github.io/fdars/reference/print.magnitudeshape.md)
  : Print Method for magnitudeshape Objects
- [`print(`*`<outliergram>`*`)`](https://sipemu.github.io/fdars/reference/print.outliergram.md)
  : Print Method for Outliergram Objects
- [`print(`*`<outliers.fdata>`*`)`](https://sipemu.github.io/fdars/reference/print.outliers.fdata.md)
  : Print method for outliers.fdata objects
- [`print(`*`<pspline>`*`)`](https://sipemu.github.io/fdars/reference/print.pspline.md)
  : Print method for pspline objects
- [`print(`*`<pspline.2d>`*`)`](https://sipemu.github.io/fdars/reference/print.pspline.2d.md)
  : Print method for pspline.2d objects
- [`print(`*`<register.fd>`*`)`](https://sipemu.github.io/fdars/reference/print.register.fd.md)
  : Print Method for register.fd Objects
- [`summary(`*`<basis.auto>`*`)`](https://sipemu.github.io/fdars/reference/summary.basis.auto.md)
  : Summary method for basis.auto objects
- [`summary(`*`<fdata>`*`)`](https://sipemu.github.io/fdars/reference/summary.fdata.md)
  : Summary method for fdata objects

## Other

- [`` `[`( ``*`<fdata>`*`)`](https://sipemu.github.io/fdars/reference/sub-.fdata.md)
  : Subset method for fdata objects
- [`kernels`](https://sipemu.github.io/fdars/reference/kernels.md) :
  Covariance Kernel Functions for Gaussian Processes
- [`fdars`](https://sipemu.github.io/fdars/reference/fdars-package.md)
  [`fdars-package`](https://sipemu.github.io/fdars/reference/fdars-package.md)
  : fdars: Functional Data Analysis in Rust
