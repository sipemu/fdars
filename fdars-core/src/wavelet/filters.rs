//! Orthonormal Daubechies filter-coefficient tables and family → filter-bank lookup.
//!
//! Each Daubechies family `dbN` (N = number of vanishing moments) is defined by a
//! single hardcoded analysis-lowpass filter `dec_lo` (length `2N`) in the standard
//! orthonormal-DWT normalization (the `dec_lo` digits match PyWavelets to full f64
//! precision). The other three filters are derived programmatically from `dec_lo` —
//! a single source of truth — so a sign/reversal mistake cannot arise from re-typing
//! tables:
//!
//! ```text
//! dec_hi[k] = (-1)^k * dec_lo[L-1-k]     (quadrature mirror; L = filter length)
//! rec_lo    = reverse(dec_lo)            (synthesis lowpass  = time-reverse of analysis)
//! rec_hi    = reverse(dec_hi)            (synthesis highpass = time-reverse of analysis)
//! ```
//!
//! **Sign convention.** The high-pass derivation above is the Mallat (1989)
//! orthonormal-DWT convention (`dec_hi[k] = (-1)^k * dec_lo[L-1-k]`, positive sign at
//! `k=0`), and synthesis is the time-reverse of analysis. This differs from
//! PyWavelets, which uses `(-1)^(k+1)` and therefore emits a **highpass of opposite
//! sign** in `dec_hi`/`rec_hi`. The two conventions are equally valid: the analysis
//! and synthesis filters here form a matched adjoint pair, so the DWT is perfectly
//! reconstructing and internally consistent regardless of the overall highpass sign.
//! Downstream consumers of `dec_hi`/`rec_hi` that compare against a PyWavelets
//! reference must account for this sign flip.
//!
//! These relations make analysis and synthesis an exact adjoint pair (perfect
//! reconstruction), which the round-trip tests in [`crate::wavelet`] verify to
//! 1e-10 relative. The normalization invariants (`dec_lo` sums to `√2`, has unit
//! L2 norm) are asserted here to 1e-12 — the correctness gate for the hardcoded
//! digits.

use crate::error::FdarError;
use crate::wavelet::WaveletFamily;

/// One orthonormal wavelet filter bank: analysis + synthesis low/high-pass filters.
///
/// All four filters share the same length `L = 2N` for `dbN` (`L = 2` for Haar).
/// The synthesis filters are the time-reverse of the analysis filters, and the
/// high-pass filters are the quadrature mirror of the corresponding low-pass
/// filters — see the module docs for the exact derivation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FilterBank {
    /// Analysis low-pass filter (`dec_lo`), length `L`. The hardcoded source of truth.
    pub dec_lo: Vec<f64>,
    /// Analysis high-pass filter (`dec_hi`), length `L`. Quadrature mirror of `dec_lo`.
    pub dec_hi: Vec<f64>,
    /// Synthesis low-pass filter (`rec_lo`), length `L`. Time-reverse of `dec_lo`.
    ///
    /// Deliverable filter-bank API (verified by the filter-invariant tests); the
    /// even-length core reconstructs via the analysis transpose, so this field is not
    /// read outside tests yet — awaiting Phase 70 consumers.
    #[allow(dead_code)]
    pub rec_lo: Vec<f64>,
    /// Synthesis high-pass filter (`rec_hi`), length `L`. Time-reverse of `dec_hi`.
    ///
    /// Deliverable filter-bank API (verified by the filter-invariant tests); the
    /// even-length core reconstructs via the analysis transpose, so this field is not
    /// read outside tests yet — awaiting Phase 70 consumers.
    #[allow(dead_code)]
    pub rec_hi: Vec<f64>,
}

impl FilterBank {
    /// The filter length `L` (`= 2N` for `dbN`, `2` for Haar/db1).
    pub(crate) fn filter_len(&self) -> usize {
        self.dec_lo.len()
    }

    /// Build the full bank from an analysis-lowpass filter, deriving the other three.
    ///
    /// `dec_hi[k] = (-1)^k * dec_lo[L-1-k]`, `rec_lo = reverse(dec_lo)`,
    /// `rec_hi = reverse(dec_hi)`.
    fn from_dec_lo(dec_lo: Vec<f64>) -> Self {
        let l = dec_lo.len();
        let dec_hi: Vec<f64> = (0..l)
            .map(|k| {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                sign * dec_lo[l - 1 - k]
            })
            .collect();
        let rec_lo: Vec<f64> = dec_lo.iter().rev().copied().collect();
        let rec_hi: Vec<f64> = dec_hi.iter().rev().copied().collect();
        Self {
            dec_lo,
            dec_hi,
            rec_lo,
            rec_hi,
        }
    }
}

/// Analysis-lowpass (`dec_lo`) coefficient table for the Haar / db1 wavelet.
///
/// `[1/√2, 1/√2]` — the orthonormal-DWT normalization (`SQRT_1_2 = 1/√2`).
const HAAR_DEC_LO: [f64; 2] = [
    std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
];

/// Daubechies analysis-lowpass (`dec_lo`) tables db2..db10, to full f64 precision
/// (PyWavelets orthonormal-DWT convention). `dbN` has length `2N`.
///
/// Each table sums to `√2` and has unit L2 norm (verified to 1e-12 in tests). The
/// remaining three filters per family are derived from these via [`FilterBank::from_dec_lo`].
const DB2_DEC_LO: [f64; 4] = [
    0.482_962_913_144_534_27,
    0.836_516_303_737_808,
    0.224_143_868_042_013_33,
    -0.129_409_522_551_260_45,
];

const DB3_DEC_LO: [f64; 6] = [
    0.332_670_552_950_082_6,
    0.806_891_509_311_092_4,
    0.459_877_502_118_491_5,
    -0.135_011_020_010_254_6,
    -0.085_441_273_882_026_62,
    0.035_226_291_885_709_554,
];

const DB4_DEC_LO: [f64; 8] = [
    0.230_377_813_308_896_45,
    0.714_846_570_552_915_6,
    0.630_880_767_929_859,
    -0.027_983_769_416_859_59,
    -0.187_034_811_719_093_09,
    0.030_841_381_835_560_63,
    0.032_883_011_666_885_17,
    -0.010_597_401_785_069_018,
];

const DB5_DEC_LO: [f64; 10] = [
    0.160_102_397_974_192_93,
    0.603_829_269_797_189_6,
    0.724_308_528_437_773,
    0.138_428_145_901_320_88,
    -0.242_294_887_066_382,
    -0.032_244_869_584_638_47,
    0.077_571_493_840_045_7,
    -0.006_241_490_212_798_271,
    -0.012_580_751_999_081_994,
    0.003_335_725_285_473_771,
];

const DB6_DEC_LO: [f64; 12] = [
    0.111_540_743_350_109_4,
    0.494_623_890_398_452_9,
    0.751_133_908_021_094_9,
    0.315_250_351_709_198,
    -0.226_264_693_965_439_3,
    -0.129_766_867_567_261_88,
    0.097_501_605_587_323_06,
    0.027_522_865_530_305_606,
    -0.031_582_039_317_485_98,
    0.000_553_842_201_161_500_1,
    0.004_777_257_510_945_505_6,
    -0.001_077_301_085_308_479_4,
];

const DB7_DEC_LO: [f64; 14] = [
    0.077_852_054_085_009_23,
    0.396_539_319_481_917_5,
    0.729_132_090_846_235_4,
    0.469_782_287_405_192_96,
    -0.143_906_003_928_565_23,
    -0.224_036_184_993_875_15,
    0.071_309_219_266_830_47,
    0.080_612_609_151_083_04,
    -0.038_029_936_935_014_41,
    -0.016_574_541_630_666_902,
    0.012_550_998_556_099_856,
    0.000_429_577_972_921_366_84,
    -0.001_801_640_704_047_493_5,
    0.000_353_713_799_974_520_6,
];

const DB8_DEC_LO: [f64; 16] = [
    0.054_415_842_243_103_95,
    0.312_871_590_914_299_6,
    0.675_630_736_297_289_2,
    0.585_354_683_654_206_5,
    -0.015_829_105_256_348_515,
    -0.284_015_542_961_546_4,
    0.000_472_484_573_913_080_16,
    0.128_747_426_620_478_72,
    -0.017_369_301_001_807_85,
    -0.044_088_253_930_794_59,
    0.013_981_027_917_398_262,
    0.008_746_094_047_405_749,
    -0.004_870_352_993_451_561,
    -0.000_391_740_373_376_947_4,
    0.000_675_449_406_450_568_5,
    -0.000_117_476_784_124_769_35,
];

const DB9_DEC_LO: [f64; 18] = [
    0.038_077_947_363_878_34,
    0.243_834_674_612_590_23,
    0.604_823_123_690_111_2,
    0.657_288_078_051_3,
    0.133_197_385_825_007_56,
    -0.293_273_783_279_174_3,
    -0.096_840_783_222_976,
    0.148_540_749_338_105_88,
    0.030_725_681_479_333_88,
    -0.067_632_829_061_330_72,
    0.000_250_947_114_831_909,
    0.022_361_662_123_678_97,
    -0.004_723_204_757_751_389,
    -0.004_281_503_682_463_433,
    0.001_847_646_883_056_228_5,
    0.000_230_385_763_523_196_16,
    -0.000_251_963_188_942_710_45,
    0.000_039_347_320_316_271_636,
];

const DB10_DEC_LO: [f64; 20] = [
    0.026_670_057_900_555_55,
    0.188_176_800_077_691_53,
    0.527_201_188_931_725_7,
    0.688_459_039_453_603_4,
    0.281_172_343_660_578_47,
    -0.249_846_424_327_315_13,
    -0.195_946_274_377_378_13,
    0.127_369_340_335_795_47,
    0.093_057_364_603_568_71,
    -0.071_394_147_166_394_58,
    -0.029_457_536_821_876_86,
    0.033_212_674_059_341_37,
    0.003_606_553_566_956_005_8,
    -0.010_733_175_483_330_54,
    0.001_395_351_747_052_900_2,
    0.001_992_405_295_185_056_6,
    -0.000_685_856_694_959_711_8,
    -0.000_116_466_855_129_285_54,
    0.000_093_588_670_320_069_6,
    -0.000_013_264_202_894_521_243,
];

/// Return the orthonormal filter bank for `family` (hardcoded coefficients).
///
/// # Errors
/// Returns [`FdarError::InvalidParameter`] for a Daubechies order outside `2..=10`
/// (db1 is spelled [`WaveletFamily::Haar`]).
pub(crate) fn filter_bank(family: &WaveletFamily) -> Result<FilterBank, FdarError> {
    let dec_lo: Vec<f64> = match family {
        WaveletFamily::Haar => HAAR_DEC_LO.to_vec(),
        WaveletFamily::Daubechies(2) => DB2_DEC_LO.to_vec(),
        WaveletFamily::Daubechies(3) => DB3_DEC_LO.to_vec(),
        WaveletFamily::Daubechies(4) => DB4_DEC_LO.to_vec(),
        WaveletFamily::Daubechies(5) => DB5_DEC_LO.to_vec(),
        WaveletFamily::Daubechies(6) => DB6_DEC_LO.to_vec(),
        WaveletFamily::Daubechies(7) => DB7_DEC_LO.to_vec(),
        WaveletFamily::Daubechies(8) => DB8_DEC_LO.to_vec(),
        WaveletFamily::Daubechies(9) => DB9_DEC_LO.to_vec(),
        WaveletFamily::Daubechies(10) => DB10_DEC_LO.to_vec(),
        WaveletFamily::Daubechies(n) => {
            return Err(FdarError::InvalidParameter {
                parameter: "family",
                message: format!(
                    "Daubechies order {n} out of range: supported orders are 2..=10 (order 1 is Haar)"
                ),
            });
        }
    };
    Ok(FilterBank::from_dec_lo(dec_lo))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::SQRT_2;

    /// Every in-scope family, for iterating invariant checks.
    fn all_families() -> Vec<WaveletFamily> {
        let mut v = vec![WaveletFamily::Haar];
        for n in 2..=10 {
            v.push(WaveletFamily::Daubechies(n));
        }
        v
    }

    #[test]
    fn dec_lo_sums_to_sqrt2() {
        for fam in all_families() {
            let fb = filter_bank(&fam).unwrap();
            let sum: f64 = fb.dec_lo.iter().sum();
            assert!(
                (sum - SQRT_2).abs() < 1e-12,
                "{fam:?}: dec_lo sums to {sum}, expected √2"
            );
        }
    }

    #[test]
    fn dec_lo_has_unit_l2_norm() {
        for fam in all_families() {
            let fb = filter_bank(&fam).unwrap();
            let ss: f64 = fb.dec_lo.iter().map(|c| c * c).sum();
            assert!(
                (ss - 1.0).abs() < 1e-12,
                "{fam:?}: dec_lo sum-of-squares is {ss}, expected 1"
            );
        }
    }

    #[test]
    fn dec_hi_is_quadrature_mirror_of_dec_lo() {
        for fam in all_families() {
            let fb = filter_bank(&fam).unwrap();
            let l = fb.filter_len();
            for k in 0..l {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                let expected = sign * fb.dec_lo[l - 1 - k];
                assert!(
                    (fb.dec_hi[k] - expected).abs() < 1e-12,
                    "{fam:?}: dec_hi[{k}] mismatch"
                );
            }
            // Orthogonality: dec_lo · dec_hi == 0.
            let dot: f64 = fb.dec_lo.iter().zip(&fb.dec_hi).map(|(a, b)| a * b).sum();
            assert!(
                (dot).abs() < 1e-12,
                "{fam:?}: dec_lo · dec_hi = {dot}, expected 0"
            );
        }
    }

    #[test]
    fn synthesis_filters_are_time_reverse_of_analysis() {
        for fam in all_families() {
            let fb = filter_bank(&fam).unwrap();
            let l = fb.filter_len();
            for k in 0..l {
                assert!(
                    (fb.rec_lo[k] - fb.dec_lo[l - 1 - k]).abs() < 1e-12,
                    "{fam:?}: rec_lo[{k}] not time-reverse of dec_lo"
                );
                assert!(
                    (fb.rec_hi[k] - fb.dec_hi[l - 1 - k]).abs() < 1e-12,
                    "{fam:?}: rec_hi[{k}] not time-reverse of dec_hi"
                );
            }
        }
    }

    #[test]
    fn filter_len_is_twice_the_order() {
        assert_eq!(filter_bank(&WaveletFamily::Haar).unwrap().filter_len(), 2);
        for n in 2..=10 {
            let fb = filter_bank(&WaveletFamily::Daubechies(n)).unwrap();
            assert_eq!(fb.filter_len(), 2 * n, "db{n} filter length");
        }
    }

    #[test]
    fn filter_bank_rejects_out_of_range_order() {
        assert!(matches!(
            filter_bank(&WaveletFamily::Daubechies(11)),
            Err(FdarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            filter_bank(&WaveletFamily::Daubechies(1)),
            Err(FdarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            filter_bank(&WaveletFamily::Daubechies(0)),
            Err(FdarError::InvalidParameter { .. })
        ));
    }
}
