//! Column-major matrix type for functional data analysis.
//!
//! [`FdMatrix`] provides safe, dimension-tracked access to the flat column-major
//! data layout used throughout this crate. It eliminates manual `data[i + j * n]`
//! index arithmetic and carries dimensions alongside the data.

use nalgebra::DMatrix;

/// Column-major matrix for functional data.
///
/// Stores data in a flat `Vec<f64>` with column-major (Fortran) layout:
/// element `(row, col)` is at index `row + col * nrows`.
///
/// # Conventions
///
/// For functional data, rows typically represent observations and columns
/// represent evaluation points. For 2D surfaces with `m1 x m2` grids,
/// the surface is flattened into `m1 * m2` columns.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
///
/// // 3 observations, 4 evaluation points
/// let data = vec![
///     1.0, 2.0, 3.0,  // column 0 (all obs at point 0)
///     4.0, 5.0, 6.0,  // column 1
///     7.0, 8.0, 9.0,  // column 2
///     10.0, 11.0, 12.0, // column 3
/// ];
/// let mat = FdMatrix::from_column_major(data, 3, 4).unwrap();
///
/// assert_eq!(mat[(0, 0)], 1.0);  // obs 0 at point 0
/// assert_eq!(mat[(1, 2)], 8.0);  // obs 1 at point 2
/// assert_eq!(mat.column(0), &[1.0, 2.0, 3.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FdMatrix {
    data: Vec<f64>,
    nrows: usize,
    ncols: usize,
}

impl FdMatrix {
    /// Create from flat column-major data with dimension validation.
    ///
    /// Returns `Err` if `data.len() != nrows * ncols`.
    pub fn from_column_major(
        data: Vec<f64>,
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, crate::FdarError> {
        if data.len() != nrows * ncols {
            return Err(crate::FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("{}", nrows * ncols),
                actual: format!("{}", data.len()),
            });
        }
        Ok(Self { data, nrows, ncols })
    }

    /// Create from a borrowed slice (copies the data).
    ///
    /// Returns `Err` if `data.len() != nrows * ncols`.
    pub fn from_slice(data: &[f64], nrows: usize, ncols: usize) -> Result<Self, crate::FdarError> {
        if data.len() != nrows * ncols {
            return Err(crate::FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("{}", nrows * ncols),
                actual: format!("{}", data.len()),
            });
        }
        Ok(Self {
            data: data.to_vec(),
            nrows,
            ncols,
        })
    }

    /// Create a zero-filled matrix.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![0.0; nrows * ncols],
            nrows,
            ncols,
        }
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Dimensions as `(nrows, ncols)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the matrix is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a contiguous column slice (zero-copy).
    ///
    /// # Panics
    /// Panics if `col >= ncols`.
    #[inline]
    pub fn column(&self, col: usize) -> &[f64] {
        let start = col * self.nrows;
        &self.data[start..start + self.nrows]
    }

    /// Get a mutable contiguous column slice (zero-copy).
    ///
    /// # Panics
    /// Panics if `col >= ncols`.
    #[inline]
    pub fn column_mut(&mut self, col: usize) -> &mut [f64] {
        let start = col * self.nrows;
        &mut self.data[start..start + self.nrows]
    }

    /// Extract a single row as a new `Vec<f64>`.
    ///
    /// This is an O(ncols) operation because rows are not contiguous
    /// in column-major layout.
    pub fn row(&self, row: usize) -> Vec<f64> {
        (0..self.ncols)
            .map(|j| self.data[row + j * self.nrows])
            .collect()
    }

    /// Copy a single row into a pre-allocated buffer (zero allocation).
    ///
    /// # Panics
    /// Panics if `buf.len() < ncols` or `row >= nrows`.
    #[inline]
    pub fn row_to_buf(&self, row: usize, buf: &mut [f64]) {
        debug_assert!(
            row < self.nrows,
            "row {row} out of bounds for {} rows",
            self.nrows
        );
        debug_assert!(
            buf.len() >= self.ncols,
            "buffer len {} < ncols {}",
            buf.len(),
            self.ncols
        );
        let n = self.nrows;
        for j in 0..self.ncols {
            buf[j] = self.data[row + j * n];
        }
    }

    /// Compute the dot product of two rows without materializing either one.
    ///
    /// The rows may come from different matrices (which must have the same `ncols`).
    ///
    /// # Panics
    /// Panics (in debug) if `row_a >= self.nrows`, `row_b >= other.nrows`,
    /// or `self.ncols != other.ncols`.
    #[inline]
    pub fn row_dot(&self, row_a: usize, other: &FdMatrix, row_b: usize) -> f64 {
        debug_assert_eq!(self.ncols, other.ncols, "ncols mismatch in row_dot");
        let na = self.nrows;
        let nb = other.nrows;
        let mut sum = 0.0;
        for j in 0..self.ncols {
            sum += self.data[row_a + j * na] * other.data[row_b + j * nb];
        }
        sum
    }

    /// Compute the squared L2 distance between two rows without allocation.
    ///
    /// Equivalent to `||self.row(row_a) - other.row(row_b)||^2` but without
    /// materializing either row vector.
    ///
    /// # Panics
    /// Panics (in debug) if `row_a >= self.nrows`, `row_b >= other.nrows`,
    /// or `self.ncols != other.ncols`.
    #[inline]
    pub fn row_l2_sq(&self, row_a: usize, other: &FdMatrix, row_b: usize) -> f64 {
        debug_assert_eq!(self.ncols, other.ncols, "ncols mismatch in row_l2_sq");
        let na = self.nrows;
        let nb = other.nrows;
        let mut sum = 0.0;
        for j in 0..self.ncols {
            let d = self.data[row_a + j * na] - other.data[row_b + j * nb];
            sum += d * d;
        }
        sum
    }

    /// Iterate over rows, yielding each row as a `Vec<f64>`.
    ///
    /// More efficient than [`to_row_major()`](Self::to_row_major) when only a
    /// subset of rows are needed or when processing rows one at a time, because
    /// it materializes only one row at a time instead of allocating the entire
    /// transposed matrix up front.
    ///
    /// Because `FdMatrix` uses column-major storage, row elements are not
    /// contiguous and a zero-copy row slice is not possible. Each yielded
    /// `Vec<f64>` is an O(ncols) allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use fdars_core::matrix::FdMatrix;
    ///
    /// let mat = FdMatrix::from_column_major(vec![
    ///     1.0, 2.0,   // col 0
    ///     3.0, 4.0,   // col 1
    ///     5.0, 6.0,   // col 2
    /// ], 2, 3).unwrap();
    ///
    /// let rows: Vec<Vec<f64>> = mat.iter_rows().collect();
    /// assert_eq!(rows, vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]]);
    /// ```
    pub fn iter_rows(&self) -> impl Iterator<Item = Vec<f64>> + '_ {
        (0..self.nrows).map(move |i| self.row(i))
    }

    /// Iterate over columns, yielding each column as a slice `&[f64]`.
    ///
    /// Zero-copy because `FdMatrix` uses column-major storage and each column
    /// is a contiguous block in memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use fdars_core::matrix::FdMatrix;
    ///
    /// let mat = FdMatrix::from_column_major(vec![
    ///     1.0, 2.0,   // col 0
    ///     3.0, 4.0,   // col 1
    ///     5.0, 6.0,   // col 2
    /// ], 2, 3).unwrap();
    ///
    /// let cols: Vec<&[f64]> = mat.iter_columns().collect();
    /// assert_eq!(cols, vec![&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
    /// ```
    pub fn iter_columns(&self) -> impl Iterator<Item = &[f64]> {
        (0..self.ncols).map(move |j| self.column(j))
    }

    /// Extract all rows as `Vec<Vec<f64>>`.
    ///
    /// Equivalent to the former `extract_curves` function.
    pub fn rows(&self) -> Vec<Vec<f64>> {
        (0..self.nrows).map(|i| self.row(i)).collect()
    }

    /// Produce a single contiguous flat buffer in row-major order.
    ///
    /// Row `i` occupies `buf[i * ncols..(i + 1) * ncols]`. This is a single
    /// allocation versus `nrows` allocations from `rows()`, and gives better
    /// cache locality for row-oriented iteration.
    pub fn to_row_major(&self) -> Vec<f64> {
        let mut buf = vec![0.0; self.nrows * self.ncols];
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                buf[i * self.ncols + j] = self.data[i + j * self.nrows];
            }
        }
        buf
    }

    /// Flat slice of the underlying column-major data (zero-copy).
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutable flat slice of the underlying column-major data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Consume and return the underlying `Vec<f64>`.
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }

    /// Convert to a nalgebra `DMatrix<f64>`.
    ///
    /// This copies the data into nalgebra's storage. Both use column-major
    /// layout, so the copy is a simple memcpy.
    pub fn to_dmatrix(&self) -> DMatrix<f64> {
        DMatrix::from_column_slice(self.nrows, self.ncols, &self.data)
    }

    /// Create from a nalgebra `DMatrix<f64>`.
    ///
    /// Both use column-major layout so this is a direct copy.
    pub fn from_dmatrix(mat: &DMatrix<f64>) -> Self {
        let (nrows, ncols) = mat.shape();
        Self {
            data: mat.as_slice().to_vec(),
            nrows,
            ncols,
        }
    }

    /// Get element at (row, col) with bounds checking.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.nrows && col < self.ncols {
            Some(self.data[row + col * self.nrows])
        } else {
            None
        }
    }

    /// Extract a submatrix by row and column indices.
    ///
    /// Returns a new `FdMatrix` containing only the specified rows and columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use fdars_core::matrix::FdMatrix;
    ///
    /// let mat = FdMatrix::from_column_major(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ///     3, 3,
    /// ).unwrap();
    ///
    /// // Extract rows [0, 2] and columns [1, 2]
    /// let sub = mat.submatrix(&[0, 2], &[1, 2]);
    /// assert_eq!(sub.shape(), (2, 2));
    /// assert_eq!(sub[(0, 0)], 4.0); // row 0, col 1 of original
    /// assert_eq!(sub[(1, 1)], 9.0); // row 2, col 2 of original
    /// ```
    pub fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Self {
        let nr = rows.len();
        let nc = cols.len();
        let mut data = vec![0.0; nr * nc];
        for (jj, &col) in cols.iter().enumerate() {
            for (ii, &row) in rows.iter().enumerate() {
                data[ii + jj * nr] = self.data[row + col * self.nrows];
            }
        }
        Self {
            data,
            nrows: nr,
            ncols: nc,
        }
    }

    /// Extract a submatrix selecting only specific rows (all columns kept).
    ///
    /// Equivalent to `submatrix(rows, &(0..ncols).collect::<Vec<_>>())` but
    /// more efficient.
    pub fn select_rows(&self, rows: &[usize]) -> Self {
        let nr = rows.len();
        let nc = self.ncols;
        let mut data = vec![0.0; nr * nc];
        for j in 0..nc {
            for (ii, &row) in rows.iter().enumerate() {
                data[ii + j * nr] = self.data[row + j * self.nrows];
            }
        }
        Self {
            data,
            nrows: nr,
            ncols: nc,
        }
    }

    /// Extract a submatrix selecting only specific columns (all rows kept).
    ///
    /// Efficient for column-major layout — each column is a contiguous copy.
    pub fn select_columns(&self, cols: &[usize]) -> Self {
        let nr = self.nrows;
        let nc = cols.len();
        let mut data = vec![0.0; nr * nc];
        for (jj, &col) in cols.iter().enumerate() {
            let src = &self.data[col * nr..(col + 1) * nr];
            data[jj * nr..(jj + 1) * nr].copy_from_slice(src);
        }
        Self {
            data,
            nrows: nr,
            ncols: nc,
        }
    }

    /// Set element at (row, col) with bounds checking.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> bool {
        if row < self.nrows && col < self.ncols {
            self.data[row + col * self.nrows] = value;
            true
        } else {
            false
        }
    }
}

impl std::ops::Index<(usize, usize)> for FdMatrix {
    type Output = f64;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        debug_assert!(
            row < self.nrows && col < self.ncols,
            "FdMatrix index ({}, {}) out of bounds for {}x{} matrix",
            row,
            col,
            self.nrows,
            self.ncols
        );
        &self.data[row + col * self.nrows]
    }
}

impl std::ops::IndexMut<(usize, usize)> for FdMatrix {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
        debug_assert!(
            row < self.nrows && col < self.ncols,
            "FdMatrix index ({}, {}) out of bounds for {}x{} matrix",
            row,
            col,
            self.nrows,
            self.ncols
        );
        &mut self.data[row + col * self.nrows]
    }
}

impl std::fmt::Display for FdMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FdMatrix({}x{})", self.nrows, self.ncols)
    }
}

/// A set of multidimensional functional curves in R^d.
///
/// Each dimension is stored as a separate [`FdMatrix`] (n curves × m points).
/// For d=1 this is equivalent to a single `FdMatrix`.
#[derive(Debug, Clone, PartialEq)]
pub struct FdCurveSet {
    /// One matrix per coordinate dimension, each n × m.
    pub dims: Vec<FdMatrix>,
}

impl FdCurveSet {
    /// Number of coordinate dimensions (d).
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Number of curves (n).
    pub fn ncurves(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims[0].nrows()
        }
    }

    /// Number of evaluation points (m).
    pub fn npoints(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims[0].ncols()
        }
    }

    /// Wrap a single 1D `FdMatrix` as a `FdCurveSet`.
    pub fn from_1d(data: FdMatrix) -> Self {
        Self { dims: vec![data] }
    }

    /// Build from multiple dimension matrices.
    ///
    /// Returns `Err` if `dims` is empty or if dimensions are inconsistent.
    pub fn from_dims(dims: Vec<FdMatrix>) -> Result<Self, crate::FdarError> {
        if dims.is_empty() {
            return Err(crate::FdarError::InvalidDimension {
                parameter: "dims",
                expected: "non-empty".to_string(),
                actual: "empty".to_string(),
            });
        }
        let (n, m) = dims[0].shape();
        if dims.iter().any(|d| d.shape() != (n, m)) {
            return Err(crate::FdarError::InvalidDimension {
                parameter: "dims",
                expected: format!("all ({n}, {m})"),
                actual: "inconsistent shapes".to_string(),
            });
        }
        Ok(Self { dims })
    }

    /// Extract the R^d point for a given curve and time index.
    pub fn point(&self, curve: usize, time_idx: usize) -> Vec<f64> {
        self.dims.iter().map(|d| d[(curve, time_idx)]).collect()
    }
}

impl std::fmt::Display for FdCurveSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FdCurveSet(d={}, n={}, m={})",
            self.ndim(),
            self.ncurves(),
            self.npoints()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_3x4() -> FdMatrix {
        // 3 rows, 4 columns, column-major
        let data = vec![
            1.0, 2.0, 3.0, // col 0
            4.0, 5.0, 6.0, // col 1
            7.0, 8.0, 9.0, // col 2
            10.0, 11.0, 12.0, // col 3
        ];
        FdMatrix::from_column_major(data, 3, 4).unwrap()
    }

    #[test]
    fn test_from_column_major_valid() {
        let mat = sample_3x4();
        assert_eq!(mat.nrows(), 3);
        assert_eq!(mat.ncols(), 4);
        assert_eq!(mat.shape(), (3, 4));
        assert_eq!(mat.len(), 12);
        assert!(!mat.is_empty());
    }

    #[test]
    fn test_from_column_major_invalid() {
        assert!(FdMatrix::from_column_major(vec![1.0, 2.0], 3, 4).is_err());
    }

    #[test]
    fn test_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = FdMatrix::from_slice(&data, 2, 3).unwrap();
        assert_eq!(mat[(0, 0)], 1.0);
        assert_eq!(mat[(1, 0)], 2.0);
        assert_eq!(mat[(0, 1)], 3.0);
    }

    #[test]
    fn test_from_slice_invalid() {
        assert!(FdMatrix::from_slice(&[1.0, 2.0], 3, 3).is_err());
    }

    #[test]
    fn test_zeros() {
        let mat = FdMatrix::zeros(2, 3);
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 3);
        for j in 0..3 {
            for i in 0..2 {
                assert_eq!(mat[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn test_index() {
        let mat = sample_3x4();
        assert_eq!(mat[(0, 0)], 1.0);
        assert_eq!(mat[(1, 0)], 2.0);
        assert_eq!(mat[(2, 0)], 3.0);
        assert_eq!(mat[(0, 1)], 4.0);
        assert_eq!(mat[(1, 1)], 5.0);
        assert_eq!(mat[(2, 3)], 12.0);
    }

    #[test]
    fn test_index_mut() {
        let mut mat = sample_3x4();
        mat[(1, 2)] = 99.0;
        assert_eq!(mat[(1, 2)], 99.0);
    }

    #[test]
    fn test_column() {
        let mat = sample_3x4();
        assert_eq!(mat.column(0), &[1.0, 2.0, 3.0]);
        assert_eq!(mat.column(1), &[4.0, 5.0, 6.0]);
        assert_eq!(mat.column(3), &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_column_mut() {
        let mut mat = sample_3x4();
        mat.column_mut(1)[0] = 99.0;
        assert_eq!(mat[(0, 1)], 99.0);
    }

    #[test]
    fn test_row() {
        let mat = sample_3x4();
        assert_eq!(mat.row(0), vec![1.0, 4.0, 7.0, 10.0]);
        assert_eq!(mat.row(1), vec![2.0, 5.0, 8.0, 11.0]);
        assert_eq!(mat.row(2), vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_rows() {
        let mat = sample_3x4();
        let rows = mat.rows();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![1.0, 4.0, 7.0, 10.0]);
        assert_eq!(rows[2], vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_as_slice() {
        let mat = sample_3x4();
        let expected = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        assert_eq!(mat.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_into_vec() {
        let mat = sample_3x4();
        let v = mat.into_vec();
        assert_eq!(v.len(), 12);
        assert_eq!(v[0], 1.0);
    }

    #[test]
    fn test_get_bounds_check() {
        let mat = sample_3x4();
        assert_eq!(mat.get(0, 0), Some(1.0));
        assert_eq!(mat.get(2, 3), Some(12.0));
        assert_eq!(mat.get(3, 0), None); // row out of bounds
        assert_eq!(mat.get(0, 4), None); // col out of bounds
    }

    #[test]
    fn test_set_bounds_check() {
        let mut mat = sample_3x4();
        assert!(mat.set(1, 1, 99.0));
        assert_eq!(mat[(1, 1)], 99.0);
        assert!(!mat.set(5, 0, 99.0)); // out of bounds
    }

    #[test]
    fn test_nalgebra_roundtrip() {
        let mat = sample_3x4();
        let dmat = mat.to_dmatrix();
        assert_eq!(dmat.nrows(), 3);
        assert_eq!(dmat.ncols(), 4);
        assert_eq!(dmat[(0, 0)], 1.0);
        assert_eq!(dmat[(1, 2)], 8.0);

        let back = FdMatrix::from_dmatrix(&dmat);
        assert_eq!(mat, back);
    }

    #[test]
    fn test_empty() {
        let mat = FdMatrix::zeros(0, 0);
        assert!(mat.is_empty());
        assert_eq!(mat.len(), 0);
    }

    #[test]
    fn test_single_element() {
        let mat = FdMatrix::from_column_major(vec![42.0], 1, 1).unwrap();
        assert_eq!(mat[(0, 0)], 42.0);
        assert_eq!(mat.column(0), &[42.0]);
        assert_eq!(mat.row(0), vec![42.0]);
    }

    #[test]
    fn test_display() {
        let mat = sample_3x4();
        assert_eq!(format!("{}", mat), "FdMatrix(3x4)");
    }

    #[test]
    fn test_clone() {
        let mat = sample_3x4();
        let cloned = mat.clone();
        assert_eq!(mat, cloned);
    }

    #[test]
    fn test_as_mut_slice() {
        let mut mat = FdMatrix::zeros(2, 2);
        let s = mat.as_mut_slice();
        s[0] = 1.0;
        s[1] = 2.0;
        s[2] = 3.0;
        s[3] = 4.0;
        assert_eq!(mat[(0, 0)], 1.0);
        assert_eq!(mat[(1, 0)], 2.0);
        assert_eq!(mat[(0, 1)], 3.0);
        assert_eq!(mat[(1, 1)], 4.0);
    }

    #[test]
    fn test_fd_curve_set_empty() {
        assert!(FdCurveSet::from_dims(vec![]).is_err());
        let cs = FdCurveSet::from_dims(vec![]).unwrap_or(FdCurveSet { dims: vec![] });
        assert_eq!(cs.ndim(), 0);
        assert_eq!(cs.ncurves(), 0);
        assert_eq!(cs.npoints(), 0);
        assert_eq!(format!("{}", cs), "FdCurveSet(d=0, n=0, m=0)");
    }

    #[test]
    fn test_fd_curve_set_from_1d() {
        let mat = sample_3x4();
        let cs = FdCurveSet::from_1d(mat.clone());
        assert_eq!(cs.ndim(), 1);
        assert_eq!(cs.ncurves(), 3);
        assert_eq!(cs.npoints(), 4);
        assert_eq!(cs.point(0, 0), vec![1.0]);
        assert_eq!(cs.point(1, 2), vec![8.0]);
    }

    #[test]
    fn test_fd_curve_set_from_dims_consistent() {
        let m1 = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let m2 = FdMatrix::from_column_major(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        let cs = FdCurveSet::from_dims(vec![m1, m2]).unwrap();
        assert_eq!(cs.ndim(), 2);
        assert_eq!(cs.point(0, 0), vec![1.0, 5.0]);
        assert_eq!(cs.point(1, 1), vec![4.0, 8.0]);
        assert_eq!(format!("{}", cs), "FdCurveSet(d=2, n=2, m=2)");
    }

    #[test]
    fn test_fd_curve_set_from_dims_inconsistent() {
        let m1 = FdMatrix::from_column_major(vec![1.0, 2.0], 2, 1).unwrap();
        let m2 = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        assert!(FdCurveSet::from_dims(vec![m1, m2]).is_err());
    }

    #[test]
    fn test_to_row_major() {
        let mat = sample_3x4();
        let rm = mat.to_row_major();
        // Row 0: [1,4,7,10], Row 1: [2,5,8,11], Row 2: [3,6,9,12]
        assert_eq!(
            rm,
            vec![1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]
        );
    }

    #[test]
    fn test_row_to_buf() {
        let mat = sample_3x4();
        let mut buf = vec![0.0; 4];
        mat.row_to_buf(0, &mut buf);
        assert_eq!(buf, vec![1.0, 4.0, 7.0, 10.0]);
        mat.row_to_buf(1, &mut buf);
        assert_eq!(buf, vec![2.0, 5.0, 8.0, 11.0]);
        mat.row_to_buf(2, &mut buf);
        assert_eq!(buf, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_row_to_buf_larger_buffer() {
        let mat = sample_3x4();
        let mut buf = vec![99.0; 6]; // bigger than ncols
        mat.row_to_buf(0, &mut buf);
        assert_eq!(&buf[..4], &[1.0, 4.0, 7.0, 10.0]);
        // Remaining elements unchanged
        assert_eq!(buf[4], 99.0);
    }

    #[test]
    fn test_row_dot_same_matrix() {
        let mat = sample_3x4();
        // row0 = [1, 4, 7, 10], row1 = [2, 5, 8, 11]
        // dot = 1*2 + 4*5 + 7*8 + 10*11 = 2 + 20 + 56 + 110 = 188
        assert_eq!(mat.row_dot(0, &mat, 1), 188.0);
        // self dot: row0 . row0 = 1+16+49+100 = 166
        assert_eq!(mat.row_dot(0, &mat, 0), 166.0);
    }

    #[test]
    fn test_row_dot_different_matrices() {
        let mat1 = sample_3x4();
        let data2 = vec![
            10.0, 20.0, 30.0, // col 0
            40.0, 50.0, 60.0, // col 1
            70.0, 80.0, 90.0, // col 2
            100.0, 110.0, 120.0, // col 3
        ];
        let mat2 = FdMatrix::from_column_major(data2, 3, 4).unwrap();
        // mat1 row0 = [1, 4, 7, 10], mat2 row0 = [10, 40, 70, 100]
        // dot = 10 + 160 + 490 + 1000 = 1660
        assert_eq!(mat1.row_dot(0, &mat2, 0), 1660.0);
    }

    #[test]
    fn test_row_l2_sq_identical() {
        let mat = sample_3x4();
        assert_eq!(mat.row_l2_sq(0, &mat, 0), 0.0);
        assert_eq!(mat.row_l2_sq(1, &mat, 1), 0.0);
    }

    #[test]
    fn test_row_l2_sq_different() {
        let mat = sample_3x4();
        // row0 = [1,4,7,10], row1 = [2,5,8,11]
        // diff = [-1,-1,-1,-1], sq sum = 4
        assert_eq!(mat.row_l2_sq(0, &mat, 1), 4.0);
    }

    #[test]
    fn test_row_l2_sq_cross_matrix() {
        let mat1 = FdMatrix::from_column_major(vec![0.0, 0.0], 1, 2).unwrap();
        let mat2 = FdMatrix::from_column_major(vec![3.0, 4.0], 1, 2).unwrap();
        // row0 = [0, 0], row0 = [3, 4], sq dist = 9 + 16 = 25
        assert_eq!(mat1.row_l2_sq(0, &mat2, 0), 25.0);
    }

    #[test]
    fn test_column_major_layout_matches_manual() {
        // Verify that FdMatrix[(i, j)] == data[i + j * n] for all i, j
        let n = 5;
        let m = 7;
        let data: Vec<f64> = (0..n * m).map(|x| x as f64).collect();
        let mat = FdMatrix::from_column_major(data.clone(), n, m).unwrap();

        for j in 0..m {
            for i in 0..n {
                assert_eq!(mat[(i, j)], data[i + j * n]);
            }
        }
    }

    #[test]
    fn test_iter_rows() {
        let mat = sample_3x4();
        let rows: Vec<Vec<f64>> = mat.iter_rows().collect();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![1.0, 4.0, 7.0, 10.0]);
        assert_eq!(rows[1], vec![2.0, 5.0, 8.0, 11.0]);
        assert_eq!(rows[2], vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_iter_rows_matches_rows() {
        let mat = sample_3x4();
        let from_iter: Vec<Vec<f64>> = mat.iter_rows().collect();
        let from_rows = mat.rows();
        assert_eq!(from_iter, from_rows);
    }

    #[test]
    fn test_iter_rows_partial() {
        // Verify that taking only a subset avoids full materialization
        let mat = sample_3x4();
        let first_two: Vec<Vec<f64>> = mat.iter_rows().take(2).collect();
        assert_eq!(first_two.len(), 2);
        assert_eq!(first_two[0], vec![1.0, 4.0, 7.0, 10.0]);
        assert_eq!(first_two[1], vec![2.0, 5.0, 8.0, 11.0]);
    }

    #[test]
    fn test_iter_rows_empty() {
        let mat = FdMatrix::zeros(0, 0);
        let rows: Vec<Vec<f64>> = mat.iter_rows().collect();
        assert!(rows.is_empty());
    }

    #[test]
    fn test_iter_rows_single_row() {
        let mat = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
        let rows: Vec<Vec<f64>> = mat.iter_rows().collect();
        assert_eq!(rows, vec![vec![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn test_iter_rows_single_column() {
        let mat = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        let rows: Vec<Vec<f64>> = mat.iter_rows().collect();
        assert_eq!(rows, vec![vec![1.0], vec![2.0], vec![3.0]]);
    }

    #[test]
    fn test_iter_columns() {
        let mat = sample_3x4();
        let cols: Vec<&[f64]> = mat.iter_columns().collect();
        assert_eq!(cols.len(), 4);
        assert_eq!(cols[0], &[1.0, 2.0, 3.0]);
        assert_eq!(cols[1], &[4.0, 5.0, 6.0]);
        assert_eq!(cols[2], &[7.0, 8.0, 9.0]);
        assert_eq!(cols[3], &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_iter_columns_partial() {
        let mat = sample_3x4();
        let first_two: Vec<&[f64]> = mat.iter_columns().take(2).collect();
        assert_eq!(first_two.len(), 2);
        assert_eq!(first_two[0], &[1.0, 2.0, 3.0]);
        assert_eq!(first_two[1], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_iter_columns_empty() {
        let mat = FdMatrix::zeros(0, 0);
        let cols: Vec<&[f64]> = mat.iter_columns().collect();
        assert!(cols.is_empty());
    }

    #[test]
    fn test_iter_columns_single_column() {
        let mat = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        let cols: Vec<&[f64]> = mat.iter_columns().collect();
        assert_eq!(cols, vec![&[1.0, 2.0, 3.0]]);
    }

    #[test]
    fn test_iter_columns_single_row() {
        let mat = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
        let cols: Vec<&[f64]> = mat.iter_columns().collect();
        assert_eq!(cols, vec![&[1.0_f64] as &[f64], &[2.0], &[3.0]]);
    }

    #[test]
    fn test_iter_rows_enumerate() {
        let mat = sample_3x4();
        for (i, row) in mat.iter_rows().enumerate() {
            assert_eq!(row, mat.row(i));
        }
    }

    #[test]
    fn test_iter_columns_enumerate() {
        let mat = sample_3x4();
        for (j, col) in mat.iter_columns().enumerate() {
            assert_eq!(col, mat.column(j));
        }
    }
}
