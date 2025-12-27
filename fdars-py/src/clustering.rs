// Clustering algorithms for functional data

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Helper to convert row-major numpy array to column-major flat vector
fn to_col_major(data: &ndarray::ArrayView2<f64>) -> Vec<f64> {
    let (nrow, ncol) = data.dim();
    (0..ncol)
        .flat_map(|j| (0..nrow).map(move |i| data[[i, j]]))
        .collect()
}

/// Helper to convert column-major flat vector to row-major ndarray
fn to_row_major_2d(flat: &[f64], nrow: usize, ncol: usize) -> ndarray::Array2<f64> {
    let mut arr = ndarray::Array2::<f64>::zeros((nrow, ncol));
    for i in 0..nrow {
        for j in 0..ncol {
            arr[[i, j]] = flat[i + j * nrow];
        }
    }
    arr
}

/// K-means clustering for functional data
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// n_clusters : int
///     Number of clusters
/// max_iter : int
///     Maximum number of iterations (default 100)
/// tol : float
///     Convergence tolerance (default 1e-6)
/// seed : int
///     Random seed for reproducibility
///
/// Returns
/// -------
/// dict with keys: 'labels', 'centers', 'inertia', 'n_iter', 'converged'
#[pyfunction]
#[pyo3(signature = (data, argvals, n_clusters, max_iter=100, tol=1e-6, seed=42))]
pub fn kmeans_fd<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 || n_clusters == 0 || n_clusters > n {
        dict.set_item("labels", Vec::<i32>::new().into_pyarray(py))?;
        dict.set_item(
            "centers",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item("inertia", f64::NAN)?;
        dict.set_item("n_iter", 0)?;
        dict.set_item("converged", false)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // kmeans_fd(data, n, m, argvals, k, max_iter, tol, seed)
    let result = fdars_core::clustering::kmeans_fd(
        &data_flat,
        n,
        n_points,
        &argvals_vec,
        n_clusters,
        max_iter,
        tol,
        seed,
    );

    let labels: Vec<i32> = result.cluster.iter().map(|&x| x as i32).collect();
    dict.set_item("labels", labels.into_pyarray(py))?;
    dict.set_item(
        "centers",
        to_row_major_2d(&result.centers, n_clusters, n_points).into_pyarray(py),
    )?;
    dict.set_item("inertia", result.tot_withinss)?;
    dict.set_item("withinss", result.withinss.into_pyarray(py))?;
    dict.set_item("n_iter", result.iter)?;
    dict.set_item("converged", result.converged)?;

    Ok(dict)
}

/// Fuzzy C-means clustering for functional data
///
/// Parameters
/// ----------
/// data : ndarray, shape (n_samples, n_points)
///     Functional data matrix
/// argvals : ndarray, shape (n_points,)
///     Evaluation points
/// n_clusters : int
///     Number of clusters
/// m : float
///     Fuzziness parameter (default 2.0)
/// max_iter : int
///     Maximum number of iterations (default 100)
/// tol : float
///     Convergence tolerance (default 1e-6)
/// seed : int
///     Random seed for reproducibility
///
/// Returns
/// -------
/// dict with keys: 'labels', 'membership', 'centers', 'inertia', 'n_iter', 'converged'
#[pyfunction]
#[pyo3(signature = (data, argvals, n_clusters, m=2.0, max_iter=100, tol=1e-6, seed=42))]
pub fn fcmeans_fd<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    argvals: PyReadonlyArray1<'py, f64>,
    n_clusters: usize,
    m: f64,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data_arr = data.as_array();
    let argvals_arr = argvals.as_array();
    let (n, n_points) = data_arr.dim();

    let dict = pyo3::types::PyDict::new(py);

    if n == 0 || n_points == 0 || n_clusters == 0 || n_clusters > n {
        dict.set_item("labels", Vec::<i32>::new().into_pyarray(py))?;
        dict.set_item(
            "membership",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item(
            "centers",
            ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py),
        )?;
        dict.set_item("inertia", f64::NAN)?;
        dict.set_item("n_iter", 0)?;
        dict.set_item("converged", false)?;
        return Ok(dict);
    }

    let data_flat = to_col_major(&data_arr);
    let argvals_vec: Vec<f64> = argvals_arr.to_vec();

    // fuzzy_cmeans_fd(data, n, m, argvals, k, fuzziness, max_iter, tol, seed)
    let result = fdars_core::clustering::fuzzy_cmeans_fd(
        &data_flat,
        n,
        n_points,
        &argvals_vec,
        n_clusters,
        m,
        max_iter,
        tol,
        seed,
    );

    // Compute hard cluster assignments from membership matrix (argmax per row)
    let labels: Vec<i32> = (0..n)
        .map(|i| {
            let mut max_val = f64::NEG_INFINITY;
            let mut max_cluster = 0;
            for c in 0..n_clusters {
                let mem = result.membership[i + c * n]; // column-major
                if mem > max_val {
                    max_val = mem;
                    max_cluster = c;
                }
            }
            max_cluster as i32
        })
        .collect();

    dict.set_item("labels", labels.into_pyarray(py))?;
    dict.set_item(
        "membership",
        to_row_major_2d(&result.membership, n, n_clusters).into_pyarray(py),
    )?;
    dict.set_item(
        "centers",
        to_row_major_2d(&result.centers, n_clusters, n_points).into_pyarray(py),
    )?;
    dict.set_item("n_iter", result.iter)?;
    dict.set_item("converged", result.converged)?;

    Ok(dict)
}
