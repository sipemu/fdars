//! fpca_dump — Standalone harness binary for the fdars benchmark agreement gate.
//!
//! Reads a full-precision CSV (produced by benchmarks/generate_data.py) and runs
//! fdars `fdata_to_pc` FPCA with n_components=5, then emits the loadings matrix
//! as JSON to stdout.
//!
//! Usage:
//!   fpca_dump --csv path/to/curves_n60_m50.csv --ncomp 5
//!
//! Output (stdout):
//!   {"loadings": [[...], ...]}   // (ncomp rows, each of length m)
//!
//! This binary is NOT part of the fdars workspace; it lives under benchmarks/fpca_dump/
//! with its own Cargo.toml. It has no new fdars-core dependency and makes no change
//! to the fdars-core public API.

use std::env;
use std::io::{self, BufRead};

use fdars_core::matrix::FdMatrix;
use fdars_core::regression::fdata_to_pc;

fn parse_args() -> (String, usize) {
    let args: Vec<String> = env::args().collect();
    let mut csv_path = String::new();
    let mut ncomp: usize = 5;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--csv" => {
                i += 1;
                csv_path = args[i].clone();
            }
            "--ncomp" => {
                i += 1;
                ncomp = args[i].parse().expect("ncomp must be a positive integer");
            }
            _ => {}
        }
        i += 1;
    }
    if csv_path.is_empty() {
        eprintln!("Usage: fpca_dump --csv <path> [--ncomp <n>]");
        std::process::exit(1);
    }
    (csv_path, ncomp)
}

fn load_csv(path: &str) -> (Vec<f64>, usize, usize) {
    // Parse the full-precision CSV produced by generate_data.py.
    // Format: header row (curve_id, t_0, t_1, ..., t_{m-1}), then n data rows.
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| { eprintln!("Cannot open {}: {}", path, e); std::process::exit(1); });
    let reader = io::BufReader::new(file);

    let mut lines = reader.lines();
    let header = lines.next()
        .expect("Empty CSV")
        .expect("Header read error");
    let cols: Vec<&str> = header.split(',').collect();
    // First column is curve_id; remaining are t_0..t_{m-1}
    let m = cols.len() - 1;

    let mut data_col_major: Vec<f64> = Vec::new();
    let mut n = 0usize;
    let mut rows: Vec<Vec<f64>> = Vec::new();

    for line in lines {
        let line = line.expect("Line read error");
        let line = line.trim();
        if line.is_empty() { continue; }
        let parts: Vec<&str> = line.split(',').collect();
        // parts[0] is curve_id; parts[1..] are the m values
        let row: Vec<f64> = parts[1..].iter()
            .map(|s| s.parse::<f64>().expect("float parse error"))
            .collect();
        assert_eq!(row.len(), m, "Row length mismatch");
        rows.push(row);
        n += 1;
    }

    // Convert row-major rows to column-major flat vector for FdMatrix
    data_col_major.resize(n * m, 0.0);
    for i in 0..n {
        for j in 0..m {
            data_col_major[i + j * n] = rows[i][j];
        }
    }

    (data_col_major, n, m)
}

fn main() {
    let (csv_path, ncomp) = parse_args();
    let (data, n, m) = load_csv(&csv_path);

    let mat = FdMatrix::from_column_major(data, n, m)
        .expect("FdMatrix construction failed");

    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    let fpca = fdata_to_pc(&mat, ncomp, &argvals)
        .expect("FPCA failed");

    // rotation is (m x ncomp) column-major — emit as (ncomp rows, each of length m)
    let mut loadings: Vec<Vec<f64>> = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let col: Vec<f64> = (0..m).map(|j| fpca.rotation[(j, k)]).collect();
        loadings.push(col);
    }

    // Emit JSON to stdout — flat manual serialisation to avoid adding serde dep
    // to fpca_dump (it's a throwaway harness binary)
    let mut out = String::from("{\"loadings\":[");
    for (k, row) in loadings.iter().enumerate() {
        if k > 0 { out.push(','); }
        out.push('[');
        for (j, &v) in row.iter().enumerate() {
            if j > 0 { out.push(','); }
            // Full precision float (%.17g equivalent)
            out.push_str(&format!("{:.17e}", v));
        }
        out.push(']');
    }
    out.push_str("]}");
    println!("{}", out);
}
