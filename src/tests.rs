use crate::factor_solve;
use num_traits::Float;

const TESTS_TOL: f64 = 1e-4;

#[test]
fn basic() {
    let a_n: usize = 10;
    let a_p: Vec<usize> = vec![0, 1, 2, 4, 5, 6, 8, 10, 12, 14, 17];
    let a_i: Vec<usize> = vec![0, 1, 1, 2, 3, 4, 1, 5, 0, 6, 3, 7, 6, 8, 1, 2, 9];
    let a_x: Vec<f64> = vec![
        1.0, 0.460641, -0.121189, 0.417928, 0.177828, 0.1, -0.0290058, -1.0, 0.350321, -0.441092,
        -0.0845395, -0.316228, 0.178663, -0.299077, 0.182452, -1.56506, -0.1,
    ];
    let mut b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    // RHS and solution to Ax = b
    let xsol = vec![
        10.2171, 3.9416, -5.69096, 9.28661, 50.0, -6.11433, -26.3104, -27.7809, -45.8099, -3.74178,
    ];

    // x replaces b during solve
    let status = factor_solve(a_n, &a_p, &a_i, &a_x, &mut b);

    assert!(status.is_ok(), "Factorisation failed");
    assert!(
        vec_diff_norm(&b, &xsol, a_n) < TESTS_TOL,
        "Solve accuracy failed"
    );
}

#[test]
fn basic_f32() {
    let a_n: usize = 10;
    let a_p: Vec<usize> = vec![0, 1, 2, 4, 5, 6, 8, 10, 12, 14, 17];
    let a_i: Vec<usize> = vec![0, 1, 1, 2, 3, 4, 1, 5, 0, 6, 3, 7, 6, 8, 1, 2, 9];
    let a_x: Vec<f32> = vec![
        1.0, 0.460641, -0.121189, 0.417928, 0.177828, 0.1, -0.0290058, -1.0, 0.350321, -0.441092,
        -0.0845395, -0.316228, 0.178663, -0.299077, 0.182452, -1.56506, -0.1,
    ];
    let mut b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    // RHS and solution to Ax = b
    let xsol: Vec<f32> = vec![
        10.2171, 3.9416, -5.69096, 9.28661, 50.0, -6.11433, -26.3104, -27.7809, -45.8099, -3.74178,
    ];

    // x replaces b during solve
    let status = factor_solve(a_n, &a_p, &a_i, &a_x, &mut b);

    assert!(status.is_ok(), "Factorisation failed");
    assert!(
        vec_diff_norm(&b, &xsol, a_n) < TESTS_TOL as f32,
        "Solve accuracy failed"
    );
}

#[test]
fn identity() {
    // A matrix data
    let a_p: Vec<usize> = vec![0, 1, 2, 3, 4];
    let a_i: Vec<usize> = vec![0, 1, 2, 3];
    let a_x: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];
    let a_n = 4;

    // RHS and solution to Ax = b
    let mut b = vec![2.0, 2.0, 2.0, 2.0];
    let xsol = vec![2.0, 2.0, 2.0, 2.0];

    // x replaces b during solve
    let status = factor_solve(a_n, &a_p, &a_i, &a_x, &mut b);

    assert!(status.is_ok(), "Factorisation failed");
    assert!(
        vec_diff_norm(&b, &xsol, a_n) < TESTS_TOL,
        "Solve accuracy failed"
    );
}

#[test]
fn osqp_kkt() {
    // Unordered A
    let a_p = vec![0, 1, 2, 5, 6, 7, 8, 12];
    let a_i = vec![0, 1, 2, 1, 0, 3, 4, 5, 5, 6, 4, 3];
    let a_x = vec![
        -0.25, -0.25, 1.0, 0.513578, 0.529142, -0.25, -0.25, 1.10274, 0.15538, 1.25883, 0.13458,
        0.621134,
    ];
    let a_n = 7;

    // RHS and solution to Ax = b
    let mut b = vec![
        -0.595598, -0.0193715, -0.576156, -0.168746, 0.61543, 0.419073, 1.31087,
    ];
    let xsol = vec![
        1.13141, -1.1367, -0.591044, 1.68867, -2.24209, 0.32254, 0.407998,
    ];

    // x replaces b during solve
    let status = factor_solve(a_n, &a_p, &a_i, &a_x, &mut b);

    assert!(status.is_ok(), "Factorisation failed");
    assert!(
        vec_diff_norm(&b, &xsol, a_n) < TESTS_TOL,
        "Solve accuracy failed"
    );
}

#[test]
fn rank_deficient() {
    // A matrix data
    let a_p = vec![0, 1, 3];
    let a_i = vec![0, 0, 1];
    let a_x = vec![1.0, 1.0, 1.0];
    let a_n = 2;

    // RHS for Ax = b (should fail to solve)
    let mut b = vec![1.0, 1.0];

    // x replaces b during solve
    let status = factor_solve(a_n, &a_p, &a_i, &a_x, &mut b);

    assert!(status.is_err(), "Rank deficiency not detected");
}

#[test]
fn singleton() {
    // A matrix data
    let a_p = vec![0, 1];
    let a_i = vec![0];
    let a_x = vec![0.2];
    let a_n = 1;

    // RHS and solution to Ax = b
    let mut b = vec![2.0];
    let xsol = vec![10.0];

    // x replaces b during solve
    let status = factor_solve(a_n, &a_p, &a_i, &a_x, &mut b);

    assert!(status.is_ok(), "Factorisation failed");
    assert!(
        vec_diff_norm(&b, &xsol, a_n) < TESTS_TOL,
        "Solve accuracy failed"
    );
}

#[test]
fn sym_structure() {
    // A matrix data
    let a_p = vec![0, 2, 4];
    let a_i = vec![0, 1, 0, 1];
    let a_x = vec![5.0, 1.0, 1.0, 5.0];
    let a_n = 2;

    // RHS for Ax = b
    let mut b = vec![1.0, 1.0];

    // x replaces b during solve
    let status = factor_solve(a_n, &a_p, &a_i, &a_x, &mut b);

    assert!(status.is_err(), "Fully symmetric input not detected");
}

#[test]
fn tril_structure() {
    // A matrix data
    let a_p = vec![0, 2, 3];
    let a_i = vec![0, 1, 1];
    let a_x = vec![5.0, 1.0, 5.0];
    let a_n = 2;

    // RHS for Ax = b
    let mut b = vec![1.0, 1.0];

    // x replaces b during solve
    let status = factor_solve(a_n, &a_p, &a_i, &a_x, &mut b);

    assert!(status.is_err(), "Tril input not detected");
}

#[test]
fn two_by_two() {
    // A matrix data
    let a_p = vec![0, 1, 3];
    let a_i = vec![0, 0, 1];
    let a_x = vec![1.0, 1.0, -1.0];
    let a_n = 2;

    // RHS and solution to Ax = b
    let mut b = vec![2.0, 4.0];
    let xsol = vec![3.0, -1.0];

    // x replaces b during solve
    let status = factor_solve(a_n, &a_p, &a_i, &a_x, &mut b);

    assert!(status.is_ok(), "Factorisation failed");
    assert!(
        vec_diff_norm(&b, &xsol, a_n) < TESTS_TOL,
        "Solve accuracy failed"
    );
}

#[test]
fn zero_on_diag() {
    // A matrix data
    let a_p = vec![0, 1, 2, 5];
    let a_i = vec![0, 0, 0, 1, 2];
    let a_x = vec![4.0, 1.0, 2.0, 1.0, -3.0];
    let a_n = 3;

    // RHS and solution to Ax = b
    let mut b = vec![6.0, 9.0, 12.0];
    let xsol = vec![17.0, -46.0, -8.0];

    // x replaces b during solve (should fill due to zero in middle)
    // NB : this system is solvable, but not by LDL
    let status = factor_solve(a_n, &a_p, &a_i, &a_x, &mut b);

    assert!(status.is_ok(), "Factorisation failed");
    assert!(
        vec_diff_norm(&b, &xsol, a_n) < TESTS_TOL,
        "Solve accuracy failed"
    );
}

fn vec_diff_norm<S: Float>(x: &[S], y: &[S], len: usize) -> S {
    let mut max_diff = S::zero();
    for i in 0..len {
        let el_diff = x[i as usize] - y[i as usize];
        max_diff = if el_diff > max_diff {
            el_diff
        } else {
            if -el_diff > max_diff {
                -el_diff
            } else {
                max_diff
            }
        };
    }
    max_diff
}
