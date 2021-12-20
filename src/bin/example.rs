extern crate ldl;

use ldl::*;

fn main() {
    let a_n: usize = 10;
    let a_p: Vec<usize> = vec![0, 1, 2, 4, 5, 6, 8, 10, 12, 14, 17];
    let a_i: Vec<usize> = vec![0, 1, 1, 2, 3, 4, 1, 5, 0, 6, 3, 7, 6, 8, 1, 2, 9];
    let a_x: Vec<f64> = vec![
        1.0, 0.460641, -0.121189, 0.417928, 0.177828, 0.1, -0.0290058, -1.0, 0.350321, -0.441092,
        -0.0845395, -0.316228, 0.178663, -0.299077, 0.182452, -1.56506, -0.1,
    ];
    let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let l_n = a_n;

    // pre-factorisation memory allocations //

    // These can happen *before* the etree is calculated
    // since the sizes are not sparsity pattern specific

    // For the elimination tree
    let mut etree_ = vec![0; a_n as usize];
    let mut l_nz = vec![0; a_n as usize];

    // For the L factors. Li and Lx are sparsity dependent
    // so must be done after the etree is constructed
    let mut l_p = vec![0; a_n as usize + 1];
    let mut d = vec![0.0; a_n as usize];
    let mut d_inv = vec![0.0; a_n as usize];

    // Working memory. Note that both the etree and factor
    // calls requires a working vector of int, with
    // the factor function requiring 3*An elements and the
    // etree only An elements. Just allocate the larger
    // amount here and use it in both places
    let mut iwork = vec![0; 3 * a_n as usize];
    let mut bwork = vec![false; a_n as usize];
    let mut fwork = vec![0.0; a_n as usize];

    // Elimination tree calculation //

    let sum_l_nz = etree(a_n, &a_p, &a_i, &mut iwork, &mut l_nz, &mut etree_).unwrap();

    // LDL factorisation //

    let mut l_i = vec![0; sum_l_nz as usize];
    let mut l_x = vec![0.0; sum_l_nz as usize];

    factor(
        a_n, &a_p, &a_i, &a_x, &mut l_p, &mut l_i, &mut l_x, &mut d, &mut d_inv, &mut l_nz,
        &etree_, &mut bwork, &mut iwork, &mut fwork,
    )
    .unwrap();

    // Solve //

    let mut x = vec![0.0; a_n as usize];

    // when solving A\b, start with x = b
    for i in 0..l_n {
        x[i as usize] = b[i as usize];
    }
    solve(l_n, &l_p, &l_i, &l_x, &d_inv, &mut x);

    // Print factors and solution //

    println!();
    println!("A (CSC format):");
    println!("--------------------------");
    println!("A.p = {:?}", a_p);
    println!("A.i = {:?}", a_i);
    println!("A.x = {:?}", a_x);
    println!();
    println!();

    println!("elimination tree:");
    println!("--------------------------");
    println!("etree = {:?}", etree_);
    println!("Lnz = {:?}", l_nz);
    println!();
    println!();

    println!("L (CSC format):");
    println!("--------------------------");
    println!("L.p = {:?}", l_p);
    println!("L.i = {:?}", l_i);
    println!("L.x = {:?}", l_x);
    println!();
    println!();

    println!("D:");
    println!("--------------------------");
    println!("diag(D)      = {:?}", d);
    println!("diag(D^{{-1}}) = {:?}", d_inv);
    println!();
    println!();

    println!("solve results:");
    println!("--------------------------");
    println!("b = {:?}", b);
    println!("A\\b = {:?}", x);
    println!();
    println!();
}
