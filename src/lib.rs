extern crate num_traits;

#[cfg(test)]
mod tests;

use num_traits::{Float, NumAssignOps};

const UNKNOWN: isize = -1;
const USED: bool = true;
const UNUSED: bool = false;

/// Compute the elimination tree for a quasidefinite matrix
/// in compressed sparse column form, where the input matrix is
/// assumed to contain data for the upper triangular part of `A` only,
/// and there are no duplicate indices.
///
/// Returns an elimination tree for the factorization `A = LDL^T` and a
/// count of the nonzeros in each column of `L` that are strictly below the
/// diagonal.
///
/// Does not allocate. It is assumed that the arrays `work`, `l_nz`, and
/// `etree` will be allocated with a number of elements equal to `n`.
///
/// The data in (`n`,`a_p`,`a_i`) are from a square matrix `A` in CSC format, and
/// should include the upper triangular part of `A` only.
///
/// This function is only intended for factorisation of QD matrices specified
/// by their upper triangular part.  An error is returned if any column has
/// data below the diagonal or is completely empty.
///
/// For matrices with a non-empty column but a zero on the corresponding diagonal,
/// this function will *not* return an error, as it may still be possible to factor
/// such a matrix in LDL form.  No promises are made in this case though...
///
/// # Arguments
///
/// * `n` - number of columns in CSC matrix `A` (assumed square)
/// * `a_p` - column pointers (size `n`+1) for columns of `A`
/// * `a_i` - row indices of `A`.  Has `a_p[n]` elements
/// * `work` - work vector (size `n`) (no meaning on return)
/// * `l_nz` - count of nonzeros in each column of `L` (size `n`) below diagonal
/// * `etree` - elimination tree (size `n`)
///
/// # Returns
///
/// Sum of `Lnz` (i.e. total nonzeros in `L` below diagonal).
/// Returns -1 if the input is not triu or has an empty column.
/// Returns -2 if the return value overflows.
pub fn etree(
    n: usize,
    a_p: &[usize],
    a_i: &[usize],
    work: &mut [usize],
    l_nz: &mut [usize],
    etree: &mut [isize],
) -> Result<usize, isize> {
    for i in 0..n {
        // Zero out Lnz and work. Set all etree values to unknown.
        work[i as usize] = 0;
        l_nz[i as usize] = 0;
        etree[i as usize] = UNKNOWN;

        // Abort if A doesn't have at least one entry
        // one entry in every column.
        if a_p[i as usize] == a_p[i as usize + 1] {
            return Err(-1);
        }
    }

    for j in 0..n {
        work[j as usize] = j;
        for p in a_p[j as usize]..a_p[j as usize + 1] {
            let mut i = a_i[p as usize];
            if i > j {
                // Abort if entries on lower triangle.
                return Err(-1);
            }
            while work[i as usize] != j {
                if etree[i as usize] == UNKNOWN {
                    etree[i as usize] = j as isize;
                }
                l_nz[i as usize] += 1; // nonzeros in this column
                work[i as usize] = j;
                i = etree[i as usize] as usize;
            }
        }
    }

    // Compute the total nonzeros in L.  This much
    // space is required to store Li and Lx.  Return
    // error code -2 if the nonzero count will overflow
    // its integer type.
    let mut sum_l_nz: usize = 0;
    for i in 0..n {
        if sum_l_nz > usize::MAX - l_nz[i as usize] {
            return Err(-2);
        } else {
            sum_l_nz += l_nz[i as usize];
        }
    }

    Ok(sum_l_nz)
}

/// Compute an LDL decomposition for a quasidefinite matrix
/// in compressed sparse column form, where the input matrix is
/// assumed to contain data for the upper triangular part of `A` only,
/// and there are no duplicate indices.
///
/// Returns factors `L`, `D` and `Dinv = 1./D`.
///
/// Does not allocate. It is assumed that `L` will be a compressed
/// sparse column matrix with data (`n`,`l_p`,`l_i`,`l_x`) with sufficient space
/// allocated, with a number of nonzeros equal to the count given
/// as a return value by `etree`.
///
/// # Arguments
///
/// * `n` - number of columns in `L` and `A` (both square)
/// * `a_p` - column pointers (size `n`+1) for columns of `A` (not modified)
/// * `a_i` - row indices of `A`.  Has `a_p[n]` elements (not modified)
/// * `a_x` - data of `A`.  Has `a_p[n]` elements (not modified)
/// * `l_p` - column pointers (size `n`+1) for columns of `L`
/// * `l_i` - row indices of `L`.  Has `l_p[n]` elements
/// * `l_x` - data of `L`.  Has `l_p[n]` elements
/// * `d` - vectorized factor `D`.  Length is `n`
/// * `d_inv` - reciprocal of `D`.  Length is `n`
/// * `l_nz` - count of nonzeros in each column of `L` below diagonal,
///   as given by `etree()` (not modified)
/// * `etree` - elimination tree as as given by `etree()` (not modified)
/// * `bwork` - working array of bools. Length is `n`
/// * `iwork` - working array of integers. Length is 3*`n`
/// * `fwork` - working array of floats. Length is `n`
///
/// # Returns
///
/// Returns a count of the number of positive elements in `D`.  
/// Returns -1 and exits immediately if any element of `D` evaluates
/// exactly to zero (matrix is not quasidefinite or otherwise LDL factorisable)
pub fn factor<S: Float + NumAssignOps>(
    n: usize,
    a_p: &[usize],
    a_i: &[usize],
    a_x: &[S],
    l_p: &mut [usize],
    l_i: &mut [usize],
    l_x: &mut [S],
    d: &mut [S],
    d_inv: &mut [S],
    l_nz: &[usize],
    etree: &[isize],
    bwork: &mut [bool],
    iwork: &mut [usize],
    fwork: &mut [S],
) -> Result<usize, isize> {
    let mut nnz_y: usize;
    let mut bidx: usize;
    let mut cidx: usize;
    let mut next_idx: isize;
    let mut nnz_e: usize;
    let mut tmp_idx: usize;

    let mut positive_values_in_d: usize = 0;

    // Partition working memory into pieces.
    let y_markers = bwork;
    let (y_idx, iwork) = iwork.split_at_mut(n as usize);
    let (elim_buffer, iwork) = iwork.split_at_mut(n as usize);
    let (l_next_space_in_col, _) = iwork.split_at_mut(n as usize);
    let y_vals = fwork;

    l_p[0] = 0; // first column starts at index zero

    for i in 0..n as usize {
        // Compute L column indices.
        l_p[i + 1] = l_p[i] + l_nz[i]; // cumsum, total at the end

        // Set all Yidx to be 'unused' initially
        // in each column of L, the next available space
        // to start is just the first space in the column
        y_markers[i] = UNUSED;
        y_vals[i] = S::zero();
        d[i] = S::zero();
        l_next_space_in_col[i] = l_p[i];
    }

    // First element of the diagonal D.
    d[0] = a_x[0];
    if d[0] == S::zero() {
        return Err(-1);
    }
    if d[0] > S::zero() {
        positive_values_in_d += 1;
    }
    d_inv[0] = S::one() / d[0];

    // Start from 1 here. The upper LH corner is trivially 0
    // in L b/c we are only computing the subdiagonal elements.
    for k in 1..n {
        // NB : For each k, we compute a solution to
        // y = L(0:(k-1),0:k-1))\b, where b is the kth
        // column of A that sits above the diagonal.
        // The solution y is then the kth row of L,
        // with an implied '1' at the diagonal entry.

        // Number of nonzeros in this row of L.
        nnz_y = 0; // Number of elements in this row.

        // This loop determines where nonzeros
        // will go in the kth row of L, but doesn't
        // compute the actual values.
        tmp_idx = a_p[k as usize + 1];

        for i in a_p[k as usize]..tmp_idx {
            bidx = a_i[i as usize]; // We are working on this element of b.

            // Initialize D[k] as the element of this column
            // corresponding to the diagonal place. Don't use
            // this element as part of the elimination step
            // that computes the k^th row of L.
            if bidx == k {
                d[k as usize] = a_x[i as usize];
                continue;
            }

            y_vals[bidx as usize] = a_x[i as usize]; // initialise y(bidx) = b(bidx)

            // Use the forward elimination tree to figure
            // out which elements must be eliminated after
            // this element of b.
            next_idx = bidx as isize;

            if y_markers[next_idx as usize] == UNUSED {
                // This y term not already visited.

                y_markers[next_idx as usize] = USED; // I touched this one.
                elim_buffer[0] = next_idx as usize; // It goes at the start of the current list.
                nnz_e = 1; // Length of unvisited elimination path from here.

                next_idx = etree[bidx as usize];

                while next_idx != UNKNOWN && (next_idx as usize) < k {
                    if y_markers[next_idx as usize] == USED {
                        break;
                    }

                    y_markers[next_idx as usize] = USED; // I touched this one.
                    elim_buffer[nnz_e as usize] = next_idx as usize; // It goes in the current list.
                    nnz_e += 1; // The list is one longer than before.
                    next_idx = etree[next_idx as usize]; // One step further along tree.
                }

                // Now I put the buffered elimination list into
                // my current ordering in reverse order.
                while nnz_e != 0 {
                    // yIdx[nnzY++] = elim_buffer[--nnzE];
                    nnz_e -= 1;
                    y_idx[nnz_y as usize] = elim_buffer[nnz_e as usize];
                    nnz_y += 1;
                }
            }
        }

        // This for loop places nonzeros values in the k^th row.
        let mut i: isize = nnz_y as isize - 1;
        while i >= 0 {
            // for i in (0..=(nnz_y - 1)).rev() {
            // for(i = (nnzY-1); i >=0; i--){

            // Which column are we working on?
            cidx = y_idx[i as usize];

            // Loop along the elements in this
            // column of L and subtract to solve to y.
            tmp_idx = l_next_space_in_col[cidx as usize];
            let y_vals_cidx = y_vals[cidx as usize];
            for j in l_p[cidx as usize]..tmp_idx {
                y_vals[l_i[j as usize] as usize] -= l_x[j as usize] * y_vals_cidx;
            }

            // Now I have the cidx^th element of y = L\b.
            // so compute the corresponding element of
            // this row of L and put it into the right place.
            l_i[tmp_idx as usize] = k;
            l_x[tmp_idx as usize] = y_vals_cidx * d_inv[cidx as usize];

            // D[k] -= yVals[cidx]*yVals[cidx]*Dinv[cidx];
            d[k as usize] -= y_vals_cidx * l_x[tmp_idx as usize];
            l_next_space_in_col[cidx as usize] += 1;

            // Reset the yvalues and indices back to zero and UNUSED
            // once I'm done with them.
            y_vals[cidx as usize] = S::zero();
            y_markers[cidx as usize] = UNUSED;

            i -= 1;
        }

        // Maintain a count of the positive entries
        // in D.  If we hit a zero, we can't factor
        // this matrix, so abort
        if d[k as usize] == S::zero() {
            return Err(-1);
        }
        if d[k as usize] > S::zero() {
            positive_values_in_d += 1;
        }

        // Compute the inverse of the diagonal.
        d_inv[k as usize] = S::one() / d[k as usize];
    }

    Ok(positive_values_in_d)
}

/// Solves `LDL'x = b`
///
/// It is assumed that `L` will be a compressed
/// sparse column matrix with data (`n`,`l_p`,`l_i`,`l_x`).
///
/// # Arguments
///
/// * `n` - number of columns in `L`
/// * `l_p` - column pointers (size `n`+1) for columns of `L`
/// * `l_i` - row indices of `L`.  Has `l_p[n]` elements
/// * `l_x` - data of `L`.  Has `l_p[n]` elements
/// * `d_inv` - reciprocal of `D`.  Length is `n`
/// * `x` - initialized to `b`.  Equal to `x` on return
pub fn solve<S: Float + NumAssignOps>(
    n: usize,
    l_p: &[usize],
    l_i: &[usize],
    l_x: &[S],
    d_inv: &[S],
    x: &mut [S],
) {
    lsolve(n, l_p, l_i, l_x, x);
    for i in 0..n {
        x[i as usize] *= d_inv[i as usize];
    }
    ltsolve(n, l_p, l_i, l_x, x);
}

/// Solves `(L+I)x = b`
///  
/// It is assumed that `L` will be a compressed
/// sparse column matrix with data (`n`,`l_p`,`l_i`,`l_x`).
///
/// # Arguments
///
/// * `n` - number of columns in `L`
/// * `l_p` - column pointers (size `n`+1) for columns of `L`
/// * `l_i` - row indices of `L`.  Has `l_p[n]` elements
/// * `l_x` - data of `L`.  Has `l_p[n]` elements
/// * `x` - initialized to `b`.  Equal to `x` on return
pub fn lsolve<S: Float + NumAssignOps>(
    n: usize,
    l_p: &[usize],
    l_i: &[usize],
    l_x: &[S],
    x: &mut [S],
) {
    for i in 0..n {
        let val = x[i as usize];
        for j in l_p[i as usize]..l_p[i as usize + 1] {
            x[l_i[j as usize] as usize] -= l_x[j as usize] * val;
        }
    }
}

/// Solves `(L+I)'x = b`
///
/// It is assumed that `L` will be a compressed
/// sparse column matrix with data (`n`,`l_p`,`l_i`,`l_x`).
///
/// # Arguments
///
/// * `n` - number of columns in `L`
/// * `l_p` - column pointers (size n+1) for columns of `L`
/// * `l_i` - row indices of `L`.  Has `l_p[n]` elements
/// * `l_x` - data of `L`.  Has `l_p[n]` elements
/// * `x` - initialized to `b`.  Equal to `x` on return
pub fn ltsolve<S: Float + NumAssignOps>(
    n: usize,
    l_p: &[usize],
    l_i: &[usize],
    l_x: &[S],
    x: &mut [S],
) {
    for i in (0..=n - 1).rev() {
        //for(i = n-1; i>=0; i--){
        let mut val = x[i as usize];
        for j in l_p[i as usize]..l_p[i as usize + 1] {
            val -= l_x[j as usize] * x[l_i[j as usize] as usize];
        }
        x[i as usize] = val;
    }
}

pub fn factor_solve<S: Float + NumAssignOps>(
    a_n: usize,
    a_p: &[usize],
    a_i: &[usize],
    a_x: &[S],
    b: &mut [S],
) -> Result<(), isize> {
    let l_n = a_n;

    // Pre-factorisation memory allocations //

    // These can happen *before* the etree is calculated
    // since the sizes are not sparsity pattern specific.

    // For the elimination tree.
    let mut etree_: Vec<isize> = vec![0; a_n as usize];
    let mut l_nz: Vec<usize> = vec![0; a_n as usize];

    // For the L factors. Li and Lx are sparsity dependent
    // so must be done after the etree is constructed.
    let mut l_p: Vec<usize> = vec![0; a_n as usize + 1];
    let mut d: Vec<S> = vec![S::zero(); a_n as usize];
    let mut d_inv: Vec<S> = vec![S::zero(); a_n as usize];

    // Working memory. Note that both the etree and factor
    // calls requires a working vector of int, with
    // the factor function requiring 3*An elements and the
    // etree only An elements. Just allocate the larger
    // amount here and use it in both places.
    let mut iwork: Vec<usize> = vec![0; 3 * a_n as usize];
    let mut bwork: Vec<bool> = vec![false; a_n as usize];
    let mut fwork: Vec<S> = vec![S::zero(); a_n as usize];

    // Elimination tree calculation //

    let sum_l_nz = etree(a_n, &a_p, &a_i, &mut iwork, &mut l_nz, &mut etree_)?;

    // LDL factorisation //

    let mut l_i = vec![0; sum_l_nz as usize];
    let mut l_x: Vec<S> = vec![S::zero(); sum_l_nz as usize];

    factor(
        a_n, &a_p, &a_i, &a_x, &mut l_p, &mut l_i, &mut l_x, &mut d, &mut d_inv, &mut l_nz,
        &etree_, &mut bwork, &mut iwork, &mut fwork,
    )?;

    // Solve //

    solve(l_n, &l_p, &l_i, &l_x, &d_inv, b);

    Ok(())
}
