extern crate num_traits;

#[cfg(test)]
mod tests;

use num_traits::{Bounded, Float, FromPrimitive, NumAssignOps, PrimInt, Signed, ToPrimitive};
use std::clone::Clone;

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
pub fn etree<
    I: PrimInt + NumAssignOps + Bounded + ToPrimitive + FromPrimitive,
    S: Signed + ToPrimitive + FromPrimitive,
>(
    n: I,
    a_p: &[I],
    a_i: &[I],
    work: &mut [I],
    l_nz: &mut [I],
    etree: &mut [S],
) -> Result<I, S> {
    for i in 0..n.to_usize().unwrap() {
        // Zero out Lnz and work. Set all etree values to unknown.
        work[i] = I::zero();
        l_nz[i] = I::zero();
        etree[i] = S::from_isize(UNKNOWN).unwrap();

        // Abort if A doesn't have at least one entry
        // one entry in every column.
        if a_p[i] == a_p[i + 1] {
            return Err(-S::one());
        }
    }

    for j in 0..n.to_usize().unwrap() {
        work[j] = I::from_usize(j).unwrap();
        for p in a_p[j].to_usize().unwrap()..a_p[j + 1].to_usize().unwrap() {
            let mut i = a_i[p].to_usize().unwrap();
            if i > j {
                // Abort if entries on lower triangle.
                return Err(-S::one());
            }
            while work[i].to_usize().unwrap() != j {
                if etree[i] == S::from_isize(UNKNOWN).unwrap() {
                    etree[i] = S::from_usize(j).unwrap();
                }
                l_nz[i] += I::one(); // nonzeros in this column
                work[i] = I::from_usize(j).unwrap();
                i = etree[i].to_usize().unwrap();
            }
        }
    }

    // Compute the total nonzeros in L.  This much
    // space is required to store Li and Lx.  Return
    // error code -2 if the nonzero count will overflow
    // its integer type.
    let mut sum_l_nz = I::zero();
    for i in 0..n.to_usize().unwrap() {
        if sum_l_nz > I::max_value() - l_nz[i] {
            return Err(S::from_i32(-2).unwrap());
        } else {
            sum_l_nz += l_nz[i];
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
pub fn factor<
    F: Float + NumAssignOps,
    I: PrimInt + NumAssignOps + FromPrimitive + ToPrimitive,
    S: Signed + FromPrimitive + ToPrimitive,
>(
    n: I,
    a_p: &[I],
    a_i: &[I],
    a_x: &[F],
    l_p: &mut [I],
    l_i: &mut [I],
    l_x: &mut [F],
    d: &mut [F],
    d_inv: &mut [F],
    l_nz: &[I],
    etree: &[S],
    bwork: &mut [bool],
    iwork: &mut [I],
    fwork: &mut [F],
) -> Result<I, S> {
    let un = n.to_usize().unwrap();

    let mut nnz_y: usize;
    let mut bidx: usize;
    let mut cidx: usize;
    let mut next_idx: isize;
    let mut nnz_e: usize;
    let mut tmp_idx: usize;

    let mut positive_values_in_d = I::zero();

    // Partition working memory into pieces.
    let y_markers = bwork;
    let (y_idx, iwork) = iwork.split_at_mut(un);
    let (elim_buffer, iwork) = iwork.split_at_mut(un);
    let (l_next_space_in_col, _) = iwork.split_at_mut(un);
    let y_vals = fwork;

    l_p[0] = I::zero(); // first column starts at index zero

    for i in 0..un {
        // Compute L column indices.
        l_p[i + 1] = l_p[i] + l_nz[i]; // cumsum, total at the end

        // Set all Yidx to be 'unused' initially
        // in each column of L, the next available space
        // to start is just the first space in the column
        y_markers[i] = UNUSED;
        y_vals[i] = F::zero();
        d[i] = F::zero();
        l_next_space_in_col[i] = l_p[i];
    }

    // First element of the diagonal D.
    d[0] = a_x[0];
    if d[0] == F::zero() {
        return Err(-S::one());
    }
    if d[0] > F::zero() {
        positive_values_in_d += I::one();
    }
    d_inv[0] = F::one() / d[0];

    // Start from 1 here. The upper LH corner is trivially 0
    // in L b/c we are only computing the subdiagonal elements.
    for k in 1..un {
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
        tmp_idx = a_p[k + 1].to_usize().unwrap();

        for i in a_p[k].to_usize().unwrap()..tmp_idx {
            bidx = a_i[i].to_usize().unwrap(); // We are working on this element of b.

            // Initialize D[k] as the element of this column
            // corresponding to the diagonal place. Don't use
            // this element as part of the elimination step
            // that computes the k^th row of L.
            if bidx == k {
                d[k] = a_x[i];
                continue;
            }

            y_vals[bidx] = a_x[i]; // initialise y(bidx) = b(bidx)

            // Use the forward elimination tree to figure
            // out which elements must be eliminated after
            // this element of b.
            next_idx = bidx as isize;

            if y_markers[next_idx as usize] == UNUSED {
                // This y term not already visited.

                y_markers[next_idx as usize] = USED; // I touched this one.
                elim_buffer[0] = I::from_isize(next_idx).unwrap(); // It goes at the start of the current list.
                nnz_e = 1; // Length of unvisited elimination path from here.

                next_idx = etree[bidx].to_isize().unwrap();

                while next_idx != UNKNOWN && (next_idx as usize) < k {
                    if y_markers[next_idx as usize] == USED {
                        break;
                    }

                    y_markers[next_idx as usize] = USED; // I touched this one.
                    elim_buffer[nnz_e] = I::from_usize(next_idx as usize).unwrap(); // It goes in the current list.
                    nnz_e += 1; // The list is one longer than before.
                    next_idx = etree[next_idx as usize].to_isize().unwrap(); // One step further along tree.
                }

                // Now I put the buffered elimination list into
                // my current ordering in reverse order.
                while nnz_e != 0 {
                    // yIdx[nnzY++] = elim_buffer[--nnzE];
                    nnz_e -= 1;
                    y_idx[nnz_y] = elim_buffer[nnz_e];
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
            cidx = y_idx[i as usize].to_usize().unwrap();

            // Loop along the elements in this
            // column of L and subtract to solve to y.
            tmp_idx = l_next_space_in_col[cidx].to_usize().unwrap();
            let y_vals_cidx = y_vals[cidx];
            for j in l_p[cidx].to_usize().unwrap()..tmp_idx {
                y_vals[l_i[j].to_usize().unwrap()] -= l_x[j] * y_vals_cidx;
            }

            // Now I have the cidx^th element of y = L\b.
            // so compute the corresponding element of
            // this row of L and put it into the right place.
            l_i[tmp_idx] = I::from_usize(k).unwrap();
            l_x[tmp_idx] = y_vals_cidx * d_inv[cidx];

            // D[k] -= yVals[cidx]*yVals[cidx]*Dinv[cidx];
            d[k] -= y_vals_cidx * l_x[tmp_idx];
            l_next_space_in_col[cidx] += I::one();

            // Reset the yvalues and indices back to zero and UNUSED
            // once I'm done with them.
            y_vals[cidx] = F::zero();
            y_markers[cidx] = UNUSED;

            i -= 1;
        }

        // Maintain a count of the positive entries
        // in D.  If we hit a zero, we can't factor
        // this matrix, so abort
        if d[k] == F::zero() {
            return Err(-S::one());
        }
        if d[k] > F::zero() {
            positive_values_in_d += I::one();
        }

        // Compute the inverse of the diagonal.
        d_inv[k] = F::one() / d[k];
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
pub fn solve<F: Float + NumAssignOps, I: PrimInt>(
    n: I,
    l_p: &[I],
    l_i: &[I],
    l_x: &[F],
    d_inv: &[F],
    x: &mut [F],
) {
    lsolve(n, l_p, l_i, l_x, x);
    for i in 0..n.to_usize().unwrap() {
        x[i] *= d_inv[i];
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
pub fn lsolve<F: Float + NumAssignOps, I: PrimInt>(
    n: I,
    l_p: &[I],
    l_i: &[I],
    l_x: &[F],
    x: &mut [F],
) {
    for i in 0..n.to_usize().unwrap() {
        let val = x[i];
        for j in l_p[i].to_usize().unwrap()..l_p[i + 1].to_usize().unwrap() {
            x[l_i[j].to_usize().unwrap()] -= l_x[j] * val;
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
pub fn ltsolve<F: Float + NumAssignOps, I: PrimInt>(
    n: I,
    l_p: &[I],
    l_i: &[I],
    l_x: &[F],
    x: &mut [F],
) {
    for i in (0..=n.to_usize().unwrap() - 1).rev() {
        //for(i = n-1; i>=0; i--){
        let mut val = x[i];
        for j in l_p[i].to_usize().unwrap()..l_p[i + 1].to_usize().unwrap() {
            val -= l_x[j] * x[l_i[j].to_usize().unwrap()];
        }
        x[i] = val;
    }
}

pub fn factor_solve<
    F: Float + NumAssignOps,
    I: PrimInt + NumAssignOps + FromPrimitive + ToPrimitive + Clone,
    S: Signed + FromPrimitive + ToPrimitive + Clone,
>(
    a_n: I,
    a_p: &[I],
    a_i: &[I],
    a_x: &[F],
    b: &mut [F],
) -> Result<(), S> {
    let un = a_n.to_usize().unwrap();

    let l_n = a_n;

    // Pre-factorisation memory allocations //

    // These can happen *before* the etree is calculated
    // since the sizes are not sparsity pattern specific.

    // For the elimination tree.
    let mut etree_: Vec<S> = vec![S::zero(); un];
    let mut l_nz: Vec<I> = vec![I::zero(); un];

    // For the L factors. Li and Lx are sparsity dependent
    // so must be done after the etree is constructed.
    let mut l_p: Vec<I> = vec![I::zero(); un + 1];
    let mut d: Vec<F> = vec![F::zero(); un];
    let mut d_inv: Vec<F> = vec![F::zero(); un];

    // Working memory. Note that both the etree and factor
    // calls requires a working vector of int, with
    // the factor function requiring 3*An elements and the
    // etree only An elements. Just allocate the larger
    // amount here and use it in both places.
    let mut iwork: Vec<I> = vec![I::zero(); 3 * un];
    let mut bwork: Vec<bool> = vec![false; un];
    let mut fwork: Vec<F> = vec![F::zero(); un];

    // Elimination tree calculation //

    let sum_l_nz = etree(a_n, a_p, a_i, &mut iwork, &mut l_nz, &mut etree_)?;

    // LDL factorisation //

    let mut l_i: Vec<I> = vec![I::zero(); sum_l_nz.to_usize().unwrap()];
    let mut l_x: Vec<F> = vec![F::zero(); sum_l_nz.to_usize().unwrap()];

    factor(
        a_n, a_p, a_i, a_x, &mut l_p, &mut l_i, &mut l_x, &mut d, &mut d_inv, &l_nz, &etree_,
        &mut bwork, &mut iwork, &mut fwork,
    )?;

    // Solve //

    solve(l_n, &l_p, &l_i, &l_x, &d_inv, b);

    Ok(())
}
