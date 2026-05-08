use core::panic;
use std::{collections::HashSet, ops::Range};

use ff::poly::cmont::MLE;
use ff::{FieldElem, FieldMont, i64_to_mont, i128_to_mont};
use log::debug;
use stream::bigvec::BigVec;

use crate::mat::{Matrix, MatrixData};

impl Matrix<FieldElem> {
    fn from_dense_slice(slice: &[FieldElem], transpose: bool) -> Self {
        let mut height = 1;
        let mut width = slice.len();
        if transpose {
            std::mem::swap(&mut width, &mut height);
        }
        let data = MatrixData::Dense(BigVec::from_vec(slice.to_vec()));
        Matrix::new(data, width, height, None, "".to_string())
    }
    // kronecker prod_i: [1-r_i, r_i], separated by split points
    pub fn trunc_kronecker_prod_split(
        r: &[FieldElem],
        splits: &[usize],
        transpose: bool,
        mont: &FieldMont,
    ) -> Vec<Matrix<FieldElem>> {
        // compute full kronecker product
        let mut data = vec![mont.one()];
        for &ri in r.iter().rev() {
            let mut new_data = Vec::with_capacity(data.len() * 2);
            let omr = mont.sub(mont.one(), ri);
            for &v in &data {
                new_data.push(mont.mul(v, omr));
                new_data.push(mont.mul(v, ri));
            }
            data = new_data;
        }

        let mut result = Vec::new();
        let mut start = 0;
        // Separate into subarrays based on split points
        for &end in splits {
            result.push(Self::from_dense_slice(&data[start..end], transpose));
            start = end;
        }
        result
    }

    pub fn trunc_kronecker_prod(
        r: &[FieldElem],
        len: usize,
        transpose: bool,
        mont: &FieldMont,
    ) -> Self {
        let mut mats = Self::trunc_kronecker_prod_split(r, &[len], transpose, mont);
        mats.remove(0)
    }
    pub fn i64_rat_to_m64(matrix: &Matrix<i64>, den: FieldElem, mont: &FieldMont) -> Self {
        // Convert the rational matrix to a FieldMontgomery matrix
        let data = match &matrix.data() {
            MatrixData::Dense(values) => MatrixData::Dense({
                // Note this can be parallelized
                let mut new_values = BigVec::new(values.len()).unwrap();
                for (v, w) in values.iter().zip(new_values.iter_mut()) {
                    *w = mont.mul(i64_to_mont(v, mont), den);
                }
                new_values
            }),
            MatrixData::COO(entries) => MatrixData::COO({
                let mut new_entries = BigVec::new(entries.len()).unwrap();
                for ((r1, c1, v1), (r2, c2, v2)) in entries.iter().zip(new_entries.iter_mut()) {
                    *r2 = *r1; // copy row index
                    *c2 = *c1; // copy column index
                    *v2 = mont.mul(i64_to_mont(v1, mont), den);
                }
                new_entries
            }),
        };
        Matrix::new(
            data,
            matrix.width(),
            matrix.height(),
            matrix.ranges().cloned(),
            matrix.comment().to_string(),
        )
    }

    pub fn i128_rat_to_m64(matrix: &Matrix<i128>, den: FieldElem, mont: &FieldMont) -> Self {
        let data = match &matrix.data() {
            MatrixData::Dense(values) => MatrixData::Dense({
                let mut new_values = BigVec::new(values.len()).unwrap();
                for (v, w) in values.iter().zip(new_values.iter_mut()) {
                    *w = mont.mul(i128_to_mont(v, mont), den);
                }
                new_values
            }),
            MatrixData::COO(entries) => MatrixData::COO({
                let mut new_entries = BigVec::new(entries.len()).unwrap();
                for ((r1, c1, v1), (r2, c2, v2)) in entries.iter().zip(new_entries.iter_mut()) {
                    *r2 = *r1;
                    *c2 = *c1;
                    *v2 = mont.mul(i128_to_mont(v1, mont), den);
                }
                new_entries
            }),
        };
        Matrix::new(
            data,
            matrix.width(),
            matrix.height(),
            matrix.ranges().cloned(),
            matrix.comment().to_string(),
        )
    }

    // multiply matrices in FieldMontgomery form
    pub fn mont_mul(&self, b: &Matrix<FieldElem>, mont: &FieldMont) -> Matrix<FieldElem> {
        // Ensure matrices are compatible for multiplication
        assert_eq!(
            self.width(),
            b.height(),
            "Incompatible matrix dimensions for multiplication"
        );
        // Create a new dense matrix to hold the result
        let mut result_data = BigVec::new(self.height() * b.width()).unwrap();
        // specialize for all cases:
        match (&self.data(), &b.data()) {
            (MatrixData::Dense(a_values), MatrixData::Dense(b_values)) => {
                // Dense * Dense
                for i in 0..self.height() {
                    for j in 0..b.width() {
                        let mut sum = mont.zero();
                        for k in 0..self.width() {
                            sum = mont.add(
                                sum,
                                mont.mul(
                                    a_values[i * self.width() + k],
                                    b_values[k * b.width() + j],
                                ),
                            );
                        }
                        result_data[i * b.width() + j] = sum;
                    }
                }
            }
            (MatrixData::COO(a_entries), MatrixData::Dense(b_values)) => {
                // COO * Dense
                if b.width() > 1 {
                    for j in 0..b.width() {
                        for (r, c, val) in a_entries.iter() {
                            result_data[r * b.width() + j] = mont.add(
                                result_data[r * b.width() + j],
                                mont.mul(*val, b_values[c * b.width() + j]),
                            );
                        }
                    }
                } else {
                    // If b is a column vector, we can optimize
                    for (r, c, val) in a_entries.iter() {
                        result_data[*r] = mont.add(result_data[*r], mont.mul(*val, b_values[*c]));
                    }
                }
            }
            (MatrixData::Dense(a_values), MatrixData::COO(b_entries)) => {
                // Dense * COO
                if self.height() > 1 {
                    for i in 0..self.height() {
                        for (r, c, val) in b_entries.iter() {
                            result_data[i * b.width() + c] = mont.add(
                                result_data[i * b.width() + c],
                                mont.mul(a_values[i * self.width() + r], *val),
                            );
                        }
                    }
                } else {
                    // If height is 1, use optimized approach
                    for (r, c, val) in b_entries.iter() {
                        result_data[*c] = mont.add(result_data[*c], mont.mul(a_values[*r], *val));
                    }
                }
            }
            (MatrixData::COO(a_entries), MatrixData::COO(b_entries)) => {
                // COO * COO
                for (ar, ac, aval) in a_entries.iter() {
                    for (br, bc, bval) in b_entries.iter() {
                        if ac == br {
                            // only multiply if dimensions match
                            result_data[ar * b.width() + bc] =
                                mont.add(result_data[ar * b.width() + bc], mont.mul(*aval, *bval));
                        }
                    }
                }
            }
        }
        // Return the resulting matrix
        Matrix::new(
            MatrixData::Dense(result_data),
            b.width(),
            self.height(),
            None,
            "".to_string(),
        )
    }
    pub fn print_column(&self, i: usize, mont: &FieldMont) {
        // print the ith column of the matrix
        assert!(i < self.width(), "Column index out of bounds");
        match &self.data() {
            MatrixData::Dense(values) => {
                for j in 0..self.height() {
                    let val = values[j * self.width() + i];
                    println!("Column {} Row {}: {}", i, j, mont.to_normal(val));
                }
            }
            MatrixData::COO(entries) => {
                for (r, c, val) in entries.iter() {
                    if *c == i {
                        println!("Column {} Row {}: {}", i, r, mont.to_normal(*val));
                    }
                }
            }
        }
    }
    // compute v^T * M
    pub fn vec_mul(&self, v: &[FieldElem], mont: &FieldMont) -> Vec<FieldElem> {
        // Ensure the vector length matches the matrix height
        assert_eq!(
            self.height(),
            v.len(),
            "Matrix and vector dimensions do not match"
        );
        // Create a new vector to hold the result
        let mut result = vec![mont.zero(); self.width()];
        // Perform multiplication
        match &self.data() {
            MatrixData::Dense(values) => {
                for i in 0..self.height() {
                    for j in 0..self.width() {
                        result[j] =
                            mont.add(result[j], mont.mul(values[i * self.width() + j], v[i]));
                    }
                }
            }
            MatrixData::COO(entries) => {
                for (r, c, val) in entries.iter() {
                    if *r < self.height() {
                        result[*c] = mont.add(result[*c], mont.mul(*val, v[*r]));
                    }
                }
            }
        }
        result
    }
    // compute M * v
    pub fn mul_vec(&self, v: &[FieldElem], mont: &FieldMont) -> Vec<FieldElem> {
        // Ensure the vector length matches the matrix width
        assert_eq!(
            self.width(),
            v.len(),
            "Matrix and vector dimensions do not match"
        );
        // Create a new vector to hold the result
        let mut result = vec![mont.zero(); self.height()];
        // Perform multiplication
        match &self.data() {
            MatrixData::Dense(values) => {
                for i in 0..self.height() {
                    for j in 0..self.width() {
                        result[i] =
                            mont.add(result[i], mont.mul(values[i * self.width() + j], v[j]));
                    }
                }
            }
            MatrixData::COO(entries) => {
                for (r, c, val) in entries.iter() {
                    if *c < self.width() {
                        result[*r] = mont.add(result[*r], mont.mul(*val, v[*c]));
                    }
                }
            }
        }
        result
    }
    pub fn mul_scalar(&mut self, scale_factor: FieldElem, mont: &FieldMont) {
        // Scale the matrix by a given factor in FieldMontgomery form
        match self.mut_data() {
            MatrixData::Dense(values) => {
                for value in values.iter_mut() {
                    *value = mont.mul(*value, scale_factor);
                }
            }
            MatrixData::COO(entries) => {
                for (_, _, value) in entries.iter_mut() {
                    *value = mont.mul(*value, scale_factor);
                }
            }
        }
    }
    fn remap_ranges(&self, default_len: usize) -> (Vec<Range<usize>>, HashSet<usize>, usize) {
        #[allow(clippy::single_range_in_vec_init)]
        let default_range = vec![0..default_len];
        let ranges = self.ranges().unwrap_or(&default_range);

        let subrange_len = ranges
            .iter()
            .map(|r| r.len())
            .max()
            .unwrap_or(0)
            .next_power_of_two();

        let mut pad_indices = HashSet::new();

        let mut curr_end = 0;
        let remapped_ranges = ranges
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let start = i * subrange_len;
                let end = start + r.len();

                // Round the end up to the next even value if it's not already even
                let padded_end = if end % 2 == 0 { end } else { end + 1 };
                curr_end += padded_end - start;
                if padded_end > end {
                    pad_indices.insert(curr_end - 1);
                }

                start..padded_end
            })
            .collect::<Vec<_>>();

        (remapped_ranges, pad_indices, subrange_len)
    }
    pub fn to_mle(&self, mont: &FieldMont) -> MLE {
        if self.height() == 1 {
            match &self.data() {
                MatrixData::Dense(values) => {
                    let (ranges, pad_indices, _) = self.remap_ranges(self.width());
                    let num_evals = self.width() + pad_indices.len();
                    let mut evals = BigVec::<FieldElem>::new(num_evals).unwrap();
                    let mut value_iter = values.iter();
                    for (i, slot) in evals.iter_mut().enumerate() {
                        if pad_indices.contains(&i) {
                            *slot = mont.zero();
                        } else {
                            *slot = *value_iter.next().unwrap();
                        }
                    }
                    MLE::from_buffer(evals, ranges)
                }
                MatrixData::COO(_) => {
                    panic!("Requires several MLEs in the sparse case");
                }
            }
        } else if self.width() == 1 {
            match &self.data() {
                MatrixData::Dense(values) => {
                    // print length of values
                    debug!("Length of values: {}", values.len());
                    let (ranges, pad_indices, _) = self.remap_ranges(self.height());
                    let num_evals = self.height() + pad_indices.len();
                    let mut evals = BigVec::<FieldElem>::new(num_evals).unwrap();
                    let mut value_iter = values.iter();
                    for (i, slot) in evals.iter_mut().enumerate() {
                        if pad_indices.contains(&i) {
                            *slot = mont.zero();
                        } else {
                            *slot = *value_iter.next().unwrap();
                        }
                    }
                    MLE::from_buffer(evals, ranges)
                }
                MatrixData::COO(_) => {
                    panic!("Requires several MLEs in the sparse case");
                }
            }
        } else {
            match &self.data() {
                MatrixData::Dense(values) => {
                    let (col_ranges, pad_indices, col_stride) = self.remap_ranges(self.height());
                    let num_col_evals = self.height() + pad_indices.len();
                    let mut evals = BigVec::<FieldElem>::new(self.width() * num_col_evals).unwrap();
                    let mut ranges = Vec::with_capacity(self.width() * col_ranges.len());
                    for j in 0..self.width() {
                        let range_start = j * col_stride;
                        let start = j * num_col_evals;
                        for r in &col_ranges {
                            ranges.push(range_start + r.start..range_start + r.end);
                        }
                        for row in 0..num_col_evals {
                            let idx = start + row;
                            if pad_indices.contains(&row) {
                                evals[idx] = mont.zero();
                            } else {
                                let flat_idx = row * self.width() + j;
                                evals[idx] = values[flat_idx];
                            }
                        }
                    }
                    MLE::from_buffer(evals, ranges)
                }
                MatrixData::COO(_) => {
                    panic!("Requires several MLEs in the sparse case");
                }
            }
        }
    }
    pub fn to_mle_evals(
        &self,
        mont: &FieldMont,
        shift: &FieldElem,
    ) -> (BigVec<FieldElem>, usize, usize) {
        let h = self.height();
        let w = self.width();
        assert!(h > 0 && w > 0, "matrix cannot be empty ");
        let num_row_vars = h.next_power_of_two().trailing_zeros() as usize;
        let num_col_vars = w.next_power_of_two().trailing_zeros() as usize;
        let num_vars = num_row_vars + num_col_vars;
        let num_evals = 1 << num_vars;

        let mut evals = BigVec::new(num_evals).unwrap();
        for i in 0..num_evals {
            evals[i] = *shift;
        }

        match &self.data() {
            MatrixData::Dense(values) => {
                for r in 0..h {
                    for c in 0..w {
                        let idx = (r << num_col_vars) | c;
                        evals[idx] = mont.add(evals[idx], values[r * w + c]);
                    }
                }
            }
            MatrixData::COO(_) => {
                panic!("Only dense matrix (witness) can be turned into a set of MLE evals")
            }
        }
        (evals, num_row_vars, num_col_vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mat_ranges() {
        let ranges = vec![0..5, 5..10, 10..35];
        let ranges_len = ranges.len();
        let mont = FieldMont::new(13);
        let width = 1;
        let height = ranges.iter().map(|r| r.len()).sum::<usize>();
        let data = (0..height)
            .map(|i| mont.to_mont(i as u128))
            .collect::<Vec<_>>();
        let data = MatrixData::Dense(BigVec::from_vec(data));
        let matrix = Matrix::new(
            data.clone(),
            width,
            height,
            Some(ranges),
            "Test matrix".to_string(),
        );
        // Test that ranges expand correctly
        let (remapped_ranges, _, _) = matrix.remap_ranges(height);
        assert_eq!(remapped_ranges.len(), ranges_len);
        assert_eq!(remapped_ranges[0], 0..6);
        assert_eq!(remapped_ranges[1], 32..38);
        assert_eq!(remapped_ranges[2], 64..90);
        let _ = matrix.to_mle(&mont);
    }
}
