use crate::mat::{Matrix, MatrixData};
use core::panic;
use model::{AFloat, FromPrimitive, ToPrimitive};
use rug::Integer;
use stream::bigvec::BigVec;

// convert from f64 matrix to i64 integer (fixed point)
// scaled by scale factor and clipped below that
// col_shift is an optional column shift to apply to all non-zero column indices,
// used to shift a1 column values to the correct secondary witness range
impl Matrix<i64> {
    pub fn f64_to_i64(
        matrix: &Matrix<f64>,
        scale_factor: AFloat,
        col_shift: Option<usize>,
    ) -> Self {
        let col_shift = col_shift.unwrap_or(0);
        let data = match &matrix.data() {
            MatrixData::Dense(values) => {
                if col_shift > 0 {
                    panic!("Column shifts only supported for COO matrices");
                }
                let mut scaled_values = BigVec::new(values.len()).unwrap();
                for (i, &v) in values.iter().enumerate() {
                    let tmp = AFloat::from_f64(v).unwrap() * scale_factor.clone();
                    let tmp_f64 = tmp.to_f64().unwrap();
                    if tmp_f64 < i64::MIN as f64 || tmp_f64 > i64::MAX as f64 || tmp_f64.is_nan() {
                        panic!("Value out of bounds for i64: {:?}", tmp);
                    }
                    scaled_values[i] = tmp.to_i64().unwrap();
                }
                MatrixData::Dense(scaled_values)
            }
            MatrixData::COO(entries) => {
                let mut scaled_entries = BigVec::new(entries.len()).unwrap();
                for (i, &(r, c, v)) in entries.iter().enumerate() {
                    let tmp = AFloat::from_f64(v).unwrap() * scale_factor.clone();
                    let tmp_f64 = tmp.to_f64().unwrap();
                    if tmp_f64 < i64::MIN as f64 || tmp_f64 > i64::MAX as f64 || tmp_f64.is_nan() {
                        panic!("Value out of bounds for i64: {:?}", tmp);
                    }
                    let shifted_c = c
                        .checked_add(col_shift)
                        .expect("Column index overflowing when applying shift");
                    scaled_entries[i] = (r, shifted_c, tmp.to_i64().unwrap());
                }
                MatrixData::COO(scaled_entries)
            }
        };
        Matrix::new(
            data,
            matrix.width(),
            matrix.height(),
            matrix.ranges().cloned(),
            matrix.comment().to_string(),
        )
    }
    pub fn f64_index_to_i64(matrix: &Matrix<f64>, table: &Matrix<i64>, shift: usize) -> Self {
        let delta = table.width();
        let data = match &matrix.data() {
            MatrixData::COO(entries) => {
                let table = match table.data() {
                    MatrixData::Dense(values) => values,
                    _ => panic!("Expected Dense matrix for table"),
                };
                let mut scaled_entries = BigVec::new(entries.len()).unwrap();
                for (i, &(r, c, v)) in entries.iter().enumerate() {
                    let idx = delta * (v as usize + shift);
                    // if v is not an integer, panic
                    if v.fract() != 0.0 {
                        panic!("Value is not an integer: {}", v);
                    }
                    if idx >= table.len() {
                        panic!("Index out of bounds: {}", idx);
                    }
                    scaled_entries[i] = (r, c, table[idx]);
                }
                MatrixData::COO(scaled_entries)
            }
            _ => panic!("Expected COO matrix for f64_index_to_i64 conversion"),
        };
        Matrix::new(
            data,
            matrix.width(),
            matrix.height(),
            matrix.ranges().cloned(),
            matrix.comment().to_string(),
        )
    }
    pub fn print_column(&self, i: usize) {
        // print the ith column of the matrix
        assert!(i < self.width(), "Column index out of bounds");
        match &self.data() {
            MatrixData::Dense(values) => {
                for j in 0..self.height() {
                    let val = values[j * self.width() + i];
                    println!("Column {} Row {}: {}", i, j, val);
                }
            }
            MatrixData::COO(entries) => {
                for (r, c, val) in entries.iter() {
                    if *c == i {
                        println!("Column {} Row {}: {}", i, r, val);
                    }
                }
            }
        }
    }
    pub fn mul_iter<'a>(a: &'a Matrix<i64>, b: &'a Matrix<i64>) -> SparseDenseMulIter<'a> {
        // construct
        assert_eq!(
            a.width(),
            b.height(),
            "Incompatible matrix dimensions for multiplication"
        );
        SparseDenseMulIter::new(vec![a], b)
    }
    pub fn to_mle_evals(&self, shift: i64) -> (Vec<Integer>, usize, usize) {
        let h = self.height();
        let w = self.width();
        assert!(h > 0 && w > 0, "matrix cannot be empty ");
        let num_row_vars = h.next_power_of_two().trailing_zeros() as usize;
        let num_col_vars = w.next_power_of_two().trailing_zeros() as usize;
        let num_vars = num_row_vars + num_col_vars;
        let num_evals = 1 << num_vars;

        let mut evals = vec![Integer::from(shift); num_evals];

        match &self.data() {
            MatrixData::Dense(values) => {
                for r in 0..h {
                    for c in 0..w {
                        let v = values[r * w + c];
                        let idx = (r << num_col_vars) | c;
                        evals[idx] = Integer::from(v) + Integer::from(shift);
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

pub struct SparseDenseMulIter<'a> {
    a_tensor: Vec<&'a BigVec<(usize, usize, i64)>>,
    b: &'a BigVec<i64>,
    row: usize,   // row in a
    col: usize,   // col in b
    width: usize, // width of b
    idx: usize,
}

impl<'a> SparseDenseMulIter<'a> {
    pub fn new(a_tensor: Vec<&'a Matrix<i64>>, b: &'a Matrix<i64>) -> Self {
        assert!(
            !a_tensor.is_empty(),
            "a tensor must have at least one matrix"
        );
        if a_tensor.len() > 1 {
            assert_eq!(
                a_tensor.len(),
                b.width(),
                "tensor depth must match b's width"
            );
        }
        assert_eq!(
            a_tensor[0].width(),
            b.height(),
            "Incompatible matrix dimensions for multiplication"
        );
        let width = b.width();
        let a_tensor = a_tensor
            .iter()
            .map(|a| match a.data() {
                MatrixData::COO(entries) => entries,
                _ => panic!("Expected COO matrix for sparse multiplication"),
            })
            .collect();
        let b = match b.data() {
            MatrixData::Dense(values) => values,
            _ => panic!("Expected Dense matrix for sparse multiplication"),
        };
        SparseDenseMulIter {
            a_tensor,
            b,
            row: 0,
            col: 0,
            width,
            idx: 0,
        }
    }
}

impl<'a> Iterator for SparseDenseMulIter<'a> {
    type Item = i128;

    fn next(&mut self) -> Option<Self::Item> {
        let a = if self.a_tensor.len() == 1 {
            self.a_tensor[0]
        } else {
            self.a_tensor[self.col]
        };
        if self.idx >= a.len() {
            return None;
        }
        let mut res: i128 = 0;
        let mut tmp_idx = self.idx;
        while tmp_idx < a.len() && a[tmp_idx].0 == self.row {
            let (_, c, val) = a[tmp_idx];
            res += val as i128 * self.b[self.col + c * self.width] as i128;
            tmp_idx += 1;
        }
        self.col += 1;
        if self.col >= self.width {
            self.col = 0;
            self.row += 1;
            self.idx = tmp_idx; // update idx to the next row
        }
        Some(res)
    }
}
