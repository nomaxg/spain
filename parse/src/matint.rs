use model::AFloat;
use model::{FromPrimitive, ToPrimitive};
use rug::Integer;

use crate::mat::{Matrix, MatrixData};

// convert from f64 matrix to rug integer (fixed point)
// scaled by 2^n for a parameter n and clipped below that
impl Matrix<Integer> {
    pub fn f64_to_integer(matrix: &Matrix<f64>, scale_factor: AFloat) -> Self {
        let data = match &matrix.data() {
            MatrixData::Dense(values) => {
                let scaled_values = values
                    .iter()
                    .map(|&v| {
                        let tmp = AFloat::from_f64(v).unwrap() * scale_factor.clone();
                        Integer::from_f64(tmp.to_f64().unwrap()).unwrap()
                    })
                    .collect();
                MatrixData::Dense(scaled_values)
            }
            MatrixData::COO(entries) => {
                let scaled_entries = entries
                    .iter()
                    .map(|&(r, c, v)| {
                        let tmp = AFloat::from_f64(v).unwrap() * scale_factor.clone();
                        (r, c, Integer::from_f64(tmp.to_f64().unwrap()).unwrap())
                    })
                    .collect();
                MatrixData::COO(scaled_entries)
            }
        };
        Matrix::new(
            data,
            matrix.width(),
            matrix.height(),
            matrix.ranges().clone(),
            matrix.comment().to_string(),
        )
    }
    // for debugging purposes
    pub fn get_max_magnitude(&self) -> Integer {
        // Get the maximum absolute value in the matrix
        match &self.data() {
            MatrixData::Dense(values) => values
                .iter()
                .map(|v| v.clone().abs())
                .max()
                .unwrap_or(Integer::default()),
            MatrixData::COO(entries) => entries
                .iter()
                .map(|(_, _, v)| v.clone().abs())
                .max()
                .unwrap_or(Integer::default()),
        }
    }
    pub fn mul(&self, b: &Matrix<Integer>) -> Matrix<Integer> {
        // Ensure matrices are compatible for multiplication
        assert_eq!(
            self.width(),
            b.height(),
            "Incompatible matrix dimensions for multiplication"
        );
        // Create a new dense matrix to hold the result
        let mut result_data = vec![Integer::default(); self.height() * b.width()];
        // specialize for all 2 cases:
        match (&self.data(), &b.data()) {
            (MatrixData::Dense(a_values), MatrixData::Dense(b_values)) => {
                // Dense * Dense
                for i in 0..self.height() {
                    for j in 0..b.width() {
                        let mut sum = Integer::default();
                        for k in 0..self.width() {
                            sum += &a_values[i * self.width() + k] * &b_values[k * b.width() + j];
                        }
                        result_data[i * b.width() + j] = sum;
                    }
                }
            }
            (MatrixData::COO(a_entries), MatrixData::Dense(b_values)) => {
                // COO * Dense
                for (r, c, val) in a_entries {
                    for j in 0..b.width() {
                        result_data[r * b.width() + j] += val * &b_values[c * b.width() + j];
                    }
                }
            }
            (MatrixData::Dense(a_values), MatrixData::COO(b_entries)) => {
                // Dense * COO
                for (r, c, val) in b_entries {
                    for i in 0..self.height() {
                        result_data[i * b.width() + c] += &a_values[i * self.width() + r] * val;
                    }
                }
            }
            (MatrixData::COO(a_entries), MatrixData::COO(b_entries)) => {
                // COO * COO
                for (ar, ac, aval) in a_entries {
                    for (br, bc, bval) in b_entries {
                        if ac == br {
                            // only multiply if dimensions match
                            result_data[ar * b.width() + bc] += aval * bval;
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
            "".to_string(),
        )
    }
    pub fn mul_scalar(&mut self, scale_factor: Integer) {
        // Scale the matrix by a given factor
        match self.mut_data() {
            MatrixData::Dense(values) => {
                for value in values.iter_mut() {
                    *value *= &scale_factor;
                }
            }
            MatrixData::COO(entries) => {
                for (_, _, value) in entries.iter_mut() {
                    *value *= &scale_factor;
                }
            }
        }
    }
}
