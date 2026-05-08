use stream::bigvec::BigVec;

use crate::mat::{Matrix, MatrixData};

impl Matrix<f64> {
    pub fn seq_dense_matrix(width: usize, height: usize) -> Self {
        let mut values = BigVec::new(width * height).unwrap();
        for i in 0..width * height {
            values[i] = i as f64;
        }
        Matrix::new(
            MatrixData::Dense(values),
            width,
            height,
            None,
            "Test: sequential dense matrix".to_string(),
        )
    }
    pub fn diagonal_coo_matrix(size: usize) -> Self {
        let mut values = BigVec::new(size).unwrap();
        for i in 0..size {
            values[i] = (i, i, i as f64);
        }
        Matrix::new(
            MatrixData::COO(values),
            size,
            size,
            None,
            "Test: diagonal COO matrix".to_string(),
        )
    }
    pub fn get_f64_max_magnitude(&self) -> f64 {
        match &self.data() {
            MatrixData::Dense(values) => values.iter().cloned().fold(0.0, f64::max),
            MatrixData::COO(values) => values
                .iter()
                .map(|(_, _, v)| v)
                .cloned()
                .fold(0.0, f64::max),
        }
    }
}
