use crate::mat::{Matrix, MatrixData};
use ff::poly::int::MLE;
use ff::{FieldElem, FieldMont, i256_to_mont, int_to_mont};
pub use i256::I256;
use i256::I512;
use model::{HighPrecision, ToPrimitiveExt};
use rug::Integer;
use std::ops::{Add, Div, Mul, Range, Sub};
use stream::bigvec::BigVec;

/// Vector of the location of the BigVec where randomness needs to be injected, as well as the
/// index of the randomness to be injected
pub type InjectionInfo = Vec<(usize, usize)>;

pub trait HighPrecisionInt:
    Sized
    + Copy
    + PartialOrd
    + Ord
    + Eq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Default
{
    fn from_hp<T: HighPrecision>(x: T) -> Self;
    fn from_f64(x: f64) -> Self;
    fn from_i128(x: i128) -> Self;
    fn to_rug_int(&self) -> Integer;
    fn is_in_bound<T: HighPrecision>(x: &T) -> bool;
    fn to_field_elem(&self, mont: &FieldMont) -> FieldElem {
        int_to_mont(&self.to_rug_int(), mont)
    }
}

impl<M: HighPrecisionInt> Matrix<M> {
    pub fn from_hp<T: HighPrecision>(matrix: &Matrix<T>, scale_factor: T) -> Self {
        let data = match &matrix.data() {
            MatrixData::Dense(values) => {
                let mut scaled_values = BigVec::new(values.len()).unwrap();
                for (i, v) in values.iter().enumerate() {
                    let tmp = v.clone() * scale_factor.clone();
                    scaled_values[i] = M::from_hp(tmp);
                }
                MatrixData::Dense(scaled_values)
            }
            _ => {
                panic!("COO not implemented yet");
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

    pub fn from_f64<T: HighPrecision>(
        matrix: &Matrix<f64>,
        scale_factor: T,
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
                    let v = T::from_f64(v).unwrap() * scale_factor.clone();
                    scaled_values[i] = M::from_hp(v);
                }
                MatrixData::Dense(scaled_values)
            }
            MatrixData::COO(entries) => {
                let mut scaled_entries = BigVec::new(entries.len()).unwrap();
                for (i, &(r, c, v)) in entries.iter().enumerate() {
                    let v = T::from_f64(v).unwrap() * scale_factor.clone();
                    let shifted_c = c
                        .checked_add(col_shift)
                        .expect("Column index overflowing when applying shift");
                    scaled_entries[i] = (r, shifted_c, M::from_hp(v));
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

    pub fn from_f64_index(matrix: &Matrix<f64>, table: &[M], shift: usize) -> Self {
        let data = match &matrix.data() {
            MatrixData::COO(entries) => {
                let mut scaled_entries = BigVec::new(entries.len()).unwrap();
                for (i, &(r, c, v)) in entries.iter().enumerate() {
                    let idx = v as usize + shift;
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
            _ => panic!("Expected COO matrix for from_f64_index"),
        };
        Matrix::new(
            data,
            matrix.width(),
            matrix.height(),
            matrix.ranges().cloned(),
            matrix.comment().to_string(),
        )
    }

    pub fn from_f64_inject(matrix: &Matrix<f64>, shift: usize) -> (Self, InjectionInfo) {
        let mut inject_info: InjectionInfo = vec![];
        let data = match &matrix.data() {
            MatrixData::COO(entries) => {
                let mut scaled_entries = BigVec::new(entries.len()).unwrap();
                for (i, &(r, c, v)) in entries.iter().enumerate() {
                    let idx = v as usize + shift;
                    if v.fract() != 0.0 {
                        panic!("Value is not an integer: {}", v);
                    }
                    scaled_entries[i] = (r, c, M::from_f64(0.));
                    inject_info.push((i, idx));
                }
                MatrixData::COO(scaled_entries)
            }
            _ => panic!("Expected COO matrix for from_f64_inject"),
        };
        (
            Matrix::new(
                data,
                matrix.width(),
                matrix.height(),
                matrix.ranges().cloned(),
                matrix.comment().to_string(),
            ),
            inject_info,
        )
    }

    pub fn to_m64(&self, den: FieldElem, mont: &FieldMont) -> Matrix<FieldElem> {
        let data = match &self.data() {
            MatrixData::Dense(values) => MatrixData::Dense({
                let mut new_values = BigVec::new(values.len()).unwrap();
                for (v, w) in values.iter().zip(new_values.iter_mut()) {
                    *w = mont.mul(v.to_field_elem(mont), den);
                }
                new_values
            }),
            MatrixData::COO(entries) => MatrixData::COO({
                let mut new_entries = BigVec::new(entries.len()).unwrap();
                for ((r1, c1, v1), (r2, c2, v2)) in entries.iter().zip(new_entries.iter_mut()) {
                    *r2 = *r1;
                    *c2 = *c1;
                    *v2 = mont.mul(v1.to_field_elem(mont), den);
                }
                new_entries
            }),
        };
        Matrix::new(
            data,
            self.width(),
            self.height(),
            self.ranges().cloned(),
            self.comment().to_string(),
        )
    }

    pub fn extract_rows_to_mle(&self, rows: Option<&Range<usize>>) -> MLE {
        let tmp = 0..self.height();
        let rows = rows.unwrap_or(&tmp);
        let h = rows.len();
        let w = self.width();
        assert!(h > 0 && w > 0, "MLE cannot be empty");
        let num_row_vars = h.next_power_of_two().trailing_zeros() as usize;
        let num_col_vars = w.next_power_of_two().trailing_zeros() as usize;
        let num_vars = num_row_vars + num_col_vars;
        let num_evals = 1 << num_vars;
        let start = rows.start * self.width();
        let mut evals = vec![Integer::from(0); num_evals];

        match &self.data() {
            MatrixData::Dense(values) => {
                for r in 0..h {
                    for c in 0..w {
                        let v = values[start + r * w + c];
                        let idx = (r << num_col_vars) | c;
                        evals[idx] = v.to_rug_int();
                    }
                }
            }
            MatrixData::COO(_) => {
                panic!("Only dense matrix (witness) can be turned into a set of MLE evals")
            }
        }
        MLE::from_buffer(evals, num_vars)
    }
}

impl HighPrecisionInt for I256 {
    fn from_hp<T: HighPrecision>(x: T) -> Self {
        x.to_i256()
    }

    fn from_f64(x: f64) -> Self {
        x.to_i256()
    }

    fn from_i128(x: i128) -> Self {
        Self::from(x)
    }

    fn to_rug_int(&self) -> Integer {
        Integer::from_str_radix(self.to_string().as_str(), 10).unwrap()
    }

    fn is_in_bound<T: HighPrecision>(x: &T) -> bool {
        let x_int = x.to_rug_integer();
        !(x_int <= Self::MIN.to_rug_int() || x_int >= Self::MAX.to_rug_int() || x.is_nan())
    }

    fn to_field_elem(&self, mont: &FieldMont) -> FieldElem {
        i256_to_mont(self, mont)
    }
}

impl HighPrecisionInt for I512 {
    fn from_hp<T: HighPrecision>(x: T) -> Self {
        x.to_i512()
    }

    fn from_f64(x: f64) -> Self {
        x.to_i512()
    }

    fn from_i128(x: i128) -> Self {
        Self::from(x)
    }

    fn to_rug_int(&self) -> Integer {
        Integer::from_str_radix(self.to_string().as_str(), 10).unwrap()
    }

    fn is_in_bound<T: HighPrecision>(x: &T) -> bool {
        let x_int = x.to_rug_integer();
        !(x_int <= Self::MIN.to_rug_int() || x_int >= Self::MAX.to_rug_int() || x.is_nan())
    }
}

impl HighPrecisionInt for i128 {
    fn from_hp<T: HighPrecision>(x: T) -> Self {
        x.to_i128().unwrap()
    }

    fn from_f64(x: f64) -> Self {
        x as i128
    }

    fn from_i128(x: i128) -> Self {
        x
    }

    fn to_rug_int(&self) -> Integer {
        Integer::from_str_radix(self.to_string().as_str(), 10).unwrap()
    }

    fn is_in_bound<T: HighPrecision>(x: &T) -> bool {
        let x_int = x.to_rug_integer();
        !(x_int <= Self::MIN.to_rug_int() || x_int >= Self::MAX.to_rug_int() || x.is_nan())
    }

    fn to_field_elem(&self, mont: &FieldMont) -> FieldElem {
        mont.from_i128(*self)
    }
}

impl HighPrecisionInt for i64 {
    fn from_hp<T: HighPrecision>(x: T) -> Self {
        x.to_i64().unwrap()
    }

    fn from_f64(x: f64) -> Self {
        x as i64
    }

    fn from_i128(x: i128) -> Self {
        Self::try_from(x).unwrap()
    }

    fn to_rug_int(&self) -> Integer {
        Integer::from(*self)
    }

    fn is_in_bound<T: HighPrecision>(x: &T) -> bool {
        let x_int = x.to_rug_integer();
        !(x_int <= Self::MIN.to_rug_int() || x_int >= Self::MAX.to_rug_int() || x.is_nan())
    }

    fn to_field_elem(&self, mont: &FieldMont) -> FieldElem {
        mont.from_i128(*self as i128)
    }
}
