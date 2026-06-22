use std::ops::Range;

use ff::{FieldElem, FieldMont, poly::int::MLE as IntMLE};
use i256::{I256, I512, I1024};
use model::HighPrecision;
use parse::{
    generalized::{HighPrecisionInt, InjectionInfo},
    mat::Matrix,
};

use crate::inputs::{Metadata, R1CSMatrices};

pub trait R1CSInstance<P, T>
where
    P: HighPrecision,
    T: HighPrecisionInt,
{
    // TODO: add input param
    fn get_matrices(
        &self,
        scale_factor: P,
        randomness: Option<&Vec<T>>,
    ) -> (R1CSMatrices<T>, Option<InjectionInfo>);
    fn get_meta(&self) -> Metadata;
    fn compute_commit_witness(&mut self, scale_factor: P, batch_size: usize) -> Matrix<T>;
    fn compute_full_witness(
        &mut self,
        metadata: &Metadata,
        random_values: Vec<P>,
        scale_factor: P,
    ) -> Matrix<T>;
}

pub trait ToI512 {
    fn to_i512(self) -> I512;
}

pub trait ToI1024 {
    fn to_i1024(self) -> I1024;
}

impl ToI512 for i64 {
    fn to_i512(self) -> I512 {
        I512::from(self)
    }
}

impl ToI512 for i128 {
    fn to_i512(self) -> I512 {
        I512::from(self)
    }
}

impl ToI1024 for i128 {
    fn to_i1024(self) -> I1024 {
        I1024::from(self)
    }
}

impl ToI1024 for I256 {
    fn to_i1024(self) -> I1024 {
        I1024::from_str_radix(self.to_string().as_str(), 10).unwrap()
    }
}

impl ToI1024 for I512 {
    fn to_i1024(self) -> I1024 {
        I1024::from_str_radix(self.to_string().as_str(), 10).unwrap()
    }
}

impl ToI512 for I256 {
    fn to_i512(self) -> I512 {
        I512::from_str_radix(self.to_string().as_str(), 10).unwrap()
    }
}

impl ToI512 for I512 {
    fn to_i512(self) -> I512 {
        self
    }
}

pub trait MatrixIntOps {
    fn rat_to_mont(&self, den_mont: FieldElem, mont: &FieldMont) -> Matrix<FieldElem>;
    fn extract_rows_to_mle(&self, rows: &Range<usize>) -> IntMLE;
}

impl MatrixIntOps for Matrix<i64> {
    fn rat_to_mont(&self, den_mont: FieldElem, mont: &FieldMont) -> Matrix<FieldElem> {
        Matrix::<FieldElem>::i64_rat_to_m64(self, den_mont, mont)
    }

    fn extract_rows_to_mle(&self, rows: &Range<usize>) -> IntMLE {
        Matrix::<i64>::extract_rows_to_mle(self, Some(rows))
    }
}

impl MatrixIntOps for Matrix<i128> {
    fn rat_to_mont(&self, den_mont: FieldElem, mont: &FieldMont) -> Matrix<FieldElem> {
        Matrix::<FieldElem>::i128_rat_to_m64(self, den_mont, mont)
    }

    fn extract_rows_to_mle(&self, rows: &Range<usize>) -> IntMLE {
        Matrix::<i128>::extract_rows_to_mle(self, Some(rows))
    }
}

impl MatrixIntOps for Matrix<I256> {
    fn rat_to_mont(&self, den_mont: FieldElem, mont: &FieldMont) -> Matrix<FieldElem> {
        self.to_m64(den_mont, mont)
    }

    fn extract_rows_to_mle(&self, rows: &Range<usize>) -> IntMLE {
        self.extract_rows_to_mle(Some(rows))
    }
}

impl MatrixIntOps for Matrix<I512> {
    fn rat_to_mont(&self, den_mont: FieldElem, mont: &FieldMont) -> Matrix<FieldElem> {
        self.to_m64(den_mont, mont)
    }

    fn extract_rows_to_mle(&self, rows: &Range<usize>) -> IntMLE {
        self.extract_rows_to_mle(Some(rows))
    }
}
