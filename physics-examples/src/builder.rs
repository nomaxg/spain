#![allow(unused)]
use std::ops::{Add, Mul, Sub};

use model::{AFloat, FromPrimitive};
use ndarray::{Array, IxDyn};
use parse::generalized::HighPrecisionInt;
use parse::mat::{Matrix, MatrixData};
use stream::bigvec::BigVec;

// Simple R1CS builder for physics examples
#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq, Hash)]
pub struct Var(pub usize);

#[derive(Debug, Clone)]
pub struct LC {
    // linear combination of variables
    terms: Vec<(f64, Var)>,
    constant: f64,
}

impl LC {
    pub fn var(v: Var) -> Self {
        Self {
            terms: vec![(1.0, v)],
            constant: 0.0,
        }
    }
    pub fn constant(c: f64) -> Self {
        Self {
            terms: Vec::new(),
            constant: c,
        }
    }
    pub fn zero() -> Self {
        Self {
            terms: Vec::new(),
            constant: 0.0,
        }
    }
    pub fn one() -> Self {
        Self {
            terms: Vec::new(),
            constant: 1.0,
        }
    }
    pub fn add_lc(&self, other: &LC) -> Self {
        let mut terms = vec![];
        let mut i = 0;
        let mut j = 0;
        while i < self.terms.len() && j < other.terms.len() {
            let (coeff_a, var_a) = &self.terms[i];
            let (coeff_b, var_b) = &other.terms[j];
            if var_a < var_b {
                terms.push((*coeff_a, var_a.clone()));
                i += 1;
            } else if var_a > var_b {
                terms.push((*coeff_b, var_b.clone()));
                j += 1;
            } else {
                let coeff = coeff_a + coeff_b;
                if !(coeff == 0.0) {
                    terms.push((coeff, var_a.clone()));
                }
                i += 1;
                j += 1;
            }
        }
        while i < self.terms.len() {
            terms.push(self.terms[i].clone());
            i += 1;
        }
        while j < other.terms.len() {
            terms.push(other.terms[j].clone());
            j += 1;
        }
        Self {
            terms,
            constant: self.constant + other.constant,
        }
    }
    pub fn sub_lc(&self, other: &LC) -> Self {
        self.add_lc(&other.const_mul(-1.0))
    }
    pub fn const_mul(&self, c: f64) -> Self {
        let terms = self
            .terms
            .iter()
            .map(|(coeff, var)| (coeff * c, var.clone()))
            .collect();
        Self {
            terms,
            constant: self.constant * c,
        }
    }
    pub fn const_add(&self, c: f64) -> Self {
        Self {
            terms: self.terms.clone(),
            constant: self.constant + c,
        }
    }
    pub fn eval(&self, witness: &[f64]) -> f64 {
        let mut acc = self.constant;
        for (coeff, Var(i)) in self.terms.iter() {
            acc += coeff * witness[*i];
        }
        acc
    }
}

impl Sub<&LC> for f64 {
    type Output = LC;
    fn sub(self, rhs: &LC) -> Self::Output {
        LC::constant(self).sub_lc(rhs)
    }
}

impl Add<LC> for LC {
    type Output = LC;
    fn add(self, rhs: LC) -> Self::Output {
        self.add_lc(&rhs)
    }
}

impl Sub<&LC> for LC {
    type Output = LC;
    fn sub(self, rhs: &LC) -> Self::Output {
        self.sub_lc(rhs)
    }
}

impl Sub<LC> for LC {
    type Output = LC;
    fn sub(self, rhs: LC) -> Self::Output {
        self.sub_lc(&rhs)
    }
}

impl Add<&LC> for LC {
    type Output = LC;
    fn add(self, rhs: &LC) -> Self::Output {
        self.add_lc(rhs)
    }
}

impl Add<f64> for LC {
    type Output = LC;
    fn add(self, rhs: f64) -> Self::Output {
        self.const_add(rhs)
    }
}

impl Sub<f64> for LC {
    type Output = LC;
    fn sub(self, rhs: f64) -> Self::Output {
        self.const_add(-rhs)
    }
}

impl Mul<f64> for LC {
    type Output = LC;
    fn mul(self, rhs: f64) -> Self::Output {
        self.const_mul(rhs)
    }
}

// builder for R1CS constraints
#[derive(Clone, Debug, Default)]
pub struct R1CSBuilder {
    pub constraints: Vec<(LC, LC, LC)>,
    pub num_vars: usize,
}

impl R1CSBuilder {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            num_vars: 0,
        }
    }

    pub fn new_variable(&mut self) -> Var {
        self.num_vars += 1;
        Var(self.num_vars)
    }

    pub fn new_lc(&mut self) -> LC {
        let var = self.new_variable();
        LC::var(var)
    }

    pub fn add_constraint(&mut self, a: &LC, b: &LC, c: &LC) {
        self.constraints.push((a.clone(), b.clone(), c.clone()));
    }

    pub fn to_r1cs_int<T: HighPrecisionInt>(
        &self,
        scale_factor: AFloat,
    ) -> (Matrix<T>, Matrix<T>, Matrix<T>) {
        let num_rows = self.constraints.len();
        let width = self.num_vars + 1;

        let mut a_entries = Vec::new();
        let mut b_entries = Vec::new();
        let mut c_entries = Vec::new();

        for (row_idx, (a_lc, b_lc, c_lc)) in self.constraints.iter().enumerate() {
            Self::push_lc_entries(&mut a_entries, row_idx, a_lc);
            Self::push_lc_entries(&mut b_entries, row_idx, b_lc);
            Self::push_lc_entries(&mut c_entries, row_idx, c_lc);
        }

        let build_matrix = |entries: Vec<(usize, usize, f64)>, label: &str| -> Matrix<T> {
            let data = MatrixData::COO(BigVec::from_vec(entries));
            let mat_f64 = Matrix::new(data, width, num_rows, None, label.to_string());
            Matrix::<T>::from_f64(&mat_f64, scale_factor.clone(), None)
        };

        let a = build_matrix(a_entries, "PhysicsExamples A");
        let b = build_matrix(b_entries, "PhysicsExamples B");
        let c = build_matrix(c_entries, "PhysicsExamples C");

        (a, b, c)
    }

    pub fn to_r1cs(&self, scale_factor: AFloat) -> (Matrix<i64>, Matrix<i64>, Matrix<i64>) {
        self.to_r1cs_int(scale_factor)
    }

    fn push_lc_entries(entries: &mut Vec<(usize, usize, f64)>, row: usize, lc: &LC) {
        if lc.constant != 0.0 {
            entries.push((row, 0, lc.constant));
        }
        for (coeff, Var(idx)) in lc.terms.iter() {
            if *coeff == 0.0 {
                continue;
            }
            entries.push((row, *idx, *coeff));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_r1cs_builds_sparse_matrices() {
        let mut builder = R1CSBuilder::new();
        let x = builder.new_lc();
        let y = builder.new_lc();
        let c = LC::constant(6.0);

        builder.add_constraint(&x, &y, &c);

        let scale = AFloat::from_f64(1.0).unwrap();
        let (a, b, c) = builder.to_r1cs(scale);

        assert_eq!(a.width(), 3);
        assert_eq!(a.height(), 1);
        assert_eq!(b.width(), 3);
        assert_eq!(c.width(), 3);
    }
}
