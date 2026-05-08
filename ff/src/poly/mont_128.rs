use super::int::MLE as IntMLE;
use crate::ops_128::{M128, Mont};
use rayon::prelude::*;
use rug::Integer;
use stream::bigvec::BigVec;

// various utilities for multivariate and univariate polynomials
// over finite fields with at most 128 bits.

#[derive(Debug, Clone)]
pub struct MLE {
    // evaluations over the boolean hypercube (in montgomery form)
    pub evals: BigVec<M128>,
    // number of variables in the mle
    num_vars: usize,
}

impl MLE {
    // constructor (not necessarily initialized)
    pub fn new(num_vars: usize) -> Self {
        let evals = BigVec::new(1usize << num_vars).unwrap();
        Self { evals, num_vars }
    }
    // constructor from explicit buffer
    pub fn from_buffer(evals: BigVec<M128>, num_vars: usize) -> Self {
        assert!(evals.len() == (1usize << num_vars));
        Self { evals, num_vars }
    }
    // constructor from buffer without num_vars as input
    pub fn from_buffer_pure(evals: BigVec<M128>) -> Self {
        // check that length is a power of 2
        let num_vars = evals.len().trailing_zeros() as usize;
        assert!(evals.len() == (1usize << num_vars));
        Self { evals, num_vars }
    }
    // get number of variables
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
    // linearly transform an MLE
    // given a and b, turn self into a * f(x) + b
    pub fn lin_transform(&mut self, a: M128, b: M128, mont: &Mont) {
        // ensure there is a variable to bind
        assert!(self.num_vars > 0);
        // get iterator for evals
        let eval_iter = self.evals.par_iter_mut();
        // iterate over evals
        eval_iter.for_each(|eval| {
            // eval = a * eval + b
            *eval = mont.add(mont.mul(*eval, a), b);
        });
    }
    // bind outer variable to the value x (in montgomery form)
    pub fn bind(&mut self, x: M128, mont: &Mont) {
        // ensure there is a variable to bind
        assert!(self.num_vars > 0);
        // decrement number of variables
        self.num_vars -= 1;
        // create new buffer
        let mut new_evals = BigVec::new(1 << self.num_vars).unwrap();
        if rayon::current_num_threads() == 1 {
            for i in 0..self.evals.len() / 2 {
                if self.evals[2 * i + 1] == mont.zero() && self.evals[2 * i] == mont.zero() {
                    // if both evaluations are zero, set new eval to zero
                    new_evals[i] = mont.zero();
                    continue;
                }
                // new = (old[1] - old[0]) * x + old[0]
                let d = mont.sub(self.evals[2 * i + 1], self.evals[2 * i]);
                let s = mont.mul(d, x);
                new_evals[i] = mont.add(s, self.evals[2 * i]);
            }
        } else {
            // get iterator for old and new
            let old_iter = self.evals.par_chunks(2);
            let new_iter = new_evals.par_iter_mut();
            // iterate over old and new
            old_iter.zip(new_iter).for_each(|(old, new)| {
                // new = (old[1] - old[0]) * x + old[0]
                let d = mont.sub(old[1], old[0]);
                let s = mont.mul(d, x);
                *new = mont.add(s, old[0]);
            });
        }
        // set evals to new
        self.evals = new_evals;
    }
    // If f is our original MLE, return MLEs fl, fr such that:
    // f(x_1,...,x_n) = fl(x_1,..x_(n-1))) + xn * fr(x_1,...,x_(n-1))
    pub fn split(&self, mont: &Mont) -> (Self, Self) {
        // ensure there is a variable to split
        assert!(self.num_vars > 0);
        // create new buffers
        let mut left_evals = BigVec::new(1 << (self.num_vars - 1)).unwrap();
        let mut right_evals = BigVec::new(1 << (self.num_vars - 1)).unwrap();
        // get iterator for old and new
        let old_iter = self.evals.par_chunks(2);
        let left_iter = left_evals.par_iter_mut();
        let right_iter = right_evals.par_iter_mut();
        // iterate over old and new
        old_iter
            .zip(left_iter)
            .zip(right_iter)
            .for_each(|((old, left), right)| {
                *left = old[0];
                *right = mont.sub(old[1], old[0]);
            });
        // return new MLEs
        (
            Self::from_buffer(left_evals, self.num_vars - 1),
            Self::from_buffer(right_evals, self.num_vars - 1),
        )
    }
    pub fn split_msb(&self, mont: &Mont, shift: Option<M128>) -> (Self, Self) {
        assert!(self.num_vars > 0);
        let half = 1 << (self.num_vars - 1);
        let mut left_evals = BigVec::new(half).unwrap();
        let mut right_evals = BigVec::new(half).unwrap();
        let shift = shift.unwrap_or(mont.zero());

        for i in 0..half {
            let a0 = self.evals[i];
            let a1 = self.evals[i + half];
            left_evals[i] = mont.add(a0, shift);
            right_evals[i] = mont.add(mont.sub(a1, a0), shift);
        }

        (
            Self::from_buffer(left_evals, self.num_vars - 1),
            Self::from_buffer(right_evals, self.num_vars - 1),
        )
    }
    // Convenience function that computes full MLE evaulation through iterative binding
    pub fn eval(&mut self, x: &[M128], mont: &Mont) -> M128 {
        assert!(x.len() == self.num_vars);
        for xi in x {
            self.bind(*xi, mont);
        }
        self.evals[0].clone()
    }
    // Lift montgomery MLE to integer polynomial
    pub fn lift_to_int(&self, mont: &Mont) -> IntMLE {
        let evals_int: Vec<Integer> = self
            .evals
            .iter()
            .map(|mval| {
                let native_u128: u128 = mont.to_normal(*mval);
                Integer::from(native_u128)
            })
            .collect();

        IntMLE::from_buffer(evals_int, self.num_vars)
    }
    // Generate a new MLE which which is the sum of self and a shifted and scaled rhs
    pub fn fold(&self, mont: &Mont, rhs: &Self, shift: M128, scale: M128) -> Self {
        assert!(self.num_vars == rhs.num_vars);
        let new_evals = self
            .evals
            .iter()
            .zip(rhs.evals.iter())
            .map(|(l, r)| mont.add(mont.sub(*l, shift), mont.mul(scale, mont.sub(*r, shift))))
            .collect();
        let new_evals = BigVec::from_vec(new_evals);
        Self::from_buffer(new_evals, rhs.num_vars())
    }
}
