use super::mont::MLE as MontMLE;
use crate::FieldMont;
use rug::{Complete, Integer};
use stream::bigvec::BigVec;

#[derive(Debug, Clone)]
pub struct MLE {
    // integer evaluations over the boolean hypercube
    pub evals: Vec<Integer>,
    // number of variables in the mle
    num_vars: usize,
}

impl MLE {
    // constructor (all evaluations are zero)
    pub fn new(num_vars: usize) -> Self {
        let evals = vec![Integer::from(0); 1 << num_vars];
        Self { evals, num_vars }
    }
    // constructor from explicit buffer
    pub fn from_buffer(evals: Vec<Integer>, num_vars: usize) -> Self {
        assert!(evals.len() == (1 << num_vars));
        Self { evals, num_vars }
    }
    // constructor from buffer without num_vars as input
    pub fn from_buffer_pure(evals: Vec<Integer>) -> Self {
        // check that length is a power of 2
        let num_vars = evals.len().trailing_zeros() as usize;
        assert!(evals.len() == (1 << num_vars));
        Self { evals, num_vars }
    }
    // get number of variables
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
    // bind outer variable to the value x
    pub fn bind(&mut self, x: &Integer) {
        // ensure there is a variable to bind
        assert!(self.num_vars > 0);
        // decrement number of variables
        self.num_vars -= 1;
        // create new buffer
        let mut new_evals = vec![Integer::from(0); 1 << self.num_vars];
        // get iterator for old and new
        let old_iter = self.evals.chunks(2);
        let new_iter = new_evals.iter_mut();
        // iterate over old and new
        old_iter.zip(new_iter).for_each(|(old, new)| {
            // new = (old[1] - old[0]) * x + old[0]
            let mut d = (&old[1] - &old[0]).complete();
            d *= x;
            *new = d + &old[0];
        });
        // set evals to new
        self.evals = new_evals;
    }
    // If f is our original MLE, return MLEs fl, fr such that:
    // f(x_1,...,x_n) = fl(x_1,..x_(n-1))) + xn * fr(x_1,...,x_(n-1))
    pub fn split(&self, shift: u64) -> (Self, Self) {
        // ensure there is a variable to split
        assert!(self.num_vars > 0);
        // create new buffers
        let mut left_evals = vec![Integer::from(0); 1 << (self.num_vars - 1)];
        let mut right_evals = vec![Integer::from(0); 1 << (self.num_vars - 1)];
        // get iterator for old and new
        let old_iter = self.evals.chunks(2);
        let left_iter = left_evals.iter_mut();
        let right_iter = right_evals.iter_mut();
        // iterate over old and new
        old_iter
            .zip(left_iter)
            .zip(right_iter)
            .for_each(|((old, left), right)| {
                *left = old[0].clone();
                *right = (&old[1] - &old[0]).complete() + Integer::from(shift);
            });
        // return new MLEs
        (
            Self::from_buffer(left_evals, self.num_vars - 1),
            Self::from_buffer(right_evals, self.num_vars - 1),
        )
    }
    pub fn split_msb(&self, shift: Option<&Integer>) -> (Self, Self) {
        assert!(self.num_vars > 0);
        let half = 1 << (self.num_vars - 1);
        let mut left_evals = Vec::with_capacity(half);
        let mut right_evals = Vec::with_capacity(half);
        let zero = Integer::from(0);
        let shift = shift.unwrap_or(&zero);

        for i in 0..half {
            let left = self.evals[i].clone() + shift;
            let right = (&self.evals[i + half] - &self.evals[i]).complete() + shift;
            left_evals.push(left);
            right_evals.push(right);
        }

        (
            Self::from_buffer(left_evals, self.num_vars - 1),
            Self::from_buffer(right_evals, self.num_vars - 1),
        )
    }
    // Convenience function that computes full MLE evaulation through iterative binding
    pub fn eval(&mut self, x: &[Integer]) -> Integer {
        assert!(x.len() == self.num_vars);
        for xi in x {
            self.bind(xi);
        }
        self.evals[0].clone()
    }
    // Reduce to a montgomery polynomial
    pub fn reduce_to_mont(&self, mont: &FieldMont) -> MontMLE {
        let evals: Vec<_> = self
            .evals
            .iter()
            .map(|e| mont.from_bigint(e.clone()))
            .collect();
        MontMLE::from_buffer_pure(BigVec::from_vec(evals))
    }
    // Generate a new MLE which which is the sum of self and a shifted and scaled rhs
    pub fn fold(&self, rhs: &MLE, shift: &Integer, scale: &Integer) -> MLE {
        assert!(self.num_vars == rhs.num_vars);
        let evals = self
            .evals
            .iter()
            .zip(rhs.evals.iter())
            .map(|(l, r)| (l - shift).complete() + scale.clone() * (r - shift).complete())
            .collect::<Vec<_>>();
        MLE::from_buffer(evals, self.num_vars)
    }
}
