use super::int::MLE as IntMLE;
use crate::{FieldElem, FieldMont, usize_to_mont};
use rug::Integer;
use stream::bigvec::BigVec;

// various utilities for multivariate and univariate polynomials
// over finite fields with at most 64 bits.

#[derive(Debug, Clone)]
pub struct MLE {
    // evaluations over the boolean hypercube (in montgomery form)
    pub evals: BigVec<FieldElem>,
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
    pub fn from_buffer(evals: BigVec<FieldElem>, num_vars: usize) -> Self {
        assert!(evals.len() == (1usize << num_vars));
        Self { evals, num_vars }
    }
    // constructor from buffer without num_vars as input
    pub fn from_buffer_pure(evals: BigVec<FieldElem>) -> Self {
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
    pub fn lin_transform(&mut self, a: FieldElem, b: FieldElem, mont: &FieldMont) {
        // ensure there is a variable to bind
        assert!(self.num_vars > 0);
        // get iterator for evals
        let eval_iter = self.evals.iter_mut();
        // iterate over evals
        eval_iter.for_each(|eval| {
            // eval = a * eval + b
            *eval = mont.add(mont.mul(*eval, a), b);
        });
    }
    // bind outer variable to the value x (in montgomery form)
    pub fn bind(&mut self, x: FieldElem, mont: &FieldMont) {
        // ensure there is a variable to bind
        assert!(self.num_vars > 0);
        // decrement number of variables
        self.num_vars -= 1;
        // create new buffer
        let mut new_evals = BigVec::new(1 << self.num_vars).unwrap();
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
        // set evals to new
        self.evals = new_evals;
    }
    // If f is our original MLE, return MLEs fl, fr such that:
    // f(x_1,...,x_n) = fl(x_1,..x_(n-1))) + xn * fr(x_1,...,x_(n-1))
    pub fn split(&self, mont: &FieldMont) -> (Self, Self) {
        // ensure there is a variable to split
        assert!(self.num_vars > 0);
        // create new buffers
        let mut left_evals = BigVec::new(1 << (self.num_vars - 1)).unwrap();
        let mut right_evals = BigVec::new(1 << (self.num_vars - 1)).unwrap();
        // loop over pairs of evaluations and split
        for i in 0..self.evals.len() / 2 {
            left_evals[i] = self.evals[2 * i];
            right_evals[i] = mont.sub(self.evals[2 * i + 1], self.evals[2 * i]);
        }
        // return new MLEs
        (
            Self::from_buffer(left_evals, self.num_vars - 1),
            Self::from_buffer(right_evals, self.num_vars - 1),
        )
    }
    pub fn split_msb(&self, mont: &FieldMont, shift: Option<FieldElem>) -> (Self, Self) {
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
    pub fn eval(&mut self, x: &[FieldElem], mont: &FieldMont) -> FieldElem {
        assert!(x.len() == self.num_vars);
        for xi in x {
            self.bind(*xi, mont);
        }
        self.evals[0]
    }
    // Lift montgomery MLE to integer polynomial
    pub fn lift_to_int(&self, mont: &FieldMont) -> IntMLE {
        let evals_int: Vec<Integer> = self
            .evals
            .iter()
            .map(|mval| {
                //let native = mont.to_normal(*mval);
                mont.to_integer(*mval)
                //Integer::from(native)
            })
            .collect();

        IntMLE::from_buffer(evals_int, self.num_vars)
    }
    // Generate a new MLE which which is the sum of self and a shifted and scaled rhs
    pub fn fold(&self, mont: &FieldMont, rhs: &Self, shift: FieldElem, scale: FieldElem) -> Self {
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

// consider a polynomial p(x) of degree d
// provided a a vec of evaluations (in montgomery form):
// [p(0), p(1), ..., p(d)]
// and a particular value x in montgomery form
// return p(x) in montgomery form
// Note. This can be sped up a lot by precomputing c
// and inverses of c[i] - c[j]
// however, this is a lower order concern
pub fn lagrange_interpolate(evals: &[FieldElem], x: FieldElem, mont: &FieldMont) -> FieldElem {
    let mut sum = mont.zero();
    let d = evals.len();
    // convert 0 though d to montgomery form
    let mut c = vec![mont.zero(); d];
    for (i, ci) in c.iter_mut().enumerate() {
        *ci = usize_to_mont(i, mont);
    }
    // compute the lagrange polynomial
    for i in 0..d {
        let mut t = evals[i];
        for j in 0..d {
            if i != j {
                let l = mont.sub(x, c[j]);
                let r = mont.inv(mont.sub(c[i], c[j]));
                let p = mont.mul(l, r);
                t = mont.mul(t, p);
            }
        }
        sum = mont.add(sum, t);
    }
    sum
}

// given a degree 1 polynomial p(x)
// as evaluations at 0, 1
// return a vector of evaluations at 0, 1, 2, ..., n
pub fn lin_batch_eval(p0: FieldElem, p1: FieldElem, n: usize, mont: &FieldMont) -> Vec<FieldElem> {
    // create new buffer
    let mut evals = vec![mont.zero(); n + 1];
    // if p0 and p1 are both zero, return all zeros
    if p0 == mont.zero() && p1 == mont.zero() {
        return evals;
    }
    // set first two evaluations
    evals[0] = p0;
    evals[1] = p1;
    // get the difference between the two evaluations
    let t = mont.sub(p1, p0);
    // compute the rest iteratively
    for i in 2..=n {
        evals[i] = mont.add(evals[i - 1], t);
    }
    evals
}
