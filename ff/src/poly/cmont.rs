use std::ops::Range;

use crate::{FieldElem, FieldMont};
use stream::bigvec::BigVec;

// various utilities for multivariate and univariate polynomials
// over finite fields with at most 64 bits.

#[derive(Debug, Clone)]
pub struct MLE {
    // evaluations over the boolean hypercube (in montgomery form)
    pub evals: BigVec<FieldElem>,
    pub ranges: Vec<Range<usize>>,
    // number of variables in the mle
    num_vars: usize,
}

impl MLE {
    // constructor from explicit buffer
    pub fn from_buffer(evals: BigVec<FieldElem>, ranges: Vec<Range<usize>>) -> Self {
        // check that ranges are sorted and non-overlapping
        assert!(ranges.windows(2).all(|w| w[0].end <= w[1].start));
        // check that all ranges have even start and end points
        assert!(ranges.iter().all(|r| r.start % 2 == 0 && r.end % 2 == 0));
        // check that evals is large enough
        assert!(evals.len() == ranges.iter().map(|r| r.len()).sum::<usize>());
        // compute number of variables
        let num_vars = ranges[ranges.len() - 1]
            .end
            .next_power_of_two()
            .trailing_zeros() as usize;
        // check that length is a power of 2
        Self {
            evals,
            ranges,
            num_vars,
        }
    }
    // get number of variables
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
    // scale an MLE by a constant
    pub fn scale(&mut self, c: FieldElem, mont: &FieldMont) {
        // iterate over evals
        self.evals.iter_mut().for_each(|eval| {
            *eval = mont.mul(*eval, c);
        });
    }
    // linearly transform an MLE
    // given a and b, turn self into a * f(x) + b
    pub fn lin_transform(&mut self, a: FieldElem, b: FieldElem, mont: &FieldMont) {
        // iterate over evals
        self.evals.iter_mut().for_each(|eval| {
            *eval = mont.add(mont.mul(*eval, a), b);
        });
    }
    // adjacent ranges are merged [a, b) and [b, c) -> [a, c)
    fn merge_ranges(&mut self) {
        let mut new_ranges = vec![];
        let mut prev: Range<usize> = self.ranges[0].clone();
        for range in &self.ranges[1..] {
            // if the end of the previous range is the start of the current range, merge them
            if prev.end == range.start {
                prev.end = range.end;
            } else {
                new_ranges.push(prev);
                prev = range.clone();
            }
        }
        // push last range
        new_ranges.push(prev);
        // set ranges to new
        self.ranges = new_ranges;
    }
    // get ranges after bind
    fn get_shrunk_ranges(&self) -> Vec<Range<usize>> {
        // shrink ranges to fit evals
        let mut new_ranges = vec![];
        for range in &self.ranges {
            let mut start = range.start / 2;
            let mut end = range.end / 2;
            // if start is odd, subtract 1
            if start % 2 == 1 {
                start -= 1;
            }
            // if end is odd, add 1
            if end % 2 == 1 {
                end += 1;
            }
            new_ranges.push(start..end);
        }
        new_ranges
    }
    // bind outer variable to the value x (in montgomery form)
    pub fn bind(&mut self, x: FieldElem, mont: &FieldMont) {
        // ensure there is a variable to bind
        assert!(self.num_vars > 0);
        // decrement number of variables
        self.num_vars -= 1;
        // simplify ranges
        self.merge_ranges();
        let new_ranges = self.get_shrunk_ranges();
        // create new buffer
        let mut new_evals = BigVec::new(new_ranges.iter().map(|r| r.len()).sum::<usize>()).unwrap();
        let mut old_idx = 0;
        let mut new_idx = 0;
        for (i, range) in self.ranges.iter().enumerate() {
            let new_range = &new_ranges[i];
            if range.start / 2 != new_range.start {
                // padding
                new_evals[new_idx] = mont.zero();
                new_idx += 1;
            }
            let range_len = range.len();
            let old_end = old_idx + range_len;
            loop {
                if self.evals[old_idx] == mont.zero() && self.evals[old_idx + 1] == mont.zero() {
                    // if both evaluations are zero, set new eval to zero
                    new_evals[new_idx] = mont.zero();
                } else {
                    // new = (old[1] - old[0]) * x + old[0]
                    let d = mont.sub(self.evals[old_idx + 1], self.evals[old_idx]);
                    let s = mont.mul(d, x);
                    new_evals[new_idx] = mont.add(s, self.evals[old_idx]);
                }
                old_idx += 2;
                new_idx += 1;
                if old_idx == old_end {
                    break;
                }
            }
            if range.end / 2 != new_range.end {
                // padding
                new_evals[new_idx] = mont.zero();
                new_idx += 1;
            }
            //}
        }
        // rescale ranges
        self.ranges = new_ranges;
        // set evals to new
        self.evals = new_evals;
    }
    // Convenience function that computes full MLE evaluation through iterative binding
    pub fn eval(&mut self, x: &[FieldElem], mont: &FieldMont) -> FieldElem {
        assert!(x.len() == self.num_vars);
        for xi in x {
            self.bind(*xi, mont);
        }
        self.evals[0]
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
        *ci = mont.to_mont(i as u128);
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
// return a vector of evaluations at 1, 2, ..., n (0 is omitted)
#[inline(always)]
pub fn lin_batch_eval(p0: FieldElem, p1: FieldElem, n: usize, mont: &FieldMont) -> Vec<FieldElem> {
    // create new buffer
    let mut evals = vec![mont.zero(); n];
    // if p0 and p1 are both zero, return all zeros
    if p0 == mont.zero() && p1 == mont.zero() {
        return evals;
    }
    // set first evaluation
    evals[0] = p1;
    // get the difference between the two evaluations
    let t = mont.sub(p1, p0);
    // compute the rest iteratively
    for i in 1..n {
        evals[i] = mont.add(evals[i - 1], t);
    }
    evals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prime_128::{rand_elem, rand_prime};
    use rand::SeedableRng;
    use stream::bigvec::BigVec;
    #[test]
    fn test_mle_eval() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let p = rand_prime(&mut rng);
        let mont = FieldMont::new(p);
        let range_1 = 0..50;
        let range_2 = 102..240;
        let mut evals_sparse = Vec::new();
        let mut evals_dense = vec![mont.zero(); 256];
        for i in range_1.clone() {
            evals_sparse.push(mont.to_mont(i as u128));
            evals_dense[i] = mont.to_mont(i as u128);
        }
        for i in range_2.clone() {
            evals_sparse.push(mont.to_mont(i as u128));
            evals_dense[i] = mont.to_mont(i as u128);
        }
        let evals_sparse = BigVec::from_vec(evals_sparse);
        let evals_dense = BigVec::from_vec(evals_dense);
        let mut mle_sparse = MLE::from_buffer(evals_sparse, vec![range_1, range_2]);
        let mut mle_dense = MLE::from_buffer(evals_dense, vec![0..256]);
        let x = Vec::from_iter((0..8).map(|_| mont.to_mont(rand_elem(mont.modulus(), &mut rng))));
        // test evals
        let eval_sparse = mle_sparse.eval(&x, &mont);
        let eval_dense = mle_dense.eval(&x, &mont);
        // print evaluations
        println!("Sparse eval: {:?}", eval_sparse);
        println!("Dense eval: {:?}", eval_dense);
        // check that evals are equal
        assert_eq!(eval_sparse, eval_dense);
    }
}
