// outer polynomial for Spartan
// The polynomial: eq(tau, x) * (Az(x) * Bz(x) - Cz(x))

use std::ops::Range;
use stream::bigvec::BigVec;

use crate::{
    FieldElem, FieldMont,
    poly::cmont::{MLE, lagrange_interpolate, lin_batch_eval},
};

// generates the eq polynomial eq(r, x)
// code slightly duplicated with build_eval_tbl in verifier.rs
fn build_eq_tbl(mont: &FieldMont, ranges: Vec<Range<usize>>, tau: &[FieldElem]) -> MLE {
    let n = tau.len();
    let mut eq = MLE::from_buffer(BigVec::new(1usize << n).unwrap(), ranges);

    eq.evals[0] = mont.one();
    let mut cur_len = 1;

    for &ri in tau.iter().rev() {
        let one = mont.one();
        let omr = mont.sub(one, ri); // 1 - ri

        for j in (0..cur_len).rev() {
            let t = eq.evals[j];
            let base_idx = 2 * j;
            eq.evals[base_idx] = mont.mul(t, omr); // t * (1 - ri)
            eq.evals[base_idx + 1] = mont.mul(t, ri); // t * ri
        }

        cur_len <<= 1;
    }

    eq
}

// get eq(tau, x) for a particular x
// eq(tau, x) = prod_i x[i] * tau[i] + (1 - x[i]) * (1 - tau[i])
//            = prod_i 1 - x[i] - tau[i] + 2 * x[i] * tau[i]
fn eval_eq_at(mont: &FieldMont, tau: &[FieldElem], x: &[FieldElem]) -> FieldElem {
    assert_eq!(tau.len(), x.len());
    let mut res = mont.one();
    for (ri, xi) in tau.iter().zip(x.iter()) {
        let p = mont.mul(*ri, *xi); // ri * xi
        let dp = mont.add(p, p); // 2 * ri * xi
        let t = mont.add(mont.sub(mont.sub(mont.one(), *ri), *xi), dp); // 1 - ri - xi + 2 * ri * xi
        res = mont.mul(res, t);
    }
    res
}

#[derive(Clone, Debug)]
pub struct OuterPolyEq {
    // MLEs
    eq: MLE,
    az: MLE,
    bz: MLE,
    cz: MLE,
    // number of variables
    num_vars: usize,
}

impl OuterPolyEq {
    // constructor from explicit buffer
    pub fn from_buffers(az: MLE, bz: MLE, cz: MLE, tau: &[FieldElem], mont: &FieldMont) -> Self {
        assert!(az.num_vars() == bz.num_vars());
        assert!(az.num_vars() == cz.num_vars());
        assert!(az.num_vars() == tau.len());
        let num_vars = az.num_vars();
        let eq = build_eq_tbl(mont, az.ranges.clone(), tau);
        Self {
            eq,
            az,
            bz,
            cz,
            num_vars,
        }
    }
    pub fn degree(&self) -> usize {
        3
    }
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
    // as a univariate degree 3 polynomial in outer variable
    // as evaluations at 1, 2, 3 (0 is omitted)
    pub fn as_poly(&self, mont: &FieldMont) -> Vec<FieldElem> {
        let mut res = vec![mont.zero(); 3];
        for i in 0..self.az.evals.len() / 2 {
            // if all a, b, c evaluations are zero, skip
            /*if self.az.evals[2 * i] == mont.zero()
                && self.az.evals[2 * i + 1] == mont.zero()
                && self.bz.evals[2 * i] == mont.zero()
                && self.bz.evals[2 * i + 1] == mont.zero()
                && self.cz.evals[2 * i] == mont.zero()
                && self.cz.evals[2 * i + 1] == mont.zero()
            {
                continue;
            }*/
            // get evaluations of MLEs at 1 through 3
            let eq_vals = lin_batch_eval(self.eq.evals[2 * i], self.eq.evals[2 * i + 1], 3, mont);
            let az_vals = lin_batch_eval(self.az.evals[2 * i], self.az.evals[2 * i + 1], 3, mont);
            let bz_vals = lin_batch_eval(self.bz.evals[2 * i], self.bz.evals[2 * i + 1], 3, mont);
            let cz_vals = lin_batch_eval(self.cz.evals[2 * i], self.cz.evals[2 * i + 1], 3, mont);
            // combine evaluations
            for j in 0..3 {
                // poly[i] = eq[i] * (az_vals[i] * bz_vals[i] - cz_vals[i])
                let d = mont.mul(az_vals[j], bz_vals[j]);
                let s = mont.sub(d, cz_vals[j]);
                let e = mont.mul(eq_vals[j], s);
                res[j] = mont.add(res[j], e);
            }
        }
        res
    }
    // bind outer variable to the value x (in montgomery form)
    pub fn bind(&mut self, x: FieldElem, mont: &FieldMont) {
        // assert there is a variable to bind
        assert!(self.num_vars > 0);
        // bind each of the MLEs
        self.eq.bind(x, mont);
        self.az.bind(x, mont);
        self.bz.bind(x, mont);
        self.cz.bind(x, mont);
        // decrement number of variables
        self.num_vars -= 1;
    }
    // returns eq, az, bz, cz at random point at the end of the protocol
    pub fn final_evals(&self) -> Vec<FieldElem> {
        // assert there are no variables to bind
        assert!(self.num_vars == 0);
        vec![
            self.eq.evals[0],
            self.az.evals[0],
            self.bz.evals[0],
            self.cz.evals[0],
        ]
    }
    // check final evals
    pub fn check_final_evals(
        mont: &FieldMont,
        p: &[FieldElem],
        r: FieldElem,
        aux: &[FieldElem], // tau, r
        evals: &[FieldElem],
    ) -> Result<(), String> {
        // check that p(r) = evals[0] * (evals[1] * evals[2] - evals[3])
        let actual = mont.mul(evals[0], mont.sub(mont.mul(evals[1], evals[2]), evals[3]));
        let expected = lagrange_interpolate(p, r, mont);
        if actual != expected {
            return Err(format!(
                "OuterEq final evaluations did not match: expected {:?}, got {:?}",
                expected, actual
            ));
        }
        // check that eq(tau, xs) = evals[0]
        let tau = &aux[..aux.len() / 2];
        let x = &aux[aux.len() / 2..];
        let eq_expected = eval_eq_at(mont, tau, x);
        if eq_expected == evals[0] {
            Ok(())
        } else {
            Err(format!(
                "eq(tau, r) did not match: expected {:?}, got {:?}",
                eq_expected, evals[0]
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FieldMont, poly::cmont::MLE, prime_128::rand_prime};
    use rand::SeedableRng;
    use stream::bigvec::BigVec;
    #[test]
    fn test_eval_eq_at() {
        let num_vars = 5;
        let size = 1usize << num_vars;
        // create random montgomery context
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let p = rand_prime(&mut rng);
        let mont = FieldMont::new(p);
        // create non-zero dummy MLEs for a, b, and c
        let a = MLE::from_buffer(
            BigVec::from_vec((0..size).map(|i| mont.to_mont((i as u128) + 1)).collect()),
            vec![0..size],
        );
        let b = MLE::from_buffer(
            BigVec::from_vec((0..size).map(|i| mont.to_mont((i as u128) + 3)).collect()),
            vec![0..size],
        );
        let c = MLE::from_buffer(
            BigVec::from_vec((0..size).map(|i| mont.to_mont((i as u128) + 7)).collect()),
            vec![0..size],
        );
        // create random tau
        let tau: Vec<_> = (0..num_vars)
            .map(|_| mont.to_mont(rand::random::<u128>() % p))
            .collect();
        // create outer polynomial
        let mut outer = OuterPolyEq::from_buffers(a.clone(), b.clone(), c.clone(), &tau, &mont);
        // get random r
        let r: Vec<_> = (0..num_vars)
            .map(|_| mont.to_mont(rand::random::<u128>() % p))
            .collect();
        // get expected sum
        let mut expected = mont.zero();
        for i in 0..size {
            expected = mont.add(
                expected,
                mont.mul(
                    outer.eq.evals[i],
                    mont.sub(
                        mont.mul(outer.az.evals[i], outer.bz.evals[i]),
                        outer.cz.evals[i],
                    ),
                ),
            );
        }
        // bind r
        let mut p = vec![];
        for &ri in r.iter() {
            p = outer.as_poly(&mont);
            p.insert(0, mont.sub(expected, p[0]));
            outer.bind(ri, &mont);
            expected = lagrange_interpolate(&p, ri, &mont);
        }
        // get final evals
        let evals = outer.final_evals();
        // copy r to the end of tau (a quirk of how we need to pass the data into the check function)
        let aux = [tau.clone(), r.clone()].concat();
        // check final evals
        let check = OuterPolyEq::check_final_evals(&mont, &p, *r.last().unwrap(), &aux, &evals);
        assert!(check.is_ok());
    }
    #[test]
    fn test_eval_eq_at_range() {
        let num_vars = 5;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let p = rand_prime(&mut rng);
        let mont = FieldMont::new(p);

        let a = MLE::from_buffer(BigVec::new((1usize << num_vars) - 2).unwrap(), vec![0..30]);
        let b = MLE::from_buffer(BigVec::new((1usize << num_vars) - 2).unwrap(), vec![0..30]);
        let c = MLE::from_buffer(BigVec::new((1usize << num_vars) - 2).unwrap(), vec![0..30]);
        let tau: Vec<_> = (0..num_vars)
            .map(|_| mont.to_mont(rand::random::<u128>() % p))
            .collect();

        let _ = OuterPolyEq::from_buffers(a, b, c, &tau, &mont);
    }
}
