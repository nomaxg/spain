// outer polynomial
// The polynomial: (Az(x) * Bz(x) - Cz(x))^2

use crate::{
    FieldElem, FieldMont,
    poly::cmont::{MLE, lagrange_interpolate, lin_batch_eval},
};

#[derive(Clone, Debug)]
pub struct OuterPoly {
    // MLEs
    az: MLE,
    bz: MLE,
    cz: MLE,
    // number of variables
    num_vars: usize,
}

impl OuterPoly {
    // constructor from explicit buffer
    pub fn from_buffers(az: MLE, bz: MLE, cz: MLE) -> Self {
        assert!(az.num_vars() == bz.num_vars());
        assert!(az.num_vars() == cz.num_vars());
        let num_vars = az.num_vars();
        Self {
            az,
            bz,
            cz,
            num_vars,
        }
    }
    pub fn degree(&self) -> usize {
        4
    }
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
    // as a univariate degree 4 polynomial in outer variable
    // as evaluations at 1, 2, 3, 4 (0 is omitted)
    pub fn as_poly(&self, mont: &FieldMont) -> Vec<FieldElem> {
        let mut res = vec![mont.zero(); 4];
        for i in 0..self.az.evals.len() / 2 {
            // if all evaluations are zero, skip
            if self.az.evals[2 * i] == mont.zero()
                && self.az.evals[2 * i + 1] == mont.zero()
                && self.bz.evals[2 * i] == mont.zero()
                && self.bz.evals[2 * i + 1] == mont.zero()
                && self.cz.evals[2 * i] == mont.zero()
                && self.cz.evals[2 * i + 1] == mont.zero()
            {
                continue;
            }
            // get evaluations of MLEs at 0 through 4
            let az_vals = lin_batch_eval(self.az.evals[2 * i], self.az.evals[2 * i + 1], 4, mont);
            let bz_vals = lin_batch_eval(self.bz.evals[2 * i], self.bz.evals[2 * i + 1], 4, mont);
            let cz_vals = lin_batch_eval(self.cz.evals[2 * i], self.cz.evals[2 * i + 1], 4, mont);
            // combine evaluations
            for j in 0..4 {
                // poly[i] = (az_vals[i] * bz_vals[i] - cz_vals[i])^2
                let d = mont.mul(az_vals[j], bz_vals[j]);
                let s = mont.sub(d, cz_vals[j]);
                res[j] = mont.add(res[j], mont.sqr(s));
            }
        }
        res
    }
    // bind outer variable to the value x (in montgomery form)
    pub fn bind(&mut self, x: FieldElem, mont: &FieldMont) {
        // assert there is a variable to bind
        assert!(self.num_vars > 0);
        // bind each of the MLEs
        self.az.bind(x, mont);
        self.bz.bind(x, mont);
        self.cz.bind(x, mont);
        // decrement number of variables
        self.num_vars -= 1;
    }
    // returns az, bz, cz at random point at the end of the protocol
    pub fn final_evals(&self) -> Vec<FieldElem> {
        // assert there are no variables to bind
        assert!(self.num_vars == 0);
        vec![self.az.evals[0], self.bz.evals[0], self.cz.evals[0]]
    }
    // check final evals
    pub fn check_final_evals(
        mont: &FieldMont,
        p: &[FieldElem],
        r: FieldElem,
        _aux: &[FieldElem],
        evals: &[FieldElem],
    ) -> Result<(), String> {
        // check that p(r) = (evals[0] * evals[1] - evals[2])^2
        let actual = mont.sqr(mont.sub(mont.mul(evals[0], evals[1]), evals[2]));
        let expected = lagrange_interpolate(p, r, mont);
        if actual == expected {
            Ok(())
        } else {
            Err(format!(
                "Final evaluations did not match: expected {:?}, got {:?}",
                expected, actual
            ))
        }
    }
}
