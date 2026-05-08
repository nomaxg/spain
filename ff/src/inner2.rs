// inner polynomial
// The polynomial: a(x)*z(x) + br1(x)*z(x) + cr2(x)*z(x)

use crate::{
    FieldElem, FieldMont,
    poly::cmont::{MLE, lagrange_interpolate},
};

#[derive(Clone, Debug)]
pub struct InnerPoly {
    // MLEs
    ar1br2c: MLE,
    z: MLE,
    // number of variables
    num_vars: usize,
}

impl InnerPoly {
    // constructor from explicit buffer
    pub fn from_buffers(
        mut a: MLE,
        b: &MLE,
        c: &MLE,
        z: MLE,
        r1: FieldElem,
        r2: FieldElem,
        mont: &FieldMont,
    ) -> Self {
        assert!(a.num_vars() == b.num_vars());
        assert!(a.num_vars() == c.num_vars());
        assert!(a.num_vars() == z.num_vars());
        let num_vars = a.num_vars();
        for i in 0..a.evals.len() {
            // scale b and c by r1 and r2
            let br1 = mont.mul(b.evals[i], r1);
            let cr2 = mont.mul(c.evals[i], r2);
            // combine a, br1, and cr2
            a.evals[i] = mont.add(mont.add(a.evals[i], br1), cr2);
        }
        Self {
            ar1br2c: a,
            z,
            num_vars,
        }
    }
    pub fn degree(&self) -> usize {
        2
    }
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
    // as a univariate degree 2 polynomial in outer variable
    // as evaluations at 1, 2 (0 omitted)
    pub fn as_poly(&self, mont: &FieldMont) -> Vec<FieldElem> {
        let mut res = vec![mont.zero(); 2];
        for i in 0..self.z.evals.len() / 2 {
            // skip cases where all z evaluations are zero
            if self.z.evals[2 * i] == mont.zero() && self.z.evals[2 * i + 1] == mont.zero() {
                continue;
            }
            let p0 = self.ar1br2c.evals[2 * i];
            let p1 = self.ar1br2c.evals[2 * i + 1];
            let z0 = self.z.evals[2 * i];
            let z1 = self.z.evals[2 * i + 1];
            res[0] = mont.add(res[0], mont.mul(p1, z1));
            let p2 = mont.sub(mont.add(p1, p1), p0);
            let z2 = mont.sub(mont.add(z1, z1), z0);
            res[1] = mont.add(res[1], mont.mul(p2, z2));
        }
        res
    }
    // bind outer variable to the value x (in montgomery form)
    pub fn bind(&mut self, x: FieldElem, mont: &FieldMont) {
        // assert there is a variable to bind
        assert!(self.num_vars > 0);
        // bind each of the MLEs
        self.ar1br2c.bind(x, mont);
        self.z.bind(x, mont);
        // decrement number of variables
        self.num_vars -= 1;
    }
    // returns a, br1, cr2, and z evaluations at the bound point
    pub fn final_evals(&self) -> Vec<FieldElem> {
        // assert there are no variables to bind
        assert!(self.num_vars == 0);
        vec![self.ar1br2c.evals[0], self.z.evals[0]]
    }
    // check final evals
    pub fn check_final_evals(
        mont: &FieldMont,
        p: &[FieldElem],
        r: FieldElem,
        _aux: &[FieldElem],
        evals: &[FieldElem],
    ) -> Result<(), String> {
        // check that p(r) = evals[0] * evals[1]
        let actual = mont.mul(evals[0], evals[1]);
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
