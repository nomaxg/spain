// inner polynomial
// The polynomial: a(x)*z(x) + br1(x)*z(x) + cr2(x)*z(x)

use crate::{
    FieldElem, FieldMont,
    poly::cmont::{MLE, lagrange_interpolate, lin_batch_eval},
};

#[derive(Clone, Debug)]
pub struct InnerPoly {
    // MLEs
    a: MLE,
    b: MLE,
    c: MLE,
    z: MLE,
    // scaling factors
    r1: FieldElem, // for br1
    r2: FieldElem, // for cr2
    // number of variables
    num_vars: usize,
}

impl InnerPoly {
    // constructor from explicit buffer
    pub fn from_buffers(a: MLE, b: MLE, c: MLE, z: MLE, r1: FieldElem, r2: FieldElem) -> Self {
        assert!(a.num_vars() == b.num_vars());
        assert!(a.num_vars() == c.num_vars());
        assert!(a.num_vars() == z.num_vars());
        let num_vars = a.num_vars();
        Self {
            a,
            b,
            c,
            z,
            r1,
            r2,
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
    // as evaluations at 0, 1, 2
    pub fn as_poly(&self, mont: &FieldMont) -> Vec<FieldElem> {
        let mut tmp = vec![vec![mont.zero(); 3]; 3];
        for i in 0..self.a.evals.len() / 2 {
            // skip cases where all z evaluations are zero
            if self.z.evals[2 * i] == mont.zero() && self.z.evals[2 * i + 1] == mont.zero() {
                continue;
            }
            // get evaluations of MLEs at 0 through 2
            let a_vals = lin_batch_eval(self.a.evals[2 * i], self.a.evals[2 * i + 1], 2, mont);
            let b_vals = lin_batch_eval(self.b.evals[2 * i], self.b.evals[2 * i + 1], 2, mont);
            let c_vals = lin_batch_eval(self.c.evals[2 * i], self.c.evals[2 * i + 1], 2, mont);
            let z_vals = lin_batch_eval(self.z.evals[2 * i], self.z.evals[2 * i + 1], 2, mont);
            // combine evaluations
            for j in 0..3 {
                let az = mont.mul(a_vals[j], z_vals[j]);
                let bz = mont.mul(b_vals[j], z_vals[j]);
                let cz = mont.mul(c_vals[j], z_vals[j]);
                tmp[0][j] = mont.add(tmp[0][j], az);
                tmp[1][j] = mont.add(tmp[1][j], mont.mul(self.r1, bz));
                tmp[2][j] = mont.add(tmp[2][j], mont.mul(self.r2, cz));
            }
        }
        // combine evaluations and return
        let mut res = vec![mont.zero(); 3];
        for i in 0..3 {
            res[i] = tmp[0][i];
            res[i] = mont.add(res[i], tmp[1][i]);
            res[i] = mont.add(res[i], tmp[2][i]);
        }
        res
    }
    // bind outer variable to the value x (in montgomery form)
    pub fn bind(&mut self, x: FieldElem, mont: &FieldMont) {
        // assert there is a variable to bind
        assert!(self.num_vars > 0);
        // bind each of the MLEs
        self.a.bind(x, mont);
        self.b.bind(x, mont);
        self.c.bind(x, mont);
        self.z.bind(x, mont);
        // decrement number of variables
        self.num_vars -= 1;
    }
    // returns a, br1, cr2, and z evaluations at the bound point
    pub fn final_evals(&self) -> Vec<FieldElem> {
        // assert there are no variables to bind
        assert!(self.num_vars == 0);
        vec![
            self.a.evals[0],
            self.b.evals[0],
            self.c.evals[0],
            self.z.evals[0],
        ]
    }
    // check final evals
    pub fn check_final_evals(
        mont: &FieldMont,
        p: &[FieldElem],
        r: FieldElem,
        aux: &[FieldElem],
        evals: &[FieldElem],
    ) -> Result<(), String> {
        // check that p(r) = (evals[0] + r1 * evals[1] + r2 * evals[2]) * evals[3]
        let actual = mont.mul(
            mont.add(
                mont.add(evals[0], mont.mul(aux[0], evals[1])),
                mont.mul(aux[1], evals[2]),
            ),
            evals[3],
        );
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
