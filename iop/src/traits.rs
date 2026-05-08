use ff::{FieldElem, FieldMont, inner2::InnerPoly, outer::OuterPoly};

// multivariate polynomial in the context of the sum-check protocol
pub trait SumCheckPoly {
    // degree of the polynomial
    fn degree(&self) -> usize;
    // number of variables in the polynomial
    fn num_vars(&self) -> usize;
    // as a univariate polynomial in outer variable
    // as evaluations at 1, 2, 3, 4 etc., leaving out 0
    fn as_poly(&self, mont: &FieldMont) -> Vec<FieldElem>;
    // bind outer variable to the value x (in montgomery form)
    fn bind(&mut self, x: FieldElem, mont: &FieldMont);
    // final evals
    // when there are no variables left to bind
    // returns MLE evaluations at the bound point
    fn final_evals(&self) -> Vec<FieldElem>;
    // check final evals
    fn check_final_evals(
        mont: &FieldMont,
        p: &[FieldElem],
        r: FieldElem,
        aux: &[FieldElem],
        evals: &[FieldElem],
    ) -> Result<(), String>;
}

impl SumCheckPoly for OuterPoly {
    fn degree(&self) -> usize {
        self.degree()
    }
    fn num_vars(&self) -> usize {
        self.num_vars()
    }
    fn as_poly(&self, mont: &FieldMont) -> Vec<FieldElem> {
        self.as_poly(mont)
    }
    fn bind(&mut self, x: FieldElem, mont: &FieldMont) {
        self.bind(x, mont);
    }
    fn final_evals(&self) -> Vec<FieldElem> {
        self.final_evals()
    }
    fn check_final_evals(
        mont: &FieldMont,
        p: &[FieldElem],
        r: FieldElem,
        aux: &[FieldElem],
        evals: &[FieldElem],
    ) -> Result<(), String> {
        OuterPoly::check_final_evals(mont, p, r, aux, evals)
    }
}

impl SumCheckPoly for InnerPoly {
    fn degree(&self) -> usize {
        self.degree()
    }
    fn num_vars(&self) -> usize {
        self.num_vars()
    }
    fn as_poly(&self, mont: &FieldMont) -> Vec<FieldElem> {
        self.as_poly(mont)
    }
    fn bind(&mut self, x: FieldElem, mont: &FieldMont) {
        self.bind(x, mont);
    }
    fn final_evals(&self) -> Vec<FieldElem> {
        self.final_evals()
    }
    fn check_final_evals(
        mont: &FieldMont,
        p: &[FieldElem],
        r: FieldElem,
        aux: &[FieldElem],
        evals: &[FieldElem],
    ) -> Result<(), String> {
        InnerPoly::check_final_evals(mont, p, r, aux, evals)
    }
}
