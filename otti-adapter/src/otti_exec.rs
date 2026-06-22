use model::AFloat;
use parse::{generalized::I256, mat::Matrix};
use spain::{
    inputs::{Metadata, R1CSMatrices},
    traits::R1CSInstance,
};
use std::time::Instant;

use crate::cons_adapter::{filter_failing_constraints, get_otti_r1cs, get_otti_r1cs_matrices};

#[derive(Clone, Debug)]
pub struct OttiExec {
    matrices: R1CSMatrices<I256>,
    z: Matrix<I256>,
    num_public_values: usize,
    full_witness: Option<Matrix<I256>>,
}

impl OttiExec {
    pub fn new(model_name: &str) -> Self {
        eprintln!("Initializing OttiExec for model: {}", model_name);
        let total_start = Instant::now();
        let (matrices, z) = get_otti_r1cs_matrices(model_name);
        let (matrices, num_failed) = filter_failing_constraints(&matrices, &z);
        eprintln!("OttiExec::new: loading raw R1CS metadata");
        let r1cs = get_otti_r1cs(model_name);
        let num_public_values = r1cs.inputs.len() + 1;
        let total_constraints = matrices.a.height() + num_failed;
        let failed_pct = (num_failed as f64) * 100.0 / (total_constraints as f64);
        eprintln!(
            "otti_exec filtered constraints: removed {} ({:.4}%), kept {}",
            num_failed,
            failed_pct,
            matrices.a.height()
        );
        eprintln!("OttiExec::new: finished in {:?}", total_start.elapsed());
        Self {
            matrices,
            z,
            num_public_values,
            full_witness: None,
        }
    }
}

impl R1CSInstance<AFloat, I256> for OttiExec {
    fn compute_commit_witness(&mut self, _scale_factor: AFloat, batch_size: usize) -> Matrix<I256> {
        let mut z = self.z.repeat_column(batch_size);
        let ranges = self.get_meta().get_ranges();
        z.set_ranges(&ranges);
        self.full_witness = Some(z.clone());
        z.extract_rows(&z.ranges().unwrap()[1])
    }

    fn compute_full_witness(
        &mut self,
        _metadata: &Metadata,
        _random_values: Vec<AFloat>,
        _scale_factor: AFloat,
    ) -> Matrix<I256> {
        self.full_witness.take().unwrap()
    }

    fn get_matrices(
        &self,
        _scale_factor: AFloat,
        randomness: Option<&Vec<I256>>,
    ) -> (
        R1CSMatrices<I256>,
        Option<parse::generalized::InjectionInfo>,
    ) {
        assert!(randomness.is_none(), "no need for randomness for otti");
        (self.matrices.clone(), None)
    }

    fn get_meta(&self) -> Metadata {
        Metadata {
            num_public_values: self.num_public_values,
            num_random_values: 0,
            num_witness_values: self.z.height() - self.num_public_values,
            num_secondary_witness_values: 0,
            num_secondary_constraint_variables: 0,
            primary_output_labels: vec![],
            secondary_output_labels: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::OttiExec;
    use model::AFloat;
    use parse::generalized::I256;
    use spain::simulate::{SpainConfig, stateful_simulate};

    #[test]
    fn test_stateful_simulate_i256_otti_exec() {
        let exec = OttiExec::new("afiro");
        let mut cfg = SpainConfig::default();
        cfg.precision = 256;
        cfg.num_chunks = 4;
        cfg.scale_factor_bits = 0;
        let result = stateful_simulate::<AFloat, _, I256>(exec, Some(cfg));
        result.report_prover_time();
        result.report_verifier_time();
    }
}
