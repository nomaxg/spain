use model::AFloat;
use parse::mat::Matrix;
use spain::{
    inputs::{Metadata, R1CSMatrices},
    traits::R1CSInstance,
};

use crate::{
    builder::R1CSBuilder,
    fluid::{flatten_inputs, initial_state, simulate_ops},
    r1cs::{ConstraintGenerator, R1CSExec, WitnessGenerator},
};

#[derive(Clone, Debug)]
pub struct PhysicsExampleExecutor {
    steps: usize,
    witness: Option<Matrix<i128>>,
}

impl Default for PhysicsExampleExecutor {
    fn default() -> Self {
        Self {
            steps: 20,
            witness: None,
        }
    }
}

impl PhysicsExampleExecutor {
    pub fn new(steps: usize) -> Self {
        Self {
            steps,
            witness: None,
        }
    }

    fn build_witness_generator(&self) -> WitnessGenerator {
        let inputs = flatten_inputs(&initial_state());
        let mut wit_gen = WitnessGenerator::new_from_tensored_inputs(inputs);
        let _ = simulate_ops(&mut wit_gen, self.steps);
        wit_gen
    }

    fn build_r1cs(&self) -> R1CSBuilder {
        let mut cons_gen = ConstraintGenerator::new();
        let _ = simulate_ops(&mut cons_gen, self.steps);
        cons_gen.finish()
    }
}

impl R1CSInstance<AFloat, i128> for PhysicsExampleExecutor {
    fn compute_commit_witness(&mut self, scale_factor: AFloat, batch_size: usize) -> Matrix<i128> {
        let ret = self
            .build_witness_generator()
            .witness_int(scale_factor)
            .repeat_column(batch_size);
        self.witness = Some(ret.clone());
        ret.extract_rows(&ret.ranges().unwrap()[1])
    }

    fn compute_full_witness(
        &mut self,
        _metadata: &Metadata,
        _random_values: Vec<AFloat>,
        _scale_factor: AFloat,
    ) -> Matrix<i128> {
        self.witness.take().unwrap()
    }

    fn get_matrices(
        &self,
        scale_factor: AFloat,
        _randomness: Option<&Vec<i128>>,
    ) -> (
        R1CSMatrices<i128>,
        Option<parse::generalized::InjectionInfo>,
    ) {
        let (a, b, c) = self.build_r1cs().to_r1cs_int(scale_factor);
        (R1CSMatrices { a, b, c }, None)
    }

    fn get_meta(&self) -> Metadata {
        self.build_witness_generator().metadata()
    }
}
