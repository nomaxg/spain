use model::AFloat;
use parse::mat::Matrix;
use spain::{
    inputs::{Metadata, R1CSMatrices},
    traits::R1CSInstance,
};

use crate::{
    builder::R1CSBuilder,
    fluid::{DEFAULT_GRID_SIZE, flatten_inputs, initial_state, simulate_ops, simulate_ops_with_progress},
    r1cs::{ConstraintGenerator, R1CSExec, WitnessGenerator},
};

#[derive(Clone, Debug)]
pub struct PhysicsExampleExecutor {
    grid_size: usize,
    num_steps: usize,
    arith_progress: bool,
    witness: Option<Matrix<i128>>,
}

impl Default for PhysicsExampleExecutor {
    fn default() -> Self {
        Self {
            grid_size: DEFAULT_GRID_SIZE,
            num_steps: 20,
            arith_progress: false,
            witness: None,
        }
    }
}

impl PhysicsExampleExecutor {
    pub fn new(grid_size: usize, num_steps: usize) -> Self {
        assert!(grid_size >= 2, "grid_size must be >= 2");
        Self {
            grid_size,
            num_steps,
            arith_progress: false,
            witness: None,
        }
    }

    pub fn with_arith_progress(mut self, arith_progress: bool) -> Self {
        self.arith_progress = arith_progress;
        self
    }

    fn build_witness_generator(&self) -> WitnessGenerator {
        let inputs = flatten_inputs(&initial_state(self.grid_size));
        let mut wit_gen = WitnessGenerator::new_from_tensored_inputs(inputs);
        let _ = simulate_ops(&mut wit_gen, self.num_steps, self.grid_size);
        wit_gen
    }

    fn build_r1cs(&self) -> R1CSBuilder {
        let mut cons_gen = ConstraintGenerator::new();
        if self.arith_progress {
            eprintln!(
                "Starting physics arithmetization: {} steps, grid size {}",
                self.num_steps, self.grid_size
            );
            let _ = simulate_ops_with_progress(
                &mut cons_gen,
                self.num_steps,
                self.grid_size,
                |step, total| eprintln!("Arithmetizing physics step {step}/{total}"),
            );
            eprintln!("Finished physics arithmetization");
        } else {
            let _ = simulate_ops(&mut cons_gen, self.num_steps, self.grid_size);
        }
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
