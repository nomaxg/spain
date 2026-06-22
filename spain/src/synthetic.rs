use model::HighPrecision;
use parse::{
    generalized::{HighPrecisionInt, InjectionInfo},
    mat::Matrix,
};
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    inputs::{Metadata, R1CSMatrices},
    traits::R1CSInstance,
};

#[derive(Clone)]
pub struct SyntheticR1CS<T: HighPrecisionInt> {
    pub num_constraints: usize,
    pub num_inputs: usize,
    pub a: Matrix<T>,
    pub b: Matrix<T>,
    pub c: Matrix<T>,
    pub z: Matrix<T>,
}

impl<T: HighPrecisionInt> SyntheticR1CS<T> {
    pub fn new(num_constraints: usize, num_inputs: usize, _scale_factor_bits: usize) -> Self {
        // synthetic R1CS generation logic copied from Spartan: https://github.com/microsoft/Spartan/blob/3a2c097cab39ffa191560f445440a41ed40db5b3/src/r1cs.rs#L160
        // produce a random assignment z, one variable per constraint
        let z_size = 1 + num_constraints + num_inputs; // 1 for the constant term, and each constraint introduces a new variable
        let mut rng = StdRng::seed_from_u64(synthetic_seed(num_constraints, num_inputs));
        let upper = 1_i128 << 31;
        let lower = 1_i128 << 30;

        let z_values = (0..z_size)
            .map(|_| T::from_i128(rng.random_range(lower..=upper)))
            .collect::<Vec<_>>();

        let mut a_entries: Vec<(usize, usize, T)> = Vec::new();
        let mut b_entries: Vec<(usize, usize, T)> = Vec::new();
        let mut c_entries: Vec<(usize, usize, T)> = Vec::new();

        for i in 0..num_constraints {
            // Derive R1CS entries based on satisfying assignment
            let a_idx = i % z_size;
            let b_idx = (i + 1) % z_size;
            let c_idx = a_idx;

            let alpha = rng.random_range(lower..=upper);
            let beta = rng.random_range(lower..=upper);
            let a_coeff = T::from_i128(alpha);
            let b_coeff = T::from_i128(beta);
            let c_coeff = T::from_i128(alpha * beta) * z_values[b_idx];

            a_entries.push((i, a_idx, a_coeff));
            b_entries.push((i, b_idx, b_coeff));
            c_entries.push((i, c_idx, c_coeff));
        }

        let a = Matrix::from_coo(a_entries, z_size, num_constraints);
        let b = Matrix::from_coo(b_entries, z_size, num_constraints);
        let c = Matrix::from_coo(c_entries, z_size, num_constraints);
        let z = Matrix::from_vec(z_values);

        Self {
            num_constraints,
            num_inputs,
            z,
            a,
            b,
            c,
        }
    }

    pub fn get_metadata(&self) -> Metadata {
        let num_public_values = 1 + self.num_inputs;
        let total_len = self.z.height();
        Metadata {
            num_public_values,
            num_random_values: 0,
            num_witness_values: total_len - num_public_values,
            ..Default::default()
        }
    }
}

fn synthetic_seed(num_constraints: usize, num_inputs: usize) -> u64 {
    (num_constraints as u64).wrapping_mul(num_inputs as u64)
}

impl<P, T> R1CSInstance<P, T> for SyntheticR1CS<T>
where
    P: HighPrecision,
    T: HighPrecisionInt,
{
    fn get_matrices(
        &self,
        _scale_factor: P,
        _randomness: Option<&Vec<T>>,
    ) -> (R1CSMatrices<T>, Option<InjectionInfo>) {
        (
            R1CSMatrices {
                a: self.a.clone(),
                b: self.b.clone(),
                c: self.c.clone(),
            },
            None,
        )
    }

    fn get_meta(&self) -> Metadata {
        self.get_metadata()
    }

    fn compute_commit_witness(&mut self, _scale_factor: P, _batch_size: usize) -> Matrix<T> {
        let ranges = self.get_metadata().get_ranges();
        self.z.set_ranges(&ranges);
        self.z.extract_rows(&ranges[1])
    }

    fn compute_full_witness(
        &mut self,
        _metadata: &Metadata,
        _random_values: Vec<P>,
        _scale_factor: P,
    ) -> Matrix<T> {
        self.z.clone()
    }
}
