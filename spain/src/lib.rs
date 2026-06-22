pub mod actor;
pub mod broker;
pub mod executor;
pub mod inputs;
pub mod prover;
pub mod simulate;
pub mod synthetic;
pub mod timer;
pub mod traits;
pub mod verifier;
pub mod witness_gen;
use std::time::Duration;

use model::HighPrecision;
use parse::{
    generalized::HighPrecisionInt,
    mat::{Matrix, MatrixData},
};
use stream::bigvec::BigVec;

#[derive(Debug, Clone, Default)]
pub struct EvaluationResult {
    // Global metrics/information
    pub model_name: String,
    pub total_protocol_time: Duration,
    pub batch_size: usize,
    pub num_constraints: usize,
    // Prover metrics
    pub proof_size: usize,
    pub total_prover_time: Duration,
    pub total_prover_actor_time: Duration,
    // Verifier metrics
    pub total_verifier_time: Duration,
    pub total_verifier_actor_time: Duration,
    // Phase breakdown: Prover
    pub prover_preprocessing_time: Duration,
    pub prover_compute_witness: Duration,
    pub prover_compute_square_error_time: Duration,
    pub prover_prepare_outer_sc_time: Duration,
    pub prover_run_outer_sc_time: Duration,
    pub prover_prepare_inner_sc_time: Duration,
    pub prover_run_inner_sc_time: Duration,
    pub prover_poly_commit_time: Duration,
    pub prover_poly_eval_time: Duration,
    pub prover_misc_time: Duration,
    pub prover_serialization_time: Duration,
    pub prover_bytes_sent: usize,
    // Phase breakdown: Verifier
    pub verifier_setup_time: Duration,
    pub verifier_epsilon_check_time: Duration,
    pub verifier_sample_time: Duration,
    pub verifier_run_outer_sc_time: Duration,
    pub verifier_run_inner_sc_time: Duration,
    pub verifier_poly_eval_time: Duration,
    pub verifier_spark_time: Duration,
    pub verifier_claim_interpolate_time: Duration,
    pub verifier_smart_eval_time: Duration,
    pub verifier_misc_time: Duration,
    pub verifier_serialization_time: Duration,
    pub verifier_bytes_sent: usize,
}

impl EvaluationResult {
    pub fn calc_totals(&mut self) {
        self.total_prover_time = self.prover_compute_square_error_time
            + self.prover_prepare_outer_sc_time
            + self.prover_run_outer_sc_time
            + self.prover_prepare_inner_sc_time
            + self.prover_run_inner_sc_time
            + self.prover_poly_commit_time
            + self.prover_poly_eval_time
            + self.prover_compute_witness
            + self.prover_misc_time;
        self.total_verifier_time = self.verifier_sample_time
            + self.verifier_run_outer_sc_time
            + self.verifier_run_inner_sc_time
            + self.verifier_poly_eval_time
            + self.verifier_spark_time
            + self.verifier_claim_interpolate_time
            + self.verifier_epsilon_check_time
            + self.verifier_smart_eval_time
            + self.verifier_misc_time;
    }

    pub fn report_prover_time(&self) {
        eprintln!("Prover time: {:#?} \n", self.total_prover_time);
        eprintln!(
            "Prover actorized time: {:#?} \n",
            self.total_prover_actor_time
        );
    }

    pub fn report_verifier_time(&self) {
        eprintln!("Verifier time: {:#?} \n", self.total_verifier_time);
        eprintln!(
            "Verifier actorized time: {:#?} \n",
            self.total_verifier_actor_time
        );
        if self.batch_size > 1 {
            eprintln!(
                "Verifier time per instance: {:#?} \n",
                self.total_verifier_time / self.batch_size as u32
            );
        }
    }

    pub fn report_prover_phases(&self) {
        eprintln!("Prover phase breakdown:");
        eprintln!("  Compute w: {:#?}", self.prover_compute_witness);
        eprintln!("  Compute J: {:#?}", self.prover_compute_square_error_time);
        eprintln!(
            "  Sum-check: {:#?}",
            self.prover_prepare_inner_sc_time
                + self.prover_run_inner_sc_time
                + self.prover_prepare_outer_sc_time
                + self.prover_run_outer_sc_time
                + self.prover_misc_time
        );
        eprintln!("  Commit to w: {:#?}", self.prover_poly_commit_time);
        eprintln!("  Open commitment: {:#?}", self.prover_poly_eval_time);
        eprintln!("  Time total: {:#?}", self.total_prover_time);
        eprintln!("Proof size: {:#?}", self.prover_bytes_sent)
    }

    pub fn report_verifier_phases(&self) {
        eprintln!("Verifier phase breakdown:");
        eprintln!("  Setup: {:#?}", self.verifier_setup_time);
        eprintln!(
            "  Steps 4-6: {:#?}",
            self.verifier_epsilon_check_time
                + self.verifier_sample_time
                + self.verifier_run_inner_sc_time
                + self.verifier_run_outer_sc_time
                + self.verifier_misc_time
        );
        eprintln!("  Open commitment: {:#?}", self.verifier_poly_eval_time);
        eprintln!("  Miscellaneous: {:#?}", self.verifier_misc_time);
        eprintln!(
            "  Test w consistency: {:#?}",
            self.verifier_claim_interpolate_time
        );
        eprintln!(
            "  A/B/C matrix evaluation: {:#?}",
            self.verifier_smart_eval_time
        );
        eprintln!("  Time total: {:#?}", self.total_verifier_time);
    }

    pub fn zklp_prover_time(&self) -> Duration {
        self.prover_prepare_inner_sc_time
            + self.prover_prepare_outer_sc_time
            + self.prover_run_outer_sc_time
            + self.prover_run_inner_sc_time
            + self.prover_poly_commit_time
            + self.prover_poly_eval_time
            + self.prover_misc_time
    }

    pub fn zklp_verifier_time(&self) -> Duration {
        self.verifier_sample_time
            + self.verifier_run_outer_sc_time
            + self.verifier_run_inner_sc_time
            + self.verifier_poly_eval_time
            + self.verifier_misc_time
            + self.verifier_claim_interpolate_time
            + self.verifier_smart_eval_time
    }
}

#[derive(Debug, Clone, Default)]
pub struct Witness<T: HighPrecision> {
    pub inputs: Vec<(T, String)>,
    pub outputs: Vec<(T, String)>,
    pub random_vars: Vec<(T, String)>,
    pub primary_witness: Vec<(T, String)>,
    pub secondary_witness: Vec<(T, String)>,
    pub secondary_constraints: Vec<(T, String)>,
}

#[allow(clippy::len_without_is_empty)]
impl<T: HighPrecision> Witness<T> {
    pub fn len(&self) -> usize {
        self.inputs.len()
            + self.outputs.len()
            + self.random_vars.len()
            + self.primary_witness.len()
            + self.secondary_witness.len()
            + self.secondary_constraints.len()
    }

    pub fn extend(&mut self, label: &str, value: T, comment: &str) {
        if label == "witness" {
            self.secondary_witness.push((value, comment.to_string()));
        } else if label.contains("primary") {
            self.primary_witness.push((value, comment.to_string()));
        } else if label == "secondary_constraint" {
            self.secondary_constraints
                .push((value, comment.to_string()));
        } else if label.contains("public") {
            self.outputs.push((value, comment.to_string()));
        } else if label == "input" {
            self.inputs.push((value, comment.to_string()));
        } else if label == "random" {
            self.random_vars.push((value, comment.to_string()));
        } else {
            panic!("unknown value label");
        }
    }

    pub fn into_matrix(self) -> (Matrix<T>, Vec<String>) {
        let len = self.len() + 1;
        let mut names = Vec::new();
        let mut values = BigVec::new(len).unwrap();
        let mut index = 1;
        values[0] = T::from_f64(1_f64).unwrap();
        names.push("one".to_string());
        self.inputs.into_iter().for_each(|(v, name)| {
            names.push(name);
            values[index] = v;
            index += 1;
        });
        self.outputs.into_iter().for_each(|(v, name)| {
            names.push(name);
            values[index] = v;
            index += 1;
        });
        self.random_vars.into_iter().for_each(|(v, name)| {
            names.push(name);
            values[index] = v;
            index += 1;
        });
        self.primary_witness.into_iter().for_each(|(v, name)| {
            names.push(name);
            values[index] = v;
            index += 1;
        });
        self.secondary_witness.into_iter().for_each(|(v, name)| {
            names.push(name);
            values[index] = v;
            index += 1;
        });
        self.secondary_constraints
            .into_iter()
            .for_each(|(v, name)| {
                names.push(name);
                values[index] = v;
                index += 1;
            });
        (
            Matrix::new(
                MatrixData::Dense(values),
                1,
                len,
                None,
                "HighPrecision Witness".to_string(),
            ),
            names,
        )
    }

    pub fn to_scaled_primary_matrix<TI: HighPrecisionInt>(&self, scale_factor: T) -> Matrix<TI> {
        let len = self.primary_witness.len();
        let mut values = BigVec::new(len).unwrap();
        self.primary_witness
            .iter()
            .zip(values.iter_mut())
            .for_each(|((wit, _), val)| *val = TI::from_hp(wit.clone() * scale_factor.clone()));
        Matrix::new(
            MatrixData::Dense(values),
            1,
            len,
            None,
            "HighPrecision Primary Witness (for commit)".to_string(),
        )
    }
}

// count the number of constraints of the form const * var * 1 = 1 * var
// print all constraints along the way
pub fn debug_count_dummy(a: &Matrix<i64>, b: &Matrix<i64>, c: &Matrix<i64>, print: bool) {
    let mut count = 0;
    match (a.data(), b.data(), c.data()) {
        (MatrixData::COO(a_entries), MatrixData::COO(b_entries), MatrixData::COO(c_entries)) => {
            let mut ai = 0;
            let mut bi = 0;
            let mut ci = 0;
            for row in 0..a.height() {
                if print {
                    print!("{} | ", row);
                    print!("0 == (");
                }
                // A
                let mut is_dummy = true;
                let mut first = true;
                while ai < a_entries.len() && a_entries[ai].0 <= row {
                    if a_entries[ai].0 == row {
                        if print {
                            if !first {
                                print!(" + ");
                            }
                            print!("{} * [{}]", a_entries[ai].2, a_entries[ai].1);
                        }
                        if !first {
                            is_dummy = false;
                        }
                        first = false;
                        ai += 1;
                    }
                }
                if print {
                    print!(") * (");
                }
                // B
                first = true;
                while bi < b_entries.len() && b_entries[bi].0 <= row {
                    if b_entries[bi].0 == row {
                        if print {
                            if !first {
                                print!(" + ")
                            }
                            print!("{} * [{}]", b_entries[bi].2, b_entries[bi].1);
                        }
                        if !first || b_entries[bi].1 != 0 {
                            is_dummy = false;
                        }
                        first = false;
                        bi += 1;
                    }
                }
                if print {
                    print!(") - (");
                }
                // C
                first = true;
                while ci < c_entries.len() && c_entries[ci].0 <= row {
                    if c_entries[ci].0 == row {
                        if print {
                            if !first {
                                print!(" + ");
                            }
                            print!("{} * [{}]", c_entries[ci].2, c_entries[ci].1);
                        }
                        if !first || c_entries[ci].1 == 0 || c_entries[ci].2 != 8388608 {
                            is_dummy = false;
                        }
                        first = false;
                        ci += 1;
                    }
                }
                if print {
                    print!(")");
                }
                if is_dummy {
                    if print {
                        print!("!");
                    }
                    count += 1;
                }
                if print {
                    println!();
                }
            }
            println!("Number of dummy constraints: {} / {}", count, a.height());
        }
        _ => panic!("Expected COO matrices for constraints"),
    }
}

// for testing soundness of the protocol, scan one constraint and modify the error to be some
// target value
pub fn increase_error(
    a: &Matrix<f64>,
    b: &Matrix<f64>,
    c: &mut Matrix<f64>,
    z: &Matrix<f64>,
    target: f64,
) {
    let (a_data, b_data, c_data, z_data) = match (a.data(), b.data(), c.mut_data(), z.data()) {
        (
            MatrixData::COO(a_data),
            MatrixData::COO(b_data),
            MatrixData::COO(c_data),
            MatrixData::Dense(z_data),
        ) => (a_data, b_data, c_data, z_data),
        _ => panic!("Invalid matrix types: expected COO, COO, COO (mutable), Dense"),
    };

    let dot_row0 = |coo: &BigVec<(usize, usize, f64)>, z: &BigVec<f64>| -> f64 {
        let mut s: f64 = 0.0;
        let mut i = 0usize;
        while i < coo.len() && coo[i].0 == 0 {
            let (_, col, val) = coo[i];
            s += val * z[col];
            i += 1;
        }
        s
    };

    let az = dot_row0(a_data, z_data);
    let bz = dot_row0(b_data, z_data);
    let cz = dot_row0(c_data, z_data);

    let current_error = az * bz - cz;
    let need = current_error - target;

    // Panic if c has no entries
    if c_data.len() == 0 {
        panic!("cant adjust, c has no entries");
    }

    let (_, col0, ref mut c0_val) = c_data[0];

    // Panic if z contribution is zero
    if z_data[col0] == 0.0 {
        panic!("cant adjust, z[col0] is zero");
    }

    let delta = need / z_data[col0];
    *c0_val += delta;

    let cz_new = cz + delta * z_data[col0];
    let new_error = az * bz - cz_new;
    dbg!(current_error, target, need, delta, new_error);
}
