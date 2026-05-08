// This module defines handy "simulate" functions that run all the steps of Spain in a single function call.

use std::{fmt::Debug, time::Instant};

use crate::{
    inputs::{Metadata, R1CSMatrices},
    prover::{
        compute_squared_error, compute_squared_error_i1024, convert_instance_to_mont,
        error_to_mont, error_to_mont_i1024, limbs_to_u128, prepare_inner_sum_check,
        prepare_outer_sum_check, scale_factor, ProverState,
    },
    timer::{ProverPhase, Timer, VerifierPhase},
    traits::{MatrixIntOps, R1CSInstance, ToI1024, ToI512},
    verifier::{
        check_final_claim, check_r1cs_claim, epsilon_check, epsilon_check_i1024, smart_r1cs_eval,
        VerifierState,
    },
    EvaluationResult,
};
use dark::DARK;
use i256::I512;
use model::{AFloat, HighPrecision};
use parse::{generalized::HighPrecisionInt, mat::Matrix};

#[derive(Debug, Clone, Copy)]
pub struct SpainConfig {
    pub scale_factor_bits: usize,
    pub max_epsilon: f64,
    pub num_chunks: usize,
    pub precision: u16,
    pub q_bits: usize,
    pub batch_size: usize,
}

impl Default for SpainConfig {
    fn default() -> Self {
        Self {
            scale_factor_bits: 70,
            max_epsilon: 0.1,
            num_chunks: 16,
            precision: 128,
            q_bits: 30_000,
            batch_size: 1,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_hp<T>(
    tensors: &R1CSMatrices<T>,
    z: &Matrix<T>,
    metadata: &Metadata,
    scale_factor_bits: usize,
    _scale_factor: AFloat,
    max_epsilon: f64,
    dark: &mut DARK,
    verbose: bool,
    eval_result: &mut EvaluationResult,
) where
    T: Copy + Clone + Default + PartialEq + ToI512 + Debug + HighPrecisionInt + ToI1024,
    Matrix<T>: MatrixIntOps,
{
    let protocol_start = std::time::Instant::now();

    // Compute squared error
    let error_start = std::time::Instant::now();
    let scale_factor_int = I512::from(1u64) << (scale_factor_bits as u32);
    let error = compute_squared_error_i1024(tensors, z, &scale_factor_int, verbose);
    eval_result.prover_compute_square_error_time = error_start.elapsed();

    // First, the verifier asserts that the constraint deviation (l_inf norm of error, bounded by
    // l2 norm) is sufficiently low
    let verifier_epsilon_check_start = std::time::Instant::now();
    epsilon_check_i1024(&error, scale_factor_bits, max_epsilon, verbose);
    eval_result.verifier_epsilon_check_time = verifier_epsilon_check_start.elapsed();

    if verbose {
        println!("Committing to w");
    }

    // Prover commits to witness rows and to sparse polynomials for A, B, C
    let prover_comm_start = std::time::Instant::now();
    let has_secondary_constraints = metadata.num_secondary_constraint_variables > 0;
    let has_randomness = metadata.num_random_values > 0;
    let num_range_variables: usize = {
        if has_randomness {
            2
        } else {
            1
        }
    };
    let ranges = metadata.get_ranges();
    let w_mle = z.extract_rows_to_mle(Some(&ranges[1]));
    let commit_time = std::time::Instant::now();
    let chunked_comm = dark.prover.commit(w_mle.clone(), &dark.public);
    println!("commit time: {:?}", commit_time.elapsed());
    eval_result.prover_poly_commit_time += prover_comm_start.elapsed();

    if verbose {
        println!("Done committing to w");
    }

    // Verifier now samples a random prime
    let verifier_sample_start = std::time::Instant::now();
    let mont = dark.public.small_mont;
    eval_result.verifier_sample_time += verifier_sample_start.elapsed();
    let scale_limbs = (scale_factor_int % I512::from(mont.modulus())).to_le_limbs();
    let scale_mod = limbs_to_u128(&scale_limbs);
    let scale_mont = mont.to_mont(scale_mod);
    let scale_den = mont.inv(scale_mont);

    // Prover prepares the outer sum check
    let outer_prep_start = std::time::Instant::now();
    let error_mont = error_to_mont_i1024(&mont, error, scale_mont, verbose);
    let (tensors_mont, z_mont) = convert_instance_to_mont(&mont, tensors, z, scale_den, verbose);
    let mut prover_state = prepare_outer_sum_check(&mont, &tensors_mont, &z_mont, verbose);
    eval_result.prover_prepare_outer_sc_time = outer_prep_start.elapsed();

    // Prover and verifier run the outer sum check
    let (r_outer, claims, outer_sc_eval_result) = prover_state.simulate(error_mont, &[], verbose);
    eval_result.prover_run_outer_sc_time = outer_sc_eval_result.prover_time;
    eval_result.verifier_run_outer_sc_time = outer_sc_eval_result.verifier_time;

    // Verifier now samples randomness and gets linear combo of claims
    let verifier_sample_start = std::time::Instant::now();
    let r1 = mont.to_mont(ff::prime_128::rand_elem(mont.modulus(), &mut rand::rng()));
    let r2 = mont.to_mont(ff::prime_128::rand_elem(mont.modulus(), &mut rand::rng()));
    let mut claim = claims[0];
    claim = mont.add(claim, mont.mul(r1, claims[1]));
    claim = mont.add(claim, mont.mul(r2, claims[2]));
    eval_result.verifier_sample_time += verifier_sample_start.elapsed();

    // Prover prepares the inner sum check
    let inner_prep_start = std::time::Instant::now();
    let a_height = tensors_mont.a.height();
    let split_point = a_height.next_power_of_two().ilog2() as usize;
    let r_row = r_outer[0..split_point].to_vec();
    let r_z_col = r_outer[split_point..].to_vec();
    let mut prover_state_inner = prepare_inner_sum_check(
        &mont,
        &tensors_mont,
        &z_mont,
        r1,
        r2,
        &r_row,
        &r_z_col,
        verbose,
    );
    eval_result.prover_prepare_inner_sc_time = inner_prep_start.elapsed();

    // Prover and verifier run the inner sum check
    let (r_col, claims, inner_sc_eval_result) =
        prover_state_inner.simulate(claim, &[r1, r2], verbose);
    eval_result.prover_run_inner_sc_time = inner_sc_eval_result.prover_time;
    eval_result.verifier_run_inner_sc_time = inner_sc_eval_result.verifier_time;

    // Prover extracts the witness and evaluates it using mock dark
    let prover_dark_prep = std::time::Instant::now();
    let mut eval_point = vec![];
    eval_point.extend(r_z_col.clone());
    eval_point.extend(r_col[..r_col.len() - num_range_variables].to_vec());
    let mut w_mont_mle = w_mle.reduce_to_mont(&mont);
    let ez1_ref_dark = w_mont_mle.eval(&eval_point, &mont);
    eval_result.prover_poly_eval_time += prover_dark_prep.elapsed();
    let mut dark_eval_result = dark.run_protocol(
        chunked_comm,
        w_mle.clone(),
        &eval_point,
        ez1_ref_dark,
        &mut rand::rng(),
    );
    dark_eval_result.calc_prover_total();
    eval_result.prover_poly_eval_time += dark_eval_result.prover_time;
    eval_result.verifier_poly_eval_time = dark_eval_result.verifier_time;

    // Verifier computes A/B/C evaluations
    let verifier_smart_eval_start = std::time::Instant::now();
    let ez1 = mont.mul(ez1_ref_dark, scale_den);
    let (a_eval, b_eval, c_eval) =
        smart_r1cs_eval(&mont, &tensors_mont, &r_row, &r_col, &metadata.get_ranges());
    check_r1cs_claim(&mont, a_eval, b_eval, c_eval, r1, r2, claims[0]);
    eval_result.verifier_smart_eval_time += verifier_smart_eval_start.elapsed();

    // Verifier checks the final claim
    let verifier_claim_start = Instant::now();
    check_final_claim(
        &mont,
        ez1,
        &z_mont,
        &r_col,
        &r_z_col,
        &claims,
        &metadata.get_ranges(),
        has_randomness,
        has_secondary_constraints,
    );
    eval_result.verifier_claim_interpolate_time = verifier_claim_start.elapsed();
    eval_result.total_protocol_time = protocol_start.elapsed();
}

#[allow(clippy::too_many_arguments)]
pub fn simulate<T>(
    tensors: &R1CSMatrices<T>,
    z: &Matrix<T>,
    metadata: &Metadata,
    scale_factor_bits: usize,
    _scale_factor: AFloat,
    max_epsilon: f64,
    dark: &mut DARK,
    verbose: bool,
    eval_result: &mut EvaluationResult,
) where
    T: Copy + Clone + Default + PartialEq + ToI512 + Debug,
    Matrix<T>: MatrixIntOps,
{
    let protocol_start = std::time::Instant::now();

    // Compute squared error
    let error_start = std::time::Instant::now();
    let scale_factor_int = I512::from(1u64) << (scale_factor_bits as u32);
    let error = compute_squared_error(tensors, z, &scale_factor_int, verbose);
    dbg!(&error);
    eval_result.prover_compute_square_error_time = error_start.elapsed();

    // First, the verifier asserts that the constraint deviation (l_inf norm of error, bounded by
    // l2 norm) is sufficiently low
    let verifier_epsilon_check_start = std::time::Instant::now();
    epsilon_check(&error, scale_factor_bits, max_epsilon, verbose);
    eval_result.verifier_epsilon_check_time = verifier_epsilon_check_start.elapsed();

    if verbose {
        println!("Committing to w");
    }

    // Prover commits to witness rows and to sparse polynomials for A, B, C
    let prover_comm_start = std::time::Instant::now();
    let has_secondary_constraints = metadata.num_secondary_constraint_variables > 0;
    let has_randomness = metadata.num_random_values > 0;
    let num_range_variables: usize = {
        if has_randomness {
            2
        } else {
            1
        }
    };
    let ranges = metadata.get_ranges();
    let w_mle = z.extract_rows_to_mle(&ranges[1]);
    let commit_time = std::time::Instant::now();
    let chunked_comm = dark.prover.commit(w_mle.clone(), &dark.public);
    println!("commit time: {:?}", commit_time.elapsed());
    eval_result.prover_poly_commit_time += prover_comm_start.elapsed();

    if verbose {
        println!("Done committing to w");
    }

    // Verifier now samples a random prime
    let verifier_sample_start = std::time::Instant::now();
    let mont = dark.public.small_mont;
    eval_result.verifier_sample_time += verifier_sample_start.elapsed();
    let scale_limbs = (scale_factor_int % I512::from(mont.modulus())).to_le_limbs();
    let scale_mod = limbs_to_u128(&scale_limbs);
    let scale_mont = mont.to_mont(scale_mod);
    let scale_den = mont.inv(scale_mont);

    // Prover prepares the outer sum check
    let outer_prep_start = std::time::Instant::now();
    let error_mont = error_to_mont(&mont, error, scale_mont, verbose);
    let (tensors_mont, z_mont) = convert_instance_to_mont(&mont, tensors, z, scale_den, verbose);
    let mut prover_state = prepare_outer_sum_check(&mont, &tensors_mont, &z_mont, verbose);
    eval_result.prover_prepare_outer_sc_time = outer_prep_start.elapsed();

    // Prover and verifier run the outer sum check
    let (r_outer, claims, outer_sc_eval_result) = prover_state.simulate(error_mont, &[], verbose);
    eval_result.prover_run_outer_sc_time = outer_sc_eval_result.prover_time;
    eval_result.verifier_run_outer_sc_time = outer_sc_eval_result.verifier_time;

    // Verifier now samples randomness and gets linear combo of claims
    let verifier_sample_start = std::time::Instant::now();
    let r1 = mont.to_mont(ff::prime_128::rand_elem(mont.modulus(), &mut rand::rng()));
    let r2 = mont.to_mont(ff::prime_128::rand_elem(mont.modulus(), &mut rand::rng()));
    let mut claim = claims[0];
    claim = mont.add(claim, mont.mul(r1, claims[1]));
    claim = mont.add(claim, mont.mul(r2, claims[2]));
    eval_result.verifier_sample_time += verifier_sample_start.elapsed();

    // Prover prepares the inner sum check
    let inner_prep_start = std::time::Instant::now();
    let a_height = tensors_mont.a.height();
    let split_point = a_height.next_power_of_two().ilog2() as usize;
    let r_row = r_outer[0..split_point].to_vec();
    let r_z_col = r_outer[split_point..].to_vec();
    let mut prover_state_inner = prepare_inner_sum_check(
        &mont,
        &tensors_mont,
        &z_mont,
        r1,
        r2,
        &r_row,
        &r_z_col,
        verbose,
    );
    eval_result.prover_prepare_inner_sc_time = inner_prep_start.elapsed();

    // Prover and verifier run the inner sum check
    let (r_col, claims, inner_sc_eval_result) =
        prover_state_inner.simulate(claim, &[r1, r2], verbose);
    eval_result.prover_run_inner_sc_time = inner_sc_eval_result.prover_time;
    eval_result.verifier_run_inner_sc_time = inner_sc_eval_result.verifier_time;

    // Prover extracts the witness and evaluates it using mock dark
    let prover_dark_prep = std::time::Instant::now();
    let mut eval_point = vec![];
    eval_point.extend(r_z_col.clone());
    eval_point.extend(r_col[..r_col.len() - num_range_variables].to_vec());
    let mut w_mont_mle = w_mle.reduce_to_mont(&mont);
    let ez1_ref = w_mont_mle.eval(&eval_point, &mont);
    eval_result.prover_poly_eval_time += prover_dark_prep.elapsed();
    let mut dark_eval_result = dark.run_protocol(
        chunked_comm,
        w_mle.clone(),
        &eval_point,
        ez1_ref,
        &mut rand::rng(),
    );
    dark_eval_result.calc_prover_total();
    eval_result.prover_poly_eval_time += dark_eval_result.prover_time;
    eval_result.verifier_poly_eval_time = dark_eval_result.verifier_time;

    // Verifier computes A/B/C evaluations
    let verifier_smart_eval_start = std::time::Instant::now();
    let ez1 = mont.mul(ez1_ref, scale_den);
    let (a_eval, b_eval, c_eval) =
        smart_r1cs_eval(&mont, &tensors_mont, &r_row, &r_col, &metadata.get_ranges());
    check_r1cs_claim(&mont, a_eval, b_eval, c_eval, r1, r2, claims[0]);
    eval_result.verifier_smart_eval_time += verifier_smart_eval_start.elapsed();

    // Verifier checks the final claim
    let verifier_claim_start = Instant::now();
    check_final_claim(
        &mont,
        ez1,
        &z_mont,
        &r_col,
        &r_z_col,
        &claims,
        &metadata.get_ranges(),
        has_randomness,
        has_secondary_constraints,
    );
    eval_result.verifier_claim_interpolate_time = verifier_claim_start.elapsed();
    eval_result.total_protocol_time = protocol_start.elapsed();
}

pub fn stateful_simulate<P, E>(wit_exec: E, config: Option<SpainConfig>) -> EvaluationResult
where
    P: HighPrecision,
    E: R1CSInstance<P, i128> + Clone,
{
    stateful_simulate_with_config(wit_exec, config.unwrap_or_default())
}

pub fn measure_setup_time<P, E>(wit_exec: E, config: SpainConfig)
where
    P: HighPrecision,
    E: R1CSInstance<P, i128> + Clone,
{
    let metadata = wit_exec.get_meta();
    let scale_factor: P = scale_factor(config.scale_factor_bits);
    let verifier_setup_start = Instant::now();

    let mut verifier: VerifierState<i128, P, E> = VerifierState::new(
        config.max_epsilon,
        config.batch_size,
        config.scale_factor_bits,
        config.q_bits,
        config.precision,
        config.num_chunks,
        wit_exec.clone(),
        metadata.clone(),
    );

    verifier.dark_setup();

    let verifier_setup_time = verifier_setup_start.elapsed();

    let public_params = verifier.get_dark_public_params();

    let prover_setup_start = Instant::now();

    let mut prover: ProverState<i128, P, E> =
        ProverState::new(wit_exec, scale_factor, metadata, config.batch_size);

    let randomness = verifier.sample_normal_randomness();
    prover.set_randomness(randomness);
    prover.import_r1cs_matrices();

    public_params.build_pippenger_bench();

    let prover_setup_time = prover_setup_start.elapsed();

    println!("Prover setup time: {:?}", prover_setup_time);

    println!("Verifier setup time: {:?}", verifier_setup_time);
}

pub fn stateful_simulate_with_config<P, E>(wit_exec: E, config: SpainConfig) -> EvaluationResult
where
    P: HighPrecision,
    E: R1CSInstance<P, i128> + Clone,
{
    let metadata = wit_exec.get_meta();
    let scale_factor: P = scale_factor(config.scale_factor_bits);

    let mut prover: ProverState<i128, P, E> = ProverState::new(
        wit_exec.clone(),
        scale_factor,
        metadata.clone(),
        config.batch_size,
    );

    let mut verifier: VerifierState<i128, P, E> = VerifierState::new(
        config.max_epsilon,
        config.batch_size,
        config.scale_factor_bits,
        config.q_bits,
        config.precision,
        config.num_chunks,
        wit_exec,
        metadata,
    );

    verifier.dark_setup();
    let public_params = verifier.get_dark_public_params();
    prover.set_dark_public_params(public_params);

    // Core protocol
    let mut timer = Timer::new();

    timer.prover(ProverPhase::ComputeWitness, || {
        prover.compute_commit_witness()
    });
    let comm = timer.prover(ProverPhase::PolyCommit, || prover.commit());
    timer.verifier(VerifierPhase::Misc, || verifier.set_commit(comm));
    let randomness = timer.verifier(VerifierPhase::Misc, || {
        let randomness = verifier.sample_normal_randomness();
        verifier.inject_randomness();
        randomness
    });
    timer.prover(ProverPhase::ComputeWitness, || {
        prover.set_randomness(randomness);
        prover.inject_randomness();
        prover.compute_full_witness()
    });
    let squared_error = timer.prover(ProverPhase::ComputeSquaredError, || {
        prover.compute_squared_error()
    });
    timer.verifier(VerifierPhase::EpsilonCheck, || {
        verifier.epsilon_check(&squared_error)
    });

    let mont = timer.verifier(VerifierPhase::Sample, || verifier.sample_mont());
    timer.prover(ProverPhase::PrepareOuterSc, || {
        prover.set_mont(mont);
        prover.convert_instance_to_mont();
        prover.prepare_outer_sc();
        verifier.prepare_outer_sc();
    });

    let mut claim = timer.prover(ProverPhase::RunOuterSc, || prover.outer_sc_claim());

    loop {
        let r = timer
            .verifier(VerifierPhase::RunOuterSc, || {
                verifier.outer_sc_verify(&mut claim)
            })
            .expect("outer sc verify failed");

        if prover.outer_last_round() {
            let final_evals = timer.prover(ProverPhase::RunOuterSc, || prover.outer_final_evals(r));
            timer
                .verifier(VerifierPhase::RunOuterSc, || {
                    verifier.outer_sc_check_final_evals(&claim, &final_evals)
                })
                .expect("outer sc final evals check failed");
            break;
        }

        claim = timer.prover(ProverPhase::RunOuterSc, || prover.outer_sc_prove(Some(r)));
    }

    let (r1, r2) = timer.verifier(VerifierPhase::Sample, || verifier.sample_lc_challenges());

    let num_vars = timer.prover(ProverPhase::PrepareInnerSc, || {
        prover.prepare_inner_sc(r1, r2)
    });
    timer.verifier(VerifierPhase::RunInnerSc, || {
        verifier.prepare_inner_sc(num_vars)
    });

    let mut inner_claim = timer.prover(ProverPhase::RunInnerSc, || prover.inner_sc_claim());

    loop {
        let r = timer
            .verifier(VerifierPhase::RunInnerSc, || {
                verifier.inner_sc_verify(&mut inner_claim)
            })
            .expect("inner sc verify failed");

        if prover.inner_last_round() {
            let final_evals = timer.prover(ProverPhase::RunInnerSc, || prover.inner_final_evals(r));
            timer
                .verifier(VerifierPhase::RunInnerSc, || {
                    verifier.inner_sc_check_final_evals(&inner_claim, &final_evals)
                })
                .expect("inner sc final evals check failed");
            break;
        }
        inner_claim = timer.prover(ProverPhase::RunInnerSc, || prover.inner_sc_prove(Some(r)));
    }

    let eval_point = timer.verifier(VerifierPhase::PolyEval, || verifier.dark_eval_point());
    let dark_claim = timer.prover(ProverPhase::PolyEval, || prover.dark_mle_eval(&eval_point));
    timer.verifier(VerifierPhase::PolyEval, || {
        verifier.set_dark_claim(dark_claim)
    });

    loop {
        let challenge = timer.verifier(VerifierPhase::PolyEval, || verifier.start_dark_round());
        let round_claim = timer.prover(ProverPhase::PolyEval, || prover.dark_respond(challenge));
        timer.verifier(VerifierPhase::PolyEval, || {
            verifier.verify_dark_round(&round_claim)
        });
        if round_claim.final_claim.is_some() {
            break;
        }
    }

    let z_openings = timer.prover(ProverPhase::Misc, || prover.witness_openings());
    timer.verifier(VerifierPhase::SmartEval, || verifier.matrices_claim_check());
    timer.verifier(VerifierPhase::ClaimInterpolate, || {
        verifier.witness_claim_check(&z_openings)
    });
    eprintln!("Final verification success!!!");

    let mut result = timer.finish();
    result.num_constraints = prover.num_constraints();
    result
}
#[cfg(test)]
mod simulate_single_tests {
    use model::F128;

    use crate::inputs::{import_metadata, DEFAULT_DATA_DIR};
    use crate::simulate::{stateful_simulate, SpainConfig};
    use crate::synthetic::SyntheticR1CS;
    use crate::witness_gen::OnnxExecutor;
    use std::path::PathBuf;

    #[test]
    fn test_stateful_simulate_synthetic_r1cs() {
        let mut config = SpainConfig::default();
        config.num_chunks = 4;
        let num_cons = 123456;
        let num_inputs = 10;
        let wit_exec = SyntheticR1CS::<i128>::new(num_cons, num_inputs, config.scale_factor_bits);
        let result = stateful_simulate::<F128, _>(wit_exec, Some(config));
        dbg!(&result);
        assert_eq!(result.num_constraints, num_cons);
    }

    #[test]
    fn test_stateful_simulate_layernorm() {
        let model = "layernorm_32x768";
        let path = PathBuf::from(DEFAULT_DATA_DIR);
        let metadata = import_metadata(&path, model);
        let wit_exec =
            OnnxExecutor::<F128>::new(model.to_string(), path.clone(), metadata.clone(), true);
        let result = stateful_simulate(wit_exec, None);
        dbg!(result);
    }

    #[test]
    #[ignore]
    fn test_stateful_simulate_gpt() {
        let model = "gpt2-seq-2";
        let path = PathBuf::from(DEFAULT_DATA_DIR);
        let metadata = import_metadata(&path, model);
        let wit_exec =
            OnnxExecutor::<F128>::new(model.to_string(), path.clone(), metadata.clone(), true);
        dbg!(stateful_simulate(wit_exec, None));
    }

    #[test]
    fn test_stateful_simulate_softmax() {
        let model = "softmax_32x32";
        let path = PathBuf::from(DEFAULT_DATA_DIR);
        let metadata = import_metadata(&path, model);
        let wit_exec =
            OnnxExecutor::<F128>::new(model.to_string(), path.clone(), metadata.clone(), true);
        dbg!(stateful_simulate(wit_exec, None));
    }
}
#[cfg(test)]
mod simulate_batched_tests {
    use model::F128;

    use crate::inputs::{import_metadata, DEFAULT_DATA_DIR};
    use crate::simulate::{stateful_simulate, SpainConfig};
    use crate::witness_gen::OnnxExecutor;
    use std::path::PathBuf;

    #[test]
    fn test_batched_stateful_simulate_layernorm() {
        let model = "layernorm-32x768";
        let path = PathBuf::from(DEFAULT_DATA_DIR);
        let metadata = import_metadata(&path, model);
        let wit_exec =
            OnnxExecutor::<F128>::new(model.to_string(), path.clone(), metadata.clone(), true);
        let mut config = SpainConfig::default();
        config.batch_size = 2;
        let result = stateful_simulate(wit_exec, Some(config));
        dbg!(result);
    }

    #[test]
    fn test_batched_stateful_simulate_softmax() {
        let model = "softmax-32x32";
        let path = PathBuf::from(DEFAULT_DATA_DIR);
        let metadata = import_metadata(&path, model);
        let wit_exec =
            OnnxExecutor::<F128>::new(model.to_string(), path.clone(), metadata.clone(), true);
        let mut config = SpainConfig::default();
        config.batch_size = 2;
        let result = stateful_simulate(wit_exec, Some(config));
        dbg!(result);
    }
}
