use clap::Parser;
use eval_utils::cost_model::fit_regression;
use model::F128;
use spain::simulate::{SpainConfig, stateful_simulate_with_config};
use spain::synthetic::SyntheticR1CS;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Fit Spain prover/verifier cost models over synthetic R1CS instances",
    long_about = None
)]
struct Cli {
    // Smallest exponent for instance size (2^x)
    #[arg(long, default_value_t = 9)]
    min_exp: u32,

    // Largest exponent for instance size 2^x
    #[arg(long, default_value_t = 20)]
    max_exp: u32,

    // Number of samples per instance size
    #[arg(long, default_value_t = 5)]
    samples: usize,

    // Number of public inputs in the synthetic instance
    #[arg(long, default_value_t = 10)]
    num_inputs: usize,

    // Batch size for Spain runs
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    // Scale factor bits used for fixed-point representation
    #[arg(long, default_value_t = 70)]
    scale_factor_bits: usize,

    // Max epsilon for approximate checks
    #[arg(long, default_value_t = 0.1)]
    max_epsilon: f64,

    // Chunk size for DARK commitment
    #[arg(long, default_value_t = 16)]
    chunk_size: usize,

    // Number of bits for q in DARK
    #[arg(long, default_value_t = 30000)]
    q_bits: usize,

    // DARK precision
    #[arg(long, default_value_t = 128)]
    precision: u16,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    assert!(
        cli.min_exp <= cli.max_exp,
        "min_exponent must be <= max_exponent"
    );
    assert!(cli.samples > 0, "samples must be at least 1");
    assert!(cli.num_inputs > 0, "num_inputs must be at least 1");

    let config = SpainConfig {
        scale_factor_bits: cli.scale_factor_bits,
        max_epsilon: cli.max_epsilon,
        num_chunks: cli.chunk_size,
        precision: cli.precision,
        q_bits: cli.q_bits,
        batch_size: cli.batch_size,
        spartan_poly: true,
    };

    let mut num_constraints = Vec::new();
    let mut avg_prover_ms = Vec::new();
    let mut avg_verifier_ms = Vec::new();

    println!(
        "sweep=min(2^{}) max(2^{}) samples={} num_inputs={} batch_size={}",
        cli.min_exp, cli.max_exp, cli.samples, cli.num_inputs, cli.batch_size
    );
    println!("num_constraints\tavg_prover_ms\tavg_verifier_ms");

    for exponent in cli.min_exp..=cli.max_exp {
        let target_constraints = 1usize << exponent;
        let mut prover_samples_ms = Vec::with_capacity(cli.samples);
        let mut verifier_samples_ms = Vec::with_capacity(cli.samples);
        let mut measured_constraints = None;

        for _ in 0..cli.samples {
            let wit_exec = SyntheticR1CS::<i128>::new(
                target_constraints,
                cli.num_inputs,
                cli.scale_factor_bits,
            );
            let result = stateful_simulate_with_config::<F128, _, i128>(wit_exec, config);
            measured_constraints.get_or_insert(result.num_constraints);
            prover_samples_ms.push(result.zklp_prover_time().as_secs_f64() * 1000.0);
            verifier_samples_ms.push(result.zklp_verifier_time().as_secs_f64() * 1000.0);
        }

        let measured_constraints = measured_constraints.expect("missing measured constraints");
        let mean_prover_ms = prover_samples_ms.iter().sum::<f64>() / cli.samples as f64;
        let mean_verifier_ms = verifier_samples_ms.iter().sum::<f64>() / cli.samples as f64;

        num_constraints.push(measured_constraints);
        avg_prover_ms.push(mean_prover_ms);
        avg_verifier_ms.push(mean_verifier_ms);

        println!(
            "{}\t{:.6}\t{:.6}",
            measured_constraints, mean_prover_ms, mean_verifier_ms
        );
    }

    let (prover_slope, prover_intercept, prover_r_squared) =
        fit_regression(&num_constraints, &avg_prover_ms, false);
    let (verifier_slope, verifier_intercept, verifier_r_squared) =
        fit_regression(&num_constraints, &avg_verifier_ms, false);

    println!();
    println!(
        "prover_model_ms = {:.12} * num_constraints + {:.12} (r^2 = {:.6})",
        prover_slope, prover_intercept, prover_r_squared
    );
    println!(
        "verifier_model_ms = {:.12} * num_constraints + {:.12} (r^2 = {:.6})",
        verifier_slope, verifier_intercept, verifier_r_squared
    );
}
