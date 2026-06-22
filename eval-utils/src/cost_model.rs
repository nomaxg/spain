use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use libspartan::{Instance, SNARK, SNARKGens};
use merlin::Transcript;
use serde::{Deserialize, Serialize};

pub const MIN_INSTANCE_SIZE: u32 = 9;
pub const MAX_INSTANCE_SIZE: u32 = 20;
const NUM_SAMPLES: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProverCostModel {
    pub slope_ms_per_constraint: f64,
    pub intercept_ms: f64,
    pub r_squared: f64,
}

impl ProverCostModel {
    pub fn estimate_ms(&self, num_constraints: usize) -> f64 {
        self.slope_ms_per_constraint * num_constraints as f64 + self.intercept_ms
    }

    fn fit(num_constraints: &[usize], avg_prove_ms: &[f64]) -> Self {
        let (slope, intercept, r_squared) = fit_regression(num_constraints, avg_prove_ms, false);

        Self {
            slope_ms_per_constraint: slope,
            intercept_ms: intercept,
            r_squared,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VerifierCostModel {
    pub slope_ms_per_sqrt_constraint: f64,
    pub intercept_ms: f64,
    pub r_squared: f64,
}

impl VerifierCostModel {
    pub fn estimate_ms(&self, num_constraints: usize) -> f64 {
        self.slope_ms_per_sqrt_constraint * (num_constraints as f64).sqrt() + self.intercept_ms
    }

    fn fit(num_constraints: &[usize], avg_verify_ms: &[f64]) -> Self {
        let (slope, intercept, r_squared) = fit_regression(num_constraints, avg_verify_ms, true);

        Self {
            slope_ms_per_sqrt_constraint: slope,
            intercept_ms: intercept,
            r_squared,
        }
    }
}

pub fn fit_regression(
    num_constraints: &[usize],
    avg_times_ms: &[f64],
    is_sqrt_model: bool,
) -> (f64, f64, f64) {
    assert_eq!(num_constraints.len(), avg_times_ms.len());
    assert!(!num_constraints.is_empty(), "need at least one sample");

    let x_values: Vec<f64> = if is_sqrt_model {
        num_constraints.iter().map(|&n| (n as f64).sqrt()).collect()
    } else {
        num_constraints.iter().map(|&n| n as f64).collect()
    };

    let m = avg_times_ms.len() as f64;
    let sum_x = x_values.iter().sum::<f64>();
    let sum_y = avg_times_ms.iter().copied().sum::<f64>();
    let sum_xy = x_values
        .iter()
        .zip(avg_times_ms.iter())
        .map(|(x, y)| x * y)
        .sum::<f64>();
    let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();

    let denom = m * sum_x2 - sum_x * sum_x;
    assert!(denom > 0.0, "degenerate regression denominator");
    let slope = (m * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / m;

    let mean_y = sum_y / m;
    let ss_tot = avg_times_ms
        .iter()
        .map(|&y| {
            let delta = y - mean_y;
            delta * delta
        })
        .sum::<f64>();
    let ss_res = x_values
        .iter()
        .zip(avg_times_ms.iter())
        .map(|(&x, &y)| {
            let y_hat = slope * x + intercept;
            let delta = y - y_hat;
            delta * delta
        })
        .sum::<f64>();
    let r_squared = if ss_tot <= f64::EPSILON {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    };

    (slope, intercept, r_squared)
}

pub fn prover_cost_model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("prover_cost.json")
}

pub fn verifier_cost_model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("verifier_cost.json")
}

pub fn cost_models_exist() -> bool {
    prover_cost_model_path().is_file() && verifier_cost_model_path().is_file()
}

pub fn load_prover_cost_model() -> ProverCostModel {
    let path = prover_cost_model_path();
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read prover cost model {}: {err}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|err| {
        panic!(
            "failed to parse prover cost model {}: {err}",
            path.display()
        )
    })
}

pub fn load_verifier_cost_model() -> VerifierCostModel {
    let path = verifier_cost_model_path();
    let raw = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read verifier cost model {}: {err}",
            path.display()
        )
    });
    serde_json::from_str(&raw).unwrap_or_else(|err| {
        panic!(
            "failed to parse verifier cost model {}: {err}",
            path.display()
        )
    })
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) {
    let data = serde_json::to_string_pretty(value)
        .unwrap_or_else(|err| panic!("failed to serialize {}: {err}", path.display()));
    fs::write(path, format!("{data}\n"))
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", path.display()));
}

fn ensure_single_thread_pool() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
}

fn measure_once(num_cons: usize, num_inputs: usize) -> (Duration, Duration) {
    let num_vars = num_cons;
    let gens = SNARKGens::new(num_cons, num_vars, num_inputs, num_cons);
    let (inst, vars, inputs) = Instance::produce_synthetic_r1cs(num_cons, num_vars, num_inputs);
    let (comm, decomm) = SNARK::encode(&inst, &gens);

    let mut prover_transcript = Transcript::new(b"bench");
    let prove_start = Instant::now();
    let proof = SNARK::prove(
        &inst,
        &comm,
        &decomm,
        vars,
        &inputs,
        &gens,
        &mut prover_transcript,
    );
    let prove_time = prove_start.elapsed();

    let mut verifier_transcript = Transcript::new(b"bench");
    let verify_start = Instant::now();
    proof
        .verify(&comm, &inputs, &mut verifier_transcript, &gens)
        .unwrap();
    let verify_time = verify_start.elapsed();

    (prove_time, verify_time)
}

#[derive(Debug, Clone)]
struct CostData {
    verifier_timings: Vec<Vec<f64>>,
    prover_timings: Vec<Vec<f64>>,
}

fn collect_samples(sizes: &[usize], num_inputs: usize) -> CostData {
    let mut prover_timings = Vec::with_capacity(sizes.len());
    let mut verifier_timings = Vec::with_capacity(sizes.len());

    for &n in sizes {
        println!("Collecting cost-model data for instance of size {}", n);
        let mut prove_samples = Vec::with_capacity(NUM_SAMPLES);
        let mut verify_samples = Vec::with_capacity(NUM_SAMPLES);
        for i in 0..NUM_SAMPLES {
            println!("sample {}/{}", i + 1, NUM_SAMPLES);
            let (prove, verify) = measure_once(n, num_inputs);
            prove_samples.push(prove.as_secs_f64() * 1000.0);
            verify_samples.push(verify.as_secs_f64() * 1000.0);
        }
        prover_timings.push(prove_samples);
        verifier_timings.push(verify_samples);
    }

    CostData {
        verifier_timings,
        prover_timings,
    }
}

fn mean_and_std_ms(samples: &[f64]) -> (f64, f64) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = if samples.len() > 1 {
        samples
            .iter()
            .map(|&s| {
                let delta = s - mean;
                delta * delta
            })
            .sum::<f64>()
            / (samples.len() as f64 - 1.0)
    } else {
        0.0
    };
    (mean, variance.sqrt())
}

fn average_times(sizes: &[usize], timings: &[Vec<f64>], std_dev_fraction_limit: f64) -> Vec<f64> {
    assert_eq!(sizes.len(), timings.len());

    sizes
        .iter()
        .zip(timings.iter())
        .map(|(&n, samples)| {
            let (mean, std_dev) = mean_and_std_ms(samples);
            assert!(mean > 0.0, "Mean time should be positive for n={}", n);
            if std_dev > mean * std_dev_fraction_limit {
                eprintln!(
                    "warning: time std dev high for n={}: mean {:.6} ms, std {:.6} ms",
                    n, mean, std_dev
                );
            }
            println!("n={}, avg={:.6} ms (std {:.6} ms)", n, mean, std_dev);
            mean
        })
        .collect()
}

fn average_prover_times(sizes: &[usize], cost_data: &CostData) -> Vec<f64> {
    average_times(sizes, &cost_data.prover_timings, 0.05)
}

fn average_verifier_times(sizes: &[usize], cost_data: &CostData) -> Vec<f64> {
    average_times(sizes, &cost_data.verifier_timings, 0.1)
}

fn fit_prover_model(sizes: &[usize], avg_prove_ms: &[f64]) -> ProverCostModel {
    ProverCostModel::fit(sizes, avg_prove_ms)
}

fn fit_verifier_model(sizes: &[usize], avg_verify_ms: &[f64]) -> VerifierCostModel {
    VerifierCostModel::fit(sizes, avg_verify_ms)
}

fn instance_sizes(max_instance_size_exponent: u32) -> Vec<usize> {
    assert!(
        max_instance_size_exponent >= MAX_INSTANCE_SIZE,
        "max instance size exponent must be at least {}",
        MIN_INSTANCE_SIZE
    );
    (MIN_INSTANCE_SIZE..=max_instance_size_exponent)
        .map(|k| 1usize << k)
        .collect()
}

pub fn derive_cost_model(max_instance_size_exponent: u32) {
    ensure_single_thread_pool();

    let num_inputs = 10;
    let sizes = instance_sizes(max_instance_size_exponent);
    let cost_data = collect_samples(&sizes, num_inputs);
    dbg!(&cost_data);
    let avg_prove_ms = average_prover_times(&sizes, &cost_data);
    let avg_verify_ms = average_verifier_times(&sizes, &cost_data);

    let prover_model = fit_prover_model(&sizes, &avg_prove_ms);
    write_json_file(&prover_cost_model_path(), &prover_model);
    println!(
        "prover linear fit (ms): time = {:.6} * n + {:.6} (r^2={:.6})",
        prover_model.slope_ms_per_constraint, prover_model.intercept_ms, prover_model.r_squared
    );

    let verifier_model = fit_verifier_model(&sizes, &avg_verify_ms);
    write_json_file(&verifier_cost_model_path(), &verifier_model);
    println!(
        "verifier linear fit (ms): time = {:.6} * sqrt(n) + {:.6} (r^2={:.6})",
        verifier_model.slope_ms_per_sqrt_constraint,
        verifier_model.intercept_ms,
        verifier_model.r_squared
    );
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn slope_stability_single_thread() {
        ensure_single_thread_pool();

        let num_inputs = 10;
        let sizes = instance_sizes(MAX_INSTANCE_SIZE);

        let mut prove_times = Vec::with_capacity(sizes.len());
        let mut verify_times = Vec::with_capacity(sizes.len());
        for &num_cons in &sizes {
            let (prove, verify) = measure_avg(num_cons, num_inputs, NUM_SAMPLES);
            prove_times.push(prove);
            verify_times.push(verify);
        }

        let slopes_prove: Vec<f64> = prove_times
            .windows(2)
            .zip(sizes.windows(2))
            .map(|(times, cons)| {
                let delta_t = times[1].saturating_sub(times[0]);
                let delta_n = (cons[1] - cons[0]) as f64;
                (delta_t.as_secs_f64() * 1000.0) / delta_n
            })
            .collect();

        let slopes_verify: Vec<f64> = verify_times
            .windows(2)
            .zip(sizes.windows(2))
            .map(|(times, cons)| {
                let delta_t = times[1].saturating_sub(times[0]);
                let delta_n = (cons[1] - cons[0]) as f64;
                (delta_t.as_secs_f64() * 1000.0) / delta_n
            })
            .collect();

        println!("prove slopes:");
        for ((start, end), slope) in sizes
            .windows(2)
            .map(|pair| (pair[0], pair[1]))
            .zip(slopes_prove.iter())
        {
            println!("  {}-{}: {:.6} ms/con", start, end, slope);
        }

        println!("verify slopes:");
        for ((start, end), slope) in sizes
            .windows(2)
            .map(|pair| (pair[0], pair[1]))
            .zip(slopes_verify.iter())
        {
            println!("  {}-{}: {:.6} ms/con", start, end, slope);
        }
    }

    #[test]
    fn derive_spartan_cost_model() {
        derive_cost_model(MAX_INSTANCE_SIZE)
    }

    #[test]
    fn derive_spartan_verifier_cost_model() {
        derive_cost_model(MAX_INSTANCE_SIZE);
    }

    #[test]
    fn bundled_models_parse() {
        let prover = load_prover_cost_model();
        let verifier = load_verifier_cost_model();
        assert!(prover.estimate_ms(1 << 16) > 0.0);
        assert!(verifier.estimate_ms(1 << 16) > 0.0);
    }

    #[cfg(test)]
    fn measure_avg(num_cons: usize, num_inputs: usize, num_samples: usize) -> (Duration, Duration) {
        let mut prove = Duration::ZERO;
        let mut verify = Duration::ZERO;
        for _ in 0..num_samples {
            let (p, v) = measure_once(num_cons, num_inputs);
            prove += p;
            verify += v;
        }
        (prove / num_samples as u32, verify / num_samples as u32)
    }
}
