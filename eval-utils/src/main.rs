mod cost_model;

use std::time::Instant;

use clap::{Parser, ValueEnum};
use cost_model::{
    cost_models_exist, derive_cost_model, load_prover_cost_model, load_verifier_cost_model,
    MAX_INSTANCE_SIZE,
};
use libspartan::{Instance, SNARKGens, SNARK};
use merlin::Transcript;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ComputationType {
    #[value(name = "gpt2-seq-2")]
    GPT2Seq2,
    #[value(name = "gpt2-seq-32")]
    GPT2Seq32,
    #[value(name = "layernorm-32x768")]
    LayerNorm,
    #[value(name = "gelu-32x3072")]
    Gelu,
    #[value(name = "softmax-32x32")]
    Softmax,
    Debug,
}

impl ComputationType {
    fn uses_cost_model(self) -> bool {
        matches!(
            self,
            Self::GPT2Seq2 | Self::GPT2Seq32 | Self::LayerNorm | Self::Gelu
        )
    }

    fn all_non_debug() -> &'static [Self] {
        &[
            Self::GPT2Seq2,
            Self::GPT2Seq32,
            Self::LayerNorm,
            Self::Gelu,
            Self::Softmax,
        ]
    }
}

#[derive(Debug)]
struct InstanceSize {
    num_inputs: usize,
    target_num_cons: usize,
    synthetic_num_cons: usize,
    synthetic_num_vars: usize,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Run a Spartan SNARK example", long_about = None)]
struct Args {
    /// computation profile to size the synthetic R1CS instance
    #[arg(long = "computation-type", value_enum)]
    computation_type: Vec<ComputationType>,
    /// Serialize fitted prover/verifier cost models to JSON instead of running a profile
    #[arg(long = "serialize-cost-models", default_value_t = false)]
    serialize_cost_models: bool,
    /// Run all supplied computation types, or all non-debug computation types if none are supplied
    #[arg(long = "run-all", default_value_t = false)]
    run_all: bool,
    /// Maximum exponent n used when fitting cost models over sizes 2^9 through 2^n
    #[arg(long = "max-instance-size", default_value_t = MAX_INSTANCE_SIZE)]
    max_instance_size: u32,
}

// Extremely generous sizing based on number of constraints from Spain's arithmetization
// Does not account for any extra floating point constraints/overhead
fn get_instance_size_lower_bound(computation_type: ComputationType) -> InstanceSize {
    let (num_inputs, target_num_cons): (usize, usize) = match computation_type {
        ComputationType::GPT2Seq2 => (10, 18_504_774_754),
        ComputationType::GPT2Seq32 => (10, 22_590_626_044),
        ComputationType::LayerNorm => (49_153, 10_148_320),
        ComputationType::Gelu => (196_609, 47_087_616),
        ComputationType::Softmax => (2_049, 1_307_360),
        ComputationType::Debug => (10, 259_923),
    };

    // We never overshoot target number of constraints, round up to next power of two and halve.
    let synthetic_num_cons = target_num_cons.next_power_of_two() / 2;

    InstanceSize {
        num_inputs,
        target_num_cons,
        synthetic_num_cons,
        synthetic_num_vars: synthetic_num_cons,
    }
}

fn run_computation(computation_type: ComputationType) {
    let size = get_instance_size_lower_bound(computation_type);
    let label = computation_type
        .to_possible_value()
        .map(|v| v.get_name().to_string())
        .unwrap_or("unknown".to_string());

    if computation_type.uses_cost_model() {
        let prover_model = load_prover_cost_model();
        let verifier_model = load_verifier_cost_model();
        let prover_time_ms = prover_model.estimate_ms(size.target_num_cons);
        let verifier_time_ms = verifier_model.estimate_ms(size.target_num_cons);

        println!("Model: {}", label);
        println!("Prover time: {:.6}ms", prover_time_ms);
        println!("Verifier time: {:.6}ms", verifier_time_ms);
        println!("Num constraints: {} \n", size.target_num_cons);
        return;
    }

    println!("Running benchmark: {}", label);
    println!("Producing public parameters..");
    let gens = SNARKGens::new(
        size.synthetic_num_cons,
        size.synthetic_num_vars,
        size.num_inputs,
        size.synthetic_num_cons,
    );

    println!("Producing instance...");
    let (inst, vars, inputs) = Instance::produce_synthetic_r1cs(
        size.synthetic_num_cons,
        size.synthetic_num_vars,
        size.num_inputs,
    );

    println!("Producing commitment...");
    let (comm, decomm) = SNARK::encode(&inst, &gens);

    let proof_start = Instant::now();

    println!("Running prover...");
    let mut prover_transcript = Transcript::new(b"snark_example");
    let proof = SNARK::prove(
        &inst,
        &comm,
        &decomm,
        vars,
        &inputs,
        &gens,
        &mut prover_transcript,
    );
    let proof_time = proof_start.elapsed();

    println!("Running verifier...");
    let verifier_time_start = Instant::now();
    let mut verifier_transcript = Transcript::new(b"snark_example");
    assert!(proof
        .verify(&comm, &inputs, &mut verifier_transcript, &gens)
        .is_ok());
    let verifier_time = verifier_time_start.elapsed();

    println!("\nBenchmark: {}", label);
    println!("Prover time: {:?}", proof_time);
    println!("Verifier time: {:?}", verifier_time);
    println!("Num constraints: {} \n", size.synthetic_num_cons);
}

fn main() {
    let args = Args::parse();
    if args.serialize_cost_models {
        derive_cost_model(args.max_instance_size);
        return;
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    let computations = if args.run_all {
        if args.computation_type.is_empty() {
            ComputationType::all_non_debug().to_vec()
        } else {
            args.computation_type.clone()
        }
    } else if args.computation_type.len() == 1 {
        args.computation_type.clone()
    } else if args.computation_type.is_empty() {
        panic!("computation type is required unless serializing cost models or using --run-all");
    } else {
        panic!("pass exactly one --computation-type unless using --run-all");
    };

    if computations.iter().any(|c| c.uses_cost_model()) && !cost_models_exist() {
        eprintln!("cost models not found; deriving them now");
        derive_cost_model(args.max_instance_size);
    }

    for computation_type in computations {
        run_computation(computation_type);
    }
}
