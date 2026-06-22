use clap::Parser;
use model::F128;
use spain::inputs::{DEFAULT_DATA_DIR, import_metadata};
use spain::simulate::{SpainConfig, measure_setup_time, stateful_simulate_with_config};
use spain::witness_gen::OnnxExecutor;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Evaluate Spain's stateful simulator on an ONNX-backed model",
    long_about = None
)]
struct Cli {
    /// Model name
    #[arg(long, default_value = "layernorm_32x768")]
    model: String,

    /// Batch size
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Scale factor bits used for fixed-point representation
    #[arg(long, default_value_t = 70)]
    scale_factor_bits: usize,

    /// Max epsilon for approximate checks
    #[arg(long, default_value_t = 0.1)]
    max_epsilon: f64,

    /// Chunk size for DARK commitment
    #[arg(long, default_value_t = 16)]
    chunk_size: usize,

    /// Number of bits for q in DARK
    #[arg(long, default_value_t = 30000)]
    q_bits: usize,

    /// DARK precision
    #[arg(long, default_value_t = 128)]
    precision: u16,

    /// Directory containing model data files
    #[arg(long, value_name = "DIR", default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,

    /// Use the precomputed input instead of a randomly generated one
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    use_same_input: bool,

    /// If true, will only measure setup time
    #[arg(long, default_value_t = false)]
    measure_setup: bool,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let metadata = import_metadata(&cli.data_dir, &cli.model);
    let wit_exec = OnnxExecutor::<F128>::new(
        cli.model.clone(),
        cli.data_dir.clone(),
        metadata,
        cli.use_same_input,
    );

    let config = SpainConfig {
        scale_factor_bits: cli.scale_factor_bits,
        max_epsilon: cli.max_epsilon,
        num_chunks: cli.chunk_size,
        precision: cli.precision,
        q_bits: cli.q_bits,
        batch_size: cli.batch_size,
        spartan_poly: false,
    };

    if cli.measure_setup {
        measure_setup_time::<F128, _, i128>(wit_exec, config);
        return;
    }

    let mut result = stateful_simulate_with_config::<F128, _, i128>(wit_exec, config);
    result.model_name = cli.model;
    result.batch_size = cli.batch_size;

    dbg!("Eval report");
    dbg!("sys: spain");
    dbg!(result);
}
