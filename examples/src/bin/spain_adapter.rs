use clap::Parser;
use examples::executor::PhysicsExampleExecutor;
use spain::simulate::{SpainConfig, stateful_simulate_with_config};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Run the physics example circuit through the Spain prover stack",
    long_about = None
)]
struct Cli {
    /// Batch size for the prover
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// Number of fractional bits for fixed-point scaling
    #[arg(long, default_value_t = 53)]
    scale_factor_bits: usize,

    /// Maximum epsilon tolerated when checking the execution trace
    #[arg(long, default_value_t = 0.1)]
    max_epsilon: f64,

    /// Chunk size for DARK commitments
    #[arg(long, default_value_t = 4)]
    chunk_size: usize,

    /// Bit length for DARK's base field
    #[arg(long, default_value_t = 30000)]
    q_bits: usize,

    /// Number of simulated fluid steps
    #[arg(long, default_value_t = 20)]
    steps: usize,
    /// Grid size for fluid simulation
    #[arg(long, default_value_t = 4)]
    grid_size: usize,
}

fn main() {
    env_logger::init();
    let Cli {
        batch_size,
        scale_factor_bits,
        max_epsilon,
        chunk_size,
        q_bits,
        steps,
        grid_size,
    } = Cli::parse();

    let exec = PhysicsExampleExecutor::new(grid_size, steps);
    let config = SpainConfig {
        scale_factor_bits,
        max_epsilon,
        num_chunks: chunk_size,
        q_bits,
        batch_size,
        ..SpainConfig::default()
    };

    stateful_simulate_with_config::<model::AFloat, _, i128>(exec, config);
}
