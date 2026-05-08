use clap::Parser;
use model::AFloat;
use physics_examples::executor::PhysicsExampleExecutor;
use protocol::broker::JsonBroker;
use protocol::machine::run_actor;
use spain::actor::{SpainMessage, SpainVerifier};
use spain::traits::R1CSInstance;
use spain::verifier::VerifierState;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Physics example verifier actor over JsonBroker"
)]
struct Cli {
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 70)]
    scale_factor_bits: usize,
    #[arg(long, default_value_t = 0.1)]
    max_epsilon: f64,
    #[arg(long, default_value_t = 16)]
    chunk_size: usize,
    #[arg(long, default_value_t = 30000)]
    q_bits: usize,
    #[arg(long, default_value_t = 128)]
    precision: u16,
    #[arg(long, default_value_t = 20)]
    steps: usize,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let exec = PhysicsExampleExecutor::new(cli.steps);
    let metadata = exec.get_meta();

    let verifier_state: VerifierState<i128, AFloat, PhysicsExampleExecutor> = VerifierState::new(
        cli.max_epsilon,
        cli.batch_size,
        cli.scale_factor_bits,
        cli.q_bits,
        cli.precision,
        cli.chunk_size,
        exec,
        metadata,
    );
    let mut verifier = SpainVerifier::new(verifier_state);
    verifier.set_eval_model_name(format!("physics-steps-{}", cli.steps));

    run_actor::<SpainMessage, _, _>(&mut verifier, JsonBroker::new())
        .expect("physics verifier actor loop failed");
    verifier.print_eval();
}
