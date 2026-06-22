use clap::Parser;
use examples::executor::PhysicsExampleExecutor;
use model::AFloat;
use protocol::machine::run_actor;
use spain::actor::{SpainMessage, SpainVerifier};
use spain::broker::SpainBroker;
use spain::traits::R1CSInstance;
use spain::verifier::VerifierState;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Physics example verifier actor over SpainBroker"
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
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    phase_breakdown: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    arith_progress: bool,
    #[arg(long, default_value_t = 20)]
    steps: usize,
    #[arg(long, default_value_t = 4)]
    grid_size: usize,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let exec = PhysicsExampleExecutor::new(cli.grid_size, cli.steps)
        .with_arith_progress(cli.arith_progress);
    let metadata = exec.get_meta();

    let verifier_state: VerifierState<i128, AFloat, PhysicsExampleExecutor> = VerifierState::new(
        cli.max_epsilon,
        cli.batch_size,
        cli.scale_factor_bits,
        cli.q_bits,
        cli.precision,
        cli.chunk_size,
        false,
        exec,
        metadata,
    );
    let mut verifier = SpainVerifier::new(verifier_state);
    verifier.set_eval_model_name(format!(
        "physics-grid-{}-steps-{}",
        cli.grid_size, cli.steps
    ));

    let (bytes_sent, time_spent, serialization_time) =
        run_actor::<SpainMessage, _, _>(&mut verifier, SpainBroker::new())
            .expect("physics verifier actor loop failed");

    let mut eval_result = verifier.get_eval_result();
    if !verifier.is_done() {
        panic!(
            "Verifier terminated without completing, ensure that are running the prover with sufficient memory."
        );
    }

    eval_result.num_constraints = verifier.num_constraints();
    eval_result.verifier_bytes_sent = bytes_sent;
    eval_result.total_verifier_actor_time = time_spent;
    eval_result.verifier_serialization_time = serialization_time;
    eval_result.calc_totals();

    eprintln!("Num constraints: {}", eval_result.num_constraints);

    if cli.phase_breakdown {
        eval_result.report_verifier_phases();
    } else {
        eval_result.report_verifier_time();
    }
}
