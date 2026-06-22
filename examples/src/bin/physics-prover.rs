use clap::Parser;
use examples::executor::PhysicsExampleExecutor;
use model::AFloat;
use protocol::machine::run_actor;
use spain::actor::{SpainMessage, SpainProver};
use spain::broker::SpainBroker;
use spain::prover::{ProverState, scale_factor};
use spain::traits::R1CSInstance;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Physics example prover actor over SpainBroker"
)]
struct Cli {
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 70)]
    scale_factor_bits: usize,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    phase_breakdown: bool,
    #[arg(long, default_value_t = 20)]
    steps: usize,
    #[arg(long, default_value_t = 4)]
    grid_size: usize,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let exec = PhysicsExampleExecutor::new(cli.grid_size, cli.steps);
    let metadata = exec.get_meta();
    let scale_factor: AFloat = scale_factor(cli.scale_factor_bits);

    let prover_state: ProverState<i128, AFloat, PhysicsExampleExecutor> =
        ProverState::new(exec, scale_factor, metadata, cli.batch_size, false);
    let mut prover = SpainProver::new(prover_state);
    prover.set_eval_model_name(format!(
        "physics-grid-{}-steps-{}",
        cli.grid_size, cli.steps
    ));

    let (bytes_sent, time_spent, serialization_time) =
        run_actor::<SpainMessage, _, _>(&mut prover, SpainBroker::new())
            .expect("physics prover actor loop failed");

    if !prover.is_done() {
        panic!(
            "Prover terminated before protocol completion, verifier may have panicked or disconnected",
        );
    }

    let mut eval_result = prover.get_eval_result();
    eval_result.num_constraints = prover.num_constraints();
    eval_result.prover_bytes_sent = bytes_sent;
    eval_result.total_prover_actor_time = time_spent;
    eval_result.prover_serialization_time = serialization_time;
    eval_result.calc_totals();

    eprintln!("Num constraints: {}", eval_result.num_constraints);

    if cli.phase_breakdown {
        eval_result.report_prover_phases();
    } else {
        eval_result.report_prover_time();
    }
}
