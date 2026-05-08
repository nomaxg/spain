use clap::Parser;
use model::AFloat;
use physics_examples::executor::PhysicsExampleExecutor;
use protocol::broker::JsonBroker;
use protocol::machine::run_actor;
use spain::actor::{SpainMessage, SpainProver};
use spain::prover::{ProverState, scale_factor};
use spain::traits::R1CSInstance;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Physics example prover actor over JsonBroker"
)]
struct Cli {
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    #[arg(long, default_value_t = 70)]
    scale_factor_bits: usize,
    #[arg(long, default_value_t = 20)]
    steps: usize,
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let exec = PhysicsExampleExecutor::new(cli.steps);
    let metadata = exec.get_meta();
    let scale_factor: AFloat = scale_factor(cli.scale_factor_bits);

    let prover_state: ProverState<i128, AFloat, PhysicsExampleExecutor> =
        ProverState::new(exec, scale_factor, metadata, cli.batch_size);
    let mut prover = SpainProver::new(prover_state);
    prover.set_eval_model_name(format!("physics-steps-{}", cli.steps));

    run_actor::<SpainMessage, _, _>(&mut prover, JsonBroker::new())
        .expect("physics prover actor loop failed");
    prover.print_eval();
}
