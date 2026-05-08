use clap::Parser;
use model::OnnxRunner;
#[allow(unused)]
use model::{AFloat, TFloat};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "../spain/data/gpt/primary_model.onnx")]
    dataset: String,
}

fn main() {
    let args = Args::parse();
    let mut runner = OnnxRunner::<f64>::read(args.dataset.as_str()).unwrap();
    println!("\n\n\n\n---\nonnx outputs\n---\n");
    let (_, perf) = runner.run_with_perf_breakdown(runner.rand_input());
    dbg!(perf);
}
