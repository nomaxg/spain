use clap::Parser;
use lpsolve::{MPSOptions, ProblemBuilder};
use otti_adapter::LPSpain;
use statrs::statistics::Statistics;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "LP optimality certificate benchmark")]
struct Cli {
    #[arg(
        long,
        default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/datasets/sc105.mps")
    )]
    mps_path: PathBuf,
    #[arg(long, default_value_t = 5)]
    num_samples: usize,
    #[arg(long, default_value_t = 3000)]
    iters_per_sample: usize,
    #[arg(long, default_value_t = 3)]
    warmup_samples: usize,
    #[arg(long, default_value_t = false)]
    print_samples: bool,
    #[arg(long, default_value_t = 10)]
    lp_solve_iters: usize,
}

fn main() {
    let cli = Cli::parse();
    assert!(
        cli.num_samples >= 2,
        "num-samples must be >= 2 to compute standard deviation"
    );
    assert!(cli.iters_per_sample >= 1, "iters-per-sample must be >= 1");

    let mps_path = cli.mps_path.to_str().expect("mps path invalid").to_string();
    let lp = LPSpain::parse_mps(&mps_path);
    for _ in 0..cli.warmup_samples {}
    let problem =
        ProblemBuilder::from_fixedmps_file(mps_path.clone(), MPSOptions::empty()).unwrap();

    // bench lp solve
    let mut lp_solve_samples_ms = Vec::with_capacity(cli.lp_solve_iters);
    for _ in 0..cli.lp_solve_iters {
        let mut problem = problem.clone();
        let start = Instant::now();
        let status = problem.solve();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(status, lpsolve::SolveStatus::Optimal, "LP solve failed");
        lp_solve_samples_ms.push(elapsed_ms);
    }

    let mean_solve_ms = lp_solve_samples_ms.as_slice().mean();
    let std_dev_ms = lp_solve_samples_ms.as_slice().std_dev();
    let rel_std_dev = std_dev_ms / mean_solve_ms;

    println!("dataset: {}", mps_path);
    println!("LP SOLVE");
    println!("num samples: {}", cli.num_samples);
    println!("mean solve time: {:.6} ms", mean_solve_ms);
    println!("std_dev: {:.6} ms", std_dev_ms);
    println!("relative std_dev: {:.2}%", rel_std_dev * 100.0);

    if rel_std_dev > 0.05 {
        println!(
            "WARNING: relative std deviation too high: {:.2}% > 5%",
            rel_std_dev * 100.0
        );
    }

    let sols = lp.solve();

    // Cert check
    // Discard warmup samples
    for _ in 0..cli.warmup_samples {
        for _ in 0..cli.iters_per_sample {
            lp.check_opt_certifiate(&sols);
        }
    }

    let mut samples_ms = Vec::with_capacity(cli.num_samples);
    for sample_idx in 0..cli.num_samples {
        let start = Instant::now();
        for _ in 0..cli.iters_per_sample {
            lp.check_opt_certifiate(&sols);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0 / cli.iters_per_sample as f64;
        samples_ms.push(elapsed_ms);
        if cli.print_samples {
            println!(
                "sample {}/{}: {:.6} ms",
                sample_idx + 1,
                cli.num_samples,
                elapsed_ms
            );
        }
    }

    let mean_ms = samples_ms.as_slice().mean();
    let std_dev_ms = samples_ms.as_slice().std_dev();
    let rel_std_dev = std_dev_ms / mean_ms;

    println!("num samples: {}", cli.num_samples);
    println!("iters/sample: {}", cli.iters_per_sample);
    println!("warmup samples: {}", cli.warmup_samples);
    println!("mean: {:.6} ms", mean_ms);
    println!("std_dev: {:.6} ms", std_dev_ms);
    println!("relative std_dev: {:.2}%", rel_std_dev * 100.0);

    if rel_std_dev > 0.05 {
        println!(
            "WARNING: relative std deviation too high: {:.2}% > 5%",
            rel_std_dev * 100.0
        );
    }

    println!("TOTAL native time: {:.6} ms", mean_ms + mean_solve_ms);
}
