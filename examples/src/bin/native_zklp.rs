use clap::Parser;
use examples::circuit_zklp;
use examples::r1cs::NativeExec;
use statrs::statistics::Statistics;
use std::hint::black_box;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "Native physics simulation benchmark")]
struct Cli {
    #[arg(long, default_value_t = 5)]
    num_samples: usize,
    #[arg(long, default_value_t = 5000)]
    iters_per_sample: usize,
    #[arg(long, default_value_t = 2)]
    warmup_samples: usize,
    #[arg(long, default_value_t = false)]
    print_samples: bool,
}

fn run_native_location_privacy_once() -> () {
    let mut exec = NativeExec::zklp_init();
    circuit_zklp(&mut exec);
}

fn main() {
    let cli = Cli::parse();

    // Discard warmup samples
    for _ in 0..cli.warmup_samples {
        for _ in 0..cli.iters_per_sample {
            black_box(run_native_location_privacy_once())
        }
    }

    let mut samples_ms = Vec::with_capacity(cli.num_samples);
    for sample_idx in 0..cli.num_samples {
        let start = Instant::now();
        for _ in 0..cli.iters_per_sample {
            black_box(run_native_location_privacy_once())
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
}
