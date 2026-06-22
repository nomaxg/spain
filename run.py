#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_LABEL = "measurements"
VERIFIER_WAIT = 1
NUM_RUNS = 5
CIRCUIT_EXPORT_DIR = ROOT / "circuit" / "export"
LP_DATASET_DIR = ROOT / "otti-adapter" / "datasets"
PHYSICS_EXAMPLES_DIR = ROOT / "examples"

NATIVE_ONNX_TIMING = {
    "softmax-32x32": {"warmup_samples": 1, "measured_samples": 10, "inner_iterations": 200000},
    "layernorm-32x768": {"warmup_samples": 1, "measured_samples": 10, "inner_iterations": 100000},
    "gelu-32x3072": {"warmup_samples": 1, "measured_samples": 10, "inner_iterations": 5000},
    "gpt2-seq-2": {"warmup_samples": 1, "measured_samples": 10, "inner_iterations": 500},
    "gpt2-seq-32": {"warmup_samples": 1, "measured_samples": 10, "inner_iterations": 500},
}
NATIVE_LP_TIMING = {
    "adlittle": {"num_samples": 5, "iters_per_sample": 3000, "warmup_samples": 3, "lp_solve_iters": 20},
    "afiro": {"num_samples": 5, "iters_per_sample": 3000, "warmup_samples": 3, "lp_solve_iters": 20},
    "sc105": {"num_samples": 5, "iters_per_sample": 3000, "warmup_samples": 3, "lp_solve_iters": 20},
    "scagr7": {"num_samples": 5, "iters_per_sample": 3000, "warmup_samples": 3, "lp_solve_iters": 20},
    "scsd8": {"num_samples": 5, "iters_per_sample": 3000, "warmup_samples": 3, "lp_solve_iters": 20},
}
NATIVE_PHYSICS_TIMING = {
    "fluid-small": {"grid_size": 8, "num_steps": 10, "num_samples": 5, "iters_per_sample": 5000, "warmup_samples": 2},
    "fluid-large": {"grid_size": 16, "num_steps": 10, "num_samples": 5, "iters_per_sample": 1, "warmup_samples": 2},
}
NATIVE_ZKLP_TIMING = {
    "geolocation": {"num_samples": 5, "iters_per_sample": 5000, "warmup_samples": 2},
}
PHYSICS_BENCHMARKS = list(NATIVE_PHYSICS_TIMING.keys())
LOCATION_PRIVACY_BENCHMARKS = list(NATIVE_ZKLP_TIMING.keys())
PHYSICS_MODEL_NAMES = {
    "fluid-small": "physics-d8-t10",
    "fluid-large": "physics-d16-t10",
}

ONNX_BENCHMARKS = [
    "softmax-32x32",
    "layernorm-32x768",
    "gelu-32x3072",
    "gpt2-seq-2",
    "gpt2-seq-32",
]

ZKLP_BENCHMARKS = [
    "softmax-32x32",
    "layernorm-32x768",
    "gelu-32x3072",
    "physics-d8-t10",
    "physics-d16-t10",
]

LP_BENCHMARKS = [
    "adlittle",
    "afiro",
    "sc105",
    "scagr7",
    "scsd8",
]

EVAL_DIR = ROOT / "eval" / RUN_LABEL


def physics_model_name(benchmark: str) -> str:
    return PHYSICS_MODEL_NAMES[benchmark]


def run_eval_actor_command(
    cmd: list[str],
    cwd: Path,
    system: str,
    benchmark_dir: str,
    file_stem: str,
    prover: bool,
    run_idx: int,
) -> None:
    role = "prover" if prover else "verifier"
    out_dir = EVAL_DIR / system / benchmark_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_stem}_{role}_{run_idx}.txt"
    with out_path.open("w") as stderr_file:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stderr=stderr_file,
            text=True,
        )
    result.check_returncode()


def run_eval_actor_output_command(
    cmd: list[str],
    cwd: Path,
    system: str,
    benchmark_dir: str,
    output_name: str,
) -> None:
    out_dir = EVAL_DIR / system / benchmark_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    with out_path.open("w") as stderr_file:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stderr=stderr_file,
            text=True,
        )
    result.check_returncode()


def run_command(
    cmd, cwd, env=None
) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_fp_spartan_estimates() -> None:
    env = dict(os.environ)
    pythonpath_entries = [str(ROOT / "circuit")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)
    run_command(
        [
            sys.executable,
            str(ROOT / "circuit" / "script" / "fp_spartan_estimate.py"),
        ],
        cwd=ROOT / "circuit",
        env=env,
    )


def run_zklp_fe_cost_model() -> None:
    run_command(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "cost_model",
        ],
        cwd=ROOT / "spain",
    )


def run_onnx_eval(model: str, passes: int, phases: bool = False) -> None:
    env = dict(os.environ)
    spain_dir = ROOT / "spain"
    make_args = ["MODEL=" + model, f"BATCH_SIZE={passes}"]
    if phases:
        make_args.append("PHASE_BREAKDOWN=true")

    prover = subprocess.Popen(["make", "run_prover", *make_args], cwd=spain_dir, env=env)
    time.sleep(VERIFIER_WAIT)
    run_command(["make", "run_verifier", *make_args], cwd=spain_dir, env=env)
    prover.wait()


def run_physics_eval(model: str, passes: int, phases: bool = False) -> None:
    timing = NATIVE_PHYSICS_TIMING[model]
    env = dict(os.environ)
    physics_dir = ROOT / "examples"
    make_args = [
        f"STEPS={timing['num_steps']}",
        f"GRID_SIZE={timing['grid_size']}",
        f"BATCH_SIZE={passes}",
    ]
    if phases:
        make_args.append("PHASE_BREAKDOWN=true")

    prover = subprocess.Popen(["make", "run_prover", *make_args], cwd=physics_dir, env=env)
    time.sleep(VERIFIER_WAIT)
    run_command(["make", "run_verifier", *make_args], cwd=physics_dir, env=env)
    prover.wait()


def run_spain_zklp_baseline_eval(model: str, passes: int) -> None:
    env = dict(os.environ)
    spain_dir = ROOT / "spain"
    spain_model = physics_model_name(model) if model in PHYSICS_MODEL_NAMES else model
    make_args = ["MODEL=" + spain_model, f"BATCH_SIZE={passes}", "ZKLP=true"]

    prover = subprocess.Popen(["make", "run_prover", *make_args], cwd=spain_dir, env=env)
    time.sleep(VERIFIER_WAIT)
    run_command(["make", "run_verifier", *make_args], cwd=spain_dir, env=env)
    prover.wait()


def run_onnx_native_eval(model: str) -> None:
    timing = NATIVE_ONNX_TIMING[model]
    run_command(
        [
            sys.executable,
            str(ROOT / "circuit" / "eval.py"),
            str(CIRCUIT_EXPORT_DIR / model),
            "--warmup-samples",
            str(timing["warmup_samples"]),
            "--measured-samples",
            str(timing["measured_samples"]),
            "--inner-iterations",
            str(timing["inner_iterations"]),
        ],
        cwd=ROOT / "circuit",
    )


def run_onnx_native_times() -> None:
    for model in ONNX_BENCHMARKS:
        print(f"native {model}")
        run_onnx_native_eval(model)


def run_lp_native_eval(model: str) -> None:
    timing = NATIVE_LP_TIMING[model]
    run_command(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "cert_check",
            "--",
            "--mps-path",
            str(LP_DATASET_DIR / f"{model}.mps"),
            "--num-samples",
            str(timing["num_samples"]),
            "--iters-per-sample",
            str(timing["iters_per_sample"]),
            "--warmup-samples",
            str(timing["warmup_samples"]),
            "--lp-solve-iters",
            str(timing["lp_solve_iters"]),
        ],
        cwd=ROOT / "otti-adapter",
    )


def run_physics_native_eval(model: str) -> None:
    timing = NATIVE_PHYSICS_TIMING[model]
    run_command(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "native_sim",
            "--",
            "--grid-size",
            str(timing["grid_size"]),
            "--num-steps",
            str(timing["num_steps"]),
            "--num-samples",
            str(timing["num_samples"]),
            "--iters-per-sample",
            str(timing["iters_per_sample"]),
            "--warmup-samples",
            str(timing["warmup_samples"]),
        ],
        cwd=PHYSICS_EXAMPLES_DIR,
    )


def run_zklp_native_eval(model: str) -> None:
    timing = NATIVE_ZKLP_TIMING[model]
    run_command(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "native_zklp",
            "--",
            "--num-samples",
            str(timing["num_samples"]),
            "--iters-per-sample",
            str(timing["iters_per_sample"]),
            "--warmup-samples",
            str(timing["warmup_samples"]),
        ],
        cwd=PHYSICS_EXAMPLES_DIR,
    )


def run_native_times() -> None:
    total_start = time.perf_counter()
    for model in ONNX_BENCHMARKS:
        print(f"native onnx {model}")
        run_onnx_native_eval(model)
        print()
    for model in LP_BENCHMARKS:
        print(f"native lp {model}")
        run_lp_native_eval(model)
        print()
    for model in NATIVE_PHYSICS_TIMING:
        print(f"native physics {model}")
        run_physics_native_eval(model)
        print()
    for model in LOCATION_PRIVACY_BENCHMARKS:
        print(f"native zklp {model}")
        run_zklp_native_eval(model)
        print()
    total_elapsed_s = time.perf_counter() - total_start
    print(f"eval native total wall time: {total_elapsed_s:.6f}s")


def run_onnx_setup(model: str, passes: int) -> None:
    run_command(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "eval",
            "--",
            "--model",
            model,
            "--batch-size",
            str(passes),
            "--measure-setup",
        ],
        cwd=ROOT / "spain",
    )


def build_onnx_circuits() -> None:
    circuit_dir = ROOT / "circuit"
    for benchmark in ONNX_BENCHMARKS:
        run_command(
            [sys.executable, "-m", "script.serialize", benchmark],
            cwd=circuit_dir,
        )


def run_lp_eval(model: str, passes: int = 1) -> None:
    otti_dir = ROOT / "otti-adapter"

    mps_arg = f"MPS_PATH=./datasets/{model}.mps"
    make_args = [mps_arg, f"BATCH_SIZE={passes}"]
    prover = subprocess.Popen(["make", "run_prover", *make_args], cwd=otti_dir)
    time.sleep(VERIFIER_WAIT)
    run_command(["make", "run_verifier", *make_args], cwd=otti_dir)
    prover.wait()


def run_lp_baseline_eval(model: str, passes: int = 1) -> None:
    otti_dir = ROOT / "otti-adapter"

    mps_arg = f"MPS_PATH=./datasets/{model}.mps"
    make_args = [mps_arg, f"BATCH_SIZE={passes}", "OTTI_SID=true"]
    prover = subprocess.Popen(["make", "run_prover", *make_args], cwd=otti_dir)
    time.sleep(VERIFIER_WAIT)
    run_command(["make", "run_verifier", *make_args], cwd=otti_dir)
    prover.wait()


def run_zklp_eval(model: str, passes: int = 1, phases: bool = False) -> None:
    examples_dir = ROOT / "examples"
    make_args = [f"BATCH_SIZE={passes}"]
    if phases:
        make_args.append("PHASE_BREAKDOWN=true")

    prover = subprocess.Popen(["make", "run_prover_zklp", *make_args], cwd=examples_dir)
    time.sleep(VERIFIER_WAIT)
    run_command(["make", "run_verifier_zklp", *make_args], cwd=examples_dir)
    prover.wait()


def run_test() -> None:
    run_lp_eval(LP_BENCHMARKS[0])


def run_eval(name: str, passes: int, baseline: bool) -> None:
    if baseline and name in LOCATION_PRIVACY_BENCHMARKS:
        raise ValueError(
            "see https://github.com/tumberger/zk-Location ZKLP repo to reproduce that baseline's experimental results"
        )

    if name in ONNX_BENCHMARKS:
        if baseline:
            run_spain_zklp_baseline_eval(name, passes=passes)
        else:
            run_onnx_eval(name, passes=passes)

    elif name in LP_BENCHMARKS:
        if baseline:
            run_lp_baseline_eval(name, passes=passes)
        else:
            run_lp_eval(name, passes=passes)

    elif name in PHYSICS_BENCHMARKS:
        if baseline:
            run_spain_zklp_baseline_eval(name, passes=passes)
        else:
            run_physics_eval(name, passes=passes)

    elif name in LOCATION_PRIVACY_BENCHMARKS:
        run_zklp_eval(name, passes=passes)

    else:
        known_names = ", ".join(
            ONNX_BENCHMARKS + LP_BENCHMARKS + PHYSICS_BENCHMARKS + LOCATION_PRIVACY_BENCHMARKS
        )
        raise ValueError(f"Unknown benchmark '{name}'. Known benchmarks: {known_names}")
    
def eval_actor(prover = True): 
    make_cmd = ["make", "run_prover"] if prover else ["make", "run_verifier"]
    gpt2_seq_32_sweep_batch_sizes = [16, 8, 4, 2]

    # Physics benchmarks (Spain arithmetization), T=10 and D in {8, 16}
    for grid_size in (8, 16):
        benchmark_dir = f"physics-d{grid_size}-t10"
        file_stem = benchmark_dir
        for run_idx in range(1, NUM_RUNS + 1):
            physics_dir = ROOT / "examples"
            make_args = [
                "STEPS=10",
                f"GRID_SIZE={grid_size}",
                "PHASE_BREAKDOWN=true",
            ]
            print(f"spain {benchmark_dir} run {run_idx}")
            run_eval_actor_command(
                make_cmd + make_args,
                cwd=physics_dir,
                system="spain",
                benchmark_dir=benchmark_dir,
                file_stem=file_stem,
                prover=prover,
                run_idx=run_idx,
            )
            if not prover:
                # sleep a bit so that prover has time to run again
                time.sleep(VERIFIER_WAIT)


    # ONNX benchmarks (Spain arithmetization)
    for model in ONNX_BENCHMARKS:
        file_stem = model.replace("-", "_")
        for run_idx in range(1, NUM_RUNS + 1):
            spain_dir = ROOT / "spain"
            make_args = ["MODEL=" + model, "PHASE_BREAKDOWN=true"]
            print(f"spain {model} run {run_idx}")
            run_eval_actor_command(
                make_cmd + make_args,
                cwd=spain_dir,
                system="spain",
                benchmark_dir=model,
                file_stem=file_stem,
                prover=prover,
                run_idx=run_idx,
            )
            if not prover: 
                # sleep a bit so that prover has time to run again
                time.sleep(VERIFIER_WAIT)

    # ZKLP benchmarks
    for batch in [512]:
        benchmark_dir = f"location-privacy"
        for run_idx in range (1, NUM_RUNS + 1):
            examples_dir = ROOT / "examples"
            zklp_make_cmd = ["make", "run_prover_zklp"] if prover else ["make", "run_verifier_zklp"]
            make_args = [f"BATCH_SIZE={batch}", "PHASE_BREAKDOWN=true"]
            file_stem = f"location_privacy_b{batch}"
            run_eval_actor_command(
                zklp_make_cmd + make_args,
                cwd=examples_dir,
                system="spain",
                benchmark_dir=benchmark_dir,
                file_stem=file_stem,
                prover=prover,
                run_idx=run_idx,
            )
            if not prover:
                # sleep a bit so that prover has time to run again
                time.sleep(VERIFIER_WAIT)

                
    # Otti benchmarks (Spain arithmetization)
    for model in LP_BENCHMARKS:
        file_stem = model
        for run_idx in range(1, NUM_RUNS + 1):
            spain_dir = ROOT / "otti-adapter"
            make_args = [f"MPS_PATH=./datasets/{model}.mps", "PHASE_BREAKDOWN=true"]
            print(f"spain {model} run {run_idx}")
            run_eval_actor_command(
                make_cmd + make_args,
                cwd=spain_dir,
                system="spain",
                benchmark_dir=model,
                file_stem=file_stem,
                prover=prover,
                run_idx=run_idx,
            )
            if not prover: 
                # sleep a bit so that prover has time to run again
                time.sleep(VERIFIER_WAIT)

    # Otti benchmarks (Otti arithmetization)
    for model in LP_BENCHMARKS:
        file_stem = model
        for run_idx in range(1, NUM_RUNS + 1):
            spain_dir = ROOT / "otti-adapter"
            make_args = [f"MPS_PATH=./datasets/{model}.mps", "PHASE_BREAKDOWN=true", "OTTI_SID=true"]
            print(f"otti-sid {model} run {run_idx}")
            run_eval_actor_command(
                make_cmd + make_args,
                cwd=spain_dir,
                system="otti-sid",
                benchmark_dir=model,
                file_stem=file_stem,
                prover=prover,
                run_idx=run_idx,
            )
            if not prover: 
                # sleep a bit so that prover has time to run again
                time.sleep(VERIFIER_WAIT)

    # ONNX benchmarks (ZKLP arithmetization)
    for model in ZKLP_BENCHMARKS:
        file_stem = model.replace("-", "_")
        for run_idx in range(1, NUM_RUNS + 1):
            spain_dir = ROOT / "spain"
            make_args = ["MODEL=" + model, "PHASE_BREAKDOWN=true", "ZKLP=true"]
            print(f"zklp-sid {model} run {run_idx}")
            run_eval_actor_command(
                make_cmd + make_args,
                cwd=spain_dir,
                system="zklp-sid",
                benchmark_dir=model,
                file_stem=file_stem,
                prover=prover,
                run_idx=run_idx,
            )
            if not prover: 
                # sleep a bit so that prover has time to run again
                time.sleep(VERIFIER_WAIT)


    # GPT-2 seq-32 batch sweep (Spain arithmetization)
    for batch_size in gpt2_seq_32_sweep_batch_sizes:
        benchmark_dir = "gpt2-seq-32"
        file_stem = benchmark_dir.replace("-", "_")
        spain_dir = ROOT / "spain"
        make_args = [
            "MODEL=gpt2-seq-32",
            f"BATCH_SIZE={batch_size}",
            "PHASE_BREAKDOWN=true",
        ]
        for run_idx in range(1, NUM_RUNS + 1):
            role = "prover" if prover else "verifier"
            print(f"spain {benchmark_dir} {role} b{batch_size} run {run_idx}")
            run_eval_actor_output_command(
                make_cmd + make_args,
                cwd=spain_dir,
                system="spain",
                benchmark_dir=benchmark_dir,
                output_name=f"{file_stem}_{role}_b{batch_size}_{run_idx}.txt",
            )
            if not prover:
                # sleep a bit so that prover has time to run again
                time.sleep(VERIFIER_WAIT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        nargs="?",
        help="Benchmark name. See README for options.",
    )
    parser.add_argument(
        "--build-onnx-circuits",
        action="store_true",
        help="Serialize all ONNX benchmark circuits via circuit/script.serialize",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test on the smallest LP benchmark",
    )
    parser.add_argument(
        "--eval-native",
        action="store_true",
        help="Run native timings for ONNX, LP, and physics benchmarks.",
    )
    parser.add_argument(
        "--onnx-zklp-constraints",
        action="store_true",
        help="Calculate ZKLP-FE constraint counts for ONNX benchmarks.",
    )
    parser.add_argument(
        "--zklp-fe-cost-model",
        action="store_true",
        help="Derive cost model for ZKLP-FE",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Batch size for Spain ONNX runs.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run the baseline system for the selected benchmark: ZKLP-SID for ONNX/physics and Otti-FE for LP benchmarks.",
    )
    parser.add_argument(
        "--eval-prover",
        action="store_true",
        help="Run prover eval script",
    )
    parser.add_argument(
        "--eval-verifier",
        action="store_true",
        help="Run verifier eval script",
    )
    parser.add_argument(
        "--show-help",
        action="store_true",
        help="Show help message",
    )
    args = parser.parse_args()

    if args.show_help:
        parser.print_help()
        return

    if args.eval_native:
        run_native_times()

    if args.onnx_zklp_constraints:
        run_fp_spartan_estimates()

    if args.zklp_fe_cost_model:
        run_zklp_fe_cost_model()

    if args.eval_prover:
        eval_actor(prover=True)

    if args.eval_verifier:
        eval_actor(prover=False)

    if args.benchmark:
        run_eval(
            args.benchmark,
            args.passes,
            args.baseline,
        )

    if args.build_onnx_circuits:
        build_onnx_circuits()
        print("ONNX benchmark circuits exported successfully")

    if args.test:
        run_test()


if __name__ == "__main__":
    main()
