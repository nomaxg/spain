#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FP_SPARTAN_BENCHMARKS = [
    "softmax-32x32",
    "layernorm-32x768",
    "gelu-32x3072",
    "gpt2-seq-2",
    "gpt2-seq-32",
]

ONNX_BENCHMARKS = [
    "softmax-32x32",
    "layernorm-32x768",
    "gelu-32x3072",
    "gpt2-seq-2",
    "gpt2-seq-32",
]

LP_BENCHMARKS = [
    "adlittle",
    "afiro",
    "bnl1",
    "sc105",
    "sc50a",
    "sc50b",
    "scagr7",
    "scsd8",
]


def run_command(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
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


def run_fp_spartan_eval() -> None:
    manifest_path = ROOT / "fp-spartan-exp" / "Cargo.toml"
    cmd = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(manifest_path),
        "--",
        "--run-all",
    ]
    for computation_type in FP_SPARTAN_BENCHMARKS:
        cmd.extend(["--computation-type", computation_type])
    run_command(cmd, cwd=ROOT)


def run_onnx_eval(model: str, passes: int, phases: bool = False) -> None:
    spain_dir = ROOT / "spain"
    make_args = ["MODEL=" + model, f"BATCH_SIZE={passes}"]
    if phases:
        make_args.append("PHASE_BREAKDOWN=true")

    prover = subprocess.Popen(["make", "run_prover", *make_args], cwd=spain_dir)
    time.sleep(1)
    run_command(["make", "run_verifier", *make_args], cwd=spain_dir)
    prover.wait()


def run_onnx_native_eval(model: str) -> None:
    run_command(
        [
            sys.executable,
            str(ROOT / "circuit" / "eval.py"),
            str(ROOT / "circuit" / "export" / model),
        ],
        cwd=ROOT / "circuit",
    )


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


def run_lp_eval(model: str) -> None:
    otti_dir = ROOT / "otti-adapter"
    mps_arg = f"MPS_PATH=./datasets/{model}.mps"
    prover = subprocess.Popen(["make", "run_prover", mps_arg], cwd=otti_dir)
    time.sleep(1)
    run_command(["make", "run_verifier", mps_arg], cwd=otti_dir)
    prover.wait()


def run_test() -> None:
    run_lp_eval(LP_BENCHMARKS[0])


def run_eval(name: str, setup: bool, phases: bool, native: bool, passes: int) -> None:
    if name in ONNX_BENCHMARKS:
        if setup and native:
            raise ValueError("--setup-costs and --native cannot be used together")
        if phases and native:
            raise ValueError("--phases and --native cannot be used together")
        if native:
            run_onnx_native_eval(name)
        elif setup:
            run_onnx_setup(name, passes=passes)
        else:
            run_onnx_eval(name, passes=passes, phases=phases)
    elif name in LP_BENCHMARKS:
        if setup:
            raise ValueError("--setup-costs is only supported for ONNX benchmarks")
        if native:
            raise ValueError("--native is only supported for ONNX benchmarks")
        run_lp_eval(name)
    else:
        known_names = ", ".join(ONNX_BENCHMARKS + LP_BENCHMARKS)
        raise ValueError(f"Unknown benchmark '{name}'. Known benchmarks: {known_names}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        nargs="?",
        help="Benchmark name. ONNX names use code/spain; Otti LP names use code/otti-adapter.",
    )
    parser.add_argument(
        "--fp-spartan-estimates",
        action="store_true",
        dest="fp_spartan_estimates",
        help="Run circuit/script/fp_spartan_estimate.py.",
    )
    parser.add_argument(
        "--eval-fp-spartan",
        action="store_true",
        help="Run fp-spartan-exp for all computation benchmarks",
    )
    parser.add_argument(
        "--setup-costs",
        action="store_true",
        help="For ONNX benchmarks, measure setup costs",
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
        "--phases",
        action="store_true",
        help="For ONNX/Spain runs, print prover/verifier phase breakdowns.",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="For ONNX benchmarks, run circuit/eval.py on circuit/export/<benchmark>.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Batch size for Spain ONNX runs.",
    )
    args = parser.parse_args()

    if (
        not args.fp_spartan_estimates
        and not args.eval_fp_spartan
        and not args.benchmark
        and not args.build_onnx_circuits
        and not args.test
    ):
        parser.print_help()
        return

    if args.benchmark:
        run_eval(
            args.benchmark,
            args.setup_costs,
            args.phases,
            args.native,
            args.passes,
        )

    if args.fp_spartan_estimates:
        run_fp_spartan_estimates()

    if args.eval_fp_spartan:
        run_fp_spartan_eval()

    if args.build_onnx_circuits:
        build_onnx_circuits()
        print("ONNX benchmark circuits exported successfully")

    if args.test:
        run_test()


if __name__ == "__main__":
    main()
