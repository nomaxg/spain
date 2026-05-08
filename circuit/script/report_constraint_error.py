import argparse
import os
import graph
from eval import load_onnx_model, evaluate_onnx_model
from ir import report_constraint_error
from r1cs import *


BASE_PATH = "onnx"


def run_report(model_name):
    model_path = os.path.join(BASE_PATH, model_name) + ".onnx"
    original_model = load_onnx_model(model_path)
    _, random_input = evaluate_onnx_model(original_model)
    r1cs = graph.get_r1cs(original_model, model_name, False)
    r1cs.constrain()
    witness, secondary_inputs = r1cs.eval_primary_model(random_input)
    r1cs.eval_secondary_model(witness, secondary_inputs)
    report = report_constraint_error(r1cs, witness)
    print("Constraint error report")
    print(f"model:{model_name}")
    print(f"l2:{report['l2']}")
    print(f"l4:{report['l4']}")
    print(f"constraints:{report['satisfied_constraints']}/{report['total_constraints']}")
    print(f"num_failed:{report['num_failed_constraints']}")
    print(f"histogram_step:{report['histogram_step']}")
    histogram = report["histogram"]
    for bin_start in sorted(histogram.keys()):
        bin_end = bin_start + report["histogram_step"]
        print(
            f"histogram:{bin_start:.3e}-{bin_end:.3e}:{histogram[bin_start]}"
        )
    print("End report")


def main():
    parser = argparse.ArgumentParser(
        description="Report constraint error norms and histogram for models."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model names without the .onnx extension",
    )
    args = parser.parse_args()
    for model_name in args.models:
        run_report(model_name)


if __name__ == "__main__":
    main()
