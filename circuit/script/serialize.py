from eval import *
from ir import *
from r1cs import *

import argparse
import graph
import time


def serialize_model(name, batched=False):
    start = time.time()
    model_path = f"onnx/{name}.onnx"
    model = load_onnx_model(model_path)
    r1cs = graph.get_r1cs(model, name, batched)
    r1cs.export_constraints_continuously = True
    r1cs.constrain()
    r1cs.serializer.export(r1cs)
    print(f"elapsed: {time.time() - start}")


def main():
    parser = argparse.ArgumentParser(description="Serialize ONNX model to R1CS.")
    parser.add_argument(
        "model", help="Name of the ONNX model file (without the .onnx extension)"
    )
    args = parser.parse_args()

    serialize_model(args.model, True)


if __name__ == "__main__":
    main()
