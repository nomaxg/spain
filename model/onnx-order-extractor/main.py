import argparse
import onnx
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Extract the output order from given onnx file, in the same dir")
    parser.add_argument("model", type=str, help="Model path")
    args = parser.parse_args()

    model = onnx.load(args.model)
    order = [tensor.name for tensor in model.graph.output if "public" in tensor.doc_string]
    order.extend([tensor.name for tensor in model.graph.output if "primary" in tensor.doc_string])
    with open(Path(args.model).parent / ("output_order.json"), "w") as f:
        json.dump(order, f)


if __name__ == "__main__":
    main()
