import numpy as np
import onnx
from onnxsim import simplify
from onnx import version_converter, numpy_helper
import argparse


def preprocess_model(in_path, out_path):
    original_model = onnx.load(in_path)
    converted_model = version_converter.convert_version(original_model, 18)
    model_simp, check = simplify(converted_model)
    assert check, "Simplified ONNX model could not be validated"
    for idx, initializer in enumerate(model_simp.graph.initializer):
        const_data = numpy_helper.to_array(initializer)
        if np.any(np.isneginf(const_data)):
            # -inf is unhandled by R1CS, so we replace it with a negative number.
            # These infinities show up in GPT-2's attention mask.
            # In theory, it is possible to to ignore these negative infinities and 
            # only constrain the upper triangular portion of the input to Softmax.
            # This is an optimization that would reduce the number of Softmax constraints 
            # by half.
            new_data = np.where(np.isneginf(const_data), -10, const_data)
            new_initializer = numpy_helper.from_array(new_data, name=initializer.name)
            model_simp.graph.initializer[idx].CopyFrom(new_initializer)
    print("ONNX model opset updated and simplified.")
    print(f"Saving preprocessed model to {out_path}")
    onnx.save(model_simp, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess ONNX for R1CS serialization."
    )
    parser.add_argument("in_path", help="Name of the ONNX model file to preprocess")
    parser.add_argument("out_path", help="Name of the output ONNX model file")
    args = parser.parse_args()
    update_opset(args.in_path, args.out_path)
