import unittest
from pprint import pprint
import os
import graph
from eval import *
from ir import *
from r1cs import *


BASE_PATH = "onnx"


class TestConstrainableModelConsistency(unittest.TestCase):
    def compare_outputs(self, outputs1, outputs2, tolerance=1e-5):
        o1 = outputs1[-1]
        o2 = outputs2[-1]
        # Get difference bteween the outputs
        self.assertTrue(
            np.allclose(o1, o2, atol=tolerance, rtol=tolerance, equal_nan=True),
            f"Outputs differ.",
        )
        print("Outputs are consistent.")

    def run_consistency_check(self, model, tolerance=1e-5):
        model_path = os.path.join(BASE_PATH, model) + ".onnx"
        original_model = load_onnx_model(model_path)
        outputs1, random_input = evaluate_onnx_model(original_model)
        r1cs = graph.get_r1cs(original_model, model, False)
        r1cs.constrain()
        witness, secondary_inputs = r1cs.eval_primary_model(random_input)
        r1cs.eval_secondary_model(witness, secondary_inputs)
        outputs2, _ = evaluate_onnx_model(r1cs.primary_model, input_data=random_input)
        eval_constraints(r1cs, witness)
        self.compare_outputs(outputs1, outputs2, tolerance)

    def test_simple_models(self):
        simple_model_files = [
            # "layer_norm",
            # "layer_norm_simple",
            # "erf_model"
            # "softmax_simple"
            # "matmul_small",
            # "filtered_model",
            # "softmax_multi_dim",
            # "eval_nano_64_v18"
            # "nano_gpt_first_transformer"
            # "split_softmax"
            # "simplified_dummy_model"
            # "reshape_ops",
            # "layer_norm_large",
            # "simple_ops_model",
            # "hybrid_model3"
            # "linear_w_transpose",
            # "simple_ops_model"
        ]
        for model_file in simple_model_files:
            with self.subTest(model_file=model_file):
                self.run_consistency_check(model_file)

        # self.run_consistency_check("nano_gpt_first_transformer", tolerance=1e-1)
        # self.run_consistency_check("erf_model", tolerance=1e-1)
        # self.run_consistency_check("layer_norm_softmax", tolerance=1e-4)
        # self.run_consistency_check("filtered_model", tolerance=1e-1)
        # self.run_consistency_check("simple_ops_model", tolerance=1e-1)
        self.run_consistency_check("logreg", tolerance=1e-1)


if __name__ == "__main__":
    unittest.main()
