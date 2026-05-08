import onnx
from onnx import helper, TensorProto, shape_inference, numpy_helper
import numpy as np


def build_secondary_model(r1cs):
    R_tensor = helper.make_tensor_value_info(
        f"R", onnx.TensorProto.FLOAT, [r1cs.num_random_variables, 1]
    )
    secondary_nodes = []
    secondary_inputs = [R_tensor]
    secondary_outputs = []
    initializers = []
    index = 0
    for meta in r1cs.nodes.values():
        if meta.op == "MatMul":
            inputs = meta.node.input
            A = r1cs.nodes.get(inputs[0])
            B = r1cs.nodes.get(inputs[1])
            if B.is_var and not r1cs.full_variable_matmul:
                B_tensor = helper.make_tensor_value_info(
                    f"Input_{index}", onnx.TensorProto.FLOAT, B.shape
                )
                B_r_tensor = helper.make_tensor_value_info(
                    f"B_r_{index}", TensorProto.FLOAT, list(B.shape[:-1])
                )
                B_r_tensor.doc_string = "secondary_constraint"

                starts_name = f"starts_{index}"
                ends_name = f"ends_{index}"
                R_slice_name = f"R_slice_{index}"

                starts_initializer = numpy_helper.from_array(
                    np.array([0], dtype=np.int64), name=starts_name
                )
                ends_initializer = numpy_helper.from_array(
                    np.array([B.shape[-1]], dtype=np.int64), name=ends_name
                )
                R_slice = helper.make_node(
                    "Slice",
                    inputs=[R_tensor.name, starts_name, ends_name],
                    outputs=[R_slice_name],
                )
                B_r_node = helper.make_node(
                    "MatMul",
                    inputs=[B_tensor.name, R_slice_name],
                    outputs=[B_r_tensor.name],
                )
                secondary_nodes.extend(
                    [
                        R_slice,
                        B_r_node,
                    ]
                )
                secondary_inputs.append(B_tensor)
                secondary_outputs.append(B_r_tensor)
                initializers.extend([starts_initializer, ends_initializer])
                index += 1
            elif not B.is_var:
                A_tensor = helper.make_tensor_value_info(
                    f"Input_{index}", onnx.TensorProto.FLOAT, A.shape
                )
                starts_name = f"starts_{index}"
                ends_name = f"ends_{index}"
                R_slice_name = f"R_slice_{index}"

                starts_initializer = numpy_helper.from_array(
                    np.array([0], dtype=np.int64), name=starts_name
                )
                ends_initializer = numpy_helper.from_array(
                    np.array([A.shape[-2]], dtype=np.int64), name=ends_name
                )
                R_slice = helper.make_node(
                    "Slice",
                    inputs=[R_tensor.name, starts_name, ends_name],
                    outputs=[R_slice_name],
                )
                r_t = helper.make_node(
                    "Transpose",
                    inputs=[R_slice_name],
                    outputs=[f"R_transpose_{index}"],
                    perm=[1, 0],
                )
                r_t_A_tensor = helper.make_tensor_value_info(
                    f"r_t_A_{index}",
                    onnx.TensorProto.FLOAT,
                    list(A.shape[:-2] + A.shape[-1:]),
                )
                r_t_A_node = helper.make_node(
                    "MatMul",
                    inputs=[f"R_transpose_{index}", A_tensor.name],
                    outputs=[r_t_A_tensor.name],
                )
                secondary_nodes.extend([R_slice, r_t, r_t_A_node])
                secondary_inputs.append(A_tensor)
                secondary_outputs.append(r_t_A_tensor)
                initializers.extend([starts_initializer, ends_initializer])
                index += 1

    secondary_model_def = helper.make_graph(
        nodes=secondary_nodes,
        name="MatMulGraph",
        inputs=secondary_inputs,
        outputs=secondary_outputs,
        initializer=initializers,
    )

    secondary_model = helper.make_model(
        secondary_model_def,
        producer_name="onnx-matmul-example",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    secondary_model.ir_version = 10
    secondary_model = shape_inference.infer_shapes(secondary_model)
    onnx.checker.check_model(secondary_model)
    return secondary_model
