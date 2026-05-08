import copy
import math

import numpy as np

import nodes
import onnx
import secondary
from eval import INFERENCE_TYPE, PRECISION_OPS
from ir import VarType
from onnx import numpy_helper
from r1cs import R1CS, R1CSNode

reshape_ops = ["Split", "Transpose", "Reshape", "Flatten"]
unhandled_ops = ["Gather"]

PUBLIC_LABEL = VarType.PUBLIC.to_str()
PRIMARY_LABEL = VarType.PRIMARY.to_str()
SECONDARY_LABEL = VarType.SECONDARY.to_str()


def collect_input_dims(tensor):
    return tuple(dim.dim_value for dim in tensor.type.tensor_type.shape.dim)


def collect_initialier_dims(initializer):
    return tuple(dims for dims in initializer.dims)


def append_graph_output(graph, node, output_type=PRIMARY_LABEL):
    # check if node already exists in graph outputs
    for output in graph.output:
        if output.name == node:
            # update the doc_string (label) if it already exists
            # small subtleties: if the output exists and is secondary, we add a new labels
            # else, we are dealing with a primary variable turned secondary, so we overwrite the lable
            if output.doc_string.count("public") > 0:
                output.doc_string += output_type
            else:
                output.doc_string = output_type
            return

    # if not found, append a new output
    graph.output.append(
        onnx.helper.make_tensor_value_info(
            node, INFERENCE_TYPE, None, doc_string=output_type
        )
    )


def convert_model_to_precision(model, precision=INFERENCE_TYPE):
    np_dtype = np.float32 if precision == onnx.TensorProto.FLOAT else np.float64
    # convert graph inputs
    for inp in model.graph.input:
        et = inp.type.tensor_type.elem_type
        if et in PRECISION_OPS:
            inp.type.tensor_type.elem_type = precision

    # convert intermediate value_info
    for vi in model.graph.value_info:
        et = vi.type.tensor_type.elem_type
        if et in PRECISION_OPS:
            vi.type.tensor_type.elem_type = precision

    # convert graph outputs
    for out in model.graph.output:
        et = out.type.tensor_type.elem_type
        if et in PRECISION_OPS:
            out.type.tensor_type.elem_type = precision

    # convert initializers (constants)
    new_inits = []
    for init in list(model.graph.initializer):
        if init.data_type in PRECISION_OPS:
            arr = numpy_helper.to_array(init).astype(np_dtype)
            new_init = numpy_helper.from_array(arr, init.name)
        else:
            new_init = init
        model.graph.initializer.remove(init)
        new_inits.append(new_init)
    model.graph.initializer.extend(new_inits)

    # convert Constant node attributes if FLOAT/DOUBLE
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type in PRECISION_OPS:
                    arr = numpy_helper.to_array(attr.t).astype(np_dtype)
                    attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
    return model


def get_r1cs(model, name, full_variable_matmul):
    # save refercence to original models
    original_model = copy.deepcopy(model)
    r1cs = R1CS(name, None, None, None)
    r1cs.unhandled_ops = unhandled_ops
    r1cs.full_variable_matmul = full_variable_matmul
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph
    shapes = {}

    for output in graph.output:
        shapes[output.name] = collect_input_dims(output)

    # Wipe outputs
    while len(graph.output) > 0:
        graph.output.pop()

    for value_info in graph.value_info:
        shapes[value_info.name] = collect_input_dims(value_info)

    # Create metadata for all initializers (constants)
    for initializer in graph.initializer:
        meta = R1CSNode(r1cs)
        meta.is_var = False
        meta.in_wit = False
        meta.node = initializer
        meta.shape = collect_initialier_dims(initializer)
        meta.op = "Initializer"
        r1cs.nodes[initializer.name] = meta

    # Create metadata for all inputs
    for input_tensor in graph.input:
        meta = R1CSNode(r1cs)
        meta.is_var = True
        meta.node = input_tensor
        meta.in_wit = True
        meta.op = "Input"
        meta.shape = collect_input_dims(input_tensor)
        r1cs.nodes[input_tensor.name] = meta
        r1cs.num_pretermined_public_variables += int(np.prod(meta.shape))

    for node in list(graph.node):
        inputs = node.input
        outputs = node.output
        # Populate successors of each input with the current node
        # Adding duplicates for nodes with multiple outputs
        for inp in inputs:
            pred = r1cs.nodes.get(inp)
            for _ in outputs:
                pred.successors.append(node.name)
        op = node.op_type
        meta = R1CSNode(r1cs)
        meta.node = node
        meta.op = op
        if len(outputs) > 1:
            for output in outputs:
                if shapes[outputs[0]] != shapes[output]:
                    raise Exception(
                        f"Node {node.name} has outputs with differing shapes: {shapes[outputs[0]]} vs {shapes[output]}"
                    )
        meta.shape = shapes[outputs[0]]
        intermediate_nodes = []
        if op in unhandled_ops:
            meta.is_var = True
            meta.in_wit = True
        elif op == "Sin":
            last_node, intermediate_nodes = nodes.wire_sin(node, graph, 0)
            meta.is_var = True
            meta.in_wit = False
        elif op == "Cos":
            last_node, intermediate_nodes = nodes.wire_sin(node, graph, math.pi / 2)
            meta.is_var = True
            meta.in_wit = False
        elif op == "Relu":
            last_node, intermediate_nodes = nodes.wire_relu(node, graph)
            meta.is_var = True
            meta.in_wit = True
        elif op == "MaxPool":
            last_node, intermediate_nodes = nodes.wire_maxpool(node, graph)
            meta.is_var = True
            meta.in_wit = True
        elif op == "Add":
            left = r1cs.nodes.get(inputs[0])
            right = r1cs.nodes.get(inputs[1])
            meta.is_var = left.is_var or right.is_var
            meta.in_wit = False
        elif op == "Sub":
            left = r1cs.nodes.get(inputs[0])
            right = r1cs.nodes.get(inputs[1])
            meta.is_var = left.is_var or right.is_var
            meta.in_wit = False
        elif op == "Identity":
            inp = r1cs.nodes.get(inputs[0])
            meta.is_var = inp.is_var
            meta.in_wit = False
        elif op == "Erf":
            last_node, intermediate_nodes = nodes.wire_erf(node, graph)
            meta.is_var = True
            meta.in_wit = False
        elif op == "Mul":
            left = r1cs.nodes.get(inputs[0])
            right = r1cs.nodes.get(inputs[1])
            meta.is_var = left.is_var or right.is_var
            meta.in_wit = left.is_var and right.is_var
        elif op == "Div":
            left = r1cs.nodes.get(inputs[0])
            right = r1cs.nodes.get(inputs[1])
            if right.is_var:
                raise ValueError("Div constraints not supported for variable divisor")
            meta.is_var = left.is_var
            meta.in_wit = True
        elif op == "LayerNormalization":
            inp = r1cs.nodes.get(inputs[0])
            meta.is_var = True
            meta.in_wit = False
            last_node, intermediate_nodes = nodes.wire_layer_norm(node, graph)
        elif op == "Softmax":
            inp = r1cs.nodes.get(inputs[0])
            meta.is_var = True
            meta.in_wit = True
            last_node, intermediate_nodes = nodes.wire_softmax(node, graph)
        elif op == "Constant":
            pass
        elif op == "MatMul":
            meta.is_var = True
            meta.in_wit = True
            A = r1cs.nodes.get(inputs[0])
            B = r1cs.nodes.get(inputs[1])
            r_length = 0
            a_output_label = ""
            b_output_label = ""
            if B.is_var and not r1cs.full_variable_matmul:
                r_length = B.shape[-1]
                b_output_label += SECONDARY_LABEL
            elif B.is_var and r1cs.full_variable_matmul:
                intermediate_nodes = nodes.wire_matmul(r1cs, node, graph)
            elif not B.is_var:
                r_length = A.shape[-2]
                a_output_label += SECONDARY_LABEL
            r1cs.update_num_random_variables(r_length)
            if not A.in_wit:
                a_output_label += "_" + PRIMARY_LABEL
            if not B.in_wit and B.is_var:
                b_output_label += "_" + PRIMARY_LABEL
            if a_output_label != "":
                if hasattr(A.node, "output"):
                    append_graph_output(graph, A.node.output[0], a_output_label)
                else:
                    # Edge case, A is already an input or output , so we update the label
                    for output in list(graph.input) + list(graph.output):
                        if output.name == A.node.name:
                            output.doc_string += a_output_label
            if b_output_label != "":
                if hasattr(B.node, "output"):
                    append_graph_output(graph, B.node.output[0], b_output_label)
                else:
                    # Edge case, B is already an input or output, so we update the label
                    for output in list(graph.output) + list(graph.input):
                        if output.name == B.node.name:
                            output.doc_string += b_output_label
        elif op in reshape_ops:
            meta.is_var = True
            meta.in_wit = False
        else:
            raise Exception(f"Unhandled node {node.op_type}")
        for output in outputs:
            r1cs.nodes[output] = meta
        if len(outputs) > 1:
            print(f"Warning: node {node.name} has multiple outputs")
        for intermediate_node in intermediate_nodes:
            append_graph_output(graph, intermediate_node.output[0])
        if meta.in_wit:
            for output in meta.node.output:
                label = PRIMARY_LABEL
                if meta.op in unhandled_ops:
                    label = PUBLIC_LABEL
                    r1cs.num_pretermined_public_variables += int(np.prod(meta.shape))
                append_graph_output(graph, output, label)

    last_node = list(r1cs.nodes.values())[-1]
    last_node.var_type = VarType.PUBLIC
    r1cs.num_pretermined_public_variables += int(np.prod(last_node.shape))
    append_graph_output(graph, last_node.node.output[0], PUBLIC_LABEL)

    model = convert_model_to_precision(model)
    secondary_model = secondary.build_secondary_model(r1cs)
    secondary_model = convert_model_to_precision(secondary_model)

    r1cs.original_model = original_model
    r1cs.primary_model = model
    r1cs.secondary_model = secondary_model
    return r1cs
