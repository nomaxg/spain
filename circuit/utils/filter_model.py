import sys
import onnx
from onnx import helper, shape_inference

# input_model_path = "onnx/eval_nano_64_v18.onnx"
input_model_path = "onnx/gpt2-seq-32.onnx"

# nodes_to_keep = [
#     "input.1",
#     "/wte/Gather",
#     "/Add",
#     "/h.0/ln_1/LayerNormalization",
#     "/h.0/attn/c_attn/MatMul",
#     "/h.0/attn/Split",
#     "/h.0/attn/Reshape_1",
#     "/h.0/attn/Reshape",
#     "/h.0/attn/Transpose_3",
#     "/h.0/attn/Transpose_1",
#     "/h.0/attn/Mul",
#     "/h.0/attn/Mul_1",
#     "/h.0/attn/MatMul",
#     "/h.0/attn/Add",
#     "/h.0/attn/Softmax",
# ]

# # ERF conf
# nodes_to_keep = [
#     "/h.0/mlp/c_fc/Add",
#     "/h.0/mlp/gelu/Div",
#     "/h.0/mlp/gelu/Erf",
#     "/h.0/mlp/gelu/Add",
#     "/h.0/mlp/gelu/Mul",
#     "/h.0/mlp/gelu/Mul_1",
# ]

# Matmul conf
nodes_to_keep = [
    "/h.0/mlp/c_fc/MatMul",
]


def find_value_info(graph, name: str):
    """Search input, value_info, and output for a ValueInfo with this name."""
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.name == name:
            return vi
    return None


def make_value_info_from_existing(name: str, vi):
    """Create a fresh ValueInfo copying type/shape from an existing one."""
    ttype = vi.type.tensor_type
    elem_type = ttype.elem_type
    shape = None
    if ttype.HasField("shape"):
        dims = []
        for d in ttype.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append(None)
        shape = dims
    return helper.make_tensor_value_info(name, elem_type, shape)


model = onnx.load(input_model_path)
model = shape_inference.infer_shapes(model)
orig_graph = model.graph

filtered_nodes = [node for node in orig_graph.node if node.name in nodes_to_keep]
if not filtered_nodes:
    print("No nodes found matching the provided names.")
    sys.exit(1)

# Collect all inputs (tensor names) used by the filtered nodes
used_inputs = set()
for node in filtered_nodes:
    for name in node.input:
        if name:  # skip empty strings
            used_inputs.add(name)

# Keep only initializers that are actually used
new_initializers = []
for init in orig_graph.initializer:
    if init.name in used_inputs:
        new_initializers.append(init)

initializer_names = {init.name for init in new_initializers}

# Build graph inputs from original graph inputs that are used and not initializers
new_graph_inputs = []
for input_value in orig_graph.input:
    if input_value.name in used_inputs and input_value.name not in initializer_names:
        new_graph_inputs.append(input_value)

# --- ENSURE WE HAVE AT LEAST ONE GRAPH INPUT ---
if not new_graph_inputs:
    first_node = filtered_nodes[0]

    # Prefer a non-initializer input if possible
    chosen_input_name = None
    for name in first_node.input:
        if name and name not in initializer_names:
            chosen_input_name = name
            break

    # If all inputs are initializers (or empty), fall back to the first input
    # and reclassify it as a graph input (remove from initializers).
    if chosen_input_name is None and first_node.input:
        chosen_input_name = first_node.input[0]
        if chosen_input_name in initializer_names:
            new_initializers = [
                init for init in new_initializers if init.name != chosen_input_name
            ]
            initializer_names.discard(chosen_input_name)

    if chosen_input_name is None:
        print("Could not determine a suitable input name for the first node.")
        sys.exit(1)

    # Try to get proper type/shape from the inferred original graph
    vi = find_value_info(orig_graph, chosen_input_name)
    if vi is not None:
        dummy_input = make_value_info_from_existing(chosen_input_name, vi)
    else:
        # Maybe it's an initializer; if so, use its dims and dtype.
        init_map = {init.name: init for init in orig_graph.initializer}
        if chosen_input_name in init_map:
            init = init_map[chosen_input_name]
            dummy_input = helper.make_tensor_value_info(
                chosen_input_name, init.data_type, list(init.dims)
            )
        else:
            # Last-resort fallback: unknown-shape FLOAT
            dummy_input = helper.make_tensor_value_info(
                chosen_input_name, onnx.TensorProto.FLOAT, None
            )

    new_graph_inputs.append(dummy_input)

# Determine final output (output of the last filtered node)
final_node = filtered_nodes[-1]
if not final_node.output:
    print("The final filtered node does not have any outputs.")
    sys.exit(1)
final_output_name = final_node.output[0]

final_output_value_info = None
for output in orig_graph.output:
    if output.name == final_output_name:
        final_output_value_info = output
        break

if final_output_value_info is None:
    # Try to grab shape/type from value_info if available
    vi = find_value_info(orig_graph, final_output_name)
    if vi is not None:
        final_output_value_info = make_value_info_from_existing(final_output_name, vi)
    else:
        final_output_value_info = helper.make_tensor_value_info(
            final_output_name, onnx.TensorProto.FLOAT, None
        )

new_graph_outputs = [final_output_value_info]

new_graph = helper.make_graph(
    filtered_nodes,  # nodes: only filtered nodes are kept
    "filtered_graph",  # name for the new graph
    new_graph_inputs,  # computed (or synthesized) inputs
    new_graph_outputs,  # final output is the output of the last filtered node
    new_initializers,  # constant initializers needed by the filtered nodes
)

new_model = helper.make_model(
    new_graph,
    producer_name="filtered_model_creator",
    opset_imports=[
        helper.make_opsetid("", 19),
    ],
)
new_model.ir_version = 10
new_model = shape_inference.infer_shapes(new_model)

output_model_path = "onnx/matmul_32x768_768x3072.onnx"
onnx.save(new_model, output_model_path)
print(f"Filtered model saved as {output_model_path}")
