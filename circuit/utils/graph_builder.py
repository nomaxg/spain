import onnx
from onnx import helper, TensorProto

x_dim = 2048
y_dim = 6072

# I/O
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, x_dim, y_dim])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, x_dim, y_dim])

# Intermediate tensor
Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, x_dim, y_dim])

# Scalar bias (broadcasts over X)
bias_init = helper.make_tensor(
    name="bias",
    data_type=TensorProto.FLOAT,
    dims=[],  # scalar
    vals=[0.1],
)

# Z = X + bias
add_node = helper.make_node(
    "Add",
    inputs=["X", "bias"],
    outputs=["Z"],
    name="AddBias",
)

# Y = Erf(Z)
erf_node = helper.make_node(
    "Erf",
    inputs=["Z"],
    outputs=["Y"],
    name="ErfOp",
)

graph = helper.make_graph(
    nodes=[add_node, erf_node],
    name="ErfGraphWithBias",
    inputs=[X],
    outputs=[Y],
    value_info=[Z],
    initializer=[bias_init],
)

model = helper.make_model(
    graph,
    producer_name="onnx-erf-example",
    ir_version=10,  # keep IR version compatible
    opset_imports=[helper.make_opsetid("", 13)],  # Erf is supported in opset ≥9
)

onnx.checker.check_model(model)

output_path = "./onnx/big_erf.onnx"
onnx.save(model, output_path)
print(f"ONNX model {output_path} has been created successfully.")
