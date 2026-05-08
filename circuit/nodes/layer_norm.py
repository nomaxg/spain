import numpy as np

import ir
from onnx import helper, numpy_helper


def wire_layer_norm(node, graph):
    axis = None
    epsilon = None
    for attr in node.attribute:
        if attr.name == "axis":
            axis = attr.i
        elif attr.name == "epsilon":
            epsilon = attr.f
    if axis is None:
        raise Exception("LayerNormalization node is missing the 'axis' attribute.")
    if epsilon is None:
        raise Exception("LayerNormalization node is missing the 'epsilon' attribute.")
    name = node.name
    x = node.input[0]
    scale = node.input[1]
    bias = node.input[2]
    output_id = node.output[0]
    axes = [axis]
    intermediate_nodes = []
    axes_const_name = f"{name}.axes_const"
    axes_tensor = numpy_helper.from_array(
        np.array(axes, dtype=np.int64), name=axes_const_name
    )
    axes_const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[axes_const_name],
        name=axes_const_name,
        value=axes_tensor,
    )

    # Define intermediate variable names.
    reduce_mean_name = f"{name}.reduce_mean"
    centered_name = f"{name}.centered"
    squared_name = f"{name}.squared"
    variance_name = f"{name}.variance"
    epsilon_name = f"{name}.epsilon"
    var_eps_name = f"{name}.var_eps"
    std_name = f"{name}.std"
    inv_std_name = f"{name}.inv_std"
    normalized_name = f"{name}.normalized"
    scaled_name = f"{name}.scaled"
    identity_name = f"{name}.identity"

    identity_node = helper.make_node(
        "Identity", inputs=[x], outputs=[identity_name], name=identity_name
    )
    intermediate_nodes.append(identity_node)

    # Compute mean
    mean_node = helper.make_node(
        "ReduceMean",
        [x] + [axes_const_name],
        outputs=[reduce_mean_name],
        name=reduce_mean_name,
        keepdims=1,
    )

    intermediate_nodes.append(mean_node)

    # Center inputs
    centered_node = helper.make_node(
        "Sub", inputs=[x, reduce_mean_name], outputs=[centered_name], name=centered_name
    )

    # Compute x^2 for variance
    squared_node = helper.make_node(
        "Mul",
        inputs=[centered_name, centered_name],
        outputs=[squared_name],
        name=squared_name,
    )

    intermediate_nodes.append(squared_node)

    # Reduce mean to calculate variance
    variance_node = helper.make_node(
        "ReduceMean",
        [squared_name] + [axes_const_name],
        outputs=[variance_name],
        name=variance_name,
        keepdims=1,
    )

    epsilon_tensor = numpy_helper.from_array(
        np.array(epsilon, dtype=np.float32), name=f"{epsilon_name}_tensor"
    )
    epsilon_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[epsilon_name],
        name=epsilon_name,
        value=epsilon_tensor,
    )

    # Epsilon constant for stability
    var_eps_node = helper.make_node(
        "Add",
        inputs=[variance_name, epsilon_name],
        outputs=[var_eps_name],
        name=var_eps_name,
    )

    std_node = helper.make_node(
        "Sqrt", inputs=[var_eps_name], outputs=[std_name], name=std_name
    )

    intermediate_nodes.append(std_node)

    inv_std_node = helper.make_node(
        "Reciprocal", inputs=[std_name], outputs=[inv_std_name], name=inv_std_name
    )

    intermediate_nodes.append(inv_std_node)

    normalized_node = helper.make_node(
        "Mul",
        inputs=[centered_name, inv_std_name],
        outputs=[normalized_name],
        name=normalized_name,
    )

    intermediate_nodes.append(normalized_node)

    scaled_node = helper.make_node(
        "Mul", inputs=[normalized_name, scale], outputs=[scaled_name], name=scaled_name
    )

    output_node = helper.make_node(
        "Add", inputs=[scaled_name, bias], outputs=[output_id], name=name
    )

    new_nodes = [
        identity_node,
        axes_const_node,
        mean_node,
        centered_node,
        squared_node,
        variance_node,
        epsilon_node,
        var_eps_node,
        std_node,
        inv_std_node,
        normalized_node,
        scaled_node,
    ]

    graph.node.remove(node)
    graph.node.append(output_node)
    graph.node.extend(new_nodes)

    return (output_node, intermediate_nodes)


def constrain_layer_norm(r1cs, rnode):
    inputs = rnode.node.input
    x = r1cs.nodes.get(inputs[0])
    axis = None
    epsilon = None
    scale = r1cs.nodes.get(inputs[1])
    bias = r1cs.nodes.get(inputs[2])
    for attr in rnode.node.attribute:
        if attr.name == "axis":
            axis = attr.i
        elif attr.name == "epsilon":
            epsilon = attr.f
    if axis is None:
        raise Exception("LayerNormalization node is missing the 'axis' attribute.")
    if epsilon is None:
        raise Exception("LayerNormalization node is missing the 'epsilon' attribute.")

    x_output_symb = r1cs.get_output(x)
    x_output = r1cs.allocate_tensor(x_output_symb.shape)
    for idx in np.ndindex(x_output.shape):
        rnode.append_labeled_constraint(
            x_output_symb[idx],
            ir.LinearCombo.one(),
            x_output[idx],
            f"{rnode.node.name}_x_output_{idx}",
        )
    axis = axis % x_output.ndim
    reduce_mean_shape = tuple(1 if i == axis else d for i, d in enumerate(x.shape))
    reduce_sum_x = np.empty(reduce_mean_shape, dtype=object)
    np.sum(x_output, axis=axis, keepdims=True, out=reduce_sum_x)
    reduce_mean_x_condensed = r1cs.allocate_tensor(reduce_sum_x.shape)
    for idx in np.ndindex(reduce_sum_x.shape):
        rnode.append_labeled_constraint(
            reduce_mean_x_condensed[idx],
            ir.LinearCombo.from_const(float(x_output.shape[axis])),
            reduce_sum_x[idx],
            f"{rnode.node.name}_reduce_mean_{idx}",
        )
    centered = np.empty(x.shape, dtype=object)
    np.subtract(x_output, reduce_mean_x_condensed, out=centered, casting="unsafe")

    centered_squared = r1cs.allocate_tensor(centered.shape)

    for idx in np.ndindex(centered.shape):
        rnode.append_labeled_constraint(
            centered[idx],
            centered[idx],
            centered_squared[idx],
            f"{rnode.node.name}_centered_squared_{idx}",
        )

    reduce_mean_variance = np.sum(
        centered_squared, axis=axis, keepdims=True
    ) + ir.LinearCombo.from_const(epsilon * centered_squared.shape[axis])

    reduce_mean_var_sqrt = r1cs.allocate_tensor(reduce_mean_variance.shape)

    for idx in np.ndindex(reduce_mean_variance.shape):
        rnode.append_labeled_constraint(
            reduce_mean_var_sqrt[idx],
            reduce_mean_var_sqrt[idx] * centered_squared.shape[axis],
            reduce_mean_variance[idx],
            f"{rnode.node.name}_reduce_mean_var_sqrt_{idx}",
        )

    inv_std = r1cs.allocate_tensor(reduce_mean_var_sqrt.shape)

    for idx in np.ndindex(reduce_mean_var_sqrt.shape):
        rnode.append_labeled_constraint(
            reduce_mean_var_sqrt[idx],
            inv_std[idx],
            ir.LinearCombo.one(),
            f"{rnode.node.name}_inv_std_{idx}",
        )

    broadcasted_inv_std = np.broadcast_to(inv_std, centered.shape)
    normalized_node = r1cs.allocate_tensor(centered.shape)
    scale = np.broadcast_to(r1cs.get_output(scale), centered.shape)

    for idx in np.ndindex(centered.shape):
        rnode.append_labeled_constraint(
            centered[idx],
            broadcasted_inv_std[idx],
            normalized_node[idx],
            f"{rnode.node.name}_normalized_{idx}",
        )
        normalized_node[idx] *= scale[idx]

    scaled_node = normalized_node + r1cs.get_output(bias)

    rnode.output = scaled_node
