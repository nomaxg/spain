import numpy as np
from onnx import helper, numpy_helper
import ir
from eval import INFERENCE_TYPE

# POC of Maxpool, may be missing constraints.

def get_value_info_shape(graph, value_name):
    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        if value_info.name == value_name:
            dims = value_info.type.tensor_type.shape.dim
            return [d.dim_value for d in dims]
    raise RuntimeError(f"could not find shape for value {value_name!r}")


def wire_maxpool(node, graph):
    name = node.name
    x_name = node.input[0]
    y_name = node.output[0]

    # Collect attributes
    kernel_shape = None
    strides = None
    pads_attr = None

    for attr in node.attribute:
        if attr.name == "kernel_shape":
            kernel_shape = list(attr.ints)
        elif attr.name == "strides":
            strides = list(attr.ints)
        elif attr.name == "pads":
            pads_attr = list(attr.ints)

    if kernel_shape is None or len(kernel_shape) != 2:
        raise Exception("maxpool node missing or invalid 'kernel_shape'")
    kernel_h, kernel_w = kernel_shape

    if strides is None:
        strides = kernel_shape  # non-overlapping windows
    if len(strides) != 2:
        raise Exception("wire_maxpool expects 2d strides")
    stride_h, stride_w = strides

    if pads_attr is None:
        pads_attr = [0, 0, 0, 0]
    if pads_attr != [0, 0, 0, 0]:
        raise Exception("wire_maxpool assumes zero padding only (pads must be [0, 0, 0, 0])")

    if stride_h != kernel_h or stride_w != kernel_w:
        raise Exception("wire_maxpool assumes stride == kernel_shape (non-overlapping windows)")

    # get innput shape
    x_shape = get_value_info_shape(graph, x_name)
    if len(x_shape) != 4:
        raise Exception("wire_maxpool only supports 2d maxpool with nchw input")
    n_batch, n_channels, in_h, in_w = x_shape

    out_h = (in_h - kernel_h) // stride_h + 1
    out_w = (in_w - kernel_w) // stride_w + 1
    used_h = out_h * stride_h
    used_w = out_w * stride_w
    new_nodes = []
    witness_nodes = []

    # 1. Slice: X -> padded_input (cropped or no-op)
    starts_name = f"{name}.slice_starts"
    starts_tensor = numpy_helper.from_array(
        np.array([0, 0, 0, 0], dtype=np.int64), name=starts_name + "_tensor"
    )
    starts_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[starts_name],
        name=starts_name,
        value=starts_tensor,
    )
    new_nodes.append(starts_node)

    ends_name = f"{name}.slice_ends"
    ends_tensor = numpy_helper.from_array(
        np.array([n_batch, n_channels, used_h, used_w], dtype=np.int64),
        name=ends_name + "_tensor",
    )
    ends_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[ends_name],
        name=ends_name,
        value=ends_tensor,
    )
    new_nodes.append(ends_node)

    axes_name = f"{name}.slice_axes"
    axes_tensor = numpy_helper.from_array(
        np.array([0, 1, 2, 3], dtype=np.int64), name=axes_name + "_tensor"
    )
    axes_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[axes_name],
        name=axes_name,
        value=axes_tensor,
    )
    new_nodes.append(axes_node)

    padded_input_name = f"{name}.padded_input"
    slice_node = helper.make_node(
        "Slice",
        inputs=[x_name, starts_name, ends_name, axes_name],
        outputs=[padded_input_name],
        name=f"{name}.Slice",
    )
    new_nodes.append(slice_node)
    # slice_node's output is our first witness (padded_input)
    witness_nodes.append(slice_node)

    # 2. New MaxPool node consuming padded_input, producing same Y
    maxpool_new = helper.make_node(
        "MaxPool",
        inputs=[padded_input_name],
        outputs=[y_name],
        name=f"{name}.MaxPool_cropped",
        kernel_shape=kernel_shape,
        strides=strides,
        pads=[0, 0, 0, 0],
    )
    new_nodes.append(maxpool_new)

    x_reshaped_shape = np.array(
        [n_batch, n_channels, out_h, kernel_h, out_w, kernel_w], dtype=np.int64
    )
    x_reshaped_shape_name = f"{name}.x_reshaped_shape"
    x_reshaped_shape_tensor = numpy_helper.from_array(
        x_reshaped_shape, name=x_reshaped_shape_name + "_tensor"
    )
    x_reshaped_shape_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[x_reshaped_shape_name],
        name=x_reshaped_shape_name,
        value=x_reshaped_shape_tensor,
    )
    new_nodes.append(x_reshaped_shape_node)

    y_reshaped_shape = np.array(
        [n_batch, n_channels, out_h, 1, out_w, 1], dtype=np.int64
    )
    y_reshaped_shape_name = f"{name}.y_reshaped_shape"
    y_reshaped_shape_tensor = numpy_helper.from_array(
        y_reshaped_shape, name=y_reshaped_shape_name + "_tensor"
    )
    y_reshaped_shape_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[y_reshaped_shape_name],
        name=y_reshaped_shape_name,
        value=y_reshaped_shape_tensor,
    )
    new_nodes.append(y_reshaped_shape_node)

    flat_shape = np.array([n_batch, n_channels, used_h, used_w], dtype=np.int64)
    flat_shape_name = f"{name}.flat_shape"
    flat_shape_tensor = numpy_helper.from_array(
        flat_shape, name=flat_shape_name + "_tensor"
    )
    flat_shape_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[flat_shape_name],
        name=flat_shape_name,
        value=flat_shape_tensor,
    )
    new_nodes.append(flat_shape_node)

    x_reshaped_name = f"{name}.x_reshaped"
    x_reshape_node = helper.make_node(
        "Reshape",
        inputs=[padded_input_name, x_reshaped_shape_name],
        outputs=[x_reshaped_name],
        name=x_reshaped_name,
    )
    new_nodes.append(x_reshape_node)

    y_reshaped_name = f"{name}.y_reshaped"
    y_reshape_node = helper.make_node(
        "Reshape",
        inputs=[y_name, y_reshaped_shape_name],
        outputs=[y_reshaped_name],
        name=y_reshaped_name,
    )
    new_nodes.append(y_reshape_node)

    # max_minus_x = y_reshaped - x_reshaped
    max_minus_x_reshaped_name = f"{name}.max_minus_x_reshaped"
    max_minus_x_reshaped_node = helper.make_node(
        "Sub",
        inputs=[y_reshaped_name, x_reshaped_name],
        outputs=[max_minus_x_reshaped_name],
        name=max_minus_x_reshaped_name,
    )
    new_nodes.append(max_minus_x_reshaped_node)

    # sqrt_max_minus_x
    sqrt_reshaped_name = f"{name}.sqrt_max_minus_x_reshaped"
    sqrt_reshaped_node = helper.make_node(
        "Sqrt",
        inputs=[max_minus_x_reshaped_name],
        outputs=[sqrt_reshaped_name],
        name=sqrt_reshaped_name,
    )
    new_nodes.append(sqrt_reshaped_node)

    # reshape both back to padded_input shape
    max_minus_x_name = f"{name}.max_minus_x"
    max_minus_x_reshape_back_node = helper.make_node(
        "Reshape",
        inputs=[max_minus_x_reshaped_name, flat_shape_name],
        outputs=[max_minus_x_name],
        name=max_minus_x_name,
    )
    new_nodes.append(max_minus_x_reshape_back_node)
    witness_nodes.append(max_minus_x_reshape_back_node)

    sqrt_name = f"{name}.sqrt_max_minus_x"
    sqrt_reshape_back_node = helper.make_node(
        "Reshape",
        inputs=[sqrt_reshaped_name, flat_shape_name],
        outputs=[sqrt_name],
        name=sqrt_name,
    )
    new_nodes.append(sqrt_reshape_back_node)
    witness_nodes.append(sqrt_reshape_back_node)

    # is_max mask: Equal(padded_input, broadcasted max)
    is_max_reshaped_bool_name = f"{name}.is_max_reshaped_bool"
    is_max_reshaped_bool_node = helper.make_node(
        "Equal",
        inputs=[x_reshaped_name, y_reshaped_name],
        outputs=[is_max_reshaped_bool_name],
        name=is_max_reshaped_bool_name,
    )
    new_nodes.append(is_max_reshaped_bool_node)

    is_max_bool_name = f"{name}.is_max_bool"
    is_max_bool_reshape_node = helper.make_node(
        "Reshape",
        inputs=[is_max_reshaped_bool_name, flat_shape_name],
        outputs=[is_max_bool_name],
        name=is_max_bool_name,
    )
    new_nodes.append(is_max_bool_reshape_node)

    is_max_name = f"{name}.is_max"
    is_max_cast_node = helper.make_node(
        "Cast",
        inputs=[is_max_bool_name],
        outputs=[is_max_name],
        name=is_max_name,
        to=INFERENCE_TYPE,
    )
    new_nodes.append(is_max_cast_node)
    witness_nodes.append(is_max_cast_node)

    witness_nodes.append(maxpool_new)

    # rewrite graph
    graph.node.remove(node)
    graph.node.extend(new_nodes)
    return (maxpool_new, witness_nodes)

# Constrains 2d maxpool
def constrain_maxpool(r1cs, rnode):
    inputs = rnode.node.input
    x_meta = r1cs.nodes.get(inputs[0])

    if not x_meta.is_var:
        raise Exception("maxpool input must be a variable; constant not implemented")

    # original MaxPool input shape
    x_shape = x_meta.shape  # (n, c, in_h, in_w)
    if len(x_shape) != 4:
        raise ValueError("constrain_maxpool only supports 2d maxpool with nchw layout")

    n_batch, n_channels, in_h, in_w = x_shape

    # fetch attributes
    kernel_shape = None
    strides = None
    pads = None

    for attr in rnode.node.attribute:
        if attr.name == "kernel_shape":
            kernel_shape = list(attr.ints)
        elif attr.name == "strides":
            strides = list(attr.ints)
        elif attr.name == "pads":
            pads = list(attr.ints)

    if kernel_shape is None or len(kernel_shape) != 2:
        raise ValueError("maxpool node missing 'kernel_shape' attribute")
    kernel_h, kernel_w = kernel_shape

    if strides is None:
        strides = kernel_shape
    if len(strides) != 2:
        raise ValueError("constrain_maxpool expects strides of length 2")

    stride_h, stride_w = strides

    if pads is None:
        pads = [0, 0, 0, 0]
    if pads != [0, 0, 0, 0]:
        raise ValueError("constrain_maxpool assumes pads == [0, 0, 0, 0]")

    if stride_h != kernel_h or stride_w != kernel_w:
        raise ValueError("constrain_maxpool assumes stride == kernel_shape (non-overlapping windows)")

    # derive output shape  / used indices
    out_h = in_h // kernel_h
    out_w = in_w // kernel_w
    out_shape = (n_batch, n_channels, out_h, out_w)
    used_h = out_h * kernel_h
    used_w = out_w * kernel_w
    padded_shape = (n_batch, n_channels, used_h, used_w)

    # input
    x_tensor = x_meta.output

    # padded_input (cropped X)
    padded_input = r1cs.allocate_tensor(padded_shape, virtual=True)
    # max minus input, should have sqrt
    max_min_x = r1cs.allocate_tensor(padded_shape, virtual=True)
    # sqrts
    sqrts = r1cs.allocate_tensor(padded_shape, virtual=True)
    # is_max 
    is_max = r1cs.allocate_tensor(padded_shape, virtual=True)
    # max results
    max_pools = r1cs.allocate_tensor(out_shape, rnode.var_type, virtual=True)

    constraints = 0

    for n in range(n_batch):
        for c in range(n_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    # window in padded_input (no padding, stride == kernel)
                    h_start = oh * kernel_h
                    w_start = ow * kernel_w
                    h_end = h_start + kernel_h
                    w_end = w_start + kernel_w

                    max_val = max_pools[(n, c, oh, ow)]

                    sum_indicators = ir.LinearCombo.zero()

                    for ih in range(h_start, h_end):
                        for iw in range(w_start, w_end):
                            # indices into padded_input
                            p_idx = (n, c, ih, iw)

                            # corresponding coords in original X
                            x_idx = (n, c, ih, iw)  # since we just crop top-left block

                            x_val = x_tensor[x_idx]
                            padded_val = padded_input[p_idx]
                            d_val = max_min_x[p_idx]
                            sqrt_val = sqrts[p_idx]
                            indicator = is_max[p_idx]

                            # padded_input must match X on the used region
                            rnode.append_labeled_constraint(
                                padded_val,
                                ir.LinearCombo.one(),
                                x_val,
                                "maxpool: padded_input = cropped X",
                            )

                            #  d = max - padded_input
                            rnode.append_labeled_constraint(
                                max_val - padded_val,
                                ir.LinearCombo.one(),
                                d_val,
                                "maxpool: d = max - padded_input",
                            )

                            # sqrt^2 = d
                            rnode.append_labeled_constraint(
                                sqrt_val,
                                sqrt_val,
                                d_val,
                                "maxpool: sqrt^2 = max - padded_input",
                            )

                            # (max - padded_input) * indicator = 0
                            rnode.append_labeled_constraint(
                                max_val - padded_val,
                                indicator,
                                ir.LinearCombo.from_const(0),
                                "maxpool: (max - padded_input) * is_max = 0",
                            )

                            # indicator is boolean: ind * (ind - 1) = 0
                            rnode.append_labeled_constraint(
                                indicator,
                                indicator - ir.LinearCombo.from_const(1),
                                ir.LinearCombo.from_const(0),
                                "maxpool: is_max boolean",
                            )

                            sum_indicators += indicator
                            constraints += 5
                            if constraints % 10000 == 0:
                                r1cs.process_constrained_node(rnode)

    rnode.output = max_pools.materialize()

