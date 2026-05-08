import numpy as np

import ir
from onnx import numpy_helper


def constrain_unhandled(r1cs, rnode):
    rnode.output = r1cs.allocate_tensor(rnode.shape, ir.VarType.PUBLIC)


def constrain_flatten(r1cs, rnode):
    inputs = rnode.node.input
    axis = 1
    for attr in rnode.node.attribute:
        if attr.name == "axis":
            axis = attr.i
    _, x = r1cs.get_node(inputs[0])
    dim0 = int(np.prod(x.shape[:axis])) if axis > 0 else 1
    dim1 = int(np.prod(x.shape[axis:]))
    flat = x.flatten(order="C")
    rnode.output = flat.reshape(dim0, dim1)


def constrain_transpose(r1cs, rnode):
    perm = None
    for attr in rnode.node.attribute:
        if attr.name == "perm":
            perm = attr.ints
    inputs = rnode.node.input
    _, data = r1cs.get_node(inputs[0])
    rnode.output = np.transpose(data, axes=perm)


def constrain_reshape(r1cs, rnode):
    inputs = rnode.node.input
    _, linmap_output = r1cs.get_node(inputs[0])
    rnode.output = np.reshape(linmap_output, rnode.shape)


def constrain_add(r1cs, rnode):
    inputs = rnode.node.input
    left = r1cs.nodes.get(inputs[0])
    right = r1cs.nodes.get(inputs[1])
    left_output = r1cs.get_output(left)
    right_output = r1cs.get_output(right)
    out_shape = np.broadcast(left_output, right_output).shape
    rnode.output = np.empty(out_shape, dtype=object)
    if left.is_var:
        np.add(left_output, right_output, out=rnode.output)
    else:
        np.add(right_output, left_output, out=rnode.output)


def constrain_generic(r1cs, rnode):
    expected = r1cs.allocate_tensor(rnode.shape, rnode.var_type)
    output = rnode.output
    for idx in np.ndindex(rnode.shape):
        rnode.append_labeled_constraint(
            output[idx],
            ir.LinearCombo.one(),
            expected[idx],
            f"{rnode.node.name}_Final_{idx}",
        )


def constrain_sub(r1cs, rnode):
    inputs = rnode.node.input
    left = r1cs.nodes.get(inputs[0])
    right = r1cs.nodes.get(inputs[1])
    rnode.output = left.output - right.output


def constrain_split(r1cs, rnode):
    inputs = rnode.node.input
    axis = None

    for attr in rnode.node.attribute:
        if attr.name == "axis":
            axis = attr.i

    if axis is None:
        raise Exception("Split node is missing the 'axis' attribute.")

    num_outputs = len(rnode.node.output)

    data = r1cs.get_output(r1cs.nodes.get(inputs[0]))
    splits = np.split(data, num_outputs, axis=axis)

    if len(rnode.node.output) != num_outputs:
        raise Exception("The number of output names does not match num_outputs.")

    output_map = {}
    for name, split in zip(rnode.node.output, splits):
        output_map[name] = split

    rnode.output = output_map


def constrain_mul(r1cs, rnode):
    inputs = rnode.node.input
    left = r1cs.nodes.get(inputs[0])
    right = r1cs.nodes.get(inputs[1])
    left_output = r1cs.get_output(left)
    right_output = r1cs.get_output(right)
    left_broadcasted = np.broadcast_to(left_output, rnode.shape)
    right_broadcasted = np.broadcast_to(right_output, rnode.shape)
    output = np.empty(rnode.shape, dtype=object)
    for idx in np.ndindex(rnode.shape):
        if not left.is_var or not right.is_var:
            output[idx] = right_broadcasted[idx] * left_broadcasted[idx]
        else:
            output[idx] = ir.LinearCombo.from_var(r1cs.allocate_var(rnode.var_type))
            rnode.append_labeled_constraint(
                left_broadcasted[idx],
                right_broadcasted[idx],
                output[idx],
                f"{rnode.node.name}_mul_{idx}",
            )
    rnode.output = output


def constrain_div(r1cs, rnode):
    inputs = rnode.node.input
    left = r1cs.nodes.get(inputs[0])
    right = r1cs.nodes.get(inputs[1])
    left_output = r1cs.get_output(left)
    right_output = r1cs.get_output(right)
    left_broadcasted = np.broadcast_to(left_output, rnode.shape)
    right_broadcasted = np.broadcast_to(right_output, rnode.shape)
    output = np.empty(rnode.shape, dtype=object)
    for idx in np.ndindex(rnode.shape):
        output[idx] = ir.LinearCombo.from_var(r1cs.allocate_var(rnode.var_type))
        rnode.append_labeled_constraint(
            ir.LinearCombo.from_const(right_broadcasted[idx]),
            output[idx],
            left_broadcasted[idx],
            f"{rnode.node.name}_div_{idx}",
        )
    rnode.output = output


def constrain_slice(r1cs, rnode):
    inputs = rnode.node.input
    data = r1cs.nodes.get(inputs[0]).output
    start = int(r1cs.nodes.get(inputs[1]).output[0].get_const())
    end = int(r1cs.nodes.get(inputs[2]).output[0].get_const())
    axis = int(r1cs.nodes.get(inputs[3]).output[0].get_const())
    step = int(r1cs.nodes.get(inputs[4]).output[0].get_const())
    r1cs.output = np.take(data, indices=np.arange(start, end, step), axis=axis)


def constrain_initializer(_, rnode):
    const_data = numpy_helper.to_array(rnode.node)
    rnode.output = const_data


def constrain_constant(_, rnode):
    const_data = numpy_helper.to_array(rnode.node.attribute[0].t)
    rnode.output = np.empty(const_data.shape, dtype=object)
    for idx in np.ndindex(const_data.shape):
        rnode.output[idx] = ir.LinearCombo.from_const(float(const_data[idx]))


def constrain_input(r1cs, rnode):
    rnode.output = r1cs.allocate_tensor(rnode.shape, ir.VarType.PUBLIC)


def constrain_identity(r1cs, rnode):
    inputs = rnode.node.input
    rnode.output = r1cs.nodes.get(inputs[0]).output
