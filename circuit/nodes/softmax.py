import numpy as np
from onnx import helper, numpy_helper
import ir
from eval import INFERENCE_TYPE

# Rational approximation coefficients
P = [1.000073, 0.33763049, 0.0392927, 0.001555238]
Q = [1.000000, -0.665098450, 0.19090786, -0.047189207]
# Pade coefficients centered around 0
# P = [1.0, 0.5, 0.0999999, 0.008333333]
# Q = [1.0, -0.5, 0.1, -0.00833333]


def wire_softmax(node, graph):
    axis = None
    for attr in node.attribute:
        if attr.name == "axis":
            axis = attr.i
    if axis is None:
        raise Exception("Softmax node is missing the 'axis' attribute.")

    name = node.name
    x = node.input[0]
    output_id = node.output[0]
    softmax_degree = len(P) - 1
    new_nodes = []

    # Define intermediate variable names.
    reduce_max_name = f"{name}.reduce_max"
    max_mask_name = f"{name}.max_mask"
    diff_name_1 = f"{name}.input_minux_max"
    diff_name_2 = f"{name}.max_minus_input"
    sqrt_name = f"{name}.sqrt_max_minus_inputs"
    is_max_name = f"{name}.is_max"
    ones_name = f"{name}.ones"
    axes_const_name = f"{name}.axes_const"

    axes = [axis]
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
    new_nodes.append(axes_const_node)
    witness_nodes = []

    # Compute maximum values along the axis.
    reduce_max_node = helper.make_node(
        "ReduceMax",
        inputs=[x, axes_const_name],
        outputs=[reduce_max_name],
        name=reduce_max_name,
        keepdims=1,
    )
    new_nodes.append(reduce_max_node)

    # Compute a boolean mask indicating where x equals the maximum.
    max_mask_node = helper.make_node(
        "Equal",
        inputs=[x, reduce_max_name],
        outputs=[max_mask_name],
        name=max_mask_name,
    )
    new_nodes.append(max_mask_node)

    # Compute the difference: diff =  x - reduce_max
    diff_node_1 = helper.make_node(
        "Sub", inputs=[x, reduce_max_name], outputs=[diff_name_1], name=diff_name_1
    )
    diff_node_2 = helper.make_node(
        "Sub", inputs=[reduce_max_name, x], outputs=[diff_name_2], name=diff_name_2
    )
    new_nodes.append(diff_node_1)
    new_nodes.append(diff_node_2)
    witness_nodes.append(diff_node_1)

    # Compute the square root of the difference.
    sqrt_node = helper.make_node(
        "Sqrt", inputs=[diff_name_2], outputs=[sqrt_name], name=sqrt_name
    )
    new_nodes.append(sqrt_node)

    # Cast the boolean mask to float.
    is_max_node = helper.make_node(
        "Cast",
        inputs=[max_mask_name],
        outputs=[is_max_name],
        name=is_max_name,
        to=INFERENCE_TYPE,
    )
    new_nodes.append(is_max_node)

    # 6. Create a constant ones tensor
    target_dtype = helper.tensor_dtype_to_np_dtype(INFERENCE_TYPE)
    ones_tensor = numpy_helper.from_array(
        np.array(1.0, dtype=target_dtype), name=ones_name + "_tensor"
    )
    ones_node = helper.make_node(
        "Constant", inputs=[], outputs=[ones_name], name=ones_name, value=ones_tensor
    )
    new_nodes.append(ones_node)

    # 7. Compute powers of diff:
    #    powers[0] = ones, powers[1] = diff, and then successive powers.
    powers = []
    powers.append(ones_name)  # Index 0.
    powers.append(diff_name_1)  # Index 1.
    for i in range(1, softmax_degree):
        prev_power_name = powers[-1]
        curr_power_name = (
            f"{name}.pow_{i+1}"  # e.g. "Softmax.pow_2", "Softmax.pow_3", etc.
        )
        power_node = helper.make_node(
            "Mul",
            inputs=[prev_power_name, diff_name_1],
            outputs=[curr_power_name],
            name=curr_power_name,
        )
        new_nodes.append(power_node)
        witness_nodes.append(power_node)
        powers.append(curr_power_name)

    # 8. Compute weighted terms for the Pade approximant.
    weighted_num_names = []
    weighted_denom_names = []
    for i in range(softmax_degree + 1):
        # Create constant nodes for P[i] and Q[i].
        p_name = f"{name}.p_{i}"
        p_tensor = numpy_helper.from_array(
            np.array(P[i], dtype=target_dtype), name=p_name + "_tensor"
        )
        p_node = helper.make_node(
            "Constant", inputs=[], outputs=[p_name], name=p_name, value=p_tensor
        )
        new_nodes.append(p_node)

        q_name = f"{name}.q_{i}"
        q_tensor = numpy_helper.from_array(
            np.array(Q[i], dtype=target_dtype), name=q_name + "_tensor"
        )
        q_node = helper.make_node(
            "Constant", inputs=[], outputs=[q_name], name=q_name, value=q_tensor
        )
        new_nodes.append(q_node)

        # Multiply the i-th power with the corresponding coefficient.
        weighted_num_name = f"{name}.weighted_num_{i}"
        weighted_num_node = helper.make_node(
            "Mul",
            inputs=[powers[i], p_name],
            outputs=[weighted_num_name],
            name=weighted_num_name,
        )
        new_nodes.append(weighted_num_node)
        weighted_num_names.append(weighted_num_name)

        weighted_denom_name = f"{name}.weighted_denom_{i}"
        weighted_denom_node = helper.make_node(
            "Mul",
            inputs=[powers[i], q_name],
            outputs=[weighted_denom_name],
            name=weighted_denom_name,
        )
        new_nodes.append(weighted_denom_node)
        weighted_denom_names.append(weighted_denom_name)

    # 9. Sum the weighted numerator terms to form approx_num.
    current_num = weighted_num_names[0]
    for i in range(1, len(weighted_num_names)):
        tmp_num_name = f"{name}.approx_num_{i}"
        add_num_node = helper.make_node(
            "Add",
            inputs=[current_num, weighted_num_names[i]],
            outputs=[tmp_num_name],
            name=tmp_num_name,
        )
        new_nodes.append(add_num_node)
        current_num = tmp_num_name
    approx_num_final = current_num

    # 10. Sum the weighted denominator terms to form approx_denom.
    current_denom = weighted_denom_names[0]
    for i in range(1, len(weighted_denom_names)):
        tmp_denom_name = f"{name}.approx_denom_{i}"
        add_denom_node = helper.make_node(
            "Add",
            inputs=[current_denom, weighted_denom_names[i]],
            outputs=[tmp_denom_name],
            name=tmp_denom_name,
        )
        new_nodes.append(add_denom_node)
        current_denom = tmp_denom_name
    approx_denom_final = current_denom

    # 11. Compute the reciprocal of approx_denom.
    recip_name = f"{name}.approx_denom_recip"
    recip_node = helper.make_node(
        "Reciprocal", inputs=[approx_denom_final], outputs=[recip_name], name=recip_name
    )
    new_nodes.append(recip_node)

    # 12. Compute approx_e_x = approx_num * Reciprocal(approx_denom).
    approx_e_x_name = f"{name}.approx_e_x"
    approx_e_x_node = helper.make_node(
        "Mul",
        inputs=[approx_num_final, recip_name],
        outputs=[approx_e_x_name],
        name=approx_e_x_name,
    )
    new_nodes.append(approx_e_x_node)

    # 13. Compute scale factor: reduce_sum of approx_e_x along the specified axis.
    reduce_sum_name = f"{name}.reduce_sum"
    reduce_sum_node = helper.make_node(
        "ReduceSum",
        inputs=[approx_e_x_name, axes_const_name],
        outputs=[reduce_sum_name],
        name=reduce_sum_name,
        keepdims=1,
    )
    new_nodes.append(reduce_sum_node)

    # 14. Final output: softmax = approx_e_x / scale_factor.
    final_softmax_name = name
    output_node = helper.make_node(
        "Div",
        inputs=[approx_e_x_name, reduce_sum_name],
        outputs=[output_id],
        name=final_softmax_name,
    )
    new_nodes.append(output_node)

    # Remove the original node and add the new nodes to the graph.
    graph.node.remove(node)
    graph.node.extend(new_nodes)
    witness_nodes.extend(
        [reduce_max_node, sqrt_node, is_max_node, approx_e_x_node, reduce_sum_node]
    )

    return (output_node, witness_nodes)


def constrain_softmax(r1cs, rnode):
    inputs = rnode.node.input
    axis = None
    for attr in rnode.node.attribute:
        if attr.name == "axis":
            axis = attr.i
    if axis is None:
        raise Exception("Softmax node is missing the 'axis' attribute.")
    if axis == -1:
        axis = len(rnode.shape) - 1
    axes = [axis]
    input_linmap, linmap_output = r1cs.get_node(inputs[0])
    in_dims = input_linmap.shape
    softmax_degree = len(P) - 1
    normalized_input = r1cs.allocate_tensor(in_dims, virtual=True)
    powers_of_x = []
    previous_powers = normalized_input
    print("Powers...")
    for deg in range(1, softmax_degree):
        x_pow = r1cs.allocate_tensor(in_dims, virtual=True)
        print(f"  Degree {deg} of {softmax_degree}...")
        constraints = 0
        for prev_val, norm_val, x_val in zip(
            previous_powers.iter_values(),
            normalized_input.iter_values(),
            x_pow.iter_values(),
        ):
            rnode.append_labeled_constraint(prev_val, norm_val, x_val, "powers of x")
            constraints += 1
            if constraints % 10000 == 0:
                r1cs.process_constrained_node(rnode)
        powers_of_x.append(x_pow)
        previous_powers = x_pow

    iterating_shape = list(in_dims)
    for i in range(len(iterating_shape)):
        if i in axes:
            iterating_shape[i] = 1

    # Allocate tensors for maxes and other intermediate values.
    maxes = r1cs.allocate_tensor(tuple(iterating_shape))
    roots = r1cs.allocate_tensor(in_dims, virtual=True)
    indicators = r1cs.allocate_tensor(in_dims, virtual=True)
    approx_e_x = r1cs.allocate_tensor(in_dims, virtual=True)
    scale_factors = r1cs.allocate_tensor(tuple(iterating_shape))
    approx_softmax = r1cs.allocate_tensor(
        in_dims, virtual=True, var_type=rnode.var_type
    )

    # Approximate exp(normalized_input) using a rational function:
    #   e^x ~ (P[0] + P[1]*x + P[2]*x^2 + P[3]*x^3) / (Q[0] + Q[1]*x + Q[2]*x^2 + Q[3]*x^3)
    print("Constrain softmax with rational approximation...")
    constraints = 0
    for norm_val, pow1_val, pow2_val, e_x_val in zip(
        normalized_input.iter_values(),
        powers_of_x[0].iter_values(),
        powers_of_x[1].iter_values(),
        approx_e_x.iter_values(),
    ):
        constraints += 1
        numerator = (
            ir.LinearCombo.from_const(P[0])
            + norm_val * P[1]
            + pow1_val * P[2]
            + pow2_val * P[3]
        )
        denominator = (
            ir.LinearCombo.from_const(Q[0])
            + norm_val * Q[1]
            + pow1_val * Q[2]
            + pow2_val * Q[3]
        )
        rnode.append_labeled_constraint(
            denominator,  # left
            e_x_val,  # right (≈ eˣ)
            numerator,  # output
            "e_x approx",
        )
        if constraints % 10000 == 0:
            print(f"    {constraints} constraints so far...")
            r1cs.process_constrained_node(rnode)
    r1cs.process_constrained_node(rnode)
    # For each “slice” along the non-softmax axes, enforce softmax constraints.
    print("Slices...")
    for it_coords in np.ndindex(tuple(iterating_shape)):
        idx_tuple = tuple(
            slice(None) if i in axes else it_coords[i] for i in range(len(in_dims))
        )
        print("Progress: ", it_coords, "/", iterating_shape)
        in_view = linmap_output[idx_tuple]
        if not isinstance(in_view, np.ndarray):
            in_view = np.array([in_view])
        approx_softmax_view = approx_softmax[idx_tuple]
        approx_e_x_view = approx_e_x[idx_tuple]
        roots_view = roots[idx_tuple]
        indicators_view = indicators[idx_tuple]
        max_val = maxes[it_coords]
        # Ensure that the scale factors are the sum of the approximated e^x values across the softmax axis
        scale_factor_sum = ir.LinearCombo.zero()
        for e_x in approx_e_x_view.iter_values():
            scale_factor_sum += e_x
        rnode.append_labeled_constraint(
            scale_factors[it_coords],
            ir.LinearCombo.from_const(1),
            scale_factor_sum,
            "scale factor sum",
        )
        softmax_to_constraints_helper(
            in_view,
            approx_softmax_view,
            approx_e_x_view,
            input_linmap.is_var,
            rnode,
            max_val,
            roots_view,
            indicators_view,
            scale_factors[it_coords],
            r1cs,
        )
        r1cs.process_constrained_node(rnode)
    rnode.output = approx_softmax.materialize()


def softmax_to_constraints_helper(
    input_view,
    approx_softmax_view,
    approx_e_x_view,
    is_var,
    rnode,
    max_val,
    s_view,
    indicators_view,
    scale_factor,
    r1cs,
):
    if not is_var:
        raise Exception("expected variable input, constant not implemented")

    # For each element in the view, enforce that the square of s equals (max - input).
    for val, s in zip(input_view.flat, s_view.iter_values()):
        rnode.append_labeled_constraint(s, s, max_val - val, "max check0")

    # Enforce that at least one element equals the max:
    # For each element, (input - max)*indicator == 0 and indicator is Boolean.
    sum_indicators = ir.LinearCombo.zero()
    constraints = 0
    for inp_val, indicator in zip(input_view.flat, indicators_view.iter_values()):
        constraints += 1
        rnode.append_labeled_constraint(
            inp_val - max_val, indicator, ir.LinearCombo.from_const(0), "max check1"
        )
        rnode.append_labeled_constraint(
            indicator,
            indicator - ir.LinearCombo.from_const(1),
            ir.LinearCombo.from_const(0),
            "max check2",
        )
        sum_indicators += indicator
        if constraints % 10000 == 0:
            r1cs.process_constrained_node(rnode)

    # Constrain the sum of the indicators to be 1.
    rnode.append_labeled_constraint(
        sum_indicators,
        ir.LinearCombo.from_const(1),
        ir.LinearCombo.from_const(1),
        "sum check",
    )

    # Enforce normalization:
    # Let scale_factor = sum_i e^x_i, then for every element,
    # scale_factor * softmax[i] == approx_e_x[i].
    constraints = 0
    for e_x, softmax_out in zip(
        approx_e_x_view.iter_values(), approx_softmax_view.iter_values()
    ):
        constraints += 1
        rnode.append_labeled_constraint(scale_factor, softmax_out, e_x, "scale check")
        if constraints % 10000 == 0:
            r1cs.process_constrained_node(rnode)
