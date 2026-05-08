import numpy as np
from onnx import helper, numpy_helper
import ir


def wire_sin(node, graph, shift):
    name = node.name
    x_name = node.input[0]
    y_name = node.output[0]

    # Consts
    two_pi_value = np.array(2.0 * np.pi, dtype=np.float32)
    two_value = np.array(2.0, dtype=np.float32)
    # Approximating coefficients for Taylor series expansion sin(x) \approx x - x^3/6 + x^5/120
    neg_one_sixth_val = np.array(-1.0 / 6.0, dtype=np.float32)
    one_over_120_val = np.array(1.0 / 120.0, dtype=np.float32)
    pi_over_2_value = np.array(np.pi / 2.0, dtype=np.float32)
    shift_value = np.array(shift, dtype=np.float32)

    two_pi_tensor = numpy_helper.from_array(two_pi_value, name=f"{name}.two_pi_tensor")
    two_tensor = numpy_helper.from_array(two_value, name=f"{name}.two_tensor")
    shift_tensor = numpy_helper.from_array(shift_value, name=f"{name}.shift_tensor")
    neg_one_sixth_tensor = numpy_helper.from_array(
        neg_one_sixth_val, name=f"{name}.neg_one_sixth_tensor"
    )
    one_over_120_tensor = numpy_helper.from_array(
        one_over_120_val, name=f"{name}.one_over_120_tensor"
    )
    pi_over_2_tensor = numpy_helper.from_array(
        pi_over_2_value, name=f"{name}.pi_over_2_tensor"
    )

    # Nodes for all constants
    two_pi_const = f"{name}.two_pi"
    two_pi_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[two_pi_const],
        name=two_pi_const,
        value=two_pi_tensor,
    )

    two_const = f"{name}.two"
    two_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[two_const],
        name=two_const,
        value=two_tensor,
    )

    neg_one_sixth_const = f"{name}.neg_one_sixth"
    neg_one_sixth_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[neg_one_sixth_const],
        name=neg_one_sixth_const,
        value=neg_one_sixth_tensor,
    )

    one_over_120_const = f"{name}.one_over_120"
    one_over_120_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[one_over_120_const],
        name=one_over_120_const,
        value=one_over_120_tensor,
    )

    pi_over_2_const = f"{name}.pi_over_2"
    pi_over_2_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[pi_over_2_const],
        name=pi_over_2_const,
        value=pi_over_2_tensor,
    )

    shift_const = f"{name}.shift"
    shift_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[shift_const],
        name=shift_const,
        value=shift_tensor,
    )

    # x = x + shift
    shift_x = f"{name}.shift_x"
    shift_x_node = helper.make_node(
        "Add",
        inputs=[x_name, shift_const],
        outputs=[shift_x],
        name=shift_x,
    )

    # abs_x = Abs(x)
    abs_x = f"{name}.abs_x"
    abs_x_node = helper.make_node(
        "Abs",
        inputs=[shift_x],
        outputs=[abs_x],
        name=abs_x,
    )

    # scaled = abs_x / (2pi)
    scaled = f"{name}.scaled"
    scaled_node = helper.make_node(
        "Div",
        inputs=[abs_x, two_pi_const],
        outputs=[scaled],
        name=scaled,
    )

    # m_real = Floor(scaled)
    m_real = f"{name}.m_real"
    m_real_node = helper.make_node(
        "Floor",
        inputs=[scaled],
        outputs=[m_real],
        name=m_real,
    )

    # s = Sign(x)
    s_name = f"{name}.sign"
    s_node = helper.make_node(
        "Sign",
        inputs=[shift_x],
        outputs=[s_name],
        name=s_name,
    )

    # sm = s * m_real
    sm = f"{name}.sm"
    sm_node = helper.make_node(
        "Mul",
        inputs=[s_name, m_real],
        outputs=[sm],
        name=sm,
    )

    # sm_2pi = sm * (2pi)
    sm_2pi = f"{name}.sm_2pi"
    sm_2pi_node = helper.make_node(
        "Mul",
        inputs=[sm, two_pi_const],
        outputs=[sm_2pi],
        name=sm_2pi,
    )

    # x' = x - sm_2pi
    x_prime = f"{name}.x_prime"
    x_prime_node = helper.make_node(
        "Sub",
        inputs=[shift_x, sm_2pi],
        outputs=[x_prime],
        name=x_prime,
    )

    # Range witnesses: x' in [-pi/2, pi/2]

    # left_expr = pi/2 - x'
    left_expr = f"{name}.left_expr"
    left_expr_node = helper.make_node(
        "Sub",
        inputs=[pi_over_2_const, x_prime],
        outputs=[left_expr],
        name=left_expr,
    )

    # right_expr = x' + pi/2
    right_expr = f"{name}.right_expr"
    right_expr_node = helper.make_node(
        "Add",
        inputs=[x_prime, pi_over_2_const],
        outputs=[right_expr],
        name=right_expr,
    )

    # sqrt_left = sqrt(left_expr)
    sqrt_left = f"{name}.sqrt_left"
    sqrt_left_node = helper.make_node(
        "Sqrt",
        inputs=[left_expr],
        outputs=[sqrt_left],
        name=sqrt_left,
    )

    # sqrt_right = sqrt(right_expr)
    sqrt_right = f"{name}.sqrt_right"
    sqrt_right_node = helper.make_node(
        "Sqrt",
        inputs=[right_expr],
        outputs=[sqrt_right],
        name=sqrt_right,
    )

    # Taylor series approx: x': x' - x'^3/6 + x'^5/120

    # x2 = x'^2
    x2 = f"{name}.x2"
    x2_node = helper.make_node(
        "Mul",
        inputs=[x_prime, x_prime],
        outputs=[x2],
        name=x2,
    )

    # x3 = x2 * x'
    x3 = f"{name}.x3"
    x3_node = helper.make_node(
        "Mul",
        inputs=[x2, x_prime],
        outputs=[x3],
        name=x3,
    )

    # x5 = x3 * x2
    x5 = f"{name}.x5"
    x5_node = helper.make_node(
        "Mul",
        inputs=[x3, x2],
        outputs=[x5],
        name=x5,
    )

    # term3 = (-1/6) * x3
    term3 = f"{name}.term3"
    term3_node = helper.make_node(
        "Mul",
        inputs=[x3, neg_one_sixth_const],
        outputs=[term3],
        name=term3,
    )

    # term5 = (1/120) * x5
    term5 = f"{name}.term5"
    term5_node = helper.make_node(
        "Mul",
        inputs=[x5, one_over_120_const],
        outputs=[term5],
        name=term5,
    )

    # y1 = x_prime + term3
    y1 = f"{name}.y1"
    y1_node = helper.make_node(
        "Add",
        inputs=[x_prime, term3],
        outputs=[y1],
        name=y1,
    )

    # Y = y1 + term5
    output_node = helper.make_node(
        "Add",
        inputs=[y1, term5],
        outputs=[y_name],
        name=name,
    )

    # m_real must be integer, so compute witnesses to binary decomposition

    bit_nodes = []
    extra_nodes = []

    for i in range(65):
        # pow2_i = 2^i
        pow2_val = np.array(float(1 << i), dtype=np.float32)
        pow2_tensor = numpy_helper.from_array(pow2_val, name=f"{name}.pow2_{i}_tensor")
        pow2_const = f"{name}.pow2_{i}"
        pow2_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[pow2_const],
            name=pow2_const,
            value=pow2_tensor,
        )

        # div_i = m_real / 2^i
        div_i = f"{name}.m_div_{i}"
        div_i_node = helper.make_node(
            "Div",
            inputs=[m_real, pow2_const],
            outputs=[div_i],
            name=div_i,
        )

        # floor_i = Floor(div_i)
        floor_i = f"{name}.m_div_floor_{i}"
        floor_i_node = helper.make_node(
            "Floor",
            inputs=[div_i],
            outputs=[floor_i],
            name=floor_i,
        )

        # m_bit_i = Mod(floor_i, 2)
        m_bit_i = f"{name}.m_bit_{i}"
        m_bit_i_node = helper.make_node(
            "Mod",
            inputs=[floor_i, two_const],
            outputs=[m_bit_i],
            name=m_bit_i,
            fmod=1,
        )

        extra_nodes.extend([pow2_node, div_i_node, floor_i_node, m_bit_i_node])
        bit_nodes.append(m_bit_i_node)

    graph.node.remove(node)
    graph.node.extend(
        [
            two_pi_node,
            two_node,
            neg_one_sixth_node,
            one_over_120_node,
            pi_over_2_node,
            shift_node,
            shift_x_node,
            abs_x_node,
            scaled_node,
            m_real_node,
            s_node,
            sm_node,
            sm_2pi_node,
            x_prime_node,
            left_expr_node,
            right_expr_node,
            sqrt_left_node,
            sqrt_right_node,
            x2_node,
            x3_node,
            x5_node,
            term3_node,
            term5_node,
            y1_node,
            output_node,
        ]
        + extra_nodes
    )

    witness_nodes = [x_prime_node, s_node, m_real_node, sm_node]
    witness_nodes.extend(bit_nodes)
    witness_nodes.extend(
        [
            x2_node,
            x3_node,
            x5_node,
            left_expr_node,
            right_expr_node,
            sqrt_left_node,
            sqrt_right_node,
        ]
    )

    return (output_node, witness_nodes)


def constrain_sin(r1cs, rnode, shift):
    inputs = rnode.node.input
    inp = r1cs.nodes.get(inputs[0])
    dims = rnode.shape

    if not inp.is_var:
        raise Exception("Input to sin must be a variable")
    # x' = x - sm * 2pi
    x_prime = r1cs.allocate_tensor(dims, virtual=True)
    s = r1cs.allocate_tensor(dims, virtual=True)
    m = r1cs.allocate_tensor(dims, virtual=True)
    sm = r1cs.allocate_tensor(dims, virtual=True)
    # Binary decomp of m
    m_bits = [r1cs.allocate_tensor(dims, virtual=True) for _ in range(65)]
    # x^2, x^3, x^5 for Taylor series approximation
    x2 = r1cs.allocate_tensor(dims, virtual=True)
    x3 = r1cs.allocate_tensor(dims, virtual=True)
    x5 = r1cs.allocate_tensor(dims, virtual=True)
    # witnesses to enforce range via square roots (x in [-pi/2, pi/2])
    left_sq = r1cs.allocate_tensor(dims, virtual=True)
    right_sq = r1cs.allocate_tensor(dims, virtual=True)
    u_range = r1cs.allocate_tensor(dims, virtual=True)
    v_range = r1cs.allocate_tensor(dims, virtual=True)
    res = np.empty(dims, dtype=object)
    # Consts
    TWO_PI = 2.0 * np.pi
    PI_OVER2 = np.pi / 2.0
    INV_6 = 1.0 / 6.0
    INV_120 = 1.0 / 120.0

    constraints = 0

    for idx in np.ndindex(dims):
        x_val = inp.output[idx]

        x_prime_val = x_prime[idx]
        s_val = s[idx]
        m_val = m[idx]
        sm_val = sm[idx]

        m_bits_vals = [mb[idx] for mb in m_bits]

        x2_val = x2[idx]
        x3_val = x3[idx]
        x5_val = x5[idx]

        left_sq_val = left_sq[idx]
        right_sq_val = right_sq[idx]
        u_val = u_range[idx]
        v_val = v_range[idx]

        # Bit constraints: m_i \in {0,1}
        for i, mi in enumerate(m_bits_vals):
            rnode.append_labeled_constraint(
                mi,
                ir.LinearCombo.one() - mi,
                ir.LinearCombo.zero(),
                f"sin: m_bit[{i}] is 0 or 1",
            )

        # m = \sum 2^i * m_i
        m_minus_decomp = m_val
        for i, mi in enumerate(m_bits_vals):
            m_minus_decomp = m_minus_decomp - (2**i) * mi

        rnode.append_labeled_constraint(
            m_minus_decomp,
            ir.LinearCombo.one(),
            ir.LinearCombo.zero(),
            "sin: m = sum 2^i m_i",
        )

        # s \in {-1,1} via s^2 = 1
        rnode.append_labeled_constraint(
            s_val,
            s_val,
            ir.LinearCombo.one(),
            "sin: s^2 = 1",
        )

        # sm = s * m
        rnode.append_labeled_constraint(
            s_val,
            m_val,
            sm_val,
            "sin: sm = s * m",
        )

        # x = x' + s * 2pi * m - shift (shift = 0 for sin, shift = pi/2 for cos)
        rnode.append_labeled_constraint(
            x_val - x_prime_val - TWO_PI * sm_val + ir.LinearCombo.from_const(shift),
            ir.LinearCombo.one(),
            ir.LinearCombo.zero(),
            "sin: x = x' + s * 2pi * m",
        )

        # Range: x' \in [-pi/2, pi/2] via square roots
        # u^2 = left_sq
        rnode.append_labeled_constraint(
            u_val, u_val, left_sq_val, "sin: u^2 = pi/2 - x'"
        )
        # v^2 = right_sq
        rnode.append_labeled_constraint(
            v_val, v_val, right_sq_val, "sin: v^2 = x' + pi/2"
        )
        # left_sq = pi/2 - x'
        rnode.append_labeled_constraint(
            PI_OVER2 * ir.LinearCombo.one() - x_prime_val - left_sq_val,
            ir.LinearCombo.one(),
            ir.LinearCombo.zero(),
            "sin: left_sq = pi/2 - x'",
        )
        # right_sq = x' + pi/2
        rnode.append_labeled_constraint(
            x_prime_val + PI_OVER2 * ir.LinearCombo.one() - right_sq_val,
            ir.LinearCombo.one(),
            ir.LinearCombo.zero(),
            "sin: right_sq = x' + pi/2",
        )

        # Polynomial: sin(x') ≈ x' - x'^3/6 + x'^5/120
        # x2 = x'^2
        rnode.append_labeled_constraint(
            x_prime_val, x_prime_val, x2_val, "sin: x2 = x'^2"
        )
        # x3 = x2 * x' = x'^3
        rnode.append_labeled_constraint(x2_val, x_prime_val, x3_val, "sin: x3 = x'^3")
        # x5 = x3 * x2 = x'^5
        rnode.append_labeled_constraint(x3_val, x2_val, x5_val, "sin: x5 = x'^5")

        # Output r
        res[idx] = x_prime_val - INV_6 * x3_val + INV_120 * x5_val

        constraints += 1
        if constraints % 10000 == 0:
            print("sin: progress ", idx, "/", dims)
            r1cs.process_constrained_node(rnode)

    rnode.output = res
