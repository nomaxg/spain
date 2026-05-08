import numpy as np
from onnx import helper, numpy_helper, TensorProto
import ir
from eval import INFERENCE_TYPE


def wire_erf(node, graph):
    name = node.name
    input_name = node.input[0]
    output_name = node.output[0]

    # Create constant nodes for 1.0 and -1.0.
    ones_value = np.array(1.0, dtype=np.float32)
    neg_ones_value = np.array(-1.0, dtype=np.float32)
    ones_tensor = numpy_helper.from_array(ones_value, name=f"{name}.ones_tensor")
    neg_ones_tensor = numpy_helper.from_array(
        neg_ones_value, name=f"{name}.neg_ones_tensor"
    )

    ones_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{name}.ones"],
        name=f"{name}.ones",
        value=ones_tensor,
    )
    neg_ones_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{name}.neg_ones"],
        name=f"{name}.neg_ones",
        value=neg_ones_tensor,
    )

    b1_bool = f"{name}.b1_bool"
    b1_node = helper.make_node(
        "Greater",
        inputs=[input_name, f"{name}.ones"],
        outputs=[b1_bool],
        name=f"{name}.b1",
    )
    b1 = f"{name}.b1_f32"
    b1_cast_node = helper.make_node(
        "Cast",
        inputs=[b1_bool],
        outputs=[b1],
        name=f"{name}.b1_f32",
        to=INFERENCE_TYPE,
    )

    b2_bool = f"{name}.b2_bool"
    b2_node = helper.make_node(
        "Less",
        inputs=[input_name, f"{name}.neg_ones"],
        outputs=[b2_bool],
        name=f"{name}.b2",
    )
    b2 = f"{name}.b2_f32"
    b2_cast_node = helper.make_node(
        "Cast",
        inputs=[b2_bool],
        outputs=[b2],
        name=f"{name}.b2_f32",
        to=INFERENCE_TYPE,
    )

    b3_intermediate = f"{name}.b3_0"
    b3_sub1 = helper.make_node(
        "Sub",
        inputs=[f"{name}.ones", b1],
        outputs=[b3_intermediate],
        name=f"{name}.b3_0",
    )
    b3 = f"{name}.b3"
    b3_sub2 = helper.make_node(
        "Sub", inputs=[b3_intermediate, b2], outputs=[b3], name=f"{name}.b3_1"
    )

    s_sub = f"{name}.s_squared_0"
    s_sub_node = helper.make_node(
        "Sub",
        inputs=[input_name, f"{name}.ones"],
        outputs=[s_sub],
        name=f"{name}.s_squared_0",
    )
    s_squared = f"{name}.s_squared_1"
    s_mul_node = helper.make_node(
        "Mul", inputs=[s_sub, b1], outputs=[s_squared], name=f"{name}.s_squared_1"
    )
    s = f"{name}.s"
    s_sqrt_node = helper.make_node(
        "Sqrt", inputs=[s_squared], outputs=[s], name=f"{name}.s"
    )

    t_sub = f"{name}.t_squared_0"
    t_sub_node = helper.make_node(
        "Sub",
        inputs=[f"{name}.neg_ones", input_name],
        outputs=[t_sub],
        name=f"{name}.t_squared_0",
    )
    t_squared = f"{name}.t_squared_1"
    t_mul_node = helper.make_node(
        "Mul", inputs=[t_sub, b2], outputs=[t_squared], name=f"{name}.t_squared_1"
    )
    t = f"{name}.t"
    t_sqrt_node = helper.make_node(
        "Sqrt", inputs=[t_squared], outputs=[t], name=f"{name}.t"
    )

    u_sub = f"{name}.u_squared_0"
    u_sub_node = helper.make_node(
        "Sub",
        inputs=[f"{name}.ones", input_name],
        outputs=[u_sub],
        name=f"{name}.u_squared_0",
    )
    u_squared = f"{name}.u_squared_1"
    u_mul_node = helper.make_node(
        "Mul", inputs=[u_sub, b3], outputs=[u_squared], name=f"{name}.u_squared_1"
    )
    u = f"{name}.u"
    u_sqrt_node = helper.make_node(
        "Sqrt", inputs=[u_squared], outputs=[u], name=f"{name}.u"
    )

    v_add = f"{name}.v_squared_0"
    v_add_node = helper.make_node(
        "Add",
        inputs=[f"{name}.ones", input_name],
        outputs=[v_add],
        name=f"{name}.v_squared_0",
    )
    v_squared = f"{name}.v_squared_1"
    v_mul_node = helper.make_node(
        "Mul", inputs=[v_add, b3], outputs=[v_squared], name=f"{name}.v_squared_1"
    )
    v = f"{name}.v"
    v_sqrt_node = helper.make_node(
        "Sqrt", inputs=[v_squared], outputs=[v], name=f"{name}.v"
    )

    b3_times_x = f"{name}.b3_times_x"
    b3_mul_node = helper.make_node(
        "Mul", inputs=[b3, input_name], outputs=[b3_times_x], name=b3_times_x
    )

    output_node_int = helper.make_node(
        "Sub",
        inputs=[b1, b2],
        outputs=[f"{name}.output_int"],
        name=f"{name}.output_int",
    )

    output_node = helper.make_node(
        "Add",
        inputs=[output_node_int.name, b3_mul_node.name],
        outputs=[output_name],
        name=name,
    )

    new_nodes = [
        ones_node,
        neg_ones_node,
        b1_node,
        b1_cast_node,
        b2_node,
        b2_cast_node,
        b3_sub1,
        b3_sub2,
        s_sub_node,
        s_mul_node,
        s_sqrt_node,
        t_sub_node,
        t_mul_node,
        t_sqrt_node,
        u_sub_node,
        u_mul_node,
        u_sqrt_node,
        v_add_node,
        v_mul_node,
        v_sqrt_node,
        b3_mul_node,
        output_node_int,
    ]
    witness_nodes = [
        b1_cast_node,
        b2_cast_node,
        b3_sub2,
        s_mul_node,
        t_mul_node,
        u_mul_node,
        v_mul_node,
        s_sqrt_node,
        t_sqrt_node,
        u_sqrt_node,
        v_sqrt_node,
        b3_mul_node,
    ]
    graph.node.remove(node)
    graph.node.append(output_node)
    graph.node.extend(new_nodes)

    return (output_node, witness_nodes)


def constrain_erf(r1cs, rnode):
    inputs = rnode.node.input
    inp = r1cs.nodes.get(inputs[0])
    dims = rnode.shape
    b1 = r1cs.allocate_tensor(dims, virtual=True)
    b2 = r1cs.allocate_tensor(dims, virtual=True)
    b3 = r1cs.allocate_tensor(dims, virtual=True)
    s_squared = r1cs.allocate_tensor(dims, virtual=True)
    t_squared = r1cs.allocate_tensor(dims, virtual=True)
    u_squared = r1cs.allocate_tensor(dims, virtual=True)
    v_squared = r1cs.allocate_tensor(dims, virtual=True)
    s = r1cs.allocate_tensor(dims, virtual=True)
    t = r1cs.allocate_tensor(dims, virtual=True)
    u = r1cs.allocate_tensor(dims, virtual=True)
    v = r1cs.allocate_tensor(dims, virtual=True)
    b3_times_x = r1cs.allocate_tensor(dims)
    res = np.empty(dims, dtype=object)
    constraints = 0

    # For each element in the tensor, add constraints
    for idx in np.ndindex(dims):
        x = inp.output[idx]
        if not inp.is_var:
            raise Exception("Input to erf must be a variable")
        else:
            # Retrieve the corresponding elements for all allocated tensors
            b1_val = b1[idx]
            b2_val = b2[idx]
            b3_val = b3[idx]
            s_val = s[idx]
            t_val = t[idx]
            u_val = u[idx]
            v_val = v[idx]
            s_squared_val = s_squared[idx]
            t_squared_val = t_squared[idx]
            u_squared_val = u_squared[idx]
            v_squared_val = v_squared[idx]
            b3_times_x_val = b3_times_x[idx]

            # 1. Enforce that exactly one branch variable is true: b1 + b2 + b3 == 1
            rnode.append_labeled_constraint(
                b1_val + b2_val + b3_val,
                ir.LinearCombo.one(),
                ir.LinearCombo.one(),
                "erf: exactly one branch variable is true",
            )
            # 2. Enforce b1, b2, and b3 are Boolean (0 or 1)
            rnode.append_labeled_constraint(
                b1_val,
                ir.LinearCombo.one() - b1_val,
                ir.LinearCombo.zero(),
                "erf: b1 is 0 or 1",
            )
            rnode.append_labeled_constraint(
                b2_val,
                ir.LinearCombo.one() - b2_val,
                ir.LinearCombo.zero(),
                "erf: b2 is 0 or 1",
            )
            # 3. Add constraints for the square root relationships:
            #    s^2 = s * s, t^2 = t * t, etc.
            rnode.append_labeled_constraint(
                s_val, s_val, s_squared_val, "erf: s^2 = x - 1"
            )
            rnode.append_labeled_constraint(
                t_val, t_val, t_squared_val, "erf: t^2 = 1 - x"
            )
            rnode.append_labeled_constraint(
                u_val, u_val, u_squared_val, "erf: u^2 = 1 - x"
            )
            rnode.append_labeled_constraint(
                v_val, v_val, v_squared_val, "erf: v^2 = 1 + x"
            )
            # 4. Branch-specific constraints:
            # For branch b1 (x > 1): s^2 = x - 1
            rnode.append_labeled_constraint(
                b1_val,
                x - ir.LinearCombo.one(),
                s_squared_val,
                "erf: b1 implies s^2 = x - 1",
            )
            # For branch b2 (x < -1): t^2 = 1 - x
            rnode.append_labeled_constraint(
                b2_val,
                ir.LinearCombo.zero() - ir.LinearCombo.one() - x,
                t_squared_val,
                "erf: b2 implies t^2 = 1 - x",
            )
            # For branch b3 (-1 <= x <= 1): u^2 = 1 - x and v^2 = 1 + x
            rnode.append_labeled_constraint(
                b3_val,
                ir.LinearCombo.one() - x,
                u_squared_val,
                "erf: b3 implies u^2 = 1 - x",
            )
            rnode.append_labeled_constraint(
                b3_val,
                ir.LinearCombo.one() + x,
                v_squared_val,
                "erf: b3 implies v^2 = 1 + x",
            )
            # For branch b3, also constrain b3_times_x to equal x
            rnode.append_labeled_constraint(
                b3_val, x, b3_times_x_val, "erf: b3 implies b3_times_x = x"
            )
            # 5. Define the output: r = b1 - b2 + b3 * x (using b3_times_x)
            constraints += 1
            if constraints % 10000 == 0:
                print("Progress: ", idx, "/", dims)
                r1cs.process_constrained_node(rnode)
            res[idx] = b1_val - b2_val + b3_times_x_val

    rnode.output = res
