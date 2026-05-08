import numpy as np
from onnx import helper, numpy_helper
import ir
from eval import INFERENCE_TYPE

def wire_relu(node, graph):
    name = node.name
    input_name = node.input[0]
    output_name = node.output[0]

    # Consts
    zero_val = np.array(0.0, dtype=np.float32)
    one_val  = np.array(1.0, dtype=np.float32)
    two_val  = np.array(2.0, dtype=np.float32)
    half_val = np.array(0.5, dtype=np.float32)
    zero_tensor = numpy_helper.from_array(zero_val, name=f"{name}.zero_tensor")
    one_tensor  = numpy_helper.from_array(one_val,  name=f"{name}.one_tensor")
    two_tensor  = numpy_helper.from_array(two_val,  name=f"{name}.two_tensor")
    half_tensor = numpy_helper.from_array(half_val, name=f"{name}.half_tensor")
    zero_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{name}.zero"],
        name=f"{name}.zero",
        value=zero_tensor,
    )
    one_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{name}.one"],
        name=f"{name}.one",
        value=one_tensor,
    )
    two_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{name}.two"],
        name=f"{name}.two",
        value=two_tensor,
    )
    half_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[f"{name}.half"],
        name=f"{name}.half",
        value=half_tensor,
    )

    # Indicator x > 0
    b_bool = f"{name}.b_bool"
    b_bool_node = helper.make_node(
        "Greater",
        inputs=[input_name, f"{name}.zero"],
        outputs=[b_bool],
        name=f"{name}.b_bool",
    )
    b = f"{name}.b"
    b_cast_node = helper.make_node(
        "Cast",
        inputs=[b_bool],
        outputs=[b],
        name=f"{name}.b",
        to=INFERENCE_TYPE,
    )

    # Sign of x
    two_b = f"{name}.two_b"
    two_b_node = helper.make_node(
        "Mul",
        inputs=[b, f"{name}.two"],
        outputs=[two_b],
        name=f"{name}.two_b",
    )
    s = f"{name}.s"
    s_node = helper.make_node(
        "Sub",
        inputs=[two_b, f"{name}.one"],
        outputs=[s],
        name=f"{name}.s",
    )

    # -abs = s * x 
    abs_name = f"{name}.abs"
    abs_node = helper.make_node(
        "Mul",
        inputs=[s, input_name],
        outputs=[abs_name],
        name=abs_name,
    )

    # sqrt = sqrt(abs) 
    sqrt_name = f"{name}.sqrt"
    sqrt_node = helper.make_node(
        "Sqrt",
        inputs=[abs_name],
        outputs=[sqrt_name],
        name=sqrt_name,
    )

    # output = (x + abs) / 2 
    x_plus_abs = f"{name}.x_plus_abs"
    x_plus_abs_node = helper.make_node(
        "Add",
        inputs=[input_name, abs_name],
        outputs=[x_plus_abs],
        name=x_plus_abs,
    )

    scaled_node = helper.make_node(
        "Mul",
        inputs=[x_plus_abs, f"{name}.half"],
        outputs=[output_name],
        name=output_name,
    )
    graph.node.remove(node)

    new_nodes = [
        zero_node,
        one_node,
        two_node,
        half_node,
        b_bool_node,
        b_cast_node,
        two_b_node,
        s_node,
        abs_node,
        sqrt_node,
        x_plus_abs_node,
        scaled_node,
    ]

    witness_nodes = [s_node, abs_node, sqrt_node, scaled_node]
    graph.node.extend(new_nodes)
    return (scaled_node, witness_nodes)

def constrain_relu(r1cs, rnode):
    inputs = rnode.node.input
    inp_meta = r1cs.nodes.get(inputs[0])
    if not inp_meta.is_var:
        raise Exception("Input to ReLU must be a variable")
    dims = rnode.shape

    # Witness values representing sign(x), abs(x), sqrt(abs(x)), and out = x*(s+1)/2
    s = r1cs.allocate_tensor(dims, virtual=True)
    abs_v = r1cs.allocate_tensor(dims, virtual=True)
    sqrt_v = r1cs.allocate_tensor(dims, virtual=True)
    out = r1cs.allocate_tensor(dims, var_type=rnode.var_type)
    res = np.empty(dims, dtype=object)
    constraints = 0
    for idx in np.ndindex(dims):
        x = inp_meta.output[idx]
        s_val = s[idx]
        abs_val = abs_v[idx]
        sqrt_val = sqrt_v[idx]
        out_val = out[idx]
        # s^2 = 1  
        rnode.append_labeled_constraint(
            s_val,
            s_val,
            ir.LinearCombo.one(),
            "relu: s^2 = 1",
        )
        #  abs = s * x
        rnode.append_labeled_constraint(
            s_val,
            x,
            abs_val,
            "relu: abs = s * x",
        )

        # abs = sqrt^2, shows that abs is positive
        rnode.append_labeled_constraint(
            sqrt_val,
            sqrt_val,
            abs_val,
            "relu: abs = sqrt^2",
        )

        # 4) output = (x + abs) / 2  <->  2*out = x + abs
        lhs = 2 * out_val - x - abs_val
        rnode.append_labeled_constraint(
            lhs,
            ir.LinearCombo.one(),
            ir.LinearCombo.zero(),
            "relu: 2*out = x + abs",
        )

        constraints += 1
        if constraints % 10000 == 0:
            r1cs.process_constrained_node(rnode)

        res[idx] = out_val

    rnode.output = res

