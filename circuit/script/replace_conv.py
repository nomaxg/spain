# GPT script to transform all convolutions into matmuls (conv is hard to constrain as written)
#!/usr/bin/env python3
import argparse
import copy
from onnxsim import simplify
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort


# ---------- helpers ----------

def get_tensor_shape(model, name):
    """Fetch static shape for a value (initializer / input / intermediate)."""
    # 1) check initializers
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims)

    # 2) check value infos (inputs, outputs, intermediates)
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if vi.name == name:
            t = vi.type.tensor_type
            dims = [d.dim_value for d in t.shape.dim]
            if any(d == 0 for d in dims):
                raise ValueError(f"Dynamic or zero dims not supported for {name}: {dims}")
            return dims

    raise KeyError(f"Shape for value {name} not found in model.")


def get_initializer_array(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    raise KeyError(f"Initializer {name} not found.")


# Builds a convolution matrix whose
def build_conv_big_mat(x_shape, w_arr, y_shape):
    # X: [N, C_in, H, W]
    N, C_in, H, W = x_shape

    # W: [C_out, C_in, kH, kW]
    C_out, C_in_w, kH, kW = w_arr.shape
    assert C_in == C_in_w, "Conv: input channel mismatch"

    # Y: [N, C_out, H_out, W_out]
    _, C_out_y, H_out, W_out = y_shape
    assert C_out == C_out_y, "Conv: output channel mismatch"

    assert H_out == H and W_out == W, "Expected SAME_UPPER, stride=1 conv (H_out != H or W_out != W)"
    # For SAME_UPPER with stride 1 and odd kernel, padding is symmetric
    pad_y = (kH - 1) // 2
    pad_x = (kW - 1) // 2

    K_in  = C_in * H * W                 # flattened input length
    K_out = C_out * H_out * W_out        # flattened output length

    A = np.zeros((K_in, K_out), dtype=w_arr.dtype)

    # Build the linear map for ONE image (N is handled by batch MatMul).
    for oc in range(C_out):
        for oy in range(H_out):
            for ox in range(W_out):
                # Column index in Y_flat for (oc, oy, ox)
                out_col = oc * H_out * W_out + oy * W_out + ox

                for ic in range(C_in):
                    for ky in range(kH):
                        for kx in range(kW):
                            # coordinates in unpadded input
                            iy = oy + ky - pad_y
                            ix = ox + kx - pad_x
                            if 0 <= iy < H and 0 <= ix < W:
                                # Row index in X_flat for (ic, iy, ix)
                                in_row = ic * H * W + iy * W + ix

                                A[in_row, out_col] += w_arr[oc, ic, ky, kx]
                            # else: contribution is 0 (padding)

    return A


def replace_conv_with_big_matmul(model):
    """
    For every Conv node (with supported config), replace it with:
      Flatten(axis=1) -> MatMul(bigA) -> Reshape -> (optional Add for bias)

    Returns a new model.
    """
    model = copy.deepcopy(model)
    graph = model.graph

    # Run shape inference so we have shapes for intermediates
    model = shape_inference.infer_shapes(model)
    graph = model.graph

    new_nodes = []
    to_remove = []

    for node in graph.node:
        if node.op_type != "Conv":
            new_nodes.append(node)
            continue

        conv = node
        name = conv.name or "Conv"

        X_name = conv.input[0]
        W_name = conv.input[1]
        B_name = conv.input[2] if len(conv.input) > 2 else None
        Y_name = conv.output[0]

        # Only support the requested configuration
        # group=1, dilations=(1,1), strides=(1,1), auto_pad=SAME_UPPER
        attrs = {a.name: a for a in conv.attribute}

        def get_ints(name, default=None):
            if name in attrs:
                return list(attrs[name].ints)
            return default

        def get_str(name, default=None):
            if name in attrs:
                return attrs[name].s.decode("utf-8")
            return default

        strides = get_ints("strides", [1, 1])
        dilations = get_ints("dilations", [1, 1])
        group = attrs.get("group", None).i if "group" in attrs else 1
        auto_pad = get_str("auto_pad", "NOTSET")

        assert group == 1, f"{name}: only group=1 supported."
        assert strides == [1, 1], f"{name}: only strides=1 supported."
        assert dilations == [1, 1], f"{name}: only dilations=1 supported."
        assert auto_pad == "SAME_UPPER", f"{name}: only auto_pad=SAME_UPPER supported."

        # Shapes
        x_shape = get_tensor_shape(model, X_name)   # [N,C,H,W]
        y_shape = get_tensor_shape(model, Y_name)   # [N,C_out,H_out,W_out]
        w_arr = get_initializer_array(model, W_name)

        A = build_conv_big_mat(x_shape, w_arr, y_shape)
        N, C_in, H, W = x_shape
        _, C_out, H_out, W_out = y_shape
        in_feat = C_in * H * W
        out_feat = C_out * H_out * W_out

        # ----- create new initializers -----
        A_name = f"{name}.bigA"
        A_init = numpy_helper.from_array(A, name=A_name)

        shape_out_name = f"{name}.out_shape"
        shape_out_init = helper.make_tensor(
            shape_out_name, TensorProto.INT64, [4],
            [N, C_out, H_out, W_out]
        )

        graph.initializer.extend([A_init, shape_out_init])

        # ----- new nodes: Flatten -> MatMul -> Reshape -> (Add) -----
        flat_name = f"{name}.X_flat"
        flatten_node = helper.make_node(
            "Flatten",
            inputs=[X_name],
            outputs=[flat_name],
            name=f"{name}.Flatten",
            axis=1,   # [N, C,H,W] -> [N, C*H*W]
        )

        mm_name = f"{name}.Y_flat"
        matmul_node = helper.make_node(
            "MatMul",
            inputs=[flat_name, A_name],
            outputs=[mm_name],
            name=f"{name}.MatMul",
        )

        reshaped_name = Y_name
        reshape_node = helper.make_node(
            "Reshape",
            inputs=[mm_name, shape_out_name],
            outputs=[reshaped_name],
            name=f"{name}.Reshape",
        )

        if B_name is not None:
            # Add bias after reshape: broadcasting [C_out] over N,H_out,W_out
            add_out = Y_name
            add_node = helper.make_node(
                "Add",
                inputs=[reshaped_name, B_name],
                outputs=[add_out],
                name=f"{name}.BiasAdd",
            )
            new_nodes.extend([flatten_node, matmul_node, reshape_node, add_node])
        else:
            new_nodes.extend([flatten_node, matmul_node, reshape_node])

        to_remove.append(conv)

    # Rebuild node list
    for n in to_remove:
        graph.node.remove(n)
    graph.ClearField("node")
    graph.node.extend(new_nodes)

    return model


def fold_const_matmul_reshapes(model):
    """
    For every MatMul node where input[1] (B) is produced by a Reshape whose
    data input is an initializer, pre-apply the reshape and make B a direct
    initializer input to the MatMul.

    Pattern:

        init_W --(Reshape)--> W_reshaped --(MatMul)--> Y

    becomes:

        init_W_reshaped --(MatMul)--> Y

    and the Reshape node is removed (if not used elsewhere).
    """
    model = copy.deepcopy(model)
    graph = model.graph

    # Helper maps
    init_map = {init.name: init for init in graph.initializer}
    output_to_node = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    nodes_to_remove = []

    for node in graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue

        B_name = node.input[1]

        # If B is already an initializer, nothing to do.
        if B_name in init_map:
            continue

        # If B is output of a Reshape node, consider folding.
        reshape_node = output_to_node.get(B_name)
        if reshape_node is None or reshape_node.op_type != "Reshape":
            continue

        # Data input to Reshape must be an initializer.
        data_name = reshape_node.input[0]
        if data_name not in init_map:
            continue

        data_init = init_map[data_name]
        data_arr = numpy_helper.to_array(data_init)

        # Determine desired shape:
        # Prefer shape initializer if present, else fall back to inferred shape.
        shape_input = reshape_node.input[1] if len(reshape_node.input) > 1 else None
        new_shape = None

        if shape_input is not None and shape_input in init_map:
            shape_arr = numpy_helper.to_array(init_map[shape_input])
            new_shape = tuple(int(x) for x in shape_arr)
        else:
            # Fallback: use inferred shape of Reshape output.
            try:
                new_shape = tuple(get_tensor_shape(model, B_name))
            except Exception:
                new_shape = None

        if new_shape is None:
            continue

        if int(np.prod(new_shape)) != data_arr.size:
            # Shape mismatch; skip
            continue

        # Pre-apply reshape
        reshaped_arr = data_arr.reshape(new_shape)

        # Create a new initializer for the reshaped weights
        base_name = reshape_node.name or f"{data_name}_reshaped_for_{node.name}"
        new_init_name = base_name + "_constB"
        existing_init_names = {init.name for init in graph.initializer}
        if new_init_name in existing_init_names:
            i = 0
            base = new_init_name
            while new_init_name in existing_init_names:
                i += 1
                new_init_name = f"{base}_{i}"

        new_init = numpy_helper.from_array(
            reshaped_arr.astype(np.float32), name=new_init_name
        )
        graph.initializer.append(new_init)

        # Rewire MatMul to use the new initializer directly
        node.input[1] = new_init_name

        # Mark reshape node for removal (we don't aggressively clean up its
        # shape initializer; that's harmless).
        nodes_to_remove.append(reshape_node)

    # Remove folded Reshape nodes
    for n in nodes_to_remove:
        if n in graph.node:
            graph.node.remove(n)

    return model


def check_models_equivalent(orig_model, new_model, atol=1e-4, rtol=1e-4):
    """Run both models on a random input and compare outputs."""
    # Assume single input, single output
    in_name = orig_model.graph.input[0].name
    in_shape = [d.dim_value for d in orig_model.graph.input[0].type.tensor_type.shape.dim]

    sess_orig = ort.InferenceSession(orig_model.SerializeToString(), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(new_model.SerializeToString(), providers=["CPUExecutionProvider"])

    x = np.random.randn(*in_shape).astype(np.float32)

    y_orig = sess_orig.run(None, {in_name: x})[0]
    y_new = sess_new.run(None, {in_name: x})[0]

    if y_orig.shape != y_new.shape:
        raise RuntimeError(f"Output shapes differ: {y_orig.shape} vs {y_new.shape}")

    diff = np.abs(y_orig - y_new)
    max_diff = diff.max()
    print(f"max |orig - new| = {max_diff:.6g}")

    if not np.allclose(y_orig, y_new, atol=atol, rtol=rtol):
        raise RuntimeError("Models are NOT numerically equivalent within tolerance.")
    print("Models are numerically equivalent within tolerance.")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_model", help="Path to original ONNX model")
    ap.add_argument("output_model", help="Where to save transformed model")
    args = ap.parse_args()

    orig = onnx.load(args.input_model)

    # 1) Replace Convs with big MatMuls.
    new = replace_conv_with_big_matmul(orig)

    # 2) Fold constant Reshape->MatMul patterns into direct constant B inputs.
    new = fold_const_matmul_reshapes(new)

    # 3) Run shape inference on the final model and save.
    model_simp = shape_inference.infer_shapes(new)
    # model_simp, _ = simplify(new)
    onnx.save(model_simp, args.output_model)
    print(f"Saved transformed model to {args.output_model}")

    # 4) Check numerical equivalence between original and transformed.
    check_models_equivalent(orig, new)


if __name__ == "__main__":
    main()

