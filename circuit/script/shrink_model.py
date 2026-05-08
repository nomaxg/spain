# Hacky AI script to filter baseline models by smaller sequence lenghts, for our scaling experiment.
# May not be necessary when we export directly from the openai hugging face model, which has tunable sequence lengths,
# Open ai's GPT-2 is not currently suppported because it has slightly different operations than the GPT-2 exported by the NanoGpt repo
# (i.e. Gemm instead of Softmax, some Where/IsNaN guardrail nodes for infinities).

#!/usr/bin/env python3
# shrink_ctx_to_target.py
# Usage:
#   python shrink_ctx_to_target.py INPUT.onnx OUTPUT.onnx --seq-in 64 --seq-out 32 [--verify]

import sys
import argparse
import onnx
import numpy as np
from onnx import numpy_helper, shape_inference

S = 64
R = 32


def rewrite_dims_S_to_R(model: onnx.ModelProto, S=64, R=32):
    def touch_vi(vi: onnx.ValueInfoProto):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value") and d.dim_value == S:
                    d.dim_value = R

    for vi in model.graph.input:
        touch_vi(vi)
    for vi in model.graph.output:
        touch_vi(vi)
    for vi in model.graph.value_info:
        touch_vi(vi)


def rewrite_input_seq_dims(model: onnx.ModelProto, S=64, R=32):
    # Only touch model inputs to set seq length; let inference propagate.
    for vi in model.graph.input:
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            dims = vi.type.tensor_type.shape.dim
            for i, d in enumerate(dims):
                if d.HasField("dim_value") and d.dim_value == S:
                    d.dim_value = R


def wipe_all_value_info_shapes_except_inputs(model: onnx.ModelProto):
    def wipe_vi(vi: onnx.ValueInfoProto):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    d.ClearField("dim_value")
                if d.HasField("dim_param"):
                    d.ClearField("dim_param")

    # Do NOT wipe inputs; serializer needs concrete non-zero dims there
    for vi in model.graph.output:
        wipe_vi(vi)
    for vi in model.graph.value_info:
        wipe_vi(vi)


def shrink_any_axis_S_or_S1(arr: np.ndarray, S=64, R=32) -> np.ndarray:
    if arr.size == 0:
        return arr
    slices = []
    changed = False
    for d in arr.shape:
        if d == S:
            slices.append(slice(0, R))
            changed = True
        elif d == S + 1:
            slices.append(slice(0, R + 1))
            changed = True
        else:
            slices.append(slice(0, d))
    return arr[tuple(slices)] if changed else arr


def shrink_initializer_safely(t: onnx.TensorProto, S=64, R=32) -> onnx.TensorProto:
    arr = numpy_helper.to_array(t)
    if arr.size == 0:
        return t
    shape = arr.shape
    rank = arr.ndim

    # 1D tensors (bias, layernorm gamma/beta): shrink if length matches hidden size
    if rank == 1 and shape[0] == S:
        arr2 = arr[:R]
        return numpy_helper.from_array(arr2, name=t.name)

    # 2D tensors (weights): shrink hidden-size dimensions only
    if rank == 2:
        h_out, h_in = shape
        # Square hidden-size matrices
        if h_out == S and h_in == S:
            arr2 = arr[:R, :R]
            return numpy_helper.from_array(arr2, name=t.name)
        # Common case: MatMul weight [out, in]; shrink in-dim or out-dim if it equals S
        if h_in == S:
            arr2 = arr[:, :R]
            return numpy_helper.from_array(arr2, name=t.name)
        if h_out == S:
            arr2 = arr[:R, :]
            return numpy_helper.from_array(arr2, name=t.name)
        return t

    # 3D/4D+ tensors (e.g., attention masks/scores with seq length). Skip to avoid seq-len mismatch.
    return t


def shrink_initializer_all_axes(t: onnx.TensorProto, S=64, R=32) -> onnx.TensorProto:
    arr = numpy_helper.to_array(t)
    arr2 = shrink_any_axis_S_or_S1(arr, S, R)
    if arr2.shape == arr.shape:
        return t
    return numpy_helper.from_array(arr2, name=t.name)


def shrink_initializer_seq_only(t: onnx.TensorProto, S=64, R=32) -> onnx.TensorProto:
    arr = numpy_helper.to_array(t)
    if arr.size == 0:
        return t
    shape = arr.shape
    rank = arr.ndim

    # Heuristics to shrink only sequence-length axes, not hidden size.

    # Rank-1: could be position ids [S] or [S+1]. Shrink when matching.
    if rank == 1:
        if shape[0] == S:
            arr2 = arr[:R]
            return numpy_helper.from_array(arr2, name=t.name)
        if shape[0] == S + 1:
            arr2 = arr[: R + 1]
            return numpy_helper.from_array(arr2, name=t.name)
        return t

    # Rank-2:
    # - Positional embeddings: [S, C] or [S+1, C] -> shrink axis 0
    # - Attention bias like [1, S] or [S, 1] -> shrink axis with S if not both S
    # - Skip [S, S] because that is ambiguous (could be hidden-size square weight)
    if rank == 2:
        d0, d1 = shape
        if d0 == S and d1 != S:
            arr2 = arr[:R, :]
            return numpy_helper.from_array(arr2, name=t.name)
        if d0 == S + 1 and d1 != S + 1:
            arr2 = arr[: R + 1, :]
            return numpy_helper.from_array(arr2, name=t.name)
        if d1 == S and d0 != S:
            arr2 = arr[:, :R]
            return numpy_helper.from_array(arr2, name=t.name)
        if d1 == S + 1 and d0 != S + 1:
            arr2 = arr[:, : R + 1]
            return numpy_helper.from_array(arr2, name=t.name)
        return t

    # Rank-3: Common activations [B, S, C] -> shrink axis 1 if equals S (rare for initializers)
    if rank == 3:
        if shape[1] == S:
            slices = [slice(0, shape[0]), slice(0, R), slice(0, shape[2])]
            arr2 = arr[tuple(slices)]
            return numpy_helper.from_array(arr2, name=t.name)
        if shape[1] == S + 1:
            slices = [slice(0, shape[0]), slice(0, R + 1), slice(0, shape[2])]
            arr2 = arr[tuple(slices)]
            return numpy_helper.from_array(arr2, name=t.name)
        return t

    # Rank-4: Attention scores/masks [B, H, S, S] or [B, 1, 1, S] (and S+1 variants)
    if rank == 4:
        b, h, d2, d3 = shape
        changed = False
        s0 = slice(0, b)
        s1 = slice(0, h)
        s2 = slice(0, d2)
        s3 = slice(0, d3)
        if d2 == S:
            s2 = slice(0, R)
            changed = True
        elif d2 == S + 1:
            s2 = slice(0, R + 1)
            changed = True
        if d3 == S:
            s3 = slice(0, R)
            changed = True
        elif d3 == S + 1:
            s3 = slice(0, R + 1)
            changed = True
        if changed:
            arr2 = arr[(s0, s1, s2, s3)]
            return numpy_helper.from_array(arr2, name=t.name)
        return t

    # Higher ranks: generic fallback
    arr2 = shrink_any_axis_S_or_S1(arr, S, R)
    if arr2.shape != arr.shape:
        return numpy_helper.from_array(arr2, name=t.name)
    return t


def rewrite_reshape_shapes(model: onnx.ModelProto, S=64, R=32):
    # Update the shape tensors for Reshape nodes (second input or Constant attr) replacing S/S+1 -> R/R+1
    # Build a map of initializers by name for quick lookup
    init_by_name = {init.name: init for init in model.graph.initializer}

    def rewrite_shape_tensor(tproto: onnx.TensorProto) -> onnx.TensorProto:
        arr = numpy_helper.to_array(tproto)
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
            arr2 = arr.copy()
            arr2[arr2 == S] = R
            arr2[arr2 == S + 1] = R + 1
            if not np.array_equal(arr2, arr):
                return numpy_helper.from_array(arr2.astype(arr.dtype), name=tproto.name)
        return tproto

    # Rewrite initializers used as reshape shape
    for node in model.graph.node:
        if node.op_type == "Reshape" and len(node.input) >= 2:
            shape_name = node.input[1]
            if shape_name in init_by_name:
                new_t = rewrite_shape_tensor(init_by_name[shape_name])
                if new_t is not init_by_name[shape_name]:
                    # replace in initializer list
                    for i, init in enumerate(model.graph.initializer):
                        if init.name == shape_name:
                            model.graph.initializer[i].CopyFrom(new_t)
                            init_by_name[shape_name] = new_t
                            break
        elif node.op_type == "Constant":
            # Also handle Constant nodes that carry a shape vector used by Reshape
            for i, attr in enumerate(node.attribute):
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    new_t = rewrite_shape_tensor(attr.t)
                    if new_t is not attr.t:
                        node.attribute[i].t.CopyFrom(new_t)


def verify_no_remaining_S(model: onnx.ModelProto, S=64):
    offenders = []

    def check_arr(name, arr):
        if any(dim == S or dim == S + 1 for dim in arr.shape):
            offenders.append((name, arr.shape))

    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        check_arr(init.name, arr)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    arr = numpy_helper.to_array(attr.t)
                    check_arr(node.name or "Constant", arr)

    # ValueInfo dims
    def dims_of(vi):
        if not vi.type.HasField("tensor_type") or not vi.type.tensor_type.HasField(
            "shape"
        ):
            return []
        vals = []
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                vals.append(d.dim_value)
            else:
                vals.append(None)
        return vals

    for vi in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        vals = dims_of(vi)
        if any(v == S or v == S + 1 for v in vals if v is not None):
            offenders.append((vi.name, tuple(v for v in vals if v is not None)))
    return offenders


def shrink_model(inp, out, S=64, R=32):
    model = onnx.load(inp)
    rewrite_input_seq_dims(model, S, R)
    wipe_all_value_info_shapes_except_inputs(model)

    # Shrink initializers: shrink ALL axes equal to S/S+1
    new_inits = [
        shrink_initializer_all_axes(init, S, R) for init in model.graph.initializer
    ]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(new_inits)

    # Rewrite Reshape shape tensors that still encode S or S+1.
    rewrite_reshape_shapes(model, S, R)

    # Shrink Constant node payloads with ALL-axes rule as well.
    for node in model.graph.node:
        if node.op_type == "Constant":
            for i, attr in enumerate(node.attribute):
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    new_t = shrink_initializer_all_axes(attr.t, S, R)
                    if new_t is not attr.t:
                        node.attribute[i].t.CopyFrom(new_t)

    # Infer shapes non-strict.
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False)
        model = inferred
    except Exception as e:
        print(f"[warn] shape inference skipped due to: {e}")

    onnx.save(model, out)
    print(f"Saved shrunk model to: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inp")
    parser.add_argument("outp")
    parser.add_argument("--seq-in", type=int, default=64)
    parser.add_argument("--seq-out", type=int, default=32)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    S = args.seq_in
    R = args.seq_out
    shrink_model(args.inp, args.outp, S, R)


if __name__ == "__main__":
    main()
