from ir import LinearCombo, VarType
from onnx import helper, TensorProto
import numpy as np

from itertools import product

def wire_matmul(r1cs, node, graph):
    inputs = node.input
    name = node.name
    A_meta = r1cs.nodes.get(inputs[0])
    B_meta = r1cs.nodes.get(inputs[1])
    output_id = node.output[0]
    unsq_A_name = f"{name}.A_exp"
    unsq_B_name = f"{name}.B_exp"
    C_hadamard_name = f"{name}.C_hadamard"
    axes_A = helper.make_tensor(f"{name}.axes_A", TensorProto.INT64, [1], [len(A_meta.shape)])
    axes_B = helper.make_tensor(f"{name}.axes_B", TensorProto.INT64, [1], [len(B_meta.shape) - 2])
    axes_reduce = helper.make_tensor("{name}.axes_reduce", TensorProto.INT64, [1], [len(A_meta.shape) - 1 ])
    unsq_A_node = helper.make_node("Unsqueeze", [A_meta.node.output[0], axes_A.name], [unsq_A_name], name=unsq_A_name)
    unsq_B_node = helper.make_node("Unsqueeze", [B_meta.node.output[0], axes_B.name], [unsq_B_name], name=unsq_B_name)
    mul_node = helper.make_node("Mul", [unsq_A_name, unsq_B_name], [C_hadamard_name], name=C_hadamard_name)
    output_node = helper.make_node(
        "ReduceSum",
        inputs=[C_hadamard_name, axes_reduce.name],
        outputs=[output_id],
        name=output_id,
        keepdims=0,
        noop_with_empty_axes=0,
    )
    graph.node.remove(node)
    graph.node.append(output_node)
    graph.initializer.extend([axes_A, axes_B, axes_reduce])
    graph.node.extend([unsq_A_node, unsq_B_node, mul_node])
    return [mul_node]

# Yields all indices for a tensor of the given shape
def iterate_multi_index(shape):
    if len(shape) == 0:
        yield ()
    else:
        yield from product(*(range(s) for s in shape))


def constrain_matmul(r1cs, rnode):
    inputs = rnode.node.input
    A_meta = r1cs.nodes.get(inputs[0])
    B_meta = r1cs.nodes.get(inputs[1])
    B = r1cs.get_output(B_meta)
    A = r1cs.get_output(A_meta)

    if len(A.shape) < 2 or len(B.shape) < 2:
        raise ValueError("Both A and B must have at least 2 dimensions (matrix dims).")

    A_M, A_K = A.shape[-2], A.shape[-1]
    B_K, B_N = B.shape[-2], B.shape[-1]

    if A_K != B_K:
        raise ValueError(
            "Inner dimensions do not match: A's columns (%d) != B's rows (%d)"
            % (A_K, B_K)
        )

    A_batch = A.shape[:-2]
    B_batch = B.shape[:-2]
    if A_batch != B_batch:
        missing_dims = len(A_batch) - len(B_batch)
        B = B.reshape(A_batch[:missing_dims] + B.shape)

    batch_shape = A_batch
    M, K, N = A_M, A_K, B_N

    if not A_meta.in_wit:
        A_prev = A
        A = r1cs.allocate_tensor(A.shape)
        for idx in np.ndindex(A_prev.shape):
            rnode.append_labeled_constraint(
                A_prev[idx],
                LinearCombo.one(),
                A[idx],
                f"{rnode.node.name}_A_{'_'.join(map(str, idx))}",
            )

    if not B_meta.in_wit and B_meta.is_var:
        B_prev = B
        B = r1cs.allocate_tensor(B_meta.shape)
        for idx in np.ndindex(B_prev.shape):
            rnode.append_labeled_constraint(
                B_prev[idx],
                LinearCombo.one(),
                B[idx],
                f"{rnode.node.name}_B_{'_'.join(map(str, idx))}",
            )

    if B_meta.is_var and not r1cs.full_variable_matmul:
        C = r1cs.allocate_tensor(rnode.shape, rnode.var_type, virtual=True)
        B_r = r1cs.allocate_secondary_constraint_tensor(B.shape[:-1])
        for batch_idx in iterate_multi_index(batch_shape):
            # Add all constraints of the form B_r - B * r = 0
            for i in range(K):
                B_i_dot_r = LinearCombo(
                    [(j, B[batch_idx + (i, j)].get_single_var()) for j in range(N)]
                )
                rnode.append_secondary_constraint(
                    LinearCombo.from_const(B_r[batch_idx + (i,)]),
                    LinearCombo.one(),
                    B_i_dot_r,
                    f"{rnode.node.name}_B_r_{'_'.join(map(str, batch_idx))}_{i}",
                )
            # Add all constraints of the form A * B_r - C * r = 0
            for i in range(M):
                C_i_dot_r = LinearCombo(
                    [(j, C[batch_idx + (i, j)].get_single_var()) for j in range(N)]
                )
                A_i_dot_B_r = LinearCombo(
                    [
                        (
                            B_r[batch_idx + (j,)],
                            A[batch_idx + (i, j)].get_single_var(),
                        )
                        for j in range(K)
                    ]
                )
                rnode.append_secondary_constraint(
                    A_i_dot_B_r,
                    LinearCombo.one(),
                    C_i_dot_r,
                    f"{rnode.node.name}_A_i_dot_B_r_{'_'.join(map(str, batch_idx))}_{i}",
                )
    elif B_meta.is_var and r1cs.full_variable_matmul:
        C_hadamard = r1cs.allocate_tensor(A.shape + (B_N,), virtual=True)
        C = r1cs.allocate_tensor(rnode.shape, rnode.var_type, virtual=True)
        constraints = 0
        for batch_idx in iterate_multi_index(batch_shape):
            for i in range(A_M):
                for k in range(A_K):
                    a_ik = A[batch_idx + (i, k)]
                    for j in range(B_N):
                        b_kj = B[batch_idx + (k, j)]
                        ch = C_hadamard[batch_idx + (i, k, j)]
                        rnode.append_labeled_constraint(
                            a_ik,
                            b_kj,
                            ch,
                            f"{rnode.node.name}_had_{'_'.join(map(str, batch_idx))}_{i}_{k}_{j}",
                        )
                        constraints += 1
                        if constraints % 1000 == 0:
                            r1cs.process_constrained_node(rnode)

            for i in range(A_M):
                for j in range(B_N):
                    sum_over_k = LinearCombo.zero()
                    for k in range (A_K):
                        sum_over_k +=  C_hadamard[batch_idx + (i, k, j)] 
                    rnode.append_labeled_constraint(
                        sum_over_k,
                        LinearCombo.one(),
                        C[batch_idx + (i, j)],
                        f"{rnode.node.name}_sum_eq_C_{'_'.join(map(str, batch_idx))}_{i}_{j}",
                    )
                    constraints += 1
                    if constraints % 1000 == 0:
                        r1cs.process_constrained_node(rnode)

    else:
        C = r1cs.allocate_tensor(rnode.shape, rnode.var_type, virtual=True)
        r_t_A = r1cs.allocate_tensor(A.shape[:-2] + A.shape[-1:], VarType.SECONDARY)
        constraints = 0
        for batch_idx in iterate_multi_index(batch_shape):
            # Add all constraints of the form r_t_A - r^t * A  = 0
            for j in range(A_K):
                r_t_A_i = LinearCombo(
                    [(i, A[batch_idx + (i, j)].get_single_var()) for i in range(M)]
                )
                rnode.append_secondary_constraint(
                    r_t_A[batch_idx + (j,)],
                    LinearCombo.one(),
                    r_t_A_i,
                    f"{rnode.node.name}_r_t_A_{'_'.join(map(str, batch_idx))}_{j}",
                )
                constraints += 1 
                if constraints % 1000 == 0:
                    r1cs.process_constrained_node(rnode)
            # Add all constraints of the form r_t_A * B - r_t_C = 0
            for j in range(N):
                r_t_C = LinearCombo(
                    [(i, C[batch_idx + (i, j)].get_single_var()) for i in range(M)]
                )
                r_t_A_dot_B = LinearCombo(
                    [
                        (
                            B[batch_idx + (k, j)],
                            r_t_A[batch_idx + (k,)].get_single_var(),
                        )
                        for k in range(K)
                    ]
                )
                rnode.append_secondary_constraint(
                    r_t_A_dot_B,
                    LinearCombo.one(),
                    r_t_C,
                    f"{rnode.node.name}_r_t_A_dot_B_{'_'.join(map(str, batch_idx))}_{j}",
                )
                constraints += 1 
                if constraints % 1000 == 0:
                    r1cs.process_constrained_node(rnode)
    if len(rnode.successors) > 0:
        rnode.output = C.materialize()
