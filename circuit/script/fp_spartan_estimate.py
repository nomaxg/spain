from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict
from numbers import Integral
from math import prod
from typing import Dict, Tuple
from eval import load_onnx_model
import graph

class OpType(Enum):
    ADD = auto()
    SUB = auto()
    LT = auto()
    GT = auto()
    SQRT = auto()
    LEQ = auto()
    GEQ = auto()
    MUL = auto()
    INIT = auto()
    DIV = auto()
    EQ = auto()


@dataclass
class EstimateInput:
    # size of witness before inflating with fixed point op costs
    base_witness_size: int
    op_count: Dict[OpType, int]


# Estimates derived from table 1 in https://eprint.iacr.org/2024/1842.pdf
def estimate_costs(est: EstimateInput) -> Tuple[int, int]:
    witness_size = est.base_witness_size
    num_cons = 0
    for op_type, count in est.op_count.items():
        cons_increase = 0
        if (
            op_type == OpType.ADD
            or op_type == OpType.SUB
        ):
            cons_increase += 85
        elif op_type == OpType.MUL:
            cons_increase += 64
        elif op_type == OpType.DIV:
            cons_increase += 76 
        elif op_type == OpType.SQRT:
            cons_increase += 45
        elif (op_type == OpType.EQ or 
              op_type == OpType.LT or 
              op_type == OpType.GT or 
              op_type == OpType.LEQ or 
              op_type == OpType.GEQ):
            cons_increase += 33
        elif (op_type == OpType.INIT):
            cons_increase += 30
        else:
            raise ValueError(f"Unsupported OpType for estimation: {op_type}")
        num_cons += cons_increase * count
    return witness_size, num_cons


# Estimate MatMul counts of X \in R^(m x n) and Y \in R^(n x p) with Freivalds (Zr = XYr)
def get_matmul_counts_freivalds(m: int, n: int, p: int) -> EstimateInput:
    base_witness_size = m * n + n * p + m * p + p  # X, Y, Z, r
    op_count = {
        OpType.MUL: p * n + n * m + p * m,  # X * r,  Y * (X * r),  Z * r,
        OpType.ADD: (p - 1) * n  # X * r
        + (n - 1) * m  # Y * (X * r)
        + (p - 1) * m,  # Z * r
        OpType.EQ: m,  # Z * r == Y * (X * r)
    }
    return EstimateInput(base_witness_size=base_witness_size, op_count=op_count)

# Estimate MatMul counts of full variable MatMul (no Freivalds trick)
def get_matmul_counts(m: int, n: int, p: int) -> EstimateInput:
    base_witness_size = m * n + n * p
    op_count = {
        OpType.MUL: n * p * m, 
        OpType.ADD: (m * p) * (n - 1)
    }
    return EstimateInput(base_witness_size=base_witness_size, op_count=op_count)

# Var * var multiplication, both tensors \in m x n
def estimate_mul(m: int, n: int) -> EstimateInput:
    base_witness_size = m * n * 2
    op_count = {
        OpType.MUL: m *n 
    }
    return EstimateInput(base_witness_size=base_witness_size, op_count=op_count)


# Use approx Erf under the hood
def get_gelu_counts(m: int, n: int, is_first = False) -> EstimateInput:
    base_witness_size = m * n * 2  # X, Y
    op_count = defaultdict(int)
    # Erf approximation
    op_count[OpType.GT] += m * n  # x > 1
    op_count[OpType.LT] += m * n  # x < -1
    op_count[OpType.MUL] += m * n # (1 - (x > 1)) * (1 - (x < -1))
    op_count[OpType.ADD] += (m * n)*3 # add terms in erf approx
    # Another mul by x
    op_count[OpType.MUL] += m * n
    # Init constraints
    if is_first:
        op_count[OpType.INIT] += m * n 
    return EstimateInput(base_witness_size=base_witness_size, op_count=op_count)

# Use approx Erf under the hood
def get_erf_counts(m: int, n: int) -> EstimateInput: 
    base_witness_size = m * n * 2  # X, Y
    op_count = defaultdict(int)
    # Erf approximation
    op_count[OpType.GT] += m * n  # x > 1
    op_count[OpType.LT] += m * n  # x < -1
    op_count[OpType.MUL] += m * n # (1 - (x > 1)) * (1 - (x < -1))
    op_count[OpType.ADD] += (m * n)*3 # add terms in erf approx
    return EstimateInput(base_witness_size=base_witness_size, op_count=op_count)

# Estimate LayerNorm counts of Y = LayerNorm(X) where X \in R^(m x n), normalization over column dim
def get_layernorm_counts(m: int, n: int, is_first=False) -> EstimateInput:
    base_witness_size = m * n * 2  # X, Y
    op_count = defaultdict(int)
    # Mean across last axis (sum)
    op_count[OpType.ADD] += m * (n - 1)  # Sum = ReduceMean(X)
    # Subtract by mean
    op_count[OpType.SUB] += m * n  # D = X - Mean
    # Square
    op_count[OpType.MUL] += m * n  # DD = D * D
    # Variance (sum across axis)
    op_count[OpType.ADD] += m * (n - 1)  # Var = ReduceMean(DD)
    # Sqrt to get std deviation
    op_count[OpType.SQRT] += m  # StdDev = sqrt(VarEps)
    # Inverse std
    op_count[OpType.DIV] += m  # StdInv = 1 / StdDev
    # Normalize
    op_count[OpType.MUL] += m * n  # Norm = D * StdInv
    # Init Constraints
    if is_first:
        op_count[OpType.INIT] += m * n 
    return EstimateInput(base_witness_size=base_witness_size, op_count=op_count)


# Estimate Softmax counts of Y = Softmax(X) where X \in R^(m x n), normalization over column dim
# Uses same Pade approximation notion that we do
def get_softmax_counts(m: int, n: int, is_first=False) -> EstimateInput:
    base_witness_size = m * n * 2  # X, Y
    op_count = defaultdict(int)
    # Rational approximation of e^x using degree 3 rational poly
    # X^2
    op_count[OpType.MUL] += m * n  # x^2
    # x^3
    op_count[OpType.MUL] += m * n  # x^3
    # Numerator + denom for rational approx
    op_count[OpType.MUL] += (m * n * 3) * 2  # a1*x, a2*x^2, a3*x^3
    op_count[OpType.ADD] += (m * n * 3) * 2  # a0 + a1*x + a2*x^2 + a3*x^3
    op_count[OpType.DIV] += m * n  # Exp = num/denom
    # Sum across last axis
    op_count[OpType.ADD] += m * (n - 1)  # Sum = ReduceSum(Exp)
    # Get scale as 1/Sum
    op_count[OpType.DIV] += m  # Scale = 1 / Sum
    # Scale Exp by Scale
    op_count[OpType.MUL] += m * n  # Softmax = Exp * Scale
    # Init constraints
    if is_first:
        op_count[OpType.INIT] += m * n
    return EstimateInput(base_witness_size=base_witness_size, op_count=op_count)

def normalize_shape(shape):
    return tuple(int(dim) for dim in shape)


def split_batch_dims(shape):
    norm = normalize_shape(shape)
    if len(norm) < 2:
        raise ValueError("Expected shape to have at least 2 dimensions")
    return tuple(norm[:-2]), (norm[-2], norm[-1])

def matmul_dims(a_shape, b_shape):
    a_norm = normalize_shape(a_shape)
    b_norm = normalize_shape(b_shape)
    if (
        a_norm is None
        or b_norm is None
        or len(a_norm) < 2
        or len(b_norm) < 2
    ):
        return None
    batch_shape = a_norm[:-2]
    m, k_a = a_norm[-2], a_norm[-1]
    k_b, n = b_norm[-2], b_norm[-1]
    if k_a != k_b:
        return None
    return batch_shape, m, k_a, n

def batch_multiplier(batch_shape):
    if not batch_shape:
        return 1
    return prod([int(dim) for dim in batch_shape])

def estimate_model(path):
    model = load_onnx_model(path)
    r1cs = graph.get_r1cs(model, "", False)
    total_constraints = 0
    for name, node in r1cs.nodes.items():
        op = getattr(node, "op", "Unknown")
        if op not in {"MatMul", "Softmax", "Erf", "LayerNormalization"}:
            continue
        raw_shape = node.shape
        batch_dims, out_dims = split_batch_dims(raw_shape)
        in_dims = []
        constraints = None
        matmul_type = None
        if op == "Softmax" and out_dims and len(out_dims) == 2:
            m, n = out_dims
            in_dims = [(m, n)]
            est_input = get_softmax_counts(m, n, False)
            _, constraints = estimate_costs(est_input)
        elif op == "Erf" and out_dims and len(out_dims) == 2:
            m, n = out_dims
            in_dims = [(m, n)]
            est_input = get_erf_counts(m, n)
            _, constraints = estimate_costs(est_input)
        elif op == "LayerNormalization" and out_dims and len(out_dims) == 2:
            m, n = out_dims
            in_dims = [(m, n)]
            est_input = get_layernorm_counts(m, n, False)
            _, constraints = estimate_costs(est_input)
        elif op == "MatMul":
            inputs = node.node.input
            A_meta = r1cs.nodes.get(inputs[0])
            B_meta = r1cs.nodes.get(inputs[1])
            dims = matmul_dims(A_meta.shape, B_meta.shape)
            if dims:
                batch_shape, m, k, n = dims
                batch_mult = batch_multiplier(batch_shape)
                batch_dims = batch_shape
                in_dims = [(m, k), (k, n)]
                out_dims = (m, n)
                if B_meta.is_var:
                    est_input = get_matmul_counts(m, k, n)
                else:
                    est_input = get_matmul_counts_freivalds(m, k, n)
                _, base_cons = estimate_costs(est_input)
                constraints = base_cons * batch_mult
        if constraints is not None:
            total_constraints += constraints
    return total_constraints

if __name__ == "__main__":
    m = 32
    n = 32
    est_input_softmax = get_softmax_counts(m, n, True)
    _, softmax_num_constraints = estimate_costs(est_input_softmax)

    m = 32
    n = 768
    est_input_layernorm = get_layernorm_counts(m, n, True)
    _, layernorm_num_constraints = estimate_costs(est_input_layernorm)

    m = 32
    n = 3072
    est_input_gelu = get_gelu_counts(m, n, True)
    _, gelu_num_constraints = estimate_costs(est_input_gelu)

    gpt2_seq_2_path = "./export/gpt2-seq-2/original_model.onnx"
    gpt2_seq_32_path = "./export/gpt2-seq-32/original_model.onnx"
    num_gpt2_seq_2_cons = estimate_model(gpt2_seq_2_path)
    num_gpt2_seq_32_cons = estimate_model(gpt2_seq_32_path)

    print("\n")
    print("FP-Spartan constraint estimates:")
    print(f"Softmax 32x32: {softmax_num_constraints} constraints")
    print(f"LayerNorm 32x768: {layernorm_num_constraints} constraints")
    print(f"Gelu 32x3072: {gelu_num_constraints} constraints")
    print(f"GPT-2 seq 2: {num_gpt2_seq_2_cons} constraints")
    print(f"GPT-2 seq 32: {num_gpt2_seq_32_cons} constraints")
    
    
