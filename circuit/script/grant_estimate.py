from dataclasses import dataclass
from enum import Enum, auto
from collections import defaultdict
from typing import Dict, Tuple


class OpType(Enum):
    ADD = auto()
    SUB = auto()
    LT = auto()
    GT = auto()
    SQRT = auto()
    LEQ = auto()
    GEQ = auto()
    MUL = auto()
    DIV = auto()
    EQ = auto()


@dataclass
class EstimateInput:
    # size of witness before inflating with fixed point op costs
    base_witness_size: int
    op_count: Dict[OpType, int]


def estimate_costs(est: EstimateInput) -> Tuple[int, int]:
    witness_size = est.base_witness_size
    num_cons = 0
    for op_type, count in est.op_count.items():
        cons_increase = 0
        wit_increase = 0
        if (
            op_type == OpType.ADD
            or op_type == OpType.SUB
            or op_type == OpType.LEQ
            or op_type == OpType.GEQ
            or op_type == OpType.GT
            or op_type == OpType.LT
        ):
            cons_increase = 64
            wit_increase = 64
        elif op_type == OpType.MUL or op_type == OpType.SQRT:
            cons_increase = 128
            wit_increase = 128
        elif op_type == OpType.DIV:
            cons_increase = 256
            wit_increase = 256
        elif op_type == OpType.EQ:
            cons_increase = 1
            wit_increase = 0
        witness_size += wit_increase * count
        num_cons += cons_increase * count
    return witness_size, num_cons


# Estimate MatMul counts of X \in R^(m x n) and Y \in R^(n x p) with Frievalds (Zr = XYr)
def get_matmul_counts(m: int, n: int, p: int) -> EstimateInput:
    base_witness_size = m * n + n * p + m * p + p  # X, Y, Z, r
    op_count = {
        OpType.MUL: p * n + n * m + p * m,  # X * r,  Y * (X * r),  Z * r,
        OpType.ADD: (p - 1) * n  # X * r
        + (n - 1) * m  # Y * (X * r)
        + (p - 1) * m,  # Z * r
        OpType.EQ: m,  # Z * r == Y * (X * r)
    }
    return EstimateInput(base_witness_size=base_witness_size, op_count=op_count)


# Estimate LayerNorm counts of Y = LayerNorm(X) where X \in R^(m x n), normalization over column dim
def get_layernorm_counts(m: int, n: int) -> EstimateInput:
    base_witness_size = m * n * 2  # X, Y
    op_count = defaultdict(int)
    # Mean across last axis (sum)
    op_count[OpType.ADD] += m * (n - 1)  # Sum = ReduceMean(X)
    # Mean across last axis (scale)
    op_count[OpType.MUL] += m  # Mean = Sum * (1/n)
    # Subtract by mean
    op_count[OpType.SUB] += m * n  # D = X - Mean
    # Square
    op_count[OpType.MUL] += m * n  # DD = D * D
    # Variance (sum across axis)
    op_count[OpType.ADD] += m * (n - 1)  # Var = ReduceMean(DD)
    # Variance (scale)
    op_count[OpType.MUL] += m  # Var = Var * (1/n)
    # Add epsilon
    op_count[OpType.ADD] += m  # VarEps = Var + epsilon
    # Sqrt to get std deviation
    op_count[OpType.SQRT] += m  # StdDev = sqrt(VarEps)
    # Inverse std
    op_count[OpType.DIV] += m  # StdInv = 1 / StdDev
    # Normalize
    op_count[OpType.MUL] += m * n  # Norm = D * StdInv
    # Equality Constraints
    op_count[OpType.EQ] += m * n  # Y == Norm
    return EstimateInput(base_witness_size=base_witness_size, op_count=op_count)


# Estimate Softmax counts of Y = Softmax(X) where X \in R^(m x n), normalization over column dim
def get_softmax_counts(m: int, n: int) -> EstimateInput:
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
    # Equality Constraints
    op_count[OpType.EQ] += m * n  # Y == Softmax
    return EstimateInput(base_witness_size=base_witness_size, op_count=op_count)


if __name__ == "__main__":
    # Computation 1A: MatMul - GPT2
    m = 32
    n = 64
    p = 32
    est_input_matmul = get_matmul_counts(m, n, p)
    witness_size, num_constraints = estimate_costs(est_input_matmul)
    print(f"GPT2 MatMul (X \in {m}x{n} * Y \in {n}x{p}):")
    print(f"Estimated Number of Constraints: {num_constraints}")
    print(f"Estimated Witness Size: {witness_size}")
    # Computation 1B: MatMul
    m = 32
    n = 3072
    p = 768
    est_input_matmul = get_matmul_counts(m, n, p)
    witness_size, num_constraints = estimate_costs(est_input_matmul)
    print(f"Big MatMul (X \in {m}x{n} * Y \in {n}x{p}):")
    print(f"Estimated Number of Constraints: {num_constraints}")
    print(f"Estimated Witness Size: {witness_size}")
    # Computation 2: Softmax
    m = 32
    n = 32
    est_input_softmax = get_softmax_counts(m, n)
    witness_size, num_constraints = estimate_costs(est_input_softmax)
    print(f"Softmax (X \in {m}x{n}):")
    print(f"Estimated Number of Constraints: {num_constraints}")
    print(f"Estimated Witness Size: {witness_size}")
    # Computation 3: LayerNorm
    m = 32
    n = 768
    est_input_layernorm = get_layernorm_counts(m, n)
    witness_size, num_constraints = estimate_costs(est_input_layernorm)
    print(f"LayerNorm (X \in {m}x{n}):")
    print(f"Estimated Number of Constraints: {num_constraints}")
    print(f"Estimated Witness Size: {witness_size}")
