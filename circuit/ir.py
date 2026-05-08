from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np
from typing import TypeVar, Generic, List, Tuple
from enum import Enum

T = TypeVar("T", int, float)


class VarType(Enum):
    PUBLIC = "public"
    PRIMARY = "primary"
    SECONDARY = "secondary"

    to_str = lambda self: self.value

    def __lt__(self, other: "VarType") -> bool:
        order = {
            VarType.PUBLIC: 0,
            VarType.PRIMARY: 1,
            VarType.SECONDARY: 2,
        }
        return order[self] < order[other]


@dataclass(frozen=True, order=True)
class VarIdx:
    idx: int
    var_type: VarType

    @staticmethod
    def con() -> VarIdx:
        return VarIdx(0, VarType.PUBLIC)

    def __str__(self) -> str:
        return f"VarIdx({self.idx})"


def _flat(idx, shape):
    if len(idx) != len(shape):
        raise IndexError(f"Rank mismatch: idx={idx}, shape={shape}")
    flat = 0
    stride = 1
    for i, dim in zip(reversed(idx), reversed(shape)):
        if isinstance(i, int):
            if i < 0 or i >= dim:
                raise IndexError(f"Index {i} out of bounds (size {dim})")
            flat += i * stride
        elif isinstance(i, slice):
            if i != slice(None):
                raise NotImplementedError("Only full ':' slices supported in _flat")
        else:
            raise TypeError(f"Unsupported index type {type(i)} in _flat")
        stride *= dim
    return flat


class VirtualTensor:
    __slots__ = ("base", "shape", "var_type")

    def __init__(self, base, shape, var_type):
        self.base = base
        self.shape = tuple(shape)
        self.var_type = var_type

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (len(self.shape) - len(key))
        if len(key) > len(self.shape):
            raise IndexError(
                f"Too many indices {key} for tensor of rank {len(self.shape)}"
            )
        if all(isinstance(k, int) for k in key):
            flat_idx = _flat(key, self.shape)
            return LinearCombo.from_var(VarIdx(self.base + flat_idx, self.var_type))
        offset = 0
        stride = 1
        new_shape = []
        for k, dim in zip(reversed(key), reversed(self.shape)):
            if isinstance(k, int):
                if k < 0 or k >= dim:
                    raise IndexError(
                        f"Index {k} out of bounds for dimension size {dim}"
                    )
                offset += k * stride
            elif isinstance(k, slice) and k == slice(None):
                new_shape.insert(0, dim)
            else:
                raise NotImplementedError(
                    "Only full ':' slices and non-negative ints are supported"
                )
            stride *= dim

        return VirtualTensor(self.base + offset, tuple(new_shape), self.var_type)

    def iter_values(self):
        base = self.base
        var_type = self.var_type
        total = 1
        for d in self.shape:
            total *= d
        for off in range(total):
            yield LinearCombo.from_var(VarIdx(base + off, var_type))

    def materialize(self):
        arr = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            arr[idx] = self[idx]
        return arr


class LinearCombo(Generic[T]):
    __slots__ = ("terms",)

    def __init__(self, terms: List[Tuple[T, VarIdx]] = None) -> None:
        self.terms: List[Tuple[T, VarIdx]] = terms if terms is not None else []

    def __repr__(self) -> str:
        return f"LinearCombo({self.terms})"

    @classmethod
    def zero(cls) -> LinearCombo[T]:
        return cls([])

    def is_zero(self) -> bool:
        return len(self.terms) == 0

    @classmethod
    def one(cls) -> LinearCombo[T]:
        return cls([(1, VarIdx.con())])

    def is_one(self) -> bool:
        return len(self.terms) == 1 and self.terms[0] == (1, VarIdx.con())

    def get_const(self) -> T:
        if len(self.terms) == 0:
            return 0
        else:
            coeff, var = self.terms[0]
            if var != VarIdx.con():
                raise ValueError("First term is not the constant term")
            return coeff

    # Turn a linear combo representing a single primary witness variable into a public variable
    # This is used to make inputs, outputs, and unconstrained tensor outputs part of the public potion of the witness vector
    def make_public(self, new_idx):
        if (
            len(self.terms) == 1
            and self.terms[0][0] == 1
            and self.terms[0][1].var_type == VarType.PRIMARY
        ):
            self.terms = [(1, VarIdx(new_idx, VarType.PUBLIC))]

    def get_single_var(self) -> VarIdx:
        if len(self.terms) == 1 and self.terms[0][0] == 1:
            return self.terms[0][1]
        else:
            print("Terms:", self.terms)
            raise ValueError("LinearCombo does not represent a single variable")

    @classmethod
    def from_const(cls, val: T) -> LinearCombo[T]:
        return cls([(val, VarIdx.con())])

    @classmethod
    def from_usize(cls, val: int) -> LinearCombo[T]:
        return cls([(val, VarIdx.con())])

    @classmethod
    def from_var(cls, var: VarIdx) -> LinearCombo[T]:
        return cls([(1, var)])

    def __mul__(self, other: object) -> LinearCombo[T]:
        if isinstance(other, LinearCombo):
            raise NotImplementedError(
                "Create a constraint, linear combos cannot be multiplied"
            )
        else:
            # Scalar multiplication: if the scalar is zero, return zero.
            if other == 0:
                return LinearCombo.zero()
            new_terms = [(coeff * other, var) for coeff, var in self.terms]
            return LinearCombo(new_terms)

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> LinearCombo[T]:
        if isinstance(other, LinearCombo):
            raise NotImplementedError(
                "Create a constraint, linear combos cannot be divided"
            )
        else:
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return self * (1 / other)

    def __add__(self, other: LinearCombo[T]) -> LinearCombo[T]:
        if isinstance(other, LinearCombo):
            terms: List[Tuple[T, VarIdx]] = []
            i, j = 0, 0
            while i < len(self.terms) and j < len(other.terms):
                coeff_a, var_a = self.terms[i]
                coeff_b, var_b = other.terms[j]
                if var_a < var_b:
                    terms.append((coeff_a, var_a))
                    i += 1
                elif var_a > var_b:
                    terms.append((coeff_b, var_b))
                    j += 1
                else:
                    new_coeff = coeff_a + coeff_b
                    if new_coeff != 0:
                        terms.append((new_coeff, var_a))
                    i += 1
                    j += 1
            while i < len(self.terms):
                terms.append(self.terms[i])
                i += 1
            while j < len(other.terms):
                terms.append(other.terms[j])
                j += 1
            return LinearCombo(terms)
        # Other assumed to be float array, add each right hand term to the constant term of the left hand side
        else:
            terms = list(self.terms)
            if terms and terms[0][1] == VarIdx.con():
                coeff, var = terms[0]
                terms[0] = (coeff + other, var)
            else:
                # prepend constant term
                terms.insert(0, (other, VarIdx.con()))
            return LinearCombo(terms)

    def __sub__(self, other: LinearCombo[T]) -> LinearCombo[T]:
        terms: List[Tuple[T, VarIdx]] = []
        i, j = 0, 0
        while i < len(self.terms) and j < len(other.terms):
            coeff_a, var_a = self.terms[i]
            coeff_b, var_b = other.terms[j]
            if var_a < var_b:
                terms.append((coeff_a, var_a))
                i += 1
            elif var_a > var_b:
                terms.append((-coeff_b, var_b))
                j += 1
            else:
                new_coeff = coeff_a - coeff_b
                if new_coeff != 0:
                    terms.append((new_coeff, var_a))
                i += 1
                j += 1
        while i < len(self.terms):
            terms.append(self.terms[i])
            i += 1
        while j < len(other.terms):
            coeff_b, var_b = other.terms[j]
            terms.append((-coeff_b, var_b))
            j += 1
        return LinearCombo(terms)


def eval_constraints(r1cs: R1CS, witness: Witness) -> List[float]:
    n_witness_vars = len(witness.public) + len(witness.primary) + len(witness.secondary)
    expected_witness_vars = (
        r1cs.num_witness_variables
        + r1cs.num_secondary_witness_variables
        + r1cs.num_public_variables
    )
    assert (
        r1cs.num_pretermined_public_variables == r1cs.num_public_variables
    ), f"Expected {r1cs.num_pretermined_public_variables} (predetermined) public variables allocated {r1cs.num_public_variables}"
    assert (
        len(witness.public) == r1cs.num_public_variables
    ), f"Expected {r1cs.num_public_variables} pub vars, got {len(witness.public)}"
    assert (
        n_witness_vars == expected_witness_vars
    ), f"Expected {expected_witness_vars} witness variables, got {n_witness_vars}"

    satisfied_constraints = 0
    total_constraints = 0
    errors = []
    num_failed_constraints = 0
    sum_squared_errors = 0
    for node in r1cs.nodes.values():
        node_satisfied_constraints = 0
        for constraint in node.constraints + node.secondary_constraints:
            a = eval_lc(constraint.a, witness)
            b = eval_lc(constraint.b, witness)
            c = eval_lc(constraint.c, witness)
            error = abs(a * b - c)
            sum_squared_errors += error**2
            if error > 1e-3:
                print(f"Constraint {constraint.label} failed: {a} * {b} != {c}")
                num_failed_constraints += 1
            else:
                node_satisfied_constraints += 1
            errors.append(error)
            total_constraints += 1
        satisfied_constraints += node_satisfied_constraints
        if len(node.constraints) + len(node.secondary_constraints) > 0:
            print(
                f"Satisfied {node_satisfied_constraints}/{len(node.constraints)+len(node.secondary_constraints)} constraints for node {node.node.name}"
            )
        else:
            print(f"Node {node.node.name} has no constraints")
    print(f"Satisfied {satisfied_constraints}/{total_constraints} constraints")
    print(
        f"Num constraints: {total_constraints} ({r1cs.num_constraints} primary, {r1cs.num_secondary_constraints} secondary)"
    )
    print(
        f"Witness size: {n_witness_vars} ({r1cs.num_public_variables} public, {r1cs.num_witness_variables} witness, {r1cs.num_secondary_witness_variables} secondary)"
    )
    if num_failed_constraints > 0:
        raise ValueError(
            f"Constraints not satisfied: {num_failed_constraints} errors found."
        )

    print(f"sqrt(J): {math.sqrt(sum_squared_errors)}")
    return errors


def report_constraint_error(
    r1cs: R1CS, witness: Witness, histogram_step: float = 1e-22
) -> dict:
    n_witness_vars = len(witness.public) + len(witness.primary) + len(witness.secondary)
    expected_witness_vars = (
        r1cs.num_witness_variables
        + r1cs.num_secondary_witness_variables
        + r1cs.num_public_variables
    )
    assert (
        r1cs.num_pretermined_public_variables == r1cs.num_public_variables
    ), f"Expected {r1cs.num_pretermined_public_variables} (predetermined) public variables allocated {r1cs.num_public_variables}"
    assert (
        len(witness.public) == r1cs.num_public_variables
    ), f"Expected {r1cs.num_public_variables} pub vars, got {len(witness.public)}"
    assert (
        n_witness_vars == expected_witness_vars
    ), f"Expected {expected_witness_vars} witness variables, got {n_witness_vars}"

    satisfied_constraints = 0
    total_constraints = 0
    num_failed_constraints = 0
    sum_squared_errors = 0.0
    sum_fourth_errors = 0.0
    histogram = {}
    for node in r1cs.nodes.values():
        node_satisfied_constraints = 0
        for constraint in node.constraints + node.secondary_constraints:
            a = eval_lc(constraint.a, witness)
            b = eval_lc(constraint.b, witness)
            c = eval_lc(constraint.c, witness)
            error = abs(a * b - c)
            sum_squared_errors += error**2
            sum_fourth_errors += error**4
            if error > 1e-3:
                num_failed_constraints += 1
            else:
                node_satisfied_constraints += 1
            bin_idx = int(error / histogram_step)
            bin_start = bin_idx * histogram_step
            histogram[bin_start] = histogram.get(bin_start, 0) + 1
            total_constraints += 1
        satisfied_constraints += node_satisfied_constraints
    l2 = math.sqrt(sum_squared_errors)
    l4 = sum_fourth_errors ** 0.25
    return {
        "l2": l2,
        "l4": l4,
        "histogram": histogram,
        "histogram_step": histogram_step,
        "satisfied_constraints": satisfied_constraints,
        "total_constraints": total_constraints,
        "num_failed_constraints": num_failed_constraints,
    }


def eval_lc(lc: LinearCombo[float], witness: Witness) -> float:
    return sum(
        coeff * getattr(witness, var.var_type.value)[var.idx] for coeff, var in lc.terms
    )


def quantize_to_int(x: float, l: int) -> int:
    return int(round(x * (1 << l)))


def eval_lc_fixed(lc, witness: Witness, l: int) -> int:
    result = 0
    for coeff, var in lc.terms:
        coeff_q = quantize_to_int(float(coeff), l)
        val_q = quantize_to_int(getattr(witness, var.var_type.value)[var.idx], l)
        result += coeff_q * val_q
    return result


def eval_constraints_fixed_point(
    r1cs: R1CS, witness: Witness, l: int = 60
) -> List[Fraction]:
    n_witness_vars = len(witness.witness) + len(witness.random) + len(witness.secondary)
    expected_witness_vars = (
        r1cs.num_witness_variables
        + r1cs.num_random_variables
        + r1cs.num_secondary_variables
    ) + 3
    if n_witness_vars != expected_witness_vars:
        raise ValueError(
            f"Expected {expected_witness_vars} witness variables, got {n_witness_vars}"
        )

    satisfied_constraints = 0
    total_constraints = 0
    errors = []
    sum_squared_errors = Fraction(0)

    for node in r1cs.nodes.values():
        node_satisfied_constraints = 0
        total_constraints += len(node.constraints)
        for constraint in node.constraints:
            a = eval_lc_fixed(constraint.a, witness, l)
            b = eval_lc_fixed(constraint.b, witness, l)
            c = eval_lc_fixed(constraint.c, witness, l)
            numerator = a * b - (c << (2 * l))
            denominator = 1 << (4 * l)
            error = Fraction(abs(numerator), denominator)
            sum_squared_errors += error**2
            if error > Fraction(1, 10**3):
                print(
                    f"Constraint {constraint.label} failed: error = {float(error):.2e}"
                )
            else:
                node_satisfied_constraints += 1
            errors.append(error)
        satisfied_constraints += node_satisfied_constraints
        if len(node.constraints) > 0:
            print(
                f"Satisfied {node_satisfied_constraints}/{len(node.constraints)} constraints for node {node.node.name}"
            )
        else:
            print(f"Node {node.node.name} has no constraints")

    print(f"Satisfied {satisfied_constraints}/{total_constraints} constraints")
    print(f"Total sum squared error (fixpoint) = {float(sum_squared_errors):.2e}")
    return errors
