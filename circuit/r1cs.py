from __future__ import annotations
from dataclasses import dataclass, field
from typing import OrderedDict, List, Tuple
import onnx
import math
import numpy as np
import nodes
from onnx import ModelProto, mapping, helper
from ir import LinearCombo, VarIdx, VarType, eval_lc, VirtualTensor
import onnxruntime as ort
from eval import evaluate_onnx_model, random_input, INFERENCE_TYPE, PRECISION_OPS
from serializer import Serializer


@dataclass
class R1CS:
    name: str
    original_model: ModelProto | None
    primary_model: ModelProto | None
    secondary_model: ModelProto | None
    unhandled_ops: list = field(default_factory=lambda: [])
    nodes: OrderedDict[str, R1CSNode] = field(default_factory=OrderedDict)
    num_constraints: int = 0
    num_secondary_constraints: int = 0
    num_witness_variables: int = 0
    num_pretermined_public_variables: int = 1
    num_public_variables: int = 1
    num_secondary_witness_variables: int = 0
    num_secondary_constraint_variables: int = 0
    num_random_variables: int = 0
    serializer: Serializer = field(init=False, repr=False)
    full_variable_matmul: bool = False
    save_outputs_to_disk: bool = False
    export_constraints_continuously: bool = False

    def __post_init__(self):
        self.serializer = Serializer(self.name)

    def reserve_vars(self, num_vars: int, var_type: VarType = VarType.PRIMARY) -> int:
        if var_type == VarType.PUBLIC:
            base = self.num_public_variables
            self.num_public_variables += num_vars
            return base
        elif var_type == VarType.PRIMARY:
            base = self.num_witness_variables
            self.num_witness_variables += num_vars
            return base
        elif var_type == VarType.SECONDARY:
            base = self.num_secondary_witness_variables
            self.num_secondary_witness_variables += num_vars
            return base
        else:
            raise ValueError(f"Unsupported var_type {var_type}")

    def update_num_random_variables(self, num_random_variables: int) -> None:
        self.num_random_variables = max(num_random_variables, self.num_random_variables)

    def allocate_var(self, var_type: VarType = VarType.PRIMARY) -> VarIdx:
        base = self.reserve_vars(1, var_type)
        return VarIdx(base, var_type)

    def get_node(self, node_name) -> tuple[R1CSNode, np.ndarray]:
        meta = self.nodes.get(node_name)
        # If output is a dictionary, return the output indexed by input name, else
        # return the output directly (in the case of multiple outputs)
        meta_output = self.get_output(meta)
        if meta.op == "Split":
            output = meta_output[node_name]
        else:
            output = meta_output
        return (meta, output)

    def get_output(self, node: R1CSNode) -> np.ndarray:
        if self.save_outputs_to_disk:
            return self.serializer.load_output(node)
        return node.output

    def allocate_tensor(
        self, dims: tuple, var_type: VarType = VarType.PRIMARY, virtual=False
    ) -> np.ndarray:
        if virtual:
            size = math.prod(dims)
            base = self.reserve_vars(size, var_type)
            return VirtualTensor(base, dims, var_type)
        tensor = np.empty(dims, dtype=object)
        for index in np.ndindex(dims):
            tensor[index] = LinearCombo.from_var(self.allocate_var(var_type))
        return tensor

    def allocate_secondary_constraint_tensor(self, dims: tuple):
        tensor = np.empty(dims, dtype=object)
        for index in np.ndindex(dims):
            tensor[index] = self.num_secondary_constraint_variables
            self.num_secondary_constraint_variables += 1
        return tensor

    # This helper function updates the constraint counts in the R1CS instance
    # And optionally exports them to disk
    def process_constrained_node(self, node):
        if self.export_constraints_continuously:
            self.serializer.serialize_node_constraints(node)

    def constrain(self) -> None:
        for node in self.nodes.values():
            self.constrain_node(node)
            self.process_constrained_node(node)
        last_node = list(self.nodes.keys())[-1]
        last_node = self.nodes[last_node]
        if not last_node.in_wit:
            nodes.constrain_generic(self, last_node)

    def constrain_node(self, node: R1CSNode) -> None:
        op = node.op
        skip_nodes = ["transformer.wte.weight"]
        print(f"Constraining node {node.node.name} with op {op}")
        if node.node.name in skip_nodes:
            print(f"Skipping constraints for node {node.node.name}")
            return
        elif op in self.unhandled_ops:
            print(
                f"{op} constraints not implemented, witness variables associated with this node will be made public"
            )
            nodes.constrain_unhandled(self, node)
            return 
        elif op == "Add":
            nodes.constrain_add(self, node)
        elif op == "Sub":
            nodes.constrain_sub(self, node)
        elif op == "Mul":
            nodes.constrain_mul(self, node)
        elif op == "Div":
            nodes.constrain_div(self, node)
        elif op == "Slice":
            nodes.constrain_slice(self, node)
        elif op == "Input":
            nodes.constrain_input(self, node)
        elif op == "Initializer":
            nodes.constrain_initializer(self, node)
        elif op == "Constant":
            nodes.constrain_constant(self, node)
        elif op == "Identity":
            nodes.constrain_identity(self, node)
        elif op == "MatMul":
            nodes.constrain_matmul(self, node)
        elif op == "Erf":
            nodes.constrain_erf(self, node)
        elif op == "Softmax":
            nodes.constrain_softmax(self, node)
        elif op == "LayerNormalization":
            nodes.constrain_layer_norm(self, node)
        elif op == "Split":
            nodes.constrain_split(self, node)
        elif op == "Flatten":
            nodes.constrain_flatten(self, node)
        elif op == "Transpose":
            nodes.constrain_transpose(self, node)
        elif op == "Reshape":
            nodes.constrain_reshape(self, node)
        elif op == "Sin":
            nodes.constrain_sin(self, node, 0)
        elif op == "Cos":
            nodes.constrain_sin(self, node, math.pi / 2)
        elif op == "Relu":
            nodes.constrain_relu(self, node)
        elif op == "MaxPool":
            nodes.constrain_maxpool(self, node)
        else:
            raise NotImplementedError(f"Operation {op} not implemented in R1CS.")

        # Garbage collection of inputs that have been used by all successors
        if hasattr(node.node, "input") and node.node.input:
            inputs = node.node.input
            for inp in inputs:
                pred = self.nodes.get(inp)
                successor = pred.successors.pop()
                if successor == node.node.name and len(pred.successors) == 0:
                    print("Garbage collecting pred.node", pred.node.name)
                    if self.save_outputs_to_disk:
                        del pred.output

        if self.save_outputs_to_disk:
            self.serializer.write_output(node)
            del node.output

    def eval_secondary_model(
        self, witness, secondary_inputs, R_tensor=None
    ) -> List[float]:
        secondary_input_feed = {}
        secondary_constraints = []
        if R_tensor is None:
            target_dtype = helper.tensor_dtype_to_np_dtype(INFERENCE_TYPE)
            R_tensor = np.random.randn(self.num_random_variables, 1).astype(
                target_dtype
            )
        if self.num_random_variables > 0:
            secondary_input_feed[f"R"] = R_tensor

        index = 0
        for value in secondary_inputs:
            secondary_input_feed[f"Input_{index}"] = value
            index += 1

        if len(secondary_input_feed) > 0:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
            secondary_outputs, _ = evaluate_onnx_model(
                self.secondary_model,
                input_data=secondary_input_feed,
                sess_options=sess_options,
            )
            output_names = [o.name for o in self.secondary_model.graph.output]
            for name, output in zip(output_names, secondary_outputs):
                if name.startswith("r_t_A"):
                    witness.extend_values(VarType.SECONDARY, output.flatten().tolist())
                else:
                    secondary_constraints.extend(output.flatten().tolist())

        for node in self.nodes.values():
            for con in node.secondary_constraints:
                # A2 matrices contain secondary witness variable indices
                if con.label.count("B_r") > 0:
                    for i, a_term in enumerate(con.a.terms):
                        con.a.terms[i] = (
                            secondary_constraints[int(a_term[0])],
                            a_term[1],
                        )
                # C1/C2 matrices contain random variable indices
                for i, c_term in enumerate(con.c.terms):
                    con.c.terms[i] = (
                        R_tensor[int(c_term[0])],
                        c_term[1],
                    )

        return secondary_constraints

    def eval_primary_model(self, input_data=None):
        if input_data == None:
            session = ort.InferenceSession(self.primary_model.SerializeToString())
            input_data = random_input(session, None)
        for name, arr in input_data.items():
            onnx_t = mapping.NP_TYPE_TO_TENSOR_TYPE[arr.dtype]
            if onnx_t in PRECISION_OPS:
                target_dtype = helper.tensor_dtype_to_np_dtype(INFERENCE_TYPE)
                input_data[name] = arr.astype(target_dtype)
        secondary_inputs = []
        witness = Witness()
        input_labels = [o.doc_string for o in self.primary_model.graph.input]
        for label, (_, value) in zip(input_labels, input_data.items()):
            witness.extend_values(VarType.PUBLIC, value.flatten().tolist())
            if label.count("secondary") > 0:
                secondary_inputs.append(value)
        primary_outputs, _ = evaluate_onnx_model(self.primary_model, input_data)
        output_labels = [o.doc_string for o in self.primary_model.graph.output]
        for label, output in zip(output_labels, primary_outputs):
            if label.count("primary") > 0:
                witness.extend_values(VarType.PRIMARY, output.flatten().tolist())
            if label.count("secondary") > 0:
                secondary_inputs.append(output)
            if label.count("public") > 0:
                witness.extend_values(VarType.PUBLIC, output.flatten().tolist())

        return witness, secondary_inputs


@dataclass
class Constraint:
    a: LinearCombo[float]
    b: LinearCombo[float]
    c: LinearCombo[float]
    label: str = ""


@dataclass
class R1CSNode:
    r1cs: R1CS
    output: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=object))
    node: str = ""
    is_var: bool = False
    in_wit: bool = False
    var_type: VarType = VarType.PRIMARY
    op: str = ""
    successors: List[str] = field(default_factory=list)
    shape: Tuple[int, ...] = field(default_factory=tuple)
    constraints: List[Constraint] = field(default_factory=list)
    secondary_constraints: List[Constraint] = field(default_factory=list)

    def append_labeled_constraint(
        self,
        a: "LinearCombo[float]",
        b: "LinearCombo[float]",
        c: "LinearCombo[float]",
        label: str = "",
    ) -> None:
        new_constraint = Constraint(a, b, c, label)
        self.r1cs.num_constraints += 1
        self.constraints.append(new_constraint)

    def append_secondary_constraint(
        self,
        a: "LinearCombo[float]",
        b: "LinearCombo[float]",
        c: "LinearCombo[float]",
        label: str = "",
    ) -> None:
        self.r1cs.num_secondary_constraints += 1
        new_constraint = Constraint(a, b, c, label)
        self.secondary_constraints.append(new_constraint)


@dataclass
class Witness:
    public: List[float] = field(default_factory=lambda: [1.0])
    primary: List[float] = field(default_factory=lambda: [])
    secondary: List[float] = field(default_factory=lambda: [])

    def extend_values(self, var_type: VarType, new_values: List[float]) -> None:
        if var_type == VarType.PUBLIC:
            self.public.extend(new_values)
        elif var_type == VarType.PRIMARY:
            self.primary.extend(new_values)
        elif var_type == VarType.SECONDARY:
            self.secondary.extend(new_values)
