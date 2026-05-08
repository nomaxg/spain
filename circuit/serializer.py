import os, json, struct
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from eval import INFERENCE_TYPE
from ir import VarType
from onnx import helper

import numpy as np, onnx

MATRIX_NAMES = ["A0", "B0", "C0", "A1", "B1", "C1", "A2", "B2", "C2"]


def sanitize_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


class NoMemoPickler(pickle.Pickler):
    def memoize(self, obj):
        pass


@dataclass
class MatrixMeta:
    num_entries: int = 0
    width: int = 0
    height: int = 0
    witness_type: str = ""

    def to_dict(self):
        return {
            "num_entries": self.num_entries,
            "width": self.width,
            "height": self.height,
            "witness_type": self.witness_type,
        }


@dataclass
class Serializer:
    model_name: str
    output_dir: str = field(init=False)
    dump_dir: str = field(default="r1cs_dumps")
    export_root: str = field(default="export")
    num_primary_constraints: int = 0
    matrix_meta: Dict[str, MatrixMeta] = field(default_factory=dict)
    num_type_1_constraints: int = 0
    num_type_2_constraints: int = 0
    # For grant proposal estimation of encoding IEEE fp64 constraints, we count all additions and multiplicatinos
    num_additions: int = 0
    num_multiplications: int = 0
    constraint_meta: Dict[str, int] = field(default_factory=dict) # Stores num_zero values per constraint (op) type

    def __post_init__(self):
        self.output_dir = os.path.join(self.export_root, self.model_name)
        for name in MATRIX_NAMES:
            self.matrix_meta[name] = MatrixMeta()

    def load_output(self, node):
        filename = sanitize_filename(node.node.name) + ".npy"
        path = os.path.join(self.dump_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Output for node '{node.node.name}' not found at {path}"
            )
        return np.load(path, allow_pickle=True)

    def write_output(self, node) -> None:
        os.makedirs(self.dump_dir, exist_ok=True)
        filename = sanitize_filename(node.node.name) + ".npy"
        path = os.path.join(self.dump_dir, filename)
        with open(path, "wb") as fh:
            NoMemoPickler(fh, protocol=5).dump(node.output)
        del node.output

    def serialize_node_constraints(self, node):
        def init_matrix_file(name, index_type=0x00, value_type=0x01, metadata=""):
            filename = os.path.join(self.output_dir, f"{name}.bin")
            if os.path.exists(filename):
                os.remove(filename)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            file_type = 0x01
            reserved = 0x00
            comment = (
                (name + f"(Metadata: {metadata})").encode("ascii")[:60].ljust(60, b" ")
            )

            with open(filename, "wb") as f:
                f.write(struct.pack("B", file_type))
                f.write(struct.pack("B", index_type))
                f.write(struct.pack("B", value_type))
                f.write(struct.pack("B", reserved))
                f.write(comment)
                f.write(struct.pack("<Q", 0))
                f.write(struct.pack("<Q", 0))
                f.write(struct.pack("<Q", 0))

        def write_coo(name, entries, index_type=0x00, value_type=0x01):
            if len(entries) == 0:
                return

            filename = os.path.join(self.output_dir, f"{name}.bin")

            with open(filename, "ab") as f:
                idx_fmt = "<I" if index_type == 0x00 else "<Q"
                val_fmt = "<d" if value_type == 0x01 else "<f"
                for row, col, val in entries:
                    f.write(struct.pack(idx_fmt, row))
                    f.write(struct.pack(idx_fmt, col))
                    f.write(struct.pack(val_fmt, val))

            self.matrix_meta[name].num_entries += len(entries)

        def get_offset(var):
            idx = var.idx
            if var.var_type == VarType.PRIMARY:
                idx += (
                    node.r1cs.num_pretermined_public_variables
                    + node.r1cs.num_random_variables
                )
            return idx

        coo: Dict[str, List[Tuple[int, int, float]]] = {
            "A0": [],
            "B0": [],
            "C0": [],
            "A1": [],
            "B1": [],
            "C1": [],
            "A2": [],
            "B2": [],
            "C2": [],
        }

        if len(node.constraints) > 0 and self.num_primary_constraints == 0:
            init_matrix_file("A0")
            init_matrix_file("B0")
            init_matrix_file("C0")

        if len(node.secondary_constraints) > 0 and self.num_type_1_constraints == 0:
            init_matrix_file("A1")
            init_matrix_file("B1")
            init_matrix_file("C1")
            init_matrix_file("A2")
            init_matrix_file("B2")
            init_matrix_file("C2")

        def add_coo_data(name, linear_combo, row_idx, keep_zeroes=False):
            self.num_additions += len(linear_combo.terms) - 1
            self.num_multiplications += len(linear_combo.terms)
            self.constraint_meta[node.op] = self.constraint_meta.get(node.op, 0) + len(linear_combo.terms)
            for coeff, var in linear_combo.terms:
                next_type = var.var_type.to_str()
                self.matrix_meta[name].witness_type = next_type
                if coeff != 0 or keep_zeroes:
                    coo[name].append((row_idx, get_offset(var), coeff))

        for constraint in node.constraints:
            add_coo_data("A0", constraint.a, self.num_primary_constraints)
            add_coo_data("B0", constraint.b, self.num_primary_constraints)
            add_coo_data("C0", constraint.c, self.num_primary_constraints)
            self.num_primary_constraints += 1

        for constraint in node.secondary_constraints:
            if constraint.label.count("r_t_A") > 0:
                add_coo_data("A1", constraint.a, self.num_type_1_constraints)
                add_coo_data("B1", constraint.b, self.num_type_1_constraints)
                add_coo_data(
                    "C1", constraint.c, self.num_type_1_constraints, keep_zeroes=True
                )
                self.num_type_1_constraints += 1
            else:
                add_coo_data(
                    "A2", constraint.a, self.num_type_2_constraints, keep_zeroes=True
                )
                add_coo_data("B2", constraint.b, self.num_type_2_constraints)
                add_coo_data(
                    "C2", constraint.c, self.num_type_2_constraints, keep_zeroes=True
                )
                self.num_type_2_constraints += 1

        write_coo("A0", coo["A0"])
        write_coo("B0", coo["B0"])
        write_coo("C0", coo["C0"])

        if len(node.secondary_constraints) > 0:
            write_coo("A1", coo["A1"])
            write_coo("B1", coo["B1"])
            write_coo("C1", coo["C1"])
            write_coo("A2", coo["A2"])
            write_coo("B2", coo["B2"])
            write_coo("C2", coo["C2"])

        node.constraints.clear()
        node.secondary_constraints.clear()

    def export(self, r1cs) -> None:
        def update_matrix_metadata(filename, num_entries, width, height):
            with open(filename, "r+b") as f:
                f.seek(64)
                f.write(struct.pack("<Q", num_entries))
                f.write(struct.pack("<Q", width))
                f.write(struct.pack("<Q", height))

        def write_dense(name, values, index_type=0x00, value_type=0x01):
            filename = os.path.join(self.output_dir, f"{name}.bin")
            file_type = 0x00
            reserved = 0x00
            comment = (name + "(Metadata)").encode("ascii")[:60].ljust(60, b" ")
            num_entries = len(values)
            width = 1
            height = num_entries

            val_fmt = "<d" if value_type == 0x01 else "<f"

            with open(filename, "wb") as f:
                f.write(struct.pack("B", file_type))
                f.write(struct.pack("B", index_type))
                f.write(struct.pack("B", value_type))
                f.write(struct.pack("B", reserved))
                f.write(comment)
                f.write(struct.pack("<Q", num_entries))
                f.write(struct.pack("<Q", width))
                f.write(struct.pack("<Q", height))

                for v in values:
                    f.write(struct.pack(val_fmt, v))

        os.makedirs(self.output_dir, exist_ok=True)

        for node in r1cs.nodes.values():
            self.serialize_node_constraints(node)

        onnx.save_model(
            r1cs.primary_model, os.path.join(self.output_dir, "primary_model.onnx")
        )
        onnx.save_model(
            r1cs.secondary_model, os.path.join(self.output_dir, "secondary_model.onnx")
        )
        onnx.save_model(
            r1cs.original_model, os.path.join(self.output_dir, "original_model.onnx")
        )

        target_dtype = helper.tensor_dtype_to_np_dtype(INFERENCE_TYPE)
        r_value = np.ones((r1cs.num_random_variables, 1), dtype=target_dtype)
        r_value = np.full(r_value.shape, 1, dtype=target_dtype)
        witness, secondary_inputs = r1cs.eval_primary_model(None)
        secondary_constraints = r1cs.eval_secondary_model(
            witness, secondary_inputs, r_value
        )
        z = (
            witness.public
            + list(r_value.flatten())
            + witness.primary
            + witness.secondary
            + secondary_constraints
        )

        public_wit_size = len(witness.public) + len(list(r_value.flatten()))
        r1cs_public_wit_size = (
            r1cs.num_pretermined_public_variables + r1cs.num_random_variables
        )
        assert (
            public_wit_size == r1cs_public_wit_size
        ), f"Inconsistency between public witness size and precalculated public witness size ({public_wit_size} vs. {r1cs_public_wit_size})"

        write_dense("Z", z)

        primary_output_labels = [
            o.doc_string if o.doc_string else "witness"
            for o in r1cs.primary_model.graph.output
        ]
        secondary_output_labels = [
            o.doc_string if o.doc_string else "witness"
            for o in r1cs.secondary_model.graph.output
        ]
        num_secondary_constraints = (
            self.num_type_1_constraints + self.num_type_2_constraints
        )
        if (
            r1cs.num_secondary_constraints != num_secondary_constraints
            or r1cs.num_constraints != self.num_primary_constraints
        ):
            raise ValueError(
                (
                    "Inconsistency between R1CS constraints and serializer constraints detected:\n"
                    f"  Serializer has {self.num_primary_constraints} primary, R1CS has {r1cs.num_constraints};\n"
                    f"  Serializer has {num_secondary_constraints} secondary, R1CS has {r1cs.num_secondary_constraints}"
                )
            )
        else:
            print("Constraint counts between serializer and R1CS are consistent.")

        print("Updating matrix metadata...")
        z_length = len(z)

        for matrix in ["A0", "B0", "C0"]:
            self.matrix_meta[matrix].height = self.num_primary_constraints
            self.matrix_meta[matrix].width = z_length

        for matrix in ["A1", "B1", "C1"]:
            self.matrix_meta[matrix].height = self.num_type_1_constraints
            self.matrix_meta[matrix].width = z_length

        for matrix in ["A2", "B2", "C2"]:
            self.matrix_meta[matrix].height = self.num_type_2_constraints
            self.matrix_meta[matrix].width = z_length

        height = r1cs.num_constraints
        total_entries = 0
        for name in MATRIX_NAMES:
            filename = os.path.join(self.output_dir, f"{name}.bin")
            num_entries = self.matrix_meta[name].num_entries
            if num_entries > 0:
                total_entries += num_entries
                width = self.matrix_meta[name].width
                height = self.matrix_meta[name].height
                update_matrix_metadata(filename, num_entries, width, height)

        # Estimate number number of non-zero entries if we pad with IEEE 64 constraints based
        # on the results of this paper: https://par.nsf.gov/servlets/purl/10408517 (115 per (+), 24 per (*))
        grant_proposal_fp_variant_max_entries = (
            self.num_additions * 115 + self.num_multiplications * 24
        )

        print("Exporting meta.json...")

        meta = {
            "model_name": self.model_name,
            "primary_output_labels": primary_output_labels,
            "secondary_output_labels": secondary_output_labels,
            "num_total_constraints": r1cs.num_constraints
            + r1cs.num_secondary_constraints,
            "num_nonzero_entries": total_entries,
            "grant_proposal_fp_variant_max_entries": grant_proposal_fp_variant_max_entries,
            "witness_size": len(z),
            "num_random_values": r1cs.num_random_variables,
            "num_primary_constraints": r1cs.num_constraints,
            "num_secondary_constraints": r1cs.num_secondary_constraints,
            "num_public_values": r1cs.num_public_variables,
            "num_witness_values": r1cs.num_witness_variables,
            "num_secondary_witness_values": r1cs.num_secondary_witness_variables,
            "num_secondary_constraint_variables": r1cs.num_secondary_constraint_variables,
            "matrix_meta": {k: v.to_dict() for k, v in self.matrix_meta.items()},
            "constraint_meta": {k: v for k, v in self.constraint_meta.items()},
        }
        with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Exported R1CS model {self.model_name} to {self.output_dir}")
