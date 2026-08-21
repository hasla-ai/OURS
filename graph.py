from ourmir import Instruction


class Tensor:
    def __init__(self, tensor_id, dtype="F32", shape=None, device="CPU"):
        self.tensor_id = tensor_id
        self.dtype = dtype
        self.shape = tuple(shape) if shape is not None else ()
        self.device = device

    def __repr__(self):
        return (
            f"Tensor("
            f"id={self.tensor_id}, "
            f"dtype={self.dtype}, "
            f"shape={self.shape}, "
            f"device={self.device}"
            f")"
        )


class Node:
    def __init__(self, node_id, instruction):
        self.node_id = node_id
        self.instruction = instruction

    def __repr__(self):
        return (
            f"Node("
            f"id={self.node_id}, "
            f"{self.instruction}"
            f")"
        )


class Graph:
    def __init__(self, graph_id=1):
        self.graph_id = graph_id
        self.tensors = {}
        self.nodes = {}

    def add_tensor(self, tensor):
        if tensor.tensor_id in self.tensors:
            raise ValueError(
                f"Tensor {tensor.tensor_id} already exists"
            )

        self.tensors[tensor.tensor_id] = tensor

    def add_node(self, node):
        if node.node_id in self.nodes:
            raise ValueError(
                f"Node {node.node_id} already exists"
            )

        self.nodes[node.node_id] = node

    def _validate_matmul(self, node):
        instruction = node.instruction

        a = self.tensors[instruction.input_a]
        b = self.tensors[instruction.input_b]
        output = self.tensors[instruction.output]

        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError(
                f"Node {node.node_id}: "
                "MATMUL currently requires 2D tensors"
            )

        a_rows, a_cols = a.shape
        b_rows, b_cols = b.shape

        if a_cols != b_rows:
            raise ValueError(
                f"Node {node.node_id}: "
                f"MATMUL shape mismatch: "
                f"{a.shape} x {b.shape}"
            )

        expected_shape = (a_rows, b_cols)

        if output.shape != expected_shape:
            raise ValueError(
                f"Node {node.node_id}: "
                f"invalid output shape: "
                f"expected {expected_shape}, "
                f"got {output.shape}"
            )

    def validate(self):
        for node in self.nodes.values():
            instruction = node.instruction

            if instruction.input_a != 0:
                if instruction.input_a not in self.tensors:
                    raise ValueError(
                        f"Node {node.node_id}: "
                        f"missing tensor {instruction.input_a}"
                    )

            if instruction.input_b != 0:
                if instruction.input_b not in self.tensors:
                    raise ValueError(
                        f"Node {node.node_id}: "
                        f"missing tensor {instruction.input_b}"
                    )

            if instruction.output != 0:
                if instruction.output not in self.tensors:
                    raise ValueError(
                        f"Node {node.node_id}: "
                        f"missing tensor {instruction.output}"
                    )

            if instruction.opcode == "MATMUL":
                self._validate_matmul(node)

        return True

    def dump(self):
        print(f"GRAPH {self.graph_id}")

        print("\nTENSORS")
        for tensor in self.tensors.values():
            print(" ", tensor)

        print("\nNODES")
        for node in self.nodes.values():
            print(" ", node)