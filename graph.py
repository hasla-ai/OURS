from ourmir import Instruction


SUPPORTED_DTYPES = {
    "BOOL",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "FP16",
    "BF16",
    "FP32",
    "FP64",
}


class Tensor:
    def __init__(
        self,
        tensor_id,
        dtype=None,
        shape=None,
        device="CPU",
    ):
        if dtype is not None and dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.tensor_id = tensor_id
        self.dtype = dtype
        self.shape = (
            tuple(shape)
            if shape is not None
            else None
        )
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

    def _get_tensor(self, tensor_id):
        if tensor_id not in self.tensors:
            raise ValueError(
                f"Missing tensor {tensor_id}"
            )

        return self.tensors[tensor_id]

    def infer_node(self, node):
        instruction = node.instruction
        opcode = instruction.opcode

        if opcode == "MATMUL":
            a = self._get_tensor(instruction.input_a)
            b = self._get_tensor(instruction.input_b)

            if a.shape is None or b.shape is None:
                raise ValueError(
                    f"Node {node.node_id}: "
                    "input shape is unknown"
                )

            if a.dtype is None or b.dtype is None:
                raise ValueError(
                    f"Node {node.node_id}: "
                    "input dtype is unknown"
                )

            if len(a.shape) != 2 or len(b.shape) != 2:
                raise ValueError(
                    f"Node {node.node_id}: "
                    "MATMUL requires 2D tensors"
                )

            if a.shape[1] != b.shape[0]:
                raise ValueError(
                    f"Node {node.node_id}: "
                    f"MATMUL shape mismatch: "
                    f"{a.shape} x {b.shape}"
                )

            if a.dtype != b.dtype:
                raise ValueError(
                    f"Node {node.node_id}: "
                    f"MATMUL dtype mismatch: "
                    f"{a.dtype} x {b.dtype}"
                )

            output_shape = (
                a.shape[0],
                b.shape[1],
            )

            output_dtype = a.dtype

            output = self._get_tensor(
                instruction.output
            )

            output.shape = output_shape
            output.dtype = output_dtype

            return output

        if opcode in {"ADD", "SUB", "MUL", "DIV"}:
            a = self._get_tensor(
                instruction.input_a
            )

            b = self._get_tensor(
                instruction.input_b
            )

            if a.shape != b.shape:
                raise ValueError(
                    f"Node {node.node_id}: "
                    f"shape mismatch: "
                    f"{a.shape} vs {b.shape}"
                )

            if a.dtype != b.dtype:
                raise ValueError(
                    f"Node {node.node_id}: "
                    f"dtype mismatch: "
                    f"{a.dtype} vs {b.dtype}"
                )

            output = self._get_tensor(
                instruction.output
            )

            output.shape = a.shape
            output.dtype = a.dtype

            return output

        raise ValueError(
            f"Type inference not implemented "
            f"for opcode: {opcode}"
        )

    def infer(self):
        for node in self.nodes.values():
            self.infer_node(node)

    def validate(self):
        for node in self.nodes.values():
            instruction = node.instruction

            self._get_tensor(
                instruction.input_a
            )

            self._get_tensor(
                instruction.output
            )

            if instruction.input_b != 0:
                self._get_tensor(
                    instruction.input_b
                )

        self.infer()

        return True

    def dump(self):
        print(f"GRAPH {self.graph_id}")

        print("\nTENSORS")

        for tensor in self.tensors.values():
            print(" ", tensor)

        print("\nNODES")

        for node in self.nodes.values():
            print(" ", node)