class TensorValue:
    def __init__(self, tensor_id, data):
        self.tensor_id = tensor_id
        self.data = data

    def __repr__(self):
        return f"TensorValue({self.tensor_id}, {self.data})"


class CPURuntime:

    def __init__(self):
        self.values = {}

    def set_tensor(self, tensor_id, data):
        self.values[tensor_id] = TensorValue(
            tensor_id,
            data,
        )

    def get_tensor(self, tensor_id):
        if tensor_id not in self.values:
            raise ValueError(
                f"Tensor {tensor_id} has no runtime value"
            )

        return self.values[tensor_id]

    def matmul(self, a, b):
        rows_a = len(a)
        cols_a = len(a[0])

        rows_b = len(b)
        cols_b = len(b[0])

        if cols_a != rows_b:
            raise ValueError(
                f"MATMUL shape mismatch: "
                f"[{rows_a},{cols_a}] x "
                f"[{rows_b},{cols_b}]"
            )

        result = []

        for i in range(rows_a):
            row = []

            for j in range(cols_b):
                value = 0

                for k in range(cols_a):
                    value += (
                        a[i][k] *
                        b[k][j]
                    )

                row.append(value)

            result.append(row)

        return result

    def add(self, a, b):
        if len(a) != len(b):
            raise ValueError("ADD shape mismatch")

        result = []

        for i in range(len(a)):
            if len(a[i]) != len(b[i]):
                raise ValueError(
                    "ADD shape mismatch"
                )

            row = []

            for j in range(len(a[i])):
                row.append(
                    a[i][j] +
                    b[i][j]
                )

            result.append(row)

        return result

    def execute_node(self, node):
        instruction = node.instruction

        opcode = instruction.opcode

        if opcode == "MATMUL":
            a = self.get_tensor(
                instruction.input_a
            )

            b = self.get_tensor(
                instruction.input_b
            )

            result = self.matmul(
                a.data,
                b.data,
            )

            self.values[
                instruction.output
            ] = TensorValue(
                instruction.output,
                result,
            )

            return

        if opcode == "ADD":
            a = self.get_tensor(
                instruction.input_a
            )

            b = self.get_tensor(
                instruction.input_b
            )

            result = self.add(
                a.data,
                b.data,
            )

            self.values[
                instruction.output
            ] = TensorValue(
                instruction.output,
                result,
            )

            return

        raise ValueError(
            f"Unsupported opcode: {opcode}"
        )

    def execute(self, graph):
        order = graph.topological_sort()

        for node_id in order:
            node = graph.nodes[node_id]

            self.execute_node(node)

        return self.values