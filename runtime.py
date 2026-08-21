from memory import MemoryManager

class TensorValue:

    def __init__(
        self,
        tensor_id,
        address,
        size,
    ):
        self.tensor_id = tensor_id
        self.address = address
        self.size = size

    def __repr__(self):
        return (
            f"TensorValue("
            f"id={self.tensor_id}, "
            f"address={self.address}, "
            f"size={self.size}"
            f")"
        )

class CPURuntime:

    def __init__(self):
        self.memory = MemoryManager()
        self.values = {}

    def allocate_tensor(
        self,
        tensor_id,
        size,
    ):
        address = self.memory.allocate(size)

        self.values[tensor_id] = TensorValue(
            tensor_id,
            address,
            size,
        )

    def get_tensor(self, tensor_id):
        if tensor_id not in self.values:
            raise ValueError(
                f"Tensor {tensor_id} "
                "does not exist"
            )

        return self.values[tensor_id]

    def write_tensor(
        self,
        tensor_id,
        data,
    ):
        tensor = self.get_tensor(
            tensor_id
        )

        self.memory.write(
            tensor.address,
            data,
        )

    def read_tensor(
        self,
        tensor_id,
    ):
        tensor = self.get_tensor(
            tensor_id
        )

        return self.memory.read(
            tensor.address
        )

    def free_tensor(
        self,
        tensor_id,
    ):
        tensor = self.get_tensor(
            tensor_id
        )

        self.memory.free(
            tensor.address
        )

    def dump_memory(self):
        self.memory.dump()

    def execute_node(self, node):

        instruction = node.instruction
        opcode = instruction.opcode

        if opcode == "LOAD":

            source = self.get_tensor(
                instruction.input_a
            )

            self.values[
                instruction.output
            ] = TensorValue(
                instruction.output,
                source.address,
                source.size,
            )

            return

        if opcode == "STORE":

            source = self.get_tensor(
                instruction.input_a
            )

            destination = self.get_tensor(
                instruction.output
            )

            data = self.memory.read(
                source.address
            )

            self.memory.write(
                destination.address,
                data,
            )

            return

        raise ValueError(
            f"Unsupported opcode: {opcode}"
        )