from memory import MemoryManager
from dtype import DType
from tensor import TensorLayout

memory = MemoryManager()

address = memory.allocate(24)

tensor = TensorLayout(
    shape=(2, 3),
    dtype=DType.FP32,
    address=address,
    memory=memory,
)

tensor.dump()

address = memory.allocate(24)

tensor.write((0, 0), 1.5)
tensor.write((0, 1), 2.5)
tensor.write((1, 2), 7.5)

print(
    tensor.read((0, 0))
)
print(
    tensor.read((0, 1))
)
print(
    tensor.read((1, 2))
)

class MemoryBlock:
    def __init__(self, address, size):
        self.address = address
        self.size = size
        self.data = bytearray(size)
        self.active = True

    def __repr__(self):
        return (
            f"MemoryBlock("
            f"address={self.address}, "
            f"size={self.size}, "
            f"active={self.active}"
            f")"
        )


class MemoryManager:

    def __init__(self):
        self.blocks = {}
        self.next_address = 1

    def allocate(self, size):
        if size <= 0:
            raise ValueError(
                "Allocation size must be positive"
            )

        address = self.next_address
        self.next_address += size

        block = MemoryBlock(
            address,
            size,
        )

        self.blocks[address] = block

        return address

    def get_block(self, address):
        if address not in self.blocks:
            raise ValueError(
                f"Invalid memory address: {address}"
            )

        block = self.blocks[address]

        if not block.active:
            raise ValueError(
                f"Memory block {address} "
                "has already been freed"
            )

        return block

    def write_at(self, address, data):
        for block in self.blocks.values():

            start = block.address
            end = (
                block.address
                + block.size
            )
            if start <= address < end:

                offset = (
                    address - start
                )

                if offset + len(data) > block.size:
                    raise ValueError(
                        "Write exceeds memory block"
                    )

                block.data[
                    offset:
                    offset + len(data)
                ] = data

                return

        raise ValueError(
            f"Invalid memory address: {address}"
        )


    def read_at(self, address, size):
        for block in self.blocks.values():

            start = block.address
            end = (
                block.address
                + block.size
            )

            if start <= address < end:

                offset = (
                    address - start
                )

                if offset + size > block.size:
                    raise ValueError(
                        "Read exceeds memory block"
                    )

                return bytes(
                    block.data[
                        offset:
                        offset + size
                    ]
                )

        raise ValueError(
            f"Invalid memory address: {address}"
        )

    def free(self, address):
        block = self.get_block(address)

        block.active = False

    def dump(self):
        print("MEMORY")

        for block in self.blocks.values():
            print(" ", block)