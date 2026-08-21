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
        self.next_address += 1

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

    def write(self, address, data):
        block = self.get_block(address)

        if len(data) > block.size:
            raise ValueError(
                "Data exceeds allocated memory"
            )

        block.data[:len(data)] = data

    def read(self, address):
        block = self.get_block(address)

        return bytes(block.data)

    def free(self, address):
        block = self.get_block(address)

        block.active = False

    def dump(self):
        print("MEMORY")

        for block in self.blocks.values():
            print(" ", block)