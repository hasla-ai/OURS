# OUR-MIR v0.1
# Machine Intermediate Representation
#
# Instruction format: 64 bits
#
# [ OPCODE 8 ][ INPUT_A 16 ][ INPUT_B 16 ][ OUTPUT 16 ][ FLAGS 8 ]

OPCODES = {
    "NOP": 0b00000000,
    "INPUT": 0b00000001,
    "OUTPUT": 0b00000010,
    "CONST": 0b00000011,
    "ADD": 0b00000100,
    "SUB": 0b00000101,
    "MUL": 0b00000110,
    "DIV": 0b00000111,
    "MATMUL": 0b00001000,
    "TRANSPOSE": 0b00001001,
    "RESHAPE": 0b00001010,
    "REDUCE": 0b00001011,
    "COPY": 0b00001100,
    "ALLOC": 0b00001101,
    "FREE": 0b00001110,
        # Memory
    "LOAD": 0x0F,
    "STORE": 0x10,
}

OPCODE_NAMES = {value: key for key, value in OPCODES.items()}


class Instruction:
    def __init__(
        self,
        opcode,
        input_a=0,
        input_b=0,
        output=0,
        flags=0,
    ):
        if opcode not in OPCODES:
            raise ValueError(f"Unknown opcode: {opcode}")

        self.opcode = opcode
        self.input_a = input_a
        self.input_b = input_b
        self.output = output
        self.flags = flags

    def encode(self):
        opcode = OPCODES[self.opcode]

        if not 0 <= opcode <= 0xFF:
            raise ValueError("Invalid opcode")

        for name, value in (
            ("input_a", self.input_a),
            ("input_b", self.input_b),
            ("output", self.output),
        ):
            if not 0 <= value <= 0xFFFF:
                raise ValueError(f"{name} must fit in 16 bits")

        if not 0 <= self.flags <= 0xFF:
            raise ValueError("flags must fit in 8 bits")

        value = (
            (opcode << 56)
            | (self.input_a << 40)
            | (self.input_b << 24)
            | (self.output << 8)
            | self.flags
        )

        return value

    @classmethod
    def decode(cls, value):
        if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("Instruction must be 64 bits")

        opcode_value = (value >> 56) & 0xFF
        input_a = (value >> 40) & 0xFFFF
        input_b = (value >> 24) & 0xFFFF
        output = (value >> 8) & 0xFFFF
        flags = value & 0xFF

        if opcode_value not in OPCODE_NAMES:
            raise ValueError(
                f"Unknown opcode value: {opcode_value:#04x}"
            )

        opcode = OPCODE_NAMES[opcode_value]

        return cls(
            opcode,
            input_a,
            input_b,
            output,
            flags,
        )

    def binary(self):
        return format(self.encode(), "064b")

    def __repr__(self):
        return (
            f"{self.opcode} "
            f"{self.input_a} "
            f"{self.input_b} "
            f"-> {self.output} "
            f"[flags={self.flags}]"
        )


def main():
    instruction = Instruction(
        opcode="MATMUL",
        input_a=1,
        input_b=2,
        output=3,
    )

    print("=== ORIGINAL ===")
    print(instruction)

    binary = instruction.binary()

    print("\n=== BINARY ===")
    print(binary)

    restored = Instruction.decode(
        int(binary, 2)
    )

    print("\n=== DECODED ===")
    print(restored)

    print("\n=== CHECK ===")
    print(instruction.encode() == restored.encode())


if __name__ == "__main__":
    main()