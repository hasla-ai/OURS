TYPE_OPCODE = 0x01
TYPE_OPERAND = 0x02
TYPE_RESULT = 0x03
TYPE_ATTRIBUTE = 0x04


def encode_uint(value):
    return value.to_bytes(
        4,
        "little",
        signed=False,
    )


def encode_string(value):
    return value.encode("utf-8")


def encode_field(
    field_type,
    value,
):
    return (
        bytes([field_type])
        + encode_uint(len(value))
        + value
    )


def encode_instruction(
    instruction,
):
    result = bytearray()

    opcode = encode_string(
        instruction.opcode
    )

    result.extend(
        encode_field(
            TYPE_OPCODE,
            opcode,
        )
    )

    for operand in instruction.operands:

        data = (
            operand.kind
            .encode("utf-8")
            + b"\0"
            + str(
                operand.value
            ).encode("utf-8")
        )

        result.extend(
            encode_field(
                TYPE_OPERAND,
                data,
            )
        )

    for result_operand in instruction.results:

        data = (
            result_operand.kind
            .encode("utf-8")
            + b"\0"
            + str(
                result_operand.value
            ).encode("utf-8")
        )

        result.extend(
            encode_field(
                TYPE_RESULT,
                data,
            )
        )

    for attribute in instruction.attributes:

        data = (
            attribute.name
            .encode("utf-8")
            + b"\0"
            + str(
                attribute.value
            ).encode("utf-8")
        )

        result.extend(
            encode_field(
                TYPE_ATTRIBUTE,
                data,
            )
        )

    return bytes(result)

# test
binary = encode_instruction(
    store
)
print(binary.hex())
print(
    "bits:",
    "".join(
        format(
            byte,
            "08b",
        )
        for byte in binary
    )
)
## Decoder - TLV 포맷

from ir import (
    Operand,
    Attribute,
    IRInstruction,
)


TYPE_OPCODE = 0x01
TYPE_OPERAND = 0x02
TYPE_RESULT = 0x03
TYPE_ATTRIBUTE = 0x04


def read_uint(data, offset):
    if offset + 4 > len(data):
        raise ValueError("Unexpected end of binary")

    value = int.from_bytes(
        data[offset:offset + 4],
        "little",
        signed=False,
    )

    return value, offset + 4


def read_field(data, offset):

    if offset >= len(data):
        raise ValueError("Unexpected end of binary")

    field_type = data[offset]
    offset += 1

    length, offset = read_uint(
        data,
        offset,
    )

    end = offset + length

    if end > len(data):
        raise ValueError(
            "Field exceeds binary size"
        )

    value = data[offset:end]

    return (
        field_type,
        value,
        end,
    )


def decode_text(data):
    return data.decode("utf-8")


def decode_operand(data):

    parts = data.split(
        b"\0",
        1,
    )

    if len(parts) != 2:
        raise ValueError(
            "Malformed operand"
        )

    kind = parts[0].decode("utf-8")
    value = parts[1].decode("utf-8")

    return Operand(
        kind,
        value,
    )


def decode_instruction(data):

    offset = 0

    opcode = None
    operands = []
    results = []
    attributes = []

    while offset < len(data):

        (
            field_type,
            value,
            offset,
        ) = read_field(
            data,
            offset,
        )

        if field_type == TYPE_OPCODE:

            opcode = decode_text(
                value
            )

        elif field_type == TYPE_OPERAND:

            operands.append(
                decode_operand(value)
            )

        elif field_type == TYPE_RESULT:

            results.append(
                decode_operand(value)
            )

        elif field_type == TYPE_ATTRIBUTE:

            parts = value.split(
                b"\0",
                1,
            )

            if len(parts) != 2:
                raise ValueError(
                    "Malformed attribute"
                )

            name = parts[0].decode(
                "utf-8"
            )

            attr_value = parts[1].decode(
                "utf-8"
            )

            attributes.append(
                Attribute(
                    name,
                    attr_value,
                )
            )

        else:

            raise ValueError(
                f"Unknown field type: "
                f"{field_type}"
            )

    if opcode is None:
        raise ValueError(
            "Missing opcode"
        )

    return IRInstruction(
        opcode,
        operands,
        results,
        attributes,
    )

## Disassembler

def disassemble(instruction):

    lines = []

    lines.append(
        instruction.opcode
    )

    for operand in instruction.operands:

        lines.append(
            f"  operand "
            f"{operand.kind}="
            f"{operand.value}"
        )

    for result in instruction.results:

        lines.append(
            f"  result "
            f"{result.kind}="
            f"{result.value}"
        )

    for attribute in instruction.attributes:

        lines.append(
            f"  {attribute.name}="
            f"{attribute.value}"
        )

    return "\n".join(lines)

## test

from ir import (
    Operand,
    IRInstruction,
)

from binary import (
    encode_instruction,
    decode_instruction,
    disassemble,
)


original = IRInstruction(
    "STORE"
)


original.add_operand(
    Operand(
        "tensor",
        17,
    )
)

original.add_operand(
    Operand(
        "index",
        [4, 8, 2],
    )
)

original.add_operand(
    Operand(
        "value",
        7.5,
    )
)

original.add_attribute(
    "dtype",
    "FP32",
)


# Machine A
binary = encode_instruction(
    original
)


print("BINARY")
print(binary.hex())


# Machine B
decoded = decode_instruction(
    binary
)


print()
print("DISASSEMBLY")
print(
    disassemble(decoded)
)