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