from ir import (
    Operand,
    IRInstruction,
)


store = IRInstruction(
    "STORE"
)


store.add_operand(
    Operand(
        "tensor",
        17,
    )
)


store.add_operand(
    Operand(
        "index",
        [4, 8, 2],
    )
)


store.add_operand(
    Operand(
        "value",
        7.5,
    )
)


store.add_attribute(
    "dtype",
    "FP32",
)


store.dump()


add = IRInstruction(
    "ADD"
)

add.add_operand(
    Operand(
        "tensor",
        10,
    )
)

add.add_operand(
    Operand(
        "tensor",
        11,
    )
)

add.add_result(
    Operand(
        "tensor",
        12,
    )
)

add.add_attribute(
    "dtype",
    "FP32",
)

matmul = IRInstruction(
    "MATMUL"
)

matmul.add_operand(
    Operand(
        "tensor",
        1,
    )
)

matmul.add_operand(
    Operand(
        "tensor",
        2,
    )
)

matmul.add_result(
    Operand(
        "tensor",
        3,
    )
)

matmul.add_attribute(
    "dtype",
    "FP32",
)

matmul.add_attribute(
    "transpose_a",
    False,
)

matmul.add_attribute(
    "transpose_b",
    False,
)