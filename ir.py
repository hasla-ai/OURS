class Operand:

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return (
            f"Operand("
            f"{self.kind}, "
            f"{self.value}"
            f")"
        )


class Attribute:

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return (
            f"{self.name}={self.value}"
        )


class IRInstruction:

    def __init__(
        self,
        opcode,
        operands=None,
        results=None,
        attributes=None,
    ):
        self.opcode = opcode

        self.operands = (
            operands
            if operands is not None
            else []
        )

        self.results = (
            results
            if results is not None
            else []
        )

        self.attributes = (
            attributes
            if attributes is not None
            else []
        )

    def add_operand(self, operand):
        self.operands.append(operand)

    def add_result(self, result):
        self.results.append(result)

    def add_attribute(
        self,
        name,
        value,
    ):
        self.attributes.append(
            Attribute(
                name,
                value,
            )
        )

    def dump(self):

        print(
            f"OP {self.opcode}"
        )

        print(
            "  OPERANDS"
        )

        for operand in self.operands:
            print(
                "   ",
                operand,
            )

        print(
            "  RESULTS"
        )

        for result in self.results:
            print(
                "   ",
                result,
            )

        print(
            "  ATTRIBUTES"
        )

        for attribute in self.attributes:
            print(
                "   ",
                attribute,
            )

            