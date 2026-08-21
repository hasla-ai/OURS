from ourmir import Instruction
from graph import Graph, Tensor, Node


graph = Graph()


graph.add_tensor(
    Tensor(
        1,
        dtype="FP32",
        shape=(2, 3),
    )
)

graph.add_tensor(
    Tensor(
        2,
        dtype="FP32",
        shape=(3, 4),
    )
)

# shape / dtype을 지정하지 않는다.
graph.add_tensor(
    Tensor(3)
)


instruction = Instruction(
    opcode="MATMUL",
    input_a=1,
    input_b=2,
    output=3,
)


graph.add_node(
    Node(
        1,
        instruction,
    )
)


graph.validate()

graph.dump()