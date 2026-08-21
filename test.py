from ourmir import Instruction
from graph import Graph, Tensor, Node


graph = Graph()

graph.add_tensor(
    Tensor(1, dtype="FP32", shape=(2, 3))
)
graph.add_tensor(
    Tensor(2, dtype="FP32", shape=(3, 4))
)
# shape / dtype을 지정하지 않는다.
graph.add_tensor(Tensor(3))

graph.add_tensor(Tensor(4, "FP32", (2,4)))

graph.add_tensor(Tensor(5))


# N2
# Tensor 3 + Tensor 4 -> Tensor 5
node2 = Node(
    2,
    Instruction(
        "ADD",
        3,
        4,
        5,
    )
)


# N1
# Tensor 1 x Tensor 2 -> Tensor 3
node1 = Node(
    1,
    Instruction(
        "MATMUL",
        1,
        2,
        3,
    )
)


# 일부러 N2를 먼저 추가
graph.add_node(node2)
graph.add_node(node1)


graph.validate()

graph.dump()

print()

graph.execution_plan()