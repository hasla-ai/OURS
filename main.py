from ourmir import Instruction
from graph import Graph, Tensor, Node
from runtime import CPURuntime

graph = Graph()

# A = [2 x 3]
graph.add_tensor(
    Tensor(
        1,
        "FP32",
        (2, 3),
    )
)

# B = [3 x 2]
graph.add_tensor(
    Tensor(
        2,
        "FP32",
        (3, 2),
    )
)

# C = A @ B
graph.add_tensor(
    Tensor(3)
)


graph.add_node(
    Node(
        1,
        Instruction(
            "MATMUL",
            1,
            2,
            3,
        ),
    )
)


# D
graph.add_tensor(
    Tensor(
        4,
        "FP32",
        (2, 2),
    )
)

# E = C + D
graph.add_tensor(
    Tensor(5)
)

graph.add_node(
    Node(
        2,
        Instruction(
            "ADD",
            3,
            4,
            5,
        ),
    )
)


# Validate + infer
graph.validate()


runtime = CPURuntime()


runtime.set_tensor(
    1,
    [
        [1, 2, 3],
        [4, 5, 6],
    ],
)

runtime.set_tensor(
    2,
    [
        [1, 2],
        [3, 4],
        [5, 6],
    ],
)

runtime.set_tensor(
    4,
    [
        [10, 20],
        [30, 40],
    ],
)


runtime.execute(graph)


print(
    "Tensor 3:",
    runtime.get_tensor(3).data,
)

print(
    "Tensor 5:",
    runtime.get_tensor(5).data,
)

def main():
    graph = Graph(graph_id=1)

    graph.add_tensor(
        Tensor(
            tensor_id=1,
            dtype="F32",
            shape=(2, 3),
            device="CPU",
        )
    )

    graph.add_tensor(
        Tensor(
            tensor_id=2,
            dtype="F32",
            shape=(3, 4),
            device="CPU",
        )
    )

    graph.add_tensor(
        Tensor(
            tensor_id=3,
            dtype="F32",
            shape=(2, 4),
            device="CPU",
        )
    )

    instruction = Instruction(
        opcode="MATMUL",
        input_a=1,
        input_b=2,
        output=3,
    )

    graph.add_node(
        Node(
            node_id=1,
            instruction=instruction,
        )
    )

    graph.dump()

    print("\nVALIDATION")
    print(graph.validate())


if __name__ == "__main__":
    main()