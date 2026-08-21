from ourmir import Instruction
from graph import Graph, Tensor, Node


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