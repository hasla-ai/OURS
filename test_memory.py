from runtime import CPURuntime


runtime = CPURuntime()


runtime.allocate_tensor(
    tensor_id=1,
    size=16,
)


runtime.write_tensor(
    1,
    bytes(
        [
            1, 2, 3, 4
        ]
    ),
)


print(
    runtime.read_tensor(1)
)


runtime.dump_memory()


runtime.free_tensor(1)