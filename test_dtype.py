from dtype import FP32


values = [
    0.0,
    1.0,
    -1.0,
    2.5,
    10.0,
]


for value in values:

    encoded = FP32.encode(value)

    decoded = FP32.decode(encoded)

    print(
        value,
        "->",
        encoded.hex(),
        "->",
        decoded,
    )