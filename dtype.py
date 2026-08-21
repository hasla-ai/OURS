import math

class DType:

    BOOL = 0x01
    INT8 = 0x02
    INT32 = 0x03

    FP16 = 0x04
    BF16 = 0x05
    FP32 = 0x06


DTYPE_NAMES = {
    DType.BOOL: "BOOL",
    DType.INT8: "INT8",
    DType.INT32: "INT32",
    DType.FP16: "FP16",
    DType.BF16: "BF16",
    DType.FP32: "FP32",
}


DTYPE_SIZE = {
    DType.BOOL: 1,
    DType.INT8: 1,
    DType.INT32: 4,

    DType.FP16: 2,
    DType.BF16: 2,
    DType.FP32: 4,
}


def dtype_name(dtype):
    if dtype not in DTYPE_NAMES:
        raise ValueError(
            f"Unknown dtype: {dtype}"
        )

    return DTYPE_NAMES[dtype]


def dtype_size(dtype):
    if dtype not in DTYPE_SIZE:
        raise ValueError(
            f"Unknown dtype: {dtype}"
        )

    return DTYPE_SIZE[dtype]

class Codec:

    def __init__(self):
        self.codecs = {
            DType.INT8: INT8Codec,
            DType.INT32: INT32Codec,
            DType.FP32: FP32Codec,
        }

    def encode(self, dtype, value):

        if dtype not in self.codecs:
            raise ValueError(
                f"No codec for {dtype_name(dtype)}"
            )

        return self.codecs[
            dtype
        ].encode(value)

    def decode(self, dtype, data):

        if dtype not in self.codecs:
            raise ValueError(
                f"No codec for {dtype_name(dtype)}"
            )

        return self.codecs[
            dtype
        ].decode(data)

class FP32Codec:

    @staticmethod
    def encode(value):
        import math

        value = float(value)

        if math.isnan(value):
            raise ValueError("NaN unsupported")

        if math.isinf(value):
            raise ValueError("Infinity unsupported")

        sign = 0

        if value < 0:
            sign = 1
            value = -value

        if value == 0:
            bits = sign << 31

            return bits.to_bytes(
                4,
                "little",
            )

        exponent = 0

        while value >= 2.0:
            value /= 2.0
            exponent += 1

        while value < 1.0:
            value *= 2.0
            exponent -= 1

        biased = exponent + 127

        fraction = value - 1.0

        mantissa = 0

        for i in range(23):
            fraction *= 2

            if fraction >= 1:
                mantissa |= (
                    1 << (22 - i)
                )

                fraction -= 1

        bits = (
            (sign << 31)
            | (biased << 23)
            | mantissa
        )

        return bits.to_bytes(
            4,
            "little",
        )

    @staticmethod
    def decode(data):

        bits = int.from_bytes(
            data,
            "little",
        )

        sign = (
            bits >> 31
        ) & 1

        exponent = (
            bits >> 23
        ) & 0xFF

        mantissa = (
            bits & 0x7FFFFF
        )

        if exponent == 0:

            if mantissa == 0:
                return 0.0

            fraction = (
                mantissa /
                (2 ** 23)
            )

            value = (
                fraction *
                (2 ** -126)
            )

        else:

            fraction = (
                1 +
                mantissa /
                (2 ** 23)
            )

            value = (
                fraction *
                (2 ** (exponent - 127))
            )

        if sign:
            value = -value

        return value


class INT8Codec:

    @staticmethod
    def encode(value):

        if not -128 <= value <= 127:
            raise ValueError(
                "INT8 overflow"
            )

        return int(value).to_bytes(
            1,
            "little",
            signed=True,
        )

    @staticmethod
    def decode(data):

        if len(data) != 1:
            raise ValueError(
                "INT8 requires 1 byte"
            )

        return int.from_bytes(
            data,
            "little",
            signed=True,
        )

class INT32Codec:

    @staticmethod
    def encode(value):

        if not (
            -2147483648
            <= value
            <= 2147483647
        ):
            raise ValueError(
                "INT32 overflow"
            )

        return int(value).to_bytes(
            4,
            "little",
            signed=True,
        )

    @staticmethod
    def decode(data):

        if len(data) != 4:
            raise ValueError(
                "INT32 requires 4 bytes"
            )

        return int.from_bytes(
            data,
            "little",
            signed=True,
        )


# test

codec = Codec()


values = [
    (DType.INT8, -10),
    (DType.INT8, 127),
    (DType.INT32, 100000),
    (DType.FP32, 1.0),
    (DType.FP32, 3.14),
]


for dtype, value in values:

    raw = codec.encode(
        dtype,
        value,
    )

    restored = codec.decode(
        dtype,
        raw,
    )

    print(
        dtype_name(dtype),
        value,
        "->",
        raw.hex(),
        "->",
        restored,
    )