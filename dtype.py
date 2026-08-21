import math


class FP32:

    BITS = 32
    BYTES = 4

    @staticmethod
    def float_to_bits(value):
        if not isinstance(value, float):
            value = float(value)

        if math.isnan(value):
            raise ValueError(
                "NaN is not supported"
            )

        if math.isinf(value):
            raise ValueError(
                "Infinity is not supported"
            )

        sign = 0

        if value < 0:
            sign = 1
            value = -value

        if value == 0.0:
            return sign << 31

        exponent = 0

        # Normalize:
        # value = fraction * 2^exponent
        while value >= 2.0:
            value /= 2.0
            exponent += 1

        while value < 1.0:
            value *= 2.0
            exponent -= 1

        biased_exponent = exponent + 127

        fraction = value - 1.0

        mantissa = 0

        for i in range(23):
            fraction *= 2.0

            if fraction >= 1.0:
                mantissa |= (
                    1 << (22 - i)
                )
                fraction -= 1.0

        bits = (
            (sign << 31)
            | (biased_exponent << 23)
            | mantissa
        )

        return bits

    @staticmethod
    def bits_to_float(bits):
        if not 0 <= bits <= 0xFFFFFFFF:
            raise ValueError(
                "FP32 requires 32 bits"
            )

        sign = (
            (bits >> 31) & 0x1
        )

        exponent = (
            (bits >> 23) & 0xFF
        )

        mantissa = (
            bits & 0x7FFFFF
        )

        if exponent == 0:
            if mantissa == 0:
                return -0.0 if sign else 0.0

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
                1.0 +
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

    @staticmethod
    def encode(value):
        bits = FP32.float_to_bits(value)

        return bytes(
            [
                bits & 0xFF,
                (bits >> 8) & 0xFF,
                (bits >> 16) & 0xFF,
                (bits >> 24) & 0xFF,
            ]
        )

    @staticmethod
    def decode(data):
        if len(data) != 4:
            raise ValueError(
                "FP32 requires 4 bytes"
            )

        bits = (
            data[0]
            | (data[1] << 8)
            | (data[2] << 16)
            | (data[3] << 24)
        )

        return FP32.bits_to_float(bits)