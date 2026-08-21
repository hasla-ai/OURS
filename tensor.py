from dtype import dtype_size, dtype_name, DType

class TensorLayout:

    def __init__(
        self,
        shape,
        dtype,
        address=0,
    ):
        if not shape:
            raise ValueError(
                "Tensor shape cannot be empty"
            )

        for dimension in shape:
            if dimension <= 0:
                raise ValueError(
                    "Tensor dimensions must be positive"
                )

        self.shape = tuple(shape)
        self.dtype = dtype
        self.address = address

        self.strides = self._calculate_strides()

    def _calculate_strides(self):
        """
        Row-major / C-order layout

        shape [2,3,4]

        strides:
        [12,4,1]
        """

        strides = [0] * len(self.shape)

        stride = 1

        for i in range(
            len(self.shape) - 1,
            -1,
            -1,
        ):
            strides[i] = stride
            stride *= self.shape[i]

        return tuple(strides)

    def num_elements(self):
        result = 1

        for dimension in self.shape:
            result *= dimension

        return result

    def nbytes(self):
        return (
            self.num_elements()
            * dtype_size(self.dtype)
        )

    def offset(self, indices):
        if len(indices) != len(self.shape):
            raise ValueError(
                "Index dimension mismatch"
            )

        element_offset = 0

        for index, dimension, stride in zip(
            indices,
            self.shape,
            self.strides,
        ):

            if not 0 <= index < dimension:
                raise IndexError(
                    f"Index {index} "
                    f"out of range for dimension "
                    f"{dimension}"
                )

            element_offset += (
                index * stride
            )

        return element_offset

    def address_of(self, indices):
        element_offset = self.offset(
            indices
        )

        byte_offset = (
            element_offset
            * dtype_size(self.dtype)
        )

        return self.address + byte_offset

    def dump(self):

        print("TENSOR LAYOUT")

        print(
            "shape:",
            self.shape,
        )

        print(
            "dtype:",
            dtype_name(self.dtype),
        )

        print(
            "address:",
            hex(self.address),
        )

        print(
            "strides:",
            self.strides,
        )

        print(
            "elements:",
            self.num_elements(),
        )

        print(
            "bytes:",
            self.nbytes(),
        )


#2D and N -dimension Tensor

tensor = TensorLayout(
    shape=(2, 3),
    dtype=DType.FP32,
    address=0x1000,
)

tensor.dump()

tensor = TensorLayout(
    shape=(2, 3, 4),
    dtype=DType.FP32,
    address=0x1000,
)

tensor.dump()