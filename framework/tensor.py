import numpy as np

from framework import core


class Tensor:
    def __init__(self, shape, data=None):
        """
        Initializes a Tensor.

        Args:
            shape (list or tuple): The shape of the tensor.
            data (numpy.ndarray, optional): Initial data for the tensor. If
                None, the tensor will be initialized with uninitialized values.
                If provided, the data type must be float32.
        """
        self.shape = tuple(shape)
        if isinstance(data, np.ndarray):
            data = core.Tensor(shape, data.flatten().tolist())
        elif isinstance(data, core.Tensor):
            pass
        elif data is None:
            data = core.Tensor(
                shape
            )  # Use the constructor that takes only shape
        else:
            raise TypeError(
                "Expected data to be one of (numpy.ndarray, "
                "framework.core.Tensor, None). "
                f"Received: {data} of type {type(data)}."
            )
        self.data = data

    @classmethod
    def from_data(cls, data):
        return cls(data.get_shape(), data)

    @classmethod
    def from_numpy(cls, numpy_array):
        """
        Creates a Tensor from a NumPy array.

        Args:
            numpy_array (numpy.ndarray): The NumPy array. The data type must
                be float32.

        Returns:
            Tensor: A new Tensor object.
        """
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if numpy_array.dtype != np.float32:
            raise ValueError("Data type must be float32")
        shape = numpy_array.shape
        return cls(shape, numpy_array)  # Use the __init__ with data

    def numpy(self):
        """Returns a copy of the tensor data as a NumPy array."""
        return self.data.get_data()

    def __del__(self):
        """Destructor."""
        pass  # The C++ destructor handles memory management
