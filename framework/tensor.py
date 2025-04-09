import numpy as np

from framework import autograd
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
        self.op_record = None
        self.shape = tuple(shape)
        self.grad = None
        if isinstance(data, np.ndarray):
            self.data = core.Tensor(shape, data.flatten().tolist())
        elif isinstance(data, core.Tensor):
            self.data = data
        elif data is None:
            self.data = core.Tensor(shape)
        else:
            raise TypeError(
                "Expected data to be one of (numpy.ndarray, "
                "framework.core.Tensor, None). "
                f"Received: {data} of type {type(data)}."
            )

    @classmethod
    def from_data(cls, data):
        return cls(data.shape, data)

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
            raise ValueError(
                f"Data type must be np.float32, but got {numpy_array.dtype}."
            )
        shape = numpy_array.shape
        return cls(shape, numpy_array)  # Use the __init__ with data

    def numpy(self):
        """Returns a copy of the tensor data as a NumPy array."""
        return self.data.copy_to_numpy()

    def backward(self):
        """Backpropagation."""
        if self.shape == (1,):
            self.grad = Tensor.from_numpy(np.ones((1,), dtype=np.float32))
        autograd.backpropagation(self)
