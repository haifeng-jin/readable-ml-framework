import ctypes
import numpy as np
from framework import core

class Tensor:
    def __init__(self, shape, data=None):
        """
        Initializes a Tensor.

        Args:
            shape (list or tuple): The shape of the tensor.
            data (numpy.ndarray, optional): Initial data for the tensor. If None,
                the tensor will be initialized with uninitialized values.
                If provided, the data type must be float32.
        """
        self._shape = shape
        if data is None:
            self._cpp_tensor = core.Tensor(shape)  # Use the constructor that takes only shape
        else:
            if not isinstance(data, np.ndarray):
                raise TypeError("Data must be a numpy.ndarray")
            if data.dtype != np.float32:
                raise ValueError("Data type must be float32")
            if data.shape != tuple(shape):
                raise ValueError(f"Data shape {data.shape} must match tensor shape {shape}")
            self._cpp_tensor = core.Tensor(shape, data.flatten().tolist())

    @classmethod
    def from_numpy(cls, numpy_array):
        """
        Creates a Tensor from a NumPy array.

        Args:
            numpy_array (numpy.ndarray): The NumPy array.  The data type must be float32.

        Returns:
            Tensor: A new Tensor object.
        """
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if numpy_array.dtype != np.float32:
            raise ValueError("Data type must be float32")
        shape = numpy_array.shape
        return cls(shape, numpy_array) # Use the __init__ with data

    def shape(self):
        """Returns the shape of the tensor."""
        return self._cpp_tensor.get_shape()

    def numpy(self):
        """Returns a copy of the tensor data as a NumPy array."""
        return self._cpp_tensor.get_data()

    def __del__(self):
        """Destructor."""
        pass  # The C++ destructor handles memory management

