import numpy as np

# framework.core.ops is the C++ implementation of the ops.
from framework import autograd
from framework import core


class Tensor:
    """The Tensor class.

    This is a Python wrapper for the C++ Tensor class. It can be initialized
    from numpy and export to numpy. It also performs backward propagation in
    `.backward()` method.

    Attributes:
        shape: Tuple. The shape of the tensor.
        data: `core.Tensor`. An instance of the C++ Tensor class.
        grad: `Tensor`. The gradient of the tensor.
        op_record: `framework.ops.OpRecord`. A record of the operation that
            produced the tensor, including the input tensors to the operation
            and its backward function.
    """

    def __init__(self, shape, data=None):
        """Initializes a Tensor.

        Args:
            shape: Tuple. The shape of the tensor.
            data: Optional `numpy.ndarray`. Initial data for the tensor. If
                None, the tensor will be initialized with uninitialized values.
                If provided, the dtype must be float32.
        """
        self.shape = tuple(shape)
        self.data = (
            # core.Tensor is the C++ Tensor class.
            core.Tensor(shape)
            if data is None
            # core.Tensor only accepts plain Python list as data.
            else core.Tensor(shape, data.flatten().tolist())
        )
        self.grad = None
        self.op_record = None

    @classmethod
    def from_numpy(cls, numpy_array):
        """Creates a Tensor from a NumPy array.

        This is the recommended way to create the tensor.

        Args:
            numpy_array: `numpy.ndarray`. The dtype must be float32.

        Returns:
            A new Tensor constructed using the data in the `numpy.ndarray`.
        """
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if numpy_array.dtype != np.float32:
            raise ValueError(
                f"Data type must be np.float32, but got {numpy_array.dtype}."
            )
        # Call Tensor.__init__() under the hood to create the Tensor.
        return cls(numpy_array.shape, numpy_array)

    def numpy(self):
        """Returns a copy of the tensor data as a NumPy array."""
        # Call the C++ member function Tensor.copy_to_numpy().
        return self.data.copy_to_numpy()

    def backward(self):
        """Backward propagation.

        The tensor has to be a single value tensor of shape (1,). The most
        common case is loss.backward(), where loss is a single float value.
        """
        # Create the gradient of the single value, which is always 1.
        self.grad = Tensor.from_numpy(np.ones((1,), dtype=np.float32))
        # Call the autograd module to perform backward propagation.
        autograd.backward_propagation(self)
