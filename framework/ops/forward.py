"""Module for tensor operations.

This module contains all the operation forward functions. Each operation
function here is a wrapper of the C++ implementation of the operation. The
operations are mainly directly used by the users to implement their layers,
models, losses, and optimizers.

For all the functions, besides applying the operations on the input tensors
to produce the output tensor, it also attach an `OpRecord` object to the
`.op_record` attribute of the output tensor so that we can trace the entire
compute graph from the final loss tensor. `OpRecord` is used by the
`framework.autograd` module to perform backward propagation.
"""

import numpy as np

# framework.core.ops is the C++ implementation of the ops.
from framework.core import ops
from framework.ops import backward
from framework.ops.op_record import OpRecord
from framework.tensor import Tensor


def matmul(x, y):
    """Matrix multiplication.

    Args:
        x: `framework.Tensor`. A tensor of shape (m, k).
        y: `framework.Tensor`. A tensor of shape (k, n).

    Returns:
        A `framework.Tensor` with shape (m, n). The result of matrix
        multiplication of x and y.

    Example:
        >>> from framework import Tensor
        >>> x_np = np.array([[1, 2], [3, 4]])
        >>> y_np = np.array([[5, 6], [7, 8]])
        >>> x = Tensor.from_numpy(x_np)
        >>> y = Tensor.from_numpy(y_np)
        >>> result = matmul(x, y)
        >>> print(result.numpy())
        [[19. 22.]
         [43. 50.]]
        >>> x_np = np.array([[1, 2, 3]])
        >>> y_np = np.array([[4], [5], [6]])
        >>> x = Tensor.from_numpy(x_np)
        >>> y = Tensor.from_numpy(y_np)
        >>> result = matmul(x, y)
        >>> print(result.numpy())
        [[32.]]
    """
    # Create the output tensor.
    result = Tensor(shape=(x.shape[0], y.shape[1]))

    # Call the op in C++.
    ops.matmul(x.data, y.data, result.data)

    # Record the tensors and the backward function.
    result.op_record = OpRecord(
        func_backward=backward.matmul_backward,
        input_tensors=(x, y),
        output_tensor=result,
    )

    return result


def add_(x, y):
    """In-place element-wise add.

    The argument x will be modified in-place for the add operation.
    No return value is needed.

    This op is not used in the forward pass of the neural network, but only
    used by the optimizer and the backward propagation, so no backward function
    needed.

    Args:
        x: `framework.Tensor`. A tensor of shape (m, n).
        y: `framework.Tensor`. A tensor of shape (m, n).

    Example:
        >>> from framework import Tensor
        >>> import numpy as np
        >>> x_np = np.array([[1, 2], [3, 4]])
        >>> y_np = np.array([[5, 6], [7, 8]])
        >>> x = Tensor.from_numpy(x_np)
        >>> y = Tensor.from_numpy(y_np)
        >>> add_(x, y)
        >>> print(x.numpy())
        [[ 6.  8.]
         [10. 12.]]
    """
    # Call the op in C++.
    ops.add_element_wise_(x.data, y.data)


def add(x, y):
    """Element-wise addition with row-broadcasting.

    Tensor x is of shape (m, n) and y of shape (1, n). This operation will add
    y to every row of x to produce the output tensor.

    Args:
        x: `framework.Tensor`. A tensor of shape (m, n).
        y: `framework.Tensor`. A tensor of shape (1, n).

    Returns:
        A `framework.Tensor` with shape (m, n). The result of the
        element-wise addition.

    Example:
        >>> from framework import Tensor
        >>> import numpy as np
        >>> x_np = np.array([[1, 2], [3, 4]])
        >>> y_np = np.array([[5, 6]])
        >>> x = Tensor.from_numpy(x_np)
        >>> y = Tensor.from_numpy(y_np)
        >>> result = add(x, y)
        >>> print(result.numpy())
        [[ 6.  8.]
         [ 8. 10.]]
    """
    # Create the output tensor.
    result = Tensor(shape=x.shape)
    # Call the C++ implementation of the add operation.
    ops.add_row_broadcast(x.data, y.data, result.data)

    # Record the input tensors and the backward function for autograd.
    result.op_record = OpRecord(
        func_backward=backward.add_backward,
        input_tensors=(x, y),
        output_tensor=result,
    )
    return result


def multiply(x, y):
    """Element-wise multiplication.

    Args:
        x: `framework.Tensor`. A tensor of shape (m, n).
        y: `framework.Tensor`. A tensor or shape (m, n) or a single
            float.

    Returns:
        A `framework.Tensor` with shape (m, n). The result of the
        element-wise multiplication.

    Example:
        >>> from framework import Tensor
        >>> import numpy as np
        >>> x_np = np.array([[1, 2], [3, 4]])
        >>> y_np = np.array([[5, 6], [7, 8]])
        >>> x = Tensor.from_numpy(x_np)
        >>> y = Tensor.from_numpy(y_np)
        >>> result = multiply(x, y)
        >>> print(result.numpy())
        [[ 5. 12.]
         [21. 32.]]
        >>> y_float = 2.0
        >>> result = multiply(x, y_float)
        >>> print(result.numpy())
        [[2. 4.]
         [6. 8.]]
    """
    if isinstance(y, float):
        # Expand the float to a full matrix with the same shape as x.
        # Every element in y is the value of the single float.
        y = Tensor.from_numpy(np.full(x.shape, y, dtype=np.float32))
    # Create the output tensor.
    result = Tensor(shape=x.shape)
    # Call the C++ implementation of element-wise multiplication.
    ops.multiply(x.data, y.data, result.data)

    # Record the tensors and the backward function.
    result.op_record = OpRecord(
        func_backward=backward.multiply_backward,
        input_tensors=(x, y),
        output_tensor=result,
    )
    return result


def relu(x):
    """Rectified Linear Unit activation function.

    Args:
        x: `framework.Tensor`.

    Returns:
        A `framework.Tensor`. The result of the ReLU operation.

    Example:
        >>> from framework import Tensor
        >>> import numpy as np
        >>> x_np = np.array([[-1, 2], [-3, 4]])
        >>> x = Tensor.from_numpy(x_np)
        >>> result = relu(x)
        >>> print(result.numpy())
        [[0. 2.]
         [0. 4.]]
    """
    # Create the output tensor.
    result = Tensor(shape=x.shape)
    # Call the C++ implementation of the ReLU operation.
    ops.relu(x.data, result.data)

    # Record the tensor and the backward function.
    result.op_record = OpRecord(
        func_backward=backward.relu_backward,
        input_tensors=(x,),
        output_tensor=result,
    )
    return result


def softmax(x):
    """Softmax activation function.

    Args:
        x: `framework.Tensor`. The input tensor of shape (m, n), where m
            is the batch size, n is the number of classes.

    Returns:
        A `framework.Tensor` with shape (m, n). The result of the
        softmax operation.

    Example:
        >>> from framework import Tensor
        >>> import numpy as np
        >>> x_np = np.array([[1.0, 2.0, 0.5]])
        >>> x = Tensor.from_numpy(x_np)
        >>> result = softmax(x)
        >>> print(result.numpy())
        [[0.26894142 0.73105858 0.1636875 ]]
        >>> x_np = np.array([[3.0, 1.0], [0.0, 2.0]])
        >>> x = Tensor.from_numpy(x_np)
        >>> result = softmax(x)
        >>> print(result.numpy())
        [[0.88079708 0.11920292]
         [0.26894142 0.73105858]]
    """
    # Create the output tensor.
    result = Tensor(shape=x.shape)
    # Call the C++ implementation of the softmax operation
    ops.softmax(x.data, result.data)

    # Record the tensor and the backward function.
    result.op_record = OpRecord(
        func_backward=backward.softmax_backward,
        input_tensors=(x,),
        output_tensor=result,
    )
    return result


def log(x):
    """Element-wise natural logarithm.

    Args:
        x: `framework.Tensor`.

    Returns:
        A `framework.Tensor`. The result of the log operation.

    Example:
        >>> from framework import Tensor
        >>> import numpy as np
        >>> x_np = np.array([[1.0, 2.71828], [3.0, 10.0]])
        >>> x = Tensor.from_numpy(x_np)
        >>> result = log(x)
        >>> print(result.numpy())
        [[0.        1.        ]
         [1.09861229 2.30258509]]
    """
    # Create the output tensor.
    result = Tensor(shape=x.shape)
    # Call the C++ implementation of the log operation.
    ops.log(x.data, result.data)

    # Record the tensor and the backward function.
    result.op_record = OpRecord(
        func_backward=backward.log_backward,
        input_tensors=(x,),
        output_tensor=result,
    )
    return result


def sum(x):
    """Sum of all elements in the tensor.

    Args:
        x: `framework.Tensor`. The input tensor of shape (m, n).

    Returns:
        A `framework.Tensor` with shape (1,). The sum of all elements in
        x.

    Example:
        >>> from framework import Tensor
        >>> import numpy as np
        >>> x_np = np.array([[1, 2], [3, 4]])
        >>> x = Tensor.from_numpy(x_np)
        >>> result = sum(x)
        >>> print(result.numpy())
        [10.]
        >>> x_np = np.array([[-1, 5, 2]])
        >>> x = Tensor.from_numpy(x_np)
        >>> result = sum(x)
        >>> print(result.numpy())
        [6.]
    """
    # Create the output tensor.
    result = Tensor(shape=(1,))
    # Call the C++ implementation of the sum operation.
    ops.sum(x.data, result.data)

    # Record the tensor and the backward function.
    result.op_record = OpRecord(
        func_backward=backward.sum_backward,
        input_tensors=(x,),
        output_tensor=result,
    )
    return result
