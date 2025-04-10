"""Module for tensor operations.

This module contains all the operation functions, both forward and backward
function of each operation. Each operation function here is a wrapper of the
C++ implementation of the operation. The operations are mainly directly used
by the users to implement their layers, models, losses, and optimizers.

It also contains the OpRecord, which records the operations performed on each
`framework.tensor.Tensor`, which is then used by the `framework.autograd`
module to perform backward propagation.
"""

import numpy as np

from framework import tensor
from framework.core import ops


class OpRecord:
    """A record of an operation performed on the tensors.

    A record contains the operation's input tensors and the backward function,
    and the output tensor. It helps the autograd module to fetch the entire
    compute graph during backward propagation.

    Each operation should have a forward function as well, which is omitted
    here. This is mainly because the forward function is not useful during
    backward propagation.

    In a compute graph, Each operation is an edge in the compute graph.  Each
    tensor is a node in the compute graph.

    It is used by `framework.tensor.Tensor` to record the operation that
    produces it in its `.op_record` attribute.

    Attributes:
        func_backward: Callable. The backward function of the operation. For
            example, if it is an add operation like `add(x, y)`, the
            `func_backward` is `add_backward`.
        input_tensors: List of `framework.tensor.Tensor`. The input tensors to
            the operation. For example, if it is an add operation like
            `add(x, y)`, the input tensors are `[x, y]`.
        output_tensor: `framework.tensor.Tensor`. The output tensor of the
            operation. For example, if it is an add operation like `add(x, y)`,
            the output tensor is the return value of the function.
    """

    def __init__(self, func_backward, input_tensors, output_tensor):
        """Initialize the record with the attribute values.

        Args:
            func_backward: Callable. Value for the `func_backward` attribute.
            input_tensors: List of `framework.tensor.Tensor`. Value for the
                `input_tensors` attribute.
            output_tensor: `framework.tensor.Tensor`. Value for the
                `output_tensor` attribute.
        """
        self.func_backward = func_backward
        self.input_tensors = input_tensors
        self.output_tensor = output_tensor


# Forward operations
def matmul(x, y):
    result = tensor.Tensor(shape=(x.shape[0], y.shape[1]))
    ops.matmul(x.data, y.data, result.data)

    result.op_record = OpRecord(
        func_backward=matmul_backward,
        input_tensors=(x, y),
        output_tensor=result,
    )
    return result


def add_(x, y):
    # element-wise in_place add
    # We haven't support backprop for this case yet.
    # So no OpRecord.
    ops.add_element_wise_(x.data, y.data)


def add(x, y):
    result = tensor.Tensor(shape=x.shape)
    ops.add_row_broadcast(x.data, y.data, result.data)

    result.op_record = OpRecord(
        func_backward=add_backward,
        input_tensors=(x, y),
        output_tensor=result,
    )
    return result


def multiply(x, y):
    if isinstance(y, float):
        # Support multiply by a single float.
        y = tensor.Tensor.from_numpy(np.full(x.shape, y, dtype=np.float32))
    result = tensor.Tensor(shape=x.shape)
    ops.multiply(x.data, y.data, result.data)

    result.op_record = OpRecord(
        func_backward=multiply_backward,
        input_tensors=(x, y),
        output_tensor=result,
    )
    return result


def relu(x):
    result = tensor.Tensor(shape=x.shape)
    ops.relu(x.data, result.data)

    result.op_record = OpRecord(
        func_backward=relu_backward,
        input_tensors=(x,),
        output_tensor=result,
    )
    return result


def softmax(x):
    result = tensor.Tensor(shape=x.shape)
    ops.softmax(x.data, result.data)

    result.op_record = OpRecord(
        func_backward=softmax_backward,
        input_tensors=(x,),
        output_tensor=result,
    )
    return result


def log(x):
    result = tensor.Tensor(shape=x.shape)
    ops.log(x.data, result.data)

    result.op_record = OpRecord(
        func_backward=log_backward,
        input_tensors=(x,),
        output_tensor=result,
    )
    return result


def sum(x):
    result = tensor.Tensor(shape=(1,))
    ops.sum(x.data, result.data)

    result.op_record = OpRecord(
        func_backward=sum_backward,
        input_tensors=(x,),
        output_tensor=result,
    )
    return result


# Backward operations
def matmul_backward(output_grad, x, y):
    x_grad = tensor.Tensor(shape=x.shape)
    y_grad = tensor.Tensor(shape=y.shape)
    ops.matmul_backward(
        output_grad.data, x.data, y.data, x_grad.data, y_grad.data
    )
    return x_grad, y_grad


def add_backward(output_grad, x, y):
    x_grad = tensor.Tensor(shape=x.shape)
    y_grad = tensor.Tensor(shape=y.shape)
    ops.add_backward(output_grad.data, x.data, y.data, x_grad.data, y_grad.data)
    return x_grad, y_grad


def multiply_backward(output_grad, x, y):
    x_grad = tensor.Tensor(shape=x.shape)
    y_grad = tensor.Tensor(shape=y.shape)
    ops.multiply_backward(
        output_grad.data, x.data, y.data, x_grad.data, y_grad.data
    )
    return x_grad, y_grad


def relu_backward(output_grad, x):
    input_grad = tensor.Tensor(shape=x.shape)
    ops.relu_backward(output_grad.data, x.data, input_grad.data)
    return input_grad


def softmax_backward(output_grad, x):
    input_grad = tensor.Tensor(shape=x.shape)
    ops.softmax_backward(output_grad.data, x.data, input_grad.data)
    return input_grad


def log_backward(output_grad, x):
    input_grad = tensor.Tensor(shape=x.shape)
    ops.log_backward(output_grad.data, x.data, input_grad.data)
    return input_grad


def sum_backward(output_grad, x):
    input_grad = tensor.Tensor(shape=x.shape)
    ops.sum_backward(output_grad.data, x.data, input_grad.data)
    return input_grad
