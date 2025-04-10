"""Module for backward functions.

This module contains all the backward function for all the supported tensor
operations. They are called during backward propagation to compute the
gradients.

For all the backward functions, their signatures all follow the same pattern:

`def name_backward(output_grad, x, ...):`

The arguments are the gradients of the output tensor followed by the input
tensors. In this way, they are easier to call during backward propagation when
we try to call different backward functions in a loop with the same code.

The implementation also follows the same pattern. They first create the tensors
for the resulting gradients, then call the C++ implementation.

All the C++ backward function all follow the same pattern in ordering the
parameters. The first argument is always the gradient of the output tensor.
Then, follows all the input tensors. Finally, the gradients for the input
tensors.
"""

# framework.core.ops is the C++ implementation of the ops.
from framework.core import ops
from framework.tensor import Tensor


def matmul_backward(output_grad, x, y):
    x_grad = Tensor(shape=x.shape)
    y_grad = Tensor(shape=y.shape)
    ops.matmul_backward(
        output_grad.data, x.data, y.data, x_grad.data, y_grad.data
    )
    return x_grad, y_grad


def add_backward(output_grad, x, y):
    x_grad = Tensor(shape=x.shape)
    y_grad = Tensor(shape=y.shape)
    ops.add_backward(output_grad.data, x.data, y.data, x_grad.data, y_grad.data)
    return x_grad, y_grad


def multiply_backward(output_grad, x, y):
    x_grad = Tensor(shape=x.shape)
    y_grad = Tensor(shape=y.shape)
    ops.multiply_backward(
        output_grad.data, x.data, y.data, x_grad.data, y_grad.data
    )
    return x_grad, y_grad


def relu_backward(output_grad, x):
    input_grad = Tensor(shape=x.shape)
    ops.relu_backward(output_grad.data, x.data, input_grad.data)
    return input_grad


def softmax_backward(output_grad, x):
    input_grad = Tensor(shape=x.shape)
    ops.softmax_backward(output_grad.data, x.data, input_grad.data)
    return input_grad


def log_backward(output_grad, x):
    input_grad = Tensor(shape=x.shape)
    ops.log_backward(output_grad.data, x.data, input_grad.data)
    return input_grad


def sum_backward(output_grad, x):
    input_grad = Tensor(shape=x.shape)
    ops.sum_backward(output_grad.data, x.data, input_grad.data)
    return input_grad
