"""
For all the backward pass functions, their signatures all follow the same
pattern:

```
def name_backward(output_grad, x, ...):
```

The arguments are the gradients of the output tensor followed by the input
tensors.  In this way, they are easier to call during backward propagation when
we try to call different backward functions in a loop with the same code.
"""

from framework import tensor

# framework.core.ops is the C++ implementation of the ops.
from framework.core import ops


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
