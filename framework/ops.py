from framework import tensor
from framework.core import ops

# wrap the ops so that we can add the tracing later.


def matmul(a, b):
    result = tensor.Tensor(shape=(a.shape[0], b.shape[1]))
    ops.matmul(a.data, b.data, result.data)
    return result


def add(a, b):
    result = tensor.Tensor(shape=a.shape)
    ops.add(a.data, b.data, result.data)
    return result


def multiply(a, b):
    result = tensor.Tensor(shape=a.shape)
    ops.multiply(a.data, b.data, result.data)
    return result


def relu(a):
    result = tensor.Tensor(shape=a.shape)
    ops.relu(a.data, result.data)
    return result


def softmax(a):
    result = tensor.Tensor(shape=a.shape)
    ops.softmax(a.data, result.data)
    return result


def log(a):
    result = tensor.Tensor(shape=a.shape)
    ops.log(a.data, result.data)
    return result


def sum(a):
    result = tensor.Tensor(shape=(1,))
    ops.sum(a.data, result.data)
    return result
