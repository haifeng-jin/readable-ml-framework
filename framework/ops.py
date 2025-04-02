from framework import tensor
from framework.core import ops


def matmul(a, b):
    return tensor.Tensor.from_data(ops.matmul(a.data, b.data))


def add(a, b):
    return tensor.Tensor.from_data(ops.add_broadcast_row(a.data, b.data))


def relu(a):
    return tensor.Tensor.from_data(ops.relu(a.data))


def softmax(a):
    return tensor.Tensor.from_data(ops.softmax(a.data))


def log(a):
    return tensor.Tensor.from_data(ops.log(a.data))
