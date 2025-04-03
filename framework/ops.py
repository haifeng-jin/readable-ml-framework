import numpy as np

from framework import tensor
from framework.core import ops

# wrap the ops so that we can add the tracing later.

def matmul(a, b):
    return tensor.Tensor.from_data(ops.matmul(a.data, b.data))


def add(a, b):
    return tensor.Tensor.from_data(ops.add(a.data, b.data))


def multiply(a, b):
    return tensor.Tensor.from_data(ops.multiply(a.data, b.data))


def relu(a):
    return tensor.Tensor.from_data(ops.relu(a.data))


def softmax(a):
    return tensor.Tensor.from_data(ops.softmax(a.data))


def log(a):
    return tensor.Tensor.from_data(ops.log(a.data))


def sum(a):
    return tensor.Tensor.from_data(ops.sum(a.data))
