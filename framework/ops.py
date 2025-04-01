from framework.core import ops
from framework import tensor


def matmul(a, b):
    return tensor.Tensor.from_data(ops.matmul(a.data, b.data))

def add(a, b):
    return tensor.Tensor.from_data(ops.add_broadcast_row(a.data, b.data))
