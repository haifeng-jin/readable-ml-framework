from framework import tensor
from framework.core import ops


class OpRecord:
    def __init__(self, func, func_backward, input_tensors, output_tensor):
        self.func = func
        self.func_backward = func_backward
        self.input_tensors = input_tensors
        self.output_tensor = output_tensor


# Forward operations
def matmul(a, b):
    result = tensor.Tensor(shape=(a.shape[0], b.shape[1]))
    ops.matmul(a.data, b.data, result.data)

    result.op_record = OpRecord(
        func=matmul,
        func_backward=matmul_backward,
        input_tensors=(a, b),
        output_tensor=result,
    )
    return result


def add(a, b):
    result = tensor.Tensor(shape=a.shape)
    ops.add(a.data, b.data, result.data)

    result.op_record = OpRecord(
        func=add,
        func_backward=add_backward,
        input_tensors=(a, b),
        output_tensor=result,
    )
    return result


def multiply(a, b):
    result = tensor.Tensor(shape=a.shape)
    ops.multiply(a.data, b.data, result.data)

    result.op_record = OpRecord(
        func=multiply,
        func_backward=multiply_backward,
        input_tensors=(a, b),
        output_tensor=result,
    )
    return result


def relu(a):
    result = tensor.Tensor(shape=a.shape)
    ops.relu(a.data, result.data)

    result.op_record = OpRecord(
        func=relu,
        func_backward=relu_backward,
        input_tensors=(a,),
        output_tensor=result,
    )
    return result


def softmax(a):
    result = tensor.Tensor(shape=a.shape)
    ops.softmax(a.data, result.data)

    result.op_record = OpRecord(
        func=softmax,
        func_backward=softmax_backward,
        input_tensors=(a,),
        output_tensor=result,
    )
    return result


def log(a):
    result = tensor.Tensor(shape=a.shape)
    ops.log(a.data, result.data)

    result.op_record = OpRecord(
        func=log,
        func_backward=log_backward,
        input_tensors=(a,),
        output_tensor=result,
    )
    return result


def sum(a):
    result = tensor.Tensor(shape=(1,))
    ops.sum(a.data, result.data)

    result.op_record = OpRecord(
        func=sum,
        func_backward=sum_backward,
        input_tensors=(a,),
        output_tensor=result,
    )
    return result


# Backward operations
def matmul_backward(output_grad, a, b):
    a_grad = tensor.Tensor(shape=a.shape)
    b_grad = tensor.Tensor(shape=b.shape)
    ops.matmul_backward(
        output_grad.data, a.data, b.data, a_grad.data, b_grad.data
    )
    return a_grad, b_grad


def add_backward(output_grad, a, b):
    a_grad = tensor.Tensor(shape=a.shape)
    b_grad = tensor.Tensor(shape=b.shape)
    ops.add_backward(output_grad.data, a.data, b.data, a_grad.data, b_grad.data)
    return a_grad, b_grad


def multiply_backward(output_grad, a, b):
    a_grad = tensor.Tensor(shape=a.shape)
    b_grad = tensor.Tensor(shape=b.shape)
    ops.multiply_backward(
        output_grad.data, a.data, b.data, a_grad.data, b_grad.data
    )
    return a_grad, b_grad


def relu_backward(output_grad, a):
    input_grad = tensor.Tensor(shape=a.shape)
    ops.relu_backward(output_grad.data, a.data, input_grad.data)
    return input_grad


def softmax_backward(output_grad, a):
    input_grad = tensor.Tensor(shape=a.shape)
    ops.softmax_backward(output_grad.data, a.data, input_grad.data)
    return input_grad


def log_backward(output_grad, a):
    input_grad = tensor.Tensor(shape=a.shape)
    ops.log_backward(output_grad.data, a.data, input_grad.data)
    return input_grad


def sum_backward(output_grad, a):
    input_grad = tensor.Tensor(shape=a.shape)
    ops.sum_backward(output_grad.data, a.data, input_grad.data)
    return input_grad
