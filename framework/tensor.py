from collections import deque

import numpy as np

from framework import core


class OpRecord:
    def __init__(self, func, func_backward, input_tensors, output_tensor):
        self.func = func
        self.func_backward = func_backward
        self.input_tensors = input_tensors
        self.output_tensor = output_tensor


def _search_compute_graph(end_tensor):
    visited = set([end_tensor])
    records = set()
    queue = deque()
    queue.append(end_tensor)

    while len(queue) > 0:
        current_tensor = queue.popleft()

        record = current_tensor.op_record

        if record is None:
            continue

        if record not in records:
            records.add(record)

        for input_tensor in record.input_tensors:
            if input_tensor in visited:
                continue
            visited.add(input_tensor)
            queue.append(input_tensor)

    return list(visited), list(records)


def _topological_sort(tensors, records):
    tensor_to_index = {}
    for index, tensor in enumerate(tensors):
        tensor_to_index[tensor] = index

    # Build edges
    output_to_input = [[] for i in range(len(tensors))]
    for record in records:
        for input_tensor in record.input_tensors:
            output_index = tensor_to_index[record.output_tensor]
            input_index = tensor_to_index[input_tensor]
            output_to_input[output_index].append(input_index)

    out_degree = [0 for i in range(len(tensors))]
    for output_index, input_indices in enumerate(output_to_input):
        for input_index in input_indices:
            out_degree[input_index] += 1

    results = []
    # Get all the end tensors
    queue = deque()
    for index, degree in enumerate(out_degree):
        if degree == 0:
            queue.append(index)

    while len(queue) > 0:
        current_index = queue.popleft()
        results.append(current_index)
        for input_index in output_to_input[current_index]:
            out_degree[input_index] -= 1
            if out_degree[input_index] == 0:
                queue.append(input_index)

    return results


def _backpropagation(end_tensor):
    tensors, records = _search_compute_graph(end_tensor)

    # Why topological sorting?
    # A-B-C-E
    #    \ /
    #     D
    # When backpropagate from E to A, B needs to gather the gradients from 2
    # branches (C & D) before backpropagate it to A.

    sorted_indices = _topological_sort(tensors, records)

    for index in sorted_indices:
        tensor = tensors[index]

        if tensor.op_record is None:
            continue

        input_tensors = tensor.op_record.input_tensors
        func_backward = tensor.op_record.func_backward
        input_grads = func_backward(tensor.grad, *input_tensors)

        if isinstance(input_grads, Tensor):
            input_grads = (input_grads,)

        for input_tensor, input_grad in zip(input_tensors, input_grads):
            input_tensor.grad = input_grad


class Tensor:
    def __init__(self, shape, data=None):
        """
        Initializes a Tensor.

        Args:
            shape (list or tuple): The shape of the tensor.
            data (numpy.ndarray, optional): Initial data for the tensor. If
                None, the tensor will be initialized with uninitialized values.
                If provided, the data type must be float32.
        """
        self.op_record = None
        self.shape = tuple(shape)
        self.grad = None
        if isinstance(data, np.ndarray):
            self.data = core.Tensor(shape, data.flatten().tolist())
        elif isinstance(data, core.Tensor):
            self.data = data
        elif data is None:
            self.data = core.Tensor(shape)
        else:
            raise TypeError(
                "Expected data to be one of (numpy.ndarray, "
                "framework.core.Tensor, None). "
                f"Received: {data} of type {type(data)}."
            )

    @classmethod
    def from_data(cls, data):
        return cls(data.shape, data)

    @classmethod
    def from_numpy(cls, numpy_array):
        """
        Creates a Tensor from a NumPy array.

        Args:
            numpy_array (numpy.ndarray): The NumPy array. The data type must
                be float32.

        Returns:
            Tensor: A new Tensor object.
        """
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("Input must be a numpy.ndarray")
        if numpy_array.dtype != np.float32:
            raise ValueError(
                f"Data type must be np.float32, but got {numpy_array.dtype}."
            )
        shape = numpy_array.shape
        return cls(shape, numpy_array)  # Use the __init__ with data

    def numpy(self):
        """Returns a copy of the tensor data as a NumPy array."""
        return self.data.copy_to_numpy()

    def backward(self):
        """Backpropagation."""
        if self.shape == (1,):
            self.grad = Tensor.from_numpy(np.ones((1,), dtype=np.float32))
        _backpropagation(self)
