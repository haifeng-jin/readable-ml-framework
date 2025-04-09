"""Module for auto computing the gradients.

The module contains all the logic for backpropagation, including:

    * Fetch the entire compute graph from the very end output tensor, which is
        the loss in most cases.

    * Topologically sort all the tensors in the compute graph

    * Produce the gradients for reach tensor.

"""

import collections

from framework import core


def _search_compute_graph(end_tensor):
    """Fetch the entire compute graph from the output tensor.

    It performs a breadth-first search (BFS) from the very end output tensor to
    fetch all the input tensors.

    Args:
        end_tensor: `framework.tensor.Tensor`. This is the very end output
            tensor of the compute graph, which is usually the loss.

    Returns:
        Two lists. The first list contains all the `framework.tensor.Tensor`s
        in the compute graph. The second list contains all ops involved in the
        compute graph in the format of `framework.ops.OpRecord`.
    """
    # visited marks if a tensor is visited during the BFS.
    visited = set([end_tensor])

    # records is one of the return values. It contains all the ops in the
    # format of framework.ops.OpRecords. Refer to the docstrings of OpRecords
    # for more details.
    records = set()
    queue = collections.deque()
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
    queue = collections.deque()
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


def backpropagation(end_tensor):
    tensors, records = _search_compute_graph(end_tensor)

    # Why topological sorting?
    # A-B-C-E
    #    \ /
    #     D
    # When backpropagate from E to A, B needs to gather the gradients from 2
    # branches (C & D) before backpropagate it to A.

    sorted_indices = _topological_sort(tensors, records)

    # Clear the grads from last round of backpropagation
    for tensor in tensors:
        if tensor is not end_tensor:
            tensor.grad = None

    for index in sorted_indices:
        tensor = tensors[index]

        if tensor.op_record is None:
            continue

        input_tensors = tensor.op_record.input_tensors
        func_backward = tensor.op_record.func_backward
        input_grads = func_backward(tensor.grad, *input_tensors)

        if not isinstance(input_grads, collections.abc.Iterable):
            input_grads = (input_grads,)

        for input_tensor, input_grad in zip(input_tensors, input_grads):
            if input_tensor.grad is None:
                input_tensor.grad = input_grad
            else:
                input_tensor.grad = core.ops.add_element_wise_(
                    input_tensor.grad, input_grad
                )
