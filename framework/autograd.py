"""Module for auto computing the gradients.

The module contains all the logic for backward propagation, including:

    * Fetch the entire compute graph from the very end output tensor, which is
        the loss in most cases.

    * Topologically sort all the tensors in the compute graph

    * Produce the gradients for reach tensor.

The module is used by `framework.tensor.Tensor.backward()` to compute the
gradients for all the tensors in the compute graph.
"""

import collections

from framework import core


def _tracing(end_tensor):
    """Fetch the entire compute graph from the output tensor.

    It performs a breadth-first search (BFS) from the very end output tensor to
    fetch all the input tensors, which is known as tracing throught the compute
    graph.

    It is used by `backward_propagation()` function as the first step to fetch
    the compute graph.

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

    # The queue used for BFS.
    queue = collections.deque()
    # Push in the end tensor as the starting point for BFS.
    queue.append(end_tensor)

    # Continue the BFS when there are still elements left in the queue.
    while len(queue) > 0:
        # Pop out a tensor to fetch its parents producing it.
        current_tensor = queue.popleft()
        record = current_tensor.op_record

        # Some tensors are not produced by any operation but directly
        # initialized by the users, thus, have no op_record. They mark that we
        # are hitting an end in the compute graph.
        if record is None:
            continue

        # Add the record to the return list of OpRecords.
        if record not in records:
            records.add(record)

        # Iterate the parents producing the tensors to push them into the
        # queue while making sure only visiting each tensor once.
        for input_tensor in record.input_tensors:
            if input_tensor in visited:
                continue
            visited.add(input_tensor)
            queue.append(input_tensor)

    # Return all the Tensors and OpRecords in the compute graph.
    return list(visited), list(records)


def _topological_sort(tensors, records):
    """Topologically sort the tensors in reverse order.

    Sort all the tensors in the compute graph topologically in reverse order,
    which means whenever there is an operation producing an output tensor from
    the input tensors, the output tensor is always sorted higher than the input
    tensors.

    We use reverse order because the backward propagation always starts from
    the output tensors.

    It is used by the `backward_propagation()` function before producing the
    gradients.

    ---

    Why do we need to topologically sort the tensors before computing
    gradients?

    Here is an example with a residual link, where A is the input tensor and F
    is the final output tensor. The compute graph looks like the following:

    A->B->C->E->F->G
       |     ^
       v     |
       D-----

    Two ops were performed on B to produce two output tensors, C and D. More
    tensors were produced after C.  In the end, D and F are added together to
    produce G.

    If we do a plain BFS from G, G would push D and F into the queue. Then, D
    would push B. F would push E and so on. The order we push the tensors into
    the search queue may look like [G,D,F,B,E,A,C].

    As you can see B can appear well before C was pushed into the queue. If we
    compute the gradients following this BFS order, it is impossible for C to
    propagate the gradient to B, before B propagate to A.

    ---

    How does topological sort work?

    Topological sort is a well-know algorithm in graph theory. The graph in our
    context is the compute graph, where each node is a tensor, and each
    operation may convert to multiple directed edges pointing from every input
    tensor to the operation to the output tensor.

    We first compute the out-degree of each node, which basically is the number
    of edges pointing to the node. Everytime, we pick a node A with out-degree
    equal to 0, and push it into the sorted array. Then, we remove all the
    edges pointing at node A and reduce the out-degree of the corresponding
    nodes, whose out-going edges have been removed. We repeat this process
    until all the nodes are pushed into the sorted array.

    ---

    Args:
        tensors: List of `framework.tensor.Tensor`. All the tensors in the
            compute graph.
        records: List of `framework.ops.OpRecord`. All the `OpRecord`s in the
            compute graph.

    Returns:
        A list of integers. The topologically sorted list of tensor indices.
        Each integer is an index of the `tensors` list in the function
        argument. The list is sorted in reverse order, where the end tensor
        (usually the loss) appears first.
    """
    # A dictionary with Tensors as its keys and the tensor indices as its
    # values.  If tensors[index_a] == tensor_a then we have
    # tensor_to_index[tensor_a] == index_a.
    tensor_to_index = {}
    for index, tensor in enumerate(tensors):
        tensor_to_index[tensor] = index

    # Construct the edges in the format of adjacency list named
    # output_to_inputs. All the nodes are represented in integers, which are
    # the indices of the tensors. output_to_inputs[index_b] is a list of
    # indices of tensors, which have out-going edges pointing at
    # tensors[index_b].  If there is an edge pointing from index_a to
    # index_b then index_a is in output_to_inputs[index_b].
    output_to_inputs = [[] for i in range(len(tensors))]
    for record in records:
        for input_tensor in record.input_tensors:
            output_index = tensor_to_index[record.output_tensor]
            input_index = tensor_to_index[input_tensor]
            output_to_inputs[output_index].append(input_index)

    # Compute the out-degree of each node. out_degree[index_a] is the number
    # of out-going edges from tensors[index_a].
    out_degree = [0 for i in range(len(tensors))]
    for output_index, input_indices in enumerate(output_to_inputs):
        for input_index in input_indices:
            out_degree[input_index] += 1

    # An empty list to store the sorted indices.
    sorted_indices = []

    # Initialize the queue with the tensor indices with zero out-degrees. In
    # our case, it should only have the end tensor. Using the queue is a
    # optimization reducing time complexity of the sort algorithm from
    # O(num_nodes^2) to O(num_nodes + num_edges). We do not need to go through
    # all the nodes to find the ones with zero out-degrees. We can just
    # maintain the queue to contain all of them as the sort goes and retrieve
    # them from the queue.
    queue = collections.deque()
    for index, degree in enumerate(out_degree):
        if degree == 0:
            queue.append(index)

    # Keep repeating until all the tensor indices are pushed into
    # sorted_indices.
    while len(queue) > 0:
        # Pop the queue and push it into the sorted list.
        current_index = queue.popleft()
        sorted_indices.append(current_index)
        # Update the out-degrees of related nodes and push the ones with zero
        # out-degrees into the queue.
        for input_index in output_to_inputs[current_index]:
            out_degree[input_index] -= 1
            if out_degree[input_index] == 0:
                queue.append(input_index)

    return sorted_indices


def backward_propagation(end_tensor):
    """Perform backward propagation to compute the gradients.

    It first traces the compute graph and sorts the tensors in topological
    order. Then, it computes the gradients for each tensor in the sorted order.
    The gradient of a tensor is stored in `framework.tensor.Tensor.grad`.

    This function is used by `framework.tensor.Tensor.backward()`. The users
    would usually do `loss.backward()` to trigger this function.

    Args:
        end_tensor: `framework.tensor.Tensor`. The tensor from which backward
            propagation starts.
    """

    # Trace the compute graph to get all the Tensors and OpRecords.
    tensors, records = _tracing(end_tensor)

    # Sort all the Tensors in reverse topological order. Note we do not sort
    # the tensors list but creating a new list of sorted indices.
    sorted_indices = _topological_sort(tensors, records)

    # Clear the gradients from last round of backward propagation.
    for tensor in tensors:
        if tensor is not end_tensor:
            tensor.grad = None

    # Iterate through the tensors in reverse topological order to compute the
    # gradients.
    for index in sorted_indices:
        tensor = tensors[index]

        # There are tensors not produce by operations but directly initialized
        # by the users, for example, the training data, and the weights of the
        # neural network. They do not have parents, for which to compute
        # gradients.
        if tensor.op_record is None:
            continue

        # Call backward function to compute the gradients. All the backward
        # function's arguments follow the same pattern, where the first
        # argument is the gradient for the output tensor, followed by all the
        # input tensors to the forward function of the operation.
        input_tensors = tensor.op_record.input_tensors
        func_backward = tensor.op_record.func_backward
        input_grads = func_backward(tensor.grad, *input_tensors)

        # Depending on the number of input tensors to the operation, the
        # gradient may be a single tensor or more. We need to format a single
        # tensor into a tuple for later usage.
        if not isinstance(input_grads, collections.abc.Iterable):
            input_grads = (input_grads,)

        # Assign the gradients to the tensors.
        for input_tensor, input_grad in zip(input_tensors, input_grads):
            if input_tensor.grad is None:
                input_tensor.grad = input_grad
            else:
                # Multiple operations may be applied on one tensor to produce
                # more than one output tensors. Each of the output tensors
                # would backward propagate the gradients to the same input
                # tensor. So, a tensor may already have some gradients when we
                # try to assign one to it. In this case, we simply element-wise
                # add the gradient to the existing one.
                input_tensor.grad = core.ops.add_element_wise_(
                    input_tensor.grad, input_grad
                )
