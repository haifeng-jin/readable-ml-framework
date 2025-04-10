class OpRecord:
    """A record of an operation performed on the tensors.

    A record contains the operation's input tensors and the backward function,
    and the output tensor. It helps the autograd module to fetch the entire
    compute graph during backward propagation.

    Each operation should have a forward function as well, which is omitted
    here. This is mainly because the forward function is not useful during
    backward propagation.

    In a compute graph, Each operation is an edge in the compute graph.  Each
    tensor is a node in the compute graph.

    It is used by `framework.Tensor` to record the operation that produces it
    in its `.op_record` attribute.

    Attributes:
        func_backward: Callable. The backward function of the operation. For
            example, if it is an add operation like `add(x, y)`, the
            `func_backward` is `add_backward`.
        input_tensors: List of `framework.Tensor`. The input tensors to
            the operation. For example, if it is an add operation like
            `add(x, y)`, the input tensors are `[x, y]`.
        output_tensor: `framework.Tensor`. The output tensor of the
            operation. For example, if it is an add operation like `add(x, y)`,
            the output tensor is the return value of the function.
    """

    def __init__(self, func_backward, input_tensors, output_tensor):
        """Initialize the record with the attribute values.

        Args:
            func_backward: Callable. Value for the `func_backward` attribute.
            input_tensors: List of `framework.Tensor`. Value for the
                `input_tensors` attribute.
            output_tensor: `framework.Tensor`. Value for the `output_tensor`
                attribute.
        """
        self.func_backward = func_backward
        self.input_tensors = input_tensors
        self.output_tensor = output_tensor
