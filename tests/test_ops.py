import numpy as np

import framework
from framework import ops


def test_matmul():
    numpy_array1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    numpy_array2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    tensor1 = framework.Tensor.from_numpy(numpy_array1)
    tensor2 = framework.Tensor.from_numpy(numpy_array2)

    expected = np.matmul(numpy_array1, numpy_array2)
    result = framework.ops.matmul(tensor1, tensor2)

    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.numpy(), expected)


def test_matmul_backward():
    # Initialize test matrices
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
    b = np.array(
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32
    )  # (3, 2)
    grad = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)  # (2, 2)

    # Create tensors
    tensor_a = framework.Tensor.from_numpy(a)
    tensor_b = framework.Tensor.from_numpy(b)
    output_grad = framework.Tensor.from_numpy(grad)

    # Calculate expected gradients using numpy
    expected_grad_a = np.matmul(grad, b.T)  # dL/dA = dL/dC * B^T
    expected_grad_b = np.matmul(a.T, grad)  # dL/dB = A^T * dL/dC

    # Calculate actual gradients
    a_grad, b_grad = framework.ops.matmul_backward(
        output_grad, tensor_a, tensor_b
    )

    # Verify shapes and values
    assert a_grad.shape == a.shape
    assert b_grad.shape == b.shape
    np.testing.assert_allclose(
        a_grad.numpy(), expected_grad_a, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        b_grad.numpy(), expected_grad_b, rtol=1e-5, atol=1e-5
    )


def test_add_backward():
    # Initialize test matrices
    a = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)  # (2, 2)
    b = np.array([[1.0, 2.0]], dtype=np.float32)  # (1, 2) for broadcasting
    grad = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)  # (2, 2)

    # Create tensors
    tensor_a = framework.Tensor.from_numpy(a)
    tensor_b = framework.Tensor.from_numpy(b)
    output_grad = framework.Tensor.from_numpy(grad)

    # Calculate expected gradients using numpy
    expected_grad_a = grad  # For a, gradient flows through unchanged
    expected_grad_b = np.sum(
        grad, axis=0, keepdims=True
    )  # Sum across broadcast dimension

    # Calculate actual gradients
    a_grad, b_grad = framework.ops.add_backward(output_grad, tensor_a, tensor_b)

    # Verify shapes and values
    assert a_grad.shape == a.shape
    assert b_grad.shape == b.shape
    np.testing.assert_allclose(
        a_grad.numpy(), expected_grad_a, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        b_grad.numpy(), expected_grad_b, rtol=1e-5, atol=1e-5
    )


def test_multiply_backward():
    # Initialize test matrices
    a = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    grad = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)

    # Create tensors
    tensor_a = framework.Tensor.from_numpy(a)
    tensor_b = framework.Tensor.from_numpy(b)
    output_grad = framework.Tensor.from_numpy(grad)

    # Calculate expected gradients using numpy
    expected_grad_a = grad * b  # dL/dA = dL/dOutput * B
    expected_grad_b = grad * a  # dL/dB = dL/dOutput * A

    # Calculate actual gradients
    a_grad, b_grad = framework.ops.multiply_backward(
        output_grad, tensor_a, tensor_b
    )

    # Verify shapes and values
    assert a_grad.shape == a.shape
    assert b_grad.shape == b.shape
    np.testing.assert_allclose(
        a_grad.numpy(), expected_grad_a, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        b_grad.numpy(), expected_grad_b, rtol=1e-5, atol=1e-5
    )


def test_relu_backward():
    # Initialize test matrix with both positive and negative values
    x = np.array([[-5.0, 6.0], [7.0, -8.0]], dtype=np.float32)
    grad = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)

    # Create tensors
    tensor_x = framework.Tensor.from_numpy(x)
    output_grad = framework.Tensor.from_numpy(grad)

    # Calculate expected gradient using numpy
    expected_grad = grad * (
        x > 0
    )  # Gradient is output_grad where input > 0, else 0

    # Calculate actual gradient
    input_grad = framework.ops.relu_backward(output_grad, tensor_x)

    # Verify shape and values
    assert input_grad.shape == x.shape
    np.testing.assert_allclose(
        input_grad.numpy(), expected_grad, rtol=1e-5, atol=1e-5
    )


def test_softmax_backward():
    # Initialize test matrix
    x = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    # Calculate softmax output
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    softmax_output = exp_x / sum_exp_x

    # Calculate expected gradient using the Jacobian-vector product
    # For each row i, each element j:
    # dL/dx_j = sum_k(dL/dy_k * dy_k/dx_j) where y = softmax(x)
    # dy_k/dx_j = y_j * (δ_jk - y_k) where δ_jk is Kronecker delta
    expected_grad = np.zeros_like(x)

    for i in range(x.shape[0]):
        s = softmax_output[i].reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        expected_grad[i] = np.dot(grad[i], jacobian)

    # Create tensors
    tensor_x = framework.Tensor.from_numpy(x)
    output_grad = framework.Tensor.from_numpy(grad)

    # Calculate actual gradient
    input_grad = framework.ops.softmax_backward(output_grad, tensor_x)

    # Verify shape and values
    assert input_grad.shape == x.shape
    np.testing.assert_allclose(
        input_grad.numpy(), expected_grad, rtol=1e-5, atol=1e-5
    )


def test_log_backward():
    # Initialize test matrix with positive values
    x = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    grad = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    # Create tensors
    tensor_x = framework.Tensor.from_numpy(x)
    output_grad = framework.Tensor.from_numpy(grad)

    # Calculate expected gradient using numpy
    expected_grad = grad / x  # dL/dx = dL/dlogx * dlogx/dx = dL/dlogx * (1/x)

    # Calculate actual gradient
    input_grad = framework.ops.log_backward(output_grad, tensor_x)

    # Verify shape and values
    assert input_grad.shape == x.shape
    np.testing.assert_allclose(
        input_grad.numpy(), expected_grad, rtol=1e-5, atol=1e-5
    )


def test_add():
    numpy_array1 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    numpy_array2 = np.array([[1.0, 2.0]], dtype=np.float32)

    tensor1 = framework.Tensor.from_numpy(numpy_array1)
    tensor2 = framework.Tensor.from_numpy(numpy_array2)

    expected = numpy_array1 + numpy_array2
    result = framework.ops.add(tensor1, tensor2)

    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.numpy(), expected)


def test_multiply():
    numpy_array1 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    numpy_array2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    tensor1 = framework.Tensor.from_numpy(numpy_array1)
    tensor2 = framework.Tensor.from_numpy(numpy_array2)

    expected = numpy_array1 * numpy_array2
    result = framework.ops.multiply(tensor1, tensor2)

    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.numpy(), expected)


def test_relu():
    numpy_array = np.array([[-5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    tensor = framework.Tensor.from_numpy(numpy_array)

    expected = np.maximum(0, numpy_array)
    result = framework.ops.relu(tensor)

    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.numpy(), expected)


def test_softmax():
    x = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    tensor = framework.Tensor.from_numpy(x)

    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    expected = exp_x / sum_exp_x
    result = framework.ops.softmax(tensor)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_log():
    x = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    tensor = framework.Tensor.from_numpy(x)

    expected = np.log(x)
    result = framework.ops.log(tensor)

    assert result.shape == (2, 2)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_sum():
    x = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    tensor = framework.Tensor.from_numpy(x)

    expected = np.sum(x)
    result = framework.ops.sum(tensor)

    assert result.shape == (1,)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_sum_backward():
    x = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    grad = np.array([2.0], dtype=np.float32)

    tensor = framework.Tensor.from_numpy(x)
    output_grad = framework.Tensor.from_numpy(grad)

    expected = np.full(x.shape, grad[0], dtype=np.float32)
    input_grad = framework.ops.sum_backward(output_grad, tensor)

    assert input_grad.shape == x.shape
    np.testing.assert_allclose(
        input_grad.numpy(), expected, rtol=1e-5, atol=1e-5
    )


def test_mlp():
    # Constants
    input_size = 20
    hidden_size = 10
    num_classes = 10
    batch_size = 32

    # Numpy implementation:

    # Initializations
    np.random.seed(0)
    x = np.random.rand(batch_size, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, batch_size)
    y = np.eye(num_classes)[y].astype(np.float32)  # One-hot

    weights_hidden = (np.random.randn(input_size, hidden_size) * 0.01).astype(
        np.float32
    )
    bias_hidden = np.zeros((1, hidden_size)).astype(np.float32)

    weights_output = (np.random.randn(hidden_size, num_classes) * 0.01).astype(
        np.float32
    )
    bias_output = np.zeros((1, num_classes)).astype(np.float32)

    # numpy loss
    hidden_linear = np.dot(x, weights_hidden) + bias_hidden
    hidden_activation = np.maximum(0, hidden_linear)

    output_linear = np.dot(hidden_activation, weights_output) + bias_output
    exp_z = np.exp(
        output_linear - np.max(output_linear, axis=1, keepdims=True)
    )  # For numerical stability
    output_probabilities = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    expected_loss = (
        -np.sum(y * np.log(output_probabilities + 1e-8)) / batch_size
    )

    # Gradient of the loss with respect to the output layer (before softmax)
    d_output_linear = output_probabilities - y

    # Gradient of weights and biases for the output layer
    d_weights_output = np.dot(hidden_activation.T, d_output_linear) / batch_size
    d_bias_output = np.sum(d_output_linear, axis=0, keepdims=True) / batch_size

    # Gradient of the hidden layer activation
    d_hidden_activation = np.dot(d_output_linear, weights_output.T)

    # Gradient of the hidden layer (before activation)
    d_hidden_linear = d_hidden_activation * (hidden_linear > 0).astype(
        int
    )  # Derivative of ReLU

    # Gradient of weights and biases for the hidden layer
    d_weights_hidden = np.dot(x.T, d_hidden_linear) / batch_size
    d_bias_hidden = np.sum(d_hidden_linear, axis=0, keepdims=True) / batch_size
    # Now we have all the grads:
    # d_weights_hidden, d_bias_hidden, d_weights_output, d_bias_output

    # Framework implementation:

    # initializations
    x = framework.Tensor.from_numpy(x)
    y = framework.Tensor.from_numpy(y)
    weights_hidden = framework.Tensor.from_numpy(weights_hidden)
    bias_hidden = framework.Tensor.from_numpy(bias_hidden)
    weights_output = framework.Tensor.from_numpy(weights_output)
    bias_output = framework.Tensor.from_numpy(bias_output)

    # framework loss
    hidden_linear = ops.add(ops.matmul(x, weights_hidden), bias_hidden)
    hidden_activation = ops.relu(hidden_linear)

    output_linear = ops.add(
        ops.matmul(hidden_activation, weights_output), bias_output
    )
    output_probabilities = ops.softmax(output_linear)

    loss = ops.multiply(
        ops.sum(ops.multiply(y, ops.log(output_probabilities))),
        framework.Tensor.from_numpy(
            np.full(y.shape, -1.0 / batch_size, dtype=np.float32)
        ),
    )
    loss.backward()

    # Checks for foward pass results
    assert loss.shape == (1,)
    np.testing.assert_allclose(
        loss.numpy(), expected_loss, rtol=1e-5, atol=1e-5
    )

    # Checks for backward pass results
