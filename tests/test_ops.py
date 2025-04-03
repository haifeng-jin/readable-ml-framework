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
    print(result.numpy())
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


def test_mlp_forward_with_loss():
    # Constants
    input_size = 20
    hidden_size = 10
    num_classes = 10
    batch_size = 32

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

    expected = -np.sum(y * np.log(output_probabilities + 1e-8)) / batch_size

    # initializations for framework
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

    result = ops.multiply(
        ops.sum(ops.multiply(y, ops.log(output_probabilities))),
        framework.Tensor.from_numpy(np.full(y.shape, -1.0 / batch_size, dtype=np.float32)),
    )

    assert result.shape == (1,)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)
