import numpy as np

import framework


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
