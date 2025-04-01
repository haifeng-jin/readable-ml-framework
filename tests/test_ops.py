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
