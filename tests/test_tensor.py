import numpy as np
import framework # Import the module

def test_create_from_numpy():
    """Test creating a Tensor from a NumPy array."""
    numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = framework.Tensor(numpy_array.shape)
    shape = numpy_array.shape
    tensor = framework.Tensor.from_numpy(numpy_array)
    assert tensor.shape == shape
    data = tensor.numpy()
    assert data.shape == shape
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data, numpy_array)
