#include "tensor.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace py = pybind11;

/*
 * Initializes a Tensor object with the specified shape. The underlying data
 * vector is resized to the total number of elements, but the elements are not
 * initialized to 0.0f.
 */
Tensor::Tensor(const std::vector<size_t> &shape) : shape(shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    // data is initialized by default at the creation of the instance.
    // Its size was 0. With resize, all the additional elements added to the
    // vector during resizing are initialized as 0.0f, which is the default
    // value of function call float().
    data.resize(size);
}

/*
 * Initializes a Tensor object with the specified shape and data. It is assumed
 * that the size of the provided data vector matches the total number of
 * elements defined by the shape. The data is copied into the Tensor's internal
 * data vector.
 */
Tensor::Tensor(const std::vector<size_t> &shape,
               const std::vector<float> &data_ptr)
    : shape(shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    data.resize(size);
    std::copy(data_ptr.begin(), data_ptr.end(), data.begin());
}

// Destructor
Tensor::~Tensor() {
    // std::vector's destructor automatically handles the deallocation
    // of the memory used by its elements. Explicitly calling clear() here
    // is redundant for memory management in this case.
}

/*
 * Returns a Pybind11 NumPy array containing a copy of the Tensor's data.
 * A new NumPy array is created with the same shape as the Tensor, and
 * the Tensor's data is copied into this array. This allows for interaction
 * with NumPy in Python without modifying the original Tensor.
 */
py::array_t<float> Tensor::copy_to_numpy() const {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }

    // Create a NumPy array with the same shape as the tensor.
    // py::array_t is the NumPy array type in Pybind11.
    auto result = py::array_t<float>(shape);
    // Get access to the NumPy array's buffer information.
    py::buffer_info buf_info = result.request();
    float *ptr = static_cast<float *>(buf_info.ptr);

    // Copy the tensor's data to the NumPy array's buffer.
    std::copy(data.begin(), data.end(), ptr);

    return result;
}
