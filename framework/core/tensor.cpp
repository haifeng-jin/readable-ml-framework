#include "tensor.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace py = pybind11;

// Constructor
Tensor::Tensor(const std::vector<size_t> &shape) : shape(shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    data.resize(size);
}

// Constructor that takes a plain python list
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
    data.clear();
    shape.clear();
}

// Get data as a numpy array.
// The py::array_t is the pybind's numpy array type.
py::array_t<float> Tensor::copy_to_numpy() const {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }

    // Initialize an empty numpy array.
    auto result = py::array_t<float>(shape);
    py::buffer_info buf_info = result.request();
    float *ptr = static_cast<float *>(buf_info.ptr);

    // Copy the data into the numpy array.
    std::copy(data.begin(), data.end(), ptr);

    return result;
}
