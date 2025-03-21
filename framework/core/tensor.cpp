#include "tensor.h"
#include <iostream>
#include <vector>
#include <algorithm>

namespace py = pybind11;

//Constructor
Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    data_float_.resize(size);
}

// Constructor that takes a numpy array data pointer
Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data_ptr) : shape_(shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    data_float_.resize(size);
    std::copy(data_ptr.begin(), data_ptr.end(), data_float_.begin());
    for (auto a : data_float_) {
        std::cout << a << std::endl;
    }
}

// Destructor
Tensor::~Tensor() {
    data_float_.clear();
    shape_.clear();
}

// Get shape
std::vector<size_t> Tensor::get_shape() const {
    return shape_;
}

// Get data as a numpy array
py::array_t<float> Tensor::get_data() const {
    size_t size = 1;
    for (size_t dim : shape_) {
        size *= dim;
    }
    auto result = py::array_t<float>(shape_);
    py::buffer_info buf_info = result.request();
    float* ptr = static_cast<float*>(buf_info.ptr);
    std::copy(data_float_.begin(), data_float_.end(), ptr);
    return result;
}
