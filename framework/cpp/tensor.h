#ifndef TENSOR_H
#define TENSOR_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
#include <pybind11/chrono.h>
#include <iostream>
#include <algorithm>
#include <vector>

namespace py = pybind11;

class Tensor {
public:
    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data_ptr);
    ~Tensor();

    std::vector<size_t> get_shape() const;
    py::array_t<float> get_data() const;
    const std::vector<float>& get_data_vector() const;

private:
    std::vector<size_t> shape_;
    std::vector<float> data_float_;
};

#endif // TENSOR_H
