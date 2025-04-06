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
    std::vector<size_t> shape;
    std::vector<float> data;

    Tensor(const std::vector<size_t>& shape);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data_ptr);
    ~Tensor();

    // Make a copy of the data in the form of a numpy array.
    py::array_t<float> copy_to_numpy() const;
};

#endif // TENSOR_H
