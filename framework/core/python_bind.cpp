#include "tensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(core, m) { // the module name matches the package name.
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, const std::vector<float>&>())
        .def("get_shape", &Tensor::get_shape)
        .def("get_data", &Tensor::get_data);
}
