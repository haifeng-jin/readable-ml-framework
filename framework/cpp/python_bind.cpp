#include "tensor.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp, m) { // the module name matches the package name.
    auto tensor_module = m.def_submodule("tensor"); // submodule tensor.
    py::class_<Tensor>(tensor_module, "Tensor")
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, const std::vector<float>&>())
        .def("get_shape", &Tensor::get_shape)
        .def("get_data", &Tensor::get_data);
}
