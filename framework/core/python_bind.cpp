#include "tensor.h"
#include "ops.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(core, m) { // the module name matches the package name.
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<size_t>&>())
        .def(py::init<const std::vector<size_t>&, const std::vector<float>&>())
        .def("get_shape", &Tensor::get_shape)
        .def("get_data", &Tensor::get_data);
    auto ops_module = m.def_submodule("ops"); // ops submodule is still present.
    ops_module.def("matmul", &matmul, "Matrix multiplication of two tensors.");
    ops_module.def("add_broadcast_row", &add_broadcast_row, "Add (1, n) to (m, n), broadcast by row.");
    ops_module.def("relu", &relu, "The relu op for tensors.");
}
