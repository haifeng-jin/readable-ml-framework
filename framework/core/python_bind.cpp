#include "ops.h"
#include "tensor.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(core, m) { // the module name matches the package name.
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<size_t> &>())
        .def(
            py::init<const std::vector<size_t> &, const std::vector<float> &>())
        .def("copy_to_numpy", &Tensor::copy_to_numpy);

    auto ops_module = m.def_submodule("ops"); // ops submodule is still present.

    // Forward operations
    ops_module.def("matmul", &ops::matmul,
                   "Matrix multiplication of two 2-d tensors.");
    ops_module.def("add_row_broadcast", &ops::add_row_broadcast,
                   "Add (1, n) to (m, n), broadcast by row.");
    ops_module.def("add_element_wise_", &ops::add_element_wise_,
                   "Add (m, n) to (m, n) element-wise in-place.");
    ops_module.def("multiply", &ops::multiply,
                   "Element-wise multiply for a tensor.");
    ops_module.def("relu", &ops::relu, "The relu op for tensors.");
    ops_module.def("softmax", &ops::softmax,
                   "The softmax op row-wise for a 2d tensor.");
    ops_module.def("log", &ops::log, "The element-wise log op for tensors.");
    ops_module.def("sum", &ops::sum, "The sum all the elements in a tensor.");

    // Backward operations
    ops_module.def("matmul_backward", &ops::matmul_backward,
                   "Backward pass for matrix multiplication.");
    ops_module.def("add_backward", &ops::add_row_broadcast_backward,
                   "Backward pass for add operation.");
    ops_module.def("multiply_backward", &ops::multiply_backward,
                   "Backward pass for element-wise multiplication.");
    ops_module.def("relu_backward", &ops::relu_backward,
                   "Backward pass for relu operation.");
    ops_module.def("softmax_backward", &ops::softmax_backward,
                   "Backward pass for softmax operation.");
    ops_module.def("log_backward", &ops::log_backward,
                   "Backward pass for log operation.");
    ops_module.def("sum_backward", &ops::sum_backward,
                   "Backward pass for sum operation.");
}
