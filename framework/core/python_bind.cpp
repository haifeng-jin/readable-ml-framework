#include "ops.h"
#include "tensor.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

/**
 * @brief Bind C++ functions & classes to Python.
 *
 * This function is how pybind11 binds the C++ implementation of the functions
 * and classes to Python functions and classes in a Python module.
 *
 * All the bindings are created by calling the .def() member function.
 * The parameters to .def() is usually:
 *   * String. The name of the function in Python.
 *   * A pointer to the function.
 *   * Docstring for the function in Python.
 *
 * An exception is when binding the constructors of a class, we only specify
 * the parameter types. Pybind11 will find the constructor on its own.
 *
 * Note that none of the function names or class names needs to be the same
 * between the Python importable object and the C++ one. You can always define
 * a different name for the functions and classes when bind them in Python.
 *
 * @param core: The name of the Python module.
 * @param m: The module object to bind classes and functions to.
 */
PYBIND11_MODULE(core, m) {
    // Bind C++ Tensor class to framework.core.Tensor with all of its member
    // functions.
    py::class_<Tensor>(m, "Tensor")
        // For shape only constructor.
        .def(py::init<const std::vector<size_t> &>())
        // For shape and data constructor.
        .def(
            py::init<const std::vector<size_t> &, const std::vector<float> &>())
        // For the copy_to_numpy() member function.
        .def("copy_to_numpy", &Tensor::copy_to_numpy);

    // Create a submodule named ops under framework.core.
    auto ops_module = m.def_submodule("ops");

    // Bind all the forward operations under the framework.core.ops.
    // For example, framework.core.ops.matmul().
    ops_module.def(
        "matmul",                  // Function name in Python.
        &ops::matmul,              // Pointer to the function.
        "(m, k), (k, n) -> (m, n)" // Docstring of the function in Python.
    );
    ops_module.def("add_row_broadcast", &ops::add_row_broadcast,
                   "(m, n), (1, n) -> (m, n)");
    ops_module.def("add_element_wise_", &ops::add_element_wise_,
                   "(m, n), (m, n)");
    ops_module.def("multiply", &ops::multiply, "(m, n), (m, n) -> (m, n)");
    ops_module.def("relu", &ops::relu, "(m, n) -> (m, n)");
    ops_module.def("softmax", &ops::softmax, "(m, n) -> (m, n)");
    ops_module.def("log", &ops::log, "(m, n) -> (m, n)");
    ops_module.def("sum", &ops::sum, "(m, n) -> (1,)");

    // Bind all the backward operations under the framework.core.ops.
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
