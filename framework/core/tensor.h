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

/**
 * @brief Represents a multi-dimensional array (tensor) of floating-point
 * numbers.
 *
 * This class provides a basic implementation of a tensor, storing its
 * shape and the underlying floating-point data. It is designed to be
 * interoperable with NumPy arrays through Pybind11.
 */
class Tensor {
public:
    /**
     * @brief The shape of the tensor, represented as a vector of size_t.
     *
     * Each element in the vector corresponds to the size of a dimension.
     * For example, a tensor with shape {2, 3} would be a 2x3 matrix.
     */
    std::vector<size_t> shape;

    /**
     * @brief The underlying data of the tensor, stored as a contiguous vector
     * of floats.
     *
     * The elements are typically ordered according to a row-major (C-style)
     * layout, though this class itself doesn't enforce a specific layout.
     */
    std::vector<float> data;

    /**
     * @brief Constructs an empty Tensor with the specified shape.
     *
     * The data vector will be initialized with a size corresponding to the
     * total number of elements in the tensor (product of the shape
     * dimensions), but the values will be initialized to 0.0f.
     *
     * @param shape: A vector of size_t representing the desired shape of the
     * tensor. Must contain at least one element (rank >= 1). When calling from
     * Python, it is a plain Python tuple of integers.
     */
    Tensor(const std::vector<size_t>& shape);

    /**
     * @brief Constructs a Tensor with the specified shape and initializes it
     * with the provided data.
     *
     * The size of the data vector must exactly match the total number of
     * elements implied by the shape.
     *
     * @param shape: A vector of size_t representing the desired shape of the
     * tensor. Must contain at least one element (rank >= 1). When calling from
     * Python, it is a plain Python tuple of integers.
     * @param data_ptr: A constant reference to a vector of floats containing
     * the initial data.  The size of this vector must be equal to the product
     * of the dimensions in the shape vector. When calling from Python, it
     * should be a plain Python list.
     */
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data_ptr);

    /**
     * @brief Destructor for the Tensor class.
     *
     * Currently, this destructor doesn't perform any specific memory
     * management as the `std::vector` handles the deallocation of the
     * underlying data.  However, it is good practice to include a destructor
     * in case future versions of the class require custom cleanup.
     *
     * When its Python binding has no reference an more and be garbage
     * collected by Python, the destructor will be triggered. So no need to
     * worry about its deconstruction when using from Python.
     */
    ~Tensor();

    /**
     * @brief Creates a NumPy array containing a copy of the tensor's data.
     *
     * The returned NumPy array will have the same shape as the Tensor object
     * and will contain a copy of the underlying float data. This allows for
     * seamless interaction between the Tensor class and NumPy in Python.
     *
     * @return A py::array_t<float> object containing a copy of the tensor's
     * data with the same shape.
     */
    py::array_t<float> copy_to_numpy() const;
};

#endif // TENSOR_H
