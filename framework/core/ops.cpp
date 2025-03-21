#include "ops.h"
#include <stdexcept>

namespace py = pybind11;

Tensor matmul(const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.get_shape();
    const auto& b_shape = b.get_shape();

    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("Matmul requires 2D tensors.");
    }

    if (a_shape[1] != b_shape[0]) {
        throw std::runtime_error("Incompatible dimensions for matmul.");
    }

    size_t m = a_shape[0];
    size_t k = a_shape[1];
    size_t n = b_shape[1];

    Tensor result({m, n});
    std::vector<float>& result_data = const_cast<std::vector<float>&>(result.get_data_vector());

    const auto& a_data = a.get_data_vector();
    const auto& b_data = b.get_data_vector();

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            result_data[i * n + j] = sum;
        }
    }

    return result;
}
