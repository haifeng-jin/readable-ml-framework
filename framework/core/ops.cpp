#include "ops.h"
#include <stdexcept>
#include <vector>
#include <thread>
#include <future>

namespace py = pybind11;

Tensor matmul_threaded(const Tensor& a, const Tensor& b, size_t m, size_t k, size_t n, size_t num_threads) {
    Tensor result({m, n});
    std::vector<float>& result_data = const_cast<std::vector<float>&>(result.get_data_vector());

    const auto& a_data = a.get_data_vector();
    const auto& b_data = b.get_data_vector();

    std::vector<std::future<void>> futures;

    // Determine the number of rows each thread will process
    size_t rows_per_thread = m / num_threads;
    size_t remaining_rows = m % num_threads;

    size_t start_row = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t num_rows = rows_per_thread + (t < remaining_rows ? 1 : 0);
        size_t end_row = start_row + num_rows;

        if (num_rows > 0) {
            futures.push_back(std::async(std::launch::async,
                [start_row, end_row, m, n, k, &a_data, &b_data, &result_data]() {
                    for (size_t i = start_row; i < end_row; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            float sum = 0.0f;
                            for (size_t l = 0; l < k; ++l) {
                                sum += a_data[i * k + l] * b_data[l * n + j];
                            }
                            result_data[i * n + j] = sum;
                        }
                    }
                }));
        }
        start_row = end_row;
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }

    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.get_shape();
    const auto& b_shape = b.get_shape();

    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("Matmul requires 2D tensors.");
    }

    if (a_shape[1] != b_shape[0]) {
        throw std::runtime_error("Incompatible dimensions for matmul.");
    }

    // Determine a reasonable number of threads to use.
    // You might want to make this configurable or base it on system resources.
    size_t num_threads = std::thread::hardware_concurrency();
    return matmul_threaded(a, b, a_shape[0], a_shape[1], b_shape[1], num_threads);
}

Tensor add_broadcast_row_threaded(const Tensor& a, const Tensor& b, size_t num_threads) {
    const auto& a_shape = a.get_shape();
    const auto& b_shape = b.get_shape();

    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("Add (broadcast row) requires 2D tensors.");
    }

    if (b_shape[0] != 1) {
        throw std::runtime_error("Second tensor for add (broadcast row) must have first dimension of size 1.");
    }

    if (a_shape[1] != b_shape[1]) {
        throw std::runtime_error("Incompatible second dimensions for add (broadcast row).");
    }

    size_t m = a_shape[0];
    size_t n = a_shape[1];

    Tensor result({m, n});
    std::vector<float>& result_data = const_cast<std::vector<float>&>(result.get_data_vector());

    const auto& a_data = a.get_data_vector();
    const auto& b_data = b.get_data_vector();

    std::vector<std::future<void>> futures;

    // Determine the number of rows each thread will process
    size_t rows_per_thread = m / num_threads;
    size_t remaining_rows = m % num_threads;

    size_t start_row = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t num_rows = rows_per_thread + (t < remaining_rows ? 1 : 0);
        size_t end_row = start_row + num_rows;

        if (num_rows > 0) {
            futures.push_back(std::async(std::launch::async,
                [start_row, end_row, n, &a_data, &b_data, &result_data]() {
                    for (size_t i = start_row; i < end_row; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            result_data[i * n + j] = a_data[i * n + j] + b_data[j];
                        }
                    }
                }));
        }
        start_row = end_row;
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }

    return result;
}

Tensor add_broadcast_row(const Tensor& a, const Tensor& b) {
    size_t num_threads = std::thread::hardware_concurrency();
    return add_broadcast_row_threaded(a, b, num_threads);
}
