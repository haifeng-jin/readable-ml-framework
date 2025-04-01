#include "ops.h"
#include <stdexcept>
#include <vector>
#include <thread>
#include <future>

namespace py = pybind11;

// Function to perform matrix multiplication for a subset of rows
void matmul_thread_task(size_t start_row, size_t end_row, size_t m, size_t n, size_t k,
                      const std::vector<float>& a_data, const std::vector<float>& b_data,
                      std::vector<float>& result_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            result_data[i * n + j] = sum;
        }
    }
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.get_shape();
    const auto& b_shape = b.get_shape();

    size_t m = a_shape[0];
    size_t k = a_shape[1];
    size_t n = b_shape[1];

    Tensor result({m, n});
    std::vector<float>& result_data = const_cast<std::vector<float>&>(result.get_data_vector());

    const auto& a_data = a.get_data_vector();
    const auto& b_data = b.get_data_vector();

    // Determine a reasonable number of threads to use.
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    // Determine the number of rows each thread will process
    size_t rows_per_thread = m / num_threads;
    size_t remaining_rows = m % num_threads;

    size_t start_row = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t num_rows = rows_per_thread + (t < remaining_rows ? 1 : 0);
        size_t end_row = start_row + num_rows;

        if (num_rows > 0) {
            // Use the named function matmul_thread_task
            futures.push_back(std::async(std::launch::async,
                                         matmul_thread_task, start_row, end_row, m, n, k,
                                         std::ref(a_data), std::ref(b_data), std::ref(result_data)));
        }
        start_row = end_row;
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }

    return result;
}

// Function to perform row-wise addition for a subset of rows
void add_broadcast_row_thread_task(size_t start_row, size_t end_row, size_t n,
                                 const std::vector<float>& a_data, const std::vector<float>& b_data,
                                 std::vector<float>& result_data) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result_data[i * n + j] = a_data[i * n + j] + b_data[j];
        }
    }
}

Tensor add_broadcast_row(const Tensor& a, const Tensor& b) {
    const auto& a_shape = a.get_shape();
    const auto& b_shape = b.get_shape();

    size_t m = a_shape[0];
    size_t n = a_shape[1];

    Tensor result({m, n});
    std::vector<float>& result_data = const_cast<std::vector<float>&>(result.get_data_vector());

    const auto& a_data = a.get_data_vector();
    const auto& b_data = b.get_data_vector();

    // Determine a reasonable number of threads to use.
    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    // Determine the number of rows each thread will process
    size_t rows_per_thread = m / num_threads;
    size_t remaining_rows = m % num_threads;

    size_t start_row = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t num_rows = rows_per_thread + (t < remaining_rows ? 1 : 0);
        size_t end_row = start_row + num_rows;

        if (num_rows > 0) {
            // Use the named function add_broadcast_row_thread_task
            futures.push_back(std::async(std::launch::async,
                                         add_broadcast_row_thread_task, start_row, end_row, n,
                                         std::ref(a_data), std::ref(b_data), std::ref(result_data)));
        }
        start_row = end_row;
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }

    return result;
}

